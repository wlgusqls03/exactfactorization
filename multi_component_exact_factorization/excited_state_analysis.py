#!/usr/bin/env python3
"""전자 local eigenstate population과 state-resolved dynamics를 분석한다.

Full density 그림만으로는 excited state에서 lower state로 population이
이동했는지 판정할 수 없다. 이 도구는 매 ``(q,R)``에서

    H_BO(x;q,R) varphi_n(x;q,R) = E_n(q,R) varphi_n(x;q,R)

을 풀고 저장된 조건부 전자 factor를 투영한다. 전역 population은

    P_n(t) = int dq dR |chi|^2 |Lambda|^2
             |<varphi_n|Phi>_x|^2

이다. 이 분석에서만 BO basis를 사용하며, direct EF 전파에는 surface
hopping이나 BO coefficient가 공급되지 않는다.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from result_paths import dated_results_dir

from .core import build_model, local_electronic_basis
from .visualize import NUMBER_FORMATTER, archive_arguments, load_archive, selected_frames


def calculate_state_decomposition(data, n_states):
    """Adiabatic energy, population, state-resolved ``(q,R)`` density 계산.

    입력 archive의 주요 shape:
        phi(nt,nx,nq,nR), lambda_wavefunction(nt,nq,nR), chi(nt,nR)

    반환 shape:
        energies(n_states,nq,nR)
        resolved(nt,n_states,nq,nR)
        populations(nt,n_states)
        residual(nt)

    ``resolved[t,n,q,R]``는 local state n에 속하는 proton-heavy
    configuration density이고, q와 R을 적분하면 ``populations[t,n]``이다.
    """
    # ------------------------------------------------------------------
    # 1. NPZ에 저장된 계산 option으로 원래 Hamiltonian/grid를 재구성한다.
    # ------------------------------------------------------------------
    options = archive_arguments(data)
    if not options:
        raise ValueError("archive에 model argument metadata가 없습니다.")
    model = build_model(SimpleNamespace(**options))
    for key, rebuilt in (("x", model.x), ("q", model.q), ("R", model.R)):
        if not np.allclose(data[key], rebuilt):
            raise ValueError(f"archive와 재구성한 {key} grid가 다릅니다.")

    # ------------------------------------------------------------------
    # 2. 모든 (q,R)의 local H_BO basis를 한 번만 계산한다.
    # ------------------------------------------------------------------
    # states shape: (state,nx,nq,nR). 시간과 무관하므로 frame loop 밖에 둔다.
    energies, states = local_electronic_basis(model, n_states)
    nt = len(data["times_fs"])
    nq, nR = len(model.q), len(model.R)
    resolved = np.empty((nt, n_states, nq, nR))                    # (nt,state,nq,nR)
    populations = np.empty((nt, n_states))                         # (nt,state)
    phi_frames = data["phi"]
    lam_frames = data["lambda_wavefunction"]
    chi_frames = data["chi"]
    norm_frames = data["norm"] if "norm" in data.files else None

    # ------------------------------------------------------------------
    # 3. 매 시간의 conditional electron Phi를 local basis에 투영한다.
    # ------------------------------------------------------------------
    for frame in range(nt):
        phi = phi_frames[frame]                                     # (nx,nq,nR)

        # c_n(q,R,t)=int dx varphi_n^*(x;q,R) Phi(x;q,R,t)
        coefficients = np.sum(
            np.conj(states)*phi[None, :, :, :], axis=1
        )*model.dx                                                   # (state,nq,nR)

        # EF의 nested probability 해석에 따라 proton-heavy joint weight는
        # |Lambda_R(q,t)|^2 |chi(R,t)|^2이다.
        nuclear_weight = (
            np.abs(lam_frames[frame])**2
            *np.abs(chi_frames[frame])[None, :]**2
        )                                                            # (nq,nR)
        resolved[frame] = np.abs(coefficients)**2*nuclear_weight[None, :, :]
        # Direct discretization의 작은 global norm drift가 state composition으로
        # 오해되지 않도록 각 frame의 population은 해당 full norm으로 나눈다.
        frame_norm = (
            float(norm_frames[frame]) if norm_frames is not None
            else float(np.sum(nuclear_weight)*model.dq*model.dR)
        )
        resolved[frame] /= max(frame_norm, 1.0e-300)
        # P_n(t)=int dq dR rho_n(q,R,t)
        populations[frame] = (
            np.sum(resolved[frame], axis=(1, 2))*model.dq*model.dR
        )

    # 계산한 n_states 밖의 전자 성분. 충분한 basis를 썼다면 0에 가까워야 한다.
    residual = np.maximum(0.0, 1.0-np.sum(populations, axis=1))
    return energies, resolved, populations, residual


def save_population_plot(times, populations, residual, outdir, dpi=180):
    """State population 전체와 작은 전이 성분 확대 그림을 함께 저장."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    ax, transfer_ax = axes
    for state in range(populations.shape[1]):
        ax.plot(times, populations[:, state], lw=2, label=rf"$P_{state}$")
    ax.plot(times, residual, color="0.35", ls="--", label="outside basis")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("electronic-state population")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
    ax.set_title("BO-state populations")
    ax.legend(frameon=False, ncol=min(4, populations.shape[1]+1))

    # 왼쪽의 0--1 scale에서는 1e-3 정도의 작은 전이가 거의 보이지 않는다.
    # t=0에서 가장 큰 상태를 제외한 성분만 오른쪽 panel에서 자동 확대한다.
    initial_state = int(np.argmax(populations[0]))
    transfer_states = [
        state for state in range(populations.shape[1]) if state != initial_state
    ]
    for state in transfer_states:
        transfer_ax.plot(
            times, populations[:, state], lw=2, label=rf"$P_{state}$"
        )
    transfer_ax.plot(times, residual, color="0.35", ls="--", label="outside basis")
    transfer_max = max(
        [float(np.max(populations[:, state])) for state in transfer_states]
        +[float(np.max(residual)), 1.0e-8]
    )
    transfer_ax.set_ylim(-0.02*transfer_max, 1.08*transfer_max)
    transfer_ax.set_xlabel("time (fs)")
    transfer_ax.set_ylabel("transferred population")
    transfer_ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
    transfer_ax.set_title(rf"Transfer from $n={initial_state}$")
    transfer_ax.legend(frameon=False, ncol=2)
    path = outdir/"electronic_state_populations.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"전자 상태 population 저장: {path}")


def save_population_animation(
    q, R, times, resolved, populations, residual, outdir,
    fps=10, max_frames=180, dpi=120, fmt="mp4",
):
    """낮은 세 전자상태의 ``(q,R)`` 분포와 population을 4분할 영상화.

    각 state map은 자기 영상 전체에서 고정된 color scale을 사용한다. State
    사이의 절대 크기는 colorbar 수치로 비교하고, 같은 state의 시간 변화는
    색의 변화로 직접 비교한다.
    """
    if resolved.shape[1] < 3:
        raise ValueError("population 동영상에는 최소 3개 electronic state가 필요합니다.")
    frames = selected_frames(len(times), min(max_frames, len(times)))
    # 세 state마다 모든 표시 frame을 훑어 고정 colorbar maximum을 정한다.
    maxima = [
        max(float(np.max(resolved[frame, state])) for frame in frames)
        for state in range(3)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    map_axes = (axes[0, 0], axes[0, 1], axes[1, 0])
    images = []
    extent = [R[0], R[-1], q[0], q[-1]]
    first = int(frames[0])
    for state, (ax, maximum) in enumerate(zip(map_axes, maxima)):
        artist = ax.imshow(
            resolved[first, state], origin="lower", aspect="auto",
            extent=extent, cmap="magma", vmin=0.0, vmax=max(maximum, 1.0e-12),
        )
        ax.set_title(rf"State {state}: $\rho_{state}(q,R,t)$")
        ax.set_xlabel("heavy R")
        ax.set_ylabel("proton q")
        fig.colorbar(
            artist, ax=ax, label=rf"$\rho_{state}$", pad=0.012,
            fraction=0.046, format=NUMBER_FORMATTER,
        )
        images.append(artist)

    # 네 번째 panel에는 전 시간 population curve와 현재 시간 marker를 둔다.
    population_ax = axes[1, 1]
    for state in range(populations.shape[1]):
        population_ax.plot(
            times, populations[:, state], lw=2, label=rf"$P_{state}$"
        )
    population_ax.plot(times, residual, color="0.35", ls="--", label="outside basis")
    marker = population_ax.axvline(times[first], color="black", lw=1.3)
    population_ax.set_xlabel("time (fs)")
    population_ax.set_ylabel("population")
    population_ax.set_ylim(-0.02, 1.02)
    population_ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
    population_ax.set_title("Global electronic-state populations")
    population_ax.legend(frameon=False, fontsize=8, ncol=2)
    title = fig.suptitle(
        f"BO-state decomposition | {times[first]:.4f} fs", fontsize=13
    )
    fig.supxlabel(
        "BO basis is used for analysis only", fontsize=8.5, color="0.35"
    )

    def update(number):
        """세 state-resolved density와 현재 시간선을 같은 frame으로 이동."""
        frame = int(frames[number])
        for state, artist in enumerate(images):
            artist.set_data(resolved[frame, state])
        marker.set_xdata([times[frame], times[frame]])
        title.set_text(f"BO-state decomposition | {times[frame]:.4f} fs")
        return (*images, marker, title)

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/"electronic_state_population_dynamics.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=2800), dpi=dpi)
    else:
        if fmt == "mp4":
            print("ffmpeg을 찾지 못해 GIF로 대신 저장합니다.")
        path = outdir/"electronic_state_population_dynamics.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"전자 상태 분해 dynamics 저장: {path}")


def run(args, data=None, decomposition=None):
    """Archive 읽기 -> state projection -> NPZ/그림/영상 저장의 전체 흐름."""
    data = data if data is not None else load_archive(
        args.archive, materialize=not getattr(args, "low_memory", False)
    )
    options = archive_arguments(data)

    # 영상은 첫 세 상태를 보여주지만 residual 판정은 최소 여섯 상태로 한다.
    initial_excitation = int(options.get("electron_excitation", 0))
    n_states = args.n_states or max(6, initial_excitation+2)
    if decomposition is None:
        decomposition = calculate_state_decomposition(data, n_states)
    energies, resolved, populations, residual = decomposition
    requested_outdir = Path(args.outdir) if args.outdir else Path(args.archive).parent/"figures"
    outdir = dated_results_dir(requested_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # 후속 분석에서 비싼 local diagonalization을 반복하지 않도록 energy와
    # state-resolved density까지 별도 NPZ에 저장한다.
    np.savez_compressed(
        outdir/"electronic_state_analysis.npz",
        times_fs=data["times_fs"], q=data["q"], R=data["R"],
        energies=energies, state_resolved_density=resolved,
        populations=populations, residual_population=residual,
    )
    save_population_plot(
        data["times_fs"], populations, residual, outdir, dpi=args.dpi
    )
    if not args.no_animation:
        save_population_animation(
            data["q"], data["R"], data["times_fs"], resolved, populations,
            residual, outdir, fps=args.fps, max_frames=args.max_frames,
            dpi=args.animation_dpi, fmt=args.format,
        )
    print("t=0 populations:", " ".join(
        f"P{state}={populations[0, state]:.6f}"
        for state in range(n_states)
    ))
    print(f"최대 truncated-basis residual: {np.max(residual):.3e}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--outdir", default="")
    parser.add_argument(
        "--n-states", type=int, default=0,
        help="0이면 max(6, initial excitation+2)를 자동 선택",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--animation-dpi", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--low-memory", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
