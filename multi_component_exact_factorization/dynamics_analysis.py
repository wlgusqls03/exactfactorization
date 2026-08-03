#!/usr/bin/env python3
"""저장된 multi-component EF 궤적에서 실제 동역학 그림을 만든다.

조건부 factor의 한 단면만 보는 대신 다음 물리량을 계산한다.

* 전자/양성자/무거운 핵의 정규화된 1D marginal과 평균·표준편차
* 전자 difference density와 좌우 population
* BO energy gap, nuclear joint density, BO population, state-resolved density

BO basis는 마지막 분석에만 사용한다. Direct EF 전파 자체에는 BO surface나
surface hopping 정보를 공급하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from result_paths import dated_results_dir

from .excited_state_analysis import calculate_state_decomposition
from .visualize import NUMBER_FORMATTER, archive_arguments, load_archive


PARTICLE_COLORS = ("#2474B5", "#D6534D", "#31915B")


def normalized_marginals(data):
    """모든 저장 frame의 실제 1D marginal과 joint density를 계산한다.

    입력 shape:
        phi(nt,nx,nq,nR), lambda_wavefunction(nt,nq,nR), chi(nt,nR)

    반환 shape:
        electron(nt,nx), proton(nt,nq), heavy(nt,nR), joint(nt,nq,nR)

    각 frame은 적분값이 1이 되도록 정규화한다. 따라서 작은 norm drift가
    packet 이동이나 폭 변화처럼 보이지 않는다.
    """
    x, q, R = data["x"], data["q"], data["R"]
    dx, dq, dR = x[1]-x[0], q[1]-q[0], R[1]-R[0]
    nt = len(data["times_fs"])
    # Compressed NPZ members are expensive to reopen for every frame. Keep
    # these three trajectory arrays alive for the complete reduction pass.
    phi_frames = data["phi"]
    lam_frames = data["lambda_wavefunction"]
    chi_frames = data["chi"]
    electron = np.empty((nt, len(x)))
    proton = np.empty((nt, len(q)))
    heavy = np.empty((nt, len(R)))
    joint = np.empty((nt, len(q), len(R)))

    # 큰 phi(nt,nx,nq,nR)의 중간 복사본을 만들지 않도록 frame별로 적분한다.
    for frame in range(nt):
        lam2 = np.abs(lam_frames[frame])**2                        # (nq,nR)
        chi2 = np.abs(chi_frames[frame])**2                        # (nR,)
        joint_frame = lam2*chi2[None, :]                            # (nq,nR)
        norm = max(float(np.sum(joint_frame)*dq*dR), 1.0e-300)
        joint[frame] = joint_frame/norm
        electron[frame] = np.sum(
            np.abs(phi_frames[frame])**2*joint[frame][None, :, :],
            axis=(1, 2),
        )*dq*dR                                                     # (nx,)
        proton[frame] = np.sum(joint[frame], axis=1)*dR            # (nq,)
        heavy[frame] = np.sum(joint[frame], axis=0)*dq             # (nR,)

        # Conditional factor의 수치 오차까지 포함해 각 1D 분포도 따로 정규화한다.
        electron[frame] /= max(float(np.sum(electron[frame])*dx), 1.0e-300)
        proton[frame] /= max(float(np.sum(proton[frame])*dq), 1.0e-300)
        heavy[frame] /= max(float(np.sum(heavy[frame])*dR), 1.0e-300)
    return electron, proton, heavy, joint


def moments(grid, density):
    """density(nt,ngrid)의 평균과 표준편차를 반환한다."""
    spacing = float(grid[1]-grid[0])
    mean = np.sum(density*grid[None, :], axis=1)*spacing
    variance = np.sum(
        density*(grid[None, :]-mean[:, None])**2, axis=1
    )*spacing
    return mean, np.sqrt(np.maximum(variance, 0.0))


def metadata_footer(data):
    """그림 아래에 넣을 짧은 초기조건 설명."""
    options = archive_arguments(data)
    if not options:
        return "initial-state metadata unavailable"
    return (
        rf"BO $n={int(options.get('electron_excitation', 0))}$  |  "
        rf"$q_0={options.get('q0', np.nan):.2f}$, "
        rf"$R_0={options.get('R0', np.nan):.2f}$  |  "
        rf"$m_p={options.get('proton_mass', np.nan):.0f}$, "
        rf"$M_H={options.get('heavy_mass', np.nan):.0f}$"
    )


def density_map(ax, times, grid, density, title, ylabel):
    """position-time density map 하나를 공통 서식으로 그린다."""
    artist = ax.pcolormesh(
        times, grid, density.T, shading="nearest", cmap="magma", rasterized=True
    )
    ax.set_title(title)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel(ylabel)
    return artist


def save_marginal_figure(data, densities, means, widths, outdir, dpi):
    """세 1D marginal의 position-time map과 평균±폭을 저장한다."""
    times = data["times_fs"]
    grids = (data["x"], data["q"], data["R"])
    titles = ("Electron marginal", "Proton marginal", "Heavy marginal")
    labels = (r"electron $x$ ($a_0$)", r"proton $q$ ($a_0$)", r"heavy $R$ ($a_0$)")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    for ax, grid, rho, title, label in zip(
        axes.flat[:3], grids, densities, titles, labels
    ):
        artist = density_map(ax, times, grid, rho, title, label)
        fig.colorbar(
            artist, ax=ax, label="probability density", pad=0.012,
            format=NUMBER_FORMATTER,
        )

    ax = axes[1, 1]
    for mean, width, color, label in zip(means, widths, PARTICLE_COLORS, ("electron", "proton", "heavy")):
        ax.plot(times, mean, color=color, lw=2, label=label)
        ax.fill_between(times, mean-width, mean+width, color=color, alpha=0.16)
    ax.set_title(r"Mean $\pm$ width")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel(r"position ($a_0$)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("Particle dynamics", fontsize=14)
    fig.supxlabel(metadata_footer(data), fontsize=9, color="0.35")
    path = outdir/"marginal_dynamics.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"1D marginal dynamics 저장: {path}")


def save_electron_transfer_figure(data, electron, mean, width, divider, outdir, dpi):
    """전자 이동을 difference density와 좌우 적분값으로 보여준다."""
    x, times = data["x"], data["times_fs"]
    dx = float(x[1]-x[0])
    difference = electron-electron[0, None, :]
    left = np.sum(electron[:, x < divider], axis=1)*dx
    right = np.sum(electron[:, x >= divider], axis=1)*dx
    rearranged = 0.5*np.sum(np.abs(difference), axis=1)*dx

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.7), constrained_layout=True)
    scale = max(float(np.max(np.abs(difference))), 1.0e-14)
    artist = axes[0, 0].pcolormesh(
        times, x, difference.T, shading="nearest", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale), rasterized=True,
    )
    axes[0, 0].set_title("Difference density")
    axes[0, 0].set_xlabel("time (fs)")
    axes[0, 0].set_ylabel(r"electron $x$ ($a_0$)")
    difference_bar = fig.colorbar(
        artist, ax=axes[0, 0], format=NUMBER_FORMATTER,
        pad=0.012, fraction=0.046,
    )
    # Matplotlib 3.5의 constrained-layout에는 매우 작은 diverging scale에서
    # 세로 colorbar label 위치가 NaN이 되는 버그가 있어 짧은 상단 제목을 쓴다.
    difference_bar.ax.set_title(r"$\Delta\rho_e$", pad=7, fontsize=9)

    snapshot_indices = sorted(set((0, len(times)//2, len(times)-1)))
    for frame in snapshot_indices:
        axes[0, 1].plot(x, electron[frame], lw=2, label=f"{times[frame]:.3f} fs")
    axes[0, 1].axvline(divider, color="0.35", ls="--", lw=1.2)
    axes[0, 1].set_title("Electron snapshots")
    axes[0, 1].set_xlabel(r"electron $x$ ($a_0$)")
    axes[0, 1].set_ylabel("probability density")
    axes[0, 1].yaxis.set_major_formatter(NUMBER_FORMATTER)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(times, left, lw=2, color="#2474B5", label=rf"$P_L$ ($x<{divider:.2f}$)")
    axes[1, 0].plot(times, right, lw=2, color="#D6534D", label=rf"$P_R$ ($x\geq{divider:.2f}$)")
    axes[1, 0].set_title("Left / right population")
    axes[1, 0].set_xlabel("time (fs)")
    axes[1, 0].set_ylabel("population")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(times, rearranged, color="#7B4EA3", lw=2, label="rearranged density")
    axes[1, 1].plot(times, mean-mean[0], color="#222222", lw=1.6, label=r"$\langle x\rangle-\langle x\rangle_0$")
    axes[1, 1].plot(times, width-width[0], color="#E28E2C", lw=1.6, label=r"$\sigma_x-\sigma_{x,0}$")
    axes[1, 1].axhline(0.0, color="0.65", lw=0.8)
    axes[1, 1].set_title("Electronic change")
    axes[1, 1].set_xlabel("time (fs)")
    axes[1, 1].set_ylabel("change")
    axes[1, 1].yaxis.set_major_formatter(NUMBER_FORMATTER)
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle("Electron transfer diagnostics", fontsize=14)
    fig.supxlabel(metadata_footer(data), fontsize=9, color="0.35")
    path = outdir/"electron_transfer.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"전자 이동 분석 저장: {path}")
    return difference, left, right, rearranged


def save_nonadiabatic_figure(data, joint, energies, resolved, populations, frame, outdir, dpi):
    """BO gap·joint density·state population을 한 장에 요약한다."""
    if len(energies) < 2:
        raise ValueError("BO gap 그림에는 최소 두 electronic state가 필요합니다.")
    q, R, times = data["q"], data["R"], data["times_fs"]
    extent = [R[0], R[-1], q[0], q[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.0), constrained_layout=True)

    gap = energies[1]-energies[0]
    gap_artist = axes[0, 0].imshow(gap, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    initial = joint[0]
    level = 0.12*float(np.max(initial))
    if level > 0.0:
        axes[0, 0].contour(R, q, initial, levels=[level], colors="white", linewidths=1.3)
    axes[0, 0].set_title(r"BO gap $E_1-E_0$")
    axes[0, 0].set_xlabel(r"heavy $R$ ($a_0$)")
    axes[0, 0].set_ylabel(r"proton $q$ ($a_0$)")
    fig.colorbar(gap_artist, ax=axes[0, 0], label="energy (a.u.)", format=NUMBER_FORMATTER)

    joint_artist = axes[0, 1].imshow(joint[frame], origin="lower", aspect="auto", extent=extent, cmap="magma")
    axes[0, 1].set_title(f"Nuclear density | {times[frame]:.3f} fs")
    axes[0, 1].set_xlabel(r"heavy $R$ ($a_0$)")
    axes[0, 1].set_ylabel(r"proton $q$ ($a_0$)")
    fig.colorbar(joint_artist, ax=axes[0, 1], label=r"$\rho_{qR}$", format=NUMBER_FORMATTER)

    for state in range(populations.shape[1]):
        axes[0, 2].plot(times, populations[:, state], lw=2, label=rf"$P_{state}$")
    axes[0, 2].set_title("BO-state populations")
    axes[0, 2].set_xlabel("time (fs)")
    axes[0, 2].set_ylabel("population")
    axes[0, 2].set_ylim(-0.02, 1.02)
    axes[0, 2].legend(frameon=False, ncol=2)

    shown_states = min(3, resolved.shape[1])
    common_max = max(float(np.max(resolved[frame, :shown_states])), 1.0e-14)
    state_artists = []
    for state, ax in enumerate(axes[1]):
        if state >= shown_states:
            ax.axis("off")
            continue
        artist = ax.imshow(
            resolved[frame, state], origin="lower", aspect="auto", extent=extent,
            cmap="magma", vmin=0.0, vmax=common_max,
        )
        state_artists.append(artist)
        ax.set_title(rf"State $n={state}$ density")
        ax.set_xlabel(r"heavy $R$ ($a_0$)")
        ax.set_ylabel(r"proton $q$ ($a_0$)")
    if state_artists:
        fig.colorbar(
            state_artists[0], ax=list(axes[1, :shown_states]), label=r"$\rho_n(q,R,t)$",
            format=NUMBER_FORMATTER, shrink=0.88,
        )
    fig.suptitle("Nonadiabatic dynamics", fontsize=14)
    fig.supxlabel(
        metadata_footer(data)+f"  |  snapshot={times[frame]:.3f} fs",
        fontsize=9, color="0.35",
    )
    path = outdir/"nonadiabatic_summary.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"nonadiabatic 요약 저장: {path}")


def support_weighted_rms(field, weight):
    """각 frame에서 probability-weighted RMS를 계산한다."""
    axes = tuple(range(1, field.ndim))
    denominator = np.sum(weight, axis=axes)
    numerator = np.sum(weight*np.abs(field)**2, axis=axes)
    return np.sqrt(numerator/np.maximum(denominator, 1.0e-300))


def save_correlation_figure(
    data, means, widths, rearranged, populations, residual, joint, outdir, dpi,
):
    """물리 변화와 constraint correction의 시작 시점을 함께 그린다."""
    times = data["times_fs"]
    heavy = np.abs(data["chi"])**2
    rms_a = support_weighted_rms(data["a"], joint)
    rms_b = support_weighted_rms(data["b"], joint)
    rms_alpha = support_weighted_rms(data["alpha"], heavy)
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.0), constrained_layout=True)

    for state in range(populations.shape[1]):
        axes[0, 0].plot(times, populations[:, state], lw=1.6, label=rf"$P_{state}$")
    axes[0, 0].plot(times, residual, color="0.3", ls="--", label="outside basis")
    axes[0, 0].set_title("BO-state composition")
    axes[0, 0].legend(frameon=False, ncol=4, fontsize=7)

    axes[0, 1].plot(times, means[1], lw=2, label=r"$\langle q\rangle$")
    axes[0, 1].plot(times, widths[1], lw=2, label=r"$\sigma_q$")
    axes[0, 1].plot(times, rearranged, lw=2, label=r"$D_{\rm rearr}$")
    axes[0, 1].set_title("Proton motion and electron rearrangement")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(times, rms_a, label=r"RMS$_\rho(a)$")
    axes[1, 0].plot(times, rms_b, label=r"RMS$_\rho(b)$")
    axes[1, 0].plot(times, rms_alpha, label=r"RMS$_\rho(\alpha)$")
    axes[1, 0].set_yscale("symlog", linthresh=1.0e-4)
    axes[1, 0].set_title("Occupied-support vector connections")
    axes[1, 0].legend(frameon=False, fontsize=8)

    gamma_keys = (
        "max_abs_support_gamma_phi_dt", "max_abs_support_gamma_lam_dt",
    ) if "max_abs_support_gamma_phi_dt" in data.files else (
        "max_abs_gamma_phi", "max_abs_gamma_lam",
    )
    for key in ("pnc_projection_correction", *gamma_keys):
        if key in data.files:
            axes[1, 1].semilogy(
                times, np.maximum(data[key], 1.0e-18), label=key,
            )
    axes[1, 1].set_title("Constraint-correction diagnostics")
    axes[1, 1].legend(frameon=False, fontsize=7)

    if "max_corrected_rate_phi" in data.files:
        axes[2, 0].semilogy(
            times, np.maximum(data["max_corrected_rate_phi"], 1.0e-20),
            label=r"corrected $r_\Phi$",
        )
        axes[2, 0].semilogy(
            times, np.maximum(data["max_corrected_rate_lam"], 1.0e-20),
            label=r"corrected $r_\Lambda$",
        )
    axes[2, 0].semilogy(
        times, np.maximum(np.abs(data["norm"]-1.0), 1.0e-20),
        label="|norm-1|",
    )
    axes[2, 0].set_title("Preserved constraints")
    axes[2, 0].legend(frameon=False, fontsize=8)

    for values, label in zip(
        means, (r"$\Delta\langle x\rangle$", r"$\Delta\langle q\rangle$", r"$\Delta\langle R\rangle$"),
    ):
        axes[2, 1].plot(times, values-values[0], label=label)
    axes[2, 1].set_title("Mean-position changes")
    axes[2, 1].legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("time (fs)")
        ax.grid(alpha=0.2)
    path = outdir/"coupled_dynamics_correlation.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"물리량-수치진단 correlation 저장: {path}")
    return rms_a, rms_b, rms_alpha


def run(args, data=None, decomposition=None):
    data = data if data is not None else load_archive(
        args.archive, materialize=not getattr(args, "low_memory", False)
    )
    requested_outdir = (
        Path(args.outdir)
        if args.outdir
        else Path(args.archive).resolve().parent/"dynamics_analysis"
    )
    outdir = dated_results_dir(requested_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    electron, proton, heavy, joint = normalized_marginals(data)
    densities = (electron, proton, heavy)
    grids = (data["x"], data["q"], data["R"])
    moment_pairs = tuple(moments(grid, density) for grid, density in zip(grids, densities))
    means = tuple(pair[0] for pair in moment_pairs)
    widths = tuple(pair[1] for pair in moment_pairs)
    save_marginal_figure(data, densities, means, widths, outdir, args.dpi)

    options = archive_arguments(data)
    divider = args.electron_divider
    if divider is None:
        divider = 0.5*(float(options.get("q0", 0.0))+float(options.get("R0", 0.0)))
    difference, left, right, rearranged = save_electron_transfer_figure(
        data, electron, means[0], widths[0], divider, outdir, args.dpi
    )

    payload = dict(
        times_fs=data["times_fs"], x=data["x"], q=data["q"], R=data["R"],
        electron_density=electron,
        proton_density=proton, heavy_density=heavy, nuclear_joint_density=joint,
        electron_mean=means[0], proton_mean=means[1], heavy_mean=means[2],
        electron_width=widths[0], proton_width=widths[1], heavy_width=widths[2],
        electron_difference=difference, electron_left_population=left,
        electron_right_population=right, electron_rearranged_density=rearranged,
        electron_divider=np.array(divider),
    )
    if not args.no_bo:
        n_states = max(2, args.n_states)
        if decomposition is None:
            decomposition = calculate_state_decomposition(data, n_states)
        energies, resolved, populations, residual = decomposition
        frame = args.frame if args.frame >= 0 else len(data["times_fs"])+args.frame
        if not 0 <= frame < len(data["times_fs"]):
            raise IndexError(f"--frame {args.frame}가 저장 frame 범위를 벗어납니다.")
        save_nonadiabatic_figure(
            data, joint, energies, resolved, populations, frame, outdir, args.dpi
        )
        payload.update(
            bo_energies=energies, state_resolved_density=resolved,
            state_populations=populations, state_basis_residual=residual,
        )
        rms_a, rms_b, rms_alpha = save_correlation_figure(
            data, means, widths, rearranged, populations, residual, joint,
            outdir, args.dpi,
        )
        payload.update(
            support_rms_a=rms_a, support_rms_b=rms_b,
            support_rms_alpha=rms_alpha,
        )

    archive = outdir/"dynamics_observables.npz"
    np.savez_compressed(archive, **payload)
    print(f"수치 observable 저장: {archive}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="multi_component_direct_ef.npz 경로")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--n-states", type=int, default=6, help="분석할 낮은 BO 상태 수")
    parser.add_argument("--frame", type=int, default=-1, help="BO/joint snapshot frame; -1은 마지막")
    parser.add_argument(
        "--electron-divider", type=float, default=None,
        help="전자 좌/우 population 경계; 생략하면 초기 q0와 R0의 중점",
    )
    parser.add_argument("--no-bo", action="store_true", help="빠른 marginal/transfer 분석만 수행")
    parser.add_argument("--low-memory", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
