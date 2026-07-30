#!/usr/bin/env python3
"""기존 2-component EF 결과의 논문형 그림과 4분할 dynamics 동영상.

정적 그림은 논문처럼 TDPES와 핵 밀도를 겹친 3-panel이다. 동영상은
전자 Phi, 핵 chi, 전체 Psi density, TDPES를 네 독립 panel로 보여준다.
Real-space와 BO-basis direct archive를 모두 읽을 수 있다.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.ticker import FuncFormatter
import numpy as np

from result_paths import dated_results_dir


def readable_number(value, _position=None):
    """1e-3 order 이하는 과학 표기, 나머지는 소수점 둘째 자리까지 표시."""
    if not np.isfinite(value):
        return ""
    if value != 0.0 and abs(value) < 1.0e-2:
        return f"{value:.2e}"
    return f"{value:.2f}"


NUMBER_FORMATTER = FuncFormatter(readable_number)


def phi_frame(data, frame):
    """저장된 Phi를 읽거나 BO coefficient와 basis로 재구성한다."""
    if "phi" in data:
        return data["phi"][frame]                                  # (nr,nR)
    if "coefficients" in data and "bo_states" in data:
        return np.einsum(
            "sR,sRr->rR", data["coefficients"][frame],
            data["bo_states"], optimize=True,
        )
    raise KeyError("phi 또는 coefficients+bo_states가 필요합니다.")


def frame_fields(data, frame):
    phi = phi_frame(data, frame)
    chi = data["chi"][frame]
    return np.abs(chi)**2, np.abs(phi)**2, np.abs(phi*chi[None, :])**2


def shifted_epsilon(data, frame, floor=1.0e-5):
    eps = np.asarray(data["epsilon"][frame], float).copy()
    rho = np.abs(data["chi"][frame])**2
    eps -= eps[int(np.argmax(rho))]
    eps[rho < floor*max(float(np.max(rho)), 1.0e-300)] = np.nan
    return eps


def frame_indices(nt, maximum):
    return np.unique(np.linspace(0, nt-1, min(nt, maximum)).round().astype(int))


def limits(arrays):
    values = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if values.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(values, [2, 98])
    return (float(lo), float(hi if hi > lo else lo+1.0))


def draw_static(data, outdir, frame=-1, dpi=180):
    if frame < 0:
        frame += len(data["times_fs"])
    r, R = data["r"], data["R"]
    rho, conditional, full = frame_fields(data, frame)
    eps = shifted_epsilon(data, frame)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), constrained_layout=True)
    axes[0].plot(R, eps, color="#275DAD", lw=2)
    axes[0].set_xlabel("nuclear R")
    axes[0].set_ylabel("shifted TDPES", color="#275DAD")
    axes[0].yaxis.set_major_formatter(NUMBER_FORMATTER)
    density_axis = axes[0].twinx()
    density_axis.plot(R, rho, color="black")
    density_axis.fill_between(R, rho, color="black", alpha=0.15)
    density_axis.set_ylabel(r"$|\chi|^2$")
    density_axis.yaxis.set_major_formatter(NUMBER_FORMATTER)
    axes[0].set_title("Nuclear field")
    conditional_image = axes[1].imshow(
        conditional, origin="lower", aspect="auto", cmap="inferno",
        extent=[R[0], R[-1], r[0], r[-1]],
    )
    axes[1].set_title(r"Conditional electron $|\Phi_R|^2$")
    full_image = axes[2].imshow(
        full, origin="lower", aspect="auto", cmap="viridis",
        extent=[R[0], R[-1], r[0], r[-1]],
    )
    axes[2].set_title(r"Full molecular $|\Phi_R\chi|^2$")
    for ax in axes[1:]:
        ax.set_xlabel("nuclear R")
        ax.set_ylabel("electron r")
    fig.colorbar(
        conditional_image, ax=axes[1], label=r"$|\Phi_R|^2$",
        pad=0.015, fraction=0.048, format=NUMBER_FORMATTER,
    )
    fig.colorbar(
        full_image, ax=axes[2], label=r"$|\Psi|^2$",
        pad=0.015, fraction=0.048, format=NUMBER_FORMATTER,
    )
    fig.suptitle(f"2-component EF   t={data['times_fs'][frame]:.4f} fs")
    path = outdir/"exact_factorization_three_panel.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"3분할 그림 저장: {path}")


def animate(data, outdir, fps=12, max_frames=180, dpi=120, fmt="mp4"):
    """Phi/chi/full Psi/TDPES를 서로 분리한 4-panel dynamics."""
    r, R, times = data["r"], data["R"], data["times_fs"]
    # 전자와 핵 위치를 같은 1D 공간 눈금에서 비교하도록 범위를 통일한다.
    common_min = min(float(r[0]), float(R[0]))
    common_max = max(float(r[-1]), float(R[-1]))
    frames = frame_indices(len(times), max_frames)
    eps_all = [shifted_epsilon(data, int(i)) for i in frames]
    eps_lo, eps_hi = limits(eps_all)
    phi_amp = chi_amp = phi_density = chi_density = full_max = 1.0e-12
    for frame in frames:
        phi = phi_frame(data, int(frame))
        chi = data["chi"][int(frame)]
        iR = int(np.argmax(np.abs(chi)**2))
        phi_line = phi[:, iR]
        _, _, full = frame_fields(data, int(frame))
        phi_amp = max(
            phi_amp, float(np.max(np.abs(phi_line.real))),
            float(np.max(np.abs(phi_line.imag))),
        )
        chi_amp = max(
            chi_amp, float(np.max(np.abs(chi.real))),
            float(np.max(np.abs(chi.imag))),
        )
        phi_density = max(phi_density, float(np.max(np.abs(phi_line)**2)))
        chi_density = max(chi_density, float(np.max(np.abs(chi)**2)))
        full_max = max(full_max, float(np.max(full)))

    first_frame = int(frames[0])
    first_phi = phi_frame(data, first_frame)
    first_chi = data["chi"][first_frame]
    first_iR = int(np.argmax(np.abs(first_chi)**2))
    first_phi_line = first_phi[:, first_iR]
    _, _, full = frame_fields(data, first_frame)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)

    def wave_panel(ax, grid, wave, amp_limit, density_limit, title, xlabel):
        real_line, = ax.plot(grid, wave.real, color="#2369BD", label="Real")
        imag_line, = ax.plot(grid, wave.imag, color="#D1495B", label="Imag")
        ax.set_ylim(-1.08*amp_limit, 1.08*amp_limit)
        ax.axhline(0, color="0.75", lw=0.7)
        ax.set_xlabel(xlabel)
        ax.set_xlim(common_min, common_max)
        ax.yaxis.set_major_formatter(NUMBER_FORMATTER)
        ax.set_ylabel("amplitude")
        ax.set_title(title)
        density_axis = ax.twinx()
        density_line, = density_axis.plot(
            grid, np.abs(wave)**2, color="black", lw=1.5, label="Density"
        )
        density_axis.set_ylim(0, 1.08*density_limit)
        density_axis.set_ylabel("density")
        density_axis.yaxis.set_major_formatter(NUMBER_FORMATTER)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        density_axis.legend(loc="upper right", frameon=False, fontsize=8)
        return real_line, imag_line, density_line

    phi_lines = wave_panel(
        axes[0, 0], r, first_phi_line, phi_amp, phi_density,
        rf"Electron $\Phi_R$  (R={R[first_iR]:.2f})", "electron r",
    )
    chi_lines = wave_panel(
        axes[0, 1], R, first_chi, chi_amp, chi_density,
        r"Nuclear $\chi(R,t)$", "nuclear R",
    )
    image_full = axes[1, 0].imshow(
        full, origin="lower", aspect="auto", cmap="viridis",
        extent=[R[0], R[-1], r[0], r[-1]], vmin=0, vmax=full_max,
    )
    axes[1, 0].set_title(r"Full $\Psi$: $|\Phi_R\chi|^2$")
    axes[1, 0].set_xlabel("nuclear R")
    axes[1, 0].set_ylabel("electron r")
    fig.colorbar(
        image_full, ax=axes[1, 0], label=r"$|\Psi|^2$",
        pad=0.015, fraction=0.048, format=NUMBER_FORMATTER,
    )
    eps_line, = axes[1, 1].plot(R, eps_all[0], color="#275DAD", lw=2)
    axes[1, 1].set_ylim(eps_lo, eps_hi)
    axes[1, 1].axhline(0, color="0.75", lw=0.7)
    axes[1, 1].set_title("TDPES")
    axes[1, 1].set_xlabel("nuclear R")
    axes[1, 1].set_ylabel("shifted energy")
    axes[1, 1].yaxis.set_major_formatter(NUMBER_FORMATTER)
    title = fig.suptitle(f"2-component EF   t={times[frames[0]]:.4f} fs")

    def update(number):
        frame = int(frames[number])
        phi = phi_frame(data, frame)
        chi = data["chi"][frame]
        iR = int(np.argmax(np.abs(chi)**2))
        phi_line = phi[:, iR]
        for artists, wave in ((phi_lines, phi_line), (chi_lines, chi)):
            artists[0].set_ydata(wave.real)
            artists[1].set_ydata(wave.imag)
            artists[2].set_ydata(np.abs(wave)**2)
        axes[0, 0].set_title(rf"Electron $\Phi_R$  (R={R[iR]:.2f})")
        _, _, full = frame_fields(data, frame)
        eps_line.set_ydata(shifted_epsilon(data, frame))
        image_full.set_data(full)
        title.set_text(f"2-component EF   t={times[frame]:.4f} fs")
        return (*phi_lines, *chi_lines, image_full, eps_line, title)

    movie = FuncAnimation(fig, update, frames=len(frames), blit=False)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/"exact_factorization_dynamics.mp4"
        movie.save(path, writer=FFMpegWriter(fps=fps, bitrate=2500), dpi=dpi)
    else:
        if fmt == "mp4":
            print("ffmpeg을 찾지 못해 GIF로 대신 저장합니다.")
        path = outdir/"exact_factorization_dynamics.gif"
        movie.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"4분할 dynamics 저장: {path}")


def run(args):
    data = np.load(args.archive, allow_pickle=True)
    requested_outdir = Path(args.outdir) if args.outdir else Path(args.archive).parent/"figures"
    outdir = dated_results_dir(requested_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    draw_static(data, outdir, frame=args.frame, dpi=args.dpi)
    if not args.no_animation:
        animate(
            data, outdir, fps=args.fps, max_frames=args.max_frames,
            dpi=args.animation_dpi, fmt=args.format,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--animation-dpi", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--no-animation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
