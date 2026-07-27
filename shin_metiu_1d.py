#!/usr/bin/env python3
"""1D Shin-Metiu nonadiabatic wavepacket dynamics.

The physical model follows Agostini et al., J. Chem. Phys. 142, 084303
(2015): one electron and one movable ion between two fixed ions.  The full
2D electron-nuclear TDSE is propagated with a split-operator FFT method, then
projected onto the instantaneous BO electronic states to monitor population
transfer and wavepacket splitting.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-shin-metiu")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.linalg import eigh
from scipy.special import erf

AU_PER_FS = 41.3413745758
STATE_COLORS = ("#0F766E", "#E76F51")
TEXT_COLOR = "#25313C"
MUTED_COLOR = "#667085"
GRID_COLOR = "#D7DCE2"
PANEL_COLOR = "#FFFFFF"
FIGURE_COLOR = "#F5F6F4"


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": "#98A2B3",
            "axes.linewidth": 0.9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": MUTED_COLOR,
            "ytick.color": MUTED_COLOR,
            "legend.fontsize": 10,
            "figure.facecolor": FIGURE_COLOR,
            "axes.facecolor": PANEL_COLOR,
            "savefig.facecolor": FIGURE_COLOR,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=4, width=0.8)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)


def occupied_R_slice(density: np.ndarray, pad_points: int = 8) -> slice:
    support = np.max(np.sum(density, axis=1), axis=0)
    occupied = np.flatnonzero(support > 0.0025 * support.max())
    if len(occupied) == 0:
        return slice(None)
    lo = max(0, int(occupied[0]) - pad_points)
    hi = min(len(support), int(occupied[-1]) + pad_points + 1)
    return slice(lo, hi)


def transfer_time_window(t: np.ndarray, populations: np.ndarray) -> tuple[float, float]:
    rate = np.abs(np.gradient(populations[:, 0], t))
    active = np.flatnonzero(rate > 0.12 * rate.max())
    if len(active) == 0:
        return float(t[0]), float(t[0])
    return float(t[active[0]]), float(t[active[-1]])


def state_cmap(name: str, color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ["#FCFCFA", color])


def erf_over_abs(x: np.ndarray, soft: float) -> np.ndarray:
    ax = np.abs(x)
    out = np.empty_like(ax, dtype=float)
    small = ax < 1.0e-10
    out[small] = 2.0 / (np.sqrt(np.pi) * soft)
    out[~small] = erf(ax[~small] / soft) / ax[~small]
    return out


def shin_metiu_potential(r: np.ndarray, R: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    rr = r[:, None]
    RR = R[None, :]
    half_L = 0.5 * args.L
    ion_repulsion = 1.0 / np.abs(half_L - RR) + 1.0 / np.abs(half_L + RR)
    electron_attraction = (
        erf_over_abs(RR - rr, args.Rf)
        + erf_over_abs(rr - half_L, args.Rl)
        + erf_over_abs(rr + half_L, args.Rr)
    )
    return ion_repulsion - electron_attraction


def compute_bo_states(
    r: np.ndarray, R: np.ndarray, potential: np.ndarray, n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    dr = r[1] - r[0]
    nr = len(r)
    kr = 2.0 * np.pi * np.fft.fftfreq(nr, d=dr)
    eye = np.eye(nr)
    kinetic = np.fft.ifft(np.fft.fft(eye, axis=0) * (0.5 * kr[:, None] ** 2), axis=0).real
    kinetic = 0.5 * (kinetic + kinetic.T)
    energies = np.empty((n_states, len(R)))
    states = np.empty((n_states, len(R), nr))
    raw_prev = np.zeros((n_states, nr))

    for iR in range(len(R)):
        vals, vecs = eigh(
            kinetic + np.diag(potential[:, iR]),
            subset_by_index=(0, n_states - 1),
            check_finite=False,
        )
        energies[:, iR] = vals
        for s in range(n_states):
            vec = vecs[:, s]
            if iR > 0 and np.dot(raw_prev[s], vec) < 0.0:
                vec = -vec
            raw_prev[s] = vec
            states[s, iR, :] = vec / np.sqrt(dr)
    return energies, states


def normalize(psi: np.ndarray, dr: float, dR: float) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dr * dR)
    return psi / norm


def project_bo(
    psi: np.ndarray, bo_states: np.ndarray, dr: float, dR: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeff = np.einsum("sRr,Rr->sR", bo_states.conj(), psi.T) * dr
    density = np.abs(coeff) ** 2
    population = np.sum(density, axis=1) * dR
    return coeff, density, population


def run_simulation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r = np.linspace(args.r_min, args.r_max, args.nr, endpoint=False)
    R = np.linspace(args.R_min, args.R_max, args.nR, endpoint=False)
    dr = r[1] - r[0]
    dR = R[1] - R[0]
    dt = args.dt_au
    n_steps = int(round(args.t_final_fs * AU_PER_FS / dt))
    save_every = max(1, args.save_every)

    potential = shin_metiu_potential(r, R, args)
    print(f"Computing {args.n_states} BO states on {args.nR} nuclear grid points...")
    bo_energies, bo_states = compute_bo_states(r, R, potential, args.n_states)

    sigma = args.sigma if args.sigma > 0.0 else 1.0 / np.sqrt(2.85)
    chi0 = (1.0 / (np.pi * sigma**2)) ** 0.25
    chi0 = chi0 * np.exp(-0.5 * ((R - args.R0) / sigma) ** 2 + 1j * args.P0 * (R - args.R0))
    psi = bo_states[args.initial_state].T * chi0[None, :]
    psi = normalize(psi, dr, dR)

    kr = 2.0 * np.pi * np.fft.fftfreq(args.nr, d=dr)
    kR = 2.0 * np.pi * np.fft.fftfreq(args.nR, d=dR)
    kinetic = kr[:, None] ** 2 / 2.0 + kR[None, :] ** 2 / (2.0 * args.mass)
    phase_T_half = np.exp(-0.5j * dt * kinetic)
    phase_V = np.exp(-1j * dt * potential)

    save_steps = list(range(0, n_steps + 1, save_every))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    frames = len(save_steps)
    times_fs = np.empty(frames)
    populations = np.empty((frames, args.n_states))
    projected_density = np.empty((frames, args.n_states, args.nR))
    nuclear_density = np.empty((frames, args.nR))
    norm = np.empty(frames)

    def save_frame(frame: int, step: int) -> None:
        _, density, pop = project_bo(psi, bo_states, dr, dR)
        times_fs[frame] = step * dt / AU_PER_FS
        projected_density[frame] = density
        populations[frame] = pop
        nuclear_density[frame] = np.sum(np.abs(psi) ** 2, axis=0) * dr
        norm[frame] = np.sum(np.abs(psi) ** 2) * dr * dR

    save_frame(0, 0)
    frame = 1
    for step in range(1, n_steps + 1):
        psi = np.fft.ifftn(np.fft.fftn(psi) * phase_T_half)
        psi *= phase_V
        psi = np.fft.ifftn(np.fft.fftn(psi) * phase_T_half)
        if frame < frames and step == save_steps[frame]:
            save_frame(frame, step)
            frame += 1
        if step % max(save_every * 20, 1) == 0:
            print(f"step {step:6d}/{n_steps}  t={step * dt / AU_PER_FS:7.3f} fs")

    np.savez_compressed(
        outdir / "shin_metiu_1d_results.npz",
        r=r,
        R=R,
        times_fs=times_fs,
        populations=populations,
        projected_density=projected_density,
        nuclear_density=nuclear_density,
        bo_energies=bo_energies,
        norm=norm,
        args=np.array([vars(args)], dtype=object),
    )
    return {
        "r": r,
        "R": R,
        "times_fs": times_fs,
        "populations": populations,
        "projected_density": projected_density,
        "nuclear_density": nuclear_density,
        "bo_energies": bo_energies,
        "norm": norm,
    }


def load_results(path: Path) -> dict[str, np.ndarray]:
    keys = (
        "r",
        "R",
        "times_fs",
        "populations",
        "projected_density",
        "nuclear_density",
        "bo_energies",
        "norm",
    )
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in keys}


def plot_summary(data: dict[str, np.ndarray], outdir: Path) -> None:
    configure_plot_style()
    R = data["R"]
    t = data["times_fs"]
    dens = data["projected_density"]
    pop = data["populations"]
    energies = data["bo_energies"]
    r_slice = occupied_R_slice(dens)
    R_view = R[r_slice]
    crossing_R = R[np.argmin(energies[1] - energies[0])]
    transfer_start, transfer_end = transfer_time_window(t, pop)

    fig = plt.figure(figsize=(13.4, 8.5), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, height_ratios=(1.06, 0.94))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    fig.suptitle("1D Shin-Metiu nonadiabatic wavepacket dynamics", fontsize=18,
                 fontweight="bold", color=TEXT_COLOR)

    cmaps = (state_cmap("state1", STATE_COLORS[0]), state_cmap("state2", STATE_COLORS[1]))
    for s, ax in enumerate(axes[:2]):
        cropped = dens[:, s, r_slice].T
        positive = cropped[cropped > 0.0]
        vmax = np.quantile(positive, 0.995) if positive.size else 1.0
        im = ax.imshow(
            cropped,
            origin="lower",
            aspect="auto",
            extent=[t[0], t[-1], R_view[0], R_view[-1]],
            cmap=cmaps[s],
            norm=PowerNorm(gamma=0.62, vmin=0.0, vmax=vmax),
            interpolation="bilinear",
        )
        ax.axhline(crossing_R, color="#7A5C00", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(t[-1] * 0.985, crossing_R + 0.16, "avoided crossing", ha="right", va="bottom",
                fontsize=9, color="#7A5C00")
        ax.set_title(f"BO state {s + 1} projected density")
        ax.set_xlabel("time (fs)")
        ax.set_ylabel(r"nuclear coordinate $R$ ($a_0$)")
        ax.xaxis.set_major_locator(MaxNLocator(7))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, pad=0.018, fraction=0.046)
        cbar.set_label(r"$|F_%d(R,t)|^2$" % (s + 1), fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    ax = axes[2]
    ax.axvspan(transfer_start, transfer_end, color="#F2C94C", alpha=0.16, linewidth=0)
    for s in range(pop.shape[1]):
        color = STATE_COLORS[s] if s < len(STATE_COLORS) else None
        ax.plot(t, pop[:, s], color=color, linewidth=2.6, label=f"BO state {s + 1}")
        ax.fill_between(t, 0.0, pop[:, s], color=color, alpha=0.055)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("BO population")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(loc="center right", frameon=False)
    ax.text(0.02, 0.07, f"final:  P1 = {pop[-1, 0]:.3f}   P2 = {pop[-1, 1]:.3f}",
            transform=ax.transAxes, color=TEXT_COLOR, fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID_COLOR})
    style_axis(ax)

    ax = axes[3]
    norm_error_scaled = (data["norm"] - 1.0) * 1.0e12
    max_error = float(np.max(np.abs(norm_error_scaled)))
    y_limit = max(0.5, 1.15 * max_error)
    ax.axhline(0.0, color="#98A2B3", linewidth=1.0)
    ax.plot(t, norm_error_scaled, color="#6C5CE7", linewidth=2.2)
    ax.fill_between(t, 0.0, norm_error_scaled, color="#6C5CE7", alpha=0.12)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel(r"norm deviation $N(t)-1$  ($10^{-12}$)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-y_limit, y_limit)
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(0.03, 0.88, rf"max $|N-1|$ = {max_error:.2f} $\times 10^{{-12}}$",
            transform=ax.transAxes, color=TEXT_COLOR, fontsize=10)
    style_axis(ax)

    fig.savefig(outdir / "shin_metiu_summary.png", dpi=200)
    fig.savefig(outdir / "shin_metiu_summary.pdf", dpi=200)
    plt.close(fig)


def make_animation(data: dict[str, np.ndarray], outdir: Path, fps: int, max_frames: int) -> None:
    configure_plot_style()
    R = data["R"]
    t = data["times_fs"]
    dens = data["projected_density"]
    pop = data["populations"]
    energies = data["bo_energies"]
    total_density = data["nuclear_density"]

    stride = max(1, int(np.ceil(len(t) / max_frames)))
    frame_ids = np.arange(0, len(t), stride)
    r_slice = occupied_R_slice(dens)
    R_view = R[r_slice]
    E_view = energies[:2, r_slice]
    crossing_R = R[np.argmin(energies[1] - energies[0])]
    transfer_start, transfer_end = transfer_time_window(t, pop)
    e_min = float(E_view.min())
    e_max = float(E_view.max())
    e_margin = 0.10 * (e_max - e_min)
    density_max = float(np.max(dens[:, :2, r_slice]))
    density_ylim = 1.10 * max(density_max, 1.0e-6)
    overlay_scale = 0.16 * (e_max - e_min) / max(density_max, 1.0e-12)

    fig = plt.figure(figsize=(13.2, 7.2), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, width_ratios=(1.24, 1.0), height_ratios=(1.0, 0.86))
    ax_energy = fig.add_subplot(grid[0, 0])
    ax_density = fig.add_subplot(grid[1, 0], sharex=ax_energy)
    ax_pop = fig.add_subplot(grid[:, 1])
    fig.suptitle("1D Shin-Metiu nonadiabatic dynamics", fontsize=18,
                 fontweight="bold", color=TEXT_COLOR)

    for s in range(2):
        ax_energy.plot(R_view, E_view[s], color=STATE_COLORS[s], linewidth=1.8,
                       label=f"BO state {s + 1}")
    energy_density_lines = [
        ax_energy.plot([], [], color=STATE_COLORS[s], linewidth=3.0, alpha=0.9)[0]
        for s in range(2)
    ]
    ax_energy.axvline(crossing_R, color="#7A5C00", linestyle="--", linewidth=1.0, alpha=0.75)
    ax_energy.set_title("Wavepacket projected onto BO surfaces")
    ax_energy.set_ylabel("energy (Ha)")
    ax_energy.set_xlim(R_view[0], R_view[-1])
    ax_energy.set_ylim(e_min - e_margin, e_max + e_margin)
    ax_energy.yaxis.set_major_locator(MaxNLocator(6))
    ax_energy.legend(loc="best", frameon=False)
    style_axis(ax_energy)

    density_lines = [
        ax_density.plot([], [], color=STATE_COLORS[s], linewidth=2.5,
                        label=rf"$|F_{s + 1}(R,t)|^2$")[0]
        for s in range(2)
    ]
    total_line, = ax_density.plot([], [], color=TEXT_COLOR, linewidth=1.6,
                                  linestyle="--", alpha=0.75, label="total nuclear density")
    ax_density.axvline(crossing_R, color="#7A5C00", linestyle="--", linewidth=1.0, alpha=0.75)
    ax_density.set_title("Instantaneous nuclear density")
    ax_density.set_xlabel(r"nuclear coordinate $R$ ($a_0$)")
    ax_density.set_ylabel("density")
    ax_density.set_ylim(0.0, density_ylim)
    ax_density.xaxis.set_major_locator(MaxNLocator(7))
    ax_density.yaxis.set_major_locator(MaxNLocator(5))
    ax_density.legend(loc="upper left", frameon=False, ncol=2)
    style_axis(ax_density)

    for s in range(2):
        ax_pop.plot(t, pop[:, s], color=STATE_COLORS[s], linewidth=1.6, alpha=0.24)
    active_population_lines = [
        ax_pop.plot([], [], color=STATE_COLORS[s], linewidth=2.8,
                    label=f"BO state {s + 1}")[0]
        for s in range(2)
    ]
    population_markers = [
        ax_pop.plot([], [], marker="o", markersize=7, color=STATE_COLORS[s],
                    markeredgecolor="white", markeredgewidth=1.2)[0]
        for s in range(2)
    ]
    ax_pop.axvspan(transfer_start, transfer_end, color="#F2C94C", alpha=0.14, linewidth=0)
    time_marker = ax_pop.axvline(t[0], color=TEXT_COLOR, linewidth=1.1, linestyle="--")
    status = ax_pop.text(0.04, 0.05, "", transform=ax_pop.transAxes, fontsize=11,
                         color=TEXT_COLOR,
                         bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": GRID_COLOR})
    ax_pop.set_title("Electronic-state populations")
    ax_pop.set_xlabel("time (fs)")
    ax_pop.set_ylabel("BO population")
    ax_pop.set_xlim(t[0], t[-1])
    ax_pop.set_ylim(0.0, 1.02)
    ax_pop.xaxis.set_major_locator(MaxNLocator(7))
    ax_pop.yaxis.set_major_locator(MaxNLocator(6))
    ax_pop.legend(loc="center right", frameon=False)
    style_axis(ax_pop)

    fills: list = [None, None]

    def update(frame_idx: int):
        i = frame_ids[frame_idx]
        artists = []
        for s in range(2):
            state_density = dens[i, s, r_slice]
            energy_density_lines[s].set_data(R_view, E_view[s] + overlay_scale * state_density)
            density_lines[s].set_data(R_view, state_density)
            active_population_lines[s].set_data(t[: i + 1], pop[: i + 1, s])
            population_markers[s].set_data([t[i]], [pop[i, s]])
            if fills[s] is not None:
                fills[s].remove()
            fills[s] = ax_density.fill_between(R_view, 0.0, state_density,
                                               color=STATE_COLORS[s], alpha=0.13)
            artists.extend((energy_density_lines[s], density_lines[s],
                            active_population_lines[s], population_markers[s], fills[s]))
        total_line.set_data(R_view, total_density[i, r_slice])
        time_marker.set_xdata([t[i], t[i]])
        status.set_text(f"t = {t[i]:5.2f} fs\nP1 = {pop[i, 0]:.3f}\nP2 = {pop[i, 1]:.3f}")
        artists.extend((total_line, time_marker, status))
        return artists

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / fps, blit=False)
    anim.save(outdir / "shin_metiu_wavepacket.gif", writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results/shin_metiu_1d")
    p.add_argument("--nr", type=int, default=192)
    p.add_argument("--nR", type=int, default=192)
    p.add_argument("--r-min", type=float, default=-15.0)
    p.add_argument("--r-max", type=float, default=15.0)
    p.add_argument("--R-min", type=float, default=-8.0)
    p.add_argument("--R-max", type=float, default=8.0)
    p.add_argument("--dt-au", type=float, default=0.25)
    p.add_argument("--t-final-fs", type=float, default=35.0)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--initial-state", type=int, default=1, help="Zero-based BO state index.")
    p.add_argument("--mass", type=float, default=1836.0)
    p.add_argument("--L", type=float, default=19.0)
    p.add_argument("--Rf", type=float, default=5.0)
    p.add_argument("--Rl", type=float, default=3.1)
    p.add_argument("--Rr", type=float, default=4.0)
    p.add_argument("--R0", type=float, default=-4.0)
    p.add_argument("--P0", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.0, help="Default is 1/sqrt(2.85) a0.")
    p.add_argument("--gif-fps", type=int, default=12)
    p.add_argument("--max-gif-frames", type=int, default=160)
    p.add_argument("--no-animation", action="store_true")
    p.add_argument("--render-from", type=Path,
                   help="Skip propagation and render plots from an existing results NPZ file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_results(args.render_from) if args.render_from else run_simulation(args)
    plot_summary(data, outdir)
    if not args.no_animation:
        make_animation(data, outdir, args.gif_fps, args.max_gif_frames)
    print(f"Saved results in {outdir}")
    print("Final BO populations:", " ".join(f"P{s + 1}={p:.4f}" for s, p in enumerate(data["populations"][-1])))
    print(f"Maximum norm error: {np.max(np.abs(data['norm'] - 1.0)):.3e}")


if __name__ == "__main__":
    main()
