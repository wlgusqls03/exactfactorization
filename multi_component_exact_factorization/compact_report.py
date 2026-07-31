#!/usr/bin/env python3
"""Create a small, question-oriented report for one completed MCEF run.

The standard report deliberately contains only four static figures and one
animation.  Each figure answers one question: what moved, which BO components
formed, what the exact potentials did, and whether the discretization remained
trustworthy.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import SymLogNorm
import numpy as np

from result_paths import dated_results_dir

from .dynamics_analysis import normalized_marginals, moments
from .potential_analysis import gauge_invariant_diagnostics, nonadiabatic_couplings
from .visualize import (
    NUMBER_FORMATTER,
    archive_arguments,
    common_position_limits,
    robust_limits,
    selected_frames,
)


COLORS = ("#2878B5", "#E07A2D", "#3A9654", "#B05279", "#7B61A8", "#8C6D31")


def _joint_normalized(joint, q, R):
    dq, dR = float(q[1]-q[0]), float(R[1]-R[0])
    norm = np.sum(joint, axis=(1, 2))*dq*dR
    return joint/np.maximum(norm[:, None, None], 1.0e-300)


def _support_mask(values, density, floor):
    cutoff = floor*max(float(np.max(density)), 1.0e-300)
    return np.where(density >= cutoff, values, np.nan)


def _shift_at_peak(values, density):
    shifted = np.asarray(values, float).copy()
    peak = np.unravel_index(int(np.argmax(density)), density.shape)
    return shifted-shifted[peak]


def _style_time_axis(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("time (fs)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18, linewidth=0.7)


def _stress_onset(data):
    """First saved time where the Phi correction becomes order one."""
    if "max_abs_gamma_phi" not in data.files:
        return None
    values = np.asarray(data["max_abs_gamma_phi"])
    indices = np.flatnonzero(values > 1.0)
    return None if not len(indices) else float(data["times_fs"][indices[0]])


def _mark_stress(ax, onset, label=False):
    if onset is None:
        return
    ax.axvline(
        onset, color="#C43C39", lw=1.1, ls=":", alpha=0.9,
        label=(r"numerical stress: $|\gamma_\Phi|>1$" if label else None),
    )


def _label_panels(axes):
    """Add stable A--F anchors used by the accompanying interpretation guide."""
    for letter, ax in zip("ABCDEF", np.asarray(axes, dtype=object).flat):
        ax.text(
            -0.075, 1.035, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left",
        )


def _footer(data):
    options = archive_arguments(data)
    return (
        rf"initial BO $n={int(options.get('electron_excitation', 0))}$; "
        rf"$q_0={options.get('q0', np.nan):.2f}$, "
        rf"$R_0={options.get('R0', np.nan):.2f}\,a_0$; "
        rf"$n_q={len(data['q'])}$, $n_R={len(data['R'])}$"
    )


def plot_particle_motion(data, densities, means, widths, outdir, dpi):
    """Three position-time marginals and their compact moment summary."""
    times = data["times_fs"]
    grids = (data["x"], data["q"], data["R"])
    names = ("Electron", "Proton", "Heavy nucleus")
    symbols = ("x", "q", "R")
    options = archive_arguments(data)
    common_min, common_max = common_position_limits(data)
    box_limits = (
        (float(options.get("x_min", grids[0][0])), float(options.get("x_max", grids[0][-1]))),
        (float(options.get("q_min", grids[1][0])), float(options.get("q_max", grids[1][-1]))),
        (float(options.get("R_min", grids[2][0])), float(options.get("R_max", grids[2][-1]))),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), constrained_layout=True)
    onset = _stress_onset(data)
    for index, (ax, grid, density, name, symbol, limits) in enumerate(zip(
        axes.flat[:3], grids, densities, names, symbols, box_limits
    )):
        ax.set_facecolor("#E8ECEF")
        image = ax.pcolormesh(
            times, grid, density.T, shading="nearest", cmap="magma",
            rasterized=True,
        )
        ax.set_ylim(common_min, common_max)
        if index > 0:
            # A white line alone disappears on the uncomputed grey area.  The
            # dark outline keeps the exact periodic-box boundary visible on
            # both the density map and the outside background.
            for boundary in limits:
                line = ax.axhline(
                    boundary, color="white", lw=1.35, ls="--", zorder=5,
                )
                line.set_path_effects([
                    path_effects.Stroke(linewidth=2.5, foreground="0.25"),
                    path_effects.Normal(),
                ])
            ax.text(
                0.985, 0.025,
                f"dashed: {symbol}-grid [{limits[0]:.2f}, {limits[1]:.2f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                color="0.2", bbox=dict(fc="white", ec="none", alpha=0.78, pad=2),
            )
        ax.plot(times, means[symbols.index(symbol)], color="white", lw=1.4)
        ax.set_title(f"{name} probability", loc="left", fontweight="semibold")
        ax.set_xlabel("time (fs)")
        ax.set_ylabel(rf"position ${symbol}$ ($a_0$)")
        _mark_stress(ax, onset)
        fig.colorbar(
            image, ax=ax, label="probability density", pad=0.012,
            format=NUMBER_FORMATTER,
        )

    ax = axes[1, 1]
    for i, (mean, width, name, symbol, color) in enumerate(zip(
        means, widths, names, symbols, COLORS
    )):
        ax.plot(times, mean-mean[0], color=color, lw=2.0, label=rf"$\Delta\langle {symbol}\rangle$")
        ax.plot(
            times, width-width[0], color=color, lw=1.5, ls="--",
            label=rf"$\Delta\sigma_{symbol}$",
        )
    ax.axhline(0.0, color="0.55", lw=0.8)
    _mark_stress(ax, onset, label=True)
    _style_time_axis(ax, "How the centers and widths changed", r"change ($a_0$)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _label_panels(axes)
    fig.suptitle(
        "1 | What moved?  Density maps (A-C) -> compact motion summary (D)",
        fontsize=14, fontweight="bold",
    )
    fig.supxlabel(_footer(data), fontsize=9, color="0.35")
    path = outdir/"01_particle_motion.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Compact report: {path}")


def _gap_nac_panel(ax, q, R, gap, nac, joint, pair, support_floor):
    shown = _support_mask(gap, joint, support_floor)
    image = ax.imshow(
        shown, origin="lower", aspect="auto",
        extent=[R[0], R[-1], q[0], q[-1]], cmap="viridis",
    )
    cutoff = support_floor*float(np.max(joint))
    ax.contour(R, q, joint, levels=[cutoff], colors="white", linewidths=1.2)
    finite_nac = nac[np.isfinite(shown)]
    if finite_nac.size:
        levels = np.unique(np.percentile(finite_nac, [35.0, 65.0, 85.0]))
        if len(levels):
            contours = ax.contour(
                R, q, np.where(np.isfinite(shown), nac, np.nan),
                levels=levels, colors="#F6C85F", linewidths=1.0,
            )
            ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    ax.set_title(
        rf"Pair {pair}: gap color + $|d^q_{{{pair}}}|$ contours",
        loc="left", fontweight="semibold",
    )
    ax.set_xlabel(r"heavy coordinate $R$ ($a_0$)")
    ax.set_ylabel(r"proton coordinate $q$ ($a_0$)")
    return image


def plot_electronic_transitions(
    data, means, widths, rearranged, decomposition, nac, joint, frame,
    support_floor, outdir, dpi,
):
    """BO composition, correlated motion, and two occupied gap/NAC maps."""
    energies, _resolved, populations, residual = decomposition
    times, q, R = data["times_fs"], data["q"], data["R"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), constrained_layout=True)
    onset = _stress_onset(data)

    ax = axes[0, 0]
    for state in range(populations.shape[1]):
        ax.plot(times, populations[:, state], lw=1.9, color=COLORS[state % len(COLORS)], label=rf"$P_{state}$")
    ax.plot(times, residual, color="0.3", ls="--", label="outside basis")
    ax.set_ylim(-0.02, 1.02)
    _mark_stress(ax, onset, label=True)
    _style_time_axis(ax, "BO-state composition", "population")
    ax.legend(frameon=False, ncol=4, fontsize=8)

    ax = axes[0, 1]
    ax.plot(times, means[1]-means[1][0], lw=2, color=COLORS[0], label=r"$\Delta\langle q\rangle$ ($a_0$)")
    ax.plot(times, widths[1]-widths[1][0], lw=2, color=COLORS[1], label=r"$\Delta\sigma_q$ ($a_0$)")
    ax.plot(times, rearranged, lw=2, color=COLORS[2], label=r"electron $D_{\rm rearr}$")
    ax.axhline(0.0, color="0.6", lw=0.8)
    _mark_stress(ax, onset)
    _style_time_axis(ax, "Motion that accompanies the state mixing", "change")
    ax.legend(frameon=False, fontsize=8)

    for ax, lower, upper in ((axes[1, 0], 0, 1), (axes[1, 1], 1, 2)):
        pair = f"{lower}{upper}"
        image = _gap_nac_panel(
            ax, q, R, energies[upper]-energies[lower],
            nac[f"nac_{pair}_q"], joint[frame], pair, support_floor,
        )
        fig.colorbar(image, ax=ax, label="BO energy gap (Hartree)", pad=0.012)

    _label_panels(axes)
    fig.suptitle(
        f"2 | Did electronic mixing occur, and where could it come from?\n"
        f"composition (A) -> correlated motion (B) -> occupied gap/NAC paths (C-D), t={times[frame]:.3f} fs",
        fontsize=14, fontweight="bold",
    )
    path = outdir/"02_electronic_transitions.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Compact report: {path}")


def plot_exact_potentials(data, diagnostics, joint, frame, support_floor, outdir, dpi):
    """The two nested TDPESs, connections, and gauge-invariant force."""
    q, R, times = data["q"], data["R"], data["times_fs"]
    density = joint[frame]
    heavy = np.sum(density, axis=0)*float(q[1]-q[0])
    extent = [R[0], R[-1], q[0], q[-1]]
    eps1 = _support_mask(_shift_at_peak(data["epsilon_1"][frame], density), density, support_floor)
    avec = _support_mask(data["a"][frame], density, support_floor)
    force = _support_mask(diagnostics["force_q"][frame], density, support_floor)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), constrained_layout=True)

    for ax, values, title, label, cmap, symmetric in (
        (axes[0, 0], eps1, r"First TDPES $\epsilon^{(1)}(q,R,t)$", "energy (Hartree)", "viridis", False),
        (axes[0, 1], avec, r"Connection $a(q,R,t)$", r"momentum ($a_0^{-1}$)", "coolwarm", True),
        (axes[1, 0], force, r"Combined result: $F_q=-\partial_q\epsilon^{(1)}+\partial_ta$", r"force (Hartree/$a_0$)", "coolwarm", True),
    ):
        finite = values[np.isfinite(values)]
        image_kwargs = {}
        if symmetric and "force" in label:
            maximum = max(float(np.nanmax(np.abs(finite))), 1.0e-14)
            typical = max(float(np.nanpercentile(np.abs(finite), 80.0)), 1.0e-6)
            image_kwargs["norm"] = SymLogNorm(
                linthresh=max(0.2*typical, 1.0e-5),
                linscale=0.8, vmin=-maximum, vmax=maximum, base=10,
            )
            limits = None
            title += rf"  (max $|F_q|={maximum:.1f}$)"
        elif symmetric:
            bound = max(float(np.nanpercentile(np.abs(finite), 98.0)), 1.0e-14)
            limits = (-bound, bound)
        else:
            limits = (float(np.nanpercentile(finite, 2.0)), float(np.nanpercentile(finite, 98.0)))
        if limits is not None:
            image_kwargs.update(vmin=limits[0], vmax=limits[1])
        image = ax.imshow(
            values, origin="lower", aspect="auto", extent=extent, cmap=cmap,
            **image_kwargs,
        )
        ax.contour(
            R, q, density, levels=[support_floor*float(np.max(density))],
            colors="white", linewidths=1.1,
        )
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_xlabel(r"heavy coordinate $R$ ($a_0$)")
        ax.set_ylabel(r"proton coordinate $q$ ($a_0$)")
        fig.colorbar(image, ax=ax, label=label, pad=0.012)

    ax = axes[1, 1]
    eps2 = np.asarray(data["epsilon_2"][frame], float)
    peak = int(np.argmax(heavy))
    eps2 = eps2-eps2[peak]
    mask = heavy >= support_floor*max(float(np.max(heavy)), 1.0e-300)
    eps2 = np.where(mask, eps2, np.nan)
    alpha = np.where(mask, data["alpha"][frame], np.nan)
    energy_line, = ax.plot(R, eps2, color=COLORS[0], lw=2, label=r"$\epsilon^{(2)}$ (left axis)")
    ax.set_xlabel(r"heavy coordinate $R$ ($a_0$)")
    ax.set_ylabel("shifted energy (Hartree)", color=COLORS[0])
    ax.tick_params(axis="y", labelcolor=COLORS[0])
    connection_axis = ax.twinx()
    alpha_line, = connection_axis.plot(R, alpha, color=COLORS[3], lw=1.8, label=r"$\alpha$ (right axis)")
    connection_axis.fill_between(R, 0.0, heavy/np.max(heavy), color="0.45", alpha=0.15, label="heavy density (scaled)")
    connection_axis.set_ylabel(r"$\alpha$ ($a_0^{-1}$)", color=COLORS[3])
    connection_axis.tick_params(axis="y", labelcolor=COLORS[3])
    ax.set_title(r"Second TDPES $\epsilon^{(2)}(R,t)$ and $\alpha(R,t)$", loc="left", fontweight="semibold")
    ax.legend(handles=[energy_line, alpha_line], frameon=False, fontsize=8, loc="upper left")

    _label_panels(axes)
    fig.suptitle(
        f"3 | How do the exact potentials act?  Scalar (A) + connection (B) -> force (C); outer level (D)\n"
        f"t={times[frame]:.3f} fs; occupied region only; scalar offsets removed",
        fontsize=14, fontweight="bold",
    )
    path = outdir/"03_exact_potentials.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Compact report: {path}")


def _edge_mass(density, spacing, points=5):
    return (np.sum(density[:, :points], axis=1)+np.sum(density[:, -points:], axis=1))*spacing


def plot_numerical_reliability(data, diagnostics, densities, outdir, dpi):
    """Constraint load, field roughness, preserved identities, and box edges."""
    times = data["times_fs"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), constrained_layout=True)
    onset = _stress_onset(data)

    ax = axes[0, 0]
    for key, label, color in (
        ("pnc_projection_correction", "PNC redistribution", COLORS[0]),
        ("max_abs_gamma_phi", r"max $|\gamma_\Phi|$", COLORS[1]),
        ("max_abs_gamma_lam", r"max $|\gamma_\Lambda|$", COLORS[2]),
    ):
        if key in data.files:
            ax.semilogy(times, np.maximum(data[key], 1.0e-18), color=color, label=label)
    _style_time_axis(ax, "How hard the constraint correction works", "magnitude (log scale)")
    _mark_stress(ax, onset, label=True)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for key, label, color in (
        ("support_rms_a", "RMS a", COLORS[0]),
        ("support_rms_b", "RMS b", COLORS[1]),
        ("support_rms_alpha", "RMS alpha", COLORS[2]),
        ("support_rms_force_q", "RMS proton force", COLORS[3]),
        ("support_rms_force_R", "RMS heavy force", COLORS[4]),
    ):
        ax.semilogy(times, np.maximum(diagnostics[key], 1.0e-18), label=label, color=color)
    _style_time_axis(ax, "Roughness inside the occupied region", "weighted RMS (log scale)")
    _mark_stress(ax, onset)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 0]
    if "max_raw_rate_phi" in data.files:
        ax.semilogy(times, np.maximum(data["max_raw_rate_phi"], 1.0e-20), color=COLORS[1], label=r"raw $r_\Phi$")
        ax.semilogy(times, np.maximum(data["max_corrected_rate_phi"], 1.0e-20), color=COLORS[0], label=r"corrected $r_\Phi$")
        ax.semilogy(times, np.maximum(data["max_corrected_rate_lam"], 1.0e-20), color=COLORS[2], label=r"corrected $r_\Lambda$")
    ax.semilogy(times, np.maximum(np.abs(data["norm"]-1.0), 1.0e-20), color="0.2", ls="--", label=r"$|N_\Psi-1|$")
    _style_time_axis(ax, "What is preserved after correction", "error / rate (log scale)")
    _mark_stress(ax, onset)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    q, R = data["q"], data["R"]
    q_edge = _edge_mass(densities[1], float(q[1]-q[0]))
    R_edge = _edge_mass(densities[2], float(R[1]-R[0]))
    ax.semilogy(times, np.maximum(q_edge, 1.0e-20), color=COLORS[0], label="proton: outer 5 points")
    ax.semilogy(times, np.maximum(R_edge, 1.0e-20), color=COLORS[2], label="heavy: outer 5 points")
    options = archive_arguments(data)
    dq, dR = float(q[1]-q[0]), float(R[1]-R[0])
    text = (
        rf"initial $\sigma_q/dq={options.get('proton_sigma', np.nan)/dq:.2f}$" "\n"
        rf"initial $\sigma_R/dR={options.get('heavy_sigma', np.nan)/dR:.2f}$"
    )
    ax.text(0.03, 0.05, text, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))
    _style_time_axis(ax, "Is the nuclear box wide and fine enough?", "edge probability (log scale)")
    _mark_stress(ax, onset)
    ax.legend(frameon=False, fontsize=8)

    _label_panels(axes)
    fig.suptitle(
        "4 | Can this trajectory be trusted?  Correction load (A) -> roughness (B) -> hidden error (C) -> grid check (D)",
        fontsize=13.5, fontweight="bold",
    )
    path = outdir/"04_numerical_reliability.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Compact report: {path}")


def make_overview_animation(
    data, decomposition, electron, joint, support_floor, outdir, fps,
    max_frames, dpi, fmt,
):
    """One synchronized movie: electron, nuclei, BO populations, and TDPES1."""
    times, x, q, R = data["times_fs"], data["x"], data["q"], data["R"]
    _energies, _resolved, populations, residual = decomposition
    frames = selected_frames(len(times), min(max_frames, len(times)))
    displayed_eps = []
    for frame in frames:
        density = joint[frame]
        values = _shift_at_peak(data["epsilon_1"][frame], density)
        displayed_eps.append(_support_mask(values, density, support_floor))
    eps_limits = robust_limits(displayed_eps)
    joint_max = max(float(np.max(joint[i])) for i in frames)
    electron_max = max(float(np.max(electron[i])) for i in frames)
    extent = [R[0], R[-1], q[0], q[-1]]
    first = int(frames[0])

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), constrained_layout=True)
    axes[0, 0].plot(
        x, electron[0], color="0.65", ls="--", lw=1.3, label="initial"
    )
    electron_line, = axes[0, 0].plot(x, electron[first], color=COLORS[0], lw=2.2, label="current")
    axes[0, 0].set(xlabel=r"electron $x$ ($a_0$)", ylabel="probability density", ylim=(0, 1.08*electron_max))
    axes[0, 0].set_title("Electron marginal", loc="left", fontweight="semibold")
    axes[0, 0].legend(frameon=False)

    nuclear_image = axes[0, 1].imshow(joint[first], origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=joint_max)
    axes[0, 1].set(xlabel=r"heavy $R$ ($a_0$)", ylabel=r"proton $q$ ($a_0$)")
    axes[0, 1].set_title("Joint nuclear density", loc="left", fontweight="semibold")
    fig.colorbar(nuclear_image, ax=axes[0, 1], label=r"$\rho_{qR}$", pad=0.012)

    for state in range(populations.shape[1]):
        axes[1, 0].plot(times, populations[:, state], color=COLORS[state % len(COLORS)], lw=1.8, label=rf"$P_{state}$")
    axes[1, 0].plot(times, residual, color="0.3", ls="--", label="outside")
    marker = axes[1, 0].axvline(times[first], color="black", lw=1.2)
    axes[1, 0].set(xlabel="time (fs)", ylabel="population", ylim=(-0.02, 1.02))
    axes[1, 0].set_title("BO-state composition", loc="left", fontweight="semibold")
    axes[1, 0].legend(frameon=False, ncol=4, fontsize=7)

    eps_image = axes[1, 1].imshow(displayed_eps[0], origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=eps_limits[0], vmax=eps_limits[1])
    axes[1, 1].set(xlabel=r"heavy $R$ ($a_0$)", ylabel=r"proton $q$ ($a_0$)")
    axes[1, 1].set_title(r"First TDPES $\epsilon^{(1)}$", loc="left", fontweight="semibold")
    fig.colorbar(eps_image, ax=axes[1, 1], label="shifted energy (Hartree)", pad=0.012)
    _label_panels(axes)
    title = fig.suptitle(
        f"Coupled MCEF story | electron (A) <-> nuclei (B) <-> BO composition (C); exact surface (D)\n"
        f"t={times[first]:.3f} fs", fontsize=13.5, fontweight="bold",
    )

    def update(number):
        frame = int(frames[number])
        electron_line.set_ydata(electron[frame])
        nuclear_image.set_data(joint[frame])
        marker.set_xdata([times[frame], times[frame]])
        eps_image.set_data(displayed_eps[number])
        title.set_text(
            f"Coupled MCEF story | electron (A) <-> nuclei (B) <-> BO composition (C); exact surface (D)\n"
            f"t={times[frame]:.3f} fs"
        )
        return electron_line, nuclear_image, marker, eps_image, title

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/"mcef_dynamics_overview.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=dpi)
    else:
        if fmt == "mp4":
            print("ffmpeg을 찾지 못해 compact animation을 GIF로 저장합니다.")
        path = outdir/"mcef_dynamics_overview.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 105))
    plt.close(fig)
    print(f"Compact report: {path}")


def make_potential_animation(
    data, joint, support_floor, outdir, fps, max_frames, dpi, fmt,
):
    """One five-field movie arranged by the two nested EF levels.

    Top row belongs to the conditional electronic factor Phi.  Bottom-row
    epsilon2/alpha belong to the proton-heavy and outer-heavy factorization;
    the final panel states where any of those fields are actually occupied.
    """
    times, q, R = data["times_fs"], data["q"], data["R"]
    frames = selected_frames(len(times), min(max_frames, len(times)))
    extent = [R[0], R[-1], q[0], q[-1]]
    fields = []
    for frame in frames:
        density = joint[frame]
        mask = density >= support_floor*max(float(np.max(density)), 1.0e-300)
        heavy = np.sum(density, axis=0)*float(q[1]-q[0])
        heavy_mask = heavy >= support_floor*max(float(np.max(heavy)), 1.0e-300)
        eps1 = np.where(
            mask, _shift_at_peak(data["epsilon_1"][frame], density), np.nan
        )
        avec = np.where(mask, data["a"][frame], np.nan)
        bvec = np.where(mask, data["b"][frame], np.nan)
        eps2 = np.asarray(data["epsilon_2"][frame], float).copy()
        eps2 -= eps2[int(np.argmax(heavy))]
        eps2[~heavy_mask] = np.nan
        alpha = np.where(heavy_mask, data["alpha"][frame], np.nan)
        fields.append((eps1, avec, bvec, eps2, alpha, density))

    eps1_lim = robust_limits([item[0] for item in fields])
    a_lim = robust_limits([item[1] for item in fields], symmetric=True)
    b_lim = robust_limits([item[2] for item in fields], symmetric=True)
    eps2_lim = robust_limits([item[3] for item in fields])
    alpha_lim = robust_limits([item[4] for item in fields], symmetric=True)
    density_max = max(float(np.max(item[5])) for item in fields)
    first = int(frames[0])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.3), constrained_layout=True)
    images = []
    for ax, values, limits, title, cmap, label in (
        (axes[0, 0], fields[0][0], eps1_lim, r"Electron level: $\epsilon^{(1)}$", "viridis", "shifted energy (Hartree)"),
        (axes[0, 1], fields[0][1], a_lim, r"Electron connection $a$ along $q$", "coolwarm", r"$a_0^{-1}$"),
        (axes[0, 2], fields[0][2], b_lim, r"Electron connection $b$ along $R$", "coolwarm", r"$a_0^{-1}$"),
    ):
        image = ax.imshow(
            values, origin="lower", aspect="auto", extent=extent, cmap=cmap,
            vmin=limits[0], vmax=limits[1],
        )
        ax.set_xlabel(r"heavy $R$ ($a_0$)")
        ax.set_ylabel(r"proton $q$ ($a_0$)")
        ax.set_title(title, loc="left", fontweight="semibold", fontsize=10)
        fig.colorbar(image, ax=ax, label=label, pad=0.01, fraction=0.046)
        images.append(image)

    eps2_line, = axes[1, 0].plot(R, fields[0][3], color=COLORS[0], lw=2)
    axes[1, 0].set(xlabel=r"heavy $R$ ($a_0$)", ylabel="shifted energy (Hartree)", ylim=eps2_lim)
    axes[1, 0].set_title(r"Proton-heavy level: $\epsilon^{(2)}$", loc="left", fontweight="semibold", fontsize=10)
    axes[1, 0].grid(alpha=0.18)

    alpha_line, = axes[1, 1].plot(R, fields[0][4], color=COLORS[3], lw=2)
    axes[1, 1].set(xlabel=r"heavy $R$ ($a_0$)", ylabel=r"$\alpha$ ($a_0^{-1}$)", ylim=alpha_lim)
    axes[1, 1].set_title(r"Outer connection $\alpha$ along $R$", loc="left", fontweight="semibold", fontsize=10)
    axes[1, 1].grid(alpha=0.18)

    support_image = axes[1, 2].imshow(
        fields[0][5], origin="lower", aspect="auto", extent=extent,
        cmap="magma", vmin=0.0, vmax=density_max,
    )
    axes[1, 2].set(xlabel=r"heavy $R$ ($a_0$)", ylabel=r"proton $q$ ($a_0$)")
    axes[1, 2].set_title("Where the nuclear packet is occupied", loc="left", fontweight="semibold", fontsize=10)
    fig.colorbar(support_image, ax=axes[1, 2], label=r"$\rho_{qR}$", pad=0.01, fraction=0.046)

    _label_panels(axes)
    title = fig.suptitle(
        f"Nested exact potentials | electron level (A-C) -> proton-heavy level (D-E); support (F)\n"
        f"t={times[first]:.3f} fs; density >= {support_floor:g} of peak",
        fontsize=14, fontweight="bold",
    )

    def update(number):
        frame = int(frames[number])
        item = fields[number]
        for image, values in zip(images, item[:3]):
            image.set_data(values)
        eps2_line.set_ydata(item[3])
        alpha_line.set_ydata(item[4])
        support_image.set_data(item[5])
        title.set_text(
            f"Nested exact potentials | electron level (A-C) -> proton-heavy level (D-E); support (F)\n"
            f"t={times[frame]:.3f} fs; density >= {support_floor:g} of peak"
        )
        return (*images, eps2_line, alpha_line, support_image, title)

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/"mcef_exact_potentials.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=3400), dpi=dpi)
    else:
        if fmt == "mp4":
            print("ffmpeg을 찾지 못해 potential animation을 GIF로 저장합니다.")
        path = outdir/"mcef_exact_potentials.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 105))
    plt.close(fig)
    print(f"Compact report: {path}")


def run(
    data, decomposition, outdir, n_states=6, frame=-1, support_floor=1.0e-3,
    dpi=180, no_animation=False, fps=12, max_frames=180,
    animation_dpi=110, fmt="mp4",
):
    """Build all compact products and return the reduced numerical payload."""
    del n_states  # decomposition already fixes the analyzed basis size.
    outdir = dated_results_dir(Path(outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    frame = frame if frame >= 0 else len(data["times_fs"])+frame
    electron, proton, heavy, joint = normalized_marginals(data)
    joint = _joint_normalized(joint, data["q"], data["R"])
    densities = (electron, proton, heavy)
    pairs = tuple(moments(grid, density) for grid, density in zip(
        (data["x"], data["q"], data["R"]), densities
    ))
    means = tuple(pair[0] for pair in pairs)
    widths = tuple(pair[1] for pair in pairs)
    dx = float(data["x"][1]-data["x"][0])
    rearranged = 0.5*np.sum(np.abs(electron-electron[0]), axis=1)*dx
    diagnostics = gauge_invariant_diagnostics(data)
    nac = nonadiabatic_couplings(data, 3)

    plot_particle_motion(data, densities, means, widths, outdir, dpi)
    plot_electronic_transitions(
        data, means, widths, rearranged, decomposition, nac, joint, frame,
        support_floor, outdir, dpi,
    )
    plot_exact_potentials(data, diagnostics, joint, frame, support_floor, outdir, dpi)
    plot_numerical_reliability(data, diagnostics, densities, outdir, dpi)
    if not no_animation:
        make_overview_animation(
            data, decomposition, electron, joint, support_floor, outdir, fps,
            max_frames, animation_dpi, fmt,
        )
        make_potential_animation(
            data, joint, support_floor, outdir, fps, max_frames,
            animation_dpi, fmt,
        )

    energies, _resolved, populations, residual = decomposition
    options = archive_arguments(data)
    divider = 0.5*(
        float(options.get("q0", 0.0))+float(options.get("R0", 0.0))
    )
    left = np.sum(electron[:, data["x"] < divider], axis=1)*dx
    right = np.sum(electron[:, data["x"] >= divider], axis=1)*dx
    payload = dict(
        times_fs=data["times_fs"], x=data["x"], q=data["q"], R=data["R"],
        electron_density=electron, proton_density=proton, heavy_density=heavy,
        nuclear_joint_density=joint,
        electron_mean=means[0], proton_mean=means[1], heavy_mean=means[2],
        electron_width=widths[0], proton_width=widths[1], heavy_width=widths[2],
        electron_rearranged_density=rearranged,
        electron_left_population=left, electron_right_population=right,
        electron_divider=np.array(divider),
        bo_energies=energies, state_populations=populations,
        state_basis_residual=residual,
        **{key: value for key, value in diagnostics.items() if value.ndim == 1},
    )
    archive = outdir/"report_observables.npz"
    np.savez_compressed(archive, **payload)
    print(f"Compact report data: {archive}")
    return payload
