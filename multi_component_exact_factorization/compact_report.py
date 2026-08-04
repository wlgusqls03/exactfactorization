#!/usr/bin/env python3
"""Create a small, question-oriented report for one completed MCEF run.

The standard report contains four static figures and three purpose-specific
animations.  Each product answers one question: what moved, which BO components
formed, what the exact potentials did, how those fields create physical
transport, and whether the discretization remained trustworthy.
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
MASK_COLOR = "#D7DCE0"


def _masked_cmap(name):
    """Colormap where masked/unoccupied cells differ clearly from value zero."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(MASK_COLOR)
    return cmap


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
    """점유 support에서 한 step tangent 이동량이 10%를 넘는 첫 시점."""
    supported = [
        np.asarray(data[key])
        for key in (
            "max_abs_support_gamma_phi_dt",
            "max_abs_support_gamma_lam_dt",
        )
        if key in data.files
    ]
    if supported:
        values = np.maximum.reduce(supported)
        indices = np.flatnonzero(values > 0.1)
        return None if not len(indices) else float(data["times_fs"][indices[0]])
    # 이전 archive에는 support-weighted 지표가 없으므로 기존 판정을 유지한다.
    if "max_abs_gamma_phi" not in data.files:
        return None
    indices = np.flatnonzero(np.asarray(data["max_abs_gamma_phi"]) > 1.0)
    return None if not len(indices) else float(data["times_fs"][indices[0]])


def _mark_stress(ax, onset, label=False):
    if onset is None:
        return
    ax.axvline(
        onset, color="#C43C39", lw=1.1, ls=":", alpha=0.9,
        label=("numerical stress onset" if label else None),
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
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), constrained_layout=True)
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
        extent=[R[0], R[-1], q[0], q[-1]], cmap=_masked_cmap("viridis"),
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
    mean_line, = ax.plot(
        times, means[1]-means[1][0], lw=2, color=COLORS[0],
        label=r"$\Delta\langle q\rangle$",
    )
    width_line, = ax.plot(
        times, widths[1]-widths[1][0], lw=2, color=COLORS[1],
        label=r"$\Delta\sigma_q$",
    )
    ax.axhline(0.0, color="0.6", lw=0.8)
    _mark_stress(ax, onset)
    _style_time_axis(
        ax, "Motion that accompanies the state mixing",
        r"proton position change ($a_0$)",
    )
    rearrange_axis = ax.twinx()
    rearrange_line, = rearrange_axis.plot(
        times, rearranged, lw=2, color=COLORS[2],
        label=r"electron $D_{\rm rearr}$",
    )
    rearrange_axis.set_ylabel(
        r"electron rearrangement (dimensionless)", color=COLORS[2]
    )
    rearrange_axis.tick_params(axis="y", labelcolor=COLORS[2])
    ax.legend(
        handles=[mean_line, width_line, rearrange_line],
        frameon=False, fontsize=8, loc="upper left",
    )

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
        f"composition (A) -> correlated motion (B) -> occupied gap/NAC paths (C-D), t={times[frame]:.3f} fs; gray=unoccupied",
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
            values, origin="lower", aspect="auto", extent=extent,
            cmap=_masked_cmap(cmap),
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
        f"t={times[frame]:.3f} fs; gray=unoccupied grid cells; scalar offsets removed",
        fontsize=14, fontweight="bold",
    )
    path = outdir/"03_exact_potentials.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Compact report: {path}")


def _edge_mass(density, spacing, points=5):
    return (np.sum(density[:, :points], axis=1)+np.sum(density[:, -points:], axis=1))*spacing


def _nyquist_power(data):
    """저장 factor의 one-cell alternating 성분을 physical weight로 측정."""
    lam = np.asarray(data["lambda_wavefunction"])
    chi = np.asarray(data["chi"])
    dq = float(data["q"][1]-data["q"][0])
    dR = float(data["R"][1]-data["R"][0])
    alternating_q = (-1.0)**np.arange(lam.shape[1])
    alternating_R = (-1.0)**np.arange(chi.shape[1])
    projection_q = np.sum(lam*alternating_q[None, :, None], axis=1)*dq
    power_q = np.sum(
        np.abs(chi)**2*np.abs(projection_q)**2, axis=1
    )*dR
    projection_R = np.sum(chi*alternating_R[None, :], axis=1)*dR
    power_R = np.abs(projection_R)**2
    return power_q, power_R


def plot_numerical_reliability(data, diagnostics, densities, outdir, dpi):
    """Constraint load, field roughness, preserved identities, and box edges."""
    times = data["times_fs"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), constrained_layout=True)
    onset = _stress_onset(data)

    ax = axes[0, 0]
    has_supported = "max_abs_support_gamma_phi_dt" in data.files
    gamma_curves = (
        (
            "max_abs_support_gamma_phi_dt",
            r"$\max|w_\Phi\gamma_\Phi|\,\Delta t$",
            COLORS[1],
        ),
        (
            "max_abs_support_gamma_lam_dt",
            r"$\max|w_\Lambda\gamma_\Lambda|\,\Delta t$",
            COLORS[2],
        ),
    ) if has_supported else (
        ("max_abs_gamma_phi", r"raw $\max|\gamma_\Phi|$", COLORS[1]),
        ("max_abs_gamma_lam", r"raw $\max|\gamma_\Lambda|$", COLORS[2]),
    )
    for key, label, color in gamma_curves:
        if key in data.files:
            ax.semilogy(
                times, np.maximum(data[key], 1.0e-18),
                color=color, lw=2.0, label=label,
            )
    if has_supported:
        ax.axhline(
            0.1, color="#C43C39", ls=":", lw=1.2,
            label="warning: 10% transfer in one step",
        )
        text = (
            rf"max support rate: $\Phi={np.max(data['max_abs_support_gamma_phi']):.1e}$, "
            rf"$\Lambda={np.max(data['max_abs_support_gamma_lam']):.1e}$ a.u.$^{{-1}}$"
        )
        ax.text(
            0.03, 0.05, text, transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9),
        )
    _style_time_axis(
        ax, "Occupied support: correction per step",
        "dimensionless step load (log scale)" if has_supported else "magnitude (log scale)",
    )
    _mark_stress(ax, onset, label=True)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    if "max_raw_logamp_phi" in data.files:
        for key, label, color, style, alpha, width in (
            ("max_raw_logamp_phi", r"tail-sensitive raw: $\Phi$", COLORS[1], ":", 0.6, 1.4),
            ("max_effective_logamp_phi", r"used by solver: $\Phi$", COLORS[1], "-", 1.0, 2.1),
            ("max_raw_logamp_lam", r"tail-sensitive raw: $\Lambda$", COLORS[2], ":", 0.6, 1.4),
            ("max_effective_logamp_lam", r"used by solver: $\Lambda$", COLORS[2], "-", 1.0, 2.1),
        ):
            ax.semilogy(
                times, np.maximum(data[key], 1.0e-18), label=label,
                color=color, ls=style, alpha=alpha, lw=width,
            )
        removed_phi = np.max(data["suppressed_probability_phi"])
        removed_lam = np.max(data["suppressed_probability_lam"])
        ax.text(
            0.03, 0.05,
            rf"max suppressed mass: $\Phi={removed_phi:.1e}$, $\Lambda={removed_lam:.1e}$",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9),
        )
        title = "Tail filter: dotted = raw, solid = used by solver"
        ylabel = r"max $|\nabla\ln A|$ (log scale)"
    else:
        for key, label, color in (
            ("support_rms_a", "RMS a", COLORS[0]),
            ("support_rms_b", "RMS b", COLORS[1]),
            ("support_rms_alpha", "RMS alpha", COLORS[2]),
            ("support_rms_force_q", "RMS proton force", COLORS[3]),
            ("support_rms_force_R", "RMS heavy force", COLORS[4]),
        ):
            ax.semilogy(times, np.maximum(diagnostics[key], 1.0e-18), label=label, color=color)
        title = "Roughness inside the occupied region"
        ylabel = "weighted RMS (log scale)"
    _style_time_axis(ax, title, ylabel)
    _mark_stress(ax, onset)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 0]
    has_product_projection = "max_product_residual_l2" in data.files
    if has_product_projection:
        ax.semilogy(
            times, np.maximum(data["max_product_residual_l2"], 1.0e-20),
            color=COLORS[1], ls=":", alpha=0.75,
            label=r"factor/full $D_2$ residual: before",
        )
        ax.semilogy(
            times,
            np.maximum(data["max_effective_product_residual_l2"], 1.0e-20),
            color=COLORS[1], lw=1.8,
            label=r"factor/full $D_2$ residual: after",
        )
        ax.semilogy(
            times, np.maximum(
                data["max_abs_full_norm_rate_after_product_projection"],
                1.0e-20,
            ),
            color=COLORS[2], lw=1.6,
            label=r"full norm-rate after projection",
        )
    elif "max_raw_rate_phi" in data.files:
        ax.semilogy(
            times, np.maximum(data["max_corrected_rate_phi"], 1.0e-20),
            color=COLORS[1], lw=1.6, label=r"corrected $r_\Phi$",
        )
        ax.semilogy(
            times, np.maximum(data["max_corrected_rate_lam"], 1.0e-20),
            color=COLORS[2], lw=1.6, label=r"corrected $r_\Lambda$",
        )
    if "pnc_projection_correction" in data.files:
        ax.semilogy(
            times, np.maximum(data["pnc_projection_correction"], 1.0e-20),
            color=COLORS[0], ls=":", alpha=0.75,
            label="global PNC before projection (tail-sensitive)",
        )
    if "pnc_error" in data.files:
        ax.semilogy(
            times, np.maximum(data["pnc_error"], 1.0e-20),
            color=COLORS[3], ls="-.", label="saved PNC residual",
        )
    norm_error = np.abs(data["norm"]-1.0)
    ax.semilogy(
        times, np.maximum(norm_error, 1.0e-20), color="0.15", lw=2.2,
        label=r"full $|N_\Psi-1|$ (physical invariant)",
    )
    ax.text(
        0.97, 0.95,
        rf"max full-norm error = {np.max(norm_error):.1e}",
        transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9),
    )
    conservation_title = (
        "Conservation: discrete product residual before/after projection"
        if has_product_projection
        else "Conservation: norm drift vs tail-sensitive PNC"
    )
    _style_time_axis(ax, conservation_title, "error / rate (log scale)")
    _mark_stress(ax, onset)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    q, R = data["q"], data["R"]
    q_edge = _edge_mass(densities[1], float(q[1]-q[0]))
    R_edge = _edge_mass(densities[2], float(R[1]-R[0]))
    ax.semilogy(times, np.maximum(q_edge, 1.0e-20), color=COLORS[0], label="proton: outer 5 points")
    ax.semilogy(times, np.maximum(R_edge, 1.0e-20), color=COLORS[2], label="heavy: outer 5 points")
    nyquist_q, nyquist_R = _nyquist_power(data)
    ax.semilogy(times, np.maximum(nyquist_q, 1.0e-20), color=COLORS[1], ls="--", label=r"$\Lambda$ q-Nyquist power")
    ax.semilogy(times, np.maximum(nyquist_R, 1.0e-20), color=COLORS[3], ls="--", label=r"$\chi$ R-Nyquist power")
    options = archive_arguments(data)
    dq, dR = float(q[1]-q[0]), float(R[1]-R[0])
    q_growth = float(nyquist_q[-1]/max(nyquist_q[0], 1.0e-30))
    R_growth = float(nyquist_R[-1]/max(nyquist_R[0], 1.0e-30))
    text = (
        rf"initial $\sigma_q/dq={options.get('proton_sigma', np.nan)/dq:.2f}$" "\n"
        rf"initial $\sigma_R/dR={options.get('heavy_sigma', np.nan)/dR:.2f}$" "\n"
        rf"Nyquist growth: $q\times{q_growth:.1e}$, $R\times{R_growth:.1e}$"
    )
    ax.text(0.03, 0.05, text, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))
    grid_title = (
        "Grid warning: R one-cell mode grows; edges remain empty"
        if R_growth > 100.0 and nyquist_R[-1] > 1.0e-12
        else "Grid check: boundaries and one-cell modes"
    )
    _style_time_axis(ax, grid_title, "probability / power (log scale)")
    _mark_stress(ax, onset)
    ax.legend(frameon=False, fontsize=8)

    _label_panels(axes)
    fig.suptitle(
        "4 | Numerical reliability: occupied dynamics (A), discarded tails (B), conservation (C), and grid artifacts (D)",
        fontsize=13.5, fontweight="bold",
    )
    path = outdir/"04_numerical_reliability.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Compact report: {path}")


def make_overview_animation(
    data, decomposition, electron, joint, diagnostics, support_floor, outdir,
    fps, max_frames, dpi, fmt,
):
    """Observed dynamics followed by proton momentum, transport, and drive."""
    times, x, q, R = data["times_fs"], data["x"], data["q"], data["R"]
    _energies, _resolved, populations, residual = decomposition
    frames = selected_frames(len(times), min(max_frames, len(times)))
    momentum_fields, current_fields, drive_fields = [], [], []
    for frame in frames:
        density = joint[frame]
        momentum_fields.append(_support_mask(
            diagnostics["momentum_q"][frame], density, support_floor
        ))
        current_fields.append(_support_mask(
            diagnostics["proton_current"][frame], density, support_floor
        ))
        drive_fields.append(_support_mask(
            diagnostics["force_q"][frame], density, support_floor
        ))
    momentum_limits = robust_limits(momentum_fields, symmetric=True)
    current_limits = robust_limits(current_fields, symmetric=True)
    drive_values = np.concatenate([
        np.abs(field[np.isfinite(field)]).ravel() for field in drive_fields
    ])
    drive_max = max(float(np.max(drive_values)), 1.0e-14)
    drive_typical = max(float(np.percentile(drive_values, 80.0)), 1.0e-6)
    drive_norm = SymLogNorm(
        linthresh=max(0.2*drive_typical, 1.0e-5), linscale=0.8,
        vmin=-drive_max, vmax=drive_max, base=10,
    )
    joint_max = max(float(np.max(joint[i])) for i in frames)
    electron_max = max(float(np.max(electron[i])) for i in frames)
    extent = [R[0], R[-1], q[0], q[-1]]
    first = int(frames[0])

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.4), constrained_layout=True)
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
        axes[0, 2].plot(times, populations[:, state], color=COLORS[state % len(COLORS)], lw=1.7, label=rf"$P_{state}$")
    axes[0, 2].plot(times, residual, color="0.3", ls="--", label="outside")
    marker = axes[0, 2].axvline(times[first], color="black", lw=1.2)
    axes[0, 2].set(xlabel="time (fs)", ylabel="population", ylim=(-0.02, 1.02))
    axes[0, 2].set_title("BO-state composition", loc="left", fontweight="semibold")
    axes[0, 2].legend(frameon=False, ncol=4, fontsize=7)

    images = []
    for ax, values, title_text, label, limits, norm in (
        (axes[1, 0], momentum_fields[0],
         r"Mechanical proton momentum $K_q=\partial_qT+a$",
         r"momentum ($a_0^{-1}$)", momentum_limits, None),
        (axes[1, 1], current_fields[0],
         r"Actual probability transport $j_q=\rho_{qR}K_q/m_p$",
         "probability current", current_limits, None),
        (axes[1, 2], drive_fields[0],
         r"Gauge-invariant drive $E_q=-\partial_q\epsilon^{(1)}+\partial_ta$",
         r"force (Hartree/$a_0$)", None, drive_norm),
    ):
        kwargs = {"norm": norm} if norm is not None else {
            "vmin": limits[0], "vmax": limits[1]
        }
        image = ax.imshow(
            values, origin="lower", aspect="auto", extent=extent,
            cmap=_masked_cmap("coolwarm"), **kwargs,
        )
        ax.set(xlabel=r"heavy $R$ ($a_0$)", ylabel=r"proton $q$ ($a_0$)")
        ax.set_title(title_text, loc="left", fontweight="semibold", fontsize=9.5)
        fig.colorbar(image, ax=ax, label=label, pad=0.01, fraction=0.046)
        images.append(image)
    _label_panels(axes)
    title = fig.suptitle(
        "Physical dynamics (A-C) -> proton momentum (D) -> transport (E) -> drive (F)\n"
        f"t={times[first]:.3f} fs; gray=unoccupied cells",
        fontsize=13.5, fontweight="bold",
    )

    def update(number):
        frame = int(frames[number])
        electron_line.set_ydata(electron[frame])
        nuclear_image.set_data(joint[frame])
        marker.set_xdata([times[frame], times[frame]])
        for image, values in zip(
            images,
            (momentum_fields[number], current_fields[number], drive_fields[number]),
        ):
            image.set_data(values)
        title.set_text(
            "Physical dynamics (A-C) -> proton momentum (D) -> transport (E) -> drive (F)\n"
            f"t={times[frame]:.3f} fs; gray=unoccupied cells"
        )
        return electron_line, nuclear_image, marker, *images, title

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


def make_physical_interpretation_animation(
    data, electron, proton, heavy, joint, diagnostics, support_floor, outdir,
    fps, max_frames, dpi, fmt,
):
    """Connect all three marginals to gauge-invariant transport and drive."""
    times = np.asarray(data["times_fs"])
    x, q, R = data["x"], data["q"], data["R"]
    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    extent = [R[0], R[-1], q[0], q[-1]]

    proton_currents, proton_drives = [], []
    heavy_currents, heavy_drives = [], []
    for frame in frames:
        density = joint[frame]
        proton_currents.append(_support_mask(
            diagnostics["proton_current"][frame], density, support_floor
        ))
        proton_drives.append(_support_mask(
            diagnostics["force_q"][frame], density, support_floor
        ))
        heavy_mask = heavy[frame] >= support_floor*max(
            float(np.max(heavy[frame])), 1.0e-300
        )
        heavy_currents.append(np.where(
            heavy_mask, diagnostics["heavy_current"][frame], np.nan
        ))
        heavy_drives.append(np.where(
            heavy_mask, diagnostics["force_R"][frame], np.nan
        ))

    current_limits = robust_limits(proton_currents, symmetric=True)
    finite_drive_values = [
        np.abs(field[np.isfinite(field)]).ravel() for field in proton_drives
        if np.any(np.isfinite(field))
    ]
    drive_values = (
        np.concatenate(finite_drive_values)
        if finite_drive_values else np.array([0.0])
    )
    drive_max = max(float(np.max(drive_values)), 1.0e-14)
    drive_typical = max(float(np.percentile(drive_values, 80.0)), 1.0e-6)
    drive_norm = SymLogNorm(
        linthresh=max(0.2*drive_typical, 1.0e-5), linscale=0.8,
        vmin=-drive_max, vmax=drive_max, base=10,
    )
    heavy_current_limits = robust_limits(heavy_currents, symmetric=True)
    heavy_drive_limits = robust_limits(heavy_drives, symmetric=True)

    fig, axes = plt.subplots(2, 3, figsize=(16.4, 8.5), constrained_layout=True)
    marginal_specs = (
        (axes[0, 0], x, electron, "Electron marginal", r"electron $x$ ($a_0$)", COLORS[0]),
        (axes[0, 1], q, proton, "Proton marginal", r"proton $q$ ($a_0$)", COLORS[1]),
        (axes[0, 2], R, heavy, "Heavy-nucleus marginal", r"heavy $R$ ($a_0$)", COLORS[2]),
    )
    marginal_lines = []
    for ax, grid, density, title_text, xlabel, color in marginal_specs:
        maximum = max(float(np.max(density[index])) for index in frames)
        ax.plot(
            grid, density[0], color="0.62", lw=1.3, ls="--",
            label="initial",
        )
        line, = ax.plot(
            grid, density[first], color=color, lw=2.2, label="current",
        )
        ax.set(
            xlabel=xlabel, ylabel="probability density",
            ylim=(0.0, 1.08*maximum),
        )
        ax.set_title(title_text, loc="left", fontweight="semibold")
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
        marginal_lines.append(line)

    current_image = axes[1, 0].imshow(
        proton_currents[0], origin="lower", aspect="auto", extent=extent,
        cmap=_masked_cmap("coolwarm"),
        vmin=current_limits[0], vmax=current_limits[1],
    )
    axes[1, 0].set(
        xlabel=r"heavy $R$ ($a_0$)", ylabel=r"proton $q$ ($a_0$)",
    )
    axes[1, 0].set_title(
        r"Transport: $j_q=\rho_{qR}(\partial_qT+a)/m_p$",
        loc="left", fontweight="semibold", fontsize=9.6,
    )
    fig.colorbar(
        current_image, ax=axes[1, 0], label="proton probability current",
        pad=0.01, fraction=0.046,
    )

    drive_image = axes[1, 1].imshow(
        proton_drives[0], origin="lower", aspect="auto", extent=extent,
        cmap=_masked_cmap("coolwarm"), norm=drive_norm,
    )
    axes[1, 1].set(
        xlabel=r"heavy $R$ ($a_0$)", ylabel=r"proton $q$ ($a_0$)",
    )
    axes[1, 1].set_title(
        r"Drive: $E_q=-\partial_q\epsilon^{(1)}+\partial_ta$",
        loc="left", fontweight="semibold", fontsize=9.6,
    )
    fig.colorbar(
        drive_image, ax=axes[1, 1],
        label=r"gauge-invariant drive (Hartree/$a_0$)",
        pad=0.01, fraction=0.046,
    )

    heavy_current_line, = axes[1, 2].plot(
        R, heavy_currents[0], color=COLORS[0], lw=2.0,
        label=r"$j_R^{(\chi)}=\rho_R(\partial_RS+\alpha)/M$",
    )
    axes[1, 2].set(
        xlabel=r"heavy $R$ ($a_0$)", ylabel="heavy probability current",
        ylim=heavy_current_limits,
    )
    axes[1, 2].grid(alpha=0.18, linewidth=0.7)
    heavy_force_axis = axes[1, 2].twinx()
    heavy_drive_line, = heavy_force_axis.plot(
        R, heavy_drives[0], color=COLORS[3], lw=1.8,
        label=r"$F_R=-\partial_R\epsilon^{(2)}+\partial_t\alpha$",
    )
    heavy_force_axis.set_ylabel(
        r"heavy drive (Hartree/$a_0$)", color=COLORS[3]
    )
    heavy_force_axis.tick_params(axis="y", labelcolor=COLORS[3])
    heavy_force_axis.set_ylim(heavy_drive_limits)
    axes[1, 2].set_title(
        "Outer heavy transport and drive", loc="left",
        fontweight="semibold", fontsize=9.6,
    )
    axes[1, 2].legend(
        handles=[heavy_current_line, heavy_drive_line],
        frameon=False, fontsize=7, loc="best",
    )

    _label_panels(axes)
    title = fig.suptitle(
        "Observed marginals (A-C) -> physical transport (D) -> "
        "scalar+vector drives (E-F)\n"
        rf"$a$ enters proton flow; $b$ enters $\alpha="
        rf"\langle\partial_RT+b\rangle_q$; t={times[first]:.3f} fs",
        fontsize=13.5, fontweight="bold",
    )

    def update(number):
        frame = int(frames[number])
        for line, density in zip(marginal_lines, (electron, proton, heavy)):
            line.set_ydata(density[frame])
        current_image.set_data(proton_currents[number])
        drive_image.set_data(proton_drives[number])
        heavy_current_line.set_ydata(heavy_currents[number])
        heavy_drive_line.set_ydata(heavy_drives[number])
        title.set_text(
            "Observed marginals (A-C) -> physical transport (D) -> "
            "scalar+vector drives (E-F)\n"
            rf"$a$ enters proton flow; $b$ enters $\alpha="
            rf"\langle\partial_RT+b\rangle_q$; t={times[frame]:.3f} fs"
        )
        return (
            *marginal_lines, current_image, drive_image,
            heavy_current_line, heavy_drive_line, title,
        )

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/"mcef_physical_interpretation.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=3300), dpi=dpi)
    else:
        if fmt == "mp4":
            print("ffmpeg을 찾지 못해 physical interpretation을 GIF로 저장합니다.")
        path = outdir/"mcef_physical_interpretation.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 105))
    plt.close(fig)
    print(f"Compact report: {path}")


def make_potential_animation(
    data, joint, diagnostics, support_floor, outdir, fps, max_frames, dpi, fmt,
):
    """Nested scalar/connections followed by their outer-heavy consequence."""
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
        phase_R = np.where(
            heavy_mask, diagnostics["phase_gradient_R_chi"][frame], np.nan
        )
        momentum_R = np.where(
            heavy_mask, diagnostics["momentum_R_outer"][frame], np.nan
        )
        current_R = np.where(
            heavy_mask, diagnostics["heavy_current"][frame], np.nan
        )
        force_R = np.where(heavy_mask, diagnostics["force_R"][frame], np.nan)
        fields.append((
            eps1, avec, bvec, eps2, alpha, phase_R, momentum_R,
            current_R, force_R, heavy,
        ))

    eps1_lim = robust_limits([item[0] for item in fields])
    a_lim = robust_limits([item[1] for item in fields], symmetric=True)
    b_lim = robust_limits([item[2] for item in fields], symmetric=True)
    eps2_lim = robust_limits([item[3] for item in fields])
    momentum_R_lim = robust_limits(
        [item[index] for item in fields for index in (4, 5, 6)],
        symmetric=True,
    )
    current_R_lim = robust_limits([item[7] for item in fields], symmetric=True)
    force_R_lim = robust_limits([item[8] for item in fields], symmetric=True)
    first = int(frames[0])

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.3), constrained_layout=True)
    images = []
    for ax, values, limits, title, cmap, label in (
        (axes[0, 0], fields[0][0], eps1_lim, r"Electron level: $\epsilon^{(1)}$", "viridis", "shifted energy (Hartree)"),
        (axes[0, 1], fields[0][1], a_lim, r"Electron connection $a$ along $q$", "coolwarm", r"$a_0^{-1}$"),
        (axes[0, 2], fields[0][2], b_lim, r"Electron connection $b$ along $R$", "coolwarm", r"$a_0^{-1}$"),
    ):
        image = ax.imshow(
            values, origin="lower", aspect="auto", extent=extent,
            cmap=_masked_cmap(cmap),
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
    density_axis = axes[1, 0].twinx()
    heavy_density_line, = density_axis.plot(
        R, fields[0][9]/max(float(np.max(fields[0][9])), 1.0e-300),
        color="0.45", lw=1.2, alpha=0.65, label=r"$\rho_R$ (scaled)",
    )
    density_axis.set_ylim(0.0, 1.08)
    density_axis.set_yticks([])

    alpha_line, = axes[1, 1].plot(R, fields[0][4], color=COLORS[3], lw=1.8, label=r"$\alpha$")
    phase_R_line, = axes[1, 1].plot(R, fields[0][5], color=COLORS[1], lw=1.6, ls="--", label=r"$\partial_RS$")
    momentum_R_line, = axes[1, 1].plot(R, fields[0][6], color="black", lw=2.1, label=r"$K_R^{(\chi)}$")
    axes[1, 1].set(xlabel=r"heavy $R$ ($a_0$)", ylabel=r"momentum ($a_0^{-1}$)", ylim=momentum_R_lim)
    axes[1, 1].set_title(r"Heavy momentum: $\partial_RS+\alpha=K_R^{(\chi)}$", loc="left", fontweight="semibold", fontsize=10)
    axes[1, 1].grid(alpha=0.18)
    axes[1, 1].legend(frameon=False, fontsize=8)

    current_R_line, = axes[1, 2].plot(
        R, fields[0][7], color=COLORS[0], lw=2, label=r"$j_R^{(\chi)}$"
    )
    axes[1, 2].set(
        xlabel=r"heavy $R$ ($a_0$)", ylabel="heavy probability current",
        ylim=current_R_lim,
    )
    force_axis = axes[1, 2].twinx()
    force_R_line, = force_axis.plot(
        R, fields[0][8], color=COLORS[3], lw=1.8,
        label=r"$F_R=-\partial_R\epsilon^{(2)}+\partial_t\alpha$",
    )
    force_axis.set_ylabel(r"heavy force (Hartree/$a_0$)", color=COLORS[3])
    force_axis.tick_params(axis="y", labelcolor=COLORS[3])
    force_axis.set_ylim(force_R_lim)
    axes[1, 2].set_title("Actual heavy transport and drive", loc="left", fontweight="semibold", fontsize=10)
    axes[1, 2].grid(alpha=0.18)
    axes[1, 2].legend(
        handles=[current_R_line, force_R_line], frameon=False, fontsize=7,
    )

    _label_panels(axes)
    title = fig.suptitle(
        "Nested fields (A-D) -> heavy mechanical momentum (E) -> transport/drive (F)\n"
        f"t={times[first]:.3f} fs; gray=unoccupied cells (< {support_floor:g} of peak)",
        fontsize=14, fontweight="bold",
    )

    def update(number):
        frame = int(frames[number])
        item = fields[number]
        for image, values in zip(images, item[:3]):
            image.set_data(values)
        eps2_line.set_ydata(item[3])
        heavy_density_line.set_ydata(
            item[9]/max(float(np.max(item[9])), 1.0e-300)
        )
        alpha_line.set_ydata(item[4])
        phase_R_line.set_ydata(item[5])
        momentum_R_line.set_ydata(item[6])
        current_R_line.set_ydata(item[7])
        force_R_line.set_ydata(item[8])
        title.set_text(
            "Nested fields (A-D) -> heavy mechanical momentum (E) -> transport/drive (F)\n"
            f"t={times[frame]:.3f} fs; gray=unoccupied cells (< {support_floor:g} of peak)"
        )
        return (
            *images, eps2_line, heavy_density_line,
            alpha_line, phase_R_line, momentum_R_line,
            current_R_line, force_R_line, title,
        )

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
            data, decomposition, electron, joint, diagnostics, support_floor,
            outdir, fps, max_frames, animation_dpi, fmt,
        )
        make_potential_animation(
            data, joint, diagnostics, support_floor, outdir, fps, max_frames,
            animation_dpi, fmt,
        )
        make_physical_interpretation_animation(
            data, electron, proton, heavy, joint, diagnostics, support_floor,
            outdir, fps, max_frames, animation_dpi, fmt,
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
