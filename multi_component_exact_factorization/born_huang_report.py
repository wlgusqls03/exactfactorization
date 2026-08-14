#!/usr/bin/env python3
"""Full compact figures and animations for Born--Huang trajectories.

The report deliberately keeps the same four-question narrative as the direct
grid report.  A compact, physically meaningful electronic picture is rebuilt
without storing the enormous ``phi_n(x;q,R)`` tensor: at each displayed frame
the one-dimensional electronic Hamiltonian is diagonalized only at the peak of
the occupied nuclear density.  The resulting curves are local conditional BO
states, not an electron marginal; every panel labels that distinction.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np

from .potential_analysis import gauge_invariant_diagnostics
from .report_plot_style import (
    COLORS,
    CURRENT_COLOR,
    FORCE_COLOR,
    HEAVY_DENSITY_COLOR,
    JOINT_CMAP,
    MASK_COLOR,
    PARTICLE_COLORS,
    SCALAR_CMAP,
    SIGNED_CMAP,
    add_fixed_center_markers,
    color_y_axis,
    density_display_alpha,
    density_weighted_shift,
    joint_density_limit,
    masked_cmap,
)
from .visualize import (
    NUMBER_FORMATTER,
    archive_arguments,
    selected_frames,
)
from .marginal_movie import make_fixed_scale_marginal_animation
from .coordinate_focus_movie import make_coordinate_focus_animation


class _ArchiveView(dict):
    @property
    def files(self):
        return tuple(self.keys())


def _diagnostics(data):
    return gauge_invariant_diagnostics(
        data if hasattr(data, "files") else _ArchiveView(data)
    )


def load_archive(path):
    """Materialize only arrays needed by the BO coefficient report."""
    required = {
        "times_fs", "q", "R", "lambda_wavefunction", "chi",
        "bo_populations", "bo_energies", "norm",
    }
    with np.load(path, allow_pickle=True) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Born--Huang report에 필요한 key가 없습니다: {missing}")
        # C_j(q,R,t), NAC tensors and x-grid basis can each be several GiB.
        # The saved factor potentials are used by the potential movie.
        wanted = required | {
            "args", "propagation_completed", "requested_final_time_fs",
            "failure_reason", "pnc_projection_correction",
            "epsilon_1", "epsilon_2", "a", "b", "alpha",
            "max_abs_full_norm_rate_after_product_projection",
            "max_abs_support_gamma_phi_dt", "max_abs_support_gamma_lam_dt",
            "max_effective_product_residual_l2",
            "max_relative_product_projection_l2",
            "max_relative_support_product_projection_l2",
            "max_raw_logamp_phi", "max_effective_logamp_phi",
            "max_raw_logamp_lam", "max_effective_logamp_lam",
            "suppressed_probability_phi", "suppressed_probability_lam",
            "deep_tail_suppressed_probability_phi",
            "deep_tail_suppressed_probability_lam",
            "bo_state_density_q", "bo_state_density_R",
            "x", "electron_density",
        }
        data = {
            key: archive[key]
            for key in archive.files
            if key in wanted
        }
        # New archives store these tiny reductions directly.  Older archives
        # can still produce the paper-style BO wave-packet plot from C, at the
        # cost of decompressing C once during report generation.
        if (
            "bo_state_density_q" not in data
            and "electronic_coefficients" in archive.files
        ):
            coefficients = np.asarray(archive["electronic_coefficients"])
            lam = np.asarray(data["lambda_wavefunction"])
            chi = np.asarray(data["chi"])
            q, R = np.asarray(data["q"]), np.asarray(data["R"])
            state_q = np.empty(
                (len(coefficients), coefficients.shape[1], len(q)), float
            )
            state_R = np.empty(
                (len(coefficients), coefficients.shape[1], len(R)), float
            )
            for frame in range(len(coefficients)):
                joint = np.abs(lam[frame])**2*np.abs(chi[frame][None, :])**2
                resolved = np.abs(coefficients[frame])**2*joint[None, :, :]
                state_q[frame] = np.sum(resolved, axis=2)*float(R[1]-R[0])
                state_R[frame] = np.sum(resolved, axis=1)*float(q[1]-q[0])
            data["bo_state_density_q"] = state_q
            data["bo_state_density_R"] = state_R
        return data


def _frame_normalize(values, q, R):
    dq, dR = float(q[1]-q[0]), float(R[1]-R[0])
    norms = np.sum(values, axis=(1, 2))*dq*dR
    return values/np.maximum(norms[:, None, None], 1.0e-300)


def _moments(grid, density, spacing):
    mean = np.sum(density*grid[None, :], axis=1)*spacing
    variance = np.sum(
        density*(grid[None, :]-mean[:, None])**2, axis=1
    )*spacing
    return mean, np.sqrt(np.maximum(variance, 0.0))


def calculate_observables(data):
    """Reduce a BO coefficient archive to nuclear and state diagnostics."""
    times = np.asarray(data["times_fs"], float)
    q, R = np.asarray(data["q"], float), np.asarray(data["R"], float)
    dq, dR = float(q[1]-q[0]), float(R[1]-R[0])
    lam = np.asarray(data["lambda_wavefunction"])
    chi = np.asarray(data["chi"])
    joint = np.abs(lam)**2*np.abs(chi[:, None, :])**2
    joint = _frame_normalize(joint, q, R)
    q_density = np.sum(joint, axis=2)*dR
    R_density = np.sum(joint, axis=1)*dq
    q_mean, q_width = _moments(q, q_density, dq)
    R_mean, R_width = _moments(R, R_density, dR)

    norm = np.asarray(data["norm"], float)
    populations = np.asarray(data["bo_populations"], float)
    normalized_populations = populations/np.maximum(norm[:, None], 1.0e-300)
    energies = np.asarray(data["bo_energies"], float)
    mean_energies = np.einsum(
        "tqR,nqR->tn", joint, energies, optimize=True
    )*dq*dR
    edge = min(5, len(q)//2, len(R)//2)
    q_outer = (
        np.sum(q_density[:, :edge], axis=1)
        +np.sum(q_density[:, -edge:], axis=1)
    )*dq
    R_outer = (
        np.sum(R_density[:, :edge], axis=1)
        +np.sum(R_density[:, -edge:], axis=1)
    )*dR
    if "bo_state_density_q" in data and "bo_state_density_R" in data:
        state_q = np.asarray(data["bo_state_density_q"], float)
        state_R = np.asarray(data["bo_state_density_R"], float)
        state_q /= np.maximum(norm[:, None, None], 1.0e-300)
        state_R /= np.maximum(norm[:, None, None], 1.0e-300)
    else:
        # Synthetic/legacy archive fallback.  It preserves integrated state
        # populations but cannot represent state-dependent packet splitting.
        state_q = normalized_populations[:, :, None]*q_density[:, None, :]
        state_R = normalized_populations[:, :, None]*R_density[:, None, :]
    electron = None
    electron_mean = electron_width = None
    if "electron_density" in data:
        candidate = np.asarray(data["electron_density"], float)
        if candidate.shape == (len(times), len(data["x"])):
            dx = float(data["x"][1]-data["x"][0])
            electron = candidate/np.maximum(
                np.sum(candidate, axis=1)[:, None]*dx, 1.0e-300
            )
            electron_mean, electron_width = _moments(
                np.asarray(data["x"], float), electron, dx
            )
    return dict(
        times_fs=times, x=(np.asarray(data["x"], float) if "x" in data else None),
        q=q, R=R, nuclear_joint_density=joint,
        proton_density=q_density, heavy_density=R_density,
        proton_mean=q_mean, proton_width=q_width,
        heavy_mean=R_mean, heavy_width=R_width,
        norm=norm, state_populations=populations,
        normalized_state_populations=normalized_populations,
        mean_bo_energies=mean_energies,
        state_resolved_q_density=state_q,
        state_resolved_R_density=state_R,
        electron_density=electron, electron_mean=electron_mean,
        electron_width=electron_width,
        outer_probability_q=q_outer, outer_probability_R=R_outer,
    )


def _diag(data, name, length):
    if name not in data:
        return np.zeros(length)
    values = np.asarray(data[name], float)
    return values if values.shape == (length,) else np.zeros(length)


def _surface_slice(data, obs, frame, coordinate):
    """BO surfaces along q or R through the joint-density maximum (not mean)."""
    density = obs["nuclear_joint_density"][frame]
    iq, iR = np.unravel_index(int(np.argmax(density)), density.shape)
    energies = np.asarray(data["bo_energies"], float)
    if coordinate == "R":
        return obs["R"], energies[:, iq, :], obs["state_resolved_R_density"][frame], (
            rf"slice at $q_{{\rm peak}}={obs['q'][iq]:.3f}$"
        )
    return obs["q"], energies[:, :, iR], obs["state_resolved_q_density"][frame], (
        rf"slice at $R_{{\rm peak}}={obs['R'][iR]:.3f}$"
    )


def _plot_bo_wavepackets(ax, grid, surfaces, packets, subtitle, *, legend=True):
    """Paper-style BO surfaces with state-projected nuclear packets attached."""
    energy_span = max(float(np.nanpercentile(surfaces, 98)-np.nanpercentile(surfaces, 2)), 1.0e-3)
    packet_scale = 0.16*energy_span
    peak = max(float(np.max(packets)), 1.0e-300)
    artists = []
    for state, (surface, packet) in enumerate(zip(surfaces, packets)):
        color = COLORS[state % len(COLORS)]
        line, = ax.plot(grid, surface, color=color, lw=1.65,
                        label=rf"$E_{state}$")
        lifted = surface+packet_scale*packet/peak
        fill = ax.fill_between(grid, surface, lifted, color=color, alpha=0.34)
        artists.extend((line, fill))
    finite = surfaces[np.isfinite(surfaces)]
    low, high = np.percentile(finite, [1.0, 99.0])
    margin = max(0.08*(high-low), packet_scale)
    ax.set_xlim(float(grid[0]), float(grid[-1]))
    ax.set_ylim(float(low-margin), float(high+1.8*packet_scale))
    ax.set_xlabel(r"nuclear coordinate ($a_0$)")
    ax.set_ylabel("BO energy (Hartree)")
    ax.set_title("State-resolved packet on BO surfaces\n"+subtitle,
                 loc="left", fontweight="semibold", fontsize=9.2)
    ax.grid(alpha=0.14)
    if legend:
        ax.legend(frameon=False, fontsize=7, ncol=3)
    return artists


def plot_nuclear_motion(obs, outdir, dpi):
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    fig = plt.figure(figsize=(17.0, 9.0), constrained_layout=True)
    spec = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.82))
    map_axes = [fig.add_subplot(spec[0, index]) for index in range(3)]
    profile_ax = fig.add_subplot(spec[1, :2])
    summary_ax = fig.add_subplot(spec[1, 2])
    populations = obs["normalized_state_populations"]
    if obs["electron_density"] is not None:
        image = map_axes[0].pcolormesh(
            times, obs["x"], obs["electron_density"].T,
            shading="nearest", cmap="magma",
        )
        map_axes[0].plot(times, obs["electron_mean"], color="white", lw=1.4)
        map_axes[0].set_title("Electron marginal", loc="left", fontweight="semibold")
        map_axes[0].set_ylabel(r"electron $x$ ($a_0$)")
        fig.colorbar(image, ax=map_axes[0], pad=0.012, label="probability density")
    else:
        positive = populations[populations > 0.0]
        vmin = max(min(float(np.min(positive)) if positive.size else 1e-12, 1e-8), 1e-14)
        image = map_axes[0].pcolormesh(
            times, np.arange(populations.shape[1]), np.maximum(populations.T, vmin),
            shading="nearest", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=1.0),
        )
        map_axes[0].set_title("Electronic BO-state character (legacy archive)", loc="left", fontweight="semibold")
        map_axes[0].set_ylabel("BO state n")
        map_axes[0].set_yticks(np.arange(populations.shape[1]))
        fig.colorbar(image, ax=map_axes[0], pad=0.012, label="population")
    map_axes[0].set_xlabel("time (fs)")
    for ax, grid, density, mean, width, title in (
        (map_axes[1], q, obs["proton_density"], obs["proton_mean"],
         obs["proton_width"], "Proton marginal"),
        (map_axes[2], R, obs["heavy_density"], obs["heavy_mean"],
         obs["heavy_width"], "Heavy-nucleus marginal"),
    ):
        image = ax.pcolormesh(times, grid, density.T, shading="nearest", cmap="magma")
        ax.plot(times, mean, color="white", lw=1.5)
        ax.plot(times, mean-width, color="white", lw=0.8, ls="--", alpha=0.8)
        ax.plot(times, mean+width, color="white", lw=0.8, ls="--", alpha=0.8)
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("position (a.u.)")
        fig.colorbar(image, ax=ax, pad=0.015, format=NUMBER_FORMATTER)

    profile_ax.plot(q, obs["proton_density"][0]/np.max(obs["proton_density"][0]), ls="--", color=COLORS[1])
    profile_ax.plot(q, obs["proton_density"][-1]/np.max(obs["proton_density"][-1]), color=COLORS[1], label="proton")
    profile_ax.plot(R, obs["heavy_density"][0]/np.max(obs["heavy_density"][0]), ls="--", color=COLORS[2])
    profile_ax.plot(R, obs["heavy_density"][-1]/np.max(obs["heavy_density"][-1]), color=COLORS[2], label="heavy")
    if obs["electron_density"] is not None:
        electron = obs["electron_density"]
        profile_ax.plot(obs["x"], electron[0]/np.max(electron[0]), ls="--", color=COLORS[0])
        profile_ax.plot(obs["x"], electron[-1]/np.max(electron[-1]), color=COLORS[0], label="electron")
    profile_ax.set_title("Nuclear marginals on one axis: initial vs final", loc="left", fontweight="semibold")
    profile_ax.set_xlabel(r"position ($a_0$)")
    profile_ax.set_ylabel("marginal shape (peak = 1)")
    profile_ax.legend(frameon=False)
    profile_ax.grid(alpha=0.18)

    summary_ax.plot(times, obs["proton_mean"]-obs["proton_mean"][0], color=COLORS[1], label=r"$\Delta\langle q\rangle$")
    summary_ax.plot(times, obs["heavy_mean"]-obs["heavy_mean"][0], color=COLORS[2], label=r"$\Delta\langle R\rangle$")
    summary_ax.plot(times, obs["proton_width"]-obs["proton_width"][0], color=COLORS[1], ls="--", label=r"$\Delta\sigma_q$")
    summary_ax.plot(times, obs["heavy_width"]-obs["heavy_width"][0], color=COLORS[2], ls="--", label=r"$\Delta\sigma_R$")
    summary_ax.set_title("How centers and widths changed", loc="left", fontweight="semibold")
    summary_ax.set_xlabel("time (fs)")
    summary_ax.set_ylabel(r"change ($a_0$)")
    summary_ax.grid(alpha=0.18)
    summary_ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("1 | What moved? Electronic character and nuclear marginals", fontsize=14, fontweight="bold")
    path = outdir/"01_particle_motion.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Born--Huang nuclear motion 저장: {path}")


def plot_state_populations(data, obs, outdir, dpi):
    times = obs["times_fs"]
    populations = obs["normalized_state_populations"]
    n_states = populations.shape[1]
    positive = populations[populations > 0.0]
    vmin = max(min(float(np.min(positive)) if positive.size else 1.0e-12, 1.0e-8), 1.0e-14)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    image = axes[0, 0].pcolormesh(
        times, np.arange(n_states), np.maximum(populations.T, vmin),
        shading="nearest", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=1.0),
    )
    axes[0, 0].set_yticks(np.arange(n_states))
    axes[0, 0].set_ylabel("BO state n")
    axes[0, 0].set_xlabel("time (fs)")
    axes[0, 0].set_title("BO-state composition", loc="left", fontweight="semibold")
    fig.colorbar(image, ax=axes[0, 0], label="population", pad=0.015)
    initial = int(np.argmax(populations[0]))
    for state in range(n_states):
        if state == initial:
            continue
        axes[0, 1].semilogy(
            times, np.maximum(populations[:, state], 1.0e-14),
            color=COLORS[state % len(COLORS)], label=rf"$P_{state}$",
        )
    axes[0, 1].semilogy(
        times, np.maximum(np.abs(obs["norm"]-1.0), 1.0e-14),
        color="black", ls="--", label=r"$|\|\Psi\|^2-1|$",
    )
    axes[0, 1].set_title(rf"Transfer away from initial BO state $n={initial}$", loc="left", fontweight="semibold")
    axes[0, 1].set_xlabel("time (fs)")
    axes[0, 1].set_ylabel("population / error")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(frameon=False, ncol=min(4, n_states))
    for ax, coordinate in zip(axes[1], ("q", "R")):
        grid, surfaces, packets, subtitle = _surface_slice(data, obs, -1, coordinate)
        _plot_bo_wavepackets(ax, grid, surfaces, packets, subtitle)
        ax.set_xlabel(rf"{coordinate} ($a_0$)")
    fig.suptitle(
        f"2 | BO transfer and state-projected nuclear wave packets | t={times[-1]:.3f} fs",
        fontsize=14, fontweight="bold",
    )
    path = outdir/"02_electronic_transitions.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Born--Huang state populations 저장: {path}")


def plot_energy_ladders(data, obs, outdir, dpi):
    times = obs["times_fs"]
    frames = selected_frames(len(times), min(4, len(times)))
    fig, axes = plt.subplots(2, len(frames), figsize=(4.8*len(frames), 9.0),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for column, frame in enumerate(frames):
        for row, coordinate in enumerate(("q", "R")):
            grid, surfaces, packets, subtitle = _surface_slice(
                data, obs, int(frame), coordinate
            )
            _plot_bo_wavepackets(
                axes[row, column], grid, surfaces, packets,
                f"t={times[frame]:.3f} fs; {subtitle}", legend=(column == 0),
            )
            axes[row, column].set_xlabel(rf"{coordinate} ($a_0$)")
    fig.suptitle(
        "5 | BO potential-energy surfaces and state-projected nuclear packets\n"
        "colored fill is the packet carried by that BO state (paper-style representation)",
        fontsize=14, fontweight="bold",
    )
    path = outdir/"05_born_huang_surface_dynamics.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Born--Huang energy ladder 저장: {path}")


def _support_field(values, density, floor=1.0e-3):
    cutoff = floor*max(float(np.max(density)), 1.0e-300)
    return np.where(density >= cutoff, values, np.nan)


def _potential_frame_fields(data, obs, diagnostics, frame, floor=1.0e-3):
    """Grid-report-equivalent occupied-support exact-potential fields."""
    density = obs["nuclear_joint_density"][frame]
    heavy = obs["heavy_density"][frame]
    eps1_raw = np.asarray(data["epsilon_1"])[frame]
    eps2_raw = np.asarray(data["epsilon_2"])[frame]
    eps1_full = density_weighted_shift(eps1_raw, density, floor)
    heavy_cutoff = floor*max(float(np.max(heavy)), 1.0e-300)
    heavy_support = heavy >= heavy_cutoff
    eps2_full = density_weighted_shift(eps2_raw, heavy, floor)
    alpha_full = np.asarray(data["alpha"])[frame]
    phase_R_full = diagnostics["phase_gradient_R_chi"][frame]
    momentum_R_full = diagnostics["momentum_R_outer"][frame]
    current_R_full = diagnostics["heavy_current"][frame]
    force_R_full = diagnostics["force_R"][frame]
    return dict(
        density=density,
        density_alpha=density_display_alpha(density, floor),
        heavy=heavy,
        heavy_support=heavy_support,
        eps1=_support_field(eps1_full, density, floor),
        eps1_full=eps1_full,
        a=_support_field(np.asarray(data["a"])[frame], density, floor),
        a_full=np.asarray(data["a"])[frame],
        b=_support_field(np.asarray(data["b"])[frame], density, floor),
        b_full=np.asarray(data["b"])[frame],
        eps2=np.where(heavy_support, eps2_full, np.nan),
        alpha=np.where(heavy_support, alpha_full, np.nan),
        phase_R=np.where(heavy_support, phase_R_full, np.nan),
        momentum_R=np.where(heavy_support, momentum_R_full, np.nan),
        current_R=np.where(heavy_support, current_R_full, np.nan),
        force_R=np.where(heavy_support, force_R_full, np.nan),
        eps2_full=eps2_full,
        alpha_full=alpha_full,
        phase_R_full=phase_R_full,
        momentum_R_full=momentum_R_full,
        current_R_full=current_R_full,
        force_R_full=force_R_full,
    )


def _potential_limits(fields):
    """Trajectory-wide scales shared by the static and animated dashboards."""
    specifications = {
        "eps1": False,
        "connection": True,
        "eps2": False,
        "momentum_component_R": True,
        "momentum_R": True,
        "current_R": True,
        "force_R": True,
    }
    bounds = {key: [] for key in specifications}

    def record(key, values):
        finite = np.asarray(values)[np.isfinite(values)]
        if not finite.size:
            return
        if specifications[key]:
            # A handful of finite tail-adjacent spikes must not bleach the
            # occupied-support color map. Values outside this plotting bound
            # remain in the data and are shown by an extended colorbar.
            bounds[key].append(float(np.percentile(np.abs(finite), 99.0)))
        else:
            bounds[key].append((float(np.min(finite)), float(np.max(finite))))

    # Only percentile scalars survive each iteration.  This is intentionally
    # streaming: a 50 fs q-R field can be several GiB if copied per frame.
    for item in fields:
        record("eps1", item["eps1"])
        record("connection", item["a"])
        record("connection", item["b"])
        record("eps2", item["eps2"])
        for key in ("alpha", "phase_R"):
            record("momentum_component_R", item[key])
        record("momentum_R", item["momentum_R"])
        record("current_R", item["current_R"])
        record("force_R", item["force_R"])

    result = {}
    for key, symmetric in specifications.items():
        values = bounds[key]
        if not values:
            result[key] = (-1.0, 1.0)
        elif symmetric:
            bound = max(float(np.percentile(values, 98.0)), 1.0e-12)
            result[key] = (-bound, bound)
        else:
            low = min(value[0] for value in values)
            high = max(value[1] for value in values)
            if high <= low:
                padding = max(abs(low)*1.0e-6, 1.0e-12)
                low, high = low-padding, high+padding
            result[key] = (low, high)
    # a and b are components of one electronic connection and therefore use
    # exactly the same symmetric color scale.
    result["a"] = result["connection"]
    result["b"] = result["connection"]
    del result["connection"]
    return result


def _scaled_density_line(axis, coordinate, density):
    """Overlay occupied support in axes-height units without changing y limits."""
    scaled = np.asarray(density, float)/max(
        float(np.max(density)), 1.0e-300
    )
    line, = axis.plot(
        coordinate, scaled, transform=axis.get_xaxis_transform(),
        color=HEAVY_DENSITY_COLOR, lw=1.1, alpha=0.42,
        label=r"$\rho_R$ (scaled)", zorder=0,
    )
    return line


def _support_tail_lines(
    axis, coordinate, occupied, full, support, *, color, label,
    linewidth=1.8, linestyle="-", zorder=2,
):
    """Draw trusted support solid and low-density continuation thin/dotted."""
    full = np.asarray(full, float)
    support = np.asarray(support, bool)
    tail = np.where(~support & np.isfinite(full), full, np.nan)
    tail_line, = axis.plot(
        coordinate, tail, color=color, lw=0.55, ls=":", alpha=0.22,
        zorder=max(zorder-1.5, 0),
    )
    support_line, = axis.plot(
        coordinate, occupied, color=color, lw=linewidth, ls=linestyle,
        label=label, zorder=zorder,
    )
    return support_line, tail_line


def plot_exact_potentials(data, obs, diagnostics, outdir, dpi, frame=-1):
    """Same six-panel scalar/connection/transport story as the grid report."""
    q, R, times = obs["q"], obs["R"], obs["times_fs"]
    item = _potential_frame_fields(data, obs, diagnostics, frame)
    scale_frames = selected_frames(len(times), min(180, len(times)))
    limits = _potential_limits(
        _potential_frame_fields(data, obs, diagnostics, int(index))
        for index in scale_frames
    )
    fields = (
        (item["eps1_full"], "eps1", r"Electron level: $\epsilon^{(1)}$", SCALAR_CMAP),
        (item["a_full"], "a", r"Electron connection $a$ along $q$", SIGNED_CMAP),
        (item["b_full"], "b", r"Electron connection $b$ along $R$", SIGNED_CMAP),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), constrained_layout=True)
    extent = [q[0], q[-1], R[0], R[-1]]
    for ax, (values, key, title, cmap) in zip(axes[0], fields):
        ax.set_facecolor(MASK_COLOR)
        image = ax.imshow(
            values.T, origin="lower", aspect="auto", extent=extent,
            cmap=masked_cmap(cmap),
            vmin=limits[key][0], vmax=limits[key][1],
            alpha=item["density_alpha"].T,
        )
        ax.contour(q, R, item["density"].T,
                   levels=[1.0e-3*np.max(item["density"])],
                   colors="white", linewidths=1.0)
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_xlabel(r"proton $q$ ($a_0$)")
        ax.set_ylabel(r"heavy $R$ ($a_0$)")
        fig.colorbar(
            image, ax=ax, pad=0.012, format=NUMBER_FORMATTER, extend="both"
        )

    epsilon_line, _epsilon_tail = _support_tail_lines(
        axes[1, 0], R, item["eps2"], item["eps2_full"],
        item["heavy_support"], color=COLORS[0],
        label=r"$\epsilon^{(2)}$", linewidth=2.0,
    )
    _scaled_density_line(axes[1, 0], R, item["heavy"])
    axes[1, 0].set_ylim(limits["eps2"])
    color_y_axis(axes[1, 0], COLORS[0], "shifted energy (Hartree)")
    axes[1, 0].set_title(r"Proton-heavy level: $\epsilon^{(2)}$", loc="left", fontweight="semibold")
    axes[1, 0].legend(frameon=False, fontsize=8)

    component_axis = axes[1, 1].twinx()
    alpha_line, _alpha_tail = _support_tail_lines(
        component_axis, R, item["alpha"], item["alpha_full"],
        item["heavy_support"], color=COLORS[3], label=r"$\alpha$",
    )
    phase_line, _phase_tail = _support_tail_lines(
        component_axis, R, item["phase_R"], item["phase_R_full"],
        item["heavy_support"], color=COLORS[1],
        label=r"$\partial_RS_\chi$", linewidth=1.6, linestyle="--",
    )
    momentum_line, _momentum_tail = _support_tail_lines(
        axes[1, 1], R, item["momentum_R"], item["momentum_R_full"],
        item["heavy_support"], color="black", label=r"$K_R^{(\chi)}$",
        linewidth=2.0,
    )
    _scaled_density_line(axes[1, 1], R, item["heavy"])
    axes[1, 1].set_ylim(limits["momentum_R"])
    component_axis.set_ylim(limits["momentum_component_R"])
    color_y_axis(axes[1, 1], "black", r"$K_R^{(\chi)}$ ($a_0^{-1}$)")
    color_y_axis(
        component_axis, COLORS[3],
        r"components $\partial_RS_\chi,\alpha$ ($a_0^{-1}$)",
    )
    axes[1, 1].set_title(r"Heavy momentum: $K_R^{(\chi)}=\partial_RS_\chi+\alpha$", loc="left", fontweight="semibold")
    axes[1, 1].legend(
        handles=[momentum_line, phase_line, alpha_line],
        frameon=False, fontsize=8,
    )

    current_line, _current_tail = _support_tail_lines(
        axes[1, 2], R, item["current_R"], item["current_R_full"],
        item["heavy_support"], color=CURRENT_COLOR,
        label=r"$j_R^{(\chi)}$", linewidth=2.0,
    )
    _scaled_density_line(axes[1, 2], R, item["heavy"])
    axes[1, 2].set_ylim(limits["current_R"])
    color_y_axis(axes[1, 2], CURRENT_COLOR, "heavy probability current")
    force_axis = axes[1, 2].twinx()
    force_line, _force_tail = _support_tail_lines(
        force_axis, R, item["force_R"], item["force_R_full"],
        item["heavy_support"], color=FORCE_COLOR,
        label=r"$F_R^{GI}$", linewidth=1.8,
    )
    force_axis.set_ylim(limits["force_R"])
    color_y_axis(force_axis, FORCE_COLOR, r"heavy drive (Hartree/$a_0$)")
    axes[1, 2].set_title(r"Heavy transport $j_R^{(\chi)}$ and drive $F_R^{GI}$", loc="left", fontweight="semibold")
    axes[1, 2].legend(
        handles=[current_line, force_line], frameon=False, fontsize=8,
        loc="upper left",
    )
    for ax in axes[1]:
        ax.set_xlabel(r"heavy $R$ ($a_0$)")
        # The occupied packet moves during the movie.  A fixed full-grid
        # window prevents late-time fields from leaving the visible x range.
        ax.set_xlim(float(R[0]), float(R[-1]))
        ax.grid(alpha=0.18)
    fig.suptitle(
        f"3 | Exact potentials, momentum, transport and force | t={times[frame]:.3f} fs\n"
        r"2D gray $\leq10^{-4}\rho_{\max}$, full color $\geq10^{-3}\rho_{\max}$; "
        "1D solid=occupied, dotted=tail; scalar offset only",
        fontsize=12.8, fontweight="bold",
    )
    path = outdir/"03_exact_potentials.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Born--Huang exact potentials 저장: {path}")


def plot_reliability(data, obs, outdir, dpi):
    times = obs["times_fs"]
    count = len(times)
    rate = _diag(data, "max_abs_full_norm_rate_after_product_projection", count)
    correction = _diag(data, "pnc_projection_correction", count)
    gamma_c = _diag(data, "max_abs_support_gamma_phi_dt", count)
    gamma_l = _diag(data, "max_abs_support_gamma_lam_dt", count)
    residual = _diag(data, "max_effective_product_residual_l2", count)
    projection = _diag(data, "max_relative_product_projection_l2", count)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    axes[0, 0].plot(times, obs["norm"]-1.0, color="black")
    axes[0, 0].axhline(0.0, color="0.6", lw=0.8)
    axes[0, 0].set_title("Signed full-wavefunction norm drift", loc="left", fontweight="semibold")
    axes[0, 0].set_ylabel(r"$\|\Psi\|^2-1$")
    axes[0, 0].set_xlabel("time (fs)")
    axes[0, 1].semilogy(times, np.maximum(rate, 1.0e-18), label="post-projection norm rate")
    axes[0, 1].semilogy(times, np.maximum(correction, 1.0e-18), label="PNC correction")
    axes[0, 1].set_title("Projection load", loc="left", fontweight="semibold")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].semilogy(times, np.maximum(gamma_c, 1.0e-18), label=r"$\Delta t\,\gamma_C$")
    axes[1, 0].semilogy(times, np.maximum(gamma_l, 1.0e-18), label=r"$\Delta t\,\gamma_\Lambda$")
    axes[1, 0].set_title("Occupied-support tangent rate", loc="left", fontweight="semibold")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].semilogy(times, np.maximum(residual, 1.0e-18), label="effective residual")
    axes[1, 1].semilogy(times, np.maximum(projection, 1.0e-18), label="relative projection")
    axes[1, 1].set_title("Product projection", loc="left", fontweight="semibold")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.set_xlabel("time (fs)")
        ax.grid(alpha=0.2)
    path = outdir/"04_numerical_reliability.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Born--Huang numerical reliability 저장: {path}")


def _save_animation(animation, fig, outdir, stem, fps, dpi, fmt):
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/f"{stem}.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=dpi)
    else:
        if fmt == "mp4":
            print(f"ffmpeg을 찾지 못해 {stem} 영상을 GIF로 저장합니다.")
        path = outdir/f"{stem}.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"Born--Huang dynamics 저장: {path}")
    return path


def make_overview_animation(data, obs, diagnostics, outdir, fps, max_frames, dpi, fmt):
    """Grid-report-equivalent dynamics, momentum, transport and drive."""
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    fig = plt.figure(figsize=(16.2, 8.6), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 3, height_ratios=(0.78, 1.22))
    marginal_ax = fig.add_subplot(grid_spec[0, :])
    field_axes = [fig.add_subplot(grid_spec[1, column]) for column in range(3)]

    # A single common position panel makes electron/proton/heavy motion
    # directly comparable.  The stored, normalized probability densities are
    # plotted without peak rescaling or smoothing.
    marginal_specs = []
    if obs["electron_density"] is not None:
        marginal_specs.append((
            np.asarray(obs["x"]), np.asarray(obs["electron_density"]),
            "electron", PARTICLE_COLORS["electron"],
            obs["electron_mean"], obs["electron_width"], "x",
        ))
    marginal_specs.extend((
        (q, np.asarray(obs["proton_density"]), "proton",
         PARTICLE_COLORS["proton"], obs["proton_mean"],
         obs["proton_width"], "q"),
        (R, np.asarray(obs["heavy_density"]), "heavy nucleus",
         PARTICLE_COLORS["heavy"], obs["heavy_mean"],
         obs["heavy_width"], "R"),
    ))
    marginal_lines = []
    for coordinate, density, name, color, _mean, _width, _symbol in marginal_specs:
        marginal_ax.plot(
            coordinate, density[0], color=color, lw=1.2, ls="--", alpha=0.38,
        )
        line, = marginal_ax.plot(
            coordinate, density[first], color=color, lw=2.2, label=name,
        )
        marginal_lines.append(line)
    coordinates = [spec[0] for spec in marginal_specs]
    position_min = min(float(values[0]) for values in coordinates)
    position_max = max(float(values[-1]) for values in coordinates)
    options = archive_arguments(
        data if hasattr(data, "files") else _ArchiveView(data)
    )
    if "x_min" in options:
        position_min = min(position_min, float(options["x_min"]))
    if "x_max" in options:
        position_max = max(position_max, float(options["x_max"]))
    add_fixed_center_markers(marginal_ax, options)
    density_maximum = max(
        float(np.nanmax(density))
        for _coordinate, density, *_rest in marginal_specs
    )
    marginal_ax.set(
        xlim=(position_min, position_max),
        ylim=(0.0, 1.05*max(density_maximum, 1.0e-300)),
        xlabel=r"common position coordinate ($a_0$)",
        ylabel=r"probability density ($a_0^{-1}$)",
    )
    marginal_ax.set_title(
        "Electron, proton and heavy-nucleus marginals on one position axis",
        loc="left", fontweight="semibold",
    )
    marginal_ax.grid(alpha=0.18, linewidth=0.7)
    marginal_ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    marginal_ax.text(
        0.995, 0.96,
        "solid = current; faint dashed = initial; stored density values",
        transform=marginal_ax.transAxes, ha="right", va="top", fontsize=7.8,
        color="0.25", bbox=dict(fc="white", ec="0.85", alpha=0.84, pad=3),
    )
    marginal_summary = marginal_ax.text(
        0.995, 0.04, "", transform=marginal_ax.transAxes,
        ha="right", va="bottom", fontsize=8.0, color="0.18",
        bbox=dict(fc="white", ec="0.85", alpha=0.84, pad=3),
    )

    def update_summary(frame):
        entries = [
            rf"$\langle {symbol}\rangle={mean[frame]:.3f}$, "
            rf"$\sigma_{symbol}={width[frame]:.3f}$"
            for _coordinate, _density, _name, _color, mean, width, symbol
            in marginal_specs
        ]
        marginal_summary.set_text("   |   ".join(entries)+r"  ($a_0$)")

    update_summary(first)

    sampled = [int(frame) for frame in frames]
    field_specs = (
        (
            "momentum_q", r"Mechanical proton momentum $K_q$",
            r"momentum ($a_0^{-1}$)", "linear",
        ),
        (
            "proton_current", r"Probability transport $j_q$",
            "proton probability current", "linear",
        ),
        (
            "force_q", r"Gauge-invariant drive $E_q$",
            r"drive (Hartree/$a_0$)", "symlog",
        ),
    )
    field_images = []
    extent = [q[0], q[-1], R[0], R[-1]]
    for ax, (key, label, unit, scale) in zip(field_axes, field_specs):
        # Scan occupied-support percentiles without retaining q-R copies for
        # every movie frame.  Rendering below always uses the unmodified raw
        # diagnostic; density only controls its display opacity.
        frame_bounds = []
        frame_typical = []
        for frame in sampled:
            occupied = _support_field(
                np.asarray(diagnostics[key][frame], float),
                obs["nuclear_joint_density"][frame],
            )
            finite = np.abs(occupied[np.isfinite(occupied)])
            if finite.size:
                frame_bounds.append(float(np.percentile(finite, 99.0)))
                frame_typical.append(float(np.percentile(finite, 80.0)))
        maximum = max(
            float(np.percentile(frame_bounds, 98.0)) if frame_bounds else 0.0,
            1.0e-12,
        )
        if scale == "symlog":
            typical = max(frame_typical+[1.0e-12])
            linear_threshold = min(max(0.2*typical, 1.0e-12), maximum)
            norm = SymLogNorm(
                linthresh=linear_threshold, linscale=0.8,
                vmin=-maximum, vmax=maximum, base=10,
            )
            image_kwargs = {"norm": norm}
            scale_note = (
                rf"symlog: linear $|E|\leq{linear_threshold:.1e}$; "
                rf"max $={maximum:.1e}$"
            )
        else:
            low, high = -maximum, maximum
            image_kwargs = {"vmin": low, "vmax": high}
            scale_note = rf"robust occupied-support scale $\pm{high:.1e}$"
        ax.set_facecolor(MASK_COLOR)
        initial_raw = np.asarray(diagnostics[key][first], float)
        initial_opacity = density_display_alpha(
            obs["nuclear_joint_density"][first]
        )
        image = ax.imshow(
            initial_raw.T, origin="lower", aspect="auto", extent=extent,
            cmap=masked_cmap(SIGNED_CMAP), alpha=initial_opacity.T,
            **image_kwargs,
        )
        ax.set_title(label, loc="left", fontweight="semibold")
        ax.set_xlabel(r"proton $q$ ($a_0$)")
        ax.set_ylabel(r"heavy $R$ ($a_0$)")
        ax.text(
            0.985, 0.025, scale_note, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.2, color="0.18",
            bbox=dict(fc="white", ec="none", alpha=0.78, pad=2),
        )
        fig.colorbar(
            image, ax=ax, pad=0.01, format=NUMBER_FORMATTER, label=unit,
            extend="both",
        )
        field_images.append((image, key))
    title = fig.suptitle(
        f"Born--Huang dynamics overview | t={times[first]:.4f} fs\n"
        r"gray below $10^{-4}\rho_{\max}$, full color above $10^{-3}\rho_{\max}$; "
        "white: signed field near zero; "
        "all color scales fixed over the trajectory"
    )

    def update(number):
        frame = int(frames[number])
        for line, (_coordinate, density, *_rest) in zip(
            marginal_lines, marginal_specs
        ):
            line.set_ydata(density[frame])
        update_summary(frame)
        frame_opacity = density_display_alpha(
            obs["nuclear_joint_density"][frame]
        )
        for image, key in field_images:
            image.set_data(np.asarray(diagnostics[key][frame], float).T)
            image.set_alpha(frame_opacity.T)
        title.set_text(
            f"Born--Huang dynamics overview | t={times[frame]:.4f} fs | "
            f"norm-1={obs['norm'][frame]-1:+.2e}\n"
            r"gray below $10^{-4}\rho_{\max}$, full color above $10^{-3}\rho_{\max}$; "
            "white: signed field near zero; "
            "all color scales fixed over the trajectory"
        )
        return (
            *marginal_lines, marginal_summary,
            *[item[0] for item in field_images], title,
        )

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(
        animation, fig, outdir, "mcef_dynamics_overview",
        fps, dpi, fmt,
    )


def _robust_animation_limit(values, frames, shift=None):
    samples = []
    for frame in frames:
        array = np.asarray(values[int(frame)], float)
        if shift is not None:
            array = array-shift(int(frame), array)
        finite = np.abs(array[np.isfinite(array)])
        if finite.size:
            samples.append(float(np.percentile(finite, 99.5)))
    return max(samples+[1.0e-12])


def make_potential_animation(data, obs, outdir, fps, max_frames, dpi, fmt):
    """Animate the full grid-style exact-potential/transport dashboard."""
    required = ("epsilon_1", "epsilon_2", "a", "b", "alpha")
    missing = [key for key in required if key not in data]
    if missing:
        print("Born--Huang potential 영상 생략; archive key 없음: " + ", ".join(missing))
        return None
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    frames = selected_frames(len(times), min(max_frames, len(times)))
    diagnostics = _diagnostics(data)
    sampled = [int(frame) for frame in frames]
    plot_limits = _potential_limits(
        _potential_frame_fields(data, obs, diagnostics, frame)
        for frame in sampled
    )
    first = sampled[0]
    first_item = _potential_frame_fields(data, obs, diagnostics, first)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), constrained_layout=True)
    map_axes = axes[0]
    arrays = (
        first_item["eps1_full"], first_item["a_full"], first_item["b_full"],
    )
    names = (r"shifted $\epsilon_1(q,R)$", r"$a(q,R)$", r"$b(q,R)$")
    map_limits = (
        plot_limits["eps1"], plot_limits["a"], plot_limits["b"],
    )
    images = []
    for ax, array, name, bound in zip(map_axes, arrays, names, map_limits):
        ax.set_facecolor(MASK_COLOR)
        image = ax.imshow(
            array.T, origin="lower", aspect="auto",
            extent=[q[0], q[-1], R[0], R[-1]],
            cmap=masked_cmap(
                SCALAR_CMAP if "epsilon" in name else SIGNED_CMAP
            ),
            vmin=bound[0], vmax=bound[1],
            alpha=first_item["density_alpha"].T,
        )
        ax.set_xlabel("proton q")
        ax.set_ylabel("heavy R")
        ax.set_title(name, loc="left", fontweight="semibold")
        fig.colorbar(
            image, ax=ax, pad=0.012, format=NUMBER_FORMATTER, extend="both"
        )
        images.append(image)

    epsilon_line, epsilon_tail = _support_tail_lines(
        axes[1, 0], R, first_item["eps2"], first_item["eps2_full"],
        first_item["heavy_support"], color=COLORS[0],
        label=r"$\epsilon^{(2)}$", linewidth=2.0,
    )
    heavy_lines = [
        _scaled_density_line(axis, R, first_item["heavy"])
        for axis in axes[1]
    ]
    axes[1, 0].set_ylim(plot_limits["eps2"])
    color_y_axis(axes[1, 0], COLORS[0], "shifted energy (Hartree)")
    axes[1, 0].set_title(r"Proton-heavy level: $\epsilon^{(2)}$", loc="left", fontweight="semibold")
    axes[1, 0].legend(frameon=False, fontsize=8)
    component_ax = axes[1, 1].twinx()
    alpha_line, alpha_tail = _support_tail_lines(
        component_ax, R, first_item["alpha"], first_item["alpha_full"],
        first_item["heavy_support"], color=COLORS[3], label=r"$\alpha$",
    )
    phase_line, phase_tail = _support_tail_lines(
        component_ax, R, first_item["phase_R"], first_item["phase_R_full"],
        first_item["heavy_support"], color=COLORS[1],
        label=r"$\partial_RS_\chi$", linewidth=1.6, linestyle="--",
    )
    momentum_line, momentum_tail = _support_tail_lines(
        axes[1, 1], R, first_item["momentum_R"],
        first_item["momentum_R_full"], first_item["heavy_support"],
        color="black", label=r"$K_R^{(\chi)}$", linewidth=2.0,
    )
    axes[1, 1].set_ylim(plot_limits["momentum_R"])
    component_ax.set_ylim(plot_limits["momentum_component_R"])
    color_y_axis(axes[1, 1], "black", r"$K_R^{(\chi)}$ ($a_0^{-1}$)")
    color_y_axis(
        component_ax, COLORS[3],
        r"components $\partial_RS_\chi,\alpha$ ($a_0^{-1}$)",
    )
    axes[1, 1].set_title(r"$K_R^{(\chi)}=\partial_RS_\chi+\alpha$", loc="left", fontweight="semibold")
    axes[1, 1].legend(
        handles=[momentum_line, phase_line, alpha_line],
        frameon=False, fontsize=8,
    )
    current_line, current_tail = _support_tail_lines(
        axes[1, 2], R, first_item["current_R"],
        first_item["current_R_full"], first_item["heavy_support"],
        color=CURRENT_COLOR, label=r"$j_R^{(\chi)}$", linewidth=2.0,
    )
    force_ax = axes[1, 2].twinx()
    force_line, force_tail = _support_tail_lines(
        force_ax, R, first_item["force_R"], first_item["force_R_full"],
        first_item["heavy_support"], color=FORCE_COLOR,
        label=r"$F_R^{GI}$", linewidth=1.8,
    )
    axes[1, 2].set_ylim(plot_limits["current_R"])
    force_ax.set_ylim(plot_limits["force_R"])
    color_y_axis(axes[1, 2], CURRENT_COLOR, "heavy probability current")
    color_y_axis(force_ax, FORCE_COLOR, r"heavy drive (Hartree/$a_0$)")
    axes[1, 2].set_title(r"Transport $j_R^{(\chi)}$ and drive $F_R^{GI}$", loc="left", fontweight="semibold")
    axes[1, 2].legend(
        handles=[current_line, force_line], frameon=False, fontsize=8,
        loc="upper left",
    )
    for ax in axes[1]:
        ax.set_xlabel("heavy R")
        ax.set_xlim(float(R[0]), float(R[-1]))
        ax.grid(alpha=0.18)
    title = fig.suptitle(
        f"Born--Huang exact potentials | t={times[first]:.4f} fs | "
        "solid=occupied, thin dotted=low density"
    )

    def update(number):
        frame = int(frames[number])
        item = _potential_frame_fields(data, obs, diagnostics, frame)
        for image, key in zip(images, ("eps1_full", "a_full", "b_full")):
            image.set_data(item[key].T)
            image.set_alpha(item["density_alpha"].T)
        epsilon_line.set_ydata(item["eps2"])
        epsilon_tail.set_ydata(np.where(
            ~item["heavy_support"], item["eps2_full"], np.nan
        ))
        scaled_heavy = item["heavy"]/max(
            float(np.max(item["heavy"])), 1.0e-300
        )
        for density_line in heavy_lines:
            density_line.set_ydata(scaled_heavy)
        alpha_line.set_ydata(item["alpha"])
        alpha_tail.set_ydata(np.where(
            ~item["heavy_support"], item["alpha_full"], np.nan
        ))
        phase_line.set_ydata(item["phase_R"])
        phase_tail.set_ydata(np.where(
            ~item["heavy_support"], item["phase_R_full"], np.nan
        ))
        momentum_line.set_ydata(item["momentum_R"])
        momentum_tail.set_ydata(np.where(
            ~item["heavy_support"], item["momentum_R_full"], np.nan
        ))
        current_line.set_ydata(item["current_R"])
        current_tail.set_ydata(np.where(
            ~item["heavy_support"], item["current_R_full"], np.nan
        ))
        force_line.set_ydata(item["force_R"])
        force_tail.set_ydata(np.where(
            ~item["heavy_support"], item["force_R_full"], np.nan
        ))
        title.set_text(
            f"Born--Huang exact potentials | t={times[frame]:.4f} fs | "
            "solid=occupied, thin dotted=low density"
        )
        return (
            *images, epsilon_line, epsilon_tail, *heavy_lines,
            alpha_line, alpha_tail, phase_line, phase_tail,
            momentum_line, momentum_tail, current_line, current_tail,
            force_line, force_tail, title,
        )

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(
        animation, fig, outdir, "mcef_exact_potentials",
        fps, dpi, fmt,
    )


def make_state_ladder_animation(data, obs, outdir, fps, max_frames, dpi, fmt):
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    populations = obs["normalized_state_populations"]
    states = np.arange(populations.shape[1])
    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    q_surface_ax, R_surface_ax, joint_ax, population_ax = axes.flat
    vmax = joint_density_limit(obs["nuclear_joint_density"])
    joint_image = joint_ax.imshow(
        obs["nuclear_joint_density"][first].T, origin="lower", aspect="auto",
        extent=[q[0], q[-1], R[0], R[-1]], cmap=JOINT_CMAP,
        vmin=0.0, vmax=vmax,
    )
    joint_ax.set_xlabel("proton q")
    joint_ax.set_ylabel("heavy R")
    joint_ax.set_title("Nuclear joint density", loc="left", fontweight="semibold")
    fig.colorbar(joint_image, ax=joint_ax, pad=0.012, format=NUMBER_FORMATTER)

    for ax, coordinate in ((q_surface_ax, "q"), (R_surface_ax, "R")):
        grid, surfaces, packets, subtitle = _surface_slice(data, obs, first, coordinate)
        _plot_bo_wavepackets(ax, grid, surfaces, packets, subtitle)
        ax.set_xlabel(rf"{coordinate} ($a_0$)")

    for state in states:
        population_ax.semilogy(
            times, np.maximum(populations[:, state], 1.0e-14),
            color=COLORS[state % len(COLORS)], label=rf"$P_{state}$",
        )
    time_marker = population_ax.axvline(times[first], color="black", lw=1.2)
    population_ax.set_ylim(1.0e-12, 1.5)
    population_ax.set_xlabel("time (fs)")
    population_ax.set_ylabel("normalized population")
    population_ax.set_title("Population transfer", loc="left", fontweight="semibold")
    population_ax.legend(frameon=False, ncol=3, fontsize=8)
    title = fig.suptitle(f"Born--Huang dynamics | t={times[first]:.4f} fs")

    def update(number):
        frame = int(frames[number])
        joint_image.set_data(obs["nuclear_joint_density"][frame].T)
        for ax, coordinate in ((q_surface_ax, "q"), (R_surface_ax, "R")):
            ax.clear()
            grid, surfaces, packets, subtitle = _surface_slice(data, obs, frame, coordinate)
            _plot_bo_wavepackets(ax, grid, surfaces, packets, subtitle)
            ax.set_xlabel(rf"{coordinate} ($a_0$)")
        time_marker.set_xdata([times[frame], times[frame]])
        title.set_text(
            f"Born--Huang dynamics | t={times[frame]:.4f} fs | "
            f"norm-1={obs['norm'][frame]-1:+.2e}"
        )
        return joint_image, time_marker, title

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(
        animation, fig, outdir, "mcef_physical_interpretation",
        fps, dpi, fmt,
    )


def make_coordinate_focus_animations(
    data, obs, diagnostics, outdir, fps, max_frames, dpi, fmt,
    marginal_ymax=1.5, marginal_xmax=12.0,
):
    """Reuse exact-potential diagnostics as q- and R-focused line movies."""
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    dR = float(R[1]-R[0]) if len(R) > 1 else 1.0
    joint = np.asarray(obs["nuclear_joint_density"], float)
    proton = np.asarray(obs["proton_density"], float)
    heavy = np.asarray(obs["heavy_density"], float)
    epsilon_q = np.zeros_like(proton)
    momentum_q = np.zeros_like(proton)
    current_q = np.sum(
        np.asarray(diagnostics["proton_current"], float), axis=2
    )*dR
    epsilon_R = np.zeros_like(heavy)
    for frame in range(len(times)):
        item = _potential_frame_fields(data, obs, diagnostics, frame)
        denominator = np.maximum(proton[frame], 1.0e-300)
        epsilon_q[frame] = np.sum(
            joint[frame]*item["eps1_full"], axis=1
        )*dR/denominator
        momentum_q[frame] = np.sum(
            joint[frame]*np.asarray(diagnostics["momentum_q"])[frame], axis=1
        )*dR/denominator
        epsilon_R[frame] = item["eps2_full"]
    options = archive_arguments(
        data if hasattr(data, "files") else _ArchiveView(data)
    )
    make_coordinate_focus_animation(
        times_fs=times, coordinate=R, marginal=heavy,
        profiles=(
            (r"Proton-heavy level $\epsilon^{(2)}(R)$",
             "shifted energy (Hartree)", epsilon_R, COLORS[0], False),
            (r"Heavy mechanical momentum $K_R^{(\chi)}$",
             r"momentum ($a_0^{-1}$)",
             diagnostics["momentum_R_outer"], "black", True),
            (r"Heavy probability transport $j_R^{(\chi)}$",
             "heavy probability current",
             diagnostics["heavy_current"], CURRENT_COLOR, True),
        ),
        options=options, outdir=outdir, fps=fps, max_frames=max_frames,
        dpi=dpi, fmt=fmt, particle_name="Heavy nucleus", coordinate_symbol="R",
        color=PARTICLE_COLORS["heavy"], stem="heavy_coordinate_dynamics",
        marginal_ymax=marginal_ymax, marginal_xmax=marginal_xmax,
    )
    make_coordinate_focus_animation(
        times_fs=times, coordinate=q, marginal=proton,
        profiles=(
            (r"Density-conditioned $\bar\epsilon^{(1)}(q)$",
             "shifted energy (Hartree)", epsilon_q, COLORS[0], False),
            (r"Density-conditioned mechanical momentum $\bar K_q$",
             r"momentum ($a_0^{-1}$)", momentum_q, "black", True),
            (r"Integrated probability transport $J_q$",
             "proton probability current", current_q, CURRENT_COLOR, True),
        ),
        options=options, outdir=outdir, fps=fps, max_frames=max_frames,
        dpi=dpi, fmt=fmt, particle_name="Proton", coordinate_symbol="q",
        color=PARTICLE_COLORS["proton"], stem="proton_coordinate_dynamics",
        marginal_ymax=marginal_ymax, marginal_xmax=marginal_xmax,
    )


def run(archive, outdir, *, dpi=180, no_animation=False, fps=12,
        max_frames=180, animation_dpi=110, fmt="mp4",
        marginal_ymax=1.5, marginal_xmax=12.0):
    """Create the coefficient-native compact report and return observables."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_archive(archive)
    obs = calculate_observables(data)
    diagnostics = _diagnostics(data)
    plot_nuclear_motion(obs, outdir, dpi)
    plot_state_populations(data, obs, outdir, dpi)
    plot_exact_potentials(data, obs, diagnostics, outdir, dpi)
    plot_reliability(data, obs, outdir, dpi)
    plot_energy_ladders(data, obs, outdir, dpi)
    if not no_animation:
        print("Born--Huang compact dynamics 기본 3개 + marginal/coordinate 영상 생성")
        make_overview_animation(
            data, obs, diagnostics, outdir, fps, max_frames, animation_dpi, fmt
        )
        make_potential_animation(
            data, obs, outdir, fps, max_frames, animation_dpi, fmt
        )
        make_state_ladder_animation(
            data, obs, outdir, fps, max_frames, animation_dpi, fmt
        )
        if obs["electron_density"] is not None:
            make_fixed_scale_marginal_animation(
                times_fs=obs["times_fs"],
                particle_series=(
                    ("electron", obs["x"], obs["electron_density"]),
                    ("proton", obs["q"], obs["proton_density"]),
                    ("heavy", obs["R"], obs["heavy_density"]),
                ),
                options=archive_arguments(
                    data if hasattr(data, "files") else _ArchiveView(data)
                ),
                outdir=outdir, fps=fps, max_frames=max_frames,
                dpi=animation_dpi, fmt=fmt, y_max=marginal_ymax,
                x_abs_max=marginal_xmax,
                title_prefix="Direct MCEF factor dynamics",
            )
        else:
            print(
                "fixed-scale 3-particle marginal 영상 생략: "
                "electron_density가 archive에 없습니다."
            )
        make_coordinate_focus_animations(
            data, obs, diagnostics, outdir, fps, max_frames,
            animation_dpi, fmt, marginal_ymax, marginal_xmax,
        )
    payload = {
        key: value for key, value in obs.items()
        if key != "nuclear_joint_density" and value is not None
    }
    np.savez_compressed(outdir/"report_observables.npz", **payload)
    stored = np.asarray(data.get("args", np.empty(0, dtype=object))).reshape(-1)
    options = stored[0] if stored.size == 1 and isinstance(stored[0], dict) else {}
    print(
        "Born--Huang compact report 완료: "
        f"{outdir}; initial state n={int(options.get('electron_excitation', 0))}"
    )
    return obs
