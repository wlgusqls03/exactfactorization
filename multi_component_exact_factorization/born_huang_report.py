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
from .visualize import NUMBER_FORMATTER, selected_frames


COLORS = ("#2878B5", "#E07A2D", "#3A9654", "#B05279", "#7B61A8", "#8C6D31")


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
    """BO surfaces along q or R through the occupied nuclear-density peak."""
    density = obs["nuclear_joint_density"][frame]
    iq, iR = np.unravel_index(int(np.argmax(density)), density.shape)
    energies = np.asarray(data["bo_energies"], float)
    if coordinate == "R":
        return obs["R"], energies[:, iq, :], obs["state_resolved_R_density"][frame], (
            rf"slice at $q_*={obs['q'][iq]:.3f}$"
        )
    return obs["q"], energies[:, :, iR], obs["state_resolved_q_density"][frame], (
        rf"slice at $R_*={obs['R'][iR]:.3f}$"
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


def plot_exact_potentials(data, obs, diagnostics, outdir, dpi, frame=-1):
    """Same scalar/connection/momentum/force story as the grid report."""
    q, R, times = obs["q"], obs["R"], obs["times_fs"]
    density = obs["nuclear_joint_density"][frame]
    heavy = obs["heavy_density"][frame]
    peak = np.unravel_index(int(np.argmax(density)), density.shape)
    eps1 = np.asarray(data["epsilon_1"])[frame]
    eps1 = _support_field(eps1-eps1[peak], density)
    fields = (
        (eps1, r"First TDPES $\epsilon^{(1)}$ (peak shifted)", "viridis", False),
        (_support_field(np.asarray(data["a"])[frame], density),
         r"Connection $a(q,R,t)$", "coolwarm", True),
        (_support_field(diagnostics["momentum_q"][frame], density),
         r"Mechanical proton momentum $K_q=\partial_qT+a$", "coolwarm", True),
        (_support_field(diagnostics["force_q"][frame], density),
         r"Gauge-invariant drive $-\partial_q\epsilon^{(1)}+\partial_ta$", "coolwarm", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    extent = [R[0], R[-1], q[0], q[-1]]
    for ax, (values, title, cmap, symmetric) in zip(axes.flat, fields):
        finite = values[np.isfinite(values)]
        kwargs = {}
        if symmetric:
            bound = max(float(np.percentile(np.abs(finite), 99.0)), 1.0e-14)
            kwargs.update(vmin=-bound, vmax=bound)
        image = ax.imshow(values, origin="lower", aspect="auto", extent=extent,
                          cmap=cmap, **kwargs)
        ax.contour(R, q, density, levels=[1.0e-3*np.max(density)],
                   colors="white", linewidths=1.0)
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_xlabel(r"heavy $R$ ($a_0$)")
        ax.set_ylabel(r"proton $q$ ($a_0$)")
        fig.colorbar(image, ax=ax, pad=0.012, format=NUMBER_FORMATTER)
    fig.suptitle(
        f"3 | Exact potentials, momentum and force | t={times[frame]:.3f} fs; gray/white boundary = occupied support",
        fontsize=14, fontweight="bold",
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
    populations = obs["normalized_state_populations"]
    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.6), constrained_layout=True)
    population_ax, joint_ax, marginal_ax = axes[0]
    vmax = max(float(np.max(obs["nuclear_joint_density"][frame])) for frame in frames)
    joint_image = joint_ax.imshow(
        obs["nuclear_joint_density"][first].T, origin="lower", aspect="auto",
        extent=[q[0], q[-1], R[0], R[-1]], cmap="magma", vmin=0.0, vmax=vmax,
    )
    peak = np.unravel_index(
        int(np.argmax(obs["nuclear_joint_density"][first])),
        obs["nuclear_joint_density"][first].shape,
    )
    peak_marker, = joint_ax.plot(q[peak[0]], R[peak[1]], "wo", ms=4)
    joint_ax.set_xlabel("proton q")
    joint_ax.set_ylabel("heavy R")
    joint_ax.set_title("Nuclear joint density", loc="left", fontweight="semibold")
    fig.colorbar(joint_image, ax=joint_ax, pad=0.012, format=NUMBER_FORMATTER)

    q_line, = marginal_ax.plot(q, obs["proton_density"][first], color=COLORS[0], label="q")
    R_line, = marginal_ax.plot(R, obs["heavy_density"][first], color=COLORS[2], label="R")
    marginal_ax.set_xlim(min(q[0], R[0]), max(q[-1], R[-1]))
    marginal_ax.set_ylim(0.0, 1.08*max(np.max(obs["proton_density"]), np.max(obs["heavy_density"])))
    marginal_ax.set_xlabel("position (a.u.)")
    marginal_ax.set_ylabel("probability density")
    marginal_ax.set_title("Nuclear marginals", loc="left", fontweight="semibold")
    marginal_ax.legend(frameon=False)

    electron_line = None
    if obs["electron_density"] is not None:
        electron = obs["electron_density"]
        population_ax.plot(obs["x"], electron[0], color="0.65", ls="--", label="initial")
        electron_line, = population_ax.plot(obs["x"], electron[first], color=COLORS[0], lw=2.0, label="current")
        population_ax.set(xlabel=r"electron $x$ ($a_0$)", ylabel="probability density")
        population_ax.set_ylim(0.0, 1.08*np.max(electron))
        population_ax.set_title("Electron marginal", loc="left", fontweight="semibold")
        population_ax.legend(frameon=False)
        time_marker = None
    else:
        for state in range(populations.shape[1]):
            population_ax.plot(times, populations[:, state], color=COLORS[state % len(COLORS)],
                               lw=1.7, label=rf"$P_{state}$")
        time_marker = population_ax.axvline(times[first], color="black", lw=1.2)
        population_ax.set(xlabel="time (fs)", ylabel="population", ylim=(-0.02, 1.02))
        population_ax.set_title("Electronic BO-state composition", loc="left", fontweight="semibold")
        population_ax.legend(frameon=False, fontsize=7, ncol=3)

    sampled = [int(frame) for frame in frames]
    field_specs = (
        ("momentum_q", r"Mechanical proton momentum $K_q$", "momentum"),
        ("proton_current", r"Probability transport $j_q$", "current"),
        ("force_q", r"Gauge-invariant drive $E_q$", "force"),
    )
    field_images = []
    extent = [R[0], R[-1], q[0], q[-1]]
    for ax, (key, label, _unit) in zip(axes[1], field_specs):
        arrays = [_support_field(diagnostics[key][frame], obs["nuclear_joint_density"][frame]) for frame in sampled]
        finite = np.concatenate([np.abs(value[np.isfinite(value)]) for value in arrays])
        bound = max(float(np.percentile(finite, 99.0)), 1.0e-14)
        image = ax.imshow(arrays[0], origin="lower", aspect="auto", extent=extent,
                          cmap="coolwarm", vmin=-bound, vmax=bound)
        ax.set_title(label, loc="left", fontweight="semibold")
        ax.set_xlabel(r"heavy $R$ ($a_0$)")
        ax.set_ylabel(r"proton $q$ ($a_0$)")
        fig.colorbar(image, ax=ax, pad=0.01, format=NUMBER_FORMATTER)
        field_images.append((image, arrays))
    title = fig.suptitle(f"Born--Huang dynamics overview | t={times[first]:.4f} fs")

    def update(number):
        frame = int(frames[number])
        density = obs["nuclear_joint_density"][frame]
        joint_image.set_data(density.T)
        peak = np.unravel_index(int(np.argmax(density)), density.shape)
        peak_marker.set_data([q[peak[0]]], [R[peak[1]]])
        q_line.set_ydata(obs["proton_density"][frame])
        R_line.set_ydata(obs["heavy_density"][frame])
        if time_marker is not None:
            time_marker.set_xdata([times[frame], times[frame]])
        if electron_line is not None:
            electron_line.set_ydata(obs["electron_density"][frame])
        for image, arrays in field_images:
            image.set_data(arrays[number])
        title.set_text(
            f"Born--Huang dynamics overview | t={times[frame]:.4f} fs | "
            f"norm-1={obs['norm'][frame]-1:+.2e}"
        )
        dynamic = [artist for artist in (time_marker, electron_line) if artist is not None]
        return joint_image, peak_marker, q_line, R_line, *dynamic, *[item[0] for item in field_images], title

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
    """Animate saved exact scalar/vector potentials on occupied coordinates."""
    required = ("epsilon_1", "epsilon_2", "a", "b", "alpha")
    missing = [key for key in required if key not in data]
    if missing:
        print("Born--Huang potential 영상 생략; archive key 없음: " + ", ".join(missing))
        return None
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    epsilon_1 = np.asarray(data["epsilon_1"], float)
    epsilon_2 = np.asarray(data["epsilon_2"], float)
    vector_a = np.asarray(data["a"], float)
    vector_b = np.asarray(data["b"], float)
    alpha = np.asarray(data["alpha"], float)

    def joint_peak_shift(frame, array):
        peak = np.unravel_index(
            int(np.argmax(obs["nuclear_joint_density"][frame])), array.shape
        )
        return float(array[peak])

    def heavy_peak_shift(frame, array):
        return float(array[int(np.argmax(obs["heavy_density"][frame]))])

    limits = (
        _robust_animation_limit(epsilon_1, frames, joint_peak_shift),
        _robust_animation_limit(vector_a, frames),
        _robust_animation_limit(vector_b, frames),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    map_axes = (axes[0, 0], axes[0, 1], axes[1, 0])
    arrays = (
        epsilon_1[first]-joint_peak_shift(first, epsilon_1[first]),
        vector_a[first], vector_b[first],
    )
    names = (r"shifted $\epsilon_1(q,R)$", r"$a(q,R)$", r"$b(q,R)$")
    images = []
    markers = []
    for ax, array, name, limit in zip(map_axes, arrays, names, limits):
        image = ax.imshow(
            array.T, origin="lower", aspect="auto",
            extent=[q[0], q[-1], R[0], R[-1]], cmap="coolwarm",
            vmin=-limit, vmax=limit,
        )
        peak = np.unravel_index(
            int(np.argmax(obs["nuclear_joint_density"][first])),
            obs["nuclear_joint_density"][first].shape,
        )
        marker, = ax.plot(q[peak[0]], R[peak[1]], "ko", ms=3.5)
        ax.set_xlabel("proton q")
        ax.set_ylabel("heavy R")
        ax.set_title(name, loc="left", fontweight="semibold")
        fig.colorbar(image, ax=ax, pad=0.012, format=NUMBER_FORMATTER)
        images.append(image)
        markers.append(marker)

    line_ax = axes[1, 1]
    shifted_epsilon_2 = epsilon_2[first]-heavy_peak_shift(first, epsilon_2[first])
    epsilon_line, = line_ax.plot(R, shifted_epsilon_2, color=COLORS[1], label=r"shifted $\epsilon_2$")
    line_ax.set_xlabel("heavy R")
    line_ax.set_ylabel(r"$\epsilon_2$ (a.u.)", color=COLORS[1])
    line_ax.tick_params(axis="y", colors=COLORS[1])
    alpha_ax = line_ax.twinx()
    alpha_line, = alpha_ax.plot(R, alpha[first], color=COLORS[4], label=r"$\alpha$")
    alpha_ax.set_ylabel(r"$\alpha$ (a.u.)", color=COLORS[4])
    alpha_ax.tick_params(axis="y", colors=COLORS[4])
    epsilon_limit = _robust_animation_limit(epsilon_2, frames, heavy_peak_shift)
    alpha_limit = _robust_animation_limit(alpha, frames)
    line_ax.set_ylim(-epsilon_limit, epsilon_limit)
    alpha_ax.set_ylim(-alpha_limit, alpha_limit)
    line_ax.set_title("Heavy scalar/vector potentials", loc="left", fontweight="semibold")
    line_ax.grid(alpha=0.18)
    title = fig.suptitle(f"Born--Huang exact potentials | t={times[first]:.4f} fs")

    def update(number):
        frame = int(frames[number])
        fields = (
            epsilon_1[frame]-joint_peak_shift(frame, epsilon_1[frame]),
            vector_a[frame], vector_b[frame],
        )
        density = obs["nuclear_joint_density"][frame]
        peak = np.unravel_index(int(np.argmax(density)), density.shape)
        for image, marker, field in zip(images, markers, fields):
            image.set_data(field.T)
            marker.set_data([q[peak[0]]], [R[peak[1]]])
        epsilon_line.set_ydata(
            epsilon_2[frame]-heavy_peak_shift(frame, epsilon_2[frame])
        )
        alpha_line.set_ydata(alpha[frame])
        title.set_text(f"Born--Huang exact potentials | t={times[frame]:.4f} fs")
        return *images, *markers, epsilon_line, alpha_line, title

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
    vmax = max(float(np.max(obs["nuclear_joint_density"][frame])) for frame in frames)
    joint_image = joint_ax.imshow(
        obs["nuclear_joint_density"][first].T, origin="lower", aspect="auto",
        extent=[q[0], q[-1], R[0], R[-1]], cmap="magma", vmin=0.0, vmax=vmax,
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


def run(archive, outdir, *, dpi=180, no_animation=False, fps=12,
        max_frames=180, animation_dpi=110, fmt="mp4"):
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
        print("Born--Huang compact dynamics 영상 3개 생성")
        make_overview_animation(
            data, obs, diagnostics, outdir, fps, max_frames, animation_dpi, fmt
        )
        make_potential_animation(
            data, obs, outdir, fps, max_frames, animation_dpi, fmt
        )
        make_state_ladder_animation(
            data, obs, outdir, fps, max_frames, animation_dpi, fmt
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
