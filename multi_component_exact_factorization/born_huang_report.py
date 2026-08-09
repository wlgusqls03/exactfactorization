#!/usr/bin/env python3
"""Compact figures and animations for electronic Born--Huang trajectories.

The coefficient archive intentionally omits the enormous static x-grid BO
eigenvectors unless ``--bo-save-basis-states`` is requested.  This report uses
only C-independent nuclear marginals, saved BO populations and BO energies, so
it remains useful for large production runs without reconstructing Phi(x,q,R).
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
import numpy as np

from .visualize import NUMBER_FORMATTER, selected_frames


COLORS = ("#2878B5", "#E07A2D", "#3A9654", "#B05279", "#7B61A8", "#8C6D31")


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
        }
        return {
            key: archive[key]
            for key in archive.files
            if key in wanted
        }


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
    return dict(
        times_fs=times, q=q, R=R, nuclear_joint_density=joint,
        proton_density=q_density, heavy_density=R_density,
        proton_mean=q_mean, proton_width=q_width,
        heavy_mean=R_mean, heavy_width=R_width,
        norm=norm, state_populations=populations,
        normalized_state_populations=normalized_populations,
        mean_bo_energies=mean_energies,
        outer_probability_q=q_outer, outer_probability_R=R_outer,
    )


def _diag(data, name, length):
    if name not in data:
        return np.zeros(length)
    values = np.asarray(data[name], float)
    return values if values.shape == (length,) else np.zeros(length)


def plot_nuclear_motion(obs, outdir, dpi):
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.5), constrained_layout=True)
    for ax, grid, density, mean, width, title in (
        (axes[0, 0], q, obs["proton_density"], obs["proton_mean"],
         obs["proton_width"], "Proton marginal"),
        (axes[0, 1], R, obs["heavy_density"], obs["heavy_mean"],
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

    axes[1, 0].plot(q, obs["proton_density"][0], ls="--", color=COLORS[0], label="q, initial")
    axes[1, 0].plot(q, obs["proton_density"][-1], color=COLORS[0], label="q, final")
    axes[1, 0].plot(R, obs["heavy_density"][0], ls="--", color=COLORS[2], label="R, initial")
    axes[1, 0].plot(R, obs["heavy_density"][-1], color=COLORS[2], label="R, final")
    axes[1, 0].set_title("Initial and final nuclear profiles", loc="left", fontweight="semibold")
    axes[1, 0].set_xlabel("position (a.u.)")
    axes[1, 0].set_ylabel("probability density")
    axes[1, 0].legend(frameon=False, ncol=2)

    axes[1, 1].semilogy(times, np.maximum(obs["outer_probability_q"], 1.0e-18), label="q outer 5")
    axes[1, 1].semilogy(times, np.maximum(obs["outer_probability_R"], 1.0e-18), label="R outer 5")
    axes[1, 1].set_title("Boundary probability", loc="left", fontweight="semibold")
    axes[1, 1].set_xlabel("time (fs)")
    axes[1, 1].set_ylabel("probability")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(frameon=False)
    path = outdir/"01_born_huang_nuclear_motion.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Born--Huang nuclear motion 저장: {path}")


def plot_state_populations(obs, outdir, dpi):
    times = obs["times_fs"]
    populations = obs["normalized_state_populations"]
    n_states = populations.shape[1]
    positive = populations[populations > 0.0]
    vmin = max(min(float(np.min(positive)) if positive.size else 1.0e-12, 1.0e-8), 1.0e-14)
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.5), constrained_layout=True)
    image = axes[0].pcolormesh(
        times, np.arange(n_states), np.maximum(populations.T, vmin),
        shading="nearest", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=1.0),
    )
    axes[0].set_yticks(np.arange(n_states))
    axes[0].set_ylabel("BO state n")
    axes[0].set_xlabel("time (fs)")
    axes[0].set_title("Normalized BO-state populations", loc="left", fontweight="semibold")
    fig.colorbar(image, ax=axes[0], label="population", pad=0.015)
    initial = int(np.argmax(populations[0]))
    for state in range(n_states):
        if state == initial:
            continue
        axes[1].semilogy(
            times, np.maximum(populations[:, state], 1.0e-14),
            color=COLORS[state % len(COLORS)], label=rf"$P_{state}$",
        )
    axes[1].semilogy(
        times, np.maximum(np.abs(obs["norm"]-1.0), 1.0e-14),
        color="black", ls="--", label=r"$|\|\Psi\|^2-1|$",
    )
    axes[1].set_title(rf"Transfer away from initial BO state $n={initial}$", loc="left", fontweight="semibold")
    axes[1].set_xlabel("time (fs)")
    axes[1].set_ylabel("population / error")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, ncol=min(4, n_states))
    path = outdir/"02_born_huang_state_populations.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Born--Huang state populations 저장: {path}")


def _ladder_scatter(ax, energies, populations, title):
    states = np.arange(len(populations))
    clipped = np.maximum(populations, 1.0e-14)
    sizes = 30.0+720.0*clipped**0.25
    artist = ax.scatter(
        states, energies, s=sizes, c=clipped, cmap="plasma",
        norm=LogNorm(vmin=1.0e-12, vmax=1.0), edgecolor="black", linewidth=0.5,
        zorder=3,
    )
    ax.plot(states, energies, color="0.65", lw=1.0, zorder=1)
    for state, (energy, population) in enumerate(zip(energies, populations)):
        ax.hlines(energy, state-0.28, state+0.28, color="0.25", lw=1.2, zorder=2)
        ax.annotate(f"{population:.1e}", (state, energy), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(states)
    ax.set_xlabel("BO state n")
    ax.set_ylabel(r"nuclear-density averaged $E_n$ (a.u.)")
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.grid(axis="y", alpha=0.18)
    return artist


def plot_energy_ladders(obs, outdir, dpi):
    times = obs["times_fs"]
    frames = selected_frames(len(times), min(5, len(times)))
    fig, axes = plt.subplots(1, len(frames), figsize=(4.0*len(frames), 5.8),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    last = None
    for ax, frame in zip(axes, frames):
        last = _ladder_scatter(
            ax, obs["mean_bo_energies"][frame],
            obs["normalized_state_populations"][frame],
            f"t = {times[frame]:.3f} fs",
        )
    if last is not None:
        fig.colorbar(last, ax=list(axes), label="BO population", pad=0.012)
    path = outdir/"03_born_huang_energy_ladder.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Born--Huang energy ladder 저장: {path}")


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
    path = outdir/"04_born_huang_numerical_reliability.png"
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


def make_overview_animation(obs, outdir, fps, max_frames, dpi, fmt):
    """Nuclear motion and BO populations in the standard overview role."""
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    populations = obs["normalized_state_populations"]
    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    joint_ax, marginal_ax, population_ax, motion_ax = axes.flat
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

    states = np.arange(populations.shape[1])
    bars = population_ax.bar(states, np.maximum(populations[first], 1.0e-12), color=[
        COLORS[state % len(COLORS)] for state in states
    ])
    population_ax.set_yscale("log")
    population_ax.set_ylim(1.0e-12, 1.5)
    population_ax.set_xticks(states)
    population_ax.set_xlabel("BO state n")
    population_ax.set_ylabel("normalized population")
    population_ax.set_title("Instantaneous BO populations", loc="left", fontweight="semibold")

    motion_ax.plot(times, obs["proton_mean"], color=COLORS[0], label=r"$\langle q\rangle$")
    motion_ax.plot(times, obs["heavy_mean"], color=COLORS[2], label=r"$\langle R\rangle$")
    edge_ax = motion_ax.twinx()
    edge_ax.semilogy(times, np.maximum(obs["outer_probability_q"], 1.0e-14),
                     color=COLORS[0], ls=":", label="q edge probability")
    edge_ax.semilogy(times, np.maximum(obs["outer_probability_R"], 1.0e-14),
                     color=COLORS[2], ls=":", label="R edge probability")
    edge_ax.set_ylabel("outer-5 probability")
    time_marker = motion_ax.axvline(times[first], color="black", lw=1.2)
    motion_ax.set_xlabel("time (fs)")
    motion_ax.set_title("Transport and boundary contact", loc="left", fontweight="semibold")
    handles, labels = motion_ax.get_legend_handles_labels()
    edge_handles, edge_labels = edge_ax.get_legend_handles_labels()
    motion_ax.legend(
        handles+edge_handles, labels+edge_labels,
        frameon=False, fontsize=8, ncol=2,
    )
    title = fig.suptitle(f"Born--Huang dynamics overview | t={times[first]:.4f} fs")

    def update(number):
        frame = int(frames[number])
        density = obs["nuclear_joint_density"][frame]
        joint_image.set_data(density.T)
        peak = np.unravel_index(int(np.argmax(density)), density.shape)
        peak_marker.set_data([q[peak[0]]], [R[peak[1]]])
        q_line.set_ydata(obs["proton_density"][frame])
        R_line.set_ydata(obs["heavy_density"][frame])
        for bar, value in zip(bars, populations[frame]):
            bar.set_height(max(float(value), 1.0e-12))
        time_marker.set_xdata([times[frame], times[frame]])
        title.set_text(
            f"Born--Huang dynamics overview | t={times[frame]:.4f} fs | "
            f"norm-1={obs['norm'][frame]-1:+.2e}"
        )
        return joint_image, peak_marker, q_line, R_line, *bars, time_marker, title

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(
        animation, fig, outdir, "born_huang_dynamics_overview",
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
        animation, fig, outdir, "born_huang_exact_potentials",
        fps, dpi, fmt,
    )


def make_state_ladder_animation(obs, outdir, fps, max_frames, dpi, fmt):
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    populations = obs["normalized_state_populations"]
    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    joint_ax, marginal_ax, ladder_ax, population_ax = axes.flat
    vmax = max(float(np.max(obs["nuclear_joint_density"][frame])) for frame in frames)
    joint_image = joint_ax.imshow(
        obs["nuclear_joint_density"][first].T, origin="lower", aspect="auto",
        extent=[q[0], q[-1], R[0], R[-1]], cmap="magma", vmin=0.0, vmax=vmax,
    )
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

    states = np.arange(populations.shape[1])
    ladder_line, = ladder_ax.plot(states, obs["mean_bo_energies"][first], color="0.65")
    ladder_points = ladder_ax.scatter(states, obs["mean_bo_energies"][first])
    ladder_ax.set_xticks(states)
    ladder_ax.set_xlabel("BO state n")
    ladder_ax.set_ylabel(r"averaged $E_n$ (a.u.)")
    ladder_ax.set_title("BO energy ladder and population", loc="left", fontweight="semibold")
    energy_min, energy_max = np.min(obs["mean_bo_energies"]), np.max(obs["mean_bo_energies"])
    margin = max(0.05*(energy_max-energy_min), 1.0e-3)
    ladder_ax.set_ylim(energy_min-margin, energy_max+margin)

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
        q_line.set_ydata(obs["proton_density"][frame])
        R_line.set_ydata(obs["heavy_density"][frame])
        energies = obs["mean_bo_energies"][frame]
        pop = np.maximum(populations[frame], 1.0e-14)
        ladder_line.set_ydata(energies)
        ladder_points.set_offsets(np.column_stack((states, energies)))
        ladder_points.set_sizes(30.0+720.0*pop**0.25)
        ladder_points.set_array(np.log10(pop))
        ladder_points.set_cmap("plasma")
        ladder_points.set_clim(-12.0, 0.0)
        time_marker.set_xdata([times[frame], times[frame]])
        title.set_text(
            f"Born--Huang dynamics | t={times[frame]:.4f} fs | "
            f"norm-1={obs['norm'][frame]-1:+.2e}"
        )
        return joint_image, q_line, R_line, ladder_line, ladder_points, time_marker, title

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(
        animation, fig, outdir, "born_huang_state_ladder_dynamics",
        fps, dpi, fmt,
    )


def run(archive, outdir, *, dpi=180, no_animation=False, fps=12,
        max_frames=180, animation_dpi=110, fmt="mp4"):
    """Create the coefficient-native compact report and return observables."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_archive(archive)
    obs = calculate_observables(data)
    plot_nuclear_motion(obs, outdir, dpi)
    plot_state_populations(obs, outdir, dpi)
    plot_energy_ladders(obs, outdir, dpi)
    plot_reliability(data, obs, outdir, dpi)
    if not no_animation:
        print("Born--Huang compact dynamics 영상 3개 생성")
        make_overview_animation(
            obs, outdir, fps, max_frames, animation_dpi, fmt
        )
        make_potential_animation(
            data, obs, outdir, fps, max_frames, animation_dpi, fmt
        )
        make_state_ladder_animation(
            obs, outdir, fps, max_frames, animation_dpi, fmt
        )
    payload = {
        key: value for key, value in obs.items()
        if key != "nuclear_joint_density"
    }
    np.savez_compressed(outdir/"report_observables.npz", **payload)
    stored = np.asarray(data.get("args", np.empty(0, dtype=object))).reshape(-1)
    options = stored[0] if stored.size == 1 and isinstance(stored[0], dict) else {}
    print(
        "Born--Huang compact report 완료: "
        f"{outdir}; initial state n={int(options.get('electron_excitation', 0))}"
    )
    return obs
