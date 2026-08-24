"""Standalone figures and movie for a direct discrete Born--Huang TDSE run.

Only reduced observables already stored by ``propagate_tdse`` are read.  In
particular, the usually very large ``tdse_coefficients`` member is never
materialized.  Plot limits are fixed over the full trajectory, and densities
and populations are displayed without smoothing or peak normalization.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from .core import AU_PER_FS, derivative
from .report_plot_style import (
    COLORS,
    CURRENT_COLOR,
    FORCE_COLOR,
    HEAVY_DENSITY_COLOR,
    JOINT_CMAP,
    LINK_CMAP,
    MASK_COLOR,
    PARTICLE_COLORS,
    SCALAR_CMAP,
    SIGNED_CMAP,
    add_fixed_center_markers,
    color_y_axis,
    density_display_alpha,
    density_weighted_shift,
    masked_cmap,
)
from .visualize import NUMBER_FORMATTER, selected_frames
from .marginal_movie import (
    make_fixed_scale_marginal_animation,
    make_relative_log_marginal_animation,
)
from .coordinate_focus_movie import make_coordinate_focus_animation


_REQUIRED = (
    "times_fs",
    "q",
    "R",
    "norm",
    "energy",
    "bo_populations",
    "joint_density",
    "proton_density",
    "heavy_density",
)


def _options(data):
    if "args" not in data.files:
        return {}
    stored = np.asarray(data["args"], dtype=object).reshape(-1)
    return stored[0] if stored.size == 1 and isinstance(stored[0], dict) else {}


def load_observables(archive):
    """Load reduced TDSE observables, deliberately skipping full coefficients."""
    archive = Path(archive)
    with np.load(archive, allow_pickle=True) as stored:
        missing = sorted(set(_REQUIRED).difference(stored.files))
        if missing:
            raise KeyError("TDSE archive에 필요한 key가 없습니다: " + ", ".join(missing))
        kind = str(np.asarray(stored.get("kind", "")).item())
        if not kind.startswith("direct_discrete_born_huang_tdse"):
            raise ValueError(f"direct discrete TDSE archive가 아닙니다: kind={kind!r}")
        values = {key: np.asarray(stored[key]) for key in _REQUIRED}
        for key in (
            "x",
            "electron_density",
            "bo_energies",
            "energy_imaginary_defect",
            "norm_rate",
            "outer_probability_q",
            "outer_probability_R",
            "fixed_center_crossing_q",
            "fixed_center_crossing_R",
            "propagation_completed",
            "requested_final_time_fs",
            "failure_reason",
        ):
            if key in stored.files:
                values[key] = np.asarray(stored[key])
        values["options"] = _options(stored)
    values["archive_path"] = archive.resolve()
    return values


def calculate_observables(data):
    times = np.asarray(data["times_fs"], float)
    q = np.asarray(data["q"], float)
    R = np.asarray(data["R"], float)
    proton = np.asarray(data["proton_density"], float)
    heavy = np.asarray(data["heavy_density"], float)
    joint = np.asarray(data["joint_density"], float)
    populations = np.asarray(data["bo_populations"], float)
    x = np.asarray(data["x"], float) if "x" in data else None
    electron = (
        np.asarray(data["electron_density"], float)
        if "electron_density" in data else None
    )
    if joint.shape != (len(times), len(q), len(R)):
        raise ValueError(
            "joint_density shape mismatch: "
            f"{joint.shape} != {(len(times), len(q), len(R))}"
        )
    if proton.shape != (len(times), len(q)) or heavy.shape != (len(times), len(R)):
        raise ValueError("stored marginal density shape가 grid/time과 맞지 않습니다")
    if electron is not None:
        if x is None or electron.shape != (len(times), len(x)):
            raise ValueError("stored electron density shape가 x grid/time과 맞지 않습니다")
    dq = float(q[1]-q[0]) if len(q) > 1 else 1.0
    dR = float(R[1]-R[0]) if len(R) > 1 else 1.0
    q_mass = np.sum(proton, axis=1)*dq
    R_mass = np.sum(heavy, axis=1)*dR
    q_mean = np.sum(proton*q[None, :], axis=1)*dq/np.maximum(q_mass, 1.0e-300)
    R_mean = np.sum(heavy*R[None, :], axis=1)*dR/np.maximum(R_mass, 1.0e-300)
    q_width = np.sqrt(
        np.maximum(
            np.sum(proton*(q[None, :]-q_mean[:, None])**2, axis=1)*dq
            /np.maximum(q_mass, 1.0e-300),
            0.0,
        )
    )
    R_width = np.sqrt(
        np.maximum(
            np.sum(heavy*(R[None, :]-R_mean[:, None])**2, axis=1)*dR
            /np.maximum(R_mass, 1.0e-300),
            0.0,
        )
    )
    result = dict(data)
    result.update(
        times_fs=times,
        x=x,
        electron_density=electron,
        q=q,
        R=R,
        proton_density=proton,
        heavy_density=heavy,
        joint_density=joint,
        bo_populations=populations,
        q_mean=q_mean,
        R_mean=R_mean,
        q_width=q_width,
        R_width=R_width,
        dq=dq,
        dR=dR,
        joint_vmax=max(float(np.nanmax(joint)), 1.0e-300),
        marginal_ymax=max(
            float(np.nanmax(proton)), float(np.nanmax(heavy)), 1.0e-300
        ),
    )
    return result


def _fixed_positions(options):
    positions = []
    for key, charge_key in (
        ("left_position", "left_charge"),
        ("right_position", "right_charge"),
    ):
        if key in options and np.isfinite(float(options[key])):
            value = float(options[key])
            charge = float(options.get(charge_key, 1.0))
            if charge != 0.0 and not any(
                np.isclose(value, old) for old in positions
            ):
                positions.append(value)
    if (
        float(options.get("heavy_trap_alpha", 0.0)) > 0.0
        and "heavy_trap_center" in options
    ):
        value = float(options["heavy_trap_center"])
        if np.isfinite(value) and not any(
            np.isclose(value, old) for old in positions
        ):
            positions.append(value)
    return positions


def _style_axis(axis):
    axis.grid(alpha=0.2, linewidth=0.7)
    axis.tick_params(direction="in")


def plot_particle_motion(obs, outdir, dpi, snapshot_count=5):
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    frames = selected_frames(len(times), min(snapshot_count, len(times)))
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.6), constrained_layout=True)
    q_ax, R_ax, mean_ax, width_ax = axes.flat
    shades = np.linspace(0.32, 1.0, len(frames))
    for alpha, frame in zip(shades, frames):
        label = f"{times[int(frame)]:.2f} fs"
        q_ax.plot(
            q, obs["proton_density"][int(frame)],
            color=PARTICLE_COLORS["proton"], alpha=alpha, lw=1.8, label=label,
        )
        R_ax.plot(
            R, obs["heavy_density"][int(frame)],
            color=PARTICLE_COLORS["heavy"], alpha=alpha, lw=1.8, label=label,
        )
    for ax, title, symbol in (
        (q_ax, "Proton marginal snapshots", "q"),
        (R_ax, "Heavy-nucleus marginal snapshots", "R"),
    ):
        add_fixed_center_markers(ax, obs["options"])
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_xlabel(rf"${symbol}$ ($a_0$)")
        ax.set_ylabel(r"probability density ($a_0^{-1}$)")
        ax.set_ylim(0.0, 1.05*obs["marginal_ymax"])
        ax.legend(frameon=False, fontsize=7, ncol=2)
        _style_axis(ax)
    mean_ax.plot(times, obs["q_mean"], color=PARTICLE_COLORS["proton"], label=r"$\langle q\rangle$")
    mean_ax.plot(times, obs["R_mean"], color=PARTICLE_COLORS["heavy"], label=r"$\langle R\rangle$")
    for position in _fixed_positions(obs["options"]):
        mean_ax.axhline(position, color="0.5", ls=":", lw=0.9)
    mean_ax.set_title("Mean positions", loc="left", fontweight="semibold")
    mean_ax.set(xlabel="time (fs)", ylabel=r"position ($a_0$)")
    mean_ax.legend(frameon=False)
    _style_axis(mean_ax)
    width_ax.plot(times, obs["q_width"], color=PARTICLE_COLORS["proton"], label=r"$\sigma_q$")
    width_ax.plot(times, obs["R_width"], color=PARTICLE_COLORS["heavy"], label=r"$\sigma_R$")
    width_ax.set_title("Packet widths", loc="left", fontweight="semibold")
    width_ax.set(xlabel="time (fs)", ylabel=r"standard deviation ($a_0$)")
    width_ax.legend(frameon=False)
    _style_axis(width_ax)
    fig.suptitle("Direct TDSE | nuclear marginal dynamics", fontweight="bold")
    path = Path(outdir)/"01_tdse_particle_motion.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE particle motion 저장: {path}")
    return path


def plot_joint_snapshots(obs, outdir, dpi, snapshot_count=6):
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    frames = selected_frames(len(times), min(snapshot_count, len(times)))
    columns = 3
    rows = int(np.ceil(len(frames)/columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(14.5, 4.0*rows), constrained_layout=True,
        squeeze=False,
    )
    image = None
    positions = _fixed_positions(obs["options"])
    for ax, frame in zip(axes.flat, frames):
        image = ax.imshow(
            obs["joint_density"][int(frame)].T,
            origin="lower", aspect="auto", interpolation="nearest",
            extent=[q[0], q[-1], R[0], R[-1]], cmap=JOINT_CMAP,
            vmin=0.0, vmax=obs["joint_vmax"],
        )
        for position in positions:
            ax.axvline(position, color="white", lw=0.75, ls=":", alpha=0.68)
            ax.axhline(position, color="white", lw=0.75, ls=":", alpha=0.68)
        ax.plot(obs["q_mean"][int(frame)], obs["R_mean"][int(frame)], "wo", ms=3.5, mec="0.15", mew=0.5)
        ax.set_title(f"t = {times[int(frame)]:.3f} fs", loc="left")
        ax.set_xlabel(r"proton $q$ ($a_0$)")
        ax.set_ylabel(r"heavy $R$ ($a_0$)")
    for ax in axes.flat[len(frames):]:
        ax.set_visible(False)
    if image is not None:
        fig.colorbar(
            image, ax=list(axes.flat[:len(frames)]), pad=0.01,
            format=NUMBER_FORMATTER,
            label=r"joint probability density ($a_0^{-2}$)",
        )
    fig.suptitle(
        "Direct TDSE | proton-heavy joint density | one exact color scale",
        fontweight="bold",
    )
    path = Path(outdir)/"02_tdse_joint_density.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE joint-density snapshots 저장: {path}")
    return path


def _relative_log_density(density, decades=6.0):
    density = np.asarray(density, dtype=float)
    axes = tuple(range(1, density.ndim))
    peak = np.maximum(np.max(density, axis=axes), 1.0e-300)
    reshape = (len(peak),)+(1,)*(density.ndim-1)
    relative = density/peak.reshape(reshape)
    return np.log10(np.maximum(relative, 10.0**(-float(decades)))), peak


def plot_joint_log_snapshots(obs, outdir, dpi, snapshot_count=6, decades=6.0):
    """Plot wavepacket shape without changing the stored joint density."""
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    log_density, peaks = _relative_log_density(obs["joint_density"], decades)
    frames = selected_frames(len(times), min(snapshot_count, len(times)))
    columns = 3
    rows = int(np.ceil(len(frames)/columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(14.5, 4.0*rows), constrained_layout=True,
        squeeze=False,
    )
    image = None
    for ax, frame in zip(axes.flat, frames):
        frame = int(frame)
        image = ax.imshow(
            log_density[frame].T,
            origin="lower", aspect="auto", interpolation="nearest",
            extent=[q[0], q[-1], R[0], R[-1]], cmap=JOINT_CMAP,
            vmin=-float(decades), vmax=0.0,
        )
        ax.set_title(
            f"t={times[frame]:.3f} fs | peak={peaks[frame]:.3e}", loc="left",
        )
        ax.set_xlabel(r"proton $q$ ($a_0$)")
        ax.set_ylabel(r"heavy $R$ ($a_0$)")
    for ax in axes.flat[len(frames):]:
        ax.set_visible(False)
    if image is not None:
        fig.colorbar(
            image, ax=list(axes.flat[:len(frames)]), pad=0.01,
            label=r"$\log_{10}[\rho_{qR}/\rho_{qR,\max}(t)]$",
        )
    fig.suptitle(
        "Direct TDSE | relative-log joint-density shape | raw density unchanged",
        fontweight="bold",
    )
    path = Path(outdir)/"08_tdse_joint_density_relative_log.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE relative-log joint density 저장: {path}")
    return path


def _sample_energies_on_mean_path(obs):
    energies = obs.get("bo_energies")
    if energies is None:
        return None
    energies = np.asarray(energies, float)
    if energies.ndim != 3 or energies.shape[1:] != (len(obs["q"]), len(obs["R"])):
        return None
    iq = np.abs(obs["q"][None, :]-obs["q_mean"][:, None]).argmin(axis=1)
    iR = np.abs(obs["R"][None, :]-obs["R_mean"][:, None]).argmin(axis=1)
    return energies[:, iq, iR].T


def _plot_state_resolved_surface(axis, obs, ef, frame, coordinate):
    energies = np.asarray(obs.get("bo_energies"), float)
    density = obs["joint_density"][frame]
    iq, iR = np.unravel_index(int(np.argmax(density)), density.shape)
    if coordinate == "q":
        grid = obs["q"]
        surfaces = energies[:, :, iR]
        packets = ef["bo_state_density_q"][frame]
        subtitle = rf"at $R_{{peak}}={obs['R'][iR]:.3f}$"
    else:
        grid = obs["R"]
        surfaces = energies[:, iq, :]
        packets = ef["bo_state_density_R"][frame]
        subtitle = rf"at $q_{{peak}}={obs['q'][iq]:.3f}$"
    packet_axis = axis.twinx()
    handles = []
    for state, (surface, packet) in enumerate(zip(surfaces, packets)):
        color = COLORS[state % len(COLORS)]
        line, = axis.plot(grid, surface, color=color, lw=1.45, label=rf"$E_{state}$")
        packet_line, = packet_axis.plot(
            grid, packet, color=color, lw=1.15, ls="--", alpha=0.72,
            label=rf"$\rho_{state}$",
        )
        handles.extend((line, packet_line))
    axis.set(xlabel=rf"{coordinate} ($a_0$)", ylabel="BO energy (Hartree)")
    packet_axis.set_ylabel(r"state-resolved marginal density ($a_0^{-1}$)", color="0.3")
    packet_axis.tick_params(axis="y", colors="0.3")
    axis.set_title(
        f"BO surfaces and raw state density\n{subtitle}",
        loc="left", fontweight="semibold", fontsize=9.5,
    )
    axis.grid(alpha=0.16)
    axis.legend(handles=handles, frameon=False, fontsize=6.5, ncol=3)


def plot_electronic_dynamics(obs, outdir, dpi, ef=None):
    times = obs["times_fs"]
    populations = obs["bo_populations"]
    states = np.arange(populations.shape[1])
    sampled_energies = _sample_energies_on_mean_path(obs)
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.6), constrained_layout=True)
    linear_ax, log_ax, energy_ax, gap_ax = axes.flat
    for state in states:
        color = COLORS[state % len(COLORS)]
        linear_ax.plot(times, populations[:, state], color=color, label=rf"$P_{state}$")
        log_ax.semilogy(times, np.maximum(populations[:, state], 1.0e-18), color=color)
    linear_ax.set_title("BO populations (linear)", loc="left", fontweight="semibold")
    linear_ax.set_ylim(0.0, max(1.0, 1.05*float(np.max(populations))))
    linear_ax.legend(frameon=False, fontsize=7, ncol=min(5, len(states)))
    log_ax.set_title("BO populations (log view)", loc="left", fontweight="semibold")
    log_ax.set_ylim(1.0e-14, max(1.0, 1.05*float(np.max(populations))))
    has_state_density = (
        ef is not None
        and "bo_state_density_q" in ef
        and "bo_state_density_R" in ef
        and "bo_energies" in obs
    )
    if has_state_density:
        _plot_state_resolved_surface(energy_ax, obs, ef, -1, "q")
        _plot_state_resolved_surface(gap_ax, obs, ef, -1, "R")
    elif sampled_energies is not None:
        for state in states:
            energy_ax.plot(times, sampled_energies[:, state], color=COLORS[state % len(COLORS)], label=rf"$E_{state}$")
        for state in range(len(states)-1):
            gap_ax.plot(
                times, sampled_energies[:, state+1]-sampled_energies[:, state],
                color=COLORS[state % len(COLORS)], label=rf"$E_{state+1}-E_{state}$",
            )
        energy_ax.set_title(
            r"BO energies at nearest $(\langle q\rangle,\langle R\rangle)$ grid point",
            loc="left", fontweight="semibold",
        )
        energy_ax.set_ylabel("energy (Hartree)")
        gap_ax.set_title("Adjacent BO gaps along mean path", loc="left", fontweight="semibold")
        gap_ax.set_ylabel("energy gap (Hartree)")
        energy_ax.legend(frameon=False, fontsize=7, ncol=min(5, len(states)))
        gap_ax.legend(frameon=False, fontsize=7, ncol=2)
    else:
        energy_ax.text(0.5, 0.5, "BO energies not stored", ha="center", va="center", transform=energy_ax.transAxes)
        gap_ax.text(0.5, 0.5, "BO gaps unavailable", ha="center", va="center", transform=gap_ax.transAxes)
    for ax in axes.flat:
        ax.set_xlabel("time (fs)")
        _style_axis(ax)
    if has_state_density:
        energy_ax.set_xlabel(r"q ($a_0$)")
        gap_ax.set_xlabel(r"R ($a_0$)")
    linear_ax.set_ylabel("population")
    log_ax.set_ylabel("population (log scale)")
    fig.suptitle("Direct TDSE | electronic-state dynamics", fontweight="bold")
    path = Path(outdir)/"03_tdse_electronic_dynamics.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE electronic dynamics 저장: {path}")
    return path


def _series(obs, key):
    return np.asarray(obs.get(key, np.zeros_like(obs["times_fs"])), float)


def plot_numerical_reliability(obs, outdir, dpi):
    times = obs["times_fs"]
    norm = np.asarray(obs["norm"], float)
    energy = np.asarray(obs["energy"], float)
    tiny = 1.0e-300
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.5), constrained_layout=True)
    axes[0, 0].plot(times, norm-1.0, color="black")
    axes[0, 0].axhline(0.0, color="0.6", lw=0.8)
    axes[0, 0].set_title("Signed norm drift", loc="left", fontweight="semibold")
    axes[0, 0].set_ylabel(r"$\|Y\|^2-1$")
    axes[0, 1].plot(times, energy-energy[0], color=COLORS[0])
    axes[0, 1].axhline(0.0, color="0.6", lw=0.8)
    axes[0, 1].set_title("Signed energy drift", loc="left", fontweight="semibold")
    axes[0, 1].set_ylabel(r"$E(t)-E(0)$ (Hartree)")
    axes[1, 0].semilogy(times, np.maximum(_series(obs, "energy_imaginary_defect"), tiny), label=r"$|\Im\langle H\rangle|$")
    axes[1, 0].semilogy(times, np.maximum(_series(obs, "norm_rate"), tiny), label=r"$|d\|Y\|^2/dt|$")
    axes[1, 0].set_title("Instantaneous Hermiticity checks", loc="left", fontweight="semibold")
    axes[1, 0].legend(frameon=False)
    boundary_series = [
        ("outer_probability_q", "q: outer five cells", PARTICLE_COLORS["proton"]),
        ("outer_probability_R", "R: outer five cells", PARTICLE_COLORS["heavy"]),
    ]
    if (
        float(obs["options"].get("right_charge", 1.0)) == 0.0
        and float(obs["options"].get("heavy_trap_alpha", 0.0)) > 0.0
    ):
        boundary_series.append((
            "fixed_center_crossing_q", r"q: outside $X_L..R_c$", COLORS[3]
        ))
    else:
        boundary_series.extend((
            ("fixed_center_crossing_q", "q: beyond fixed centers", COLORS[3]),
            ("fixed_center_crossing_R", "R: beyond fixed centers", COLORS[4]),
        ))
    for key, label, color in boundary_series:
        axes[1, 1].semilogy(times, np.maximum(_series(obs, key), tiny), color=color, label=label)
    axes[1, 1].set_title("Boundary and fixed-center probability", loc="left", fontweight="semibold")
    axes[1, 1].set_ylabel("probability")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("time (fs)")
        _style_axis(ax)
    fig.suptitle("Direct TDSE | numerical reliability", fontweight="bold")
    path = Path(outdir)/"04_tdse_numerical_reliability.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE numerical reliability 저장: {path}")
    return path


def _save_animation(animation, fig, outdir, stem, fps, dpi, fmt):
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = Path(outdir)/f"{stem}.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=3200), dpi=dpi)
    else:
        if fmt == "mp4":
            print(f"ffmpeg을 찾지 못해 {stem} 영상을 GIF로 저장합니다.")
        path = Path(outdir)/f"{stem}.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"TDSE dynamics 저장: {path}")
    return path


def make_joint_log_animation(
    obs, outdir, fps, max_frames, dpi, fmt, decades=6.0,
):
    """Animate frame-relative log joint density as a shape diagnostic."""
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    log_density, peaks = _relative_log_density(obs["joint_density"], decades)
    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    fig, axis = plt.subplots(figsize=(10.6, 7.2), constrained_layout=True)
    image = axis.imshow(
        log_density[first].T,
        origin="lower", aspect="auto", interpolation="nearest",
        extent=[q[0], q[-1], R[0], R[-1]], cmap=JOINT_CMAP,
        vmin=-float(decades), vmax=0.0,
    )
    for position in _fixed_positions(obs["options"]):
        axis.axvline(position, color="white", lw=0.75, ls=":", alpha=0.68)
        axis.axhline(position, color="white", lw=0.75, ls=":", alpha=0.68)
    axis.set(xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)")
    fig.colorbar(
        image, ax=axis, pad=0.01,
        label=r"$\log_{10}[\rho_{qR}/\rho_{qR,\max}(t)]$",
    )
    title = fig.suptitle("")

    def update(number):
        frame = int(frames[number])
        image.set_data(log_density[frame].T)
        title.set_text(
            f"Direct TDSE joint-density shape | t={times[frame]:.4f} fs | "
            f"raw peak={peaks[frame]:.3e} $a_0^{{-2}}$\n"
            f"relative log floor=$10^{{-{decades:g}}}$; no density values modified"
        )
        return image, title

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(
        animation, fig, outdir, "tdse_joint_density_relative_log",
        fps, dpi, fmt,
    )


def make_dynamics_animation(obs, outdir, fps, max_frames, dpi, fmt):
    """Animate raw TDSE marginals, joint density, populations and mean motion."""
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    fig = plt.figure(figsize=(15.8, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.78, 1.22))
    marginal_ax = fig.add_subplot(grid[0, :])
    joint_ax = fig.add_subplot(grid[1, 0])
    population_ax = fig.add_subplot(grid[1, 1])
    motion_ax = fig.add_subplot(grid[1, 2])

    marginal_ax.plot(q, obs["proton_density"][0], color=PARTICLE_COLORS["proton"], ls="--", lw=1.1, alpha=0.38)
    marginal_ax.plot(R, obs["heavy_density"][0], color=PARTICLE_COLORS["heavy"], ls="--", lw=1.1, alpha=0.38)
    proton_line, = marginal_ax.plot(q, obs["proton_density"][first], color=PARTICLE_COLORS["proton"], lw=2.1, label="proton")
    heavy_line, = marginal_ax.plot(R, obs["heavy_density"][first], color=PARTICLE_COLORS["heavy"], lw=2.1, label="heavy nucleus")
    add_fixed_center_markers(marginal_ax, obs["options"])
    marginal_ax.set(
        xlim=(min(float(q[0]), float(R[0])), max(float(q[-1]), float(R[-1]))),
        ylim=(0.0, 1.05*obs["marginal_ymax"]),
        xlabel=r"position coordinate ($a_0$)",
        ylabel=r"probability density ($a_0^{-1}$)",
    )
    marginal_ax.set_title("Nuclear marginals | solid=current, faint dashed=initial", loc="left", fontweight="semibold")
    marginal_ax.legend(frameon=False, ncol=3)
    _style_axis(marginal_ax)

    joint_image = joint_ax.imshow(
        obs["joint_density"][first].T,
        origin="lower", aspect="auto", interpolation="nearest",
        extent=[q[0], q[-1], R[0], R[-1]], cmap=JOINT_CMAP,
        vmin=0.0, vmax=obs["joint_vmax"],
    )
    for position in _fixed_positions(obs["options"]):
        joint_ax.axvline(position, color="white", lw=0.75, ls=":", alpha=0.68)
        joint_ax.axhline(position, color="white", lw=0.75, ls=":", alpha=0.68)
    joint_mean, = joint_ax.plot(obs["q_mean"][first], obs["R_mean"][first], "wo", ms=4, mec="0.15", mew=0.6)
    joint_ax.set(xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)")
    joint_ax.set_title("Proton-heavy joint density", loc="left", fontweight="semibold")
    fig.colorbar(joint_image, ax=joint_ax, pad=0.01, format=NUMBER_FORMATTER, label=r"density ($a_0^{-2}$)")

    populations = obs["bo_populations"]
    states = np.arange(populations.shape[1])
    for state in states:
        population_ax.plot(times, populations[:, state], color=COLORS[state % len(COLORS)], label=rf"$P_{state}$")
    population_time = population_ax.axvline(times[first], color="black", lw=1.2)
    population_ax.set(
        xlim=(times[0], times[-1] if times[-1] > times[0] else times[0]+1.0),
        ylim=(0.0, max(1.0, 1.05*float(np.max(populations)))),
        xlabel="time (fs)", ylabel="BO population",
    )
    population_ax.set_title("Electronic-state transfer", loc="left", fontweight="semibold")
    population_ax.legend(frameon=False, fontsize=7, ncol=min(3, len(states)))
    _style_axis(population_ax)

    motion_ax.plot(times, obs["q_mean"], color=PARTICLE_COLORS["proton"], label=r"$\langle q\rangle$")
    motion_ax.plot(times, obs["R_mean"], color=PARTICLE_COLORS["heavy"], label=r"$\langle R\rangle$")
    motion_ax.fill_between(times, obs["q_mean"]-obs["q_width"], obs["q_mean"]+obs["q_width"], color=PARTICLE_COLORS["proton"], alpha=0.13, linewidth=0)
    motion_ax.fill_between(times, obs["R_mean"]-obs["R_width"], obs["R_mean"]+obs["R_width"], color=PARTICLE_COLORS["heavy"], alpha=0.13, linewidth=0)
    motion_time = motion_ax.axvline(times[first], color="black", lw=1.2)
    for position in _fixed_positions(obs["options"]):
        motion_ax.axhline(position, color="0.5", ls=":", lw=0.8)
    motion_ax.set(
        xlim=(times[0], times[-1] if times[-1] > times[0] else times[0]+1.0),
        ylim=(min(float(q[0]), float(R[0])), max(float(q[-1]), float(R[-1]))),
        xlabel="time (fs)", ylabel=r"position ($a_0$)",
    )
    motion_ax.set_title(r"Mean position and $\pm1\sigma$", loc="left", fontweight="semibold")
    motion_ax.legend(frameon=False)
    _style_axis(motion_ax)

    energy = np.asarray(obs["energy"], float)
    title = fig.suptitle("")

    def update(number):
        frame = int(frames[number])
        proton_line.set_ydata(obs["proton_density"][frame])
        heavy_line.set_ydata(obs["heavy_density"][frame])
        joint_image.set_data(obs["joint_density"][frame].T)
        joint_mean.set_data([obs["q_mean"][frame]], [obs["R_mean"][frame]])
        population_time.set_xdata([times[frame], times[frame]])
        motion_time.set_xdata([times[frame], times[frame]])
        title.set_text(
            f"Direct discrete TDSE dynamics | t={times[frame]:.4f} fs | "
            f"norm-1={obs['norm'][frame]-1:+.2e} | "
            f"E-E(0)={energy[frame]-energy[0]:+.2e} Ha\n"
            "stored normalized densities/populations; fixed axes and one trajectory-wide color scale"
        )
        return proton_line, heavy_line, joint_image, joint_mean, population_time, motion_time, title

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(animation, fig, outdir, "tdse_dynamics_overview", fps, dpi, fmt)


def _load_ef_fields(obs):
    path = Path(obs["archive_path"]).parent/"tdse_exact_factorization_fields.npz"
    if not path.is_file():
        return None
    required = ("epsilon_1", "epsilon_2", "a", "b", "alpha")
    with np.load(path, allow_pickle=False) as stored:
        missing = [key for key in required if key not in stored.files]
        if missing:
            raise KeyError(
                f"{path.name}에 field가 없습니다: " + ", ".join(missing)
            )
        if not np.allclose(stored["times_fs"], obs["times_fs"], rtol=0.0, atol=1.0e-10):
            raise ValueError("TDSE field cache와 source archive의 저장 시각이 다릅니다")
        result = {key: np.asarray(stored[key], float) for key in required}
        for key in (
            "x",
            "electron_density",
            "factorization_residual",
            "epsilon_1_imaginary_defect",
            "epsilon_2_imaginary_defect",
            "bo_state_density_q",
            "bo_state_density_R",
        ):
            if key in stored.files:
                result[key] = np.asarray(stored[key], float)
        for key in (
            "sphi_q1", "sphi_q2", "sphi_R1", "sphi_R2",
            "sgamma_R1", "sgamma_R2",
        ):
            if key in stored.files:
                result[key] = np.asarray(stored[key], complex)
    result["path"] = path
    return result


def support_aware_temporal_lift_1d(connection, density, spacing, floor=1.0e-3):
    """Lift a principal link phase only on connected occupied support.

    ``connection`` is ``Arg(S)/spacing``.  A full-domain, frame-independent
    ``np.unwrap`` can inherit an arbitrary winding number from an empty tail
    and shift the occupied packet by ``2*pi/spacing``.  Here every connected
    occupied segment is spatially unwrapped and its integer branch is chosen
    to remain closest to the previous frame on their overlap.  Outside support
    the native principal value is retained for faint contextual plotting.
    """
    connection = np.asarray(connection, dtype=float)
    density = np.asarray(density, dtype=float)
    if connection.ndim != 2 or density.shape != connection.shape:
        raise ValueError("1D connection/density trajectory shape mismatch")
    if not np.isfinite(spacing) or spacing == 0.0:
        raise ValueError("connection spacing must be finite and nonzero")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("support floor must be finite and positive")

    principal_phase = connection*float(spacing)
    lifted_phase = principal_phase.copy()
    peak = np.maximum(np.max(density, axis=1), 1.0e-300)
    support = density >= floor*peak[:, None]
    previous = np.full(connection.shape[1], np.nan, dtype=float)
    previous_density = np.zeros(connection.shape[1], dtype=float)
    branch_turns = np.zeros(connection.shape[0], dtype=int)

    for frame in range(connection.shape[0]):
        active = support[frame]
        padded = np.pad(active.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops):
            segment = np.unwrap(principal_phase[frame, start:stop])
            overlap = np.isfinite(previous[start:stop])
            if np.any(overlap):
                weights = np.maximum(
                    density[frame, start:stop][overlap]
                    +previous_density[start:stop][overlap],
                    1.0e-300,
                )
                delta = previous[start:stop][overlap]-segment[overlap]
                turns = int(np.rint(np.sum(weights*delta)/np.sum(weights)/(2.0*np.pi)))
            else:
                local_peak = int(np.argmax(density[frame, start:stop]))
                turns = int(np.rint(
                    (principal_phase[frame, start+local_peak]-segment[local_peak])
                    /(2.0*np.pi)
                ))
            segment = segment+turns*(2.0*np.pi)
            lifted_phase[frame, start:stop] = segment
        global_peak = int(np.argmax(density[frame]))
        branch_turns[frame] = int(np.rint(
            (lifted_phase[frame, global_peak]-principal_phase[frame, global_peak])
            /(2.0*np.pi)
        ))
        previous = np.where(active, lifted_phase[frame], np.nan)
        previous_density = np.where(active, density[frame], 0.0)
    return lifted_phase/float(spacing), support, branch_turns


def continuity_current_1d(density, times_fs, spacing):
    """Reconstruct branch-free 1D current from the discrete continuity law.

    The current at cell centres is obtained from the cumulative flux through
    cell boundaries, with negligible left-boundary flux as the integration
    constant.  This is appropriate for the reported trajectories whose edge
    probability is explicitly diagnosed.  Its temporal accuracy is limited
    by the saved-frame spacing, but it cannot inherit a link-phase branch.
    """
    density = np.asarray(density, dtype=float)
    times_au = np.asarray(times_fs, dtype=float)*AU_PER_FS
    if density.ndim != 2 or density.shape[0] != len(times_au):
        raise ValueError("continuity density/time shape mismatch")
    if len(times_au) < 2:
        return np.zeros_like(density)
    edge_order = 2 if len(times_au) >= 3 else 1
    density_rate = np.gradient(
        density, times_au, axis=0, edge_order=edge_order,
    )
    edge_current = np.zeros(
        (density.shape[0], density.shape[1]+1), dtype=float,
    )
    edge_current[:, 1:] = -float(spacing)*np.cumsum(density_rate, axis=1)
    return 0.5*(edge_current[:, :-1]+edge_current[:, 1:])


def _prepared_ef_geometry(obs, ef, floor=1.0e-3):
    cached = ef.get("_prepared_geometry")
    if cached is not None:
        return cached
    alpha, heavy_support, branch_turns = support_aware_temporal_lift_1d(
        ef["alpha"], obs["heavy_density"], obs["dR"], floor,
    )
    times_au = np.asarray(obs["times_fs"], dtype=float)*AU_PER_FS
    edge_order = 2 if len(times_au) >= 3 else 1
    dalpha_dt = (
        np.gradient(alpha, times_au, axis=0, edge_order=edge_order)
        if len(times_au) >= 2 else np.zeros_like(alpha)
    )
    cached = {
        "alpha": alpha,
        "dalpha_dt": dalpha_dt,
        "heavy_support": heavy_support,
        "alpha_branch_turns": branch_turns,
        "heavy_continuity_current": continuity_current_1d(
            obs["heavy_density"], obs["times_fs"], obs["dR"],
        ),
        "proton_continuity_current": continuity_current_1d(
            obs["proton_density"], obs["times_fs"], obs["dq"],
        ),
    }
    ef["_prepared_geometry"] = cached
    return cached


def _connection(ef, key, frame, spacing, spatial_axis):
    del spacing, spatial_axis
    # ``a``, ``b`` and ``alpha`` are already native principal Arg(S)/h
    # values.  Do not spatially unwrap them through empty tails.
    return np.asarray(ef[key][frame], float)


def _connection_time_derivative(ef, key, frame, spacing, spatial_axis, times_au):
    count = len(times_au)
    if count < 2:
        return np.zeros_like(ef[key][frame], dtype=float)
    if frame == 0:
        indices = np.array([0, 1])
        local = 0
    elif frame == count-1:
        indices = np.array([count-2, count-1])
        local = 1
    else:
        indices = np.array([frame-1, frame, frame+1])
        local = 1
    phase = np.asarray(ef[key][indices], float)*spacing
    del spatial_axis
    # Track the same bond through adjacent saved times.  Spatial unwrapping
    # across empty tails is intentionally forbidden.
    phase = np.unwrap(phase, axis=0)
    edge_order = 2 if len(indices) >= 3 else 1
    return np.gradient(
        phase, times_au[indices], axis=0, edge_order=edge_order
    )[local]/spacing


def _ef_frame(obs, ef, frame, floor=1.0e-3):
    density = obs["joint_density"][frame]
    heavy = obs["heavy_density"][frame]
    density_cutoff = floor*max(float(np.max(density)), 1.0e-300)
    heavy_cutoff = floor*max(float(np.max(heavy)), 1.0e-300)
    support = density >= density_cutoff
    heavy_support = heavy >= heavy_cutoff
    eps1_full = density_weighted_shift(ef["epsilon_1"][frame], density, floor)
    eps2_full = density_weighted_shift(ef["epsilon_2"][frame], heavy, floor)
    a_full = _connection(ef, "a", frame, obs["dq"], 0)
    b_full = _connection(ef, "b", frame, obs["dR"], 1)
    prepared = _prepared_ef_geometry(obs, ef, floor)
    alpha_full = prepared["alpha"][frame]
    times_au = obs["times_fs"]*AU_PER_FS
    da_dt = _connection_time_derivative(
        ef, "a", frame, obs["dq"], 0, times_au
    )
    db_dt = _connection_time_derivative(
        ef, "b", frame, obs["dR"], 1, times_au
    )
    dalpha_dt = prepared["dalpha_dt"][frame]
    force_q_full = -derivative(eps1_full, obs["dq"], axis=0)+da_dt
    force_R_first_full = -derivative(eps1_full, obs["dR"], axis=1)+db_dt
    force_R_full = -derivative(eps2_full, obs["dR"], axis=0)+dalpha_dt
    options = obs["options"]
    proton_mass = float(options.get("proton_mass", 1836.15267343))
    heavy_mass = float(options.get("heavy_mass", 1836.15267343))
    momentum_q_full = a_full
    momentum_R_first_full = b_full
    momentum_R_full = alpha_full
    proton_current_full = density*momentum_q_full/proton_mass
    first_heavy_current_full = density*momentum_R_first_full/heavy_mass
    heavy_current_full = prepared["heavy_continuity_current"][frame]

    def map_support(values):
        return np.where(support, values, np.nan)

    def line_support(values):
        return np.where(heavy_support, values, np.nan)

    return {
        "density": density,
        "density_alpha": density_display_alpha(density, floor),
        "heavy": heavy,
        "support": support,
        "heavy_support": heavy_support,
        "eps1_full": eps1_full,
        "eps1": map_support(eps1_full),
        "a_full": a_full,
        "a": map_support(a_full),
        "b_full": b_full,
        "b": map_support(b_full),
        "eps2_full": eps2_full,
        "eps2": line_support(eps2_full),
        "alpha_full": alpha_full,
        "alpha": line_support(alpha_full),
        "momentum_q_full": momentum_q_full,
        "momentum_q": map_support(momentum_q_full),
        "momentum_R_first_full": momentum_R_first_full,
        "momentum_R_first": map_support(momentum_R_first_full),
        "momentum_R_full": momentum_R_full,
        "momentum_R": line_support(momentum_R_full),
        "proton_current_full": proton_current_full,
        "proton_current": map_support(proton_current_full),
        "first_heavy_current_full": first_heavy_current_full,
        "first_heavy_current": map_support(first_heavy_current_full),
        "heavy_current_full": heavy_current_full,
        "heavy_current": line_support(heavy_current_full),
        "heavy_current_kind": "continuity_reconstructed",
        "force_q_full": force_q_full,
        "force_q": map_support(force_q_full),
        "force_R_first_full": force_R_first_full,
        "force_R_first": map_support(force_R_first_full),
        "force_R_full": force_R_full,
        "force_R": line_support(force_R_full),
    }


def _ef_limits(items):
    scalar_keys = ("eps1", "eps2")
    symmetric_keys = (
        "connection", "momentum_R", "current_R", "force_R",
        "momentum_q", "current_q", "force_q",
    )
    values = {key: [] for key in scalar_keys+symmetric_keys}

    def finite(array):
        array = np.asarray(array)
        return array[np.isfinite(array)]

    for item in items:
        for key in scalar_keys:
            selected = finite(item[key])
            if selected.size:
                values[key].append((float(np.min(selected)), float(np.max(selected))))
        for target, keys in (
            ("connection", ("a", "b")),
            ("momentum_R", ("alpha", "momentum_R")),
            ("current_R", ("heavy_current",)),
            ("force_R", ("force_R",)),
            ("momentum_q", ("momentum_q",)),
            ("current_q", ("proton_current",)),
            ("force_q", ("force_q",)),
        ):
            for key in keys:
                selected = finite(item[key])
                if selected.size:
                    values[target].append(
                        float(np.percentile(np.abs(selected), 99.0))
                    )
    result = {}
    for key in scalar_keys:
        if not values[key]:
            result[key] = (-1.0, 1.0)
        else:
            low = min(item[0] for item in values[key])
            high = max(item[1] for item in values[key])
            if high <= low:
                padding = max(abs(low)*1.0e-6, 1.0e-12)
                low, high = low-padding, high+padding
            result[key] = (low, high)
    for key in symmetric_keys:
        bound = max(
            float(np.percentile(values[key], 98.0)) if values[key] else 0.0,
            1.0e-12,
        )
        result[key] = (-bound, bound)
    result["a"] = result["connection"]
    result["b"] = result["connection"]
    return result


def _trajectory_ef_limits(obs, ef, maximum_frames):
    cached = ef.get("plot_limits")
    if cached is not None:
        return cached
    frames = selected_frames(
        len(obs["times_fs"]), min(maximum_frames, len(obs["times_fs"]))
    )
    cached = _ef_limits(_ef_frame(obs, ef, int(index)) for index in frames)
    ef["plot_limits"] = cached
    return cached


def _support_tail_lines(axis, coordinate, occupied, full, support, *, color,
                        label, linewidth=1.8, linestyle="-"):
    tail = np.where(~np.asarray(support, bool), np.asarray(full), np.nan)
    tail_line, = axis.plot(
        coordinate, tail, color=color, lw=0.55, ls=":", alpha=0.22,
        zorder=0.5,
    )
    support_line, = axis.plot(
        coordinate, occupied, color=color, lw=linewidth, ls=linestyle,
        label=label, zorder=2.5,
    )
    return support_line, tail_line


def _scaled_heavy_density(axis, coordinate, density):
    line, = axis.plot(
        coordinate, density/max(float(np.max(density)), 1.0e-300),
        transform=axis.get_xaxis_transform(), color=HEAVY_DENSITY_COLOR,
        lw=1.0, alpha=0.42, label=r"$\rho_R$ (scaled)", zorder=0,
    )
    return line


def _draw_ef_maps(fig, axes, item, obs, limits):
    q, R = obs["q"], obs["R"]
    extent = [q[0], q[-1], R[0], R[-1]]
    images = []
    for ax, key, title, cmap in (
        (axes[0, 0], "eps1", r"First TDPES $\epsilon^{(1)}(q,R)$", SCALAR_CMAP),
        (axes[0, 1], "a", r"First vector potential: $a_q(q,R)$", SIGNED_CMAP),
        (axes[0, 2], "b", r"First vector potential: $b_R(q,R)$", SIGNED_CMAP),
    ):
        ax.set_facecolor(MASK_COLOR)
        image = ax.imshow(
            item[f"{key}_full"].T, origin="lower", aspect="auto",
            interpolation="nearest", extent=extent, cmap=masked_cmap(cmap),
            vmin=limits[key][0], vmax=limits[key][1],
            alpha=item["density_alpha"].T,
        )
        ax.set(xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)")
        ax.set_title(title, loc="left", fontweight="semibold")
        fig.colorbar(
            image, ax=ax, pad=0.01, format=NUMBER_FORMATTER, extend="both"
        )
        images.append((image, key))
    return images


def plot_exact_factorization_fields(obs, ef, outdir, dpi, frame=-1):
    limits = _trajectory_ef_limits(obs, ef, 180)
    item = _ef_frame(obs, ef, frame)
    q, R = obs["q"], obs["R"]
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.7), constrained_layout=True)
    _draw_ef_maps(fig, axes, item, obs, limits)
    eps_line, _ = _support_tail_lines(
        axes[1, 0], R, item["eps2"], item["eps2_full"], item["heavy_support"],
        color=COLORS[0], label=r"$\epsilon^{(2)}$", linewidth=2.0,
    )
    _scaled_heavy_density(axes[1, 0], R, item["heavy"])
    axes[1, 0].set_ylim(limits["eps2"])
    axes[1, 0].set_title(r"Second TDPES $\epsilon^{(2)}(R)$", loc="left", fontweight="semibold")
    color_y_axis(axes[1, 0], COLORS[0], "shifted energy (Hartree)")
    axes[1, 0].legend(handles=[eps_line], frameon=False)
    alpha_line, _ = _support_tail_lines(
        axes[1, 1], R, item["alpha"], item["alpha_full"], item["heavy_support"],
        color=COLORS[3], label=r"second vector potential $\alpha_R$",
    )
    momentum_line, _ = _support_tail_lines(
        axes[1, 1], R, item["momentum_R"], item["momentum_R_full"], item["heavy_support"],
        color="black", label=r"$K_R^{(\chi)}$", linewidth=2.0,
    )
    _scaled_heavy_density(axes[1, 1], R, item["heavy"])
    axes[1, 1].set_ylim(limits["momentum_R"])
    axes[1, 1].set_ylabel(r"momentum / connection ($a_0^{-1}$)")
    axes[1, 1].set_title("Second vector potential and heavy momentum", loc="left", fontweight="semibold")
    axes[1, 1].legend(handles=[alpha_line, momentum_line], frameon=False, fontsize=8)
    current_line, _ = _support_tail_lines(
        axes[1, 2], R, item["heavy_current"], item["heavy_current_full"], item["heavy_support"],
        color=CURRENT_COLOR, label=r"$j_R^{(\chi)}$ (continuity)", linewidth=2.0,
    )
    axes[1, 2].set_ylim(limits["current_R"])
    color_y_axis(axes[1, 2], CURRENT_COLOR, "continuity-reconstructed current")
    force_axis = axes[1, 2].twinx()
    force_line, _ = _support_tail_lines(
        force_axis, R, item["force_R"], item["force_R_full"], item["heavy_support"],
        color=FORCE_COLOR, label=r"$F_R^{GI}$", linewidth=1.8,
    )
    force_axis.set_ylim(limits["force_R"])
    color_y_axis(force_axis, FORCE_COLOR, r"drive (Hartree/$a_0$)")
    axes[1, 2].set_title("Heavy transport and gauge-invariant drive", loc="left", fontweight="semibold")
    axes[1, 2].legend(handles=[current_line, force_line], frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set(xlabel=r"heavy $R$ ($a_0$)", xlim=(R[0], R[-1]))
        _style_axis(ax)
    fig.suptitle(
        f"TDSE postprocessed nested exact factorization | t={obs['times_fs'][frame]:.4f} fs\n"
        "density gauge; scalar offsets only; solid=occupied, dotted=low density",
        fontweight="bold",
    )
    path = Path(outdir)/"05_tdse_exact_factorization_fields.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE reconstructed exact fields 저장: {path}")
    return path


def plot_transport_fields(obs, ef, outdir, dpi, frame=-1):
    limits = _trajectory_ef_limits(obs, ef, 180)
    item = _ef_frame(obs, ef, frame)
    q, R = obs["q"], obs["R"]
    extent = [q[0], q[-1], R[0], R[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.7), constrained_layout=True)
    specifications = (
        ("momentum_q", r"Mechanical proton momentum $K_q$", "momentum_q"),
        ("proton_current", r"Proton probability transport $j_q$", "current_q"),
        ("force_q", r"Gauge-invariant proton drive $F_q^{GI}$", "force_q"),
        ("momentum_R_first", r"First-level heavy momentum $K_R^{(1)}$", "momentum_R"),
        ("first_heavy_current", r"First-level heavy transport $j_R^{(1)}$", "current_R"),
        ("force_R_first", r"First-level heavy drive $F_R^{(1),GI}$", "force_R"),
    )
    for ax, (key, title, limit_key) in zip(axes.flat, specifications):
        ax.set_facecolor(MASK_COLOR)
        image = ax.imshow(
            item[f"{key}_full"].T, origin="lower", aspect="auto",
            interpolation="nearest", extent=extent,
            cmap=masked_cmap(SIGNED_CMAP),
            vmin=limits[limit_key][0], vmax=limits[limit_key][1],
            alpha=item["density_alpha"].T,
        )
        ax.set(xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)")
        ax.set_title(title, loc="left", fontweight="semibold", fontsize=9.5)
        fig.colorbar(
            image, ax=ax, pad=0.01, format=NUMBER_FORMATTER, extend="both"
        )
    fig.suptitle(
        f"TDSE postprocessed gauge-invariant transport | t={obs['times_fs'][frame]:.4f} fs\n"
        "raw reconstructed fields; density controls display opacity only",
        fontweight="bold",
    )
    path = Path(outdir)/"06_tdse_transport_and_drive.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE reconstructed transport 저장: {path}")
    return path


def plot_discrete_link_geometry(obs, ef, outdir, dpi, frame=-1):
    """Plot native complex overlap links without altering archived values."""
    required = ("sphi_q1", "sphi_R1", "sgamma_R1")
    if any(key not in ef for key in required):
        return None
    q, R = obs["q"], obs["R"]
    density = np.asarray(obs["joint_density"][frame], float)
    heavy = np.asarray(obs["heavy_density"][frame], float)
    opacity = density_display_alpha(density, 1.0e-3)
    extent = [q[0], q[-1], R[0], R[-1]]
    q_link = np.asarray(ef["sphi_q1"][frame], complex)
    R_link = np.asarray(ef["sphi_R1"][frame], complex)
    gamma_link = np.asarray(ef["sgamma_R1"][frame], complex)
    map_values = (
        (1.0-np.abs(q_link), r"$1-|S^{\Phi}_{q,+1}|$", LINK_CMAP),
        (np.angle(q_link), r"$\arg S^{\Phi}_{q,+1}$", SIGNED_CMAP),
        (1.0-np.abs(R_link), r"$1-|S^{\Phi}_{R,+1}|$", LINK_CMAP),
        (np.angle(R_link), r"$\arg S^{\Phi}_{R,+1}$", SIGNED_CMAP),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.6), constrained_layout=True)
    for ax, (values, title, cmap) in zip(axes.flat[:4], map_values):
        ax.set_facecolor(MASK_COLOR)
        if "arg" in title:
            limit = max(float(np.nanpercentile(np.abs(values), 99.5)), 1.0e-14)
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = 0.0, max(float(np.nanpercentile(values, 99.5)), 1.0e-14)
        image = ax.imshow(
            values.T, origin="lower", aspect="auto", interpolation="nearest",
            extent=extent, cmap=masked_cmap(cmap), vmin=vmin, vmax=vmax,
            alpha=opacity.T,
        )
        ax.set(xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)")
        ax.set_title(title, loc="left", fontweight="semibold")
        fig.colorbar(image, ax=ax, pad=0.01, format=NUMBER_FORMATTER)
    support = heavy >= 1.0e-3*max(float(np.max(heavy)), 1.0e-300)
    magnitude = 1.0-np.abs(gamma_link)
    phase = np.angle(gamma_link)
    _support_tail_lines(
        axes[1, 1], R, np.where(support, magnitude, np.nan), magnitude,
        support, color=COLORS[2], label=r"$1-|S^{\Gamma}_{R,+1}|$",
    )
    _support_tail_lines(
        axes[1, 2], R, np.where(support, phase, np.nan), phase,
        support, color=COLORS[3], label=r"$\arg S^{\Gamma}_{R,+1}$",
    )
    for ax in axes[1, 1:]:
        ax.set(xlabel=r"heavy $R$ ($a_0$)", xlim=(R[0], R[-1]))
        ax.legend(frameon=False, fontsize=8)
        _style_axis(ax)
    axes[1, 1].set_ylabel("link magnitude defect")
    axes[1, 2].set_ylabel("principal phase (rad)")
    fig.suptitle(
        f"TDSE-derived native discrete geometry | t={obs['times_fs'][frame]:.4f} fs\n"
        "raw complex links; color values unchanged, density controls opacity only",
        fontweight="bold",
    )
    path = Path(outdir)/"07_tdse_discrete_link_geometry.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE native discrete link geometry 저장: {path}")
    return path


def make_exact_field_animation(obs, ef, outdir, fps, max_frames, dpi, fmt):
    frames = selected_frames(len(obs["times_fs"]), min(max_frames, len(obs["times_fs"])))
    limits = _trajectory_ef_limits(obs, ef, max_frames)
    first = int(frames[0])
    first_item = _ef_frame(obs, ef, first)
    R = obs["R"]
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.7), constrained_layout=True)
    images = _draw_ef_maps(fig, axes, first_item, obs, limits)
    eps_line, eps_tail = _support_tail_lines(
        axes[1, 0], R, first_item["eps2"], first_item["eps2_full"], first_item["heavy_support"],
        color=COLORS[0], label=r"$\epsilon^{(2)}$", linewidth=2.0,
    )
    density_line = _scaled_heavy_density(axes[1, 0], R, first_item["heavy"])
    axes[1, 0].set_ylim(limits["eps2"])
    color_y_axis(axes[1, 0], COLORS[0], "shifted energy (Hartree)")
    axes[1, 0].set_title(r"Second TDPES $\epsilon^{(2)}(R)$", loc="left", fontweight="semibold")
    alpha_line, alpha_tail = _support_tail_lines(
        axes[1, 1], R, first_item["alpha"], first_item["alpha_full"], first_item["heavy_support"],
        color=COLORS[3], label=r"$\alpha_R$",
    )
    momentum_line, momentum_tail = _support_tail_lines(
        axes[1, 1], R, first_item["momentum_R"], first_item["momentum_R_full"], first_item["heavy_support"],
        color="black", label=r"$K_R^{(\chi)}$", linewidth=2.0,
    )
    axes[1, 1].set_ylim(limits["momentum_R"])
    axes[1, 1].set_title("Second vector potential and momentum", loc="left", fontweight="semibold")
    axes[1, 1].legend(frameon=False, fontsize=8)
    current_line, current_tail = _support_tail_lines(
        axes[1, 2], R, first_item["heavy_current"], first_item["heavy_current_full"], first_item["heavy_support"],
        color=CURRENT_COLOR, label=r"$j_R^{(\chi)}$ (continuity)", linewidth=2.0,
    )
    axes[1, 2].set_ylim(limits["current_R"])
    color_y_axis(axes[1, 2], CURRENT_COLOR, "continuity-reconstructed current")
    force_axis = axes[1, 2].twinx()
    force_line, force_tail = _support_tail_lines(
        force_axis, R, first_item["force_R"], first_item["force_R_full"], first_item["heavy_support"],
        color=FORCE_COLOR, label=r"$F_R^{GI}$", linewidth=1.8,
    )
    force_axis.set_ylim(limits["force_R"])
    color_y_axis(force_axis, FORCE_COLOR, r"drive (Hartree/$a_0$)")
    axes[1, 2].set_title("Heavy transport and drive", loc="left", fontweight="semibold")
    axes[1, 2].legend(handles=[current_line, force_line], frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set(xlabel=r"heavy $R$ ($a_0$)", xlim=(R[0], R[-1]))
        _style_axis(ax)
    title = fig.suptitle("")

    def update(number):
        frame = int(frames[number])
        item = _ef_frame(obs, ef, frame)
        for image, key in images:
            image.set_data(item[f"{key}_full"].T)
            image.set_alpha(item["density_alpha"].T)
        eps_line.set_ydata(item["eps2"])
        eps_tail.set_ydata(np.where(~item["heavy_support"], item["eps2_full"], np.nan))
        density_line.set_ydata(item["heavy"]/max(float(np.max(item["heavy"])), 1.0e-300))
        alpha_line.set_ydata(item["alpha"])
        alpha_tail.set_ydata(np.where(~item["heavy_support"], item["alpha_full"], np.nan))
        momentum_line.set_ydata(item["momentum_R"])
        momentum_tail.set_ydata(np.where(~item["heavy_support"], item["momentum_R_full"], np.nan))
        current_line.set_ydata(item["heavy_current"])
        current_tail.set_ydata(np.where(~item["heavy_support"], item["heavy_current_full"], np.nan))
        force_line.set_ydata(item["force_R"])
        force_tail.set_ydata(np.where(~item["heavy_support"], item["force_R_full"], np.nan))
        title.set_text(
            f"TDSE -> nested exact factorization | t={obs['times_fs'][frame]:.4f} fs\n"
            "two TDPES; first vector potential (a,b); second vector potential alpha; fixed scales"
        )
        return *(entry[0] for entry in images), eps_line, eps_tail, density_line, alpha_line, alpha_tail, momentum_line, momentum_tail, current_line, current_tail, force_line, force_tail, title

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(animation, fig, outdir, "tdse_exact_factorization_fields", fps, dpi, fmt)


def make_all_exact_potentials_animation(
    obs, ef, outdir, fps, max_frames, dpi, fmt,
):
    """Animate every native nested-EF potential/link in six outer panels.

    The complex first-level link has q and R components.  Its single outer
    panel therefore contains four compact, fixed-scale views: magnitude
    defect and principal phase for each direction.  No archived link is
    unwrapped, clipped, or otherwise changed for this display.
    """
    required = ("sphi_q1", "sphi_R1", "sgamma_R1")
    if any(key not in ef for key in required):
        print(
            "TDSE all-potential 영상 생략: nearest overlap links가 없습니다. "
            "postprocess_tdse_ef --link-output nearest 또는 full을 실행하세요."
        )
        return None

    frames = selected_frames(
        len(obs["times_fs"]), min(max_frames, len(obs["times_fs"]))
    )
    limits = _trajectory_ef_limits(obs, ef, max_frames)
    q, R = obs["q"], obs["R"]
    extent = [q[0], q[-1], R[0], R[-1]]

    def magnitude_defect(link):
        return 1.0-np.abs(np.asarray(link, complex))

    def robust_link_limit(key):
        maxima = []
        for frame in frames:
            values = magnitude_defect(ef[key][int(frame)])
            finite = np.abs(values[np.isfinite(values)])
            if finite.size:
                maxima.append(float(np.nanpercentile(finite, 99.5)))
        return max(max(maxima or [0.0]), 1.0e-14)

    q_defect_limit = robust_link_limit("sphi_q1")
    R_defect_limit = robust_link_limit("sphi_R1")
    gamma_defect_limit = robust_link_limit("sgamma_R1")
    first = int(frames[0])
    item = _ef_frame(obs, ef, first)

    fig, axes = plt.subplots(
        2, 3, figsize=(16.4, 9.2), constrained_layout=True,
    )
    map_images = _draw_ef_maps(fig, axes, item, obs, limits)

    # epsilon^(2) and alpha share R but not units, so use colored twin axes.
    eps_axis = axes[1, 0]
    alpha_axis = eps_axis.twinx()
    eps_line, = eps_axis.plot(
        R, item["eps2"], color=COLORS[0], lw=2.0,
        label=r"$\epsilon^{(2)}$",
    )
    eps_tail, = eps_axis.plot(
        R, np.where(~item["heavy_support"], item["eps2_full"], np.nan),
        color=COLORS[0], lw=0.55, ls=":", alpha=0.22,
    )
    alpha_line, = alpha_axis.plot(
        R, item["alpha"], color=COLORS[3], lw=1.8, ls="--",
        label=r"$\alpha_R$",
    )
    alpha_tail, = alpha_axis.plot(
        R, np.where(~item["heavy_support"], item["alpha_full"], np.nan),
        color=COLORS[3], lw=0.55, ls=":", alpha=0.22,
    )
    _scaled_heavy_density(eps_axis, R, item["heavy"])
    eps_axis.set(xlabel=r"heavy $R$ ($a_0$)", xlim=(R[0], R[-1]))
    eps_axis.set_ylim(limits["eps2"])
    alpha_axis.set_ylim(limits["momentum_R"])
    color_y_axis(eps_axis, COLORS[0], "shifted energy (Hartree)")
    color_y_axis(alpha_axis, COLORS[3], r"connection ($a_0^{-1}$)")
    eps_axis.set_title(
        r"Second level: $\epsilon^{(2)}(R)$ and $\alpha_R(R)$",
        loc="left", fontweight="semibold",
    )
    eps_axis.legend(
        handles=[eps_line, alpha_line], frameon=False, fontsize=8,
        loc="upper left",
    )
    _style_axis(eps_axis)

    # One S^Phi outer panel, with both coordinate directions and both pieces
    # of each complex link.  Insets avoid pretending that phase and magnitude
    # share one scalar color scale.
    sphi_axis = axes[1, 1]
    sphi_axis.set_axis_off()
    sphi_axis.set_title(
        r"First-level overlap $S^\Phi$ (native +1 links)",
        loc="left", fontweight="semibold", pad=7,
    )
    inset_specs = (
        ("sphi_q1", "defect", [0.02, 0.54, 0.46, 0.39],
         r"$1-|S^\Phi_{q,+1}|$", q_defect_limit),
        ("sphi_q1", "phase", [0.52, 0.54, 0.46, 0.39],
         r"$\arg S^\Phi_{q,+1}$", np.pi),
        ("sphi_R1", "defect", [0.02, 0.06, 0.46, 0.39],
         r"$1-|S^\Phi_{R,+1}|$", R_defect_limit),
        ("sphi_R1", "phase", [0.52, 0.06, 0.46, 0.39],
         r"$\arg S^\Phi_{R,+1}$", np.pi),
    )
    sphi_images = []
    density_alpha = density_display_alpha(obs["joint_density"][first], 1.0e-3)
    for key, component, bounds, title_text, scale in inset_specs:
        inset = sphi_axis.inset_axes(bounds)
        link = np.asarray(ef[key][first], complex)
        values = magnitude_defect(link) if component == "defect" else np.angle(link)
        if component == "defect":
            vmin, vmax, cmap = 0.0, scale, LINK_CMAP
        else:
            vmin, vmax, cmap = -np.pi, np.pi, SIGNED_CMAP
        image = inset.imshow(
            values.T, origin="lower", aspect="auto", interpolation="nearest",
            extent=extent, cmap=masked_cmap(cmap), vmin=vmin, vmax=vmax,
            alpha=density_alpha.T,
        )
        inset.set_title(title_text, fontsize=7.5, pad=2)
        inset.tick_params(labelsize=6, direction="in")
        inset.set_xticks([])
        inset.set_yticks([])
        sphi_images.append((image, key, component))

    # S^Gamma is one-dimensional: show its invariant magnitude defect and
    # principal phase on separate colored y axes.
    gamma_axis = axes[1, 2]
    gamma_phase_axis = gamma_axis.twinx()
    gamma_link = np.asarray(ef["sgamma_R1"][first], complex)
    gamma_support = item["heavy_support"]
    gamma_defect = magnitude_defect(gamma_link)
    gamma_phase = np.angle(gamma_link)
    gamma_defect_line, = gamma_axis.plot(
        R, np.where(gamma_support, gamma_defect, np.nan),
        color=COLORS[2], lw=2.0, label=r"$1-|S^\Gamma_{R,+1}|$",
    )
    gamma_defect_tail, = gamma_axis.plot(
        R, np.where(~gamma_support, gamma_defect, np.nan),
        color=COLORS[2], lw=0.55, ls=":", alpha=0.22,
    )
    gamma_phase_line, = gamma_phase_axis.plot(
        R, np.where(gamma_support, gamma_phase, np.nan),
        color=COLORS[3], lw=1.7, ls="--",
        label=r"$\arg S^\Gamma_{R,+1}$",
    )
    gamma_phase_tail, = gamma_phase_axis.plot(
        R, np.where(~gamma_support, gamma_phase, np.nan),
        color=COLORS[3], lw=0.55, ls=":", alpha=0.22,
    )
    gamma_axis.set(
        xlabel=r"heavy $R$ ($a_0$)", xlim=(R[0], R[-1]),
        ylim=(-0.05*gamma_defect_limit, gamma_defect_limit),
    )
    gamma_phase_axis.set_ylim(-np.pi, np.pi)
    color_y_axis(gamma_axis, COLORS[2], "magnitude defect")
    color_y_axis(gamma_phase_axis, COLORS[3], "principal phase (rad)")
    gamma_axis.set_title(
        r"Second-level overlap $S^\Gamma_{R,+1}$",
        loc="left", fontweight="semibold",
    )
    gamma_axis.legend(
        handles=[gamma_defect_line, gamma_phase_line], frameon=False,
        fontsize=8, loc="upper left",
    )
    _style_axis(gamma_axis)
    title = fig.suptitle("")

    def update(number):
        frame = int(frames[number])
        current = _ef_frame(obs, ef, frame)
        for image, key in map_images:
            image.set_data(current[f"{key}_full"].T)
            image.set_alpha(current["density_alpha"].T)
        eps_line.set_ydata(current["eps2"])
        eps_tail.set_ydata(np.where(
            ~current["heavy_support"], current["eps2_full"], np.nan
        ))
        alpha_line.set_ydata(current["alpha"])
        alpha_tail.set_ydata(np.where(
            ~current["heavy_support"], current["alpha_full"], np.nan
        ))
        opacity = density_display_alpha(obs["joint_density"][frame], 1.0e-3)
        for image, key, component in sphi_images:
            link = np.asarray(ef[key][frame], complex)
            values = magnitude_defect(link) if component == "defect" else np.angle(link)
            image.set_data(values.T)
            image.set_alpha(opacity.T)
        link = np.asarray(ef["sgamma_R1"][frame], complex)
        support = current["heavy_support"]
        defect = magnitude_defect(link)
        phase = np.angle(link)
        gamma_defect_line.set_ydata(np.where(support, defect, np.nan))
        gamma_defect_tail.set_ydata(np.where(~support, defect, np.nan))
        gamma_phase_line.set_ydata(np.where(support, phase, np.nan))
        gamma_phase_tail.set_ydata(np.where(~support, phase, np.nan))
        title.set_text(
            f"TDSE-derived complete nested exact potentials | "
            f"t={obs['times_fs'][frame]:.4f} fs\n"
            "density gauge; fixed trajectory-wide scales; "
            "solid=occupied, dotted=low density"
        )
        return (
            *(entry[0] for entry in map_images), eps_line, eps_tail,
            alpha_line, alpha_tail, *(entry[0] for entry in sphi_images),
            gamma_defect_line, gamma_defect_tail,
            gamma_phase_line, gamma_phase_tail, title,
        )

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(
        animation, fig, outdir, "tdse_all_exact_potentials", fps, dpi, fmt
    )


def make_transport_animation(obs, ef, outdir, fps, max_frames, dpi, fmt):
    frames = selected_frames(len(obs["times_fs"]), min(max_frames, len(obs["times_fs"])))
    limits = _trajectory_ef_limits(obs, ef, max_frames)
    first = int(frames[0])
    item = _ef_frame(obs, ef, first)
    q, R = obs["q"], obs["R"]
    extent = [q[0], q[-1], R[0], R[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)
    specifications = (
        ("momentum_q", r"Mechanical proton momentum $K_q$", "momentum_q"),
        ("proton_current", r"Probability transport $j_q$", "current_q"),
        ("force_q", r"Gauge-invariant drive $F_q^{GI}$", "force_q"),
    )
    images = []
    for ax, (key, label, limit_key) in zip(axes, specifications):
        ax.set_facecolor(MASK_COLOR)
        image = ax.imshow(
            item[f"{key}_full"].T, origin="lower", aspect="auto",
            interpolation="nearest", extent=extent,
            cmap=masked_cmap(SIGNED_CMAP),
            vmin=limits[limit_key][0], vmax=limits[limit_key][1],
            alpha=item["density_alpha"].T,
        )
        ax.set(xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)")
        ax.set_title(label, loc="left", fontweight="semibold")
        fig.colorbar(
            image, ax=ax, pad=0.01, format=NUMBER_FORMATTER, extend="both"
        )
        images.append((image, key))
    title = fig.suptitle("")

    def update(number):
        frame = int(frames[number])
        current = _ef_frame(obs, ef, frame)
        for image, key in images:
            image.set_data(current[f"{key}_full"].T)
            image.set_alpha(current["density_alpha"].T)
        title.set_text(
            f"TDSE postprocessed proton transport | t={obs['times_fs'][frame]:.4f} fs\n"
            "density changes opacity only; field values and trajectory-wide scales are unchanged"
        )
        return *(entry[0] for entry in images), title

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(animation, fig, outdir, "tdse_transport_and_drive", fps, dpi, fmt)


def make_coordinate_focus_animations(
    obs, ef, outdir, fps, max_frames, dpi, fmt,
    marginal_ymax=1.5, marginal_xmax=12.0,
):
    """Make q- and R-focused movies from the reconstructed EF fields.

    The heavy-coordinate panels are the same native one-dimensional fields
    used by the exact-field report.  The first-level q-R fields are reduced to
    q profiles using the physical joint density (or integrated directly for
    the probability current); the raw two-dimensional maps remain available
    in ``tdse_transport_and_drive``.
    """
    times = np.asarray(obs["times_fs"], float)
    q, R = np.asarray(obs["q"], float), np.asarray(obs["R"], float)
    joint = np.asarray(obs["joint_density"], float)
    proton = np.asarray(obs["proton_density"], float)
    heavy = np.asarray(obs["heavy_density"], float)
    epsilon_q = np.zeros_like(proton)
    momentum_q = np.zeros_like(proton)
    current_q = np.zeros_like(proton)
    epsilon_R = np.zeros_like(heavy)
    momentum_R = np.zeros_like(heavy)
    current_R = np.zeros_like(heavy)
    prepared_geometry = _prepared_ef_geometry(obs, ef)
    for frame in range(len(times)):
        item = _ef_frame(obs, ef, frame)
        denominator = np.maximum(proton[frame], 1.0e-300)
        epsilon_q[frame] = (
            np.sum(joint[frame]*item["eps1_full"], axis=1)
            *obs["dR"]/denominator
        )
        momentum_q[frame] = (
            np.sum(joint[frame]*item["momentum_q_full"], axis=1)
            *obs["dR"]/denominator
        )
        current_q[frame] = prepared_geometry["proton_continuity_current"][frame]
        epsilon_R[frame] = item["eps2_full"]
        momentum_R[frame] = item["momentum_R_full"]
        current_R[frame] = item["heavy_current_full"]

    make_coordinate_focus_animation(
        times_fs=times, coordinate=R, marginal=heavy,
        profiles=(
            (r"Second TDPES $\epsilon^{(2)}(R)$",
             "shifted energy (Hartree)", epsilon_R, COLORS[0], False),
            (r"Heavy mechanical momentum $K_R^{(\chi)}$",
             r"momentum ($a_0^{-1}$)", momentum_R, "black", True),
            (r"Heavy probability transport $j_R^{(\chi)}$ (continuity)",
             "continuity-reconstructed current", current_R, CURRENT_COLOR, True),
        ),
        options=obs["options"], outdir=outdir, fps=fps,
        max_frames=max_frames, dpi=dpi, fmt=fmt,
        particle_name="Heavy nucleus", coordinate_symbol="R",
        color=PARTICLE_COLORS["heavy"], stem="heavy_coordinate_dynamics",
        marginal_ymax=marginal_ymax, marginal_xmax=marginal_xmax,
    )
    make_coordinate_focus_animation(
        times_fs=times, coordinate=q, marginal=proton,
        profiles=(
            (r"Density-conditioned first TDPES $\bar\epsilon^{(1)}(q)$",
             "shifted energy (Hartree)", epsilon_q, COLORS[0], False),
            (r"Density-conditioned mechanical momentum $\bar K_q$",
             r"momentum ($a_0^{-1}$)", momentum_q, "black", True),
            (r"Integrated probability transport $J_q$ (continuity)",
             "continuity-reconstructed current", current_q, CURRENT_COLOR, True),
        ),
        options=obs["options"], outdir=outdir, fps=fps,
        max_frames=max_frames, dpi=dpi, fmt=fmt,
        particle_name="Proton", coordinate_symbol="q",
        color=PARTICLE_COLORS["proton"], stem="proton_coordinate_dynamics",
        marginal_ymax=marginal_ymax, marginal_xmax=marginal_xmax,
    )


def run(archive, outdir, *, dpi=180, no_animation=False, fps=12,
        max_frames=180, animation_dpi=110, fmt="mp4", snapshot_count=6,
        marginal_ymax=1.5, marginal_xmax=12.0):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_observables(archive)
    obs = calculate_observables(data)
    coefficient_note = "tdse_coefficients skipped"
    print(
        "TDSE reduced observables 준비 완료: "
        f"frames={len(obs['times_fs'])}, joint={obs['joint_density'].nbytes/1024**3:.2f} GiB; "
        f"{coefficient_note}"
    )
    ef = _load_ef_fields(obs)
    if (
        obs["electron_density"] is None
        and ef is not None
        and "electron_density" in ef
    ):
        obs["electron_density"] = ef["electron_density"]
        obs["x"] = ef.get("x", obs["x"])
    plot_particle_motion(obs, outdir, dpi, snapshot_count=snapshot_count)
    plot_joint_snapshots(obs, outdir, dpi, snapshot_count=snapshot_count)
    plot_joint_log_snapshots(obs, outdir, dpi, snapshot_count=snapshot_count)
    plot_electronic_dynamics(obs, outdir, dpi, ef=ef)
    plot_numerical_reliability(obs, outdir, dpi)
    if ef is not None:
        print(f"TDSE postprocessed exact-factorization fields 사용: {ef['path']}")
        plot_exact_factorization_fields(obs, ef, outdir, dpi)
        plot_transport_fields(obs, ef, outdir, dpi)
        plot_discrete_link_geometry(obs, ef, outdir, dpi)
    else:
        print(
            "TDSE exact-potential/connection 그림 생략: "
            "tdse_exact_factorization_fields.npz가 없습니다. "
            "postprocess_tdse_ef를 먼저 실행하세요."
        )
    if not no_animation:
        make_dynamics_animation(obs, outdir, fps, max_frames, animation_dpi, fmt)
        make_joint_log_animation(obs, outdir, fps, max_frames, animation_dpi, fmt)
        if obs["electron_density"] is not None and obs["x"] is not None:
            make_fixed_scale_marginal_animation(
                times_fs=obs["times_fs"],
                particle_series=(
                    ("electron", obs["x"], obs["electron_density"]),
                    ("proton", obs["q"], obs["proton_density"]),
                    ("heavy", obs["R"], obs["heavy_density"]),
                ),
                options=obs["options"], outdir=outdir, fps=fps,
                max_frames=max_frames, dpi=animation_dpi, fmt=fmt,
                y_max=marginal_ymax, x_abs_max=marginal_xmax,
                title_prefix="Direct TDSE dynamics",
            )
            make_relative_log_marginal_animation(
                times_fs=obs["times_fs"],
                particle_series=(
                    ("electron", obs["x"], obs["electron_density"]),
                    ("proton", obs["q"], obs["proton_density"]),
                    ("heavy", obs["R"], obs["heavy_density"]),
                ),
                options=obs["options"], outdir=outdir, fps=fps,
                max_frames=max_frames, dpi=animation_dpi, fmt=fmt,
                decades=6.0, x_abs_max=marginal_xmax,
                title_prefix="Direct TDSE dynamics",
            )
        else:
            print(
                "TDSE fixed-scale 3-particle marginal 영상 생략: "
                "electron_density가 없습니다. postprocess_tdse_ef를 실행하세요."
            )
        if ef is not None:
            make_exact_field_animation(
                obs, ef, outdir, fps, max_frames, animation_dpi, fmt
            )
            make_all_exact_potentials_animation(
                obs, ef, outdir, fps, max_frames, animation_dpi, fmt
            )
            make_transport_animation(
                obs, ef, outdir, fps, max_frames, animation_dpi, fmt
            )
            make_coordinate_focus_animations(
                obs, ef, outdir, fps, max_frames, animation_dpi, fmt,
                marginal_ymax, marginal_xmax,
            )
    report_payload = dict(
        times_fs=obs["times_fs"],
        q_mean=obs["q_mean"], R_mean=obs["R_mean"],
        q_width=obs["q_width"], R_width=obs["R_width"],
        norm=obs["norm"], energy=obs["energy"],
        bo_populations=obs["bo_populations"],
    )
    if obs["electron_density"] is not None:
        report_payload.update(
            x=obs["x"], electron_density=obs["electron_density"]
        )
    if ef is not None:
        geometry = _prepared_ef_geometry(obs, ef)
        report_payload.update(
            alpha_support_temporal_lift=geometry["alpha"],
            alpha_branch_turns=geometry["alpha_branch_turns"],
            heavy_continuity_current=geometry["heavy_continuity_current"],
            proton_continuity_current=geometry["proton_continuity_current"],
        )
        print(
            "TDSE branch/continuity audit: "
            f"alpha turns=[{np.min(geometry['alpha_branch_turns'])},"
            f"{np.max(geometry['alpha_branch_turns'])}], "
            f"max|j_R|={np.max(np.abs(geometry['heavy_continuity_current'])):.3e}, "
            f"max|J_q|={np.max(np.abs(geometry['proton_continuity_current'])):.3e}"
        )
    np.savez_compressed(
        outdir/"tdse_report_observables.npz", **report_payload
    )
    print(f"TDSE standalone report 완료: {outdir}")
    return obs
