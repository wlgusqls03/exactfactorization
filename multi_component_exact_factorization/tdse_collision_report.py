"""Collision diagnostics for the proton--heavy relative coordinate.

The routines in this module deliberately use only reduced probability
densities stored by the TDSE propagator.  They therefore do not manufacture a
Wigner function from a diagonal density.  The reported crossing rate is the
saved-frame derivative of the population on the ``q > R`` side; it is the
continuity-law crossing flux at ``s = q-R = 0`` up to the temporal resolution
of the saved frames.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from .core import AU_PER_FS
from .report_plot_style import COLORS, JOINT_CMAP
from .visualize import NUMBER_FORMATTER, selected_frames


def relative_observables(obs):
    """Reduce ``rho(q,R,t)`` to collision observables in ``s=q-R``.

    Cell probabilities are histogrammed rather than point-interpolated.  This
    preserves total probability to floating-point precision for unequal q and
    R grid spacings.
    """
    q = np.asarray(obs["q"], float)
    R = np.asarray(obs["R"], float)
    joint = np.asarray(obs["joint_density"], float)
    times_fs = np.asarray(obs["times_fs"], float)
    dq = float(obs.get("dq", q[1]-q[0]))
    dR = float(obs.get("dR", R[1]-R[0]))
    if joint.shape != (len(times_fs), len(q), len(R)):
        raise ValueError("joint density/grid/time shape mismatch")

    separation = q[:, None]-R[None, :]
    ds = min(abs(dq), abs(dR))
    lower = float(np.min(separation))-0.5*ds
    upper = float(np.max(separation))+0.5*ds
    bin_count = max(1, int(np.ceil((upper-lower)/ds)))
    edges = lower+ds*np.arange(bin_count+1, dtype=float)
    # Close the final bin against roundoff without changing the common spacing.
    if edges[-1] <= np.max(separation):
        edges = np.append(edges, edges[-1]+ds)
        bin_count += 1
    s = 0.5*(edges[:-1]+edges[1:])
    indices = np.floor((separation.ravel()-edges[0])/ds).astype(np.int64)
    indices = np.clip(indices, 0, bin_count-1)

    rho_s = np.empty((len(times_fs), bin_count), dtype=float)
    cell_weight = abs(dq*dR)
    for frame, density in enumerate(joint):
        probability = np.bincount(
            indices, weights=np.maximum(density, 0.0).ravel()*cell_weight,
            minlength=bin_count,
        )
        rho_s[frame] = probability/ds

    greater = separation > 0.0
    lesser = separation < 0.0
    equal = ~(greater | lesser)
    p_greater = np.sum(joint[:, greater], axis=1)*cell_weight
    p_lesser = np.sum(joint[:, lesser], axis=1)*cell_weight
    p_equal = np.sum(joint[:, equal], axis=1)*cell_weight
    times_au = times_fs*AU_PER_FS
    edge_order = 2 if len(times_au) >= 3 else 1
    crossing_rate = (
        np.gradient(p_greater, times_au, edge_order=edge_order)
        if len(times_au) >= 2 else np.zeros_like(p_greater)
    )

    q_mass = float(obs.get("options", {}).get("proton_mass", 1836.15267343))
    R_mass = float(obs.get("options", {}).get("heavy_mass", 1836.15267343))
    centre = (q_mass*q[:, None]+R_mass*R[None, :])/(q_mass+R_mass)
    probability = joint*cell_weight
    norm = np.maximum(np.sum(probability, axis=(1, 2)), 1.0e-300)
    s_mean = np.sum(probability*separation[None, :, :], axis=(1, 2))/norm
    X_mean = np.sum(probability*centre[None, :, :], axis=(1, 2))/norm
    s_width = np.sqrt(np.maximum(
        np.sum(
            probability*(separation[None, :, :]-s_mean[:, None, None])**2,
            axis=(1, 2),
        )/norm,
        0.0,
    ))

    return {
        "times_fs": times_fs,
        "s": s,
        "ds": np.array(ds),
        "relative_density": rho_s,
        "p_q_greater_R": p_greater,
        "p_q_less_R": p_lesser,
        "p_q_equal_R_cells": p_equal,
        "crossing_rate_au": crossing_rate,
        "s_mean": s_mean,
        "s_width": s_width,
        "X_mean": X_mean,
        "relative_norm": np.sum(rho_s, axis=1)*ds,
    }


def _relative_log(values, decades):
    peak = np.maximum(np.max(values, axis=1), 1.0e-300)
    relative = values/peak[:, None]
    return np.log10(np.maximum(relative, 10.0**(-float(decades))))


def _relative_log_frame(values, decades):
    values = np.asarray(values, float)
    peak = max(float(np.max(values)), 1.0e-300)
    return np.log10(np.maximum(values/peak, 10.0**(-float(decades))))


def plot_collision_snapshots(obs, collision, outdir, dpi=180,
                             snapshot_count=6, decades=6.0):
    """Joint-density snapshots with the physical contact line made explicit."""
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    frames = selected_frames(len(times), min(snapshot_count, len(times)))
    columns = 3
    rows = int(np.ceil(len(frames)/columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(14.8, 4.0*rows), constrained_layout=True,
        squeeze=False,
    )
    image = None
    contact_min = max(float(q[0]), float(R[0]))
    contact_max = min(float(q[-1]), float(R[-1]))
    for axis, frame in zip(axes.flat, frames):
        frame = int(frame)
        image = axis.imshow(
            _relative_log_frame(obs["joint_density"][frame], decades).T,
            origin="lower", aspect="auto",
            interpolation="nearest", extent=[q[0], q[-1], R[0], R[-1]],
            cmap=JOINT_CMAP, vmin=-float(decades), vmax=0.0,
        )
        if contact_min <= contact_max:
            axis.plot(
                [contact_min, contact_max], [contact_min, contact_max],
                color="white", lw=1.15, ls="--", label=r"contact $q=R$",
            )
        axis.set(
            xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)",
            title=f"t = {times[frame]:.3f} fs",
        )
        legend = axis.legend(loc="upper left", frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color("white")
    for axis in axes.flat[len(frames):]:
        axis.set_visible(False)
    if image is not None:
        fig.colorbar(
            image, ax=list(axes.flat[:len(frames)]), pad=0.01,
            label=rf"$\log_{{10}}[\rho_{{qR}}/\rho_{{qR,\max}}(t)]$",
        )
    fig.suptitle(
        "Direct TDSE | proton--heavy collision geometry\n"
        r"$q<R$: proton left of heavy; $q>R$: proton right of heavy",
        fontweight="bold",
    )
    path = Path(outdir)/"09_tdse_collision_snapshots.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE collision snapshots 저장: {path}")
    return path


def plot_relative_diagnostics(collision, outdir, dpi=180, decades=6.0):
    times = collision["times_fs"]
    s = collision["s"]
    log_rho = _relative_log(collision["relative_density"], decades)
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), constrained_layout=True)

    image = axes[0, 0].imshow(
        log_rho.T, origin="lower", aspect="auto", interpolation="nearest",
        extent=[times[0], times[-1], s[0], s[-1]], cmap=JOINT_CMAP,
        vmin=-float(decades), vmax=0.0,
    )
    axes[0, 0].axhline(0.0, color="white", lw=1.0, ls="--")
    axes[0, 0].set(
        title=r"Relative density $\rho_s(s,t)$",
        xlabel="time (fs)", ylabel=r"separation $s=q-R$ ($a_0$)",
    )
    fig.colorbar(
        image, ax=axes[0, 0], pad=0.01,
        label=rf"$\log_{{10}}[\rho_s/\rho_{{s,\max}}(t)]$",
    )

    axes[0, 1].plot(
        times, collision["p_q_less_R"], color=COLORS[0],
        label=r"$P_<:q<R$",
    )
    axes[0, 1].plot(
        times, collision["p_q_greater_R"], color=COLORS[1],
        label=r"$P_>:q>R$",
    )
    axes[0, 1].set(
        title="Side populations (not by itself a tunnelling probability)",
        xlabel="time (fs)", ylabel="probability", ylim=(-0.02, 1.02),
    )
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        times, collision["crossing_rate_au"], color=COLORS[3], lw=1.6,
    )
    axes[1, 0].axhline(0.0, color="0.5", lw=0.8)
    axes[1, 0].set(
        title=r"Saved-frame crossing rate $dP_>/dt=J_s(0,t)$",
        xlabel="time (fs)", ylabel="probability / atomic time",
    )

    axes[1, 1].plot(
        times, collision["s_mean"], color=COLORS[1], label=r"$\langle s\rangle$",
    )
    axes[1, 1].fill_between(
        times,
        collision["s_mean"]-collision["s_width"],
        collision["s_mean"]+collision["s_width"],
        color=COLORS[1], alpha=0.18, label=r"$\langle s\rangle\pm\sigma_s$",
    )
    axes[1, 1].plot(
        times, collision["X_mean"], color=COLORS[2], label=r"$\langle X\rangle$",
    )
    axes[1, 1].axhline(0.0, color="0.5", lw=0.8, ls=":")
    axes[1, 1].set(
        title="Relative and centre-of-mass motion", xlabel="time (fs)",
        ylabel=r"coordinate ($a_0$)",
    )
    axes[1, 1].legend(frameon=False)

    fig.suptitle(
        "Direct TDSE | proton--heavy crossing diagnostics\n"
        "No temporal smoothing; crossing rate is limited by saved-frame spacing",
        fontweight="bold",
    )
    path = Path(outdir)/"10_tdse_relative_collision_diagnostics.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"TDSE relative collision diagnostics 저장: {path}")
    return path


def _save_animation(animation, fig, path, fps, dpi):
    path = Path(path)
    if path.suffix.lower() == ".gif":
        animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        animation.save(
            path, writer=FFMpegWriter(fps=fps, bitrate=3200), dpi=dpi,
        )
    plt.close(fig)
    return path


def make_collision_animation(obs, collision, outdir, fps=12, max_frames=180,
                             dpi=110, fmt="mp4", decades=6.0):
    frames = selected_frames(len(obs["times_fs"]), max_frames)
    times, q, R = obs["times_fs"], obs["q"], obs["R"]
    s = collision["s"]
    # Keep only rendered frames in the log-display cache.  A production joint
    # trajectory can be several GiB and need not be duplicated in RAM.
    log_joint = np.asarray([
        _relative_log_frame(obs["joint_density"][int(frame)], decades)
        for frame in frames
    ])
    log_rho_s = _relative_log(collision["relative_density"], decades)

    fig = plt.figure(figsize=(14.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    joint_axis = fig.add_subplot(grid[:, 0])
    relative_axis = fig.add_subplot(grid[0, 1])
    history_axis = fig.add_subplot(grid[1, 1])
    joint_image = joint_axis.imshow(
        log_joint[0].T, origin="lower", aspect="auto",
        interpolation="nearest", extent=[q[0], q[-1], R[0], R[-1]],
        cmap=JOINT_CMAP, vmin=-float(decades), vmax=0.0,
    )
    contact_min = max(float(q[0]), float(R[0]))
    contact_max = min(float(q[-1]), float(R[-1]))
    if contact_min <= contact_max:
        joint_axis.plot(
            [contact_min, contact_max], [contact_min, contact_max],
            color="white", lw=1.2, ls="--",
        )
    joint_axis.set(
        xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)",
        title=r"Joint density; dashed line: $q=R$",
    )
    fig.colorbar(
        joint_image, ax=joint_axis, pad=0.01,
        label=rf"$\log_{{10}}(\rho_{{qR}}/\rho_{{qR,\max}})$",
    )

    relative_line, = relative_axis.plot(
        s, log_rho_s[int(frames[0])], color=COLORS[1], lw=1.8,
    )
    relative_axis.axvline(0.0, color="0.4", lw=1.0, ls="--")
    relative_axis.set(
        xlim=(s[0], s[-1]), ylim=(-float(decades), 0.05),
        xlabel=r"$s=q-R$ ($a_0$)",
        ylabel=rf"$\log_{{10}}(\rho_s/\rho_{{s,\max}})$",
        title=r"Relative density; $s=0$ is contact",
    )

    p_less_line, = history_axis.plot(
        [], [], color=COLORS[0], label=r"$P_<$",
    )
    p_greater_line, = history_axis.plot(
        [], [], color=COLORS[1], label=r"$P_>$",
    )
    history_axis.set(
        xlim=(times[0], times[-1]), ylim=(-0.02, 1.02),
        xlabel="time (fs)", ylabel="side probability",
        title="Crossing history",
    )
    history_axis.legend(frameon=False, loc="upper left")
    rate_axis = history_axis.twinx()
    rate_line, = rate_axis.plot(
        [], [], color=COLORS[3], alpha=0.78,
        label=r"$dP_>/dt$",
    )
    rate_limit = max(
        float(np.nanpercentile(np.abs(collision["crossing_rate_au"]), 99.5)),
        1.0e-16,
    )
    rate_axis.set_ylim(-1.08*rate_limit, 1.08*rate_limit)
    rate_axis.set_ylabel(r"$dP_>/dt$ (au$^{-1}$)", color=COLORS[3])
    rate_axis.tick_params(axis="y", colors=COLORS[3])
    title = fig.suptitle("", fontweight="bold")

    def update(number):
        frame = int(frames[number])
        joint_image.set_data(log_joint[number].T)
        relative_line.set_ydata(log_rho_s[frame])
        stop = frame+1
        p_less_line.set_data(times[:stop], collision["p_q_less_R"][:stop])
        p_greater_line.set_data(times[:stop], collision["p_q_greater_R"][:stop])
        rate_line.set_data(times[:stop], collision["crossing_rate_au"][:stop])
        title.set_text(
            f"Proton--heavy collision | t={times[frame]:.3f} fs | "
            f"P(q>R)={collision['p_q_greater_R'][frame]:.6f}"
        )
        return joint_image, relative_line, p_less_line, p_greater_line, rate_line, title

    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=1000/max(float(fps), 1.0),
        blit=False,
    )
    path = Path(outdir)/f"tdse_collision_dynamics.{fmt}"
    _save_animation(animation, fig, path, fps, dpi)
    print(f"TDSE collision dynamics 영상 저장: {path}")
    return path


def run(obs, outdir, *, dpi=180, no_animation=False, fps=12,
        max_frames=180, animation_dpi=110, fmt="mp4", snapshot_count=6,
        decades=6.0):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    collision = relative_observables(obs)
    plot_collision_snapshots(
        obs, collision, outdir, dpi=dpi, snapshot_count=snapshot_count,
        decades=decades,
    )
    plot_relative_diagnostics(collision, outdir, dpi=dpi, decades=decades)
    if not no_animation:
        make_collision_animation(
            obs, collision, outdir, fps=fps, max_frames=max_frames,
            dpi=animation_dpi, fmt=fmt, decades=decades,
        )
    np.savez_compressed(outdir/"tdse_collision_observables.npz", **collision)
    norm_error = float(np.max(np.abs(collision["relative_norm"]-1.0)))
    print(
        "TDSE collision audit: "
        f"max|integral rho_s-1|={norm_error:.3e}; "
        f"P(q>R) max/final={np.max(collision['p_q_greater_R']):.6e}/"
        f"{collision['p_q_greater_R'][-1]:.6e}"
    )
    return collision
