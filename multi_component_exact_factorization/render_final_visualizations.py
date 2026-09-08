#!/usr/bin/env python3
"""Render the compact final TDSE/MCEF analysis gallery from saved arrays.

This command deliberately reuses the reduced TDSE observables and the
postprocessed exact-factorization cache.  It never loads ``tdse_coefficients``
and does not repeat propagation or electronic factorization.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np

from . import tdse_collision_report, tdse_report
from .render_all import find_archive, resolve_run_input
from .report_plot_style import (
    COLORS,
    FORCE_COLOR,
    JOINT_CMAP,
    MASK_COLOR,
    PARTICLE_COLORS,
    SIGNED_CMAP,
    add_fixed_center_markers,
    color_y_axis,
    density_display_alpha,
    density_weighted_shift,
    masked_cmap,
)
from .visualize import NUMBER_FORMATTER, selected_frames


FINAL_PRODUCTS = (
    "marginal", "joint", "velocity", "vector", "current", "nested",
    "heavy", "bo", "bo3d", "tdpes1", "tdpes2", "geometry",
)


def _snapshot_frames(obs, count=8):
    """Use the established endpoint-inclusive uniform frame selector."""
    return selected_frames(len(obs["times_fs"]), min(int(count), len(obs["times_fs"])))


def _movie_frames(obs, maximum):
    return selected_frames(
        len(obs["times_fs"]), min(int(maximum), len(obs["times_fs"]))
    )


def _time_tag(time_fs):
    return f"{float(time_fs):09.4f}fs".replace(".", "p")


def _math_scientific(value, digits=2):
    """Scientific notation fragment for insertion inside a mathtext string."""
    mantissa, exponent = f"{float(value):.{int(digits)}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def _save_figure(fig, path, dpi):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"final visualization 저장: {path}")
    return path


def _save_individual_frames(builder, frames, times, directory, stem, dpi):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for order, frame in enumerate(frames, 1):
        frame = int(frame)
        fig = builder(frame)
        path = directory/f"{order:02d}_{stem}_{_time_tag(times[frame])}.png"
        paths.append(_save_figure(fig, path, dpi))
    return paths


def _hardlink_output_alias(source, target):
    """Give one rendered product a second scientifically equivalent name."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        # Output directories can exceptionally cross filesystem boundaries.
        # Stream the already encoded file instead of rerendering every frame.
        with source.open("rb") as reader, target.open("wb") as writer:
            while True:
                block = reader.read(8*1024*1024)
                if not block:
                    break
                writer.write(block)
    print(f"final visualization alias 저장: {target}")
    return target


def _support_limits(coordinate, density, floor=1.0e-4, padding=0.06,
                    requested=None):
    """Trajectory-wide occupied coordinate window with display-only padding."""
    coordinate = np.asarray(coordinate, float)
    density = np.asarray(density, float)
    peak = np.maximum(np.max(density, axis=1), 1.0e-300)
    active = np.any(density >= float(floor)*peak[:, None], axis=0)
    if np.any(active):
        indices = np.flatnonzero(active)
        lower = float(coordinate[indices[0]])
        upper = float(coordinate[indices[-1]])
        span = max(upper-lower, abs(float(coordinate[1]-coordinate[0])))
        lower -= padding*span
        upper += padding*span
    else:
        lower, upper = float(coordinate[0]), float(coordinate[-1])
    lower = max(lower, float(coordinate[0]))
    upper = min(upper, float(coordinate[-1]))
    if requested is not None:
        lower = max(lower, float(requested[0]))
        upper = min(upper, float(requested[1]))
    if lower >= upper:
        lower, upper = float(coordinate[0]), float(coordinate[-1])
        if requested is not None:
            lower = max(lower, float(requested[0]))
            upper = min(upper, float(requested[1]))
    return lower, upper


def _symmetric_support_bound(arrays, densities, floor=1.0e-4,
                             percentile=99.0):
    per_frame = []
    for values, density in zip(arrays, densities):
        values = np.asarray(values, float)
        density = np.asarray(density, float)
        support = density >= float(floor)*max(float(np.max(density)), 1.0e-300)
        selected = np.abs(values[support & np.isfinite(values)])
        if selected.size:
            per_frame.append(float(np.percentile(selected, percentile)))
    bound = max(
        float(np.percentile(per_frame, 98.0)) if per_frame else 0.0,
        1.0e-12,
    )
    return -bound, bound


def _set_density_axis(axis):
    axis.set_facecolor("black")
    axis.tick_params(direction="in")


# ---------------------------------------------------------------------------
# 1. Marginal time-position maps


def _marginal_map_data(obs, decades):
    electron = obs.get("electron_density")
    x = obs.get("x")
    if electron is None or x is None:
        raise KeyError(
            "electron marginal이 없습니다. postprocess_tdse_ef를 먼저 실행하세요."
        )
    return (
        ("Electron", np.asarray(x), np.asarray(electron)),
        ("Proton", np.asarray(obs["q"]), np.asarray(obs["proton_density"])),
        ("Heavy nucleus", np.asarray(obs["R"]), np.asarray(obs["heavy_density"])),
    ), float(decades)


def _draw_marginal_time_maps(fig, axes, obs, prepared, frame, *, colorbar=True,
                             compact=False):
    series, decades = prepared
    times = obs["times_fs"]
    images, cursors = [], []
    for axis, (name, coordinate, density) in zip(axes, series):
        log_density = tdse_collision_report._relative_log(density, decades)
        image = axis.imshow(
            log_density.T, origin="lower", aspect="auto",
            interpolation="nearest",
            extent=[times[0], times[-1], coordinate[0], coordinate[-1]],
            cmap=JOINT_CMAP, vmin=-decades, vmax=0.0,
        )
        cursor = axis.axvline(times[frame], color="white", lw=1.05, alpha=0.92)
        _set_density_axis(axis)
        axis.set_ylabel(f"{name}\nposition ($a_0$)", fontsize=(7 if compact else None))
        if compact:
            axis.tick_params(labelsize=6)
        images.append(image)
        cursors.append(cursor)
    axes[-1].set_xlabel("dynamics time (fs)")
    if colorbar:
        fig.colorbar(
            images[0], ax=list(axes), pad=0.012,
            label=rf"$\log_{{10}}[\rho/\rho_{{\max}}(t)]$",
        )
    return {"images": images, "cursors": cursors}


def render_marginal_time_position(obs, outdir, args, snapshots):
    prepared = _marginal_map_data(obs, args.decades)
    times = obs["times_fs"]

    def individual(frame):
        fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.0), constrained_layout=True)
        _draw_marginal_time_maps(fig, axes, obs, prepared, frame)
        fig.suptitle(
            f"TDSE particle marginal histories | cursor t={times[frame]:.4f} fs\n"
            "frame-relative log display; archived densities unchanged",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times,
        Path(outdir)/"marginal_time_position_frames",
        "marginal_time_position", args.dpi,
    )
    # Each outer cell contains three very wide time-history panels.  A shallow
    # 2x4 canvas keeps the two rows contiguous instead of leaving unused
    # vertical space around the nested axes.
    fig = plt.figure(figsize=(22.0, 7.2), constrained_layout=False)
    outer = fig.add_gridspec(
        2, 4, left=0.035, right=0.945, bottom=0.075, top=0.91,
        wspace=0.22, hspace=0.22,
    )
    for slot, frame in zip(outer, snapshots):
        inner = slot.subgridspec(3, 1, hspace=0.03)
        axes = [fig.add_subplot(inner[row, 0]) for row in range(3)]
        _draw_marginal_time_maps(
            fig, axes, obs, prepared, int(frame), colorbar=False, compact=True,
        )
        axes[0].set_title(f"t = {times[int(frame)]:.3f} fs", color="0.15", fontsize=9)
    scalar = ScalarMappable(norm=Normalize(-args.decades, 0.0), cmap=JOINT_CMAP)
    colorbar_axis = fig.add_axes([0.957, 0.16, 0.010, 0.68])
    fig.colorbar(
        scalar, cax=colorbar_axis,
        label=rf"$\log_{{10}}[\rho/\rho_{{\max}}(t)]$",
    )
    fig.suptitle(
        "Electron / proton / heavy marginal time-position maps",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/"marginal_time_position_snapshots.png", args.dpi,
    ))

    if not args.no_animation:
        fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.0), constrained_layout=True)
        state = _draw_marginal_time_maps(fig, axes, obs, prepared, 0)
        title = fig.suptitle("", fontweight="bold")
        frames = _movie_frames(obs, args.max_frames)

        def update(number):
            frame = int(frames[number])
            for cursor in state["cursors"]:
                cursor.set_xdata([times[frame], times[frame]])
            title.set_text(
                f"TDSE particle marginal histories | t={times[frame]:.4f} fs\n"
                "frame-relative log display; white cursor is current time"
            )
            return *state["cursors"], title

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(tdse_report._save_animation(
            animation, fig, outdir, "marginal_time_position_movie",
            args.fps, args.animation_dpi, args.format,
        ))
    return products


# ---------------------------------------------------------------------------
# 2. Proton-heavy joint density


def _draw_joint_density(axis, obs, frame, decades, *, compact=False):
    q, R = obs["q"], obs["R"]
    values = tdse_collision_report._relative_log_frame(
        obs["joint_density"][frame], decades,
    )
    image = axis.imshow(
        values.T, origin="lower", aspect="auto", interpolation="nearest",
        extent=[q[0], q[-1], R[0], R[-1]], cmap=JOINT_CMAP,
        vmin=-float(decades), vmax=0.0,
    )
    contact_min = max(float(q[0]), float(R[0]))
    contact_max = min(float(q[-1]), float(R[-1]))
    if contact_min <= contact_max:
        axis.plot(
            [contact_min, contact_max], [contact_min, contact_max],
            color="white", lw=(0.75 if compact else 1.15), ls="--",
        )
    axis.set_xlim(tdse_collision_report._q_display_limits(q))
    axis.set(xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)")
    _set_density_axis(axis)
    if compact:
        axis.tick_params(labelsize=7)
    return image


def render_joint_density(obs, outdir, args, snapshots):
    times = obs["times_fs"]

    def individual(frame):
        fig, axis = plt.subplots(figsize=(9.4, 7.0), constrained_layout=True)
        image = _draw_joint_density(axis, obs, frame, args.decades)
        fig.colorbar(
            image, ax=axis, pad=0.012,
            label=rf"$\log_{{10}}[\rho_{{qR}}/\rho_{{qR,\max}}(t)]$",
        )
        fig.suptitle(
            f"Proton-heavy joint density | t={times[frame]:.4f} fs",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times, Path(outdir)/"joint_density_qR_frames",
        "joint_density_qR", args.dpi,
    )
    fig, axes = plt.subplots(2, 4, figsize=(20.5, 9.3), constrained_layout=True)
    image = None
    for axis, frame in zip(axes.flat, snapshots):
        image = _draw_joint_density(axis, obs, int(frame), args.decades, compact=True)
        axis.set_title(f"t = {times[int(frame)]:.3f} fs", color="white", fontsize=9)
    fig.colorbar(
        image, ax=list(axes.flat), pad=0.008, shrink=0.82,
        label=rf"$\log_{{10}}[\rho_{{qR}}/\rho_{{qR,\max}}(t)]$",
    )
    fig.suptitle(
        r"Proton-heavy joint density; dashed line is contact $q=R$",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/"joint_density_qR_snapshots.png", args.dpi,
    ))

    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        first = int(frames[0])
        fig, axis = plt.subplots(figsize=(9.4, 7.0), constrained_layout=True)
        image = _draw_joint_density(axis, obs, first, args.decades)
        fig.colorbar(
            image, ax=axis, pad=0.012,
            label=rf"$\log_{{10}}[\rho_{{qR}}/\rho_{{qR,\max}}(t)]$",
        )
        title = fig.suptitle("", fontweight="bold")

        def update(number):
            frame = int(frames[number])
            image.set_data(tdse_collision_report._relative_log_frame(
                obs["joint_density"][frame], args.decades,
            ).T)
            title.set_text(f"Proton-heavy joint density | t={times[frame]:.4f} fs")
            return image, title

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(tdse_report._save_animation(
            animation, fig, outdir, "joint_density_qR_movie",
            args.fps, args.animation_dpi, args.format,
        ))
    return products


# ---------------------------------------------------------------------------
# 2b. Proton-heavy joint density + positive-gauge velocity field


def _uniform_sample_indices(coordinate, limits, count):
    """Fixed, approximately uniform indices inside one display window."""
    coordinate = np.asarray(coordinate, float)
    inside = np.flatnonzero(
        (coordinate >= float(limits[0])) & (coordinate <= float(limits[1]))
    )
    if not inside.size:
        return np.array([
            int(np.argmin(np.abs(coordinate-np.mean(limits))))
        ])
    selected = np.linspace(
        int(inside[0]), int(inside[-1]), min(int(count), len(inside)),
    )
    return np.unique(np.rint(selected).astype(int))


def _joint_velocity_preparation(obs, ef, args):
    """Prepare one trajectory-wide sampling grid and physical arrow scale."""
    q, R = np.asarray(obs["q"], float), np.asarray(obs["R"], float)
    q_limits = tdse_collision_report._q_display_limits(q)
    R_limits = (float(R[0]), float(R[-1]))
    q_indices = _uniform_sample_indices(q, q_limits, args.velocity_q_points)
    R_indices = _uniform_sample_indices(R, R_limits, args.velocity_R_points)
    q_mesh, R_mesh = np.meshgrid(q[q_indices], R[R_indices])
    proton_mass = float(obs["options"].get("proton_mass", 1836.15267343))
    heavy_mass = float(obs["options"].get("heavy_mass", 1836.15267343))

    supported_speeds = []
    focus_floor = getattr(args, "analysis_focus_floor", args.support_floor)
    # Robust trajectory-wide limits do not need an expensive full-resolution
    # decomposition pass over every movie frame.  Evenly sample the complete
    # trajectory; the plotted physical arrays themselves are never sampled or
    # altered by this choice.
    for frame in _movie_frames(obs, min(
        args.max_frames, getattr(args, "scale_sample_frames", 32),
    )):
        frame = int(frame)
        density = obs["joint_density"][frame][np.ix_(q_indices, R_indices)]
        support = density >= focus_floor*max(
            float(np.max(obs["joint_density"][frame])), 1.0e-300,
        )
        velocity_q = (
            ef["a"][frame][np.ix_(q_indices, R_indices)]/proton_mass
        )
        velocity_R = (
            ef["b"][frame][np.ix_(q_indices, R_indices)]/heavy_mass
        )
        speed = np.hypot(velocity_q, velocity_R)
        valid = support & np.isfinite(speed)
        if np.any(valid):
            supported_speeds.append(speed[valid])

    if supported_speeds:
        reference_speed = float(np.percentile(
            np.concatenate(supported_speeds), 95.0,
        ))
    else:
        reference_speed = 1.0
    reference_speed = max(reference_speed, 1.0e-14)
    reference_length = 0.055*min(
        float(q_limits[1]-q_limits[0]), float(R_limits[1]-R_limits[0]),
    )
    quiver_scale = reference_speed/max(reference_length, 1.0e-12)
    return {
        "q_indices": q_indices,
        "R_indices": R_indices,
        "q_mesh": q_mesh,
        "R_mesh": R_mesh,
        "q_limits": q_limits,
        "R_limits": R_limits,
        "proton_mass": proton_mass,
        "heavy_mass": heavy_mass,
        "reference_speed": reference_speed,
        "quiver_scale": quiver_scale,
    }


def _joint_velocity_frame(obs, ef, prep, frame, floor):
    """Return mass-scaled velocity components on density-supported sites."""
    q_indices, R_indices = prep["q_indices"], prep["R_indices"]
    density = obs["joint_density"][frame][np.ix_(q_indices, R_indices)]
    cutoff = float(floor)*max(
        float(np.max(obs["joint_density"][frame])), 1.0e-300,
    )
    support = density >= cutoff
    velocity_q = (
        ef["a"][frame][np.ix_(q_indices, R_indices)]/prep["proton_mass"]
    )
    velocity_R = (
        ef["b"][frame][np.ix_(q_indices, R_indices)]/prep["heavy_mass"]
    )
    invalid = ~support | ~np.isfinite(velocity_q) | ~np.isfinite(velocity_R)
    # imshow uses (q, R).T while quiver's mesh is (R rows, q columns).
    return (
        np.ma.array(velocity_q.T, mask=invalid.T),
        np.ma.array(velocity_R.T, mask=invalid.T),
    )


def _draw_joint_velocity(axis, obs, ef, prep, frame, args, *, compact=False):
    image = _draw_joint_density(
        axis, obs, frame, args.decades, compact=compact,
    )
    velocity_q, velocity_R = _joint_velocity_frame(
        obs, ef, prep, frame, args.analysis_focus_floor,
    )
    arrows = axis.quiver(
        prep["q_mesh"], prep["R_mesh"], velocity_q, velocity_R,
        color="#55DDE0", edgecolor="#102A30",
        linewidth=(0.18 if compact else 0.28),
        angles="xy", scale_units="xy", scale=prep["quiver_scale"],
        width=(0.0022 if compact else 0.0028),
        headwidth=3.5, headlength=4.5, headaxislength=4.0,
        pivot="mid", zorder=4,
    )
    if not compact:
        axis.quiverkey(
            arrows, 0.985, 1.025, prep["reference_speed"],
            rf"$v_{{95}}={_math_scientific(prep['reference_speed'])}"
            rf"\ a_0/t_{{\rm au}}$",
            labelpos="W", coordinates="axes", color="#102A30",
            labelcolor="#102A30", fontproperties={"size": 8},
        )
    return image, arrows


def render_joint_velocity(obs, ef, outdir, args, snapshots):
    """Render joint density with (a/m_p, b/M) positive-gauge arrows."""
    times = obs["times_fs"]
    prep = _joint_velocity_preparation(obs, ef, args)

    def individual(frame):
        fig, axis = plt.subplots(figsize=(9.8, 7.2), constrained_layout=True)
        image, _ = _draw_joint_velocity(axis, obs, ef, prep, frame, args)
        fig.colorbar(
            image, ax=axis, pad=0.012,
            label=rf"$\log_{{10}}[\rho_{{qR}}/\rho_{{qR,\max}}(t)]$",
        )
        fig.suptitle(
            f"Joint density and positive-gauge velocity | "
            f"t={times[frame]:.4f} fs\n"
            r"$(v_q,v_R)=(K_q/m_p,K_R^{(1)}/M)=(a/m_p,b/M)$",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times, Path(outdir)/"joint_velocity_frames",
        "joint_velocity", args.dpi,
    )
    fig, axes = plt.subplots(2, 4, figsize=(21.0, 9.6), constrained_layout=True)
    image = None
    for axis, frame in zip(axes.flat, snapshots):
        image, _ = _draw_joint_velocity(
            axis, obs, ef, prep, int(frame), args, compact=True,
        )
        axis.set_title(f"t = {times[int(frame)]:.3f} fs", color="white", fontsize=9)
    fig.colorbar(
        image, ax=list(axes.flat), pad=0.008, shrink=0.82,
        label=rf"$\log_{{10}}[\rho_{{qR}}/\rho_{{qR,\max}}(t)]$",
    )
    fig.suptitle(
        r"Proton-heavy probability density and mechanical velocity field",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/"joint_velocity_snapshots.png", args.dpi,
    ))

    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        first = int(frames[0])
        fig, axis = plt.subplots(figsize=(9.8, 7.2), constrained_layout=True)
        image, arrows = _draw_joint_velocity(axis, obs, ef, prep, first, args)
        fig.colorbar(
            image, ax=axis, pad=0.012,
            label=rf"$\log_{{10}}[\rho_{{qR}}/\rho_{{qR,\max}}(t)]$",
        )
        title = fig.suptitle("", fontweight="bold")

        def update(number):
            frame = int(frames[number])
            image.set_data(tdse_collision_report._relative_log_frame(
                obs["joint_density"][frame], args.decades,
            ).T)
            velocity_q, velocity_R = _joint_velocity_frame(
                obs, ef, prep, frame, args.analysis_focus_floor,
            )
            arrows.set_UVC(velocity_q, velocity_R)
            title.set_text(
                f"Joint density and positive-gauge velocity | "
                f"t={times[frame]:.4f} fs\n"
                r"$(v_q,v_R)=(a/m_p,b/M)$"
            )
            return image, arrows, title

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(tdse_report._save_animation(
            animation, fig, outdir, "joint_velocity_movie",
            args.fps, args.animation_dpi, args.format,
        ))
    prep["arrow_support_floor"] = float(args.analysis_focus_floor)
    return products, prep


# ---------------------------------------------------------------------------
# 3. Particle dynamics + positive-gauge vector potentials


def _vector_preparation(obs, ef, args):
    floor = getattr(args, "analysis_focus_floor", args.support_floor)
    frames = _movie_frames(obs, args.max_frames)
    densities = [obs["joint_density"][int(frame)] for frame in frames]
    a_values = [ef["a"][int(frame)] for frame in frames]
    b_values = [ef["b"][int(frame)] for frame in frames]
    connection_limits = _symmetric_support_bound(
        a_values+b_values, densities+densities, floor,
    )
    alpha_lifted, heavy_support, _ = tdse_report.support_aware_temporal_lift_1d(
        ef["alpha"], obs["heavy_density"], obs["dR"], args.support_floor,
    )
    heavy_support = obs["heavy_density"] >= floor*np.maximum(
        np.max(obs["heavy_density"], axis=1, keepdims=True), 1.0e-300,
    )
    alpha_limits = _symmetric_support_bound(
        [alpha_lifted[int(frame)] for frame in frames],
        [obs["heavy_density"][int(frame)] for frame in frames], floor,
    )
    q_limits = _support_limits(obs["q"], obs["proton_density"], floor)
    R_limits = _support_limits(obs["R"], obs["heavy_density"], floor)
    return {
        "connection_limits": connection_limits,
        "alpha_limits": alpha_limits,
        "alpha_lifted": alpha_lifted,
        "heavy_support": heavy_support,
        "q_limits": q_limits,
        "R_limits": R_limits,
    }


def _new_vector_axes(figsize=(15.6, 8.8), *, compact=False, subplot_spec=None,
                     figure=None):
    if subplot_spec is None:
        figure = plt.figure(figsize=figsize, constrained_layout=True)
        grid = figure.add_gridspec(2, 3, height_ratios=(0.72, 1.0))
    else:
        grid = subplot_spec.subgridspec(2, 3, height_ratios=(0.62, 1.0), hspace=0.12)
    axes = {
        "marginal": figure.add_subplot(grid[0, :]),
        "a": figure.add_subplot(grid[1, 0]),
        "b": figure.add_subplot(grid[1, 1]),
        "alpha": figure.add_subplot(grid[1, 2]),
    }
    if compact:
        for axis in axes.values():
            axis.tick_params(labelsize=5.5)
    return figure, axes


def _draw_particle_marginal_panel(axis, obs, frame, args, *, compact=False):
    """Draw the shared upper panel used by vector/current composites."""
    marginal_lines = []
    for name, coordinate, density in (
        ("electron", obs["x"], obs["electron_density"]),
        ("proton", obs["q"], obs["proton_density"]),
        ("heavy", obs["R"], obs["heavy_density"]),
    ):
        line, = axis.plot(
            coordinate, density[frame], color=PARTICLE_COLORS[name],
            lw=(1.15 if compact else 2.0), label=name,
        )
        marginal_lines.append(line)
    add_fixed_center_markers(axis, obs["options"])
    axis.set(
        xlim=(-args.marginal_xmax, args.marginal_xmax),
        ylim=(0.0, args.marginal_ymax),
        xlabel=("" if compact else r"common position coordinate ($a_0$)"),
        ylabel=r"density ($a_0^{-1}$)",
    )
    axis.set_title(
        "Particle marginals" if compact else
        "Electron, proton and heavy-nucleus marginals | fixed display scale",
        loc="left", fontweight="semibold", fontsize=(7 if compact else None),
    )
    axis.legend(frameon=False, ncol=3, fontsize=(5.5 if compact else 8))
    axis.grid(alpha=0.18)
    return marginal_lines


def _add_attached_colorbar(fig, axis, image, label):
    """Attach a narrow colorbar directly to a final-composite map panel."""
    color_axis = axis.inset_axes([1.020, 0.035, 0.034, 0.93])
    colorbar = fig.colorbar(
        image, cax=color_axis, format=NUMBER_FORMATTER,
        extend="both", label=label,
    )
    colorbar.ax.tick_params(labelsize=8, pad=1.5)
    return colorbar


def _draw_vector_composite(fig, axes, obs, ef, prep, frame, args, *,
                           colorbars=True, compact=False):
    q, R = obs["q"], obs["R"]
    marginal_lines = _draw_particle_marginal_panel(
        axes["marginal"], obs, frame, args, compact=compact,
    )

    density = obs["joint_density"][frame]
    active, _, frame_limits = _frame_focus(
        obs, frame, args.analysis_focus_floor,
    )
    opacity = active.astype(float)
    extent = [q[0], q[-1], R[0], R[-1]]
    images = []
    for key, label in (("a", r"$a(q,R,t)$"), ("b", r"$b(q,R,t)$")):
        axis = axes[key]
        axis.set_facecolor(MASK_COLOR)
        image = axis.imshow(
            ef[key][frame].T, origin="lower", aspect="auto",
            interpolation="nearest", extent=extent,
            cmap=masked_cmap(SIGNED_CMAP),
            vmin=prep["connection_limits"][0],
            vmax=prep["connection_limits"][1], alpha=opacity.T,
        )
        axis.set(
            xlim=frame_limits[0], ylim=frame_limits[1],
            xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)",
        )
        axis.set_title(label+" | positive-density gauge", fontsize=(7 if compact else 9))
        if colorbars:
            _add_attached_colorbar(
                fig, axis, image, r"connection ($a_0^{-1}$)",
            )
        images.append((image, key))

    heavy = obs["heavy_density"][frame]
    support = prep["heavy_support"][frame]
    occupied = np.where(support, prep["alpha_lifted"][frame], np.nan)
    alpha_line, alpha_tail = tdse_report._support_tail_lines(
        axes["alpha"], R, occupied, prep["alpha_lifted"][frame], support,
        color=COLORS[3], label=r"$\alpha(R,t)$", linewidth=(1.2 if compact else 2.0),
    )
    density_line = tdse_report._scaled_heavy_density(axes["alpha"], R, heavy)
    axes["alpha"].set(
        xlim=prep["R_limits"], ylim=prep["alpha_limits"],
        xlabel=r"heavy $R$ ($a_0$)", ylabel=r"$\alpha$ ($a_0^{-1}$)",
    )
    axes["alpha"].set_title(
        r"$\alpha(R,t)$ | positive-density gauge", fontsize=(7 if compact else 9),
    )
    axes["alpha"].legend(frameon=False, fontsize=(5.5 if compact else 8))
    axes["alpha"].grid(alpha=0.18)
    return {
        "marginal_lines": marginal_lines,
        "images": images,
        "alpha_line": alpha_line,
        "alpha_tail": alpha_tail,
        "density_line": density_line,
        "axes": axes,
    }


def _update_vector_composite(state, obs, ef, prep, frame, args):
    for line, density in zip(state["marginal_lines"], (
        obs["electron_density"], obs["proton_density"], obs["heavy_density"],
    )):
        line.set_ydata(density[frame])
    active, _, frame_limits = _frame_focus(
        obs, frame, args.analysis_focus_floor,
    )
    opacity = active.astype(float)
    for image, key in state["images"]:
        image.set_data(ef[key][frame].T)
        image.set_alpha(opacity.T)
    for axis_name in ("a", "b"):
        state["axes"][axis_name].set_xlim(frame_limits[0])
        state["axes"][axis_name].set_ylim(frame_limits[1])
    support = prep["heavy_support"][frame]
    alpha = prep["alpha_lifted"][frame]
    state["alpha_line"].set_ydata(np.where(support, alpha, np.nan))
    state["alpha_tail"].set_ydata(np.where(~support, alpha, np.nan))
    heavy = obs["heavy_density"][frame]
    state["density_line"].set_ydata(heavy/max(float(np.max(heavy)), 1.0e-300))


def render_vector_composite(obs, ef, outdir, args, snapshots):
    if obs.get("electron_density") is None or obs.get("x") is None:
        raise KeyError("vector composite에는 저장된 electron marginal이 필요합니다")
    prep = _vector_preparation(obs, ef, args)
    times = obs["times_fs"]

    def individual(frame):
        fig, axes = _new_vector_axes()
        _draw_vector_composite(fig, axes, obs, ef, prep, frame, args)
        fig.suptitle(
            f"Particle dynamics and positive-gauge vector potentials | "
            f"t={times[frame]:.4f} fs",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times,
        Path(outdir)/"vector_potential_composite_frames",
        "vector_potential_composite", args.dpi,
    )
    # Match the aspect of the four-panel single-frame product inside each
    # 2x4 cell; a tall canvas makes the nested panels collapse horizontally.
    fig = plt.figure(figsize=(24.0, 8.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 4)
    for slot, frame in zip(outer, snapshots):
        _, axes = _new_vector_axes(compact=True, subplot_spec=slot, figure=fig)
        _draw_vector_composite(
            fig, axes, obs, ef, prep, int(frame), args,
            colorbars=False, compact=True,
        )
        axes["marginal"].text(
            0.99, 0.92, f"t={times[int(frame)]:.3f} fs",
            transform=axes["marginal"].transAxes, ha="right", va="top",
            fontsize=6.5,
        )
    fig.suptitle(
        "Particle dynamics with positive-density-gauge vector potentials",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/"vector_potential_composite_snapshots.png", args.dpi,
    ))

    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        first = int(frames[0])
        fig, axes = _new_vector_axes()
        state = _draw_vector_composite(fig, axes, obs, ef, prep, first, args)
        title = fig.suptitle("", fontweight="bold")

        def update(number):
            frame = int(frames[number])
            _update_vector_composite(state, obs, ef, prep, frame, args)
            title.set_text(
                "Particle dynamics and positive-density-gauge vector potentials | "
                f"t={times[frame]:.4f} fs"
            )
            return (
                *state["marginal_lines"],
                *(image for image, _ in state["images"]),
                state["alpha_line"], state["alpha_tail"],
                state["density_line"], title,
            )

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(tdse_report._save_animation(
            animation, fig, outdir, "vector_potential_composite_movie",
            args.fps, args.animation_dpi, args.format,
        ))
    return products, prep


# ---------------------------------------------------------------------------
# 3b. Particle dynamics + positive-gauge probability currents


def _new_current_axes(figsize=(15.6, 8.8), *, compact=False,
                      subplot_spec=None, figure=None):
    if subplot_spec is None:
        figure = plt.figure(figsize=figsize, constrained_layout=True)
        grid = figure.add_gridspec(2, 3, height_ratios=(0.72, 1.0))
    else:
        grid = subplot_spec.subgridspec(
            2, 3, height_ratios=(0.62, 1.0), hspace=0.12,
        )
    axes = {
        "marginal": figure.add_subplot(grid[0, :]),
        "proton": figure.add_subplot(grid[1, 0]),
        "heavy_joint": figure.add_subplot(grid[1, 1]),
        "heavy_marginal": figure.add_subplot(grid[1, 2]),
    }
    if compact:
        for axis in axes.values():
            axis.tick_params(labelsize=5.5)
    return figure, axes


def _supported_percentile_bound(values, density, floor, percentile=99.0):
    support = density >= float(floor)*max(float(np.max(density)), 1.0e-300)
    selected = np.abs(np.asarray(values, float)[support])
    selected = selected[np.isfinite(selected)]
    return float(np.percentile(selected, percentile)) if selected.size else 0.0


def _current_preparation(obs, ef, args):
    frames = _movie_frames(obs, args.max_frames)
    proton_mass = float(obs["options"].get("proton_mass", 1836.15267343))
    heavy_mass = float(obs["options"].get("heavy_mass", 1836.15267343))
    alpha_lifted, heavy_support, _ = tdse_report.support_aware_temporal_lift_1d(
        ef["alpha"], obs["heavy_density"], obs["dR"], args.support_floor,
    )
    display_floor = getattr(args, "analysis_focus_floor", args.support_floor)
    heavy_support = obs["heavy_density"] >= display_floor*np.maximum(
        np.max(obs["heavy_density"], axis=1, keepdims=True), 1.0e-300,
    )
    proton_bounds, heavy_joint_bounds, heavy_marginal_bounds = [], [], []
    for frame in frames:
        frame = int(frame)
        density = obs["joint_density"][frame]
        heavy = obs["heavy_density"][frame]
        proton_current = density*ef["a"][frame]/proton_mass
        heavy_joint_current = density*ef["b"][frame]/heavy_mass
        heavy_marginal_current = heavy*alpha_lifted[frame]/heavy_mass
        proton_bounds.append(_supported_percentile_bound(
            proton_current, density, display_floor,
        ))
        heavy_joint_bounds.append(_supported_percentile_bound(
            heavy_joint_current, density, display_floor,
        ))
        heavy_marginal_bounds.append(_supported_percentile_bound(
            heavy_marginal_current, heavy, display_floor,
        ))

    def limits(bounds):
        bound = max(
            float(np.percentile(bounds, 98.0)) if bounds else 0.0,
            1.0e-18,
        )
        return -bound, bound

    return {
        "proton_mass": proton_mass,
        "heavy_mass": heavy_mass,
        "alpha_lifted": alpha_lifted,
        "heavy_support": heavy_support,
        "proton_limits": limits(proton_bounds),
        "heavy_joint_limits": limits(heavy_joint_bounds),
        "heavy_marginal_limits": limits(heavy_marginal_bounds),
        "q_limits": _support_limits(
            obs["q"], obs["proton_density"], display_floor,
        ),
        "R_limits": _support_limits(
            obs["R"], obs["heavy_density"], display_floor,
        ),
    }


def _current_frame(obs, ef, prep, frame):
    density = obs["joint_density"][frame]
    heavy = obs["heavy_density"][frame]
    return {
        "proton": density*ef["a"][frame]/prep["proton_mass"],
        "heavy_joint": density*ef["b"][frame]/prep["heavy_mass"],
        "heavy_marginal": (
            heavy*prep["alpha_lifted"][frame]/prep["heavy_mass"]
        ),
    }


def _draw_current_composite(fig, axes, obs, ef, prep, frame, args, *,
                            colorbars=True, compact=False):
    q, R = obs["q"], obs["R"]
    marginal_lines = _draw_particle_marginal_panel(
        axes["marginal"], obs, frame, args, compact=compact,
    )
    current = _current_frame(obs, ef, prep, frame)
    density = obs["joint_density"][frame]
    active, _, frame_limits = _frame_focus(
        obs, frame, args.analysis_focus_floor,
    )
    opacity = active.astype(float)
    extent = [q[0], q[-1], R[0], R[-1]]
    specifications = (
        (
            "proton", prep["proton_limits"],
            r"$J_A^p=\rho_{qR}K_A^p/m_p=\rho_{qR}a/m_p$",
        ),
        (
            "heavy_joint", prep["heavy_joint_limits"],
            r"$J_c^R=\rho_{qR}K_c^R/M=\rho_{qR}b/M$",
        ),
    )
    images = []
    for key, value_limits, label in specifications:
        axis = axes[key]
        axis.set_facecolor(MASK_COLOR)
        image = axis.imshow(
            current[key].T, origin="lower", aspect="auto",
            interpolation="nearest", extent=extent,
            cmap=masked_cmap(SIGNED_CMAP),
            vmin=value_limits[0], vmax=value_limits[1], alpha=opacity.T,
        )
        axis.set(
            xlim=frame_limits[0], ylim=frame_limits[1],
            xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)",
        )
        axis.set_title(label, fontsize=(5.8 if compact else 8.5))
        if colorbars:
            _add_attached_colorbar(
                fig, axis, image, "joint probability current (a.u.)",
            )
        images.append((image, key))

    support = prep["heavy_support"][frame]
    heavy_marginal = current["heavy_marginal"]
    current_line, current_tail = tdse_report._support_tail_lines(
        axes["heavy_marginal"], R,
        np.where(support, heavy_marginal, np.nan),
        heavy_marginal, support, color=COLORS[0],
        label=r"$\overline{J_c^R}$", linewidth=(1.2 if compact else 2.0),
    )
    density_line = tdse_report._scaled_heavy_density(
        axes["heavy_marginal"], R, obs["heavy_density"][frame],
    )
    axes["heavy_marginal"].set(
        xlim=prep["R_limits"], ylim=prep["heavy_marginal_limits"],
        xlabel=r"heavy $R$ ($a_0$)",
        ylabel=r"marginal probability current (a.u.)",
    )
    axes["heavy_marginal"].set_title(
        r"$\overline{J_c^R}=\int dq\,J_c^R=\rho_R\alpha/M$",
        fontsize=(5.8 if compact else 8.5),
    )
    axes["heavy_marginal"].legend(
        frameon=False, fontsize=(5.5 if compact else 8),
    )
    axes["heavy_marginal"].grid(alpha=0.18)
    return {
        "marginal_lines": marginal_lines,
        "images": images,
        "current_line": current_line,
        "current_tail": current_tail,
        "density_line": density_line,
        "axes": axes,
    }


def _update_current_composite(state, obs, ef, prep, frame, args):
    for line, density in zip(state["marginal_lines"], (
        obs["electron_density"], obs["proton_density"], obs["heavy_density"],
    )):
        line.set_ydata(density[frame])
    current = _current_frame(obs, ef, prep, frame)
    active, _, frame_limits = _frame_focus(
        obs, frame, args.analysis_focus_floor,
    )
    opacity = active.astype(float)
    for image, key in state["images"]:
        image.set_data(current[key].T)
        image.set_alpha(opacity.T)
    for axis_name in ("proton", "heavy_joint"):
        state["axes"][axis_name].set_xlim(frame_limits[0])
        state["axes"][axis_name].set_ylim(frame_limits[1])
    support = prep["heavy_support"][frame]
    heavy_marginal = current["heavy_marginal"]
    state["current_line"].set_ydata(
        np.where(support, heavy_marginal, np.nan),
    )
    state["current_tail"].set_ydata(
        np.where(~support, heavy_marginal, np.nan),
    )
    heavy = obs["heavy_density"][frame]
    state["density_line"].set_ydata(
        heavy/max(float(np.max(heavy)), 1.0e-300),
    )


def render_current_composite(obs, ef, outdir, args, snapshots):
    if obs.get("electron_density") is None or obs.get("x") is None:
        raise KeyError("current composite에는 저장된 electron marginal이 필요합니다")
    prep = _current_preparation(obs, ef, args)
    times = obs["times_fs"]

    def individual(frame):
        fig, axes = _new_current_axes()
        _draw_current_composite(fig, axes, obs, ef, prep, frame, args)
        fig.suptitle(
            "Particle dynamics and positive-density-gauge probability "
            f"currents | t={times[frame]:.4f} fs",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times,
        Path(outdir)/"current_density_composite_frames",
        "current_density_composite", args.dpi,
    )
    fig = plt.figure(figsize=(24.0, 8.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 4)
    for slot, frame in zip(outer, snapshots):
        _, axes = _new_current_axes(
            compact=True, subplot_spec=slot, figure=fig,
        )
        _draw_current_composite(
            fig, axes, obs, ef, prep, int(frame), args,
            colorbars=False, compact=True,
        )
        axes["marginal"].text(
            0.99, 0.92, f"t={times[int(frame)]:.3f} fs",
            transform=axes["marginal"].transAxes, ha="right", va="top",
            fontsize=6.5,
        )
    fig.suptitle(
        "Particle dynamics with positive-density-gauge probability currents",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/"current_density_composite_snapshots.png", args.dpi,
    ))

    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        first = int(frames[0])
        fig, axes = _new_current_axes()
        state = _draw_current_composite(
            fig, axes, obs, ef, prep, first, args,
        )
        title = fig.suptitle("", fontweight="bold")

        def update(number):
            frame = int(frames[number])
            _update_current_composite(state, obs, ef, prep, frame, args)
            title.set_text(
                "Particle dynamics and positive-density-gauge probability "
                f"currents | t={times[frame]:.4f} fs"
            )
            return (
                *state["marginal_lines"],
                *(image for image, _ in state["images"]),
                state["current_line"], state["current_tail"],
                state["density_line"], title,
            )

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(tdse_report._save_animation(
            animation, fig, outdir, "current_density_composite_movie",
            args.fps, args.animation_dpi, args.format,
        ))
    return products, prep


# ---------------------------------------------------------------------------
# 4. Nested-factorization potential and conditional-density analysis


def _robust_shifted_limits(arrays, densities, floor):
    """Trajectory-wide scalar limits after one occupied-density offset."""
    lower, upper = [], []
    for values, density in zip(arrays, densities):
        shifted = density_weighted_shift(values, density, floor)
        support = (
            np.asarray(density, float)
            >= float(floor)*max(float(np.max(density)), 1.0e-300)
        )
        selected = shifted[support & np.isfinite(shifted)]
        if selected.size:
            lower.append(float(np.percentile(selected, 1.0)))
            upper.append(float(np.percentile(selected, 99.0)))
    if not lower:
        return -1.0, 1.0
    low = float(np.percentile(lower, 2.0))
    high = float(np.percentile(upper, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = 0.5*(low+high) if np.isfinite(low+high) else 0.0
        width = max(abs(center)*1.0e-6, 1.0e-12)
        return center-width, center+width
    padding = 0.04*(high-low)
    return low-padding, high+padding


def _robust_shifted_symmetric_limits(arrays, densities, floor):
    """Trajectory-wide zero-centred limits for shifted occupied fields."""
    per_frame = []
    for values, density in zip(arrays, densities):
        shifted = density_weighted_shift(values, density, floor)
        support = (
            np.asarray(density, float)
            >= float(floor)*max(float(np.max(density)), 1.0e-300)
        )
        selected = np.abs(shifted[support & np.isfinite(shifted)])
        if selected.size:
            per_frame.append(float(np.percentile(selected, 99.0)))
    bound = max(
        float(np.percentile(per_frame, 98.0)) if per_frame else 0.0,
        1.0e-12,
    )
    return -1.04*bound, 1.04*bound


def _conditional_proton_density(obs, frame):
    """Return rho(q|R)=rho(q,R)/rho_R without dividing at exact nodes."""
    joint = np.asarray(obs["joint_density"][frame], float)
    heavy = np.asarray(obs["heavy_density"][frame], float)
    conditional = np.zeros_like(joint)
    np.divide(
        joint, heavy[None, :], out=conditional,
        where=heavy[None, :] > np.finfo(np.float64).tiny,
    )
    return conditional


def _nested_frame(obs, ef_positive, frame, args):
    joint = obs["joint_density"][frame]
    heavy = obs["heavy_density"][frame]
    conditional = _conditional_proton_density(obs, frame)
    focus_floor = getattr(args, "analysis_focus_floor", args.support_floor)
    heavy_support = heavy >= focus_floor*max(
        float(np.max(heavy)), 1.0e-300,
    )
    joint_support = joint >= focus_floor*max(
        float(np.max(joint)), 1.0e-300,
    )
    return {
        "electron_proton": np.maximum(
            np.asarray(ef_positive["electron_proton_density"][frame], float), 0.0,
        ),
        "conditional": conditional,
        "conditional_opacity": np.broadcast_to(
            heavy_support[None, :].astype(float), conditional.shape,
        ),
        "joint_density": np.maximum(np.asarray(joint, float), 0.0),
        "joint_opacity": joint_support.astype(float),
        "epsilon_1": density_weighted_shift(
            ef_positive.get("tdpes1_total", ef_positive["epsilon_1"])[frame],
            joint, focus_floor,
        ),
        "epsilon_2": density_weighted_shift(
            ef_positive.get("tdpes2_total", ef_positive["epsilon_2"])[frame],
            heavy, focus_floor,
        ),
        "heavy_support": heavy_support,
    }


def _nested_preparation(obs, ef_positive, args):
    electron_proton = np.asarray(ef_positive["electron_proton_density"], float)
    expected = (len(obs["times_fs"]), len(obs["x"]), len(obs["q"]))
    if electron_proton.shape != expected:
        raise ValueError(
            "electron_proton_density shape mismatch: "
            f"{electron_proton.shape} != {expected}"
        )
    frames = _movie_frames(obs, args.max_frames)
    epsilon_1_source = ef_positive.get("tdpes1_total", ef_positive["epsilon_1"])
    epsilon_2_source = ef_positive.get("tdpes2_total", ef_positive["epsilon_2"])
    epsilon_1_limits = _robust_shifted_symmetric_limits(
        [epsilon_1_source[int(frame)] for frame in frames],
        [obs["joint_density"][int(frame)] for frame in frames],
        args.analysis_focus_floor,
    )
    epsilon_2_limits = _robust_shifted_limits(
        [epsilon_2_source[int(frame)] for frame in frames],
        [obs["heavy_density"][int(frame)] for frame in frames],
        args.analysis_focus_floor,
    )
    dx = float(obs["x"][1]-obs["x"][0])
    electron_proton_vmax = max(
        float(np.nanmax(electron_proton)), 1.0e-300,
    )
    all_frames = range(len(obs["times_fs"]))
    ep_mass_error = max(
        abs(
            float(np.sum(electron_proton[int(frame)])*dx*obs["dq"])-1.0
        )
        for frame in all_frames
    )
    conditional_normalization_error = 0.0
    conditional_vmax = 0.0
    for frame in range(len(obs["times_fs"])):
        frame = int(frame)
        heavy = obs["heavy_density"][frame]
        support = (
            heavy
            >= args.support_floor*max(float(np.max(heavy)), 1.0e-300)
        )
        if np.any(support):
            conditional = _conditional_proton_density(obs, frame)
            normalization = np.sum(conditional, axis=0)*obs["dq"]
            conditional_normalization_error = max(
                conditional_normalization_error,
                float(np.max(np.abs(normalization[support]-1.0))),
            )
            conditional_vmax = max(
                conditional_vmax,
                float(np.nanmax(conditional[:, support])),
            )
    return {
        "epsilon_1_limits": epsilon_1_limits,
        "epsilon_2_limits": epsilon_2_limits,
        "electron_proton_vmax": electron_proton_vmax,
        "conditional_vmax": max(conditional_vmax, 1.0e-300),
        "x_limits": _support_limits(
            obs["x"], obs["electron_density"], args.support_floor,
            requested=(-args.marginal_xmax, args.marginal_xmax),
        ),
        "q_limits": _support_limits(
            obs["q"], obs["proton_density"], args.support_floor,
        ),
        "R_limits": _support_limits(
            obs["R"], obs["heavy_density"], args.support_floor,
        ),
        "electron_proton_mass_error": ep_mass_error,
        "conditional_normalization_error": conditional_normalization_error,
    }


def _new_nested_axes(figsize=(15.4, 10.2), *, compact=False,
                     subplot_spec=None, figure=None):
    if subplot_spec is None:
        figure = plt.figure(figsize=figsize, constrained_layout=True)
        grid = figure.add_gridspec(2, 2)
    else:
        grid = subplot_spec.subgridspec(2, 2, wspace=0.12, hspace=0.20)
    axes = {
        "electron_proton": figure.add_subplot(grid[0, 0]),
        "conditional": figure.add_subplot(grid[0, 1]),
        "epsilon_1": figure.add_subplot(grid[1, 0]),
        "epsilon_2": figure.add_subplot(grid[1, 1]),
    }
    if compact:
        for axis in axes.values():
            axis.tick_params(labelsize=5.2, direction="in")
    return figure, axes


def _joint_contours(axis, obs, log_density, decades, compact=False, *,
                    q=None, R=None, color="white", halo_color="0.08"):
    # Quarter-decade spacing resolves shoulders and weakly connected branches
    # without changing or smoothing the underlying physical density.  Two
    # additional core contours distinguish the dense centre of each lobe.
    lower = np.ceil(-float(decades)*4.0)/4.0
    levels = np.arange(lower, -0.24, 0.25)
    levels = np.unique(np.concatenate((
        levels, np.log10(np.array((0.65, 0.82))),
    )))
    if levels.size < 3:
        levels = np.linspace(-0.9*float(decades), -0.1*float(decades), 3)
    contours = axis.contour(
        obs["q"] if q is None else q, obs["R"] if R is None else R,
        log_density.T, levels=levels,
        colors=color, linewidths=np.linspace(0.42, 1.05, len(levels)),
        alpha=(0.72 if compact else 0.88),
    )
    # A dark halo keeps the physical-density contours legible over both the
    # bright and dark ends of the scalar-potential colour map.  This changes
    # only the line rendering; contour levels and field values are untouched.
    halo_width = 1.05 if compact else 1.55
    for collection in contours.collections:
        collection.set_path_effects((
            path_effects.Stroke(
                linewidth=halo_width, foreground=halo_color, alpha=0.72,
            ),
            path_effects.Normal(),
        ))
    return contours


def _joint_linear_contours(axis, obs, density, compact=False, *,
                           color="black"):
    """Draw nested-analysis density contours on an ordinary linear scale."""
    density = np.maximum(np.asarray(density, float), 0.0)
    peak = max(float(np.max(density)), 1.0e-300)
    relative = density/peak
    levels = np.array((
        0.025, 0.05, 0.075, 0.10, 0.15, 0.20,
        0.30, 0.40, 0.50, 0.60, 0.75, 0.90,
    ))
    contours = axis.contour(
        obs["q"], obs["R"], relative.T, levels=levels,
        colors=color,
        linewidths=np.linspace(
            0.24 if compact else 0.34,
            0.50 if compact else 0.68,
            len(levels),
        ),
        linestyles="solid", alpha=(0.72 if compact else 0.84),
    )
    return contours


def _heavy_silhouette(axis, R, density, compact=False):
    normalized = density/max(float(np.max(density)), 1.0e-300)
    height = (0.18 if compact else 0.23)*normalized
    transform = axis.get_xaxis_transform()
    fill = axis.fill_between(
        R, 0.0, height, transform=transform,
        color=PARTICLE_COLORS["heavy"], alpha=0.22, linewidth=0,
    )
    line, = axis.plot(
        R, height, transform=transform,
        color=PARTICLE_COLORS["heavy"],
        lw=(0.8 if compact else 1.5), label=r"heavy $\rho_R$ silhouette",
    )
    return fill, line


def _draw_nested_composite(fig, axes, obs, ef_positive, prep, frame, args, *,
                           colorbars=True, compact=False):
    q, R, x = obs["q"], obs["R"], obs["x"]
    current = _nested_frame(obs, ef_positive, frame, args)
    _, _, joint_limits = _frame_focus(
        obs, frame, args.analysis_focus_floor,
    )
    _, _, heavy_limits = _frame_heavy_focus(
        obs, frame, args.analysis_focus_floor,
    )
    density_extent = [x[0], x[-1], q[0], q[-1]]
    qR_extent = [q[0], q[-1], R[0], R[-1]]

    electron_proton_image = axes["electron_proton"].imshow(
        current["electron_proton"].T, origin="lower", aspect="auto",
        interpolation="nearest", extent=density_extent, cmap=JOINT_CMAP,
        vmin=0.0, vmax=prep["electron_proton_vmax"],
    )
    axes["electron_proton"].set(
        xlim=prep["x_limits"], ylim=prep["q_limits"],
        xlabel=r"electron $x$ ($a_0$)", ylabel=r"proton $q$ ($a_0$)",
    )
    axes["electron_proton"].set_title(
        r"Absolute $\rho_{ep}(x,q)=\int dR\,|\Psi|^2$",
        loc="left", fontweight="semibold", fontsize=(6.2 if compact else 10),
    )
    _set_density_axis(axes["electron_proton"])

    conditional_image = axes["conditional"].imshow(
        current["conditional"].T, origin="lower", aspect="auto",
        interpolation="nearest", extent=qR_extent, cmap=JOINT_CMAP,
        vmin=0.0, vmax=prep["conditional_vmax"],
        alpha=current["conditional_opacity"].T,
    )
    axes["conditional"].set(
        xlim=joint_limits[0], ylim=joint_limits[1],
        xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)",
    )
    axes["conditional"].set_title(
        r"Conditional proton $\rho(q|R)=\rho_{qR}/\rho_R=|\Lambda_R|^2$",
        loc="left", fontweight="semibold", fontsize=(6.2 if compact else 10),
    )
    _set_density_axis(axes["conditional"])

    axes["epsilon_1"].set_facecolor(MASK_COLOR)
    epsilon_1_image = axes["epsilon_1"].imshow(
        current["epsilon_1"].T, origin="lower", aspect="auto",
        interpolation="nearest", extent=qR_extent,
        cmap=masked_cmap(SIGNED_CMAP),
        vmin=prep["epsilon_1_limits"][0],
        vmax=prep["epsilon_1_limits"][1],
        alpha=current["joint_opacity"].T,
    )
    contours = _joint_linear_contours(
        axes["epsilon_1"], obs, current["joint_density"], compact,
    )
    axes["epsilon_1"].set(
        xlim=joint_limits[0], ylim=joint_limits[1],
        xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)",
    )
    axes["epsilon_1"].set_title(
        r"First TDPES $\epsilon_{\rm PG}^{(1)}(q,R)$ + $\rho_{qR}$ contours "
        r"(linear density contours; denser inward)",
        loc="left", fontweight="semibold", fontsize=(6.2 if compact else 10),
    )

    support = current["heavy_support"]
    epsilon_2_line, epsilon_2_tail = tdse_report._support_tail_lines(
        axes["epsilon_2"], R,
        np.where(support, current["epsilon_2"], np.nan),
        current["epsilon_2"], support, color=COLORS[0],
        label=r"$\epsilon_{\rm PG}^{(2)}(R,t)$",
        linewidth=(1.0 if compact else 2.2),
    )
    heavy_fill, heavy_line = _heavy_silhouette(
        axes["epsilon_2"], R, obs["heavy_density"][frame], compact,
    )
    axes["epsilon_2"].axhline(0.0, color="0.72", lw=0.65, zorder=0)
    axes["epsilon_2"].set(
        xlim=heavy_limits, ylim=prep["epsilon_2_limits"],
        xlabel=r"heavy $R$ ($a_0$)", ylabel="shifted energy (Hartree)",
    )
    axes["epsilon_2"].set_title(
        r"Second TDPES $\epsilon_{\rm PG}^{(2)}(R)$ and heavy support",
        loc="left", fontweight="semibold", fontsize=(6.2 if compact else 10),
    )
    axes["epsilon_2"].grid(alpha=0.16)
    axes["epsilon_2"].legend(
        handles=(epsilon_2_line, heavy_line), frameon=False,
        fontsize=(5.0 if compact else 8), loc="best",
    )

    if colorbars:
        fig.colorbar(
            electron_proton_image, ax=axes["electron_proton"],
            pad=0.014, format=NUMBER_FORMATTER,
            label=rf"$\rho_{{ep}}$ ($a_0^{{-2}}$)",
        )
        fig.colorbar(
            conditional_image, ax=axes["conditional"],
            pad=0.014, format=NUMBER_FORMATTER,
            label=rf"$\rho(q|R)$ ($a_0^{{-1}}$)",
        )
        fig.colorbar(
            epsilon_1_image, ax=axes["epsilon_1"], pad=0.014,
            format=NUMBER_FORMATTER, extend="both",
            label="shifted energy (Hartree)",
        )
    return {
        "electron_proton_image": electron_proton_image,
        "conditional_image": conditional_image,
        "epsilon_1_image": epsilon_1_image,
        "contours": contours,
        "epsilon_2_line": epsilon_2_line,
        "epsilon_2_tail": epsilon_2_tail,
        "heavy_fill": heavy_fill,
        "heavy_line": heavy_line,
        "axes": axes,
    }


def _update_nested_composite(state, obs, ef_positive, prep, frame, args):
    current = _nested_frame(obs, ef_positive, frame, args)
    _, _, joint_limits = _frame_focus(
        obs, frame, args.analysis_focus_floor,
    )
    _, _, heavy_limits = _frame_heavy_focus(
        obs, frame, args.analysis_focus_floor,
    )
    state["electron_proton_image"].set_data(
        current["electron_proton"].T,
    )
    state["conditional_image"].set_data(current["conditional"].T)
    state["conditional_image"].set_alpha(
        current["conditional_opacity"].T,
    )
    state["epsilon_1_image"].set_data(current["epsilon_1"].T)
    state["epsilon_1_image"].set_alpha(current["joint_opacity"].T)
    for axis_name in ("conditional", "epsilon_1"):
        state["axes"][axis_name].set_xlim(joint_limits[0])
        state["axes"][axis_name].set_ylim(joint_limits[1])
    state["axes"]["epsilon_2"].set_xlim(heavy_limits)
    for collection in state["contours"].collections:
        collection.remove()
    state["contours"] = _joint_linear_contours(
        state["axes"]["epsilon_1"], obs, current["joint_density"],
    )
    support = current["heavy_support"]
    state["epsilon_2_line"].set_ydata(
        np.where(support, current["epsilon_2"], np.nan),
    )
    state["epsilon_2_tail"].set_ydata(
        np.where(~support, current["epsilon_2"], np.nan),
    )
    state["heavy_fill"].remove()
    state["heavy_fill"], temporary_line = _heavy_silhouette(
        state["axes"]["epsilon_2"], obs["R"],
        obs["heavy_density"][frame],
    )
    temporary_line.remove()
    normalized = (
        obs["heavy_density"][frame]
        /max(float(np.max(obs["heavy_density"][frame])), 1.0e-300)
    )
    state["heavy_line"].set_ydata(0.23*normalized)


def render_nested_factorization(obs, ef_positive, outdir, args, snapshots):
    if obs.get("electron_density") is None or obs.get("x") is None:
        raise KeyError(
            "nested analysis에는 electron marginal과 x grid가 필요합니다"
        )
    prep = _nested_preparation(obs, ef_positive, args)
    times = obs["times_fs"]

    def individual(frame):
        fig, axes = _new_nested_axes()
        _draw_nested_composite(fig, axes, obs, ef_positive, prep, frame, args)
        fig.suptitle(
            "Nested factorization: correlated densities and exact potentials | "
            f"t={times[frame]:.4f} fs\n"
            r"absolute densities: trajectory-fixed linear scales; potentials: "
            r"positive-density gauge; contours: physical $\rho_{qR}$",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times,
        Path(outdir)/"nested_factorization_analysis_frames",
        "nested_factorization_analysis", args.dpi,
    )
    fig = plt.figure(figsize=(24.0, 10.8), constrained_layout=True)
    outer = fig.add_gridspec(2, 4)
    for slot, frame in zip(outer, snapshots):
        _, axes = _new_nested_axes(
            compact=True, subplot_spec=slot, figure=fig,
        )
        _draw_nested_composite(
            fig, axes, obs, ef_positive, prep, int(frame), args,
            colorbars=False, compact=True,
        )
        axes["electron_proton"].text(
            0.98, 0.92, f"t={times[int(frame)]:.3f} fs",
            transform=axes["electron_proton"].transAxes,
            ha="right", va="top", color="white", fontsize=6.0,
        )
    fig.suptitle(
        "Heavy-integrated electronic dynamics and subsequent proton factorization",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/"nested_factorization_analysis_snapshots.png",
        args.dpi,
    ))

    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        first = int(frames[0])
        fig, axes = _new_nested_axes()
        state = _draw_nested_composite(
            fig, axes, obs, ef_positive, prep, first, args,
        )
        title = fig.suptitle("", fontweight="bold")

        def update(number):
            frame = int(frames[number])
            _update_nested_composite(
                state, obs, ef_positive, prep, frame, args,
            )
            title.set_text(
                "Nested factorization: correlated densities and exact "
                f"potentials | t={times[frame]:.4f} fs\n"
                "absolute densities on trajectory-fixed scales; "
                "positive-density gauge; no smoothing"
            )
            return (
                state["electron_proton_image"],
                state["conditional_image"], state["epsilon_1_image"],
                state["epsilon_2_line"], state["epsilon_2_tail"],
                state["heavy_line"], title,
            )

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(tdse_report._save_animation(
            animation, fig, outdir, "nested_factorization_analysis_movie",
            args.fps, args.animation_dpi, args.format,
        ))
    return products, prep


# ---------------------------------------------------------------------------
# 5. Heavy-coordinate force/momentum analysis


def _heavy_preparation(obs, ef_zero, alpha_positive, args):
    R = obs["R"]
    floor = args.support_floor
    heavy_support = (
        obs["heavy_density"]
        >= floor*np.maximum(np.max(obs["heavy_density"], axis=1), 1.0e-300)[:, None]
    )
    epsilon_zero = np.asarray([
        density_weighted_shift(
            ef_zero["epsilon_2"][frame], obs["heavy_density"][frame], floor,
        )
        for frame in range(len(obs["times_fs"]))
    ])
    trap_alpha = float(obs["options"].get("heavy_trap_alpha", 0.0))
    trap_center = float(obs["options"].get(
        "heavy_trap_center",
        0.5*float(obs["options"].get("fixed_ion_separation", 0.0)),
    ))
    trap_potential = trap_alpha*(R-trap_center)**2
    # All three forces live on the same forward R bond as S^Gamma.  Applying
    # one discrete derivative to both the exact TDPES and the explicit trap
    # makes the finite-grid decomposition an identity (including its closing
    # PBC bond), rather than mixing a bond force with a site-centred analytic
    # force.
    total_force = -tdse_report._forward_bond_derivative(
        epsilon_zero, obs["dR"], axis=1,
    )
    harmonic_force = -tdse_report._forward_bond_derivative(
        trap_potential, obs["dR"], axis=0,
    )
    driven_force = total_force-harmonic_force[None, :]
    force_decomposition_max_abs = float(np.max(np.abs(
        total_force-(driven_force+harmonic_force[None, :])
    )))
    requested = (args.heavy_min, args.heavy_max)
    R_limits = _support_limits(
        R, obs["heavy_density"], floor, padding=0.22, requested=requested,
    )
    # Keep the requested 5--15 analysis window whenever it lies on the grid;
    # it is intentionally wider than the dense heavy support so the incoming
    # proton silhouette is visible before and after closest approach.
    requested_limits = (
        max(float(R[0]), float(args.heavy_min)),
        min(float(R[-1]), float(args.heavy_max)),
    )
    if requested_limits[0] < requested_limits[1]:
        R_limits = requested_limits
    frames = _movie_frames(obs, args.max_frames)
    dynamic_force_limits = _symmetric_support_bound(
        [
            values[int(frame)]
            for frame in frames
            for values in (total_force, driven_force)
        ],
        [
            obs["heavy_density"][int(frame)]
            for frame in frames
            for _ in range(2)
        ], floor,
        percentile=99.0,
    )
    view = (R >= R_limits[0]) & (R <= R_limits[1])
    harmonic_bound = (
        float(np.max(np.abs(harmonic_force[view]))) if np.any(view) else 0.0
    )
    force_bound = 1.08*max(
        abs(dynamic_force_limits[0]), abs(dynamic_force_limits[1]),
        harmonic_bound, 1.0e-12,
    )
    alpha_limits = _symmetric_support_bound(
        [alpha_positive[int(frame)] for frame in frames],
        [obs["heavy_density"][int(frame)] for frame in frames], floor,
    )
    alpha_bound = 1.08*max(abs(alpha_limits[0]), abs(alpha_limits[1]), 1.0e-12)
    return {
        "epsilon_zero": epsilon_zero,
        "total_force": total_force,
        "driven_force": driven_force,
        "harmonic_force": harmonic_force,
        "trap_potential": trap_potential,
        "force_decomposition_max_abs": force_decomposition_max_abs,
        "alpha_positive": alpha_positive,
        "heavy_support": heavy_support,
        "R_limits": R_limits,
        "force_limits": (-force_bound, force_bound),
        "alpha_limits": (-alpha_bound, alpha_bound),
        "trap_alpha": trap_alpha,
        "trap_center": trap_center,
    }


def _draw_silhouettes(axis, obs, frame, x_limits, scale=0.24):
    R, q = obs["R"], obs["q"]
    heavy = obs["heavy_density"][frame]
    proton = obs["proton_density"][frame]
    heavy_shape = scale*heavy/max(float(np.max(heavy)), 1.0e-300)
    proton_shape = scale*proton/max(float(np.max(proton)), 1.0e-300)
    q_view = (q >= x_limits[0]) & (q <= x_limits[1])
    R_view = (R >= x_limits[0]) & (R <= x_limits[1])
    transform = axis.get_xaxis_transform()
    heavy_fill = axis.fill_between(
        R[R_view], 0.0, heavy_shape[R_view], transform=transform,
        color=PARTICLE_COLORS["heavy"], alpha=0.20, linewidth=0,
    )
    heavy_line, = axis.plot(
        R[R_view], heavy_shape[R_view], transform=transform,
        color=PARTICLE_COLORS["heavy"], lw=1.8,
        label=r"heavy $\rho_R$ silhouette",
    )
    proton_fill = axis.fill_between(
        q[q_view], 0.0, proton_shape[q_view], transform=transform,
        color=PARTICLE_COLORS["proton"], alpha=0.18, linewidth=0,
    )
    proton_line, = axis.plot(
        q[q_view], proton_shape[q_view], transform=transform,
        color=PARTICLE_COLORS["proton"], lw=1.55,
        label=r"proton $\rho_q$ silhouette",
    )
    return {
        "heavy_fill": heavy_fill, "heavy_line": heavy_line,
        "proton_fill": proton_fill, "proton_line": proton_line,
        "q_view": q_view, "R_view": R_view,
    }


def _draw_heavy_analysis(fig, force_axis, obs, prep, frame, args, *,
                         compact=False):
    R = obs["R"]
    support = prep["heavy_support"][frame]
    total = prep["total_force"][frame]
    driven = prep["driven_force"][frame]
    total_line, total_tail = tdse_report._support_tail_lines(
        force_axis, R, np.where(support, total, np.nan), total, support,
        color="0.10",
        label=r"$F_{\mathrm{total}}=-\partial_R\epsilon_{\mathrm{ZP}}^{(2)}$",
        linewidth=(1.55 if compact else 2.7), linestyle="-",
    )
    driven_line, driven_tail = tdse_report._support_tail_lines(
        force_axis, R, np.where(support, driven, np.nan), driven, support,
        color=FORCE_COLOR,
        label=(
            r"$F_{\mathrm{driven}}="
            r"-\partial_R[\epsilon_{\mathrm{ZP}}^{(2)}-V_{\mathrm{trap}}]$"
        ),
        linewidth=(1.0 if compact else 1.75), linestyle="--",
    )
    harmonic_line, = force_axis.plot(
        R, prep["harmonic_force"], color=COLORS[4],
        lw=(1.0 if compact else 1.75), ls="--",
        label=r"$F_{\mathrm{harm}}=-\partial_RV_{\mathrm{trap}}$",
    )
    force_axis.axhline(0.0, color="0.65", lw=0.7, zorder=0)
    force_axis.axvline(
        prep["trap_center"], color=COLORS[4], lw=0.8, ls=":", alpha=0.75,
    )
    force_axis.set(
        xlim=prep["R_limits"], ylim=prep["force_limits"],
        xlabel=r"heavy coordinate / common position $R$ ($a_0$)",
        ylabel=r"force (Hartree/$a_0$)",
    )
    color_y_axis(force_axis, "0.10", r"force (Hartree/$a_0$)")
    force_axis.grid(alpha=0.16)

    alpha_axis = force_axis.twinx()
    alpha = prep["alpha_positive"][frame]
    alpha_line, alpha_tail = tdse_report._support_tail_lines(
        alpha_axis, R, np.where(support, alpha, np.nan), alpha, support,
        color=COLORS[0], label=r"$\alpha_{\mathrm{PG}}(R,t)=K_R$",
        linewidth=(1.1 if compact else 2.0),
    )
    alpha_axis.set_ylim(prep["alpha_limits"])
    color_y_axis(alpha_axis, COLORS[0], r"$\alpha_{\mathrm{PG}}$ ($a_0^{-1}$)")
    silhouettes = _draw_silhouettes(
        force_axis, obs, frame, prep["R_limits"], scale=(0.20 if compact else 0.25),
    )
    handles = [
        silhouettes["heavy_line"], silhouettes["proton_line"],
        alpha_line, total_line, driven_line, harmonic_line,
    ]
    force_axis.legend(
        handles=handles, frameon=False, fontsize=(5.2 if compact else 8),
        ncol=(2 if compact else 3), loc="upper left",
    )
    force_axis.set_title(
        "Heavy-coordinate momentum and separated force contributions",
        loc="left", fontweight="semibold", fontsize=(7 if compact else None),
    )
    return {
        "force_axis": force_axis, "alpha_axis": alpha_axis,
        "total_line": total_line, "total_tail": total_tail,
        "driven_line": driven_line, "driven_tail": driven_tail,
        "harmonic_line": harmonic_line,
        "alpha_line": alpha_line, "alpha_tail": alpha_tail,
        "silhouettes": silhouettes,
    }


def _update_heavy_analysis(state, obs, prep, frame, compact=False):
    support = prep["heavy_support"][frame]
    total = prep["total_force"][frame]
    driven = prep["driven_force"][frame]
    alpha = prep["alpha_positive"][frame]
    state["total_line"].set_ydata(np.where(support, total, np.nan))
    state["total_tail"].set_ydata(np.where(~support, total, np.nan))
    state["driven_line"].set_ydata(np.where(support, driven, np.nan))
    state["driven_tail"].set_ydata(np.where(~support, driven, np.nan))
    state["alpha_line"].set_ydata(np.where(support, alpha, np.nan))
    state["alpha_tail"].set_ydata(np.where(~support, alpha, np.nan))
    for key in ("heavy_fill", "proton_fill"):
        state["silhouettes"][key].remove()
    old_heavy = state["silhouettes"]["heavy_line"]
    old_proton = state["silhouettes"]["proton_line"]
    old_heavy.remove()
    old_proton.remove()
    state["silhouettes"] = _draw_silhouettes(
        state["force_axis"], obs, frame, prep["R_limits"],
        scale=(0.20 if compact else 0.25),
    )


def render_heavy_analysis(obs, ef_zero, alpha_positive, outdir, args, snapshots):
    prep = _heavy_preparation(obs, ef_zero, alpha_positive, args)
    times = obs["times_fs"]

    def individual(frame):
        fig, axis = plt.subplots(figsize=(13.2, 7.0), constrained_layout=True)
        _draw_heavy_analysis(fig, axis, obs, prep, frame, args)
        fig.suptitle(
            f"Heavy-coordinate analysis | t={times[frame]:.4f} fs\n"
            r"$\alpha_{\rm PG}$: positive gauge; forces: "
            r"$\alpha_{\rm ZP}=0$ gauge; "
            r"$F_{\rm total}=F_{\rm driven}+F_{\rm harm}$",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times, Path(outdir)/"heavy_analysis_frames",
        "heavy_analysis", args.dpi,
    )
    fig = plt.figure(figsize=(22.0, 7.6), constrained_layout=True)
    outer = fig.add_gridspec(2, 4)
    for slot, frame in zip(outer, snapshots):
        axis = fig.add_subplot(slot)
        _draw_heavy_analysis(fig, axis, obs, prep, int(frame), args, compact=True)
        axis.text(
            0.98, 0.92, f"t={times[int(frame)]:.3f} fs",
            transform=axis.transAxes, ha="right", va="top", fontsize=7,
        )
    fig.suptitle(
        "Heavy wavepacket, positive-gauge momentum, and separated forces",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/"heavy_analysis_snapshots.png", args.dpi,
    ))

    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        first = int(frames[0])
        fig, axis = plt.subplots(figsize=(13.2, 7.0), constrained_layout=True)
        state = _draw_heavy_analysis(fig, axis, obs, prep, first, args)
        title = fig.suptitle("", fontweight="bold")

        def update(number):
            frame = int(frames[number])
            _update_heavy_analysis(state, obs, prep, frame)
            title.set_text(
                f"Heavy-coordinate analysis | t={times[frame]:.4f} fs\n"
                r"$\alpha_{\rm PG}=K_R$; "
                r"$F_{\rm total}=-\partial_R\epsilon_{\rm ZP}^{(2)}"
                r"=F_{\rm driven}+F_{\rm harm}$"
            )
            return (
                state["total_line"], state["total_tail"],
                state["driven_line"], state["driven_tail"],
                state["harmonic_line"], state["alpha_line"],
                state["alpha_tail"], state["silhouettes"]["heavy_line"],
                state["silhouettes"]["proton_line"], title,
            )

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(tdse_report._save_animation(
            animation, fig, outdir, "heavy_analysis_movie",
            args.fps, args.animation_dpi, args.format,
        ))
    return products, prep


# ---------------------------------------------------------------------------
# 5. Existing BO panel 1 + panel 4


def _bo_preparation(obs, ef, args):
    energies = obs.get("bo_energies")
    required = ("bo_state_density_q", "bo_state_density_R")
    if energies is None or any(key not in ef for key in required):
        raise KeyError(
            "BO energies/state-resolved densities가 없습니다. "
            "postprocess_tdse_ef를 먼저 실행하세요."
        )
    energies = np.asarray(energies, float)
    density_q = np.asarray(ef["bo_state_density_q"], float)
    populations = np.asarray(obs["bo_populations"], float)
    n_states = min(
        max(1, int(args.surface_count)), energies.shape[0],
        density_q.shape[1], populations.shape[1],
    )
    q = obs["q"]
    q_min, q_max = tdse_report._clipped_q_limits(q, tdse_report._BO_Q_DISPLAY_LIMITS)
    q_mask = (q >= q_min) & (q <= q_max)
    frames = _movie_frames(obs, args.max_frames)
    samples = []
    for frame in frames:
        _, iR = np.unravel_index(
            int(np.argmax(obs["joint_density"][int(frame)])),
            obs["joint_density"][int(frame)].shape,
        )
        values = energies[:n_states, q_mask, iR]
        finite = values[np.isfinite(values)]
        if finite.size:
            samples.append(finite)
    values = np.concatenate(samples) if samples else np.array([-1.0, 1.0])
    lower, upper = np.nanpercentile(values, (1.0, 99.0))
    span = max(float(upper-lower), 1.0e-3)
    energy_limits = (float(lower-0.08*span), float(upper+0.08*span))
    density_max = max(float(np.nanmax(density_q[:, :n_states, q_mask])), 1.0e-14)
    packet_lift = (
        tdse_report._BO_PACKET_VISUAL_AMPLIFICATION*0.34*
        max(energy_limits[1]-energy_limits[0], 1.0e-3)
    )
    display_limits = (energy_limits[0], energy_limits[1]+1.08*packet_lift)
    return {
        "energies": energies, "density_q": density_q,
        "populations": populations, "n_states": n_states,
        "q_mask": q_mask, "q_limits": (q_min, q_max),
        "density_max": density_max, "packet_lift": packet_lift,
        "display_limits": display_limits,
    }


def _draw_bo_combined(fig, q_axis, population_axis, obs, prep, frame, *,
                      compact=False):
    q, R, times = obs["q"], obs["R"], obs["times_fs"]
    iq, iR = np.unravel_index(
        int(np.argmax(obs["joint_density"][frame])),
        obs["joint_density"][frame].shape,
    )
    energy_lines, packet_fills = [], []
    for state in range(prep["n_states"]):
        color = COLORS[state % len(COLORS)]
        surface = prep["energies"][state, :, iR]
        top = (
            surface+prep["packet_lift"]*prep["density_q"][frame, state]
            /prep["density_max"]
        )
        line, = q_axis.plot(
            q, surface, color=color, lw=(1.0 if compact else 1.5),
            label=rf"$E_{state}$",
        )
        fill = q_axis.fill_between(
            q[prep["q_mask"]], surface[prep["q_mask"]], top[prep["q_mask"]],
            color=color, alpha=0.58, edgecolor=color, linewidth=0.4,
        )
        energy_lines.append(line)
        packet_fills.append(fill)
        state_name = "ground" if state == 0 else (
            "first excited" if state == 1 else f"state {state}"
        )
        population_axis.plot(
            times, 100.0*prep["populations"][:, state], color=color,
            lw=(1.0 if compact else 1.55),
            label=rf"$P_{state}$ ({state_name})",
        )
    q_axis.set(
        xlim=prep["q_limits"], ylim=prep["display_limits"],
        xlabel=r"proton $q$ ($a_0$)", ylabel="BO energy (Hartree)",
    )
    q_axis.set_title(
        rf"BO cuts and channel packets | $R_{{peak}}={R[iR]:.3f}$",
        loc="left", fontweight="semibold", fontsize=(7 if compact else None),
    )
    population_marker = population_axis.axvline(times[frame], color="black", lw=1.1)
    population_axis.set(
        xlim=(times[0], times[-1]), ylim=(0.0, 100.0),
        xlabel="time (fs)", ylabel=r"$P_j(t)$ (%)",
    )
    population_axis.set_title(
        "BO-channel population transfer", loc="left", fontweight="semibold",
        fontsize=(7 if compact else None),
    )
    for axis in (q_axis, population_axis):
        axis.grid(alpha=0.16)
        axis.tick_params(labelsize=(5.5 if compact else None), direction="in")
        axis.legend(frameon=False, fontsize=(5 if compact else 7))
    return {
        "q_axis": q_axis, "energy_lines": energy_lines,
        "packet_fills": packet_fills, "population_marker": population_marker,
    }


def _update_bo_combined(state, obs, prep, frame):
    q, R = obs["q"], obs["R"]
    _, iR = np.unravel_index(
        int(np.argmax(obs["joint_density"][frame])),
        obs["joint_density"][frame].shape,
    )
    for index in range(prep["n_states"]):
        color = COLORS[index % len(COLORS)]
        surface = prep["energies"][index, :, iR]
        state["energy_lines"][index].set_ydata(surface)
        state["packet_fills"][index].remove()
        top = (
            surface+prep["packet_lift"]*prep["density_q"][frame, index]
            /prep["density_max"]
        )
        state["packet_fills"][index] = state["q_axis"].fill_between(
            q[prep["q_mask"]], surface[prep["q_mask"]], top[prep["q_mask"]],
            color=color, alpha=0.58, edgecolor=color, linewidth=0.4,
        )
    state["q_axis"].set_title(
        rf"BO cuts and channel packets | $R_{{peak}}={R[iR]:.3f}$",
        loc="left", fontweight="semibold",
    )
    time = obs["times_fs"][frame]
    state["population_marker"].set_xdata([time, time])


def render_bo_combined(obs, ef, outdir, args, snapshots):
    prep = _bo_preparation(obs, ef, args)
    times = obs["times_fs"]

    def individual(frame):
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), constrained_layout=True)
        _draw_bo_combined(fig, axes[0], axes[1], obs, prep, frame)
        fig.suptitle(
            f"BO cut/channel packet and population | t={times[frame]:.4f} fs\n"
            "same logic and fixed lift as tdse_bo_surface_dynamics",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times, Path(outdir)/"bo_combined_frames",
        "bo_combined", args.dpi,
    )
    fig = plt.figure(figsize=(22.0, 7.8), constrained_layout=True)
    outer = fig.add_gridspec(2, 4)
    for slot, frame in zip(outer, snapshots):
        inner = slot.subgridspec(2, 1, hspace=0.14)
        q_axis = fig.add_subplot(inner[0, 0])
        population_axis = fig.add_subplot(inner[1, 0])
        _draw_bo_combined(
            fig, q_axis, population_axis, obs, prep, int(frame), compact=True,
        )
        q_axis.text(
            0.98, 0.90, f"t={times[int(frame)]:.3f} fs",
            transform=q_axis.transAxes, ha="right", va="top", fontsize=6.5,
        )
    fig.suptitle(
        "BO cuts/channel packets and BO-channel populations",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/"bo_combined_snapshots.png", args.dpi,
    ))

    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        first = int(frames[0])
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), constrained_layout=True)
        state = _draw_bo_combined(fig, axes[0], axes[1], obs, prep, first)
        title = fig.suptitle("", fontweight="bold")

        def update(number):
            frame = int(frames[number])
            _update_bo_combined(state, obs, prep, frame)
            title.set_text(
                f"BO cut/channel packet and population | t={times[frame]:.4f} fs\n"
                "same fixed-scale BO logic as tdse_bo_surface_dynamics"
            )
            return (
                *state["energy_lines"], *state["packet_fills"],
                state["population_marker"], title,
            )

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(tdse_report._save_animation(
            animation, fig, outdir, "bo_combined_movie",
            args.fps, args.animation_dpi, args.format,
        ))
    return products


# ---------------------------------------------------------------------------
# 6. Two physical BO-channel densities on the fixed two-dimensional BOPES


def _frame_focus(obs, frame, floor):
    """Bounding box of all occupied branches, with padding in grid cells."""
    density = obs['joint_density'][frame]
    active = np.isfinite(density) & (density >= floor*max(float(np.max(density)), 1e-300))
    limits = []
    indices = []
    for coordinate, occupied in ((obs['q'], np.any(active, axis=1)),
                                 (obs['R'], np.any(active, axis=0))):
        found = np.flatnonzero(occupied)
        if not found.size:
            found = np.arange(len(coordinate))
        pad = max(2, int(np.ceil(0.08*(found[-1]-found[0]+1))))
        start, stop = max(0, found[0]-pad), min(len(coordinate), found[-1]+pad+1)
        indices.append(np.arange(start, stop))
        limits.append((float(coordinate[start]), float(coordinate[stop-1])))
    return active, indices, limits


def _save_analysis_movie(animation, fig, outdir, stem, args):
    """Encode dense scientific plots sharply without the former slow preset."""
    if args.format == 'mp4' and FFMpegWriter.isAvailable():
        path = Path(outdir)/f'{stem}.mp4'
        writer = FFMpegWriter(fps=args.fps, codec='libx264', bitrate=-1,
                             extra_args=['-crf', '18', '-preset',
                                         getattr(args, 'movie_preset', 'medium'),
                                         '-pix_fmt', 'yuv420p', '-vf',
                                         'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
        # 120 dpi on the 16.5 x 9.2 inch analysis canvas is already near 1080p.
        # Higher values remain available through --animation-dpi.
        animation.save(path, writer=writer, dpi=max(120, args.animation_dpi))
        plt.close(fig)
        return path
    return tdse_report._save_animation(animation, fig, outdir, stem,
                                      args.fps, args.animation_dpi, args.format)


def _bo3d_preparation(obs, ef, args):
    energies = np.asarray(obs.get("bo_energies"), float)
    channel = np.asarray(ef.get("bo_channel_density_qR"), float)
    if energies.ndim != 3 or channel.ndim != 4:
        raise KeyError(
            "2D BO channel density가 없습니다. postprocess_tdse_ef를 "
            "--channel-density-states 2 --overwrite로 다시 실행하세요."
        )
    n_states = min(2, int(args.surface_count), energies.shape[0], channel.shape[1])
    q_limits = _support_limits(
        obs["q"], obs["proton_density"], args.support_floor, padding=0.08,
    )
    R_limits = _support_limits(
        obs["R"], obs["heavy_density"], args.support_floor, padding=0.12,
    )
    q_indices = np.flatnonzero((obs["q"] >= q_limits[0]) & (obs["q"] <= q_limits[1]))
    R_indices = np.flatnonzero((obs["R"] >= R_limits[0]) & (obs["R"] <= R_limits[1]))
    q_indices = q_indices[np.linspace(
        0, len(q_indices)-1, min(len(q_indices), args.bo3d_q_points), dtype=int,
    )]
    R_indices = R_indices[np.linspace(
        0, len(R_indices)-1, min(len(R_indices), args.bo3d_R_points), dtype=int,
    )]
    sampled_energy = energies[:n_states][:, q_indices][:, :, R_indices]
    finite = sampled_energy[np.isfinite(sampled_energy)]
    lo, hi = np.percentile(finite, (1.0, 99.0))
    span = max(float(hi-lo), 1.0e-3)
    density_max = max(float(np.max(channel[:, :n_states])), 1.0e-300)
    return {
        "energies": energies, "channel": channel, "n_states": n_states,
        "q_indices": q_indices, "R_indices": R_indices,
        "energy_limits": (float(lo-0.06*span), float(hi+0.42*span)),
        "packet_lift": 0.30*span, "density_max": density_max,
        "q_limits": q_limits, "R_limits": R_limits,
        "floor": float(args.support_floor),
        "focus_floor": args.analysis_focus_floor,
        "q_points": args.bo3d_q_points,
        "R_points": args.bo3d_R_points,
        "movie_q_points": args.movie_bo3d_q_points,
        "movie_R_points": args.movie_bo3d_R_points,
    }


def _draw_bo3d_axis(axis, obs, prep, frame, states, compact=False):
    active, indices, limits = _frame_focus(obs, frame, prep['focus_floor'])
    qi, Ri = indices
    qi = qi[np.linspace(0, len(qi)-1, min(len(qi), prep['q_points']), dtype=int)]
    Ri = Ri[np.linspace(0, len(Ri)-1, min(len(Ri), prep['R_points']), dtype=int)]
    Q, RR = np.meshgrid(obs["q"][qi], obs["R"][Ri], indexing="ij")
    for state in states:
        energy = prep["energies"][state][np.ix_(qi, Ri)]
        density = prep["channel"][frame, state][np.ix_(qi, Ri)]
        relative = density/prep["density_max"]
        axis.plot_surface(
            Q, RR, energy, color=COLORS[state % len(COLORS)], alpha=0.18,
            linewidth=0.0, antialiased=True, shade=False,
            rcount=len(qi), ccount=len(Ri),
        )
        occupied = active[np.ix_(qi, Ri)] & (relative >= prep['floor'])
        lifted = np.where(
            occupied, energy+prep["packet_lift"]*np.sqrt(relative), np.nan,
        )
        face = plt.get_cmap(JOINT_CMAP)(np.clip(relative**0.22, 0.0, 1.0))
        face[..., 3] = np.where(occupied, 0.35+0.65*np.clip(relative**0.18, 0, 1), 0)
        axis.plot_surface(
            Q, RR, lifted, facecolors=face, linewidth=0.0,
            antialiased=True, shade=False, rcount=len(qi), ccount=len(Ri),
        )
    axis.set(
        xlim=limits[0], ylim=limits[1],
        zlim=prep["energy_limits"], xlabel=r"proton $q$ ($a_0$)",
        ylabel=r"heavy $R$ ($a_0$)", zlabel="BO energy (Hartree)",
    )
    axis.view_init(elev=29, azim=-132)
    axis.tick_params(labelsize=(5 if compact else 7), pad=0)
    state_text = ", ".join(rf"$j={j}$" for j in states)
    axis.set_title(
        state_text+r": fixed BOPES + physical $\rho_j$ (display lift)",
        fontsize=(6 if compact else 9), pad=(1 if compact else 5),
    )


def _draw_bo3d(fig, axes, obs, prep, frame, compact=False):
    if len(axes) == prep["n_states"]:
        for state, axis in enumerate(axes):
            _draw_bo3d_axis(axis, obs, prep, frame, (state,), compact)
    else:
        _draw_bo3d_axis(axes[0], obs, prep, frame, range(prep["n_states"]), compact)


def _setup_bo3d_movie_axis(axis, obs, prep, states):
    """Draw time-independent BO surfaces once and return dynamic state IDs."""
    qi, Ri = prep["q_indices"], prep["R_indices"]
    Q, RR = np.meshgrid(obs["q"][qi], obs["R"][Ri], indexing="ij")
    for state in states:
        energy = prep["energies"][state][np.ix_(qi, Ri)]
        axis.plot_surface(
            Q, RR, energy, color=COLORS[state % len(COLORS)], alpha=0.18,
            linewidth=0.0, antialiased=False, shade=False,
            rcount=len(qi), ccount=len(Ri),
        )
    axis.set(
        zlim=prep["energy_limits"], xlabel=r"proton $q$ ($a_0$)",
        ylabel=r"heavy $R$ ($a_0$)", zlabel="BO energy (Hartree)",
    )
    axis.view_init(elev=29, azim=-132)
    axis.tick_params(labelsize=7, pad=0)
    state_text = ", ".join(rf"$j={j}$" for j in states)
    axis.set_title(
        state_text+r": fixed BOPES + physical $\rho_j$ (display lift)",
        fontsize=9, pad=5,
    )
    return tuple(states)


def _draw_bo3d_movie_packets(axis, obs, prep, frame, states):
    """Draw only the moving density surfaces on an already initialized axis."""
    active, indices, limits = _frame_focus(obs, frame, prep["focus_floor"])
    qi, Ri = indices
    qi = qi[np.linspace(
        0, len(qi)-1, min(len(qi), prep["movie_q_points"]), dtype=int,
    )]
    Ri = Ri[np.linspace(
        0, len(Ri)-1, min(len(Ri), prep["movie_R_points"]), dtype=int,
    )]
    Q, RR = np.meshgrid(obs["q"][qi], obs["R"][Ri], indexing="ij")
    packets = []
    for state in states:
        energy = prep["energies"][state][np.ix_(qi, Ri)]
        density = prep["channel"][frame, state][np.ix_(qi, Ri)]
        relative = density/prep["density_max"]
        occupied = active[np.ix_(qi, Ri)] & (relative >= prep["floor"])
        lifted = np.where(
            occupied, energy+prep["packet_lift"]*np.sqrt(relative), np.nan,
        )
        face = plt.get_cmap(JOINT_CMAP)(np.clip(relative**0.22, 0.0, 1.0))
        face[..., 3] = np.where(
            occupied, 0.35+0.65*np.clip(relative**0.18, 0, 1), 0,
        )
        packets.append(axis.plot_surface(
            Q, RR, lifted, facecolors=face, linewidth=0.0,
            antialiased=False, shade=False, rcount=len(qi), ccount=len(Ri),
        ))
    axis.set_xlim(limits[0])
    axis.set_ylim(limits[1])
    return packets


def render_bo3d_channels(obs, ef, outdir, args, snapshots):
    prep = _bo3d_preparation(obs, ef, args)
    times = obs["times_fs"]
    def individual(frame):
        fig = plt.figure(figsize=(15.0, 6.5), constrained_layout=True)
        axes = [fig.add_subplot(1, prep["n_states"], j+1, projection="3d")
                for j in range(prep["n_states"])]
        _draw_bo3d(fig, axes, obs, prep, frame)
        fig.suptitle(
            f"Physical BO-channel density on fixed BOPES | t={times[frame]:.4f} fs\n"
            r"$\rho_j(q,R,t)=\rho_{qR}|C_j|^2=|Y_j|^2$; vertical lift is display-only",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times, Path(outdir)/"bo_3d_channel_frames",
        "bo_3d_channel_dynamics", args.dpi,
    )
    fig = plt.figure(figsize=(24.0, 13.0), constrained_layout=True)
    for slot, frame in zip(fig.add_gridspec(2, 4), snapshots):
        axis = fig.add_subplot(slot, projection="3d")
        _draw_bo3d(fig, [axis], obs, prep, int(frame), compact=True)
        axis.text2D(
            0.98, 0.94, f"t={times[int(frame)]:.3f} fs",
            transform=axis.transAxes, ha="right", va="top", fontsize=6,
        )
    fig.suptitle(
        r"BO surfaces are time independent; physical channel densities $\rho_j$ redistribute",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/"bo_3d_channel_dynamics_snapshots.png", args.dpi,
    ))
    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        fig = plt.figure(figsize=(15.0, 6.5), constrained_layout=True)
        axes = [fig.add_subplot(1, prep["n_states"], j+1, projection="3d")
                for j in range(prep["n_states"])]
        title = fig.suptitle("", fontweight="bold")
        axis_states = []
        for state, axis in enumerate(axes):
            axis_states.append(_setup_bo3d_movie_axis(axis, obs, prep, (state,)))
        packet_artists = [[] for _ in axes]

        def update(number):
            frame = int(frames[number])
            for index, (axis, states) in enumerate(zip(axes, axis_states)):
                for artist in packet_artists[index]:
                    artist.remove()
                packet_artists[index] = _draw_bo3d_movie_packets(
                    axis, obs, prep, frame, states,
                )
            title.set_text(
                f"Physical BO-channel density on fixed BOPES | t={times[frame]:.4f} fs\n"
                r"$\rho_j=|Y_j|^2$; fixed trajectory scale; display lift only"
            )
            return (*[item for group in packet_artists for item in group], title)

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(_save_analysis_movie(
            animation, fig, outdir, "bo_3d_channel_dynamics_movie", args,
        ))
    return products, prep


# ---------------------------------------------------------------------------
# 7. First-level TDPES origin: exact discrete pieces and link-metric limit


def _site_link_metric(link, spacing, axis):
    """Site-centred continuum-limit metric from adjacent overlap magnitudes."""
    forward = (1.0-np.abs(link)**2)/(float(spacing)**2)
    return 0.5*(forward+np.roll(forward, 1, axis=axis))


def _tdpes1_origin_frame(obs, ef_zero, prep, frame):
    density = obs["joint_density"][frame]
    native_total = np.asarray(ef_zero["epsilon_1"][frame], float)
    native_gi = np.asarray(ef_zero["epsilon_1_gi"][frame], float)
    wbo_raw = np.asarray(ef_zero["epsilon_1_wbo"][frame], float)
    support = density >= prep["floor"]*max(float(np.max(density)), 1.0e-300)
    stored = all(key in ef_zero for key in (
        "tdpes1_total", "tdpes1_wbo_0", "tdpes1_wbo_excited",
        "tdpes1_gd", "tdpes1_geo_q", "tdpes1_geo_R",
    ))
    if stored:
        # New caches are self-contained: plotting reads all six raw fields
        # calculated at EF-postprocessing time and performs no physical
        # reconstruction.
        total_raw = np.asarray(ef_zero["tdpes1_total"][frame], float)
        ground_raw = np.asarray(ef_zero["tdpes1_wbo_0"][frame], float)
        excited_raw = np.asarray(
            ef_zero["tdpes1_wbo_excited"][frame], float,
        )
        gd_raw = np.asarray(ef_zero["tdpes1_gd"][frame], float)
        geo_q = np.asarray(ef_zero["tdpes1_geo_q"][frame], float)
        geo_R = np.asarray(ef_zero["tdpes1_geo_R"][frame], float)
        wbo_raw = ground_raw+excited_raw
    else:
        # Compatibility for older caches.  Regenerate the cache to eliminate
        # this plotting-time continuum-limit reconstruction.
        gd_raw = native_total-native_gi
        geo_q = _site_link_metric(
            ef_zero["sphi_q1"][frame], obs["dq"], axis=0,
        )/(2.0*prep["proton_mass"])
        geo_R = _site_link_metric(
            ef_zero["sphi_R1"][frame], obs["dR"], axis=1,
        )/(2.0*prep["heavy_mass"])
        total_raw = wbo_raw+geo_q+geo_R+gd_raw
        ground_raw = None
        excited_raw = None
    # Preparation fixes this value once from frame zero.  The fallback is used
    # only by that bootstrap call, so every subsequently rendered frame keeps
    # the same physical energy zero and retains genuine temporal offsets.
    reference_mode = prep.get("energy_reference_mode", "initial")
    energy_reference = (
        prep.get("fixed_energy_reference")
        if reference_mode == "initial" else None
    )
    if energy_reference is None:
        energy_reference = (
            np.average(total_raw[support], weights=density[support])
            if np.any(support) else 0.0
        )
    energy_reference = float(energy_reference)
    total = total_raw-energy_reference

    # Use physical channel density / joint density, not global populations.
    # Apply the same energy reference to every BO surface.  The second plotted
    # BO panel is the complete j>=1 sector, so the six displayed panels remain
    # an exact identity even when the propagation retained more than 2 states.
    channels = np.asarray(ef_zero["bo_channel_density_qR"][frame, :2], float)
    weights = np.divide(channels, density[None], out=np.zeros_like(channels),
                        where=density[None] > 0)
    energies = np.asarray(obs["bo_energies"][:2], float)
    ground = (
        ground_raw-weights[0]*energy_reference
        if ground_raw is not None else
        weights[0]*(energies[0]-energy_reference)
    )
    first_excited = weights[1]*(energies[1]-energy_reference)
    wbo = wbo_raw-energy_reference
    excited_sector = (
        excited_raw-(1.0-weights[0])*energy_reference
        if excited_raw is not None else wbo-ground
    )
    higher_bo = excited_sector-first_excited
    identity_residual = total-(
        ground+excited_sector+gd_raw+geo_q+geo_R
    )
    return {
        "total": total, "wbo": wbo,
        "wbo_1": ground, "wbo_2": excited_sector,
        "wbo_2_pure": first_excited, "wbo_higher": higher_bo,
        "native_gi": native_gi, "total_raw": total_raw,
        "gi_limit": wbo+geo_q+geo_R,
        "native_total": native_total, "gd_raw": gd_raw,
        "geo_q": geo_q, "geo_R": geo_R, "gd": gd_raw,
        "energy_reference": energy_reference,
        "identity_residual": identity_residual,
        "stored_decomposition": stored,
        "opacity": density_display_alpha(density, floor=prep["floor"]),
        "joint_log": np.log10(np.maximum(density/max(float(np.max(density)), 1e-300), 1e-300)),
    }


def _tdpes1_origin_preparation(obs, ef_zero, args):
    if ("bo_channel_density_qR" not in ef_zero
            or ef_zero["bo_channel_density_qR"].shape[1] < 2
            or np.asarray(obs.get("bo_energies")).ndim != 3
            or len(obs["bo_energies"]) < 2):
        raise ValueError("TDPES channel panels require two BO energies and "
                         "bo_channel_density_qR; rebuild EF cache with "
                         "--channel-density-states 2 --overwrite")
    floor = float(args.support_floor)
    proton_mass = float(obs["options"].get("proton_mass", 1836.15267343))
    heavy_mass = float(obs["options"].get("heavy_mass", 1836.15267343))
    provisional = {
        "floor": floor, "proton_mass": proton_mass, "heavy_mass": heavy_mass,
        "energy_reference_mode": getattr(
            args, "tdpes_energy_reference", "initial",
        ),
        "decades": float(args.decades),
        "focus_floor": getattr(args, 'analysis_focus_floor', 1e-3),
        "q_limits": _support_limits(
            obs["q"], obs["proton_density"], floor, padding=0.08,
        ),
        "R_limits": _support_limits(
            obs["R"], obs["heavy_density"], floor, padding=0.12,
        ),
        "contour_q_points": int(getattr(args, "tdpes_contour_q_points", 180)),
        "contour_R_points": int(getattr(args, "tdpes_contour_R_points", 120)),
        "stored_decomposition": all(key in ef_zero for key in (
            "tdpes1_total", "tdpes1_wbo_0", "tdpes1_wbo_excited",
            "tdpes1_gd", "tdpes1_geo_q", "tdpes1_geo_R",
        )),
    }
    if provisional["energy_reference_mode"] == "initial":
        initial = _tdpes1_origin_frame(obs, ef_zero, provisional, 0)
        provisional["fixed_energy_reference"] = initial["energy_reference"]
    samples, geo_samples = [], []
    identity_errors, higher_bo_sizes = [], []
    for frame in _movie_frames(obs, min(
        args.max_frames, getattr(args, "scale_sample_frames", 32),
    )):
        current = _tdpes1_origin_frame(obs, ef_zero, provisional, int(frame))
        support = obs['joint_density'][int(frame)] >= provisional['focus_floor']*np.max(obs['joint_density'][int(frame)])
        for key in ("total", "wbo_1", "wbo_2", "gd"):
            values = np.abs(current[key][support & np.isfinite(current[key])])
            if values.size:
                samples.append(float(np.percentile(values, 99.0)))
        for key in ("geo_q", "geo_R"):
            values = current[key][support & np.isfinite(current[key])]
            if values.size:
                geo_samples.append(float(np.percentile(values, 99.0)))
        if np.any(support):
            identity_errors.append(float(np.max(np.abs(
                current["identity_residual"][support]
            ))))
            higher_bo_sizes.append(float(np.max(np.abs(
                current["wbo_higher"][support]
            ))))
    signed_bound = max(max(samples, default=0.0), 1.0e-10)
    geo_all = np.asarray(geo_samples) if geo_samples else np.array([0.0, 1.0e-10])
    geo_bound = max(float(np.max(np.abs(geo_all))), 1.0e-10)
    geo_limits = (-geo_bound, geo_bound)
    common_bound = max(signed_bound, geo_bound)
    max_identity_residual = max(identity_errors, default=0.0)
    closure_tolerance = 256.0*np.finfo(np.float64).eps*max(
        common_bound, 1.0,
    )
    if max_identity_residual > closure_tolerance:
        raise RuntimeError(
            "displayed TDPES decomposition does not close: "
            f"max residual={max_identity_residual:.6e}, "
            f"tolerance={closure_tolerance:.6e}"
        )
    provisional.update({
        "signed_bound": signed_bound, "geo_limits": geo_limits,
        "common_bound": common_bound,
        "linear_threshold": max(1.0e-2*common_bound, 1.0e-12),
        "color_scale": getattr(args, "tdpes_color_scale", "linear"),
        "max_identity_residual": max_identity_residual,
        "identity_tolerance": closure_tolerance,
        "max_higher_bo_contribution": max(higher_bo_sizes, default=0.0),
    })
    return provisional


_TDPES1_KEYS = ("total", "wbo_1", "wbo_2", "gd", "geo_q", "geo_R")
_TDPES1_TITLES = (
    r"Total $\widetilde\epsilon_{\rm total}^{(1)}$ (stored EF decomposition)",
    r"$\epsilon_{\rm wBO,1}^{(1)}=|C_0|^2(E_0^{\rm BO}-E_{\rm ref})$",
    r"$\epsilon_{\rm wBO,2+}^{(1)}=\sum_{j\geq1}|C_j|^2(E_j^{\rm BO}-E_{\rm ref})$",
    r"Gauge dependent $\epsilon_{\rm GD}^{(1)}$",
    r"Proton geometry $\epsilon_{q,\rm geo}^{(1)}$ (link-metric limit)",
    r"Heavy geometry $\epsilon_{R,\rm geo}^{(1)}$ (link-metric limit)",
)


def _tdpes1_shared_norm(prep):
    """One zero-centred norm shared by all six energy contributions."""
    bound = prep["common_bound"]
    if prep.get("color_scale", "linear") == "linear":
        return Normalize(-bound, bound)
    return SymLogNorm(
        linthresh=prep["linear_threshold"], linscale=0.65,
        vmin=-bound, vmax=bound, base=10,
    )


def _tdpes1_colorbar(fig, prep, cax):
    """Draw the shared TDPES scale in a dedicated, layout-stable axis."""
    scale = "symmetric log" if prep.get("color_scale") == "symlog" else "linear"
    colorbar = fig.colorbar(
        ScalarMappable(norm=_tdpes1_shared_norm(prep), cmap=SIGNED_CMAP),
        cax=cax, extend="both", format=NUMBER_FORMATTER,
    )
    colorbar.set_label(
        f"energy contribution (Hartree; shared {scale} scale)",
        labelpad=9,
    )
    colorbar.ax.tick_params(labelsize=7, pad=3)
    return colorbar


def _draw_tdpes1_origin(fig, axes, obs, ef_zero, prep, frame, colorbars=True,
                        compact=False, colorbar_axis=None):
    current = _tdpes1_origin_frame(obs, ef_zero, prep, frame)
    keys = _TDPES1_KEYS
    active, indices, limits = _frame_focus(obs, frame, prep['focus_floor'])
    qi, Ri = indices
    active_crop = active[np.ix_(qi, Ri)]
    images, contours = [], []
    extent = [obs["q"][qi[0]], obs["q"][qi[-1]],
              obs["R"][Ri[0]], obs["R"][Ri[-1]]]
    contour_qi = qi[np.linspace(
        0, len(qi)-1, min(len(qi), prep["contour_q_points"]), dtype=int,
    )]
    contour_Ri = Ri[np.linspace(
        0, len(Ri)-1, min(len(Ri), prep["contour_R_points"]), dtype=int,
    )]
    shared_norm = _tdpes1_shared_norm(prep)
    titles = list(_TDPES1_TITLES)
    if not prep.get("stored_decomposition", False):
        titles[0] = (
            r"Total $\widetilde\epsilon_{\rm total}^{(1)}$ "
            r"(legacy plotting reconstruction)"
        )
    for index, (axis, key, title) in enumerate(zip(axes, keys, titles)):
        cmap = masked_cmap(SIGNED_CMAP)
        image = axis.imshow(
            np.ma.masked_where(
                ~active_crop | ~np.isfinite(current[key][np.ix_(qi, Ri)]),
                current[key][np.ix_(qi, Ri)],
            ).T,
            origin="lower", aspect="auto", extent=extent,
            cmap=cmap, norm=shared_norm, interpolation="nearest",
        )
        contours.append(_joint_contours(
            axis, obs, current["joint_log"][np.ix_(contour_qi, contour_Ri)],
            -np.log10(prep['focus_floor']), q=obs["q"][contour_qi],
            R=obs["R"][contour_Ri], color="black", halo_color="white",
            compact=compact,
        ))
        axis.set_facecolor(MASK_COLOR)
        axis.set_title(title, loc="left", fontweight="semibold",
                       fontsize=(5.5 if compact else 8.5))
        axis.set_xlabel(r"proton $q$ ($a_0$)")
        axis.set_ylabel(r"heavy $R$ ($a_0$)")
        axis.set_xlim(limits[0])
        axis.set_ylim(limits[1])
        axis.tick_params(labelsize=(5 if compact else 7), direction="in")
        images.append(image)
    if colorbars:
        if colorbar_axis is None:
            raise ValueError("TDPES origin colorbar requires a dedicated axis")
        _tdpes1_colorbar(fig, prep, colorbar_axis)
    return {"images": images, "contours": contours}


def _update_tdpes1_origin(state, axes, obs, ef_zero, prep, frame):
    """Update image buffers and density contours without rebuilding axes."""
    current = _tdpes1_origin_frame(obs, ef_zero, prep, frame)
    keys = _TDPES1_KEYS
    active, indices, limits = _frame_focus(obs, frame, prep["focus_floor"])
    qi, Ri = indices
    active_crop = active[np.ix_(qi, Ri)]
    extent = [obs["q"][qi[0]], obs["q"][qi[-1]],
              obs["R"][Ri[0]], obs["R"][Ri[-1]]]
    contour_qi = qi[np.linspace(
        0, len(qi)-1, min(len(qi), prep["contour_q_points"]), dtype=int,
    )]
    contour_Ri = Ri[np.linspace(
        0, len(Ri)-1, min(len(Ri), prep["contour_R_points"]), dtype=int,
    )]
    artists = []
    for index, (axis, image, key) in enumerate(zip(axes, state["images"], keys)):
        cropped = current[key][np.ix_(qi, Ri)]
        image.set_data(np.ma.masked_where(
            ~active_crop | ~np.isfinite(cropped), cropped,
        ).T)
        image.set_extent(extent)
        axis.set_xlim(limits[0])
        axis.set_ylim(limits[1])
        for collection in state["contours"][index].collections:
            collection.remove()
        state["contours"][index] = _joint_contours(
            axis, obs, current["joint_log"][np.ix_(contour_qi, contour_Ri)],
            -np.log10(prep["focus_floor"]), q=obs["q"][contour_qi],
            R=obs["R"][contour_Ri], color="black", halo_color="white",
        )
        artists.append(image)
        artists.extend(state["contours"][index].collections)
    return artists


def _tdpes1_origin_axes(fig, slot=None):
    """Six equally sized panels with a reserved colorbar margin.

    The full-frame layout deliberately reserves the right margin for a colorbar.
    Letting constrained-layout attach one colorbar to axes from two nested
    GridSpecs can collapse both panel rows to nearly zero height.
    """
    if slot is None:
        grid = fig.add_gridspec(
            2, 1, left=0.055, right=0.915, bottom=0.075, top=0.855,
            hspace=0.34,
        )
    else:
        grid = slot.subgridspec(2, 1, hspace=0.30)
    top = grid[0].subgridspec(1, 3, wspace=0.27)
    bottom = grid[1].subgridspec(1, 3, wspace=0.27)
    return [fig.add_subplot(top[i]) for i in range(3)] + [
        fig.add_subplot(bottom[i]) for i in range(3)
    ]


def _tdpes1_full_colorbar_axis(fig):
    """Fixed colorbar slot shared by the six full-size TDPES panels."""
    return fig.add_axes((0.938, 0.145, 0.014, 0.645))


def render_tdpes1_origin(obs, ef_fields, outdir, args, snapshots, *,
                         stem="tdpes1_origin_positive_gauge",
                         gauge_label="positive-density gauge"):
    if ef_fields.get("gauge") != "positive_density":
        raise ValueError(
            "final TDPES1 visualization requires positive-density gauge"
        )
    prep = _tdpes1_origin_preparation(obs, ef_fields, args)
    if prep["energy_reference_mode"] == "framewise":
        stem += "_framewise_reference"
    reference_label = (
        "fixed t=0 density-weighted energy origin"
        if prep["energy_reference_mode"] == "initial"
        else "framewise density-weighted energy origin"
    )
    times = obs["times_fs"]

    def individual(frame):
        fig = plt.figure(figsize=(16.5, 9.2), constrained_layout=False)
        axes = _tdpes1_origin_axes(fig)
        cax = _tdpes1_full_colorbar_axis(fig)
        _draw_tdpes1_origin(
            fig, axes, obs, ef_fields, prep, frame,
            colorbar_axis=cax,
        )
        fig.suptitle(
            f"Origin of first-level TDPES structure | t={times[frame]:.4f} fs\n"
            f"{gauge_label}; {reference_label}; "
            "contours are occupied physical density; "
            "link metrics are continuum-limit diagnostics",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times, Path(outdir)/f"{stem}_frames",
        stem, args.dpi,
    )
    fig = plt.figure(figsize=(28.0, 16.0), constrained_layout=False)
    summary_grid = fig.add_gridspec(
        2, 4, left=0.025, right=0.945, bottom=0.045, top=0.925,
        wspace=0.16, hspace=0.20,
    )
    for slot, frame in zip(summary_grid, snapshots):
        axes = _tdpes1_origin_axes(fig, slot)
        _draw_tdpes1_origin(fig, axes, obs, ef_fields, prep, int(frame),
                            colorbars=False, compact=True)
        axes[0].text(0.98, 0.92, f"t={times[int(frame)]:.3f} fs",
                     transform=axes[0].transAxes, ha="right", va="top",
                     fontsize=5.5)
    summary_cax = fig.add_axes((0.958, 0.16, 0.009, 0.65))
    _tdpes1_colorbar(fig, prep, summary_cax)
    fig.suptitle(
        f"First-level TDPES origin ({gauge_label}; {reference_label}): "
        "8 representative times",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/f"{stem}_snapshots.png", args.dpi,
    ))
    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        fig = plt.figure(figsize=(16.5, 9.2), constrained_layout=False)
        axes = _tdpes1_origin_axes(fig)
        cax = _tdpes1_full_colorbar_axis(fig)
        title = fig.suptitle("", fontweight="bold")
        _tdpes1_colorbar(fig, prep, cax)
        state = _draw_tdpes1_origin(
            fig, axes, obs, ef_fields, prep, int(frames[0]), colorbars=False,
        )

        def update(number):
            frame = int(frames[number])
            artists = _update_tdpes1_origin(
                state, axes, obs, ef_fields, prep, frame,
            )
            title.set_text(
                f"Origin of first-level TDPES structure | t={times[frame]:.4f} fs\n"
                f"{gauge_label}; {reference_label}; "
                "displayed decomposition closes exactly"
            )
            return (*artists, title)

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(_save_analysis_movie(
            animation, fig, outdir, f"{stem}_movie", args,
        ))
    return products, prep


# ---------------------------------------------------------------------------
# 8. Second-level TDPES origin on the heavy coordinate


def _frame_heavy_focus(obs, frame, floor):
    """Return the padded heavy-density support used as the live R window."""
    density = np.asarray(obs["heavy_density"][frame], float)
    active = np.isfinite(density) & (
        density >= float(floor)*max(float(np.max(density)), 1.0e-300)
    )
    found = np.flatnonzero(active)
    if not found.size:
        found = np.arange(len(obs["R"]))
    pad = max(2, int(np.ceil(0.12*(found[-1]-found[0]+1))))
    start = max(0, int(found[0])-pad)
    stop = min(len(obs["R"]), int(found[-1])+pad+1)
    return active, np.arange(start, stop), (
        float(obs["R"][start]), float(obs["R"][stop-1]),
    )


def _tdpes2_origin_frame(obs, ef_positive, prep, frame):
    """Exact displayed continuum-limit decomposition of epsilon^(2).

    At finite spacing the native second GI scalar contains the BO average and
    the complete internal q kinetic/link contribution.  The outer R metric is
    carried by S^Gamma and is restored explicitly for the continuum-limit
    diagnostic.  One scalar E_ref fixed from the occupied density at t=0 is
    subtracted from total and every BO energy at all times, leaving the
    geometric and GD terms unchanged.
    """
    joint = np.asarray(obs["joint_density"][frame], float)
    heavy = np.asarray(obs["heavy_density"][frame], float)
    conditional = np.divide(
        joint, heavy[None, :], out=np.zeros_like(joint),
        where=heavy[None, :] > np.finfo(np.float64).tiny,
    )
    native_total = np.asarray(ef_positive["epsilon_2"][frame], float)
    native_gi = np.asarray(ef_positive["epsilon_2_gi"][frame], float)
    stored = all(key in ef_positive for key in (
        "tdpes2_total", "tdpes2_wbo_0", "tdpes2_wbo_excited",
        "tdpes2_gd", "tdpes2_geo_q", "tdpes2_geo_R",
    ))
    if stored:
        total_raw = np.asarray(ef_positive["tdpes2_total"][frame], float)
        ground_raw = np.asarray(ef_positive["tdpes2_wbo_0"][frame], float)
        excited_raw = np.asarray(
            ef_positive["tdpes2_wbo_excited"][frame], float,
        )
        gd = np.asarray(ef_positive["tdpes2_gd"][frame], float)
        geo_q = np.asarray(ef_positive["tdpes2_geo_q"][frame], float)
        geo_R = np.asarray(ef_positive["tdpes2_geo_R"][frame], float)
        wbo_raw = ground_raw+excited_raw
    else:
        gd = native_total-native_gi
        wbo_raw = np.sum(
            conditional*np.asarray(
                ef_positive["epsilon_1_wbo"][frame], float,
            ), axis=0, dtype=np.float64,
        )*obs["dq"]
        # Exact native internal-q contribution plus the continuum outer-R
        # metric.  This branch exists only for legacy caches.
        geo_q = native_gi-wbo_raw
        geo_R = _site_link_metric(
            ef_positive["sgamma_R1"][frame], obs["dR"], axis=0,
        )/(2.0*prep["heavy_mass"])
        total_raw = wbo_raw+geo_q+geo_R+gd
        ground_raw = None
        excited_raw = None
    support = heavy >= prep["floor"]*max(float(np.max(heavy)), 1.0e-300)
    reference_mode = prep.get("energy_reference_mode", "initial")
    energy_reference = (
        prep.get("fixed_energy_reference")
        if reference_mode == "initial" else None
    )
    if energy_reference is None:
        energy_reference = (
            np.average(total_raw[support], weights=heavy[support])
            if np.any(support) else 0.0
        )
    energy_reference = float(energy_reference)
    total = total_raw-energy_reference
    wbo = wbo_raw-energy_reference

    channels = np.asarray(
        ef_positive["bo_channel_density_qR"][frame, :2], float,
    )
    energies = np.asarray(obs["bo_energies"][:2], float)
    ground_from_channels = np.sum(
        np.divide(
            channels[0], heavy[None, :], out=np.zeros_like(channels[0]),
            where=heavy[None, :] > np.finfo(np.float64).tiny,
        )*(energies[0]-energy_reference),
        axis=0, dtype=np.float64,
    )*obs["dq"]
    ground_probability = np.sum(
        np.divide(
            channels[0], heavy[None, :], out=np.zeros_like(channels[0]),
            where=heavy[None, :] > np.finfo(np.float64).tiny,
        ), axis=0, dtype=np.float64,
    )*obs["dq"]
    ground = (
        ground_raw-ground_probability*energy_reference
        if ground_raw is not None else ground_from_channels
    )
    first_excited = np.sum(
        np.divide(
            channels[1], heavy[None, :], out=np.zeros_like(channels[1]),
            where=heavy[None, :] > np.finfo(np.float64).tiny,
        )*(energies[1]-energy_reference),
        axis=0, dtype=np.float64,
    )*obs["dq"]
    excited_sector = (
        excited_raw-(1.0-ground_probability)*energy_reference
        if excited_raw is not None else wbo-ground
    )
    higher_bo = excited_sector-first_excited
    gi = ground+excited_sector+geo_q+geo_R

    # Dotted reference curves: bare BO surfaces averaged only over the
    # conditional proton density, without electronic-channel weighting.
    bo_reference = np.sum(
        conditional[None, :, :]*(energies-energy_reference),
        axis=1, dtype=np.float64,
    )*obs["dq"]
    identity_residual = total-(gi+gd)
    return {
        "total": total, "gi": gi, "wbo": wbo,
        "wbo_1": ground, "wbo_2": excited_sector,
        "wbo_2_pure": first_excited, "wbo_higher": higher_bo,
        "gd": gd, "geo_q": geo_q, "geo_R": geo_R,
        "native_total": native_total, "native_gi": native_gi,
        "total_raw": total_raw,
        "energy_reference": energy_reference,
        "bo_reference": bo_reference,
        "identity_residual": identity_residual,
        "stored_decomposition": stored,
        "support": support,
    }


_TDPES2_KEYS = ("total", "wbo_1", "wbo_2", "gd", "geo_q", "geo_R")
_TDPES2_TITLES = (
    r"$\widetilde\epsilon_{\rm total}^{(2)}="
    r"\widetilde\epsilon_{\rm GI}^{(2)}+\epsilon_{\rm GD}^{(2)}$",
    r"$\epsilon_{\rm wBO,1}^{(2)}$ (ground, $j=0$)",
    r"$\epsilon_{\rm wBO,2+}^{(2)}$ (all $j\geq1$)",
    r"Gauge dependent $\epsilon_{\rm GD}^{(2)}$",
    r"Internal proton $\epsilon_{q,\rm geo}^{(2)}$",
    r"Outer heavy $\epsilon_{R,\rm geo}^{(2)}$ (link-metric limit)",
)


def _tdpes2_origin_preparation(obs, ef_positive, args):
    required = (
        "epsilon_2", "epsilon_2_gi", "epsilon_1_wbo",
        "bo_channel_density_qR", "sgamma_R1",
    )
    missing = [key for key in required if key not in ef_positive]
    if missing:
        raise KeyError(
            "TDPES2 decomposition requires: " + ", ".join(missing)
            + "; rebuild the EF cache with --channel-density-states 2 "
              "--link-output nearest --overwrite"
        )
    if ef_positive["bo_channel_density_qR"].shape[1] < 2:
        raise ValueError("TDPES2 BO panels require at least two channel densities")
    provisional = {
        "floor": float(args.support_floor),
        "focus_floor": float(args.analysis_focus_floor),
        "heavy_mass": float(obs["options"].get("heavy_mass", 1836.15267343)),
        "stored_decomposition": all(key in ef_positive for key in (
            "tdpes2_total", "tdpes2_wbo_0", "tdpes2_wbo_excited",
            "tdpes2_gd", "tdpes2_geo_q", "tdpes2_geo_R",
        )),
        "energy_reference_mode": getattr(
            args, "tdpes_energy_reference", "initial",
        ),
    }
    if provisional["energy_reference_mode"] == "initial":
        initial = _tdpes2_origin_frame(obs, ef_positive, provisional, 0)
        provisional["fixed_energy_reference"] = initial["energy_reference"]
    samples, identity_errors, higher_sizes = [], [], []
    frames = _movie_frames(obs, min(
        args.max_frames, getattr(args, "scale_sample_frames", 32),
    ))
    for frame in frames:
        frame = int(frame)
        current = _tdpes2_origin_frame(obs, ef_positive, provisional, frame)
        support = (
            obs["heavy_density"][frame]
            >= provisional["focus_floor"]
            *max(float(np.max(obs["heavy_density"][frame])), 1.0e-300)
        )
        for key in _TDPES2_KEYS:
            selected = np.abs(current[key][support & np.isfinite(current[key])])
            if selected.size:
                samples.append(float(np.percentile(selected, 99.0)))
        selected_gi = np.abs(
            current["gi"][support & np.isfinite(current["gi"])]
        )
        if selected_gi.size:
            samples.append(float(np.percentile(selected_gi, 99.0)))
        for reference in current["bo_reference"]:
            selected = np.abs(reference[support & np.isfinite(reference)])
            if selected.size:
                samples.append(float(np.percentile(selected, 99.0)))
        if np.any(support):
            identity_errors.append(float(np.max(np.abs(
                current["identity_residual"][support]
            ))))
            higher_sizes.append(float(np.max(np.abs(
                current["wbo_higher"][support]
            ))))
    bound = max(
        float(np.percentile(samples, 98.0)) if samples else 0.0, 1.0e-10,
    )
    bound *= 1.06
    maximum_residual = max(identity_errors, default=0.0)
    tolerance = 256.0*np.finfo(np.float64).eps*max(bound, 1.0)
    if maximum_residual > tolerance:
        raise RuntimeError(
            "displayed TDPES2 decomposition does not close: "
            f"max residual={maximum_residual:.6e}, tolerance={tolerance:.6e}"
        )
    provisional.update({
        "energy_limits": (-bound, bound),
        "max_identity_residual": maximum_residual,
        "identity_tolerance": tolerance,
        "max_higher_bo_contribution": max(higher_sizes, default=0.0),
    })
    return provisional


def _tdpes2_axes(fig, slot=None):
    if slot is None:
        grid = fig.add_gridspec(
            2, 3, left=0.065, right=0.975, bottom=0.085, top=0.855,
            wspace=0.27, hspace=0.38,
        )
    else:
        grid = slot.subgridspec(2, 3, wspace=0.30, hspace=0.40)
    return [fig.add_subplot(grid[row, column])
            for row in range(2) for column in range(3)]


def _draw_tdpes2_origin(axes, obs, ef_positive, prep, frame, *, compact=False):
    current = _tdpes2_origin_frame(obs, ef_positive, prep, frame)
    active, _, limits = _frame_heavy_focus(
        obs, frame, prep["focus_floor"],
    )
    R = obs["R"]
    lines, reference_lines = [], []
    balance_lines = []
    for panel, (axis, key, title) in enumerate(zip(
            axes, _TDPES2_KEYS, _TDPES2_TITLES)):
        line, = axis.plot(
            R, np.where(active, current[key], np.nan),
            color="0.08", lw=(1.15 if compact else 2.2),
            label=(r"$\widetilde\epsilon_{\rm total}^{(2)}$"
                   if panel == 0 else None), zorder=5,
        )
        if panel == 0:
            gi_line, = axis.plot(
                R, np.where(active, current["gi"], np.nan),
                color="#2A7F62", lw=(0.95 if compact else 1.9),
                label=r"$\widetilde\epsilon_{\rm GI}^{(2)}$",
                zorder=4,
            )
            gd_line, = axis.plot(
                R, np.where(active, current["gd"], np.nan),
                color="#B23A48", lw=(0.95 if compact else 1.9),
                label=r"$\epsilon_{\rm GD}^{(2)}$",
                zorder=4,
            )
            balance_lines.extend((gi_line, gd_line))
        refs = []
        for state, (color, label) in enumerate((
            (COLORS[0], r"$\overline{E}_0^{\rm BO}(R,t)$"),
            (COLORS[1], r"$\overline{E}_1^{\rm BO}(R,t)$"),
        )):
            reference, = axis.plot(
                R, np.where(active, current["bo_reference"][state], np.nan),
                color=color, lw=(0.65 if compact else 1.25), ls="--",
                alpha=0.82, label=label, zorder=3,
            )
            refs.append(reference)
        axis.axhline(0.0, color="0.70", lw=0.55, zorder=0)
        axis.set(
            xlim=limits, ylim=prep["energy_limits"],
            xlabel=r"heavy $R$ ($a_0$)", ylabel="energy (Hartree)",
        )
        axis.set_title(
            title, loc="left", fontweight="semibold",
            fontsize=(5.2 if compact else 8.5),
        )
        axis.tick_params(labelsize=(4.8 if compact else 7), direction="in")
        axis.grid(alpha=0.16)
        lines.append(line)
        reference_lines.append(refs)
    axes[0].legend(
        handles=[lines[0], *balance_lines, *reference_lines[0]],
        frameon=False, fontsize=(3.7 if compact else 6.5),
        loc="best", ncol=(2 if compact else 1),
    )
    return {
        "lines": lines, "balance_lines": balance_lines,
        "reference_lines": reference_lines,
    }


def _update_tdpes2_origin(state, axes, obs, ef_positive, prep, frame):
    current = _tdpes2_origin_frame(obs, ef_positive, prep, frame)
    active, _, limits = _frame_heavy_focus(
        obs, frame, prep["focus_floor"],
    )
    artists = []
    for index, (axis, key) in enumerate(zip(axes, _TDPES2_KEYS)):
        state["lines"][index].set_ydata(
            np.where(active, current[key], np.nan),
        )
        axis.set_xlim(limits)
        artists.append(state["lines"][index])
        for bo_state, reference in enumerate(state["reference_lines"][index]):
            reference.set_ydata(np.where(
                active, current["bo_reference"][bo_state], np.nan,
            ))
            artists.append(reference)
    for line, key in zip(state["balance_lines"], ("gi", "gd")):
        line.set_ydata(np.where(active, current[key], np.nan))
        artists.append(line)
    return artists


def render_tdpes2_origin(obs, ef_positive, outdir, args, snapshots):
    prep = _tdpes2_origin_preparation(obs, ef_positive, args)
    times = obs["times_fs"]
    reference_label = (
        "fixed t=0 energy origin"
        if prep["energy_reference_mode"] == "initial"
        else "framewise density-weighted energy origin"
    )

    def individual(frame):
        fig = plt.figure(figsize=(16.5, 9.2), constrained_layout=False)
        axes = _tdpes2_axes(fig)
        _draw_tdpes2_origin(axes, obs, ef_positive, prep, frame)
        fig.suptitle(
            "Origin of second-level TDPES structure | "
            f"t={times[frame]:.4f} fs\npositive-density gauge; "
            f"{reference_label} and one y scale; dashed curves are "
            "proton-conditioned bare BO references",
            fontweight="bold",
        )
        return fig

    stem = "tdpes2_origin_positive_gauge"
    if prep["energy_reference_mode"] == "framewise":
        stem += "_framewise_reference"
    products = _save_individual_frames(
        individual, snapshots, times, Path(outdir)/f"{stem}_frames",
        stem, args.dpi,
    )
    fig = plt.figure(figsize=(28.0, 16.0), constrained_layout=False)
    outer = fig.add_gridspec(
        2, 4, left=0.025, right=0.985, bottom=0.04, top=0.92,
        wspace=0.18, hspace=0.22,
    )
    for slot, frame in zip(outer, snapshots):
        axes = _tdpes2_axes(fig, slot)
        _draw_tdpes2_origin(
            axes, obs, ef_positive, prep, int(frame), compact=True,
        )
        axes[0].text(
            0.98, 0.92, f"t={times[int(frame)]:.3f} fs",
            transform=axes[0].transAxes, ha="right", va="top", fontsize=5.2,
        )
    fig.suptitle(
        "Second-level TDPES origin (positive-density gauge): "
        f"8 representative times; {reference_label}",
        fontweight="bold",
    )
    products.append(_save_figure(
        fig, Path(outdir)/f"{stem}_snapshots.png", args.dpi,
    ))
    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        fig = plt.figure(figsize=(16.5, 9.2), constrained_layout=False)
        axes = _tdpes2_axes(fig)
        state = _draw_tdpes2_origin(
            axes, obs, ef_positive, prep, int(frames[0]),
        )
        title = fig.suptitle("", fontweight="bold")

        def update(number):
            frame = int(frames[number])
            artists = _update_tdpes2_origin(
                state, axes, obs, ef_positive, prep, frame,
            )
            title.set_text(
                "Origin of second-level TDPES structure | "
                f"t={times[frame]:.4f} fs\npositive-density gauge; "
                f"{reference_label} and one y scale; dashed curves are "
                "proton-conditioned bare BO references"
            )
            return (*artists, title)

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        products.append(_save_analysis_movie(
            animation, fig, outdir, f"{stem}_movie", args,
        ))
    return products, prep


# ---------------------------------------------------------------------------
# 9. Four nonnegative geometry contributions on one shared logarithmic scale


def _tdpes_geometry_frame(obs, ef_positive, prep, frame):
    """Return reference-independent first/second-level geometry energies."""
    joint = np.asarray(obs["joint_density"][frame], float)
    if prep["stored_decomposition"]:
        # Keep the standalone geometry diagnostic on exactly the same stored
        # arrays as the six-panel TDPES figures.  No plot-time reconstruction
        # is performed for a newly generated EF cache.
        geo1_q = np.asarray(ef_positive["tdpes1_geo_q"][frame], float)
        geo1_R = np.asarray(ef_positive["tdpes1_geo_R"][frame], float)
        geo2_q = np.asarray(ef_positive["tdpes2_geo_q"][frame], float)
        geo2_R = np.asarray(ef_positive["tdpes2_geo_R"][frame], float)
    else:
        # Compatibility path for caches created before the decomposed fields
        # were stored.  Rebuilding the cache removes this branch.
        heavy = np.asarray(obs["heavy_density"][frame], float)
        conditional = np.divide(
            joint, heavy[None, :], out=np.zeros_like(joint),
            where=heavy[None, :] > np.finfo(np.float64).tiny,
        )
        geo1_q = _site_link_metric(
            ef_positive["sphi_q1"][frame], obs["dq"], axis=0,
        )/(2.0*prep["proton_mass"])
        geo1_R = _site_link_metric(
            ef_positive["sphi_R1"][frame], obs["dR"], axis=1,
        )/(2.0*prep["heavy_mass"])
        wbo2 = np.sum(
            conditional*np.asarray(
                ef_positive["epsilon_1_wbo"][frame], float,
            ), axis=0, dtype=np.float64,
        )*obs["dq"]
        geo2_q = np.asarray(
            ef_positive["epsilon_2_gi"][frame], float,
        )-wbo2
        geo2_R = _site_link_metric(
            ef_positive["sgamma_R1"][frame], obs["dR"], axis=0,
        )/(2.0*prep["heavy_mass"])
    peak = max(float(np.max(joint)), 1.0e-300)
    return {
        "geo1_q": geo1_q, "geo1_R": geo1_R,
        "geo2_q": geo2_q, "geo2_R": geo2_R,
        "joint_log": np.log10(np.maximum(joint/peak, 1.0e-300)),
    }


def _tdpes_geometry_preparation(obs, ef_positive, args):
    stored_keys = (
        "tdpes1_geo_q", "tdpes1_geo_R",
        "tdpes2_geo_q", "tdpes2_geo_R",
    )
    stored_decomposition = all(key in ef_positive for key in stored_keys)
    required = stored_keys if stored_decomposition else (
        "epsilon_2_gi", "epsilon_1_wbo",
        "sphi_q1", "sphi_R1", "sgamma_R1",
    )
    missing = [key for key in required if key not in ef_positive]
    if missing:
        raise KeyError(
            "geometry comparison requires: " + ", ".join(missing)
            + "; rebuild the EF cache with --link-output nearest --overwrite"
        )
    prep = {
        "floor": float(args.support_floor),
        "focus_floor": float(args.analysis_focus_floor),
        "stored_decomposition": stored_decomposition,
        "proton_mass": float(obs["options"].get("proton_mass", 1836.15267343)),
        "heavy_mass": float(obs["options"].get("heavy_mass", 1836.15267343)),
        "decades": float(getattr(args, "geometry_decades", 8.0)),
        "contour_q_points": int(args.tdpes_contour_q_points),
        "contour_R_points": int(args.tdpes_contour_R_points),
    }
    samples = []
    negative_minima = {key: 0.0 for key in (
        "geo1_q", "geo1_R", "geo2_q", "geo2_R",
    )}
    frames = _movie_frames(obs, min(args.max_frames, args.scale_sample_frames))
    for frame in frames:
        frame = int(frame)
        current = _tdpes_geometry_frame(obs, ef_positive, prep, frame)
        joint = np.asarray(obs["joint_density"][frame], float)
        heavy = np.asarray(obs["heavy_density"][frame], float)
        support2d = joint >= prep["focus_floor"]*max(float(np.max(joint)), 1e-300)
        support1d = heavy >= prep["focus_floor"]*max(float(np.max(heavy)), 1e-300)
        for key, support in (("geo1_q", support2d), ("geo1_R", support2d),
                             ("geo2_q", support1d), ("geo2_R", support1d)):
            values = current[key][support & np.isfinite(current[key])]
            if values.size:
                positive = values[values > 0.0]
                if positive.size:
                    # Keep scale selection O(number of sampled frames), not
                    # O(full q-R trajectory), for multi-GiB production grids.
                    samples.append(float(np.percentile(positive, 99.5)))
                negative_minima[key] = min(
                    negative_minima[key], float(np.min(values)),
                )
    samples = np.asarray(samples, float)
    samples = samples[np.isfinite(samples) & (samples > 0.0)]
    bound = (
        float(np.percentile(samples, 98.0)) if samples.size else 1.0e-12
    )
    bound = max(1.06*bound, 1.0e-14)
    lower = max(bound*10.0**(-prep["decades"]), 1.0e-18)
    prep.update({
        "bound": bound,
        "lower": lower,
        "negative_minima": negative_minima,
        "reference_dependence": "none",
    })
    return prep


def _tdpes_geometry_norm(prep):
    return LogNorm(vmin=prep["lower"], vmax=prep["bound"], clip=False)


def _tdpes_geometry_axes(fig):
    grid = fig.add_gridspec(
        2, 2, left=0.070, right=0.900, bottom=0.090, top=0.850,
        wspace=0.25, hspace=0.38,
    )
    return [fig.add_subplot(grid[row, column])
            for row in range(2) for column in range(2)]


_TDPES_GEOMETRY_TITLES = (
    r"First level: $\epsilon^{(1)}_{q,\mathrm{geo}}(q,R,t)$",
    r"First level: $\epsilon^{(1)}_{R,\mathrm{geo}}(q,R,t)$",
    r"Second level: $\epsilon^{(2)}_{q,\mathrm{geo}}(R,t)$",
    r"Second level: $\epsilon^{(2)}_{R,\mathrm{geo}}(R,t)$",
)


def _draw_tdpes_geometry(fig, axes, obs, ef_positive, prep, frame,
                         *, colorbar=True):
    current = _tdpes_geometry_frame(obs, ef_positive, prep, frame)
    active2d, indices, limits2d = _frame_focus(
        obs, frame, prep["focus_floor"],
    )
    qi, Ri = indices
    cropped_active = active2d[np.ix_(qi, Ri)]
    extent = [obs["q"][qi[0]], obs["q"][qi[-1]],
              obs["R"][Ri[0]], obs["R"][Ri[-1]]]
    cq = qi[np.linspace(
        0, len(qi)-1, min(len(qi), prep["contour_q_points"]), dtype=int,
    )]
    cR = Ri[np.linspace(
        0, len(Ri)-1, min(len(Ri), prep["contour_R_points"]), dtype=int,
    )]
    norm = _tdpes_geometry_norm(prep)
    images, contours = [], []
    for axis, key, title in zip(
            axes[:2], ("geo1_q", "geo1_R"), _TDPES_GEOMETRY_TITLES[:2]):
        values = current[key][np.ix_(qi, Ri)]
        image = axis.imshow(
            np.ma.masked_where(
                ~cropped_active | ~np.isfinite(values) | (values <= 0.0),
                values,
            ).T,
            origin="lower", aspect="auto", extent=extent,
            cmap=masked_cmap(JOINT_CMAP), norm=norm, interpolation="nearest",
        )
        contours.append(_joint_contours(
            axis, obs, current["joint_log"][np.ix_(cq, cR)],
            -np.log10(prep["focus_floor"]), q=obs["q"][cq], R=obs["R"][cR],
            color="black", halo_color="white",
        ))
        axis.set_facecolor(MASK_COLOR)
        axis.set(
            xlim=limits2d[0], ylim=limits2d[1],
            xlabel=r"proton $q$ ($a_0$)", ylabel=r"heavy $R$ ($a_0$)",
        )
        axis.set_title(title, loc="left", fontweight="semibold", fontsize=9)
        axis.tick_params(labelsize=7, direction="in")
        images.append(image)

    active1d, _, limits1d = _frame_heavy_focus(
        obs, frame, prep["focus_floor"],
    )
    lines = []
    for axis, key, title, color in zip(
            axes[2:], ("geo2_q", "geo2_R"), _TDPES_GEOMETRY_TITLES[2:],
            (PARTICLE_COLORS["proton"], PARTICLE_COLORS["heavy"])):
        values = np.where(active1d, current[key], np.nan)
        line, = axis.plot(
            obs["R"], np.where(values > 0.0, values, np.nan),
            color=color, lw=2.2,
        )
        axis.set_yscale("log", base=10)
        axis.set(
            xlim=limits1d, ylim=(prep["lower"], prep["bound"]),
            xlabel=r"heavy $R$ ($a_0$)", ylabel="geometry energy (Hartree)",
        )
        axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        axis.set_title(title, loc="left", fontweight="semibold", fontsize=9)
        axis.tick_params(labelsize=7, direction="in")
        axis.grid(which="major", alpha=0.20)
        axis.grid(which="minor", alpha=0.07)
        lines.append(line)
    if colorbar:
        cax = fig.add_axes((0.925, 0.515, 0.014, 0.285))
        bar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=JOINT_CMAP), cax=cax,
            extend="both", format=LogFormatterMathtext(base=10.0),
        )
        bar.locator = LogLocator(base=10.0, subs=(1.0,))
        bar.update_ticks()
        bar.set_label("geometry energy (Hartree; shared log scale)")
        bar.ax.tick_params(labelsize=7)
    return {"images": images, "contours": contours, "lines": lines}


def _update_tdpes_geometry(state, axes, obs, ef_positive, prep, frame):
    current = _tdpes_geometry_frame(obs, ef_positive, prep, frame)
    active2d, indices, limits2d = _frame_focus(
        obs, frame, prep["focus_floor"],
    )
    qi, Ri = indices
    cropped_active = active2d[np.ix_(qi, Ri)]
    extent = [obs["q"][qi[0]], obs["q"][qi[-1]],
              obs["R"][Ri[0]], obs["R"][Ri[-1]]]
    cq = qi[np.linspace(
        0, len(qi)-1, min(len(qi), prep["contour_q_points"]), dtype=int,
    )]
    cR = Ri[np.linspace(
        0, len(Ri)-1, min(len(Ri), prep["contour_R_points"]), dtype=int,
    )]
    artists = []
    for index, (axis, image, key) in enumerate(zip(
            axes[:2], state["images"], ("geo1_q", "geo1_R"))):
        values = current[key][np.ix_(qi, Ri)]
        image.set_data(np.ma.masked_where(
            ~cropped_active | ~np.isfinite(values) | (values <= 0.0), values,
        ).T)
        image.set_extent(extent)
        axis.set_xlim(limits2d[0])
        axis.set_ylim(limits2d[1])
        for collection in state["contours"][index].collections:
            collection.remove()
        state["contours"][index] = _joint_contours(
            axis, obs, current["joint_log"][np.ix_(cq, cR)],
            -np.log10(prep["focus_floor"]), q=obs["q"][cq], R=obs["R"][cR],
            color="black", halo_color="white",
        )
        artists.append(image)
        artists.extend(state["contours"][index].collections)
    active1d, _, limits1d = _frame_heavy_focus(
        obs, frame, prep["focus_floor"],
    )
    for axis, line, key in zip(
            axes[2:], state["lines"], ("geo2_q", "geo2_R")):
        values = np.where(active1d, current[key], np.nan)
        line.set_ydata(np.where(values > 0.0, values, np.nan))
        axis.set_xlim(limits1d)
        artists.append(line)
    return artists


def render_tdpes_geometry_log(obs, ef_positive, outdir, args, snapshots):
    """Render once; fixed/framewise names are identical reference-free aliases."""
    if ef_positive.get("gauge") != "positive_density":
        raise ValueError("geometry comparison requires the positive-density cache")
    prep = _tdpes_geometry_preparation(obs, ef_positive, args)
    times = obs["times_fs"]

    def individual(frame):
        fig = plt.figure(figsize=(15.5, 9.0), constrained_layout=False)
        axes = _tdpes_geometry_axes(fig)
        _draw_tdpes_geometry(fig, axes, obs, ef_positive, prep, frame)
        fig.suptitle(
            "First- and second-level geometry energies | "
            f"t={times[frame]:.4f} fs\n"
            "one shared positive-log Hartree scale; energy-reference independent",
            fontweight="bold",
        )
        return fig

    products = _save_individual_frames(
        individual, snapshots, times, Path(outdir)/"tdpes_geometry_log_frames",
        "tdpes_geometry_log", args.dpi,
    )
    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        fig = plt.figure(figsize=(15.5, 9.0), constrained_layout=False)
        axes = _tdpes_geometry_axes(fig)
        title = fig.suptitle("", fontweight="bold")
        state = _draw_tdpes_geometry(
            fig, axes, obs, ef_positive, prep, int(frames[0]),
        )

        def update(number):
            frame = int(frames[number])
            artists = _update_tdpes_geometry(
                state, axes, obs, ef_positive, prep, frame,
            )
            title.set_text(
                "First- and second-level geometry energies | "
                f"t={times[frame]:.4f} fs\n"
                "shared positive-log scale; independent of fixed/framewise E_ref"
            )
            return (*artists, title)

        update(0)
        animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
        fixed = _save_analysis_movie(
            animation, fig, outdir,
            "tdpes_geometry_log_fixed_reference_movie", args,
        )
        products.append(fixed)
        suffix = Path(fixed).suffix
        products.append(_hardlink_output_alias(
            fixed,
            Path(outdir)/f"tdpes_geometry_log_framewise_reference_movie{suffix}",
        ))
    return products, prep


# ---------------------------------------------------------------------------
# Orchestration


def run(args):
    archive, run_dir = find_archive(resolve_run_input(args.run))
    obs = tdse_report.calculate_observables(tdse_report.load_observables(archive))
    output = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir else
        (run_dir/"report"/"final_visualizations").resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    selected = tuple(args.only or FINAL_PRODUCTS)
    snapshots = _snapshot_frames(obs, args.snapshot_count)
    print(
        f"final visualization: archive={archive}; output={output}; "
        f"products={','.join(selected)}"
    )
    print(
        "snapshot frames: "+", ".join(
            f"{int(frame)} ({obs['times_fs'][int(frame)]:.6f} fs)"
            for frame in snapshots
        )
    )

    products = []
    if "marginal" in selected:
        products.extend(render_marginal_time_position(obs, output, args, snapshots))
    if "joint" in selected:
        products.extend(render_joint_density(obs, output, args, snapshots))

    ef_needed = any(
        name in selected
        for name in (
            "velocity", "vector", "current", "nested", "heavy", "bo",
            "bo3d", "tdpes1", "tdpes2", "geometry",
        )
    )
    ef = None
    if ef_needed:
        ef_cache_path = run_dir/"tdse_exact_factorization_fields.npz"
        decomposition_keys = ()
        if ef_cache_path.is_file():
            with np.load(ef_cache_path, allow_pickle=False) as stored:
                available = set(stored.files)
            candidates = (
                "tdpes1_total", "tdpes1_wbo_0",
                "tdpes1_wbo_excited", "tdpes1_gd",
                "tdpes1_geo_q", "tdpes1_geo_R",
                "tdpes2_total", "tdpes2_wbo_0",
                "tdpes2_wbo_excited", "tdpes2_gd",
                "tdpes2_geo_q", "tdpes2_geo_R",
                "tdpes1_closure_defect", "tdpes2_closure_defect",
                "tdpes1_native_representation_difference",
                "tdpes2_native_representation_difference",
            )
            decomposition_keys = tuple(
                key for key in candidates if key in available
            )
        field_keys = []
        if "velocity" in selected:
            field_keys.extend(("a", "b"))
        if "vector" in selected:
            field_keys.extend(("a", "b", "alpha"))
        if "current" in selected:
            field_keys.extend(("a", "b", "alpha"))
        if "nested" in selected:
            field_keys.extend((
                "epsilon_1", "epsilon_2", "a", "b", "alpha",
                "electron_proton_density",
            ))
        if "heavy" in selected:
            field_keys.extend(("epsilon_2", "alpha"))
        if "bo" in selected:
            field_keys.extend(("bo_state_density_q", "bo_state_density_R"))
        if "bo3d" in selected:
            field_keys.append("bo_channel_density_qR")
        if "tdpes1" in selected:
            field_keys.extend((
                "epsilon_1", "epsilon_1_gi", "epsilon_1_wbo",
                "bo_channel_density_qR",
            ))
            field_keys.extend(
                key for key in decomposition_keys if key.startswith("tdpes1_")
            )
        if "tdpes2" in selected:
            field_keys.extend((
                "epsilon_2", "epsilon_2_gi", "epsilon_1_wbo",
                "bo_channel_density_qR",
            ))
            field_keys.extend(
                key for key in decomposition_keys if key.startswith("tdpes2_")
            )
        if "geometry" in selected:
            field_keys.extend(("epsilon_2_gi", "epsilon_1_wbo"))
            field_keys.extend(
                key for key in decomposition_keys
                if key in (
                    "tdpes1_geo_q", "tdpes1_geo_R",
                    "tdpes2_geo_q", "tdpes2_geo_R",
                )
            )
        if "nested" in selected:
            field_keys.extend(
                key for key in decomposition_keys
                if key in ("tdpes1_total", "tdpes2_total")
            )
        field_keys = tuple(dict.fromkeys(field_keys))
        link_keys = []
        if "nested" in selected:
            link_keys.extend(("sphi_q1", "sphi_R1", "sgamma_R1"))
        else:
            if "tdpes1" in selected:
                link_keys.extend(("sphi_q1", "sphi_R1"))
            if "tdpes2" in selected:
                link_keys.append("sgamma_R1")
            if "geometry" in selected:
                stored_geometry = all(
                    key in decomposition_keys for key in (
                        "tdpes1_geo_q", "tdpes1_geo_R",
                        "tdpes2_geo_q", "tdpes2_geo_R",
                    )
                )
                if not stored_geometry:
                    link_keys.extend(("sphi_q1", "sphi_R1", "sgamma_R1"))
            if "heavy" in selected:
                link_keys.append("sgamma_R1")
        ef = tdse_report._load_ef_fields(
            obs, field_keys=field_keys, link_keys=tuple(link_keys),
        )
        if ef is None:
            raise FileNotFoundError(
                f"{run_dir/'tdse_exact_factorization_fields.npz'}가 없습니다."
            )

    velocity_prep = None
    if "velocity" in selected:
        generated, velocity_prep = render_joint_velocity(
            obs, ef, output, args, snapshots,
        )
        products.extend(generated)

    vector_prep = None
    if "vector" in selected:
        generated, vector_prep = render_vector_composite(
            obs, ef, output, args, snapshots,
        )
        products.extend(generated)
    current_prep = None
    if "current" in selected:
        generated, current_prep = render_current_composite(
            obs, ef, output, args, snapshots,
        )
        products.extend(generated)
    if "bo" in selected:
        products.extend(render_bo_combined(obs, ef, output, args, snapshots))
    bo3d_prep = None
    if "bo3d" in selected:
        generated, bo3d_prep = render_bo3d_channels(
            obs, ef, output, args, snapshots,
        )
        products.extend(generated)

    requested_reference_mode = args.tdpes_energy_reference
    reference_modes = (
        ("initial", "framewise")
        if requested_reference_mode == "both"
        else (requested_reference_mode,)
    )
    tdpes1_preps = {}
    if "tdpes1" in selected:
        for reference_mode in reference_modes:
            args.tdpes_energy_reference = reference_mode
            generated, prep = render_tdpes1_origin(
                obs, ef, output, args, snapshots,
            )
            products.extend(generated)
            tdpes1_preps[reference_mode] = prep

    tdpes2_preps = {}
    if "tdpes2" in selected:
        for reference_mode in reference_modes:
            args.tdpes_energy_reference = reference_mode
            generated, prep = render_tdpes2_origin(
                obs, ef, output, args, snapshots,
            )
            products.extend(generated)
            tdpes2_preps[reference_mode] = prep
    args.tdpes_energy_reference = requested_reference_mode

    geometry_prep = None
    if "geometry" in selected:
        generated, geometry_prep = render_tdpes_geometry_log(
            obs, ef, output, args, snapshots,
        )
        products.extend(generated)

    alpha_positive = None
    branch_turns = None
    if "heavy" in selected:
        alpha_positive, _, branch_turns = tdse_report.support_aware_temporal_lift_1d(
            ef["alpha"], obs["heavy_density"], obs["dR"], args.support_floor,
        )
        alpha_positive = alpha_positive.copy()

    nested_prep = None
    if "nested" in selected:
        generated, nested_prep = render_nested_factorization(
            obs, ef, output, args, snapshots,
        )
        products.extend(generated)

    tdpes1_prep = (
        tdpes1_preps.get("initial")
        or next(iter(tdpes1_preps.values()), None)
    )
    tdpes2_prep = (
        tdpes2_preps.get("initial")
        or next(iter(tdpes2_preps.values()), None)
    )

    heavy_prep = None
    if "heavy" in selected:
        tdse_report.transform_second_level_to_zero_potential_gauge(obs, ef)
        generated, heavy_prep = render_heavy_analysis(
            obs, ef, alpha_positive, output, args, snapshots,
        )
        products.extend(generated)
        print(
            "heavy positive-gauge alpha branch turns: "
            f"min={int(np.min(branch_turns))}, max={int(np.max(branch_turns))}"
        )
        print(
            "heavy force decomposition audit: "
            "max|F_total-(F_driven+F_harm)|="
            f"{heavy_prep['force_decomposition_max_abs']:.3e}"
        )

    manifest = [
        f"source_archive={archive}",
        f"analysis_focus_floor={args.analysis_focus_floor}",
        "analysis_focus=per_frame_joint_density_peak_relative_bounding_box",
        "snapshot_frames="+",".join(str(int(frame)) for frame in snapshots),
        "snapshot_times_fs="+",".join(
            f"{obs['times_fs'][int(frame)]:.12g}" for frame in snapshots
        ),
        "products="+",".join(str(Path(path).resolve()) for path in products),
    ]
    if vector_prep is not None:
        manifest.extend((
            f"vector_q_limits={vector_prep['q_limits']}",
            f"vector_R_limits={vector_prep['R_limits']}",
        ))
    if velocity_prep is not None:
        manifest.extend((
            f"velocity_q_limits={velocity_prep['q_limits']}",
            f"velocity_R_limits={velocity_prep['R_limits']}",
            (
                "velocity_reference_speed="
                f"{velocity_prep['reference_speed']:.16g}"
            ),
            f"velocity_quiver_scale={velocity_prep['quiver_scale']:.16g}",
            (
                "velocity_arrow_support_floor="
                f"{velocity_prep['arrow_support_floor']:.16g}"
            ),
            "velocity_components=(a/proton_mass,b/heavy_mass)",
            "velocity_arrow_scaling=trajectory_wide_no_field_normalization",
        ))
    if current_prep is not None:
        manifest.extend((
            "proton_current=joint_density*a/proton_mass",
            "heavy_joint_current=joint_density*b/heavy_mass",
            "heavy_marginal_current=heavy_density*alpha/heavy_mass",
            f"current_q_limits={current_prep['q_limits']}",
            f"current_R_limits={current_prep['R_limits']}",
        ))
    if nested_prep is not None:
        manifest.extend((
            "nested_potential_gauge=positive_density",
            "electron_proton_density=integral_dR_abs_Psi_squared",
            "conditional_proton_density=joint_density/heavy_density",
            "nested_density_display=absolute_linear_trajectory_fixed",
            (
                "nested_electron_proton_vmax="
                f"{nested_prep['electron_proton_vmax']:.16g}"
            ),
            (
                "nested_conditional_proton_vmax="
                f"{nested_prep['conditional_vmax']:.16g}"
            ),
            "epsilon_1_overlay=physical_joint_density_relative_contours",
            "nested_density_contours=12_linear_relative_levels_from_0.025_to_0.90",
            "nested_density_contour_style=thin_black_solid",
            (
                "electron_proton_mass_error="
                f"{nested_prep['electron_proton_mass_error']:.16g}"
            ),
            (
                "conditional_proton_normalization_error="
                f"{nested_prep['conditional_normalization_error']:.16g}"
            ),
            f"nested_x_limits={nested_prep['x_limits']}",
            f"nested_q_limits={nested_prep['q_limits']}",
            f"nested_R_limits={nested_prep['R_limits']}",
        ))
    if bo3d_prep is not None:
        manifest.extend((
            "bo3d_density=rho_j(q,R,t)=rho_qR*abs(C_j)^2=abs(Y_j)^2",
            "bo3d_surfaces=time_independent_BO_energies",
            "bo3d_vertical_density_lift=display_only_fixed_trajectory_scale",
            f"bo3d_states={bo3d_prep['n_states']}",
            f"bo3d_q_limits={bo3d_prep['q_limits']}",
            f"bo3d_R_limits={bo3d_prep['R_limits']}",
        ))
    if tdpes1_prep is not None:
        manifest.extend((
            "tdpes1_rendered_energy_reference_modes="
            + ",".join(tdpes1_preps),
            "tdpes1_panels=total,wBO_1,wBO_2,GD,q_geo,R_geo",
            "tdpes1_identity=total=wBO_1+wBO_2plus+GD+q_geo+R_geo",
            f"tdpes1_energy_reference_mode={tdpes1_prep['energy_reference_mode']}",
            (
                "tdpes1_fixed_energy_reference="
                f"{tdpes1_prep.get('fixed_energy_reference', float('nan')):.16g}"
            ),
            "tdpes1_wBO_1=abs(C_0)^2*(E_0_BO-E_ref)",
            "tdpes1_wBO_2plus=sum_j_ge_1_abs(C_j)^2*(E_j_BO-E_ref)",
            f"tdpes1_max_identity_residual={tdpes1_prep['max_identity_residual']:.16g}",
            f"tdpes1_identity_tolerance={tdpes1_prep['identity_tolerance']:.16g}",
            f"tdpes1_max_grouped_j_ge_2_BO_contribution={tdpes1_prep['max_higher_bo_contribution']:.16g}",
            "tdpes1_gauge=positive_density",
            "tdpes1_discrete_identity=E_total(gauge)=E_GI_native+E_GD(gauge)",
            "tdpes1_weighted_bo=sum_all_stored_abs(C_j)^2*E_j_BO",
            "tdpes1_q_metric=(1-abs(Sphi_q1)^2)/dq^2_site_centered",
            "tdpes1_R_metric=(1-abs(Sphi_R1)^2)/dR^2_site_centered",
            "tdpes1_continuum_limit_interpretation=E_wBO+q_metric/(2m_q)+R_metric/(2M)+GD",
            "tdpes1_warning=link_metrics_are_continuum_limit_diagnostics_not_termwise_finite_difference_replacements_of_the_native_discrete_GI_scalar",
            f"tdpes1_signed_bound={tdpes1_prep['signed_bound']:.16g}",
            f"tdpes1_geo_limits={tdpes1_prep['geo_limits']}",
            f"tdpes1_shared_bound={tdpes1_prep['common_bound']:.16g}",
            f"tdpes1_color_scale={tdpes1_prep['color_scale']}",
            f"tdpes1_symlog_linear_threshold={tdpes1_prep['linear_threshold']:.16g}",
            "tdpes1_density_contours=black_with_white_halo",
        ))
    if tdpes2_prep is not None:
        manifest.extend((
            "tdpes2_rendered_energy_reference_modes="
            + ",".join(tdpes2_preps),
            "tdpes2_gauge=positive_density",
            "tdpes2_panels=total,wBO_1,wBO_2plus,GD,q_geo,R_geo",
            "tdpes2_identity=total=wBO_1+wBO_2plus+GD+q_geo+R_geo",
            f"tdpes2_energy_reference_mode={tdpes2_prep['energy_reference_mode']}",
            (
                "tdpes2_fixed_energy_reference="
                f"{tdpes2_prep.get('fixed_energy_reference', float('nan')):.16g}"
            ),
            "tdpes2_wBO_1=integral_dq_rho_0_over_rho_R_times_(E_0_BO-E_ref)",
            "tdpes2_wBO_2plus=all_j_ge_1_sector_of_integral_dq_rho_conditional_times_(epsilon_1_wBO-E_ref)",
            "tdpes2_q_geo=native_epsilon_2_GI-minus-proton_average_epsilon_1_wBO",
            "tdpes2_R_geo=(1-abs(SGamma_R1)^2)/(2M*dR^2)_site_centered",
            "tdpes2_BO_reference_j=integral_dq_rho(q_given_R)*(E_j_BO-E_ref)",
            f"tdpes2_max_identity_residual={tdpes2_prep['max_identity_residual']:.16g}",
            f"tdpes2_identity_tolerance={tdpes2_prep['identity_tolerance']:.16g}",
            f"tdpes2_max_grouped_j_ge_2_BO_contribution={tdpes2_prep['max_higher_bo_contribution']:.16g}",
            f"tdpes2_energy_limits={tdpes2_prep['energy_limits']}",
            "tdpes2_x_window=per_frame_heavy_density_support",
        ))
    if geometry_prep is not None:
        manifest.extend((
            "geometry_panels=tdpes1_q_geo,tdpes1_R_geo,tdpes2_q_geo,tdpes2_R_geo",
            "geometry_gauge=positive_density_input_but_all_four_terms_are_gauge_invariant",
            "geometry_energy_reference_dependence=none",
            "geometry_fixed_and_framewise_movies=identical_aliases_by_definition",
            "geometry_scale=one_shared_positive_log_Hartree_scale",
            f"geometry_decades={geometry_prep['decades']:.16g}",
            f"geometry_shared_bound={geometry_prep['bound']:.16g}",
            f"geometry_lower_positive_limit={geometry_prep['lower']:.16g}",
            f"geometry_negative_minima={geometry_prep['negative_minima']}",
            "geometry_nonpositive_display=masked_and_reported_not_absolute_valued",
            "geometry_display_values=raw_positive_unsmoothed_no_energy_reference_subtraction",
            "geometry_2d_window=per_frame_joint_density_support",
            "geometry_1d_window=per_frame_heavy_density_support",
        ))
    if heavy_prep is not None:
        manifest.extend((
            "harmonic_potential=heavy_trap_alpha*(R-heavy_trap_center)^2",
            "total_force=-partial_R*epsilon_ZP^(2)",
            "harmonic_force=-partial_R*harmonic_potential",
            "driven_force=total_force-harmonic_force",
            "force_identity=total_force=driven_force+harmonic_force",
            "force_coordinates=forward_R_bonds",
            "force_derivative=forward_finite_difference_in_R",
            (
                "force_decomposition_max_abs="
                f"{heavy_prep['force_decomposition_max_abs']:.16g}"
            ),
            f"heavy_trap_alpha={heavy_prep['trap_alpha']:.16g}",
            f"heavy_trap_center={heavy_prep['trap_center']:.16g}",
            f"heavy_R_limits={heavy_prep['R_limits']}",
        ))
    manifest.extend((
        "default_scalar_vector_gauge=positive_density",
        "zero_potential_gauge_usage=heavy_force_from_minus_partial_R_epsilon_2_only",
    ))
    manifest_path = output/"final_visualizations_manifest.txt"
    manifest_path.write_text("\n".join(manifest)+"\n", encoding="utf-8")
    products.append(manifest_path)
    print(f"final visualization manifest 저장: {manifest_path}")
    del ef
    gc.collect()
    return products


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="completed TDSE run directory or archive")
    parser.add_argument(
        "--outdir", default="",
        help="default: RUN_DIRECTORY/report/final_visualizations",
    )
    parser.add_argument(
        "--only", nargs="+", choices=FINAL_PRODUCTS,
        help="render only selected product groups",
    )
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--snapshot-count", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--animation-dpi", type=int, default=110)
    parser.add_argument("--decades", type=float, default=6.0)
    parser.add_argument(
        "--geometry-decades", type=float, default=8.0,
        help="positive-log dynamic range reserved for the four geometry terms",
    )
    parser.add_argument("--support-floor", type=float, default=1.0e-4)
    parser.add_argument(
        "--velocity-q-points", type=int, default=36,
        help="number of fixed proton-coordinate arrow samples",
    )
    parser.add_argument(
        "--velocity-R-points", type=int, default=18,
        help="number of fixed heavy-coordinate arrow samples",
    )
    parser.add_argument("--marginal-ymax", type=float, default=1.5)
    parser.add_argument("--marginal-xmax", type=float, default=12.0)
    parser.add_argument("--heavy-min", type=float, default=5.0)
    parser.add_argument("--heavy-max", type=float, default=15.0)
    parser.add_argument("--surface-count", type=int, default=2)
    parser.add_argument("--bo3d-q-points", type=int, default=144)
    parser.add_argument("--bo3d-R-points", type=int, default=108)
    parser.add_argument("--movie-bo3d-q-points", type=int, default=72,
                        help="BO3D movie-only proton mesh; snapshots keep --bo3d-q-points")
    parser.add_argument("--movie-bo3d-R-points", type=int, default=54,
                        help="BO3D movie-only heavy mesh; snapshots keep --bo3d-R-points")
    parser.add_argument("--tdpes-contour-q-points", type=int, default=180,
                        help="maximum q samples used only to trace movie density contours")
    parser.add_argument("--tdpes-contour-R-points", type=int, default=120,
                        help="maximum R samples used only to trace movie density contours")
    parser.add_argument("--scale-sample-frames", type=int, default=32,
                        help="evenly spaced frames used to choose robust display limits")
    parser.add_argument("--tdpes-color-scale", choices=("symlog", "linear"),
                        default="linear",
                        help="one shared TDPES1 norm; linear is the readable default, symlog reveals small structure")
    parser.add_argument(
        "--tdpes-energy-reference",
        choices=("initial", "framewise", "both"), default="initial",
        help=(
            "TDPES energy zero: fixed occupied-density mean at t=0 "
            "(default), legacy per-frame mean, or render both"
        ),
    )
    parser.add_argument("--movie-preset", choices=("ultrafast", "veryfast", "fast", "medium", "slow"),
                        default="medium", help="libx264 encoding preset for analysis movies")
    parser.add_argument(
        '--analysis-focus-floor', type=float, default=1e-3,
        help=(
            'shared visible support for BO/TDPES/vector/current/nested and '
            'velocity arrows: density >= this fraction of each frame peak '
            '(default: 1e-3 = 0.1%%)'
        ),
    )
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.analysis_focus_floor < 1:
        parser.error('--analysis-focus-floor must lie strictly between 0 and 1')
    positive = (
        "fps", "max_frames", "snapshot_count", "dpi", "animation_dpi",
        "decades", "support_floor", "marginal_ymax", "marginal_xmax",
        "surface_count", "velocity_q_points", "velocity_R_points",
        "bo3d_q_points", "bo3d_R_points",
        "movie_bo3d_q_points", "movie_bo3d_R_points",
        "tdpes_contour_q_points", "tdpes_contour_R_points",
        "scale_sample_frames", "geometry_decades",
    )
    for name in positive:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.heavy_min) or not np.isfinite(args.heavy_max):
        parser.error("heavy coordinate limits must be finite")
    if args.heavy_min >= args.heavy_max:
        parser.error("--heavy-min must be smaller than --heavy-max")
    return args


if __name__ == "__main__":
    run(parse_args())
