#!/usr/bin/env python3
"""Render the compact final TDSE/MCEF analysis gallery from saved arrays.

This command deliberately reuses the reduced TDSE observables and the
postprocessed exact-factorization cache.  It never loads ``tdse_coefficients``
and does not repeat propagation or electronic factorization.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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
    "marginal", "joint", "velocity", "vector", "current", "heavy", "bo",
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
    for frame in _movie_frames(obs, args.max_frames):
        frame = int(frame)
        density = obs["joint_density"][frame][np.ix_(q_indices, R_indices)]
        support = density >= args.support_floor*max(
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
        obs, ef, prep, frame, args.support_floor,
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
            rf"$v_{{95}}={prep['reference_speed']:.2e}\ a_0/t_{{\rm au}}$",
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
                obs, ef, prep, frame, args.support_floor,
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
    prep["arrow_support_floor"] = float(args.support_floor)
    return products, prep


# ---------------------------------------------------------------------------
# 3. Particle dynamics + positive-gauge vector potentials


def _vector_preparation(obs, ef, args):
    floor = args.support_floor
    frames = _movie_frames(obs, args.max_frames)
    densities = [obs["joint_density"][int(frame)] for frame in frames]
    a_values = [ef["a"][int(frame)] for frame in frames]
    b_values = [ef["b"][int(frame)] for frame in frames]
    connection_limits = _symmetric_support_bound(
        a_values+b_values, densities+densities, floor,
    )
    alpha_lifted, heavy_support, _ = tdse_report.support_aware_temporal_lift_1d(
        ef["alpha"], obs["heavy_density"], obs["dR"], floor,
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
    opacity = density_display_alpha(density, args.support_floor)
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
            xlim=prep["q_limits"], ylim=prep["R_limits"],
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
    }


def _update_vector_composite(state, obs, ef, prep, frame, args):
    for line, density in zip(state["marginal_lines"], (
        obs["electron_density"], obs["proton_density"], obs["heavy_density"],
    )):
        line.set_ydata(density[frame])
    opacity = density_display_alpha(obs["joint_density"][frame], args.support_floor)
    for image, key in state["images"]:
        image.set_data(ef[key][frame].T)
        image.set_alpha(opacity.T)
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
    proton_bounds, heavy_joint_bounds, heavy_marginal_bounds = [], [], []
    for frame in frames:
        frame = int(frame)
        density = obs["joint_density"][frame]
        heavy = obs["heavy_density"][frame]
        proton_current = density*ef["a"][frame]/proton_mass
        heavy_joint_current = density*ef["b"][frame]/heavy_mass
        heavy_marginal_current = heavy*alpha_lifted[frame]/heavy_mass
        proton_bounds.append(_supported_percentile_bound(
            proton_current, density, args.support_floor,
        ))
        heavy_joint_bounds.append(_supported_percentile_bound(
            heavy_joint_current, density, args.support_floor,
        ))
        heavy_marginal_bounds.append(_supported_percentile_bound(
            heavy_marginal_current, heavy, args.support_floor,
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
            obs["q"], obs["proton_density"], args.support_floor,
        ),
        "R_limits": _support_limits(
            obs["R"], obs["heavy_density"], args.support_floor,
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
    opacity = density_display_alpha(density, args.support_floor)
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
            xlim=prep["q_limits"], ylim=prep["R_limits"],
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
    }


def _update_current_composite(state, obs, ef, prep, frame, args):
    for line, density in zip(state["marginal_lines"], (
        obs["electron_density"], obs["proton_density"], obs["heavy_density"],
    )):
        line.set_ydata(density[frame])
    current = _current_frame(obs, ef, prep, frame)
    opacity = density_display_alpha(
        obs["joint_density"][frame], args.support_floor,
    )
    for image, key in state["images"]:
        image.set_data(current[key].T)
        image.set_alpha(opacity.T)
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
# 4. Heavy-coordinate force/momentum analysis


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
        for name in ("velocity", "vector", "current", "heavy", "bo")
    )
    ef = None
    if ef_needed:
        field_keys = []
        if "velocity" in selected:
            field_keys.extend(("a", "b"))
        if "vector" in selected:
            field_keys.extend(("a", "b", "alpha"))
        if "current" in selected:
            field_keys.extend(("a", "b", "alpha"))
        if "heavy" in selected:
            field_keys.extend(("epsilon_2", "alpha"))
        if "bo" in selected:
            field_keys.extend(("bo_state_density_q", "bo_state_density_R"))
        field_keys = tuple(dict.fromkeys(field_keys))
        link_keys = ("sgamma_R1",) if "heavy" in selected else ()
        ef = tdse_report._load_ef_fields(
            obs, field_keys=field_keys, link_keys=link_keys,
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
    heavy_prep = None
    if "heavy" in selected:
        alpha_positive, _, branch_turns = tdse_report.support_aware_temporal_lift_1d(
            ef["alpha"], obs["heavy_density"], obs["dR"], args.support_floor,
        )
        alpha_positive = alpha_positive.copy()
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
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args(argv)
    positive = (
        "fps", "max_frames", "snapshot_count", "dpi", "animation_dpi",
        "decades", "support_floor", "marginal_ymax", "marginal_xmax",
        "surface_count", "velocity_q_points", "velocity_R_points",
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
