"""Four-panel coordinate movies with density-following field windows."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from .report_plot_style import add_fixed_center_markers
from .visualize import selected_frames


def _support_intervals(coordinate, density, floor, padding_fraction=0.16):
    spacing = abs(float(coordinate[1]-coordinate[0])) if len(coordinate) > 1 else 1.0
    limits = []
    masks = []
    for values in density:
        support = values >= floor*max(float(np.max(values)), 1.0e-300)
        indices = np.flatnonzero(support)
        if not indices.size:
            indices = np.array([int(np.argmax(values))])
        left = float(coordinate[indices[0]])
        right = float(coordinate[indices[-1]])
        span = max(right-left, 6.0*spacing)
        padding = max(padding_fraction*span, 3.0*spacing)
        limits.append((
            max(float(coordinate[0]), left-padding),
            min(float(coordinate[-1]), right+padding),
        ))
        masks.append(support)
    return np.asarray(limits), np.asarray(masks)


def _profile_ylim(values, support, symmetric):
    frame_bounds = []
    frame_lows = []
    frame_highs = []
    for frame in range(len(values)):
        selected = np.asarray(values[frame])[support[frame]]
        selected = selected[np.isfinite(selected)]
        if not selected.size:
            continue
        if symmetric:
            frame_bounds.append(float(np.percentile(np.abs(selected), 99.0)))
        else:
            # Keep scalar-potential panels equivalent to the existing exact-
            # potential line plots: every finite value on occupied support is
            # inside the displayed y range. Robust clipping is reserved for
            # signed momentum/current panels with isolated spikes.
            frame_lows.append(float(np.min(selected)))
            frame_highs.append(float(np.max(selected)))
    if symmetric:
        bound = max(
            float(np.percentile(frame_bounds, 98.0)) if frame_bounds else 0.0,
            1.0e-12,
        )
        return -1.08*bound, 1.08*bound
    if not frame_lows:
        return -1.0, 1.0
    low = min(frame_lows)
    high = max(frame_highs)
    span = max(high-low, 1.0e-12)
    return low-0.06*span, high+0.06*span


def make_coordinate_focus_animation(
    *, times_fs, coordinate, marginal, profiles, options, outdir, fps,
    max_frames, dpi, fmt, particle_name, coordinate_symbol, color,
    stem, marginal_ymax=1.5, marginal_xmax=12.0, support_floor=1.0e-3,
):
    """Animate one marginal plus three fields over its moving support."""
    times = np.asarray(times_fs, float)
    coordinate = np.asarray(coordinate, float)
    marginal = np.asarray(marginal, float)
    if marginal.shape != (len(times), len(coordinate)):
        raise ValueError("coordinate marginal shape mismatch")
    if len(profiles) != 3:
        raise ValueError("exactly three coordinate profiles are required")
    limits, support = _support_intervals(
        coordinate, marginal, support_floor
    )
    prepared = []
    for title, ylabel, values, profile_color, symmetric in profiles:
        values = np.asarray(values, float)
        if values.shape != marginal.shape:
            raise ValueError(f"{title} profile shape mismatch")
        prepared.append((
            title, ylabel, values, profile_color, symmetric,
            _profile_ylim(values, support, symmetric),
        ))

    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.3), constrained_layout=True)
    marginal_axis = axes[0, 0]
    marginal_line, = marginal_axis.plot(
        coordinate, marginal[first], color=color, lw=2.3,
    )
    left_marker = marginal_axis.axvline(limits[first, 0], color="0.35", ls=":", lw=1.0)
    right_marker = marginal_axis.axvline(limits[first, 1], color="0.35", ls=":", lw=1.0)
    add_fixed_center_markers(marginal_axis, options)
    marginal_axis.set(
        xlim=(max(float(coordinate[0]), -marginal_xmax),
              min(float(coordinate[-1]), marginal_xmax)),
        ylim=(0.0, marginal_ymax),
        xlabel=rf"{coordinate_symbol} ($a_0$)",
        ylabel=r"probability density ($a_0^{-1}$)",
    )
    marginal_axis.set_title(
        f"{particle_name} marginal | dotted lines bound field windows",
        loc="left", fontweight="semibold",
    )

    profile_artists = []
    for axis, item in zip((axes[0, 1], axes[1, 0], axes[1, 1]), prepared):
        title_text, ylabel, values, profile_color, _symmetric, ylim = item
        occupied = np.where(support[first], values[first], np.nan)
        tail = np.where(~support[first], values[first], np.nan)
        tail_line, = axis.plot(
            coordinate, tail, color=profile_color, lw=0.55, ls=":",
            alpha=0.22, zorder=0.5,
        )
        support_line, = axis.plot(
            coordinate, occupied, color=profile_color, lw=2.15, zorder=2.5,
        )
        axis.set(
            xlim=tuple(limits[first]), ylim=ylim,
            xlabel=rf"{coordinate_symbol} ($a_0$)", ylabel=ylabel,
        )
        axis.set_title(title_text, loc="left", fontweight="semibold")
        add_fixed_center_markers(axis, options)
        profile_artists.append((axis, support_line, tail_line, values))

    for axis in axes.flat:
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.tick_params(direction="in")
    moment_text = marginal_axis.text(
        0.985, 0.95, "", transform=marginal_axis.transAxes,
        ha="right", va="top", fontsize=8.7,
        bbox=dict(fc="white", ec="0.85", alpha=0.86, pad=3),
    )
    title = fig.suptitle("")
    spacing = float(coordinate[1]-coordinate[0]) if len(coordinate) > 1 else 1.0
    mass = np.sum(marginal, axis=1)*spacing
    mean = np.sum(marginal*coordinate[None, :], axis=1)*spacing/np.maximum(mass, 1e-300)
    width = np.sqrt(np.maximum(
        np.sum(marginal*(coordinate[None, :]-mean[:, None])**2, axis=1)
        *spacing/np.maximum(mass, 1e-300), 0.0,
    ))

    def update(number):
        frame = int(frames[number])
        marginal_line.set_ydata(marginal[frame])
        left_marker.set_xdata([limits[frame, 0], limits[frame, 0]])
        right_marker.set_xdata([limits[frame, 1], limits[frame, 1]])
        for axis, support_line, tail_line, values in profile_artists:
            support_line.set_ydata(np.where(support[frame], values[frame], np.nan))
            tail_line.set_ydata(np.where(~support[frame], values[frame], np.nan))
            axis.set_xlim(*limits[frame])
        moment_text.set_text(
            rf"$\langle {coordinate_symbol}\rangle={mean[frame]:.3f},\ "
            rf"\sigma_{coordinate_symbol}={width[frame]:.3f}\ a_0$"
        )
        title.set_text(
            f"{particle_name}-coordinate dynamics | t={times[frame]:.4f} fs\n"
            rf"field x-window follows $\rho_{{{coordinate_symbol}}}\geq"
            rf"{support_floor:g}\rho_{{{coordinate_symbol},\max}}$; raw values"
        )
        return marginal_line, left_marker, right_marker, *(
            artist for _, line, tail, _ in profile_artists for artist in (line, tail)
        ), moment_text, title

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/f"{stem}.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=3200), dpi=dpi)
    else:
        if fmt == "mp4":
            print(f"ffmpeg을 찾지 못해 {stem} 영상을 GIF로 저장합니다.")
        path = outdir/f"{stem}.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"coordinate-focus dynamics 저장: {path}")
    return path
