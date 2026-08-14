"""Shared fixed-display-scale movie for physical particle marginals.

The movie deliberately changes only the visible y range. Stored probability
densities are neither peak-normalized nor clipped before plotting, so a peak
above the display ceiling is simply outside the axes and remains unchanged in
the underlying data.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from .report_plot_style import PARTICLE_COLORS, add_fixed_center_markers
from .visualize import selected_frames


def make_fixed_scale_marginal_animation(
    *,
    times_fs,
    particle_series,
    options,
    outdir,
    fps,
    max_frames,
    dpi,
    fmt,
    y_max=1.5,
    x_abs_max=12.0,
    title_prefix="Dynamics",
    stem="particle_marginals_fixed_scale",
):
    """Plot electron/proton/heavy marginals on one unmodified density scale."""
    times = np.asarray(times_fs, dtype=float)
    if times.ndim != 1 or not len(times):
        raise ValueError("times_fs must be a nonempty one-dimensional array")
    if not np.isfinite(y_max) or y_max <= 0.0:
        raise ValueError("marginal y maximum must be finite and positive")
    if not np.isfinite(x_abs_max) or x_abs_max <= 0.0:
        raise ValueError("marginal position maximum must be finite and positive")

    prepared = []
    for name, coordinate, density in particle_series:
        coordinate = np.asarray(coordinate, dtype=float)
        density = np.asarray(density, dtype=float)
        expected = (len(times), len(coordinate))
        if coordinate.ndim != 1 or density.shape != expected:
            raise ValueError(
                f"{name} marginal shape mismatch: {density.shape} != {expected}"
            )
        if not np.all(np.isfinite(density)):
            raise ValueError(f"{name} marginal contains non-finite values")
        spacing = float(coordinate[1]-coordinate[0]) if len(coordinate) > 1 else 1.0
        mass = np.sum(density, axis=1)*spacing
        safe_mass = np.maximum(mass, 1.0e-300)
        mean = np.sum(density*coordinate[None, :], axis=1)*spacing/safe_mass
        width = np.sqrt(np.maximum(
            np.sum(density*(coordinate[None, :]-mean[:, None])**2, axis=1)
            *spacing/safe_mass,
            0.0,
        ))
        prepared.append((name, coordinate, density, mean, width))
    if not prepared:
        raise ValueError("at least one particle marginal is required")

    frames = selected_frames(len(times), min(max_frames, len(times)))
    first = int(frames[0])
    fig, axis = plt.subplots(figsize=(14.8, 6.5), constrained_layout=True)
    lines = []
    for name, coordinate, density, _mean, _width in prepared:
        color = PARTICLE_COLORS.get(name)
        line, = axis.plot(
            coordinate, density[first], color=color, lw=2.25, label=name,
        )
        lines.append(line)

    add_fixed_center_markers(axis, options)
    available_min = min(float(coordinate[0]) for _, coordinate, *_ in prepared)
    available_max = max(float(coordinate[-1]) for _, coordinate, *_ in prepared)
    x_min = max(available_min, -float(x_abs_max))
    x_max = min(available_max, float(x_abs_max))
    axis.set(
        xlim=(x_min, x_max), ylim=(0.0, float(y_max)),
        xlabel=r"common position coordinate ($a_0$)",
        ylabel=r"probability density ($a_0^{-1}$)",
    )
    axis.set_title(
        "Electron, proton and heavy-nucleus marginals | fixed display scale",
        loc="left", fontweight="semibold",
    )
    axis.grid(alpha=0.18, linewidth=0.7)
    axis.tick_params(direction="in")
    axis.legend(frameon=False, ncol=max(1, len(prepared)), loc="upper left")
    moment_text = axis.text(
        0.995, 0.965, "", transform=axis.transAxes,
        ha="right", va="top", fontsize=8.4, color="0.18",
        bbox=dict(fc="white", ec="0.85", alpha=0.86, pad=3),
    )
    title = fig.suptitle("")

    def update(number):
        frame = int(frames[number])
        moments = []
        for line, (name, _coordinate, density, mean, width) in zip(lines, prepared):
            line.set_ydata(density[frame])
            symbol = {"electron": "x", "proton": "q", "heavy": "R"}.get(name, name)
            moments.append(
                rf"$⟨{symbol}⟩={mean[frame]:.3f},\ "
                rf"\sigma_{symbol}={width[frame]:.3f}$"
            )
        moment_text.set_text("   |   ".join(moments)+r"  ($a_0$)")
        title.set_text(
            f"{title_prefix} | t={times[frame]:.4f} fs\n"
            f"raw densities; display window only: position ±{x_abs_max:g} $a_0$, "
            f"density ≤ {y_max:g} $a_0^{{-1}}$"
        )
        return *lines, moment_text, title

    update(0)
    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if fmt == "mp4" and shutil.which("ffmpeg"):
        path = outdir/f"{stem}.mp4"
        animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=dpi)
    else:
        if fmt == "mp4":
            print(f"ffmpeg을 찾지 못해 {stem} 영상을 GIF로 저장합니다.")
        path = outdir/f"{stem}.gif"
        animation.save(path, writer=PillowWriter(fps=fps), dpi=min(dpi, 110))
    plt.close(fig)
    print(f"fixed-scale particle marginals 저장: {path}")
    return path
