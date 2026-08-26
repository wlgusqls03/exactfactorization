"""Shared visual conventions for continuum and discrete MCEF reports.

This module deliberately contains plotting-only policy.  Keeping the same
palette, orientation and trajectory-wide limits here prevents two reports of
the same saved field from suggesting different physics merely because each
Matplotlib routine autoscaled independently.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


COLORS = ("#2878B5", "#E07A2D", "#3A9654", "#B05279", "#7B61A8", "#8C6D31")
PARTICLE_COLORS = {
    "electron": COLORS[0],
    "proton": COLORS[1],
    "heavy": COLORS[2],
}
CURRENT_COLOR = COLORS[0]
FORCE_COLOR = COLORS[3]
HEAVY_DENSITY_COLOR = "0.45"
JOINT_CMAP = "magma"
SCALAR_CMAP = "viridis"
# Keep support-masked physical fields visually distinct from the gray tail.
# Their numerical limits remain unchanged; this only changes value-to-color
# mapping so that a near-zero signed field no longer looks like empty space.
SIGNED_CMAP = SCALAR_CMAP
LINK_CMAP = SCALAR_CMAP
MASK_COLOR = "#D7DCE0"


def masked_cmap(name):
    """Copy a colormap with one consistent low-support cell color."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(MASK_COLOR)
    return cmap


def joint_density_limit(joint):
    """One color maximum for a complete trajectory of normalized densities."""
    return max(float(np.nanmax(np.asarray(joint, float))), 1.0e-300)


def density_display_alpha(density, floor=1.0e-4, transition_decades=2.0):
    """Smooth plotting opacity from gray tail to fully occupied support.

    Values at or above ``floor`` of the instantaneous density peak are fully
    opaque.  Values ``transition_decades`` lower are gray.  A quintic
    smoothstep between them avoids a one-frame flash when a cell barely
    crosses a hard cutoff.
    """
    density = np.asarray(density, float)
    # Opacity is a rendering attribute, not a propagated physical field.
    # Invalid/negative density cells are shown as the gray background while
    # the archived arrays remain untouched.
    density = np.where(np.isfinite(density) & (density > 0.0), density, 0.0)
    relative = density/max(float(np.max(density)), 1.0e-300)
    upper = np.log10(max(float(floor), 1.0e-300))
    lower = upper-max(float(transition_decades), 1.0e-12)
    scaled = np.clip(
        (np.log10(np.maximum(relative, 1.0e-300))-lower)/(upper-lower),
        0.0, 1.0,
    )
    opacity = scaled**3*(scaled*(6.0*scaled-15.0)+10.0)
    # The quintic is mathematically in [0, 1], but its factored floating-point
    # evaluation may overshoot 1 by a few ulps (observed: 1+1.6e-15).  Unlike
    # color data, Matplotlib validates alpha strictly, so close the interval.
    return np.clip(opacity, 0.0, 1.0)


def density_weighted_shift(values, density, floor=1.0e-4):
    """Remove one smoothly weighted scalar offset on occupied support."""
    values = np.asarray(values, float)
    density = np.maximum(np.asarray(density, float), 0.0)
    support = (
        np.isfinite(values)
        & (density >= float(floor)*max(float(np.max(density)), 1.0e-300))
    )
    if not np.any(support):
        return values.copy()
    weights = density[support]
    offset = np.sum(weights*values[support])/max(float(np.sum(weights)), 1.0e-300)
    return values-offset


def color_y_axis(axis, color, label):
    """Tie a twin-axis label, ticks and visible spine to its data curve."""
    axis.set_ylabel(label, color=color)
    axis.tick_params(axis="y", colors=color)
    side = "right" if axis.yaxis.get_label_position() == "right" else "left"
    axis.spines[side].set_color(color)


def add_fixed_center_markers(axis, options):
    """Mark active fixed charges and the heavy harmonic-trap center."""
    positions = []
    for key, charge_key in (
        ("left_position", "left_charge"),
        ("right_position", "right_charge"),
    ):
        value = options.get(key)
        if value is None:
            continue
        value = float(value)
        charge = float(options.get(charge_key, 1.0))
        if (
            charge != 0.0 and np.isfinite(value)
            and not any(np.isclose(value, old) for old in positions)
        ):
            positions.append(value)
    for index, position in enumerate(positions):
        axis.axvline(
            position, color="0.42", lw=1.0, ls=":", alpha=0.82,
            label=("fixed + charge" if index == 0 else None), zorder=1,
        )
    trap_alpha = float(options.get("heavy_trap_alpha", 0.0))
    trap_center = options.get("heavy_trap_center")
    if trap_alpha > 0.0 and trap_center is not None:
        trap_center = float(trap_center)
        if np.isfinite(trap_center):
            axis.axvline(
                trap_center, color="#6A3D9A", lw=1.0, ls="--", alpha=0.78,
                label="heavy trap center", zorder=1,
            )
            if not any(np.isclose(trap_center, old) for old in positions):
                positions.append(trap_center)
    return positions
