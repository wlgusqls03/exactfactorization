"""Decade guides and uniformly spaced minor density contours."""
import numpy as np


DECADE_COLORS = ('#008f39', '#c000c0', '#d89000', '#00a6a6', '#6842c2')


def decade_levels(cutoff, upper):
    """5% of each decade's upper bound; final decade stops at its midpoint.

    For cutoff=1e-3: the final black contours run .0095,...,.005.
    Colored decade boundaries include the cutoff, but no contour is below it.
    """
    if not (np.isfinite(cutoff) and np.isfinite(upper) and 0 < cutoff < upper):
        return np.array([]), np.array([])
    bottom = int(np.floor(np.log10(cutoff)+1e-12))
    top = int(np.ceil(np.log10(upper)-1e-12))
    major = 10.**np.arange(bottom, top+1)
    major = major[(major >= cutoff*(1-1e-12)) & (major <= upper*(1+1e-12))]
    minor = []
    for exponent in range(bottom+1, top+1):
        # Start at .95 H; .10 H is the next colored boundary.
        smallest = 10 if exponent == bottom+1 else 3
        minor.extend(10.**exponent*np.arange(smallest, 20)*.05)
    minor = np.asarray(minor)
    minor = minor[(minor > cutoff) & (minor < upper)]
    return np.sort(major), np.sort(minor)


def decade_color(value):
    # Shared absolute masking boundary: black on both signed maps and legends.
    # Keep the existing major-contour linewidth and all other decade colors.
    if np.isclose(value, 1e-3, rtol=1e-10, atol=0):
        return 'black'
    return DECADE_COLORS[int(round(-np.log10(value))) % len(DECADE_COLORS)]


def automatic_absolute_cutoff(initial_density, relative_floor):
    """Fixed power of ten at/below the initial relative cutoff, never per-frame."""
    value = max(float(np.max(initial_density))*relative_floor, 1e-300)
    return 10.**np.floor(np.log10(value))
