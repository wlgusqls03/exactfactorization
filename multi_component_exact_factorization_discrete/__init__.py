"""Spatially discrete, time-continuous multi-component exact factorization.

This package is intentionally separate from the continuum-derived MCEF
implementation.  It factorizes the already discretized Shin--Metiu
Hamiltonian and never invokes a spatial Leibniz rule.
"""

from .core import (
    OFFSETS,
    discrete_born_huang_rhs,
    kinetic_weights,
    pnc_retract,
    reconstruct_coefficient_wavefunction,
)

__all__ = (
    "OFFSETS",
    "discrete_born_huang_rhs",
    "kinetic_weights",
    "pnc_retract",
    "reconstruct_coefficient_wavefunction",
)
