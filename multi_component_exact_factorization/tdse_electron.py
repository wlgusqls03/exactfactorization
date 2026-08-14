"""CPU reconstruction utilities for TDSE Born--Huang electron marginals."""

from __future__ import annotations

import numpy as np


def electron_marginal_from_bo(y_cpu, states, dq, dR, block_R=24):
    """Reconstruct the exact x marginal without forming full Psi(x,q,R)."""
    if block_R <= 0:
        raise ValueError("block_R must be positive")
    y_cpu = np.asarray(y_cpu)
    norm = np.sum(np.abs(y_cpu)**2, dtype=np.float64)*dq*dR
    if not np.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("TDSE frame has a non-positive/non-finite norm")
    density = np.zeros(states.shape[1], dtype=np.float64)
    for start in range(0, y_cpu.shape[2], block_R):
        stop = min(start+block_R, y_cpu.shape[2])
        psi_block = np.einsum(
            "nqR,nxqR->xqR", y_cpu[:, :, start:stop],
            states[:, :, :, start:stop], optimize=True,
        )
        density += np.sum(
            np.abs(psi_block)**2, axis=(1, 2), dtype=np.float64,
        )*dq*dR/norm
    return density
