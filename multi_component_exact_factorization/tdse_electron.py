"""CPU reconstruction utilities for TDSE Born--Huang electron marginals."""

from __future__ import annotations

import numpy as np


def electronic_reduced_densities_from_bo(
    y_cpu, states, dq, dR, block_R=24, *,
    electron_marginal=True, electron_proton=True,
):
    """Reconstruct reduced electronic densities without forming full ``Psi``.

    ``electron_proton`` is the diagonal two-body density
    ``rho_ep(x,q)=integral dR |Psi(x,q,R)|^2``.  Reconstructing the coherent
    BO sum before taking its magnitude retains all inter-state cross terms.
    The R-block loop bounds the largest temporary at ``(nx,nq,block_R)``.
    """
    if block_R <= 0:
        raise ValueError("block_R must be positive")
    y_cpu = np.asarray(y_cpu)
    states = np.asarray(states)
    if y_cpu.ndim != 3 or states.ndim != 4:
        raise ValueError("expected y(n_BO,nq,nR) and states(n_BO,nx,nq,nR)")
    if (
        states.shape[0] != y_cpu.shape[0]
        or states.shape[2:] != y_cpu.shape[1:]
    ):
        raise ValueError(f"BO coefficient/basis shape mismatch: {y_cpu.shape}, {states.shape}")
    if not electron_marginal and not electron_proton:
        return {}
    norm = np.sum(np.abs(y_cpu)**2, dtype=np.float64)*dq*dR
    if not np.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("TDSE frame has a non-positive/non-finite norm")
    result = {}
    if electron_marginal:
        result["electron_density"] = np.zeros(states.shape[1], dtype=np.float64)
    if electron_proton:
        result["electron_proton_density"] = np.zeros(
            (states.shape[1], y_cpu.shape[1]), dtype=np.float64,
        )
    for start in range(0, y_cpu.shape[2], block_R):
        stop = min(start+block_R, y_cpu.shape[2])
        psi_block = np.einsum(
            "nqR,nxqR->xqR", y_cpu[:, :, start:stop],
            states[:, :, :, start:stop], optimize=True,
        )
        probability = np.abs(psi_block)**2
        if electron_marginal:
            result["electron_density"] += np.sum(
                probability, axis=(1, 2), dtype=np.float64,
            )*dq*dR/norm
        if electron_proton:
            result["electron_proton_density"] += np.sum(
                probability, axis=2, dtype=np.float64,
            )*dR/norm
    return result


def electron_marginal_from_bo(y_cpu, states, dq, dR, block_R=24):
    """Reconstruct the exact x marginal without forming full Psi(x,q,R)."""
    return electronic_reduced_densities_from_bo(
        y_cpu, states, dq, dR, block_R,
        electron_marginal=True, electron_proton=False,
    )["electron_density"]


def electron_proton_density_from_bo(y_cpu, states, dq, dR, block_R=24):
    """Return ``integral dR |Psi(x,q,R)|^2`` with coherent BO cross terms."""
    return electronic_reduced_densities_from_bo(
        y_cpu, states, dq, dR, block_R,
        electron_marginal=False, electron_proton=True,
    )["electron_proton_density"]
