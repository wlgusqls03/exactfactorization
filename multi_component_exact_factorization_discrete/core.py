"""NumPy reference algebra for the discretize-first Born--Huang MCEF.

The primary variables are ``C_j(q,R)``, ``Lambda_R(q)`` and ``chi(R)``.
For a normalized factorization ``Y_j=C_j Lambda chi``, the routines below
implement Eqs. (23)--(25) and (65) of
``paper/Discrete_Multi_Component_Exact_Factorization.pdf``.  Nuclear kinetic
energies are periodic five-point matrices.  Their action is assembled from
neighbor BO overlaps, not from a finite-difference product expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multi_component_exact_factorization.core import (
    flat_top_support_mask,
)


OFFSETS = (-2, -1, 1, 2)


@dataclass
class DiscreteRHSResult:
    dc: np.ndarray
    dlam: np.ndarray
    dchi: np.ndarray
    fields: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def kinetic_weights(spacing: float, mass: float) -> dict[int, float]:
    """Periodic fourth-order five-point ``-D2/(2M)`` coefficients."""
    if spacing <= 0.0 or mass <= 0.0:
        raise ValueError("spacing and mass must be positive")
    scale = 1.0/(24.0*mass*spacing**2)
    return {-2: scale, -1: -16.0*scale, 0: 30.0*scale,
            1: -16.0*scale, 2: scale}


def _backward_link(forward: np.ndarray, coordinate_axis: int, offset: int):
    return np.roll(
        np.swapaxes(np.conj(forward), 0, 1),
        offset, axis=coordinate_axis+1,
    )


def neighbor_transports(
    coefficients: np.ndarray, basis, coordinate_axis: int,
) -> dict[int, np.ndarray]:
    """Evaluate ``S_BO(g,g+s) C(g+s)`` for the four stencil bonds."""
    if coordinate_axis == 1:
        link1, link2 = basis.link_q1, basis.link_q2
    elif coordinate_axis == 2:
        link1, link2 = basis.link_R1, basis.link_R2
    else:
        raise ValueError("coordinate_axis must be q(1) or R(2)")
    if link1 is None or link2 is None:
        raise ValueError("Born--Huang overlap links are required")
    back1 = _backward_link(link1, coordinate_axis, 1)
    back2 = _backward_link(link2, coordinate_axis, 2)

    def action(link, values):
        return np.einsum("abqR,bqR->aqR", link, values, optimize=True)

    return {
        -2: action(back2, np.roll(coefficients, 2, axis=coordinate_axis)),
        -1: action(back1, np.roll(coefficients, 1, axis=coordinate_axis)),
        1: action(link1, np.roll(coefficients, -1, axis=coordinate_axis)),
        2: action(link2, np.roll(coefficients, -2, axis=coordinate_axis)),
    }


def _safe_flat_top_inverse(
    value: np.ndarray, density: np.ndarray, relative_on: float,
    transition_decades: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = flat_top_support_mask(
        density, relative_on, transition_decades
    )
    value_density = np.abs(value)**2
    # The conditional chart is undefined at an exact marginal node.  Even an
    # explicitly unmasked run must define its generalized inverse as zero
    # there instead of forming 1/0 (or 0/0) and contaminating the ODE.
    active = (mask > 0.0) & (value_density > 0.0)
    denominator = np.where(active, value_density, 1.0)
    inverse = np.where(active, mask*np.conj(value)/denominator, 0.0)
    return inverse, mask


def _l2(values: np.ndarray, dq: float, dR: float) -> np.ndarray:
    return np.asarray(np.sqrt(np.sum(np.abs(values)**2)*dq*dR))


def _suppressed_probability(
    density: np.ndarray, mask: np.ndarray, volume: float,
) -> np.ndarray:
    total = np.sum(density)*volume
    if total <= 0.0:
        return np.asarray(0.0)
    return np.asarray(np.sum(density*(1.0-mask))*volume/total)


def reconstruct_coefficient_wavefunction(
    coefficients: np.ndarray, lam: np.ndarray, chi: np.ndarray,
) -> np.ndarray:
    """Return the BO coefficient representation ``Y=C Lambda chi``."""
    return coefficients*(lam*chi[None, :])[None, :, :]


def discrete_tdse_action(
    coefficient_wavefunction: np.ndarray, model, basis,
) -> np.ndarray:
    """Apply the same spatially discrete BO Hamiltonian directly to ``Y``.

    ``Y_j(q,R)`` is the full molecular wavefunction in the local BO basis.
    This action is the product-rule-free TDSE oracle recombined by the
    discrete MCEF equations when their generalized inverses are unmodified.
    """
    y = np.asarray(coefficient_wavefunction)
    if y.ndim != 3 or y.shape != basis.energies.shape:
        raise ValueError("expected Y(NBO,nq,nR) matching BO energies")
    q_weights = kinetic_weights(model.dq, model.proton_mass)
    R_weights = kinetic_weights(model.dR, model.heavy_mass)
    action = (basis.energies+q_weights[0]+R_weights[0])*y
    q_transports = neighbor_transports(y, basis, 1)
    R_transports = neighbor_transports(y, basis, 2)
    for offset in OFFSETS:
        action = action+q_weights[offset]*q_transports[offset]
        action = action+R_weights[offset]*R_transports[offset]
    return action


def full_step_discrete_tdse(
    coefficient_wavefunction: np.ndarray, dt: float, model, basis,
) -> np.ndarray:
    """One classical RK4 step for ``i dY/dt = H_h Y``."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    def rhs(values):
        return -1j*discrete_tdse_action(values, model, basis)

    k1 = rhs(coefficient_wavefunction)
    k2 = rhs(coefficient_wavefunction+0.5*dt*k1)
    k3 = rhs(coefficient_wavefunction+0.5*dt*k2)
    k4 = rhs(coefficient_wavefunction+dt*k3)
    return coefficient_wavefunction+dt*(k1+2.0*k2+2.0*k3+k4)/6.0


def pnc_retract(
    coefficients: np.ndarray, lam: np.ndarray, chi: np.ndarray,
    dq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Strict product-preserving retraction to both PNC manifolds.

    This reference routine is deliberately strict.  The GPU production path
    adds a support gate so that undefined deep-tail factor charts are not
    rescaled by a large amount.
    """
    before = reconstruct_coefficient_wavefunction(coefficients, lam, chi)
    c_norm2 = np.sum(np.abs(coefficients)**2, axis=0)
    c_scale = np.sqrt(np.maximum(c_norm2, 1.0e-300))
    coefficients = coefficients/c_scale[None, :, :]
    lam = lam*c_scale
    lam_norm2 = np.sum(np.abs(lam)**2, axis=0)*dq
    lam_scale = np.sqrt(np.maximum(lam_norm2, 1.0e-300))
    lam = lam/lam_scale[None, :]
    chi = chi*lam_scale
    after = reconstruct_coefficient_wavefunction(coefficients, lam, chi)
    return coefficients, lam, chi, {
        "max_raw_pnc_phi_error": np.asarray(np.max(np.abs(c_norm2-1.0))),
        "max_raw_pnc_lam_error": np.asarray(np.max(np.abs(lam_norm2-1.0))),
        "max_product_change": np.asarray(np.max(np.abs(after-before))),
    }


def discrete_born_huang_rhs(
    coefficients: np.ndarray,
    lam: np.ndarray,
    chi: np.ndarray,
    model,
    basis,
    *,
    flat_top_on_phi: float = 0.0,
    flat_top_on_lam: float = 0.0,
    transition_decades: float = 3.0,
    horizontal_correction: bool = True,
) -> DiscreteRHSResult:
    """Evaluate the regularized discrete Born--Huang MCEF vector field.

    Setting both flat-top onsets to zero gives the unregularized equation on
    nonzero marginal support.  A nonzero onset replaces only the singular
    mass inverses by ``W/F`` and ``W/chi``; local scalar/Hamiltonian terms and
    overlap links remain unmodified.
    """
    c = np.asarray(coefficients)
    lam = np.asarray(lam)
    chi = np.asarray(chi)
    if c.ndim != 3 or lam.shape != c.shape[1:] or chi.shape != (c.shape[2],):
        raise ValueError("expected C(NBO,nq,nR), Lambda(nq,nR), chi(nR)")

    norm_c = np.sum(np.abs(c)**2, axis=0)
    norm_c_safe = np.maximum(norm_c, 1.0e-300)
    norm_lam = np.sum(np.abs(lam)**2, axis=0)*model.dq
    norm_lam_safe = np.maximum(norm_lam, 1.0e-300)
    F = lam*chi[None, :]
    rho_qR = norm_c*np.abs(F)**2
    rho_R = np.sum(rho_qR, axis=0)*model.dq
    inverse_F, mask_phi = _safe_flat_top_inverse(
        F, rho_qR, flat_top_on_phi, transition_decades
    )
    inverse_chi, mask_lam = _safe_flat_top_inverse(
        chi, rho_R, flat_top_on_lam, transition_decades
    )
    # These are the weights actually used by the generalized inverses.  They
    # equal the requested masks on nonzero support, but are exactly zero at an
    # exact marginal node even when the user requested an unmasked run.
    effective_mask_phi = F*inverse_F
    effective_mask_lam = chi*inverse_chi

    eps1_complex = np.sum(
        np.conj(c)*basis.energies*c, axis=0
    )/norm_c_safe
    epsilon_1 = eps1_complex.real
    q_weights = kinetic_weights(model.dq, model.proton_mass)
    R_weights = kinetic_weights(model.dR, model.heavy_mass)
    direct_action = basis.energies*c*F[None, :, :]
    direct_action += (q_weights[0]+R_weights[0])*c*F[None, :, :]
    coupling_c = np.zeros_like(c)
    q_action_lam = q_weights[0]*lam

    q_transports = neighbor_transports(c, basis, 1)
    for offset in OFFSETS:
        transport = q_transports[offset]
        overlap = np.sum(np.conj(c)*transport, axis=0)/norm_c_safe
        F_neighbor = np.roll(F, -offset, axis=0)
        lam_neighbor = np.roll(lam, -offset, axis=0)
        weight = q_weights[offset]
        coupling_c += weight*F_neighbor[None, :, :]*(
            transport-overlap[None, :, :]*c
        )
        q_action_lam += weight*overlap*lam_neighbor
        direct_action += weight*transport*F_neighbor[None, :, :]

    hpr_local = epsilon_1*lam+q_action_lam
    eps2_complex = (
        np.sum(np.conj(lam)*hpr_local, axis=0)*model.dq/norm_lam_safe
    )
    epsilon_2 = eps2_complex.real
    coupling_lam = np.zeros_like(lam)
    heavy_action = R_weights[0]*chi
    R_transports = neighbor_transports(c, basis, 2)
    nearest_sphi_q = None
    nearest_sphi_R = None
    nearest_sgamma_R = None
    for offset in OFFSETS:
        transport = R_transports[offset]
        overlap = np.sum(np.conj(c)*transport, axis=0)/norm_c_safe
        F_neighbor = np.roll(F, -offset, axis=1)
        lam_neighbor = np.roll(lam, -offset, axis=1)
        chi_neighbor = np.roll(chi, -offset, axis=0)
        weight = R_weights[offset]
        coupling_c += weight*F_neighbor[None, :, :]*(
            transport-overlap[None, :, :]*c
        )
        transported_lam = overlap*lam_neighbor
        overlap_gamma = (
            np.sum(np.conj(lam)*transported_lam, axis=0)*model.dq
            /norm_lam_safe
        )
        coupling_lam += weight*chi_neighbor[None, :]*(
            transported_lam-overlap_gamma[None, :]*lam
        )
        heavy_action += weight*overlap_gamma*chi_neighbor
        direct_action += weight*transport*F_neighbor[None, :, :]
        if offset == 1:
            nearest_sphi_R = overlap
            nearest_sgamma_R = overlap_gamma

    # q nearest-neighbor link is reconstructed once for native diagnostics.
    q_plus = q_transports[1]
    nearest_sphi_q = np.sum(np.conj(c)*q_plus, axis=0)/norm_c_safe

    dc = -1j*((basis.energies-epsilon_1[None, :, :])*c
              +inverse_F[None, :, :]*coupling_c)
    dlam = -1j*((epsilon_1-epsilon_2[None, :])*lam+q_action_lam
                +inverse_chi[None, :]*coupling_lam)
    dchi = -1j*(epsilon_2*chi+heavy_action)

    raw_parallel_c = np.sum(np.conj(c)*dc, axis=0)/norm_c_safe
    if horizontal_correction:
        dc = dc-raw_parallel_c[None, :, :]*c
        dlam = dlam+raw_parallel_c*lam
    raw_parallel_lam = (
        np.sum(np.conj(lam)*dlam, axis=0)*model.dq/norm_lam_safe
    )
    if horizontal_correction:
        dlam = dlam-raw_parallel_lam[None, :]*lam
        dchi = dchi+raw_parallel_lam*chi

    Y = reconstruct_coefficient_wavefunction(c, lam, chi)
    dY = dc*F[None, :, :]+c*(
        dlam*chi[None, :]+lam*dchi[None, :]
    )[None, :, :]
    recombination_residual = 1j*dY-direct_action
    predicted_mask_residual = (
        (effective_mask_phi-1.0)[None, :, :]*coupling_c
        +c*(effective_mask_lam-1.0)[None, :]*coupling_lam[None, :, :]
    )
    unexplained = recombination_residual-predicted_mask_residual
    target_l2 = max(float(_l2(direct_action, model.dq, model.dR)), 1.0e-300)
    fields = {
        "epsilon_1": epsilon_1,
        "epsilon_2": epsilon_2,
        "mask_phi": mask_phi,
        "mask_lam": mask_lam,
        "sphi_q1": nearest_sphi_q,
        "sphi_R1": nearest_sphi_R,
        "sgamma_R1": nearest_sgamma_R,
    }
    diagnostics = {
        "max_raw_horizontal_phi": np.asarray(np.max(np.abs(raw_parallel_c))),
        "max_raw_horizontal_lam": np.asarray(np.max(np.abs(raw_parallel_lam))),
        "suppressed_probability_phi": _suppressed_probability(
            rho_qR, mask_phi, model.dq*model.dR
        ),
        "suppressed_probability_lam": _suppressed_probability(
            rho_R, mask_lam, model.dR
        ),
        "recombination_residual_l2": _l2(
            recombination_residual, model.dq, model.dR
        ),
        "predicted_mask_residual_l2": _l2(
            predicted_mask_residual, model.dq, model.dR
        ),
        "unexplained_residual_l2": _l2(unexplained, model.dq, model.dR),
        "relative_unexplained_residual": np.asarray(
            float(_l2(unexplained, model.dq, model.dR))/target_l2
        ),
        "epsilon_1_imaginary_defect": np.asarray(
            np.max(np.abs(eps1_complex.imag))
        ),
        "epsilon_2_imaginary_defect": np.asarray(
            np.max(np.abs(eps2_complex.imag))
        ),
        "full_norm_rate": np.asarray(
            2.0*np.real(np.sum(np.conj(Y)*dY)*model.dq*model.dR)
        ),
    }
    return DiscreteRHSResult(dc, dlam, dchi, fields, diagnostics)
