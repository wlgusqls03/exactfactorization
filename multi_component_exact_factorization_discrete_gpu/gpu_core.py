"""Optimized CUDA algebra for discrete Born--Huang MCEF propagation."""

from __future__ import annotations

from dataclasses import dataclass

from multi_component_exact_factorization_discrete.core import (
    OFFSETS,
    kinetic_weights,
)
from multi_component_exact_factorization_gpu.gpu_born_huang import (
    neighbor_transports,
    pnc_project_coefficients,
)
from multi_component_exact_factorization_gpu.gpu_core import (
    GPUModel,
    cp,
    flat_top_support_mask,
    suppressed_probability,
)


@dataclass
class DiscreteGPUResult:
    dc: cp.ndarray
    dlam: cp.ndarray
    dchi: cp.ndarray
    fields: dict[str, cp.ndarray]
    diagnostics: dict[str, cp.ndarray]


def _safe_masked_inverse(value, density, relative_on, decades, model):
    mask = flat_top_support_mask(density, relative_on, decades, model)
    value_density = cp.real(value*cp.conj(value))
    active = (mask > 0.0) & (value_density > 0.0)
    # Do not evaluate a division at inactive nodes.  CuPy's where evaluates
    # both already-formed branches, so the denominator itself is selected.
    denominator = cp.where(active, value_density, 1.0)
    inverse = cp.where(
        active, mask*cp.conj(value)/denominator, 0.0
    ).astype(model.complex_dtype, copy=False)
    return inverse, mask


def make_discrete_gpu_model(cpu_model):
    """Upload only data used by the discrete coefficient equations.

    The BO energies and overlap links live in ``GPUBornHuangBasis``.  Unlike
    the direct-grid solver, this backend never reads the full ``V(x,q,R)`` or
    electronic sine-kinetic arrays, so allocating them on the GPU would waste
    roughly ``8*nx*nq*nR`` bytes without changing a single operation.
    """
    return GPUModel(
        dx=cpu_model.dx, dq=cpu_model.dq, dR=cpu_model.dR,
        proton_mass=cpu_model.proton_mass, heavy_mass=cpu_model.heavy_mass,
        potential=cp.empty((0,), dtype=cp.float64),
        kinetic_energies=cp.empty((0,), dtype=cp.float64),
        real_dtype=cp.float64, complex_dtype=cp.complex128,
        reduction_real_dtype=cp.float64,
        reduction_complex_dtype=cp.complex128,
        coupling_mask_backend="flat_top",
        flat_top_on_phi=float(cpu_model.flat_top_on_phi),
        flat_top_on_lam=float(cpu_model.flat_top_on_lam),
        flat_top_transition_decades=float(
            cpu_model.flat_top_transition_decades
        ),
        projection_tau_phi=float(cpu_model.projection_tau_phi),
        projection_tau_lam=float(cpu_model.projection_tau_lam),
        projection_tau_chi=float(cpu_model.projection_tau_chi),
        projection_support_epsilon=float(
            cpu_model.projection_support_epsilon
        ),
        deep_tail_zero_threshold=float(cpu_model.deep_tail_zero_threshold),
    )


def _l2(values, model):
    return cp.sqrt(cp.sum(
        cp.real(values*cp.conj(values)),
        dtype=model.reduction_real_dtype,
    )*model.dq*model.dR)


def _shift(values, offset, axis):
    return cp.roll(values, -int(offset), axis=axis)


def discrete_tdse_action_gpu(coefficient_wavefunction, model, basis):
    """Apply the direct BO-coefficient TDSE Hamiltonian on the GPU."""
    y = coefficient_wavefunction
    q_weights = kinetic_weights(model.dq, model.proton_mass)
    R_weights = kinetic_weights(model.dR, model.heavy_mass)
    action = (basis.energies+q_weights[0]+R_weights[0])*y
    q_transports = neighbor_transports(y, basis, 1)
    R_transports = neighbor_transports(y, basis, 2)
    for index, offset in enumerate(OFFSETS):
        action += q_weights[offset]*q_transports[index]
        action += R_weights[offset]*R_transports[index]
    return action


def full_step_discrete_tdse_gpu(coefficient_wavefunction, dt, model, basis):
    """One classical RK4 step for the direct discrete TDSE reference."""
    def rhs(values):
        return -1j*discrete_tdse_action_gpu(values, model, basis)

    y = coefficient_wavefunction
    k1 = rhs(y)
    k2 = rhs(y+0.5*dt*k1)
    k3 = rhs(y+0.5*dt*k2)
    k4 = rhs(y+dt*k3)
    return y+dt*(k1+2.0*k2+2.0*k3+k4)/6.0


def discrete_rhs_gpu(
    coefficients, lam, chi, model, basis, *, collect_diagnostics=False,
):
    """Evaluate Eqs. (23)--(25)/(65) with flat-top mass inverses.

    The expensive BO contractions are exactly the same validated fused
    overlap-link kernel as the continuum Born--Huang backend.  Only two
    transport launches (q and R) are required per RHS evaluation.
    """
    c = coefficients
    c_norm2 = cp.sum(
        cp.real(c*cp.conj(c)), axis=0,
        dtype=model.reduction_real_dtype,
    )
    tiny = cp.asarray(1.0e-300, dtype=model.reduction_real_dtype)
    c_norm_safe = cp.maximum(c_norm2, tiny)
    lam_norm2 = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dq
    lam_norm_safe = cp.maximum(lam_norm2, tiny)
    F = lam*chi[None, :]
    F_density = cp.real(F*cp.conj(F))
    rho_qR = c_norm2*F_density
    rho_R = cp.sum(
        rho_qR, axis=0, dtype=model.reduction_real_dtype
    )*model.dq
    inverse_F, mask_phi = _safe_masked_inverse(
        F, rho_qR, model.flat_top_on_phi,
        model.flat_top_transition_decades, model,
    )
    inverse_chi, mask_lam = _safe_masked_inverse(
        chi, rho_R, model.flat_top_on_lam,
        model.flat_top_transition_decades, model,
    )
    # Match the residual prediction to the generalized inverse actually used
    # by the RHS.  At nonzero sites these equal mask_phi/mask_lam; at an exact
    # node they are zero even in the explicitly unmasked limit.
    effective_mask_phi = F*inverse_F
    effective_mask_lam = chi*inverse_chi

    eps1_complex = cp.sum(
        cp.conj(c)*basis.energies*c, axis=0,
        dtype=model.reduction_complex_dtype,
    )/c_norm_safe
    epsilon_1 = eps1_complex.real.astype(model.real_dtype, copy=False)
    q_weights = kinetic_weights(model.dq, model.proton_mass)
    R_weights = kinetic_weights(model.dR, model.heavy_mass)
    coupling_c = cp.zeros_like(c)
    q_action_lam = q_weights[0]*lam
    direct_action = None
    max_regularized_F_ratio = cp.asarray(
        0.0, dtype=model.reduction_real_dtype
    )
    max_regularized_chi_ratio = cp.asarray(
        0.0, dtype=model.reduction_real_dtype
    )
    if collect_diagnostics:
        direct_action = (
            basis.energies*c*F[None, :, :]
            +(q_weights[0]+R_weights[0])*c*F[None, :, :]
        )

    q_transports = neighbor_transports(c, basis, 1)
    sphi_q1 = None
    for index, offset in enumerate(OFFSETS):
        transport = q_transports[index]
        overlap = cp.sum(
            cp.conj(c)*transport, axis=0,
            dtype=model.reduction_complex_dtype,
        )/c_norm_safe
        F_neighbor = _shift(F, offset, 0)
        lam_neighbor = _shift(lam, offset, 0)
        weight = q_weights[offset]
        coupling_c += weight*F_neighbor[None, :, :]*(
            transport-overlap[None, :, :]*c
        )
        q_action_lam += weight*overlap*lam_neighbor
        if collect_diagnostics:
            direct_action += weight*transport*F_neighbor[None, :, :]
            max_regularized_F_ratio = cp.maximum(
                max_regularized_F_ratio,
                cp.max(cp.abs(inverse_F*F_neighbor)),
            )
        if offset == 1:
            sphi_q1 = overlap

    hpr_local = epsilon_1*lam+q_action_lam
    eps2_complex = cp.sum(
        cp.conj(lam)*hpr_local, axis=0,
        dtype=model.reduction_complex_dtype,
    )*model.dq/lam_norm_safe
    epsilon_2 = eps2_complex.real.astype(model.real_dtype, copy=False)
    coupling_lam = cp.zeros_like(lam)
    heavy_action = R_weights[0]*chi
    R_transports = neighbor_transports(c, basis, 2)
    sphi_R1 = None
    sgamma_R1 = None
    for index, offset in enumerate(OFFSETS):
        transport = R_transports[index]
        overlap = cp.sum(
            cp.conj(c)*transport, axis=0,
            dtype=model.reduction_complex_dtype,
        )/c_norm_safe
        F_neighbor = _shift(F, offset, 1)
        lam_neighbor = _shift(lam, offset, 1)
        chi_neighbor = _shift(chi, offset, 0)
        weight = R_weights[offset]
        coupling_c += weight*F_neighbor[None, :, :]*(
            transport-overlap[None, :, :]*c
        )
        transported_lam = overlap*lam_neighbor
        overlap_gamma = cp.sum(
            cp.conj(lam)*transported_lam, axis=0,
            dtype=model.reduction_complex_dtype,
        )*model.dq/lam_norm_safe
        coupling_lam += weight*chi_neighbor[None, :]*(
            transported_lam-overlap_gamma[None, :]*lam
        )
        heavy_action += weight*overlap_gamma*chi_neighbor
        if collect_diagnostics:
            direct_action += weight*transport*F_neighbor[None, :, :]
            max_regularized_F_ratio = cp.maximum(
                max_regularized_F_ratio,
                cp.max(cp.abs(inverse_F*F_neighbor)),
            )
            max_regularized_chi_ratio = cp.maximum(
                max_regularized_chi_ratio,
                cp.max(cp.abs(inverse_chi*chi_neighbor)),
            )
        if offset == 1:
            sphi_R1 = overlap
            sgamma_R1 = overlap_gamma

    dc = -1j*((basis.energies-epsilon_1[None, :, :])*c
              +inverse_F[None, :, :]*coupling_c)
    dlam = -1j*((epsilon_1-epsilon_2[None, :])*lam+q_action_lam
                +inverse_chi[None, :]*coupling_lam)
    dchi = -1j*(epsilon_2*chi+heavy_action)

    parallel_c = cp.sum(
        cp.conj(c)*dc, axis=0, dtype=model.reduction_complex_dtype,
    )/c_norm_safe
    dc = dc-parallel_c[None, :, :]*c
    dlam = dlam+parallel_c*lam
    parallel_lam = cp.sum(
        cp.conj(lam)*dlam, axis=0,
        dtype=model.reduction_complex_dtype,
    )*model.dq/lam_norm_safe
    dlam = dlam-parallel_lam[None, :]*lam
    dchi = dchi+parallel_lam*chi

    fields = {
        "epsilon_1": epsilon_1,
        "epsilon_2": epsilon_2,
        "mask_phi": mask_phi,
        "mask_lam": mask_lam,
    }
    diagnostics = {}
    if collect_diagnostics:
        fields.update(
            sphi_q1=sphi_q1, sphi_R1=sphi_R1,
            sgamma_R1=sgamma_R1,
        )
        dY = dc*F[None, :, :]+c*(
            dlam*chi[None, :]+lam*dchi[None, :]
        )[None, :, :]
        residual = 1j*dY-direct_action
        predicted = (
            (effective_mask_phi-1.0)[None, :, :]*coupling_c
            +c*(effective_mask_lam-1.0)[None, :]*coupling_lam[None, :, :]
        )
        unexplained = residual-predicted
        target_l2 = cp.maximum(_l2(direct_action, model), tiny)
        probability = cp.maximum(
            cp.sum(rho_qR, dtype=model.reduction_real_dtype)
            *model.dq*model.dR,
            tiny,
        )
        heavy_probability = cp.maximum(
            cp.sum(rho_R, dtype=model.reduction_real_dtype)*model.dR,
            tiny,
        )
        diagnostics = {
            "max_raw_horizontal_phi": cp.max(cp.abs(parallel_c)),
            "max_raw_horizontal_lam": cp.max(cp.abs(parallel_lam)),
            "max_raw_pnc_phi_error": cp.max(cp.abs(c_norm2-1.0)),
            "max_raw_pnc_lam_error": cp.max(cp.abs(lam_norm2-1.0)),
            "suppressed_probability_phi": suppressed_probability(
                rho_qR, mask_phi, model.dq*model.dR, model
            ),
            "suppressed_probability_lam": suppressed_probability(
                rho_R, mask_lam, model.dR, model
            ),
            "recombination_residual_l2": _l2(residual, model),
            "predicted_mask_residual_l2": _l2(predicted, model),
            "unexplained_residual_l2": _l2(unexplained, model),
            "relative_unexplained_residual": _l2(unexplained, model)/target_l2,
            "direct_action_l2": target_l2,
            "recombined_rhs_l2": _l2(dY, model),
            "max_abs_regularized_F_ratio": max_regularized_F_ratio,
            "max_abs_regularized_chi_ratio": max_regularized_chi_ratio,
            "weighted_link_defect_phi_q": cp.sum(
                rho_qR*cp.abs(1.0-cp.abs(sphi_q1)),
                dtype=model.reduction_real_dtype,
            )*model.dq*model.dR/probability,
            "weighted_link_defect_phi_R": cp.sum(
                rho_qR*cp.abs(1.0-cp.abs(sphi_R1)),
                dtype=model.reduction_real_dtype,
            )*model.dq*model.dR/probability,
            "weighted_link_defect_gamma_R": cp.sum(
                rho_R*cp.abs(1.0-cp.abs(sgamma_R1)),
                dtype=model.reduction_real_dtype,
            )*model.dR/heavy_probability,
            "epsilon_1_imaginary_defect": cp.max(cp.abs(eps1_complex.imag)),
            "epsilon_2_imaginary_defect": cp.max(cp.abs(eps2_complex.imag)),
            "full_norm_rate": cp.abs(2.0*cp.sum(
                cp.conj(c*F[None, :, :])*dY,
                dtype=model.reduction_complex_dtype,
            ).real*model.dq*model.dR),
            "mask_transition_fraction_phi": cp.mean(
                (mask_phi > 0.0) & (mask_phi < 1.0)
            ),
            "mask_transition_fraction_lam": cp.mean(
                (mask_lam > 0.0) & (mask_lam < 1.0)
            ),
        }
    return DiscreteGPUResult(dc, dlam, dchi, fields, diagnostics)


def pnc_retract_gpu(coefficients, lam, chi, model):
    """Apply the existing support-aware, product-preserving PNC retraction."""
    c_norm2 = cp.sum(
        cp.real(coefficients*cp.conj(coefficients)), axis=0,
        dtype=model.reduction_real_dtype,
    )
    raw_c = cp.max(cp.abs(c_norm2-1.0))
    lam_norm2 = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dq
    raw_lam = cp.max(cp.abs(lam_norm2-1.0))
    coefficients, lam, chi, correction = pnc_project_coefficients(
        coefficients, lam, chi, model
    )
    return coefficients, lam, chi, correction, {
        "max_raw_pnc_phi_error": raw_c,
        "max_raw_pnc_lam_error": raw_lam,
    }


def _product_tangent(coefficients, lam, chi, derivatives):
    dc, dlam, dchi = derivatives
    F = lam*chi[None, :]
    return dc*F[None, :, :]+coefficients*(
        dlam*chi[None, :]+lam*dchi[None, :]
    )[None, :, :]


def full_step_discrete_bh(
    coefficients, lam, chi, dt, model, basis, *,
    collect_step_diagnostics=False,
):
    """One classical RK4 step followed by a product-preserving PNC retraction."""
    def rhs(c, l, h):
        result = discrete_rhs_gpu(c, l, h, model, basis)
        return result.dc, result.dlam, result.dchi

    initial = (coefficients, lam, chi)
    k1 = rhs(*initial)
    stage2 = (
        coefficients+0.5*dt*k1[0],
        lam+0.5*dt*k1[1], chi+0.5*dt*k1[2],
    )
    k2 = rhs(*stage2)
    stage3 = (
        coefficients+0.5*dt*k2[0],
        lam+0.5*dt*k2[1], chi+0.5*dt*k2[2],
    )
    k3 = rhs(*stage3)
    stage4 = (
        coefficients+dt*k3[0], lam+dt*k3[1], chi+dt*k3[2],
    )
    k4 = rhs(*stage4)
    initial_product = None
    rk_product_increment = None
    if collect_step_diagnostics:
        initial_product = coefficients*(lam*chi[None, :])[None, :, :]
        rk_product_increment = dt*(
            _product_tangent(*initial, k1)
            +2.0*_product_tangent(*stage2, k2)
            +2.0*_product_tangent(*stage3, k3)
            +_product_tangent(*stage4, k4)
        )/6.0
    coefficients = coefficients+dt*(
        k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0]
    )/6.0
    lam = lam+dt*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])/6.0
    chi = chi+dt*(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])/6.0
    pre_pnc_product = None
    if collect_step_diagnostics:
        pre_pnc_product = coefficients*(lam*chi[None, :])[None, :, :]
    coefficients, lam, chi, correction, diagnostics = pnc_retract_gpu(
        coefficients, lam, chi, model
    )
    if collect_step_diagnostics:
        final_product = coefficients*(lam*chi[None, :])[None, :, :]
        actual_increment = final_product-initial_product
        local_defect = actual_increment-rk_product_increment
        scale = cp.maximum(
            cp.maximum(_l2(actual_increment, model),
                       _l2(rk_product_increment, model)),
            cp.asarray(1.0e-300, dtype=model.reduction_real_dtype),
        )
        diagnostics.update(
            rk_product_local_defect_l2=_l2(local_defect, model),
            rk_product_local_defect_relative=_l2(local_defect, model)/scale,
            pnc_product_change_l2=_l2(final_product-pre_pnc_product, model),
            rk_product_increment_l2=_l2(rk_product_increment, model),
        )
    return coefficients, lam, chi, correction, diagnostics
