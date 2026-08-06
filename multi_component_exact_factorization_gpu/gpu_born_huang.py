"""GPU Born--Huang coefficient propagation for electronic-only MCEF expansion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gpu_core import (
    cp,
    covariant_square,
    derivative,
    logarithmic_components,
    occupied_support_mask,
    proton_base_operator,
    remove_local_norm_generator,
    weak_log_amplitude_gradient,
)


@dataclass
class GPUBornHuangBasis:
    energies: cp.ndarray
    d_q: cp.ndarray
    D_q: cp.ndarray
    d_R: cp.ndarray
    D_R: cp.ndarray


def to_gpu_basis(basis, model):
    return GPUBornHuangBasis(
        energies=cp.asarray(basis.energies, dtype=model.real_dtype),
        d_q=cp.asarray(basis.d_q, dtype=model.complex_dtype),
        D_q=cp.asarray(basis.D_q, dtype=model.complex_dtype),
        d_R=cp.asarray(basis.d_R, dtype=model.complex_dtype),
        D_R=cp.asarray(basis.D_R, dtype=model.complex_dtype),
    )


def connection_action(connection, coefficients):
    return cp.einsum("ljqR,jqR->lqR", connection, coefficients)


def projected_gradient(coefficients, connection, spacing, axis):
    return (
        derivative(coefficients, spacing, axis=axis)
        +connection_action(connection, coefficients)
    )


def residual_momentum(coefficients, connection, vector, spacing, axis):
    return -1j*projected_gradient(
        coefficients, connection, spacing, axis
    )-vector[None, :, :]*coefficients


def residual_square(
    coefficients, first_connection, second_connection, vector, spacing, axis,
):
    first = derivative(coefficients, spacing, axis=axis)
    second = derivative(coefficients, spacing, axis=axis, order=2)
    vector_derivative = derivative(vector, spacing, axis=axis-1)
    return (
        -second
        -2.0*connection_action(first_connection, first)
        -connection_action(second_connection, coefficients)
        +1j*vector_derivative[None, :, :]*coefficients
        +2j*vector[None, :, :]*(
            first+connection_action(first_connection, coefficients)
        )
        +vector[None, :, :]**2*coefficients
    )


def projected_plain_second(
    coefficients, first_connection, second_connection, spacing, axis,
):
    first = derivative(coefficients, spacing, axis=axis)
    return (
        derivative(coefficients, spacing, axis=axis, order=2)
        +2.0*connection_action(first_connection, first)
        +connection_action(second_connection, coefficients)
    )


def coefficient_vector_potential(coefficients, connection, spacing, axis, model):
    gradient = projected_gradient(coefficients, connection, spacing, axis)
    value = -1j*cp.sum(
        cp.conj(coefficients)*gradient, axis=0,
        dtype=model.reduction_complex_dtype,
    )
    return value.real.astype(model.real_dtype, copy=False)


def pnc_project_coefficients(coefficients, lam, chi, model):
    c_norm2 = cp.sum(
        cp.real(coefficients*cp.conj(coefficients)), axis=0,
        dtype=model.reduction_real_dtype,
    )
    c_error = cp.max(cp.abs(c_norm2-1.0))
    c_norm = cp.sqrt(c_norm2).astype(model.real_dtype, copy=False)
    safe_c = cp.where(c_norm > 1.0e-14, c_norm, 1.0)
    coefficients = coefficients/safe_c[None, :, :]
    lam = lam*safe_c
    lam_norm2 = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dq
    lam_error = cp.max(cp.abs(lam_norm2-1.0))
    lam_norm = cp.sqrt(lam_norm2).astype(model.real_dtype, copy=False)
    safe_lam = cp.where(lam_norm > 1.0e-14, lam_norm, 1.0)
    lam = lam/safe_lam[None, :]
    chi = chi*safe_lam
    return coefficients, lam, chi, cp.maximum(c_error, lam_error)


def instantaneous_functionals_bh(
    coefficients, lam, chi, model, basis, ratio_floor,
    mask_threshold_phi, mask_threshold_lam,
):
    # Reuse the expensive coefficient derivatives/NAC contractions in a, b,
    # the residual momenta and their squares.
    first_q = derivative(coefficients, model.dq, axis=1)
    d_c_q = connection_action(basis.d_q, coefficients)
    gradient_q = first_q+d_c_q
    first_R = derivative(coefficients, model.dR, axis=2)
    d_c_R = connection_action(basis.d_R, coefficients)
    gradient_R = first_R+d_c_R
    a = (-1j*cp.sum(
        cp.conj(coefficients)*gradient_q, axis=0,
        dtype=model.reduction_complex_dtype,
    )).real.astype(model.real_dtype, copy=False)
    b = (-1j*cp.sum(
        cp.conj(coefficients)*gradient_R, axis=0,
        dtype=model.reduction_complex_dtype,
    )).real.astype(model.real_dtype, copy=False)
    p_R_lam = -1j*derivative(lam, model.dR, axis=1)
    alpha = cp.sum(
        cp.conj(lam)*(p_R_lam+b*lam), axis=0,
        dtype=model.reduction_complex_dtype,
    ).real*model.dq
    alpha = alpha.astype(model.real_dtype, copy=False)

    rho_R = cp.real(chi*cp.conj(chi))
    rho_qR = cp.real(lam*cp.conj(lam))*rho_R[None, :]
    mask_phi = occupied_support_mask(rho_qR, mask_threshold_phi, model)
    mask_lam = occupied_support_mask(rho_R, mask_threshold_lam, model)
    p_q_lam = -1j*derivative(lam, model.dq, axis=0)
    lam_phase_q, lam_logamp_q = logarithmic_components(
        lam, model.dq, axis=0, model=model, numerical_floor=ratio_floor,
        momentum_factor=p_q_lam,
    )
    lam_phase_R, lam_logamp_R = logarithmic_components(
        lam, model.dR, axis=1, model=model, numerical_floor=ratio_floor,
        momentum_factor=p_R_lam,
    )
    p_R_chi = -1j*derivative(chi, model.dR, axis=0)
    chi_phase_R, chi_logamp_R = logarithmic_components(
        chi, model.dR, axis=0, model=model, numerical_floor=ratio_floor,
        momentum_factor=p_R_chi,
    )
    weak_diag = {}
    if model.log_derivative_backend == "weak":
        xi = lam*chi[None, :]
        xi_logamp_q, dq_diag = weak_log_amplitude_gradient(
            xi, model.dq, 0, model
        )
        xi_logamp_R, dR_diag = weak_log_amplitude_gradient(
            xi, model.dR, 1, model
        )
        chi_logamp_used, dc_diag = weak_log_amplitude_gradient(
            chi, model.dR, 0, model
        )
        weak_diag = dict(
            weak_log_residual_q_xi=dq_diag["weak_log_residual"],
            weak_log_residual_R_xi=dR_diag["weak_log_residual"],
            weak_log_residual_R_chi=dc_diag["weak_log_residual"],
            weak_log_iterations=cp.maximum(
                dq_diag["weak_log_iterations"], cp.maximum(
                    dR_diag["weak_log_iterations"],
                    dc_diag["weak_log_iterations"],
                ),
            ),
            weak_log_unconverged_lines=(
                dq_diag["weak_log_unconverged_lines"]
                +dR_diag["weak_log_unconverged_lines"]
                +dc_diag["weak_log_unconverged_lines"]
            ),
        )
    else:
        xi_logamp_q = lam_logamp_q
        xi_logamp_R = lam_logamp_R+chi_logamp_R[None, :]
        chi_logamp_used = chi_logamp_R

    p_q = -1j*gradient_q-a[None, :, :]*coefficients
    p2_q = (
        -derivative(coefficients, model.dq, axis=1, order=2)
        -2.0*connection_action(basis.d_q, first_q)
        -connection_action(basis.D_q, coefficients)
        +1j*derivative(a, model.dq, axis=0)[None, :, :]*coefficients
        +2j*a[None, :, :]*gradient_q+a[None, :, :]**2*coefficients
    )
    p_R = -1j*gradient_R-b[None, :, :]*coefficients
    p2_R = (
        -derivative(coefficients, model.dR, axis=2, order=2)
        -2.0*connection_action(basis.d_R, first_R)
        -connection_action(basis.D_R, coefficients)
        +1j*derivative(b, model.dR, axis=1)[None, :, :]*coefficients
        +2j*b[None, :, :]*gradient_R+b[None, :, :]**2*coefficients
    )
    coefficient_q = lam_phase_q+a-1j*mask_phi*xi_logamp_q
    coefficient_R = (
        lam_phase_R+chi_phase_R[None, :]+b-1j*mask_phi*xi_logamp_R
    )
    u_c = (
        0.5*p2_q+coefficient_q[None, :, :]*p_q
    )/model.proton_mass+(
        0.5*p2_R+coefficient_R[None, :, :]*p_R
    )/model.heavy_mass
    u_c, gamma_c, raw_rate_c, corrected_rate_c = remove_local_norm_generator(
        coefficients, u_c, 1.0, axis=0, model=model
    )
    hbo_c = basis.energies*coefficients
    epsilon_1 = cp.sum(
        cp.conj(coefficients)*(hbo_c+u_c), axis=0,
        dtype=model.reduction_complex_dtype,
    ).real.astype(model.real_dtype, copy=False)

    base_lam = proton_base_operator(
        lam, a, b, alpha, chi_phase_R, chi_logamp_used, mask_lam, model
    )
    hpr_raw = base_lam+epsilon_1*lam+1j*gamma_c*lam
    hpr, gamma_lam, raw_rate_lam, corrected_rate_lam = (
        remove_local_norm_generator(lam, hpr_raw, model.dq, axis=0, model=model)
    )
    epsilon_2 = cp.sum(
        cp.conj(lam)*hpr, axis=0, dtype=model.reduction_complex_dtype,
    ).real*model.dq
    epsilon_2 = epsilon_2.astype(model.real_dtype, copy=False)
    return dict(
        a=a, b=b, alpha=alpha, epsilon_1=epsilon_1, epsilon_2=epsilon_2,
        u_c=u_c, hpr_lam=hpr, gamma_c=gamma_c, gamma_lam=gamma_lam,
        mask_phi=mask_phi, mask_lam=mask_lam, p_R_chi=p_R_chi,
        raw_rate_phi=raw_rate_c, corrected_rate_phi=corrected_rate_c,
        raw_rate_lam=raw_rate_lam, corrected_rate_lam=corrected_rate_lam,
        raw_logamp_phi=cp.maximum(
            cp.abs(lam_logamp_q),
            cp.abs(lam_logamp_R)+cp.abs(chi_logamp_R)[None, :],
        ),
        effective_logamp_phi=cp.maximum(
            cp.abs(mask_phi*xi_logamp_q), cp.abs(mask_phi*xi_logamp_R)
        ),
        **weak_diag,
    )


def project_product_residual_bh(
    coefficients, lam, chi, dc, dlam, dchi, model, basis,
):
    xi = lam*chi[None, :]
    y = coefficients*xi[None, :, :]
    product_rhs = dc*xi[None, :, :]+coefficients*(
        dlam*chi[None, :]+lam*dchi[None, :]
    )[None, :, :]
    target = -1j*(
        -0.5*projected_plain_second(
            y, basis.d_q, basis.D_q, model.dq, 1
        )/model.proton_mass
        -0.5*projected_plain_second(
            y, basis.d_R, basis.D_R, model.dR, 2
        )/model.heavy_mass
    )
    residual = target-product_rhs
    delta_xi = cp.sum(
        cp.conj(coefficients)*residual, axis=0,
        dtype=model.reduction_complex_dtype,
    ).astype(model.complex_dtype, copy=False)
    perp_c = residual-coefficients*delta_xi[None, :, :]
    xi_density = cp.real(xi*cp.conj(xi))
    tiny = cp.asarray(1.0e-30, dtype=xi_density.dtype)
    xi_peak = cp.maximum(cp.max(xi_density), tiny)
    support = xi_density/(
        xi_density+model.product_projection_floor_phi*xi_peak+tiny
    )
    if model.product_projection_backend == "weighted_tikhonov":
        ridge = model.projection_tau_phi*xi_peak/(
            support+model.projection_support_epsilon
        )
        inverse_xi = support*cp.conj(xi)/(
            support*xi_density+ridge+tiny
        )
    else:
        inverse_xi = cp.conj(xi)/(
            xi_density+model.product_projection_floor_phi*xi_peak
        )
    delta_c = perp_c*inverse_xi[None, :, :]
    parallel_chi = cp.sum(
        cp.conj(lam)*delta_xi, axis=0,
        dtype=model.reduction_complex_dtype,
    )*model.dq
    perp_lam = delta_xi-lam*parallel_chi[None, :]
    chi_density = cp.real(chi*cp.conj(chi))
    chi_peak = cp.maximum(cp.max(chi_density), tiny)
    support_R = chi_density/(
        chi_density+model.product_projection_floor_lam*chi_peak+tiny
    )
    if model.product_projection_backend == "weighted_tikhonov":
        ridge_R = model.projection_tau_lam*chi_peak/(
            support_R+model.projection_support_epsilon
        )
        inverse_chi = support_R*cp.conj(chi)/(
            support_R*chi_density+ridge_R+tiny
        )
        chi_shrink = support_R/(
            support_R
            +model.projection_tau_chi/(
                support_R+model.projection_support_epsilon
            )+tiny
        )
        delta_chi = chi_shrink*parallel_chi
    else:
        inverse_chi = cp.conj(chi)/(
            chi_density+model.product_projection_floor_lam*chi_peak
        )
        delta_chi = parallel_chi
    delta_lam = perp_lam*inverse_chi[None, :]
    dc = dc+delta_c
    dlam = dlam+delta_lam
    dchi = dchi+delta_chi
    corrected = dc*xi[None, :, :]+coefficients*(
        dlam*chi[None, :]+lam*dchi[None, :]
    )[None, :, :]
    volume = model.dq*model.dR
    l2 = lambda value: cp.sqrt(cp.sum(
        cp.real(value*cp.conj(value)), dtype=model.reduction_real_dtype
    )*volume)
    target_l2 = cp.maximum(l2(target), tiny)
    diagnostics = dict(
        max_product_residual_l2=l2(residual),
        max_effective_product_residual_l2=l2(target-corrected),
        max_relative_product_projection_l2=l2(corrected-product_rhs)/target_l2,
        max_abs_product_correction_phi=cp.max(cp.abs(delta_c)),
        max_abs_product_correction_lam=cp.max(cp.abs(delta_lam)),
        max_abs_product_correction_chi=cp.max(cp.abs(delta_chi)),
        max_inverse_support_product_correction_phi=cp.sqrt(cp.sum(
            cp.real(delta_c*cp.conj(delta_c))/(
                support[None, :, :]+model.projection_support_epsilon
            ), dtype=model.reduction_real_dtype,
        )*volume),
        max_inverse_support_product_correction_lam=cp.sqrt(cp.sum(
            cp.real(delta_lam*cp.conj(delta_lam))/(
                support_R[None, :]+model.projection_support_epsilon
            ), dtype=model.reduction_real_dtype,
        )*volume),
        max_inverse_support_product_correction_chi=cp.sqrt(cp.sum(
            cp.real(delta_chi*cp.conj(delta_chi))/(
                support_R+model.projection_support_epsilon
            ), dtype=model.reduction_real_dtype,
        )*model.dR),
        max_abs_full_norm_rate_before_product_projection=cp.abs(
            2.0*cp.sum(
                cp.conj(y)*product_rhs,
                dtype=model.reduction_complex_dtype,
            ).real*volume
        ),
        max_abs_full_norm_rate_after_product_projection=cp.abs(
            2.0*cp.sum(
                cp.conj(y)*corrected,
                dtype=model.reduction_complex_dtype,
            ).real*volume
        ),
    )
    return dc, dlam, dchi, diagnostics


def coupled_rhs_bh(
    coefficients, lam, chi, model, basis, ratio_floor,
    mask_threshold_phi, mask_threshold_lam,
):
    fields = instantaneous_functionals_bh(
        coefficients, lam, chi, model, basis, ratio_floor,
        mask_threshold_phi, mask_threshold_lam,
    )
    dc = -1j*(fields["u_c"]-fields["epsilon_1"][None, :, :]*coefficients)
    dlam = -1j*(fields["hpr_lam"]-fields["epsilon_2"][None, :]*lam)
    p2chi = covariant_square(
        chi, fields["alpha"], model.dR, axis=0, sign=+1,
        momentum_field=fields["p_R_chi"],
    )
    dchi = -1j*(
        0.5*p2chi/model.heavy_mass+fields["epsilon_2"]*chi
    )+fields["gamma_lam"]*chi
    dc, dlam, dchi, diagnostics = project_product_residual_bh(
        coefficients, lam, chi, dc, dlam, dchi, model, basis
    )
    diagnostics.update(
        max_abs_gamma_phi=cp.max(cp.abs(fields["gamma_c"])),
        max_abs_gamma_lam=cp.max(cp.abs(fields["gamma_lam"])),
        max_abs_support_gamma_phi=cp.max(cp.abs(
            fields["mask_phi"]*fields["gamma_c"]
        )),
        max_abs_support_gamma_lam=cp.max(cp.abs(
            fields["mask_lam"]*fields["gamma_lam"]
        )),
        max_raw_logamp_phi=cp.max(cp.abs(fields["raw_logamp_phi"])),
        max_effective_logamp_phi=cp.max(cp.abs(
            fields["effective_logamp_phi"]
        )),
    )
    for key in (
        "weak_log_residual_q_xi", "weak_log_residual_R_xi",
        "weak_log_residual_R_chi", "weak_log_iterations",
        "weak_log_unconverged_lines",
    ):
        if key in fields:
            diagnostics[f"max_{key}"] = cp.max(fields[key])
    return dc, dlam, dchi, fields, diagnostics


def full_step_bh(
    coefficients, lam, chi, dt, model, basis, ratio_floor,
    mask_threshold_phi, mask_threshold_lam,
):
    phase = cp.exp(-0.5j*dt*basis.energies).astype(
        model.complex_dtype, copy=False
    )
    coefficients = coefficients*phase
    stages = []

    def rhs(c, l, h):
        result = coupled_rhs_bh(
            c, l, h, model, basis, ratio_floor,
            mask_threshold_phi, mask_threshold_lam,
        )
        stages.append(result[4])
        return result[:3]

    k1 = rhs(coefficients, lam, chi)
    k2 = rhs(
        coefficients+0.5*dt*k1[0], lam+0.5*dt*k1[1], chi+0.5*dt*k1[2]
    )
    k3 = rhs(
        coefficients+0.5*dt*k2[0], lam+0.5*dt*k2[1], chi+0.5*dt*k2[2]
    )
    k4 = rhs(coefficients+dt*k3[0], lam+dt*k3[1], chi+dt*k3[2])
    coefficients = coefficients+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0
    lam = lam+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0
    chi = chi+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0
    coefficients, lam, chi, correction = pnc_project_coefficients(
        coefficients, lam, chi, model
    )
    coefficients = coefficients*phase
    coefficients, lam, chi, correction2 = pnc_project_coefficients(
        coefficients, lam, chi, model
    )
    merged = {}
    for key in stages[0]:
        value = cp.asarray(0.0, dtype=model.reduction_real_dtype)
        for stage in stages:
            value = cp.maximum(value, stage.get(key, 0.0))
        merged[key] = value
    return coefficients, lam, chi, cp.maximum(correction, correction2), merged
