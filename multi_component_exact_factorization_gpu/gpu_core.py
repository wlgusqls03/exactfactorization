"""Multi-component EF의 큰 배열 연산을 한 GPU에서 수행한다.

배열 축과 물리식은 CPU ``multi_component_exact_factorization.core``와 같다.
초기 local BO eigenproblem은 작은 tridiagonal 문제이므로 CPU에서 한 번 풀고,
시간 전파에 필요한 Phi/Lambda/chi와 potential만 GPU로 옮긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - GPU 환경에서 확인하는 경로
    raise ImportError(
        "GPU 전파에는 CuPy가 필요합니다. CUDA 11.2 서버에서는 "
        "`pip install cupy-cuda11x==11.6.0`을 사용하세요."
    ) from exc


@dataclass
class GPUModel:
    """GPU에 상주하는 Hamiltonian 자료와 선택한 계산 정밀도."""

    dx: float
    dq: float
    dR: float
    proton_mass: float
    heavy_mass: float
    potential: cp.ndarray                    # (nx,nq,nR), real32/real64
    kinetic_energies: cp.ndarray             # (nx,), real32/real64
    real_dtype: type
    complex_dtype: type
    reduction_real_dtype: type
    reduction_complex_dtype: type
    kinetic_phase_cache: dict = field(default_factory=dict)
    potential_phase_cache: dict = field(default_factory=dict)


def precision_types(precision):
    """double/single/mixed별 저장 dtype와 reduction dtype를 정한다."""
    if precision == "double":
        return cp.float64, cp.complex128, cp.float64, cp.complex128
    if precision == "single":
        return cp.float32, cp.complex64, cp.float32, cp.complex64
    if precision == "mixed":
        # 큰 3D 배열은 단정밀도지만 inner product와 norm의 합은 FP64로 누적한다.
        return cp.float32, cp.complex64, cp.float64, cp.complex128
    raise ValueError(f"지원하지 않는 precision: {precision}")


def make_gpu_model(cpu_model, precision):
    """CPU model의 고정 Hamiltonian 자료를 현재 GPU로 한 번만 복사한다."""
    real, complex_, reduction_real, reduction_complex = precision_types(precision)
    modes = cp.arange(1, len(cpu_model.x)+1, dtype=real)
    kinetic = (
        1.0-cp.cos(cp.pi*modes/(len(cpu_model.x)+1))
    )/cpu_model.dx**2
    return GPUModel(
        dx=cpu_model.dx, dq=cpu_model.dq, dR=cpu_model.dR,
        proton_mass=cpu_model.proton_mass, heavy_mass=cpu_model.heavy_mass,
        potential=cp.asarray(cpu_model.potential, dtype=real),
        kinetic_energies=kinetic.astype(real, copy=False),
        real_dtype=real, complex_dtype=complex_,
        reduction_real_dtype=reduction_real,
        reduction_complex_dtype=reduction_complex,
    )


def to_gpu_factors(phi, lam, chi, model):
    """CPU 초기 factor를 선택한 complex dtype으로 GPU에 올린다."""
    return (
        cp.asarray(phi, dtype=model.complex_dtype),
        cp.asarray(lam, dtype=model.complex_dtype),
        cp.asarray(chi, dtype=model.complex_dtype),
    )


def derivative(values, spacing, axis, order=1):
    """GPU 주기 격자의 독립적인 4차 정확도 5점 1·2차 유한차분."""
    if values.shape[axis] < 5:
        raise ValueError("5점 미분에는 해당 축에 최소 5개 격자점이 필요합니다.")
    if order == 1:
        return (
            cp.roll(values, 2, axis=axis)
            -8.0*cp.roll(values, 1, axis=axis)
            +8.0*cp.roll(values, -1, axis=axis)
            -cp.roll(values, -2, axis=axis)
        )/(12.0*spacing)
    if order == 2:
        return (
            -cp.roll(values, 2, axis=axis)
            +16.0*cp.roll(values, 1, axis=axis)
            -30.0*values
            +16.0*cp.roll(values, -1, axis=axis)
            -cp.roll(values, -2, axis=axis)
        )/(12.0*spacing**2)
    raise ValueError("order는 1 또는 2여야 합니다.")


def momentum(values, spacing, axis):
    """운동량 연산자 ``-i d/dcoordinate``."""
    return -1j*derivative(values, spacing, axis)


def _real_inner_sum(values, axis, model):
    """복소 inner-product 합을 선택한 reduction 정밀도로 누적한다."""
    summed = cp.sum(values, axis=axis, dtype=model.reduction_complex_dtype)
    # mixed에서는 큰 배열과 다시 곱할 때 complex128 승격을 막기 위해
    # 고정밀도로 합한 결과를 propagation real dtype으로 되돌린다.
    return summed.real.astype(model.real_dtype, copy=False)


def regularized_ratio(numerator, denominator, relative_floor):
    """Node/tail에서 안정화한 numerator/denominator를 GPU에서 계산한다."""
    density = cp.real(denominator*cp.conj(denominator))
    tiny = cp.asarray(1.0e-30, dtype=density.dtype)
    floor = cp.maximum(relative_floor*cp.max(density), tiny)
    return numerator*cp.conj(denominator)/(density+floor)


def logarithmic_components(
    factor, spacing, axis, model, numerical_floor=1.0e-14,
):
    """GPU에서 phase gradient와 amplitude logarithmic gradient를 분리."""
    if numerical_floor <= 0.0:
        raise ValueError("numerical_floor는 양수여야 합니다.")
    density = cp.real(factor*cp.conj(factor))
    peak = cp.max(density, axis=axis, keepdims=True)
    tiny = cp.asarray(1.0e-30, dtype=density.dtype)
    safe = density+numerical_floor*cp.maximum(peak, tiny)
    ratio = momentum(factor, spacing, axis)*cp.conj(factor)/safe
    return (
        ratio.real.astype(model.real_dtype, copy=False),
        (-ratio.imag).astype(model.real_dtype, copy=False),
    )


def occupied_support_mask(density, relative_threshold, model):
    """GPU ``rho/(rho+eta*rho_max)`` support mask."""
    if relative_threshold < 0.0:
        raise ValueError("mask threshold는 0 이상이어야 합니다.")
    if relative_threshold == 0.0:
        return cp.ones_like(density, dtype=model.real_dtype)
    tiny = cp.asarray(1.0e-30, dtype=density.dtype)
    peak = cp.maximum(cp.max(density), tiny)
    return (density/(density+relative_threshold*peak)).astype(
        model.real_dtype, copy=False
    )


def suppressed_probability(density, mask, volume, model):
    """Support mask가 감쇠한 정규화 probability mass."""
    total = cp.sum(density, dtype=model.reduction_real_dtype)*volume
    removed = cp.sum(
        density*(1.0-mask), dtype=model.reduction_real_dtype
    )*volume
    tiny = cp.asarray(1.0e-300, dtype=model.reduction_real_dtype)
    return removed/cp.maximum(total, tiny)


def remove_local_norm_generator(
    factor, action, spacing, axis, model, norm_floor=1.0e-14,
):
    """고정밀도 reduction으로 local anti-Hermitian norm generator를 제거."""
    norm2 = cp.sum(
        cp.real(factor*cp.conj(factor)), axis=axis,
        dtype=model.reduction_real_dtype,
    )*spacing
    expectation = cp.sum(
        cp.conj(factor)*action, axis=axis,
        dtype=model.reduction_complex_dtype,
    )*spacing
    floor = cp.asarray(norm_floor, dtype=model.reduction_real_dtype)
    gamma_reduction = expectation.imag/cp.maximum(norm2, floor)
    gamma = gamma_reduction.astype(model.real_dtype, copy=False)
    imaginary_unit = cp.asarray(1j, dtype=model.complex_dtype)
    corrected = (
        action-imaginary_unit*cp.expand_dims(gamma, axis=axis)*factor
    ).astype(model.complex_dtype, copy=False)
    corrected_expectation = cp.sum(
        cp.conj(factor)*corrected, axis=axis,
        dtype=model.reduction_complex_dtype,
    )*spacing
    return (
        corrected,
        gamma,
        2.0*expectation.imag,
        2.0*corrected_expectation.imag,
    )


def dst1_ortho(values, axis=0):
    """복소 배열의 orthonormal DST-I를 odd-extension FFT로 구현한다.

    CuPy 내장 DST는 type I을 지원하지 않는다. 길이 N 데이터를 길이
    ``2(N+1)``의 odd extension으로 만든 뒤

        DST-I(x) = i FFT(odd(x))[1:N+1] / sqrt(2(N+1))

    을 사용한다. Orthonormal DST-I는 자기 자신의 역변환이다.
    """
    n = values.shape[axis]
    extended_shape = list(values.shape)
    extended_shape[axis] = 2*(n+1)
    extended = cp.zeros(extended_shape, dtype=values.dtype)
    middle = [slice(None)]*values.ndim
    middle[axis] = slice(1, n+1)
    extended[tuple(middle)] = values
    tail = [slice(None)]*values.ndim
    tail[axis] = slice(n+2, None)
    extended[tuple(tail)] = -cp.flip(values, axis=axis)
    transformed = cp.fft.fft(extended, axis=axis)
    selected = [slice(None)]*values.ndim
    selected[axis] = slice(1, n+1)
    scale = cp.asarray((2.0*(n+1))**-0.5, dtype=values.real.dtype)
    imaginary_unit = cp.asarray(1j, dtype=values.dtype)
    result = imaginary_unit*transformed[tuple(selected)]*scale
    return result.astype(values.dtype, copy=False)


def electronic_kinetic_step(values, tau, model):
    """전자축에 hard-wall ``exp(-i tau T_x)``를 적용한다."""
    key = float(tau)
    phase = model.kinetic_phase_cache.get(key)
    if phase is None:
        phase = cp.exp(-1j*tau*model.kinetic_energies).astype(
            model.complex_dtype, copy=False
        )
        model.kinetic_phase_cache[key] = phase
    transformed = dst1_ortho(values, axis=0)
    transformed *= phase[:, None, None]
    return dst1_ortho(transformed, axis=0)


def electronic_split_step(phi, tau, model):
    """Hard-wall 전자 ``T_x/2 -> V -> T_x/2`` split step."""
    phi = electronic_kinetic_step(phi, 0.5*tau, model)
    key = float(tau)
    phase = model.potential_phase_cache.get(key)
    if phase is None:
        phase = cp.exp(-1j*tau*model.potential).astype(
            model.complex_dtype, copy=False
        )
        model.potential_phase_cache[key] = phase
    phi *= phase
    return electronic_kinetic_step(phi, 0.5*tau, model)


def apply_electronic_hamiltonian(phi, model):
    """Dirichlet 중앙차분 ``[-d_x^2/2+V] Phi``."""
    kinetic = phi/model.dx**2
    kinetic[1:] -= 0.5*phi[:-1]/model.dx**2
    kinetic[:-1] -= 0.5*phi[1:]/model.dx**2
    return kinetic+model.potential*phi


def _minus_covariant(field, vector, spacing, axis):
    return momentum(field, spacing, axis)-vector*field


def _plus_covariant(field, vector, spacing, axis):
    return momentum(field, spacing, axis)+vector*field


def covariant_square(field, vector, spacing, axis, sign):
    """독립 5점 D2와 Hermitian anticommutator로 covariant square 조립."""
    second = derivative(field, spacing, axis=axis, order=2)
    p_field = momentum(field, spacing, axis=axis)
    p_vector_field = momentum(vector*field, spacing, axis=axis)
    return (
        -second
        +sign*(p_vector_field+vector*p_field)
        +vector**2*field
    ).astype(field.dtype, copy=False)


def geometric_fields(phi, lam, model):
    """Vector potential ``a(q,R), b(q,R), alpha(R)``."""
    p_q_phi = momentum(phi, model.dq, axis=1)
    p_R_phi = momentum(phi, model.dR, axis=2)
    a = _real_inner_sum(cp.conj(phi)*p_q_phi, axis=0, model=model)*model.dx
    b = _real_inner_sum(cp.conj(phi)*p_R_phi, axis=0, model=model)*model.dx
    p_R_lam = momentum(lam, model.dR, axis=1)
    alpha = _real_inner_sum(
        cp.conj(lam)*(p_R_lam+b*lam), axis=0, model=model
    )*model.dq
    return a, b, alpha


def proton_base_operator(
    lam, a, b, alpha, chi_phase_R, chi_logamp_R, mask_lam, model,
):
    """epsilon_1을 제외한 proton-heavy Hamiltonian을 Lambda에 적용."""
    proton_kinetic = covariant_square(
        lam, a, model.dq, axis=0, sign=+1
    )*(0.5/model.proton_mass)

    vector_R = b-alpha[None, :]
    dplus_R = _plus_covariant(lam, vector_R, model.dR, axis=1)
    dplus_R2 = covariant_square(
        lam, vector_R, model.dR, axis=1, sign=+1
    )
    coefficient = (
        chi_phase_R+alpha-1j*mask_lam*chi_logamp_R
    ).astype(model.complex_dtype, copy=False)
    coupling = (
        0.5*dplus_R2+coefficient[None, :]*dplus_R
    )/model.heavy_mass
    return proton_kinetic+coupling


def instantaneous_functionals(
    phi, lam, chi, model, floor=1.0e-14,
    mask_threshold_phi=1.0e-10, mask_threshold_lam=1.0e-10,
):
    """현재 factor에서 두 TDPES와 세 vector potential을 계산한다."""
    a, b, alpha = geometric_fields(phi, lam, model)

    rho_R = cp.real(chi*cp.conj(chi))
    rho_qR = cp.real(lam*cp.conj(lam))*rho_R[None, :]
    mask_phi = occupied_support_mask(rho_qR, mask_threshold_phi, model)
    mask_lam = occupied_support_mask(rho_R, mask_threshold_lam, model)
    lam_phase_q, lam_logamp_q = logarithmic_components(
        lam, model.dq, axis=0, model=model, numerical_floor=floor
    )
    lam_phase_R, lam_logamp_R = logarithmic_components(
        lam, model.dR, axis=1, model=model, numerical_floor=floor
    )
    chi_phase_R, chi_logamp_R = logarithmic_components(
        chi, model.dR, axis=0, model=model, numerical_floor=floor
    )

    dminus_q = _minus_covariant(phi, a[None, :, :], model.dq, axis=1)
    dminus_q2 = covariant_square(
        phi, a[None, :, :], model.dq, axis=1, sign=-1
    )
    coefficient_q = (
        lam_phase_q+a-1j*mask_phi*lam_logamp_q
    ).astype(model.complex_dtype, copy=False)
    u_q_phi = (
        0.5*dminus_q2+coefficient_q[None, :, :]*dminus_q
    )/model.proton_mass

    dminus_R = _minus_covariant(phi, b[None, :, :], model.dR, axis=2)
    dminus_R2 = covariant_square(
        phi, b[None, :, :], model.dR, axis=2, sign=-1
    )
    coefficient_R = (
        chi_phase_R[None, :]+lam_phase_R+b
        -1j*mask_phi*(chi_logamp_R[None, :]+lam_logamp_R)
    ).astype(model.complex_dtype, copy=False)
    u_R_phi = (
        0.5*dminus_R2
        +coefficient_R[None, :, :]*dminus_R
    )/model.heavy_mass
    u_phi = u_q_phi+u_R_phi

    u_phi, gamma_phi, raw_rate_phi, corrected_rate_phi = (
        remove_local_norm_generator(
            phi, u_phi, model.dx, axis=0, model=model
        )
    )

    hbo_phi = apply_electronic_hamiltonian(phi, model)
    epsilon_1 = _real_inner_sum(
        cp.conj(phi)*(hbo_phi+u_phi), axis=0, model=model
    )*model.dx

    base_lam_raw = proton_base_operator(
        lam, a, b, alpha, chi_phase_R, chi_logamp_R, mask_lam, model
    )
    hpr_lam_raw = (
        base_lam_raw+epsilon_1*lam+1j*gamma_phi*lam
    ).astype(model.complex_dtype, copy=False)
    hpr_lam, gamma_lam, raw_rate_lam, corrected_rate_lam = (
        remove_local_norm_generator(
            lam, hpr_lam_raw, model.dq, axis=0, model=model
        )
    )
    epsilon_2 = _real_inner_sum(
        cp.conj(lam)*hpr_lam, axis=0, model=model
    )*model.dq
    return dict(
        a=a, b=b, alpha=alpha, epsilon_1=epsilon_1, epsilon_2=epsilon_2,
        u_phi=u_phi, hpr_lam=hpr_lam,
        gamma_phi=gamma_phi, gamma_lam=gamma_lam,
        support_gamma_phi=mask_phi*gamma_phi,
        support_gamma_lam=mask_lam*gamma_lam,
        raw_rate_phi=raw_rate_phi, raw_rate_lam=raw_rate_lam,
        corrected_rate_phi=corrected_rate_phi,
        corrected_rate_lam=corrected_rate_lam,
        mask_phi=mask_phi, mask_lam=mask_lam,
        suppressed_probability_phi=suppressed_probability(
            rho_qR, mask_phi, model.dq*model.dR, model
        ),
        suppressed_probability_lam=suppressed_probability(
            rho_R, mask_lam, model.dR, model
        ),
        raw_logamp_phi=cp.maximum(
            cp.abs(lam_logamp_q),
            cp.abs(lam_logamp_R)+cp.abs(chi_logamp_R)[None, :],
        ),
        effective_logamp_phi=cp.maximum(
            cp.abs(mask_phi*lam_logamp_q),
            cp.abs(mask_phi*(lam_logamp_R+chi_logamp_R[None, :])),
        ),
        raw_logamp_lam=cp.abs(chi_logamp_R),
        effective_logamp_lam=cp.abs(mask_lam*chi_logamp_R),
    )


def pnc_project(phi, lam, chi, model):
    """두 PNC를 고정밀도 reduction으로 복원하며 full Psi를 보존한다."""
    phi_density = cp.sum(
        cp.real(phi*cp.conj(phi)), axis=0, dtype=model.reduction_real_dtype
    )*model.dx
    phi_error = cp.max(cp.abs(phi_density-1.0))
    phi_norm = cp.sqrt(phi_density).astype(model.real_dtype, copy=False)
    safe_phi = cp.where(phi_norm > 1.0e-14, phi_norm, 1.0)
    phi = phi/safe_phi[None, :, :]
    lam = lam*safe_phi

    lam_density = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0, dtype=model.reduction_real_dtype
    )*model.dq
    lam_error = cp.max(cp.abs(lam_density-1.0))
    lam_norm = cp.sqrt(lam_density).astype(model.real_dtype, copy=False)
    safe_lam = cp.where(lam_norm > 1.0e-14, lam_norm, 1.0)
    lam = lam/safe_lam[None, :]
    chi = chi*safe_lam
    return phi, lam, chi, cp.maximum(phi_error, lam_error)


def pnc_error(phi, lam, model):
    """현재 저장 factor의 두 PNC 최대 오차를 GPU scalar로 반환한다."""
    phi_density = cp.sum(
        cp.real(phi*cp.conj(phi)), axis=0, dtype=model.reduction_real_dtype
    )*model.dx
    lam_density = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0, dtype=model.reduction_real_dtype
    )*model.dq
    return cp.maximum(
        cp.max(cp.abs(phi_density-1.0)), cp.max(cp.abs(lam_density-1.0))
    )


def project_discrete_product_residual(
    phi, lam, chi, dphi, dlam, dchi, model,
    support_floor_phi=1.0e-10, support_floor_lam=1.0e-10,
):
    """GPU factor RHS를 periodic full nuclear D2의 nested tangent에 투영."""
    if support_floor_phi < 0.0 or support_floor_lam < 0.0:
        raise ValueError("product projection support floor는 0 이상이어야 합니다.")
    psi = phi*lam[None, :, :]*chi[None, None, :]
    product_rhs = (
        dphi*lam[None, :, :]*chi[None, None, :]
        +phi*dlam[None, :, :]*chi[None, None, :]
        +phi*lam[None, :, :]*dchi[None, None, :]
    )
    nuclear_action = (
        -0.5*derivative(psi, model.dq, axis=1, order=2)/model.proton_mass
        -0.5*derivative(psi, model.dR, axis=2, order=2)/model.heavy_mass
    )
    target_rhs = -1j*nuclear_action
    residual = target_rhs-product_rhs

    phi_norm2 = cp.sum(
        cp.real(phi*cp.conj(phi)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dx
    phi_floor = cp.asarray(1.0e-14, dtype=phi_norm2.dtype)
    delta_xi = cp.sum(
        cp.conj(phi)*residual, axis=0,
        dtype=model.reduction_complex_dtype,
    )*model.dx/cp.maximum(phi_norm2, phi_floor)
    delta_xi = delta_xi.astype(model.complex_dtype, copy=False)
    perpendicular_phi = residual-phi*delta_xi[None, :, :]

    xi = lam*chi[None, :]
    xi_density = cp.real(xi*cp.conj(xi))
    tiny = cp.asarray(1.0e-30, dtype=xi_density.dtype)
    xi_peak = cp.maximum(cp.max(xi_density), tiny)
    inverse_xi = cp.conj(xi)/(
        xi_density+support_floor_phi*xi_peak
    )
    delta_phi = perpendicular_phi*inverse_xi[None, :, :]

    lam_norm2 = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dq
    delta_chi = cp.sum(
        cp.conj(lam)*delta_xi, axis=0,
        dtype=model.reduction_complex_dtype,
    )*model.dq/cp.maximum(lam_norm2, phi_floor)
    delta_chi = delta_chi.astype(model.complex_dtype, copy=False)
    perpendicular_lam = delta_xi-lam*delta_chi[None, :]
    chi_density = cp.real(chi*cp.conj(chi))
    chi_peak = cp.maximum(cp.max(chi_density), tiny)
    inverse_chi = cp.conj(chi)/(
        chi_density+support_floor_lam*chi_peak
    )
    delta_lam = perpendicular_lam*inverse_chi[None, :]

    dphi = (dphi+delta_phi).astype(model.complex_dtype, copy=False)
    dlam = (dlam+delta_lam).astype(model.complex_dtype, copy=False)
    dchi = (dchi+delta_chi).astype(model.complex_dtype, copy=False)
    corrected_product_rhs = (
        dphi*lam[None, :, :]*chi[None, None, :]
        +phi*dlam[None, :, :]*chi[None, None, :]
        +phi*lam[None, :, :]*dchi[None, None, :]
    )
    effective_residual = target_rhs-corrected_product_rhs
    volume = model.dx*model.dq*model.dR

    def l2(values):
        return cp.sqrt(cp.sum(
            cp.real(values*cp.conj(values)),
            dtype=model.reduction_real_dtype,
        )*volume)

    def norm_rate(values):
        overlap = cp.sum(
            cp.conj(psi)*values, dtype=model.reduction_complex_dtype
        )
        return 2.0*overlap.real*volume

    diagnostics = dict(
        product_residual_l2=l2(residual),
        effective_product_residual_l2=l2(effective_residual),
        full_norm_rate_before_product_projection=norm_rate(product_rhs),
        full_norm_rate_after_product_projection=norm_rate(
            corrected_product_rhs
        ),
        product_correction_phi=cp.max(cp.abs(delta_phi)),
        product_correction_lam=cp.max(cp.abs(delta_lam)),
        product_correction_chi=cp.max(cp.abs(delta_chi)),
    )
    return dphi, dlam, dchi, diagnostics


def coupled_rhs(
    phi, lam, chi, model, ratio_floor, mask_threshold_phi,
    mask_threshold_lam,
):
    """H_BO split 부분을 제외한 세 coupled RHS."""
    fields = instantaneous_functionals(
        phi, lam, chi, model, floor=ratio_floor,
        mask_threshold_phi=mask_threshold_phi,
        mask_threshold_lam=mask_threshold_lam,
    )
    dphi = -1j*(
        fields["u_phi"]-fields["epsilon_1"][None, :, :]*phi
    )
    dlam = -1j*(
        fields["hpr_lam"]-fields["epsilon_2"][None, :]*lam
    )
    alpha = fields["alpha"]
    p2chi = covariant_square(chi, alpha, model.dR, axis=0, sign=+1)
    dchi = (
        -1j*(0.5*p2chi/model.heavy_mass+fields["epsilon_2"]*chi)
        +fields["gamma_lam"]*chi
    )
    dphi, dlam, dchi, product_diagnostics = project_discrete_product_residual(
        phi, lam, chi, dphi, dlam, dchi, model,
        support_floor_phi=mask_threshold_phi,
        support_floor_lam=mask_threshold_lam,
    )
    fields.update(product_diagnostics)
    return dphi, dlam, dchi, field_maxima(fields)


DIAGNOSTIC_FIELDS = {
    "max_abs_gamma_phi": "gamma_phi",
    "max_abs_gamma_lam": "gamma_lam",
    "max_abs_support_gamma_phi": "support_gamma_phi",
    "max_abs_support_gamma_lam": "support_gamma_lam",
    "max_raw_rate_phi": "raw_rate_phi",
    "max_raw_rate_lam": "raw_rate_lam",
    "max_corrected_rate_phi": "corrected_rate_phi",
    "max_corrected_rate_lam": "corrected_rate_lam",
    "suppressed_probability_phi": "suppressed_probability_phi",
    "suppressed_probability_lam": "suppressed_probability_lam",
    "max_raw_logamp_phi": "raw_logamp_phi",
    "max_effective_logamp_phi": "effective_logamp_phi",
    "max_raw_logamp_lam": "raw_logamp_lam",
    "max_effective_logamp_lam": "effective_logamp_lam",
    "max_product_residual_l2": "product_residual_l2",
    "max_effective_product_residual_l2": "effective_product_residual_l2",
    "max_abs_full_norm_rate_before_product_projection": (
        "full_norm_rate_before_product_projection"
    ),
    "max_abs_full_norm_rate_after_product_projection": (
        "full_norm_rate_after_product_projection"
    ),
    "max_abs_product_correction_phi": "product_correction_phi",
    "max_abs_product_correction_lam": "product_correction_lam",
    "max_abs_product_correction_chi": "product_correction_chi",
}


def field_maxima(fields):
    """GPU 동기화 없이 현재 RHS field의 진단 최대 scalar를 만든다."""
    return {
        name: (
            cp.max(cp.abs(fields[field_name]))
            if field_name in fields else cp.asarray(0.0)
        )
        for name, field_name in DIAGNOSTIC_FIELDS.items()
    }


def merge_maxima(*diagnostics):
    """GPU scalar 진단을 RK stage/step 전체에서 key별 최대값으로 합친다."""
    merged = {}
    for name in DIAGNOSTIC_FIELDS:
        maximum = cp.asarray(0.0)
        for values in diagnostics:
            maximum = cp.maximum(maximum, values.get(name, 0.0))
        merged[name] = maximum
    return merged


def rk4_coupled_step(
    phi, lam, chi, dt, model, ratio_floor, mask_threshold_phi,
    mask_threshold_lam,
):
    """GPU에서 네 번의 RHS를 평가하는 고전 RK4 coupled substep."""
    k1p, k1l, k1c, d1 = coupled_rhs(
        phi, lam, chi, model, ratio_floor, mask_threshold_phi,
        mask_threshold_lam,
    )
    k2p, k2l, k2c, d2 = coupled_rhs(
        phi+0.5*dt*k1p, lam+0.5*dt*k1l, chi+0.5*dt*k1c,
        model, ratio_floor, mask_threshold_phi, mask_threshold_lam,
    )
    k3p, k3l, k3c, d3 = coupled_rhs(
        phi+0.5*dt*k2p, lam+0.5*dt*k2l, chi+0.5*dt*k2c,
        model, ratio_floor, mask_threshold_phi, mask_threshold_lam,
    )
    k4p, k4l, k4c, d4 = coupled_rhs(
        phi+dt*k3p, lam+dt*k3l, chi+dt*k3c,
        model, ratio_floor, mask_threshold_phi, mask_threshold_lam,
    )
    phi = phi+dt*(k1p+2*k2p+2*k3p+k4p)/6.0
    lam = lam+dt*(k1l+2*k2l+2*k3l+k4l)/6.0
    chi = chi+dt*(k1c+2*k2c+2*k3c+k4c)/6.0
    phi, lam, chi, pnc_error = pnc_project(phi, lam, chi, model)
    diagnostics = merge_maxima(d1, d2, d3, d4)
    return phi, lam, chi, pnc_error, diagnostics


def full_step(
    phi, lam, chi, dt, model, ratio_floor, mask_threshold_phi,
    mask_threshold_lam,
):
    """``H_BO 반 -> coupled RK4 -> H_BO 반`` 한 time step."""
    phi = electronic_split_step(phi, 0.5*dt, model)
    phi, lam, chi, first_error, diagnostics = rk4_coupled_step(
        phi, lam, chi, dt, model, ratio_floor, mask_threshold_phi,
        mask_threshold_lam,
    )
    phi = electronic_split_step(phi, 0.5*dt, model)
    phi, lam, chi, final_error = pnc_project(phi, lam, chi, model)
    return (
        phi, lam, chi, cp.maximum(first_error, final_error), diagnostics
    )


def all_finite(phi, lam, chi):
    """세 GPU factor가 모두 finite인지 한 번의 host sync로 확인한다."""
    valid = cp.all(cp.isfinite(phi)) & cp.all(cp.isfinite(lam)) & cp.all(cp.isfinite(chi))
    return bool(valid.get())
