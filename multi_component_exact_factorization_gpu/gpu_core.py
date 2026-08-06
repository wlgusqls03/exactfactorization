"""Multi-component EF의 큰 배열 연산을 한 GPU에서 수행한다.

배열 축과 물리식은 CPU ``multi_component_exact_factorization.core``와 같다.
초기 local BO eigenproblem은 작은 tridiagonal 문제이므로 CPU에서 한 번 풀고,
시간 전파에 필요한 Phi/Lambda/chi와 potential만 GPU로 옮긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - GPU 환경에서 확인하는 경로
    raise ImportError(
        "GPU 전파에는 CuPy가 필요합니다. CUDA 11.2 서버에서는 "
        "`pip install cupy-cuda11x==11.6.0`을 사용하세요."
    ) from exc


_FUSED_PERIODIC_DERIVATIVE = False
_DERIVATIVE_KERNEL_CACHE = {}


def configure_fused_periodic_derivative(enabled):
    """Select fused CUDA or allocation-heavy CuPy-roll five-point stencil."""
    global _FUSED_PERIODIC_DERIVATIVE
    _FUSED_PERIODIC_DERIVATIVE = bool(enabled)


def _fused_derivative_kernel(dtype, order):
    """Return a cached one-pass periodic stencil kernel for a complex dtype."""
    dtype = np.dtype(dtype)
    key = (dtype.str, int(order))
    cached = _DERIVATIVE_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    if dtype == np.dtype(np.complex128):
        vector_type, scalar_type, maker = "double2", "double", "make_double2"
    elif dtype == np.dtype(np.complex64):
        vector_type, scalar_type, maker = "float2", "float", "make_float2"
    else:
        raise TypeError("fused derivative는 complex64/complex128만 지원합니다.")
    if order == 1:
        real_expression = "vm2.x-8*vm1.x+8*vp1.x-vp2.x"
        imag_expression = "vm2.y-8*vm1.y+8*vp1.y-vp2.y"
    elif order == 2:
        real_expression = "-vm2.x+16*vm1.x-30*v0.x+16*vp1.x-vp2.x"
        imag_expression = "-vm2.y+16*vm1.y-30*v0.y+16*vp1.y-vp2.y"
    else:
        raise ValueError("order는 1 또는 2여야 합니다.")
    name = f"mcef_periodic_d{order}_{'c128' if dtype.itemsize == 16 else 'c64'}"
    code = f'''\
extern "C" __global__
void {name}(
    const {vector_type}* values, {vector_type}* output,
    const long long size, const int axis_length,
    const long long stride, const {scalar_type} scale
) {{
    const long long index = (long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (index >= size) return;
    const int coordinate = (int)((index/stride)%axis_length);
    const int cm2 = coordinate >= 2 ? coordinate-2 : coordinate+axis_length-2;
    const int cm1 = coordinate >= 1 ? coordinate-1 : axis_length-1;
    const int cp1 = coordinate+1 < axis_length ? coordinate+1 : 0;
    const int cp2 = coordinate+2 < axis_length ? coordinate+2 : coordinate+2-axis_length;
    const long long base = index-(long long)coordinate*stride;
    const {vector_type} vm2 = values[base+(long long)cm2*stride];
    const {vector_type} vm1 = values[base+(long long)cm1*stride];
    const {vector_type} v0 = values[index];
    const {vector_type} vp1 = values[base+(long long)cp1*stride];
    const {vector_type} vp2 = values[base+(long long)cp2*stride];
    output[index] = {maker}(
        ({real_expression})*scale,
        ({imag_expression})*scale
    );
}}
'''
    kernel = cp.RawKernel(code, name, options=("--std=c++11",))
    _DERIVATIVE_KERNEL_CACHE[key] = kernel
    return kernel


def _fused_periodic_derivative(values, spacing, axis, order):
    """Apply the same five-point stencil in one CUDA memory pass."""
    axis = np.core.numeric.normalize_axis_index(axis, values.ndim)
    if not values.flags.c_contiguous or values.dtype.kind != "c":
        return None
    output = cp.empty_like(values)
    stride = int(np.prod(values.shape[axis+1:], dtype=np.int64))
    scale = (
        1.0/(12.0*spacing) if order == 1
        else 1.0/(12.0*spacing**2)
    )
    scalar = np.float64(scale) if values.dtype.itemsize == 16 else np.float32(scale)
    threads = 256
    blocks = (values.size+threads-1)//threads
    _fused_derivative_kernel(values.dtype, order)(
        (blocks,), (threads,),
        (
            values, output, np.int64(values.size),
            np.int32(values.shape[axis]), np.int64(stride), scalar,
        ),
    )
    return output


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
    reuse_stage_derivatives: bool = True
    product_projection_floor_phi: float = 1.0e-10
    product_projection_floor_lam: float = 1.0e-10
    mask_residual_diagnostics: bool = False
    log_derivative_backend: str = "pointwise"
    weak_log_delta: float = 1.0e-10
    weak_log_smoothing: float = 0.04
    weak_log_tolerance: float = 1.0e-9
    weak_log_max_iterations: int = 40
    product_projection_backend: str = "nested_inverse"
    projection_tau_phi: float = 1.0e-10
    projection_tau_lam: float = 1.0e-10
    projection_tau_chi: float = 1.0e-10
    projection_support_epsilon: float = 1.0e-12
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


def make_gpu_model(
    cpu_model, precision, reuse_stage_derivatives=True,
    product_projection_floor_phi=1.0e-10,
    product_projection_floor_lam=1.0e-10,
    mask_residual_diagnostics=False,
):
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
        reuse_stage_derivatives=bool(reuse_stage_derivatives),
        product_projection_floor_phi=float(product_projection_floor_phi),
        product_projection_floor_lam=float(product_projection_floor_lam),
        mask_residual_diagnostics=bool(mask_residual_diagnostics),
        log_derivative_backend=cpu_model.log_derivative_backend,
        weak_log_delta=cpu_model.weak_log_delta,
        weak_log_smoothing=cpu_model.weak_log_smoothing,
        weak_log_tolerance=cpu_model.weak_log_tolerance,
        weak_log_max_iterations=cpu_model.weak_log_max_iterations,
        product_projection_backend=cpu_model.product_projection_backend,
        projection_tau_phi=cpu_model.projection_tau_phi,
        projection_tau_lam=cpu_model.projection_tau_lam,
        projection_tau_chi=cpu_model.projection_tau_chi,
        projection_support_epsilon=cpu_model.projection_support_epsilon,
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
    if _FUSED_PERIODIC_DERIVATIVE:
        fused = _fused_periodic_derivative(values, spacing, axis, order)
        if fused is not None:
            return fused
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
    momentum_factor=None,
):
    """GPU에서 phase gradient와 amplitude logarithmic gradient를 분리."""
    if numerical_floor <= 0.0:
        raise ValueError("numerical_floor는 양수여야 합니다.")
    density = cp.real(factor*cp.conj(factor))
    peak = cp.max(density, axis=axis, keepdims=True)
    tiny = cp.asarray(1.0e-30, dtype=density.dtype)
    safe = density+numerical_floor*cp.maximum(peak, tiny)
    if momentum_factor is None:
        momentum_factor = momentum(factor, spacing, axis)
    ratio = momentum_factor*cp.conj(factor)/safe
    return (
        ratio.real.astype(model.real_dtype, copy=False),
        (-ratio.imag).astype(model.real_dtype, copy=False),
    )


def weak_log_amplitude_gradient(
    factor, spacing, axis, model, *, delta=None, smoothing_length=None,
    tolerance=None, max_iterations=None,
):
    """Batched GPU PCG solve for the density-weighted weak log amplitude."""
    delta = model.weak_log_delta if delta is None else float(delta)
    smoothing_length = (
        model.weak_log_smoothing
        if smoothing_length is None else float(smoothing_length)
    )
    tolerance = (
        model.weak_log_tolerance if tolerance is None else float(tolerance)
    )
    max_iterations = (
        model.weak_log_max_iterations
        if max_iterations is None else int(max_iterations)
    )
    if delta <= 0.0 or smoothing_length < 0.0:
        raise ValueError("weak-log regularization 설정이 잘못되었습니다.")
    density = cp.real(factor*cp.conj(factor))
    tiny = cp.asarray(1.0e-30, dtype=density.dtype)
    peak = cp.max(density, axis=axis, keepdims=True)
    relative = density/cp.maximum(peak, tiny)
    rhs = 0.5*derivative(relative, spacing, axis=axis)
    diagonal_shift = smoothing_length**2*30.0/(12.0*spacing**2)
    preconditioner = relative+delta+diagonal_shift

    def apply(values):
        return (
            (relative+delta)*values
            -smoothing_length**2*derivative(
                values, spacing, axis=axis, order=2
            )
        )

    solution = cp.zeros_like(relative, dtype=model.real_dtype)
    residual = rhs.astype(model.real_dtype, copy=True)
    z = residual/preconditioner
    direction = z.copy()
    rz = cp.sum(
        residual*z, axis=axis, keepdims=True,
        dtype=model.reduction_real_dtype,
    ).astype(model.real_dtype, copy=False)
    rhs_norm = cp.sqrt(cp.sum(
        rhs*rhs, axis=axis, keepdims=True,
        dtype=model.reduction_real_dtype,
    )).astype(model.real_dtype, copy=False)
    scale = cp.maximum(rhs_norm, tiny)
    relative_residual = cp.ones_like(scale)
    iterations = 0
    for iterations in range(1, max_iterations+1):
        action = apply(direction)
        denominator = cp.sum(
            direction*action, axis=axis, keepdims=True,
            dtype=model.reduction_real_dtype,
        ).astype(model.real_dtype, copy=False)
        alpha = rz/cp.maximum(denominator, tiny)
        solution += alpha*direction
        residual -= alpha*action
        z = residual/preconditioner
        rz_new = cp.sum(
            residual*z, axis=axis, keepdims=True,
            dtype=model.reduction_real_dtype,
        ).astype(model.real_dtype, copy=False)
        beta = rz_new/cp.maximum(rz, tiny)
        direction = z+beta*direction
        rz = rz_new
        if iterations % 5 == 0 or iterations == max_iterations:
            relative_residual = cp.sqrt(cp.sum(
                residual*residual, axis=axis, keepdims=True,
                dtype=model.reduction_real_dtype,
            )).astype(model.real_dtype, copy=False)/scale
            if float(cp.max(relative_residual).get()) <= tolerance:
                break
    diagnostics = dict(
        weak_log_residual=cp.max(relative_residual),
        weak_log_iterations=cp.asarray(
            float(iterations), dtype=model.reduction_real_dtype
        ),
        weak_log_unconverged_lines=cp.count_nonzero(
            relative_residual > tolerance
        ).astype(model.reduction_real_dtype),
    )
    return solution, diagnostics


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


def covariant_square(
    field, vector, spacing, axis, sign, momentum_field=None,
):
    """독립 5점 D2와 Hermitian anticommutator로 covariant square 조립."""
    second = derivative(field, spacing, axis=axis, order=2)
    p_field = (
        momentum(field, spacing, axis=axis)
        if momentum_field is None else momentum_field
    )
    p_vector_field = momentum(vector*field, spacing, axis=axis)
    return (
        -second
        +sign*(p_vector_field+vector*p_field)
        +vector**2*field
    ).astype(field.dtype, copy=False)


def geometric_fields(phi, lam, model, return_momenta=False):
    """Vector potential ``a(q,R), b(q,R), alpha(R)``."""
    p_q_phi = momentum(phi, model.dq, axis=1)
    p_R_phi = momentum(phi, model.dR, axis=2)
    a = _real_inner_sum(cp.conj(phi)*p_q_phi, axis=0, model=model)*model.dx
    b = _real_inner_sum(cp.conj(phi)*p_R_phi, axis=0, model=model)*model.dx
    p_R_lam = momentum(lam, model.dR, axis=1)
    alpha = _real_inner_sum(
        cp.conj(lam)*(p_R_lam+b*lam), axis=0, model=model
    )*model.dq
    if return_momenta:
        return a, b, alpha, (p_q_phi, p_R_phi, p_R_lam)
    return a, b, alpha


def proton_base_operator(
    lam, a, b, alpha, chi_phase_R, chi_logamp_R, mask_lam, model,
    p_q_lam=None, p_R_lam=None, return_unmasked=False,
):
    """epsilon_1을 제외한 proton-heavy Hamiltonian을 Lambda에 적용."""
    proton_kinetic = covariant_square(
        lam, a, model.dq, axis=0, sign=+1,
        momentum_field=p_q_lam,
    )*(0.5/model.proton_mass)

    vector_R = b-alpha[None, :]
    if p_R_lam is None:
        p_R_lam = momentum(lam, model.dR, axis=1)
    dplus_R = p_R_lam+vector_R*lam
    dplus_R2 = covariant_square(
        lam, vector_R, model.dR, axis=1, sign=+1,
        momentum_field=p_R_lam,
    )
    coefficient = (
        chi_phase_R+alpha-1j*mask_lam*chi_logamp_R
    ).astype(model.complex_dtype, copy=False)
    coupling = (
        0.5*dplus_R2+coefficient[None, :]*dplus_R
    )/model.heavy_mass
    masked = proton_kinetic+coupling
    if not return_unmasked:
        return masked
    coefficient_unmasked = (
        chi_phase_R+alpha-1j*chi_logamp_R
    ).astype(model.complex_dtype, copy=False)
    unmasked = proton_kinetic+(
        0.5*dplus_R2+coefficient_unmasked[None, :]*dplus_R
    )/model.heavy_mass
    return masked, unmasked


def instantaneous_functionals(
    phi, lam, chi, model, floor=1.0e-14,
    mask_threshold_phi=1.0e-10, mask_threshold_lam=1.0e-10,
    include_unmasked=None,
):
    """현재 factor에서 두 TDPES와 세 vector potential을 계산한다."""
    reuse = getattr(model, "reuse_stage_derivatives", True)
    if include_unmasked is None:
        include_unmasked = getattr(model, "mask_residual_diagnostics", False)
    if reuse:
        a, b, alpha, momenta = geometric_fields(
            phi, lam, model, return_momenta=True
        )
        p_q_phi, p_R_phi, p_R_lam = momenta
        p_q_lam = momentum(lam, model.dq, axis=0)
    else:
        a, b, alpha = geometric_fields(phi, lam, model)
        p_q_phi = p_R_phi = p_q_lam = p_R_lam = None

    rho_R = cp.real(chi*cp.conj(chi))
    rho_qR = cp.real(lam*cp.conj(lam))*rho_R[None, :]
    mask_phi = occupied_support_mask(rho_qR, mask_threshold_phi, model)
    mask_lam = occupied_support_mask(rho_R, mask_threshold_lam, model)
    lam_phase_q, lam_logamp_q = logarithmic_components(
        lam, model.dq, axis=0, model=model, numerical_floor=floor,
        momentum_factor=p_q_lam,
    )
    lam_phase_R, lam_logamp_R = logarithmic_components(
        lam, model.dR, axis=1, model=model, numerical_floor=floor,
        momentum_factor=p_R_lam,
    )
    p_R_chi = momentum(chi, model.dR, axis=0) if reuse else None
    chi_phase_R, chi_logamp_R = logarithmic_components(
        chi, model.dR, axis=0, model=model, numerical_floor=floor,
        momentum_factor=p_R_chi,
    )
    weak_diagnostics = {}
    if model.log_derivative_backend == "weak":
        xi = lam*chi[None, :]
        xi_logamp_q, weak_q = weak_log_amplitude_gradient(
            xi, model.dq, axis=0, model=model
        )
        xi_logamp_R, weak_R = weak_log_amplitude_gradient(
            xi, model.dR, axis=1, model=model
        )
        chi_logamp_R_used, weak_chi = weak_log_amplitude_gradient(
            chi, model.dR, axis=0, model=model
        )
        weak_diagnostics = dict(
            weak_log_residual_q_xi=weak_q["weak_log_residual"],
            weak_log_residual_R_xi=weak_R["weak_log_residual"],
            weak_log_residual_R_chi=weak_chi["weak_log_residual"],
            weak_log_iterations=cp.maximum(
                weak_q["weak_log_iterations"], cp.maximum(
                    weak_R["weak_log_iterations"],
                    weak_chi["weak_log_iterations"],
                )
            ),
            weak_log_unconverged_lines=(
                weak_q["weak_log_unconverged_lines"]
                +weak_R["weak_log_unconverged_lines"]
                +weak_chi["weak_log_unconverged_lines"]
            ),
        )
    else:
        xi_logamp_q = lam_logamp_q
        xi_logamp_R = lam_logamp_R+chi_logamp_R[None, :]
        chi_logamp_R_used = chi_logamp_R

    dminus_q = (
        p_q_phi-a[None, :, :]*phi
        if reuse else
        _minus_covariant(phi, a[None, :, :], model.dq, axis=1)
    )
    dminus_q2 = covariant_square(
        phi, a[None, :, :], model.dq, axis=1, sign=-1,
        momentum_field=p_q_phi,
    )
    coefficient_q = (
        lam_phase_q+a-1j*mask_phi*xi_logamp_q
    ).astype(model.complex_dtype, copy=False)
    u_q_phi = (
        0.5*dminus_q2+coefficient_q[None, :, :]*dminus_q
    )/model.proton_mass

    dminus_R = (
        p_R_phi-b[None, :, :]*phi
        if reuse else
        _minus_covariant(phi, b[None, :, :], model.dR, axis=2)
    )
    dminus_R2 = covariant_square(
        phi, b[None, :, :], model.dR, axis=2, sign=-1,
        momentum_field=p_R_phi,
    )
    coefficient_R = (
        chi_phase_R[None, :]+lam_phase_R+b
        -1j*mask_phi*xi_logamp_R
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

    base_result = proton_base_operator(
        lam, a, b, alpha, chi_phase_R, chi_logamp_R_used, mask_lam, model,
        p_q_lam=p_q_lam, p_R_lam=p_R_lam,
        return_unmasked=include_unmasked,
    )
    if include_unmasked:
        base_lam_raw, base_lam_unmasked_raw = base_result
    else:
        base_lam_raw = base_result
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
    result = dict(
        a=a, b=b, alpha=alpha, epsilon_1=epsilon_1, epsilon_2=epsilon_2,
        u_phi=u_phi, hpr_lam=hpr_lam,
        # coupled_rhs의 outer-heavy covariant square가 같은 D_R chi를
        # 다시 계산하지 않도록 한 stage 안에서만 재사용한다.
        _p_R_chi=p_R_chi,
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
            cp.abs(mask_phi*xi_logamp_q),
            cp.abs(mask_phi*xi_logamp_R),
        ),
        raw_logamp_lam=cp.abs(chi_logamp_R),
        effective_logamp_lam=cp.abs(mask_lam*chi_logamp_R_used),
        **weak_diagnostics,
    )
    if include_unmasked:
        coefficient_q_unmasked = (
            lam_phase_q+a-1j*xi_logamp_q
        ).astype(model.complex_dtype, copy=False)
        coefficient_R_unmasked = (
            chi_phase_R[None, :]+lam_phase_R+b
            -1j*xi_logamp_R
        ).astype(model.complex_dtype, copy=False)
        u_phi_unmasked_raw = (
            0.5*dminus_q2
            +coefficient_q_unmasked[None, :, :]*dminus_q
        )/model.proton_mass+(
            0.5*dminus_R2
            +coefficient_R_unmasked[None, :, :]*dminus_R
        )/model.heavy_mass
        (
            u_phi_unmasked, gamma_phi_unmasked, _, _
        ) = remove_local_norm_generator(
            phi, u_phi_unmasked_raw, model.dx, axis=0, model=model
        )
        epsilon_1_unmasked = _real_inner_sum(
            cp.conj(phi)*(hbo_phi+u_phi_unmasked), axis=0, model=model
        )*model.dx
        hpr_lam_unmasked_raw = (
            base_lam_unmasked_raw+epsilon_1_unmasked*lam
            +1j*gamma_phi_unmasked*lam
        ).astype(model.complex_dtype, copy=False)
        (
            hpr_lam_unmasked, gamma_lam_unmasked, _, _
        ) = remove_local_norm_generator(
            lam, hpr_lam_unmasked_raw, model.dq, axis=0, model=model
        )
        epsilon_2_unmasked = _real_inner_sum(
            cp.conj(lam)*hpr_lam_unmasked, axis=0, model=model
        )*model.dq
        result.update(
            u_phi_unmasked=u_phi_unmasked,
            hpr_lam_unmasked=hpr_lam_unmasked,
            epsilon_1_unmasked=epsilon_1_unmasked,
            epsilon_2_unmasked=epsilon_2_unmasked,
            gamma_lam_unmasked=gamma_lam_unmasked,
        )
    return result


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
    unmasked_rhs=None, residual_support_weight=None,
):
    """GPU factor RHS를 periodic full nuclear D2의 nested tangent에 투영."""
    if support_floor_phi < 0.0 or support_floor_lam < 0.0:
        raise ValueError("product projection support floor는 0 이상이어야 합니다.")
    xi = lam*chi[None, :]
    reuse = getattr(model, "reuse_stage_derivatives", True)
    if reuse:
        # 같은 곱미분 법칙을 먼저 xi=Lambda*chi 수준에서 묶으면 큰
        # (nx,nq,nR) 임시항 하나를 만들지 않아도 된다.
        dxi = dlam*chi[None, :]+lam*dchi[None, :]
        psi = phi*xi[None, :, :]
        product_rhs = dphi*xi[None, :, :]+phi*dxi[None, :, :]
    else:
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
    if unmasked_rhs is None:
        residual_without_mask = None
        residual_due_to_mask = None
    else:
        uphi, ulam, uchi = unmasked_rhs
        if reuse:
            unmasked_dxi = ulam*chi[None, :]+lam*uchi[None, :]
            unmasked_product_rhs = (
                uphi*xi[None, :, :]+phi*unmasked_dxi[None, :, :]
            )
        else:
            unmasked_product_rhs = (
                uphi*lam[None, :, :]*chi[None, None, :]
                +phi*ulam[None, :, :]*chi[None, None, :]
                +phi*lam[None, :, :]*uchi[None, None, :]
            )
        residual_without_mask = target_rhs-unmasked_product_rhs
        residual_due_to_mask = unmasked_product_rhs-product_rhs

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

    xi_density = cp.real(xi*cp.conj(xi))
    tiny = cp.asarray(1.0e-30, dtype=xi_density.dtype)
    xi_peak = cp.maximum(cp.max(xi_density), tiny)
    projection_backend = model.product_projection_backend
    if projection_backend == "weighted_tikhonov":
        support_epsilon = cp.asarray(
            model.projection_support_epsilon, dtype=xi_density.dtype
        )
        support_phi = xi_density/(
            xi_density+support_floor_phi*xi_peak+tiny
        )
        ridge_phi = (
            model.projection_tau_phi*xi_peak/(support_phi+support_epsilon)
        )
        inverse_xi = support_phi*cp.conj(xi)/(
            support_phi*xi_density+ridge_phi+tiny
        )
    elif projection_backend == "nested_inverse":
        support_phi = xi_density/(
            xi_density+support_floor_phi*xi_peak+tiny
        )
        inverse_xi = cp.conj(xi)/(
            xi_density+support_floor_phi*xi_peak
        )
    else:
        raise ValueError(f"지원하지 않는 product projection: {projection_backend}")
    delta_phi = perpendicular_phi*inverse_xi[None, :, :]

    lam_norm2 = cp.sum(
        cp.real(lam*cp.conj(lam)), axis=0,
        dtype=model.reduction_real_dtype,
    )*model.dq
    parallel_chi = cp.sum(
        cp.conj(lam)*delta_xi, axis=0,
        dtype=model.reduction_complex_dtype,
    )*model.dq/cp.maximum(lam_norm2, phi_floor)
    parallel_chi = parallel_chi.astype(model.complex_dtype, copy=False)
    # Strong tangent gauge: the parallel chi block and perpendicular Lambda
    # block form the structured simultaneous minimum-norm decomposition.
    perpendicular_lam = delta_xi-lam*parallel_chi[None, :]
    chi_density = cp.real(chi*cp.conj(chi))
    chi_peak = cp.maximum(cp.max(chi_density), tiny)
    if projection_backend == "weighted_tikhonov":
        support_lam = chi_density/(
            chi_density+support_floor_lam*chi_peak+tiny
        )
        ridge_lam = (
            model.projection_tau_lam*chi_peak/(support_lam+support_epsilon)
        )
        inverse_chi = support_lam*cp.conj(chi)/(
            support_lam*chi_density+ridge_lam+tiny
        )
        chi_shrink = support_lam/(
            support_lam
            +model.projection_tau_chi/(support_lam+support_epsilon)
            +tiny
        )
        delta_chi = (chi_shrink*parallel_chi).astype(
            model.complex_dtype, copy=False
        )
    else:
        support_lam = chi_density/(
            chi_density+support_floor_lam*chi_peak+tiny
        )
        inverse_chi = cp.conj(chi)/(
            chi_density+support_floor_lam*chi_peak
        )
        delta_chi = parallel_chi
    delta_lam = perpendicular_lam*inverse_chi[None, :]

    dphi = (dphi+delta_phi).astype(model.complex_dtype, copy=False)
    dlam = (dlam+delta_lam).astype(model.complex_dtype, copy=False)
    dchi = (dchi+delta_chi).astype(model.complex_dtype, copy=False)
    if reuse:
        corrected_dxi = dlam*chi[None, :]+lam*dchi[None, :]
        corrected_product_rhs = (
            dphi*xi[None, :, :]+phi*corrected_dxi[None, :, :]
        )
    else:
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

    target_l2 = l2(target_rhs)
    relative_floor = cp.maximum(
        target_l2, cp.asarray(1.0e-300, dtype=target_l2.dtype)
    )
    projection_product_rhs = corrected_product_rhs-product_rhs
    relative_product_projection_l2 = l2(
        projection_product_rhs
    )/relative_floor
    support_weight = (xi_density/xi_peak)[None, :, :]

    def physical_support_l2(values):
        return cp.sqrt(cp.sum(
            support_weight*cp.real(values*cp.conj(values)),
            dtype=model.reduction_real_dtype,
        )*volume)

    support_target_l2_physical = physical_support_l2(target_rhs)
    relative_support_product_projection_l2 = physical_support_l2(
        projection_product_rhs
    )/cp.maximum(
        support_target_l2_physical,
        cp.asarray(1.0e-300, dtype=support_target_l2_physical.dtype),
    )

    joint_total = cp.maximum(
        cp.sum(xi_density, dtype=model.reduction_real_dtype)
        *model.dq*model.dR,
        cp.asarray(1.0e-300, dtype=model.reduction_real_dtype),
    )
    edge_width_q = min(2, xi_density.shape[0]//2)
    edge_width_R = min(2, xi_density.shape[1]//2)
    outer_probability_q = (
        cp.sum(xi_density[:edge_width_q], dtype=model.reduction_real_dtype)
        +cp.sum(xi_density[-edge_width_q:], dtype=model.reduction_real_dtype)
    )*model.dq*model.dR/joint_total
    outer_probability_R = (
        cp.sum(xi_density[:, :edge_width_R], dtype=model.reduction_real_dtype)
        +cp.sum(xi_density[:, -edge_width_R:], dtype=model.reduction_real_dtype)
    )*model.dq*model.dR/joint_total
    psi_norm2 = cp.maximum(
        cp.sum(
            cp.real(psi*cp.conj(psi)), dtype=model.reduction_real_dtype
        )*volume,
        cp.asarray(1.0e-300, dtype=model.reduction_real_dtype),
    )
    relative_psi_wrap_mismatch_q = cp.sqrt(
        cp.sum(
            cp.abs(psi[:, 0, :]-psi[:, -1, :])**2,
            dtype=model.reduction_real_dtype,
        )*model.dx*model.dR*model.dq/psi_norm2
    )
    relative_psi_wrap_mismatch_R = cp.sqrt(
        cp.sum(
            cp.abs(psi[:, :, 0]-psi[:, :, -1])**2,
            dtype=model.reduction_real_dtype,
        )*model.dx*model.dq*model.dR/psi_norm2
    )

    zero = cp.asarray(0.0, dtype=model.reduction_real_dtype)
    if residual_without_mask is None:
        residual_without_mask_l2 = zero
        residual_due_to_mask_l2 = zero
        alignment_floor = cp.asarray(1.0e-300, dtype=zero.dtype)
        alignment = zero
        support_without_mask_l2 = zero
        support_due_to_mask_l2 = zero
        relative_support_without_mask = zero
        relative_support_due_to_mask = zero
    else:
        residual_without_mask_l2 = l2(residual_without_mask)
        residual_due_to_mask_l2 = l2(residual_due_to_mask)
        relative_floor = cp.maximum(
            target_l2, cp.asarray(1.0e-300, dtype=target_l2.dtype)
        )
        alignment_floor = cp.asarray(1.0e-300, dtype=target_l2.dtype)
        alignment_overlap = cp.sum(
            cp.conj(residual_without_mask)*residual_due_to_mask,
            dtype=model.reduction_complex_dtype,
        ).real*volume
        alignment = cp.where(
            residual_without_mask_l2*residual_due_to_mask_l2 > alignment_floor,
            alignment_overlap/cp.maximum(
                residual_without_mask_l2*residual_due_to_mask_l2,
                alignment_floor,
            ),
            0.0,
        )
        weight = (
            cp.ones(phi.shape[1:], dtype=model.real_dtype)
            if residual_support_weight is None else residual_support_weight
        )[None, :, :]

        def weighted_l2(values):
            return cp.sqrt(cp.sum(
                weight*cp.real(values*cp.conj(values)),
                dtype=model.reduction_real_dtype,
            )*volume)

        support_target_l2 = weighted_l2(target_rhs)
        support_without_mask_l2 = weighted_l2(residual_without_mask)
        support_due_to_mask_l2 = weighted_l2(residual_due_to_mask)
        support_relative_floor = cp.maximum(
            support_target_l2,
            cp.asarray(1.0e-300, dtype=support_target_l2.dtype),
        )
        relative_support_without_mask = (
            support_without_mask_l2/support_relative_floor
        )
        relative_support_due_to_mask = (
            support_due_to_mask_l2/support_relative_floor
        )
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
        inverse_support_product_correction_phi=cp.sqrt(cp.sum(
            cp.real(delta_phi*cp.conj(delta_phi))/(
                support_phi[None, :, :]+1.0e-12
            ), dtype=model.reduction_real_dtype,
        )*volume),
        inverse_support_product_correction_lam=cp.sqrt(cp.sum(
            cp.real(delta_lam*cp.conj(delta_lam))/(
                support_lam[None, :]+1.0e-12
            ), dtype=model.reduction_real_dtype,
        )*model.dq*model.dR),
        inverse_support_product_correction_chi=cp.sqrt(cp.sum(
            cp.real(delta_chi*cp.conj(delta_chi))/(
                support_lam+1.0e-12
            ), dtype=model.reduction_real_dtype,
        )*model.dR),
        relative_product_projection_l2=relative_product_projection_l2,
        relative_support_product_projection_l2=(
            relative_support_product_projection_l2
        ),
        outer_probability_q=outer_probability_q,
        outer_probability_R=outer_probability_R,
        relative_psi_wrap_mismatch_q=relative_psi_wrap_mismatch_q,
        relative_psi_wrap_mismatch_R=relative_psi_wrap_mismatch_R,
        product_residual_without_mask_l2=residual_without_mask_l2,
        product_residual_due_to_mask_l2=residual_due_to_mask_l2,
        relative_product_residual_without_mask=(
            residual_without_mask_l2/relative_floor
        ),
        relative_product_residual_due_to_mask=(
            residual_due_to_mask_l2/relative_floor
        ),
        product_mask_nonmask_alignment=alignment,
        product_mask_nonmask_alignment_positive=cp.maximum(alignment, zero),
        product_mask_nonmask_alignment_negative_magnitude=cp.maximum(
            -alignment, zero
        ),
        support_product_residual_without_mask_l2=support_without_mask_l2,
        support_product_residual_due_to_mask_l2=support_due_to_mask_l2,
        relative_support_product_residual_without_mask=(
            relative_support_without_mask
        ),
        relative_support_product_residual_due_to_mask=(
            relative_support_due_to_mask
        ),
    )
    return dphi, dlam, dchi, diagnostics


def coupled_rhs(
    phi, lam, chi, model, ratio_floor, mask_threshold_phi,
    mask_threshold_lam, include_mask_diagnostics=False,
):
    """H_BO split 부분을 제외한 세 coupled RHS."""
    fields = instantaneous_functionals(
        phi, lam, chi, model, floor=ratio_floor,
        mask_threshold_phi=mask_threshold_phi,
        mask_threshold_lam=mask_threshold_lam,
        include_unmasked=include_mask_diagnostics,
    )
    dphi = -1j*(
        fields["u_phi"]-fields["epsilon_1"][None, :, :]*phi
    )
    dlam = -1j*(
        fields["hpr_lam"]-fields["epsilon_2"][None, :]*lam
    )
    alpha = fields["alpha"]
    p2chi = covariant_square(
        chi, alpha, model.dR, axis=0, sign=+1,
        momentum_field=fields["_p_R_chi"],
    )
    dchi = (
        -1j*(0.5*p2chi/model.heavy_mass+fields["epsilon_2"]*chi)
        +fields["gamma_lam"]*chi
    )
    unmasked_rhs = None
    if include_mask_diagnostics:
        dphi_unmasked = -1j*(
            fields["u_phi_unmasked"]
            -fields["epsilon_1_unmasked"][None, :, :]*phi
        )
        dlam_unmasked = -1j*(
            fields["hpr_lam_unmasked"]
            -fields["epsilon_2_unmasked"][None, :]*lam
        )
        dchi_unmasked = (
            -1j*(
                0.5*p2chi/model.heavy_mass
                +fields["epsilon_2_unmasked"]*chi
            )+fields["gamma_lam_unmasked"]*chi
        )
        unmasked_rhs = (dphi_unmasked, dlam_unmasked, dchi_unmasked)
    dphi, dlam, dchi, product_diagnostics = project_discrete_product_residual(
        phi, lam, chi, dphi, dlam, dchi, model,
        support_floor_phi=model.product_projection_floor_phi,
        support_floor_lam=model.product_projection_floor_lam,
        unmasked_rhs=unmasked_rhs,
        residual_support_weight=(
            fields["mask_phi"] if unmasked_rhs is not None else None
        ),
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
    "max_relative_product_projection_l2": "relative_product_projection_l2",
    "max_relative_support_product_projection_l2": (
        "relative_support_product_projection_l2"
    ),
    "max_inverse_support_product_correction_phi": (
        "inverse_support_product_correction_phi"
    ),
    "max_inverse_support_product_correction_lam": (
        "inverse_support_product_correction_lam"
    ),
    "max_inverse_support_product_correction_chi": (
        "inverse_support_product_correction_chi"
    ),
    "max_weak_log_residual_q_xi": "weak_log_residual_q_xi",
    "max_weak_log_residual_R_xi": "weak_log_residual_R_xi",
    "max_weak_log_residual_R_chi": "weak_log_residual_R_chi",
    "max_weak_log_iterations": "weak_log_iterations",
    "max_weak_log_unconverged_lines": "weak_log_unconverged_lines",
    "max_outer_probability_q": "outer_probability_q",
    "max_outer_probability_R": "outer_probability_R",
    "max_relative_psi_wrap_mismatch_q": "relative_psi_wrap_mismatch_q",
    "max_relative_psi_wrap_mismatch_R": "relative_psi_wrap_mismatch_R",
    "max_product_residual_without_mask_l2": (
        "product_residual_without_mask_l2"
    ),
    "max_product_residual_due_to_mask_l2": "product_residual_due_to_mask_l2",
    "max_relative_product_residual_without_mask": (
        "relative_product_residual_without_mask"
    ),
    "max_relative_product_residual_due_to_mask": (
        "relative_product_residual_due_to_mask"
    ),
    "max_abs_product_mask_nonmask_alignment": (
        "product_mask_nonmask_alignment"
    ),
    "max_product_mask_nonmask_alignment_positive": (
        "product_mask_nonmask_alignment_positive"
    ),
    "max_product_mask_nonmask_alignment_negative_magnitude": (
        "product_mask_nonmask_alignment_negative_magnitude"
    ),
    "max_support_product_residual_without_mask_l2": (
        "support_product_residual_without_mask_l2"
    ),
    "max_support_product_residual_due_to_mask_l2": (
        "support_product_residual_due_to_mask_l2"
    ),
    "max_relative_support_product_residual_without_mask": (
        "relative_support_product_residual_without_mask"
    ),
    "max_relative_support_product_residual_due_to_mask": (
        "relative_support_product_residual_due_to_mask"
    ),
}


def evaluate_mask_residual_diagnostics(
    phi, lam, chi, model, ratio_floor, mask_threshold_phi,
    mask_threshold_lam,
):
    """Evaluate mask-on/off product residuals without advancing the state."""
    _, _, _, diagnostics = coupled_rhs(
        phi, lam, chi, model, ratio_floor, mask_threshold_phi,
        mask_threshold_lam, include_mask_diagnostics=True,
    )
    return diagnostics


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
