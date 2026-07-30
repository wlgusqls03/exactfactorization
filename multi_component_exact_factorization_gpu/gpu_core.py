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
    """GPU periodic 중앙 유한차분. Shape은 입력과 동일하다."""
    source = cp.moveaxis(values, axis, 0)
    result = cp.empty_like(source)
    if order == 1:
        scale = 1.0/(2.0*spacing)
        result[1:-1] = (source[2:]-source[:-2])*scale
        result[0] = (source[1]-source[-1])*scale
        result[-1] = (source[0]-source[-2])*scale
        return cp.moveaxis(result, 0, axis)
    if order == 2:
        scale = 1.0/spacing**2
        result[1:-1] = (source[2:]-2.0*source[1:-1]+source[:-2])*scale
        result[0] = (source[1]-2.0*source[0]+source[-1])*scale
        result[-1] = (source[0]-2.0*source[-1]+source[-2])*scale
        return cp.moveaxis(result, 0, axis)
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


def proton_base_operator(lam, chi, a, b, alpha, model, floor):
    """epsilon_1을 제외한 proton-heavy Hamiltonian을 Lambda에 적용."""
    dplus_q = _plus_covariant(lam, a, model.dq, axis=0)
    proton_kinetic = _plus_covariant(dplus_q, a, model.dq, axis=0)
    proton_kinetic *= 0.5/model.proton_mass

    vector_R = b-alpha[None, :]
    dplus_R = _plus_covariant(lam, vector_R, model.dR, axis=1)
    dplus_R2 = _plus_covariant(dplus_R, vector_R, model.dR, axis=1)
    ratio_chi_R = regularized_ratio(
        momentum(chi, model.dR, axis=0), chi, floor
    )
    coupling = (
        0.5*dplus_R2+(ratio_chi_R+alpha)[None, :]*dplus_R
    )/model.heavy_mass
    return proton_kinetic+coupling


def instantaneous_functionals(phi, lam, chi, model, floor=1.0e-10):
    """현재 factor에서 두 TDPES와 세 vector potential을 계산한다."""
    a, b, alpha = geometric_fields(phi, lam, model)

    dminus_q = _minus_covariant(phi, a[None, :, :], model.dq, axis=1)
    dminus_q2 = _minus_covariant(dminus_q, a[None, :, :], model.dq, axis=1)
    ratio_lam_q = regularized_ratio(
        momentum(lam, model.dq, axis=0), lam, floor
    )
    u_q_phi = (
        0.5*dminus_q2+(ratio_lam_q+a)[None, :, :]*dminus_q
    )/model.proton_mass

    dminus_R = _minus_covariant(phi, b[None, :, :], model.dR, axis=2)
    dminus_R2 = _minus_covariant(dminus_R, b[None, :, :], model.dR, axis=2)
    ratio_chi_R = regularized_ratio(
        momentum(chi, model.dR, axis=0), chi, floor
    )
    ratio_lam_R = regularized_ratio(
        momentum(lam, model.dR, axis=1), lam, floor
    )
    u_R_phi = (
        0.5*dminus_R2
        +(ratio_chi_R[None, :]+ratio_lam_R+b)[None, :, :]*dminus_R
    )/model.heavy_mass
    u_phi = u_q_phi+u_R_phi

    hbo_phi = apply_electronic_hamiltonian(phi, model)
    epsilon_1 = _real_inner_sum(
        cp.conj(phi)*(hbo_phi+u_phi), axis=0, model=model
    )*model.dx

    base_lam = proton_base_operator(lam, chi, a, b, alpha, model, floor)
    hpr_lam = base_lam+epsilon_1*lam
    epsilon_2 = _real_inner_sum(
        cp.conj(lam)*hpr_lam, axis=0, model=model
    )*model.dq
    return dict(
        a=a, b=b, alpha=alpha, epsilon_1=epsilon_1, epsilon_2=epsilon_2,
        u_phi=u_phi, hpr_lam=hpr_lam,
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


def coupled_rhs(phi, lam, chi, model, density_threshold):
    """H_BO split 부분을 제외한 세 coupled RHS."""
    fields = instantaneous_functionals(
        phi, lam, chi, model, floor=density_threshold
    )
    dphi = -1j*(
        fields["u_phi"]-fields["epsilon_1"][None, :, :]*phi
    )
    dlam = -1j*(
        fields["hpr_lam"]-fields["epsilon_2"][None, :]*lam
    )
    alpha = fields["alpha"]
    pchi = -1j*derivative(chi, model.dR, axis=0)+alpha*chi
    p2chi = -1j*derivative(pchi, model.dR, axis=0)+alpha*pchi
    dchi = -1j*(0.5*p2chi/model.heavy_mass+fields["epsilon_2"]*chi)
    return dphi, dlam, dchi


def rk4_coupled_step(phi, lam, chi, dt, model, density_threshold):
    """GPU에서 네 번의 RHS를 평가하는 고전 RK4 coupled substep."""
    k1p, k1l, k1c = coupled_rhs(phi, lam, chi, model, density_threshold)
    k2p, k2l, k2c = coupled_rhs(
        phi+0.5*dt*k1p, lam+0.5*dt*k1l, chi+0.5*dt*k1c,
        model, density_threshold,
    )
    k3p, k3l, k3c = coupled_rhs(
        phi+0.5*dt*k2p, lam+0.5*dt*k2l, chi+0.5*dt*k2c,
        model, density_threshold,
    )
    k4p, k4l, k4c = coupled_rhs(
        phi+dt*k3p, lam+dt*k3l, chi+dt*k3c,
        model, density_threshold,
    )
    phi = phi+dt*(k1p+2*k2p+2*k3p+k4p)/6.0
    lam = lam+dt*(k1l+2*k2l+2*k3l+k4l)/6.0
    chi = chi+dt*(k1c+2*k2c+2*k3c+k4c)/6.0
    return pnc_project(phi, lam, chi, model)


def full_step(phi, lam, chi, dt, model, density_threshold):
    """``H_BO 반 -> coupled RK4 -> H_BO 반`` 한 time step."""
    phi = electronic_split_step(phi, 0.5*dt, model)
    phi, lam, chi, first_error = rk4_coupled_step(
        phi, lam, chi, dt, model, density_threshold
    )
    phi = electronic_split_step(phi, 0.5*dt, model)
    phi, lam, chi, final_error = pnc_project(phi, lam, chi, model)
    return phi, lam, chi, cp.maximum(first_error, final_error)


def all_finite(phi, lam, chi):
    """세 GPU factor가 모두 finite인지 한 번의 host sync로 확인한다."""
    valid = cp.all(cp.isfinite(phi)) & cp.all(cp.isfinite(lam)) & cp.all(cp.isfinite(chi))
    return bool(valid.get())
