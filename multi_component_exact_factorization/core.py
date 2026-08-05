"""Multi-component EF에서 공통으로 쓰는 격자, 연산자, 물리량 계산.

좌표와 배열 축은 코드 전체에서 다음 순서를 유지한다.

    x : 전자 좌표                 (nx,)
    q : 양성자(수소 핵) 좌표      (nq,)
    R : 오른쪽 무거운 핵 좌표     (nR,)

Nested exact factorization은

    Psi(x,q,R,t) = Phi_{R,q}(x,t) Lambda_R(q,t) chi(R,t)

로 쓴다. 따라서 주요 배열 shape은

    phi     (nx,nq,nR)   조건부 전자 파동함수
    lam     (nq,nR)      R에 조건부인 양성자 파동함수
    chi     (nR,)        무거운 핵의 marginal 파동함수

이다. 이 파일은 교육용으로 수식을 가능한 한 코드와 가까운 순서로
배치했다. 속도보다 각 항이 어디에서 오는지 읽기 쉬운 것을 우선한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.fft import dst, idst
from scipy.linalg import eigh_tridiagonal


AU_PER_FS = 41.3413745758


@dataclass
class Model:
    """격자와 Hamiltonian parameter를 한데 모은 자료구조."""

    x: np.ndarray                 # (nx,) 전자 격자
    q: np.ndarray                 # (nq,) 양성자 격자
    R: np.ndarray                 # (nR,) 무거운 핵 격자
    dx: float
    dq: float
    dR: float
    x_left: float
    x_right: float
    proton_mass: float
    heavy_mass: float
    fft_workers: int             # 전자 DST에 사용할 CPU worker 수
    potential: np.ndarray         # (nx,nq,nR)


def soft_inverse(distance: np.ndarray, softening: float) -> np.ndarray:
    """1D Coulomb 특이점을 완화한 ``1/sqrt(d^2+a^2)``."""
    return 1.0 / np.sqrt(distance**2 + softening**2)


def build_model(args) -> Model:
    """왼쪽 고정점-전자-양성자-무거운 핵의 1D model을 만든다.

    왼쪽 중심은 움직이지 않으므로 파동함수 좌표축을 갖지 않는다. 전자에게는
    potential 중심인 동시에 넘을 수 없는 왼쪽 Dirichlet 경계다. q와 R box는
    초기 packet의 tail이 수치 경계에서 충분히 작도록 잡는다.
    """
    # 전자는 왼쪽 고정점 x_L을 넘을 수 없는 Dirichlet hard-wall box에 둔다.
    # 따라서 x_L과 x_max 자체는 psi=0인 경계이고, 실제 배열에는 그 사이의
    # nx개 interior point만 저장한다. 이 배치는 DST-I kinetic과 정확히 맞는다.
    x_left = float(args.left_position)
    x_right = float(args.x_max)
    if x_right <= x_left:
        raise ValueError("--x-max는 왼쪽 고정점 --left-position보다 커야 합니다.")
    dx = (x_right-x_left)/(args.nx+1)
    x = x_left+dx*np.arange(1, args.nx+1)

    # q와 R은 density가 경계에서 충분히 작다는 전제 아래 finite box에 둔다.
    # 배열 배치는 기존 archive와 호환되도록 endpoint=False를 유지하지만,
    # 미분은 경계를 wrap하지 않는 5점 one-sided stencil을 사용한다.
    q = np.linspace(args.q_min, args.q_max, args.nq, endpoint=False)
    R = np.linspace(args.R_min, args.R_max, args.nR, endpoint=False)
    dq = float(q[1] - q[0])
    dR = float(R[1] - R[0])

    # 저장 metadata에서도 전자 box의 왼쪽 끝이 고정점임을 명시한다.
    args.x_min = x_left

    xx = x[:, None, None]       # (nx,1,1)
    qq = q[None, :, None]       # (1,nq,1)
    RR = R[None, None, :]       # (1,1,nR)

    # 전자(-1)는 세 양전하 중심에 끌리고, 양전하끼리는 서로 밀어낸다.
    # 모든 항은 broadcasting되어 최종 shape (nx,nq,nR)가 된다.
    potential = (
        -args.left_charge * soft_inverse(xx - args.left_position, args.soft_e_left)
        -soft_inverse(xx - qq, args.soft_e_proton)
        -args.heavy_charge * soft_inverse(xx - RR, args.soft_e_heavy)
        +args.left_charge * soft_inverse(qq - args.left_position, args.soft_p_left)
        +args.heavy_charge * soft_inverse(qq - RR, args.soft_p_heavy)
        +args.left_charge * args.heavy_charge
        * soft_inverse(RR - args.left_position, args.soft_left_heavy)
    )

    return Model(
        x=x, q=q, R=R, dx=dx, dq=dq, dR=dR,
        x_left=x_left, x_right=x_right,
        proton_mass=args.proton_mass, heavy_mass=args.heavy_mass,
        # 예전 archive metadata에는 이 option이 없으므로 재분석 시 fallback한다.
        fft_workers=getattr(args, "fft_workers", -1),
        potential=np.asarray(potential),
    )


def normalized_gaussian(grid, spacing, center, sigma, momentum=0.0):
    """``|psi|^2``의 표준편차가 ``sigma``인 정규화 Gaussian.

    확률밀도를 ``exp[-(x-x0)^2/(2 sigma^2)]``로 만들려면 파동함수
    amplitude는 그 제곱근인 ``exp[-(x-x0)^2/(4 sigma^2)]``여야 한다.
    무한 공간의 amplitude 정규화 상수는 ``(2*pi*sigma^2)^(-1/4)``이다.
    유한 격자/box에서는 이 상수를 그대로 쓰는 대신 아래의 수치 적분으로
    정확히 다시 정규화한다.
    """
    if sigma <= 0.0:
        raise ValueError("Gaussian 표준편차 sigma는 양수여야 합니다.")
    wave = np.exp(-0.25*((grid-center)/sigma)**2 + 1j*momentum*(grid-center))
    norm = np.sqrt(np.sum(np.abs(wave)**2)*spacing)
    return wave/norm


def local_electronic_basis(model: Model, n_states: int):
    """각 ``(q,R)``에서 clamped electronic Hamiltonian의 낮은 상태를 푼다.

    전자 초기상태는 항상 이 BO형 초기화를 사용한다. 전파 자체는 이후에도
    coupled exact-factorization 방정식을 사용한다.

    계산하는 고유값 문제는 각 configuration마다

        H_BO(x;q_j,R_k) varphi_n(x;q_j,R_k) = E_n(q_j,R_k) varphi_n

    이다. 반환 shape은 ``energies(n_states,nq,nR)``와
    ``states(n_states,nx,nq,nR)``이다. 서로 이웃한 configuration의
    eigenvector overlap이 양의 실수가 되도록 phase를 맞춰 q/R 미분에서
    임의의 eigenvector 부호가 튀는 현상을 줄인다.
    """
    if n_states < 1 or n_states > len(model.x):
        raise ValueError("전자 상태 수는 1 이상 nx 이하여야 합니다.")

    nx, nq, nR = len(model.x), len(model.q), len(model.R)

    # ------------------------------------------------------------------
    # 1. 전자 운동에너지 T_x의 실공간 matrix를 한 번만 만든다.
    # ------------------------------------------------------------------
    # Dirichlet 경계의 2차 중앙차분 T_x=-d_x^2/2는 tridiagonal이다.
    # diagonal=1/dx^2, off-diagonal=-1/(2 dx^2)이므로 전체 dense matrix를
    # 만들지 않고 필요한 낮은 고유상태만 효율적으로 구할 수 있다.
    kinetic_diagonal = np.full(nx, 1.0/model.dx**2)
    kinetic_offdiagonal = np.full(nx-1, -0.5/model.dx**2)

    energies = np.empty((n_states, nq, nR), float)                 # (state,nq,nR)
    states = np.empty((n_states, nx, nq, nR), complex)             # (state,nx,nq,nR)

    # ------------------------------------------------------------------
    # 2. 모든 (q_j,R_k)에서 작은 nx x nx Hermitian 문제를 독립적으로 푼다.
    # ------------------------------------------------------------------
    for iR in range(nR):
        for iq in range(nq):
            diagonal = kinetic_diagonal+model.potential[:, iq, iR]
            values, vectors = eigh_tridiagonal(
                diagonal, kinetic_offdiagonal,
                select="i", select_range=(0, n_states-1),
            )

            # np.linalg.eigh의 열벡터는 sum_x |v_x|^2=1이다. 연속 PNC
            # sum_x |phi_x|^2 dx=1에 맞추기 위해 sqrt(dx)로 나눈다.
            chosen = vectors.T.astype(complex)/np.sqrt(model.dx)

            # ----------------------------------------------------------
            # 3. Eigenvector가 갖는 임의의 complex phase를 매끈하게 잇는다.
            # ----------------------------------------------------------
            # H v=E v이면 exp(i beta)v도 같은 고유벡터다. 이 phase를 각
            # configuration에서 제멋대로 두면 d_q Phi, d_R Phi가 가짜 spike를
            # 만든다. 먼저 q 방향, 각 q 행의 시작에서는 R 방향 이웃과 맞춘다.
            for state in range(n_states):
                if iq > 0:
                    reference = states[state, :, iq-1, iR]
                elif iR > 0:
                    reference = states[state, :, iq, iR-1]
                else:
                    # 첫 configuration에서는 최대 성분을 양의 실수로 둔다.
                    pivot = int(np.argmax(np.abs(chosen[state])))
                    phase = np.angle(chosen[state, pivot])
                    chosen[state] *= np.exp(-1j*phase)
                    continue
                overlap = np.sum(np.conj(reference)*chosen[state])*model.dx
                if abs(overlap) > 1.0e-12:
                    chosen[state] *= np.exp(-1j*np.angle(overlap))
            energies[:, iq, iR] = values
            states[:, :, iq, iR] = chosen
    return energies, states


def independent_surface_curvatures(surface, model: Model, q0: float, R0: float):
    """BO energy를 q와 R 방향으로 각각 독립적으로 두 번 미분한다.

    먼저 ``E(q,R0)``와 ``E(q0,R)``의 1차원 slice를 만든다. 다른 좌표가
    grid point와 정확히 일치하지 않으면 선형보간하고, 각 slice에서 중심에
    가장 가까운 세 점을

        E(s) = c + g*(s-s0) + k*(s-s0)^2/2

    로 fit한다. 따라서 ``k_q=d2E(q,R0)/dq2``와
    ``k_R=d2E(q0,R)/dR2``만 계산하며 혼합미분은 만들지 않는다.
    """
    if not model.q[0] <= q0 <= model.q[-1]:
        raise ValueError(f"q0={q0}가 q grid 밖에 있습니다.")
    if not model.R[0] <= R0 <= model.R[-1]:
        raise ValueError(f"R0={R0}가 R grid 밖에 있습니다.")

    # E(q,R0): 각 q에서 R 방향으로 R0까지 보간한다. shape (nq,)
    energy_along_q = np.array([
        np.interp(R0, model.R, surface[iq, :])
        for iq in range(len(model.q))
    ])
    # E(q0,R): 각 R에서 q 방향으로 q0까지 보간한다. shape (nR,)
    energy_along_R = np.array([
        np.interp(q0, model.q, surface[:, iR])
        for iR in range(len(model.R))
    ])

    def fit_one_axis(grid, energy, center):
        indices = np.sort(np.argsort(np.abs(grid-center))[:3])
        displacement = grid[indices]-center
        design = np.column_stack((
            np.ones(3), displacement, 0.5*displacement**2,
        ))
        coefficients = np.linalg.solve(design, energy[indices])
        return float(coefficients[1]), float(coefficients[2])

    gradient_q, k_q = fit_one_axis(model.q, energy_along_q, q0)
    gradient_R, k_R = fit_one_axis(model.R, energy_along_R, R0)
    return dict(
        gradient_q=gradient_q,
        gradient_R=gradient_R,
        k_q=k_q,
        k_R=k_R,
    )


def harmonic_density_sigma(mass: float, force_constant: float):
    """1D harmonic ground-state 확률밀도의 표준편차."""
    if mass <= 0.0 or force_constant <= 0.0:
        raise ValueError("mass와 force constant는 양수여야 합니다.")
    return (1.0/(4.0*mass*force_constant))**0.25


def initial_factors(model: Model, args):
    """Local BO 전자상태와 두 독립 harmonic Gaussian으로 초기화."""
    excitation = int(args.electron_excitation)
    if excitation < 0:
        raise ValueError("--electron-excitation은 0 이상이어야 합니다.")

    # ------------------------------------------------------------------
    # 1. 모든 (q,R)에서 local H_BO를 풀어 전자 초기상태/PES를 얻는다.
    # ------------------------------------------------------------------
    energies, electronic_states = local_electronic_basis(model, excitation+1)
    phi = electronic_states[excitation]                             # (nx,nq,nR)
    curvature = independent_surface_curvatures(
        energies[excitation], model, args.q0, args.R0
    )

    # 0이면 BO surface의 국소 이차 미분을 자동 사용하고, 양수 option은
    # 사용자가 harmonic reference force constant를 직접 지정한 경우다.
    if args.proton_force_constant < 0.0 or args.heavy_force_constant < 0.0:
        raise ValueError("force constant option은 0(자동) 또는 양수여야 합니다.")
    kq = (
        curvature["k_q"]
        if args.proton_force_constant == 0.0
        else args.proton_force_constant
    )
    kR = (
        curvature["k_R"]
        if args.heavy_force_constant == 0.0
        else args.heavy_force_constant
    )
    if kq <= 0.0 or kR <= 0.0:
        raise ValueError(
            "독립적인 초기 harmonic 폭에 사용할 곡률이 양수가 아닙니다: "
            f"k_q={kq:.6g}, k_R={kR:.6g}. 양의 곡률 위치를 선택하거나 "
            "--proton-force-constant와 --heavy-force-constant를 지정하세요."
        )
    proton_sigma = harmonic_density_sigma(model.proton_mass, kq)
    heavy_sigma = harmonic_density_sigma(model.heavy_mass, kR)

    # 계산된 초기 parameter도 NPZ의 args metadata에 함께 남긴다.
    args.electron_initial_state = "local-eigenstate"
    args.proton_sigma = proton_sigma
    args.heavy_sigma = heavy_sigma
    args.initial_proton_force_constant = kq
    args.initial_heavy_force_constant = kR
    args.initial_gradient_q = curvature["gradient_q"]
    args.initial_gradient_R = curvature["gradient_R"]

    for name, sigma, spacing in (
        ("proton", proton_sigma, model.dq),
        ("heavy", heavy_sigma, model.dR),
    ):
        if sigma < 1.5*spacing:
            warnings.warn(
                f"{name} sigma={sigma:.4g}가 grid spacing={spacing:.4g}에 "
                "비해 좁습니다. 해당 grid 점 수를 늘려 convergence를 확인하세요.",
                RuntimeWarning,
            )

    # ------------------------------------------------------------------
    # 2. Heavy marginal Gaussian: 중심 R0, 폭 sigma_R, shape (nR,)
    # ------------------------------------------------------------------
    chi = normalized_gaussian(
        model.R, model.dR, args.R0, heavy_sigma, args.heavy_momentum
    )                                                               # (nR,)

    # ------------------------------------------------------------------
    # 3. Proton Gaussian은 모든 R에서 동일한 q0와 sigma_q를 갖는다.
    # ------------------------------------------------------------------
    proton_line = normalized_gaussian(
        model.q, model.dq, args.q0, proton_sigma, args.proton_momentum
    )                                                               # (nq,)
    lam = np.repeat(proton_line[:, None], len(model.R), axis=1)     # (nq,nR)
    return phi.astype(complex), lam.astype(complex), chi.astype(complex)


def derivative(values, spacing, axis, order=1):
    """주기 격자의 독립적인 4차 정확도 5점 1·2차 유한차분.

    모든 점에서 같은 central stencil을 쓰고 인덱스를 주기적으로 감싼다.
    따라서 균일 격자의 표준 내적에서 ``D1^H=-D1`` 및 ``D2^H=D2``가
    성립한다. 2차 미분은 ``D1(D1(f))``가 아니라 독립 stencil을 사용해
    Nyquist checkerboard에도 유한한 kinetic penalty를 준다.
    """
    if values.shape[axis] < 5:
        raise ValueError("5점 미분에는 해당 축에 최소 5개 격자점이 필요합니다.")
    if order == 1:
        return (
            np.roll(values, 2, axis=axis)
            -8.0*np.roll(values, 1, axis=axis)
            +8.0*np.roll(values, -1, axis=axis)
            -np.roll(values, -2, axis=axis)
        )/(12.0*spacing)
    if order == 2:
        return (
            -np.roll(values, 2, axis=axis)
            +16.0*np.roll(values, 1, axis=axis)
            -30.0*values
            +16.0*np.roll(values, -1, axis=axis)
            -np.roll(values, -2, axis=axis)
        )/(12.0*spacing**2)
    raise ValueError("order는 1 또는 2여야 합니다.")


def momentum(values, spacing, axis):
    """운동량 연산자 ``-i d/dcoordinate``를 적용한다."""
    return -1j*derivative(values, spacing, axis=axis)


def regularized_ratio(numerator, denominator, relative_floor):
    """``numerator/denominator``를 node와 low-density tail에서 안정화한다.

    mathematically undefined인 denominator=0 영역이 direct EF 불안정성의
    주원인이다. ``z/f = z*f*/|f|^2``를 사용하고 분모에 작은 floor를 둔다.
    """
    density = np.abs(denominator)**2
    floor = relative_floor*max(float(np.max(density)), 1.0e-300)
    return numerator*np.conj(denominator)/(density+floor)


def logarithmic_components(factor, spacing, axis, numerical_floor=1.0e-14):
    """``(-i d factor)/factor = phase_gradient-i*log_amplitude_gradient``.

    ``numerical_floor``는 overflow/zero division만 막는 작은 수치 floor다.
    물리적 support를 정하는 mask threshold와 의도적으로 분리한다.
    """
    if numerical_floor <= 0.0:
        raise ValueError("numerical_floor는 양수여야 합니다.")
    density = np.abs(factor)**2
    peak = np.max(density, axis=axis, keepdims=True)
    safe = density+numerical_floor*np.maximum(peak, 1.0e-300)
    ratio = momentum(factor, spacing, axis)*np.conj(factor)/safe
    return ratio.real, -ratio.imag


def occupied_support_mask(density, relative_threshold):
    """상대 density로 정의한 부드러운 ``rho/(rho+eta*rho_max)`` mask."""
    if relative_threshold < 0.0:
        raise ValueError("mask threshold는 0 이상이어야 합니다.")
    if relative_threshold == 0.0:
        return np.ones_like(density, dtype=float)
    peak = max(float(np.max(density)), 1.0e-300)
    return density/(density+relative_threshold*peak)


def suppressed_probability(density, mask, *spacings):
    """Support mask가 감쇠한 정규화 probability mass."""
    volume = float(np.prod(spacings))
    total = float(np.sum(density))*volume
    if total <= 0.0:
        return 0.0
    return float(np.sum(density*(1.0-mask)))*volume/total


def mask_threshold_for_probability_budget(
    density, budget, *, lower=0.0, upper=1.0, iterations=64,
):
    """Return the largest smooth-mask threshold within a mass budget.

    This is a diagnostic helper: it does not alter propagation.  For
    ``M=rho/(rho+eta*rho_max)``, the suppressed probability is monotone in
    ``eta``, so a scalar bisection gives the threshold corresponding to a
    requested probability budget.  The normalization and uniform-grid volume
    cancel in the ratio.
    """
    if not 0.0 <= budget < 1.0:
        raise ValueError("probability budget은 0 이상 1 미만이어야 합니다.")
    density = np.asarray(density, dtype=float)
    total = float(np.sum(density))
    peak = float(np.max(density)) if density.size else 0.0
    if budget == 0.0 or total <= 0.0 or peak <= 0.0:
        return 0.0

    def removed(eta):
        return float(np.sum(
            density*(eta*peak)/(density+eta*peak)
        ))/total

    lo, hi = float(lower), float(upper)
    while removed(hi) < budget and hi < 1.0e12:
        hi *= 10.0
    for _ in range(iterations):
        mid = 0.5*(lo+hi)
        if removed(mid) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def remove_local_norm_generator(
    factor, action, spacing, axis=0, norm_floor=1.0e-14,
):
    """Remove the local anti-Hermitian component parallel to ``factor``.

    RK4 intermediate factors do not satisfy the PNC exactly, so the imaginary
    expectation value is divided by the current local norm rather than
    assuming it equals one. The returned raw/corrected rates are
    ``2 Im <factor|action>`` before and after correction.
    """
    norm2 = np.sum(np.abs(factor)**2, axis=axis)*spacing
    expectation = np.sum(np.conj(factor)*action, axis=axis)*spacing
    safe_norm2 = np.maximum(norm2, norm_floor)
    gamma = expectation.imag/safe_norm2
    corrected = action-1j*np.expand_dims(gamma, axis=axis)*factor
    corrected_expectation = (
        np.sum(np.conj(factor)*corrected, axis=axis)*spacing
    )
    return (
        corrected,
        gamma,
        2.0*expectation.imag,
        2.0*corrected_expectation.imag,
    )


def electronic_kinetic_energies(model: Model):
    """Dirichlet 중앙차분 전자 kinetic의 DST-I 고유값 ``(nx,)``.

    왼쪽/오른쪽 경계에서 wavefunction이 0인 interior grid의 mode는
    ``sin[n*pi*(x-x_L)/L]``이고, 2차 중앙차분 kinetic 고유값은

        T_n = [1-cos(n*pi/(nx+1))]/dx^2,  n=1,...,nx

    이다.
    """
    modes = np.arange(1, len(model.x)+1, dtype=float)
    return (1.0-np.cos(np.pi*modes/(len(model.x)+1)))/model.dx**2


def electronic_kinetic_step(values, tau, model: Model):
    """전자축에 ``exp(-i*tau*T_x)``를 hard-wall sine basis로 적용."""
    transformed = dst(
        values, type=1, axis=0, norm="ortho", workers=model.fft_workers
    )
    phase = np.exp(-1j*tau*electronic_kinetic_energies(model))
    phase = phase.reshape((-1,)+(1,)*(values.ndim-1))
    return idst(
        transformed*phase, type=1, axis=0, norm="ortho",
        workers=model.fft_workers,
    )


def apply_electronic_hamiltonian(phi, model: Model):
    """Dirichlet 중앙차분 ``[-d_x^2/2+V(x,q,R)] Phi``를 적용."""
    # 배열 밖의 두 값은 hard-wall Dirichlet 조건에 따라 0이다.
    kinetic = phi/model.dx**2
    kinetic[1:] -= 0.5*phi[:-1]/model.dx**2
    kinetic[:-1] -= 0.5*phi[1:]/model.dx**2
    return kinetic + model.potential*phi


def electronic_split_step(phi, tau, model: Model):
    """Hard-wall 전자 ``T_x/2 -> V -> T_x/2`` split step."""
    phi = electronic_kinetic_step(phi, 0.5*tau, model)
    phi *= np.exp(-1j*tau*model.potential)
    return electronic_kinetic_step(phi, 0.5*tau, model)


def pnc_project(phi, lam, chi, model: Model):
    """두 partial normalization condition을 복원하며 Psi는 보존한다.

    Phi에서 빠진 local norm은 Lambda로, Lambda에서 빠진 local norm은
    chi로 옮긴다. 따라서 ``Phi*Lambda*chi``는 점별로 바뀌지 않는다.
    """
    phi_norm = np.sqrt(np.sum(np.abs(phi)**2, axis=0)*model.dx)      # (nq,nR)
    phi_error = float(np.max(np.abs(phi_norm**2-1.0)))
    safe_phi = np.where(phi_norm > 1.0e-14, phi_norm, 1.0)
    phi = phi/safe_phi[None, :, :]
    lam = lam*safe_phi

    lam_norm = np.sqrt(np.sum(np.abs(lam)**2, axis=0)*model.dq)     # (nR,)
    lam_error = float(np.max(np.abs(lam_norm**2-1.0)))
    safe_lam = np.where(lam_norm > 1.0e-14, lam_norm, 1.0)
    lam = lam/safe_lam[None, :]
    chi = chi*safe_lam
    return phi, lam, chi, max(phi_error, lam_error)


def reconstruct_psi(phi, lam, chi):
    """세 factor로 full molecular wavefunction ``Psi(nx,nq,nR)`` 재구성."""
    return phi*lam[None, :, :]*chi[None, None, :]


def project_discrete_product_residual(
    phi, lam, chi, dphi, dlam, dchi, model: Model,
    support_floor_phi=1.0e-10, support_floor_lam=1.0e-10,
    unmasked_rhs=None, residual_support_weight=None,
):
    """Factor RHS를 Hermitian discrete nuclear kinetic의 tangent로 맞춘다.

    연속 EF 유도에 쓰이는 Leibniz product rule은 finite difference에서
    정확하지 않다. 따라서 factor RHS로 재구성한 ``dPsi``와 periodic 5점
    ``D2``가 만드는 target nuclear ``dPsi``의 residual을 계산하고,
    ``Phi -> Lambda -> chi`` 순서의 orthogonal nested tangent로 분해해
    되돌린다. 점유 support에서는 product derivative가 target과 일치하고,
    나눗셈이 위험한 tail에는 factor-orthogonal residual만 남으므로 full
    norm 생성률은 만들지 않는다.
    """
    if support_floor_phi < 0.0 or support_floor_lam < 0.0:
        raise ValueError("product projection support floor는 0 이상이어야 합니다.")
    psi = reconstruct_psi(phi, lam, chi)
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

    def product_from_factor_rhs(rhs):
        rp, rl, rc = rhs
        return (
            rp*lam[None, :, :]*chi[None, None, :]
            +phi*rl[None, :, :]*chi[None, None, :]
            +phi*lam[None, :, :]*rc[None, None, :]
        )

    if unmasked_rhs is None:
        residual_without_mask = None
        residual_due_to_mask = None
    else:
        unmasked_product_rhs = product_from_factor_rhs(unmasked_rhs)
        residual_without_mask = target_rhs-unmasked_product_rhs
        residual_due_to_mask = unmasked_product_rhs-product_rhs

    phi_norm2 = np.sum(np.abs(phi)**2, axis=0)*model.dx
    safe_phi_norm2 = np.maximum(phi_norm2, 1.0e-14)
    delta_xi = (
        np.sum(np.conj(phi)*residual, axis=0)*model.dx/safe_phi_norm2
    )
    perpendicular_phi = residual-phi*delta_xi[None, :, :]

    xi = lam*chi[None, :]
    xi_density = np.abs(xi)**2
    xi_peak = max(float(np.max(xi_density)), 1.0e-300)
    # conj(xi)/(|xi|^2+eta*peak)는 (1/xi)에 smooth support weight를
    # 곱한 것과 같다. Hard cutoff가 만드는 새 경계/spike를 피한다.
    inverse_xi = np.conj(xi)/(
        xi_density+support_floor_phi*xi_peak
    )
    delta_phi = perpendicular_phi*inverse_xi[None, :, :]

    lam_norm2 = np.sum(np.abs(lam)**2, axis=0)*model.dq
    safe_lam_norm2 = np.maximum(lam_norm2, 1.0e-14)
    delta_chi = (
        np.sum(np.conj(lam)*delta_xi, axis=0)*model.dq/safe_lam_norm2
    )
    perpendicular_lam = delta_xi-lam*delta_chi[None, :]
    chi_density = np.abs(chi)**2
    chi_peak = max(float(np.max(chi_density)), 1.0e-300)
    inverse_chi = np.conj(chi)/(
        chi_density+support_floor_lam*chi_peak
    )
    delta_lam = perpendicular_lam*inverse_chi[None, :]

    dphi = dphi+delta_phi
    dlam = dlam+delta_lam
    dchi = dchi+delta_chi
    corrected_product_rhs = (
        dphi*lam[None, :, :]*chi[None, None, :]
        +phi*dlam[None, :, :]*chi[None, None, :]
        +phi*lam[None, :, :]*dchi[None, None, :]
    )
    effective_residual = target_rhs-corrected_product_rhs
    volume = model.dx*model.dq*model.dR
    def l2(values):
        return np.sqrt(np.sum(np.abs(values)**2)*volume)

    if residual_without_mask is None:
        target_l2 = 0.0
        residual_without_mask_l2 = 0.0
        residual_due_to_mask_l2 = 0.0
        relative_floor = 1.0
        residual_alignment = 0.0
        support_without_mask_l2 = 0.0
        support_due_to_mask_l2 = 0.0
        relative_support_without_mask = 0.0
        relative_support_due_to_mask = 0.0
    else:
        target_l2 = l2(target_rhs)
        residual_without_mask_l2 = l2(residual_without_mask)
        residual_due_to_mask_l2 = l2(residual_due_to_mask)
        relative_floor = max(target_l2, 1.0e-300)
        alignment_denominator = max(
            residual_without_mask_l2*residual_due_to_mask_l2, 1.0e-300
        )
        residual_alignment = np.real(np.sum(
            np.conj(residual_without_mask)*residual_due_to_mask
        ))*volume/alignment_denominator
        if residual_support_weight is None:
            residual_support_weight = np.ones(phi.shape[1:], dtype=float)
        weight = np.asarray(residual_support_weight, dtype=float)[None, :, :]
        support_target_l2 = np.sqrt(
            np.sum(weight*np.abs(target_rhs)**2)*volume
        )
        support_without_mask_l2 = np.sqrt(
            np.sum(weight*np.abs(residual_without_mask)**2)*volume
        )
        support_due_to_mask_l2 = np.sqrt(
            np.sum(weight*np.abs(residual_due_to_mask)**2)*volume
        )
        support_floor = max(support_target_l2, 1.0e-300)
        relative_support_without_mask = support_without_mask_l2/support_floor
        relative_support_due_to_mask = support_due_to_mask_l2/support_floor
    diagnostics = dict(
        product_residual_l2=np.sqrt(np.sum(np.abs(residual)**2)*volume),
        effective_product_residual_l2=np.sqrt(
            np.sum(np.abs(effective_residual)**2)*volume
        ),
        full_norm_rate_before_product_projection=(
            2.0*np.real(np.sum(np.conj(psi)*product_rhs))*volume
        ),
        full_norm_rate_after_product_projection=(
            2.0*np.real(np.sum(np.conj(psi)*corrected_product_rhs))*volume
        ),
        product_correction_phi=np.max(np.abs(delta_phi)),
        product_correction_lam=np.max(np.abs(delta_lam)),
        product_correction_chi=np.max(np.abs(delta_chi)),
        product_residual_without_mask_l2=residual_without_mask_l2,
        product_residual_due_to_mask_l2=residual_due_to_mask_l2,
        relative_product_residual_without_mask=(
            residual_without_mask_l2/relative_floor
        ),
        relative_product_residual_due_to_mask=(
            residual_due_to_mask_l2/relative_floor
        ),
        product_mask_nonmask_alignment=residual_alignment,
        product_mask_nonmask_alignment_positive=max(
            residual_alignment, 0.0
        ),
        product_mask_nonmask_alignment_negative_magnitude=max(
            -residual_alignment, 0.0
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


def _minus_covariant(field, vector, spacing, axis):
    """``(-i d - vector) field``."""
    return momentum(field, spacing, axis)-vector*field


def _plus_covariant(field, vector, spacing, axis):
    """``(-i d + vector) field``."""
    return momentum(field, spacing, axis)+vector*field


def covariant_square(field, vector, spacing, axis, sign):
    """독립적인 5점 ``D2``로 ``(-i d + sign*vector)^2 field`` 계산.

    ``sign=+1``은 ``p+A``, ``sign=-1``은 ``p-A``이다. 1차 covariant
    derivative를 연속 적용하지 않으므로 vector=0일 때 표준 5점
    ``-d^2``가 되고 one-cell checkerboard도 큰 kinetic 값을 갖는다.

    유한차분에는 정확한 Leibniz product rule이 없으므로 연속 전개식의
    ``(dA)f+2A(df)``를 직접 쓰지 않는다. 대신 ``pA+Ap``의 대칭
    anticommutator 형태를 사용한다. 실수 vector와 periodic central
    ``D1,D2``에 대해 이산 연산자도 정확히 Hermitian이다.
    """
    second = derivative(field, spacing, axis=axis, order=2)
    p_field = momentum(field, spacing, axis=axis)
    p_vector_field = momentum(vector*field, spacing, axis=axis)
    return (
        -second
        +sign*(p_vector_field+vector*p_field)
        +vector**2*field
    )


def geometric_fields(phi, lam, model: Model):
    """두 단계의 vector potential ``a(q,R), b(q,R), alpha(R)`` 계산."""
    p_q_phi = momentum(phi, model.dq, axis=1)
    p_R_phi = momentum(phi, model.dR, axis=2)
    a = np.sum(np.conj(phi)*p_q_phi, axis=0)*model.dx               # (nq,nR)
    b = np.sum(np.conj(phi)*p_R_phi, axis=0)*model.dx               # (nq,nR)
    # PNC가 정확하면 허수부는 0이다. 작은 허수부는 차분 오차이므로 제거한다.
    a = a.real
    b = b.real
    p_R_lam = momentum(lam, model.dR, axis=1)
    alpha = np.sum(
        np.conj(lam)*(p_R_lam+b*lam), axis=0
    )*model.dq                                                       # (nR,)
    return a, b, alpha.real


def proton_base_operator(
    lam, a, b, alpha, chi_phase_R, chi_logamp_R, mask_lam,
    model: Model, return_unmasked=False,
):
    """``H_pr`` 중 epsilon_1을 제외한 부분을 Lambda에 적용한다.

    반환값은 proton kinetic과 두 번째 factorization의 coupling을 합한
    ``base_lambda(nq,nR)``이다.
    """
    # [(-i d_q+a)^2/(2m_p)] Lambda
    proton_kinetic = covariant_square(
        lam, a, model.dq, axis=0, sign=+1
    )*(0.5/model.proton_mass)

    # U_{p,n}^coup Lambda: R 방향에서 b-alpha가 vector field 역할을 한다.
    vector_R = b-alpha[None, :]
    dplus_R = _plus_covariant(lam, vector_R, model.dR, axis=1)
    dplus_R2 = covariant_square(
        lam, vector_R, model.dR, axis=1, sign=+1
    )
    # K_R^(chi)=d_R S+alpha는 유지하고 node에서 singular한
    # d_R ln|chi|만 rho_R support로 감쇠한다.
    chi_coefficient = (
        chi_phase_R+alpha-1j*mask_lam*chi_logamp_R
    )                                                               # (nR,)
    proton_nuclear_coupling = (
        0.5*dplus_R2+chi_coefficient[None, :]*dplus_R
    )/model.heavy_mass
    masked = proton_kinetic+proton_nuclear_coupling
    if not return_unmasked:
        return masked
    # 같은 derivative를 재사용하여 support mask만 끈 비교 action을 만든다.
    # numerical logarithmic-derivative floor는 그대로 유지한다.
    chi_coefficient_unmasked = (
        chi_phase_R+alpha-1j*chi_logamp_R
    )
    unmasked = proton_kinetic+(
        0.5*dplus_R2+chi_coefficient_unmasked[None, :]*dplus_R
    )/model.heavy_mass
    return masked, unmasked


def instantaneous_functionals(
    phi, lam, chi, model: Model, floor=1.0e-14,
    mask_threshold_phi=1.0e-10, mask_threshold_lam=1.0e-10,
    include_unmasked=False,
):
    """현재 세 factor에서 모든 EF scalar/vector potential을 계산한다.

    두 scalar gauge는 parallel-transport gauge를 사용한다:

        epsilon_1 = <Phi|H_el|Phi>_x
        epsilon_2 = <Lambda|H_pr|Lambda>_q

    반환 dictionary의 shape은 주석에 적힌 것과 같다.
    """
    a, b, alpha = geometric_fields(phi, lam, model)

    # 각 nested conditional factor가 실제로 정의되는 physical support.
    rho_R = np.abs(chi)**2
    rho_qR = np.abs(lam)**2*rho_R[None, :]
    mask_phi = occupied_support_mask(rho_qR, mask_threshold_phi)
    mask_lam = occupied_support_mask(rho_R, mask_threshold_lam)

    lam_phase_q, lam_logamp_q = logarithmic_components(
        lam, model.dq, axis=0, numerical_floor=floor
    )
    lam_phase_R, lam_logamp_R = logarithmic_components(
        lam, model.dR, axis=1, numerical_floor=floor
    )
    chi_phase_R, chi_logamp_R = logarithmic_components(
        chi, model.dR, axis=0, numerical_floor=floor
    )

    # ----- 첫 번째 factorization: 전자 coupling U_e,pn -----
    # 양성자 좌표 q 방향 항
    dminus_q = _minus_covariant(phi, a[None, :, :], model.dq, axis=1)
    dminus_q2 = covariant_square(
        phi, a[None, :, :], model.dq, axis=1, sign=-1
    )
    # Gauge-invariant K_q=d_q T+a는 보존하고 singular amplitude gradient만
    # joint nuclear density support에서 감쇠한다.
    coefficient_q = (
        lam_phase_q+a-1j*mask_phi*lam_logamp_q
    )                                                               # (nq,nR)
    u_q_phi = (
        0.5*dminus_q2+coefficient_q[None, :, :]*dminus_q
    )/model.proton_mass

    # 무거운 핵 좌표 R 방향 항. chi와 Lambda의 R 변화가 모두 들어간다.
    dminus_R = _minus_covariant(phi, b[None, :, :], model.dR, axis=2)
    dminus_R2 = covariant_square(
        phi, b[None, :, :], model.dR, axis=2, sign=-1
    )
    coefficient_R = (
        chi_phase_R[None, :]+lam_phase_R+b
        -1j*mask_phi*(chi_logamp_R[None, :]+lam_logamp_R)
    )                                                               # (nq,nR)
    u_R_phi = (
        0.5*dminus_R2
        +coefficient_R[None, :, :]*dminus_R
    )/model.heavy_mass
    u_phi = u_q_phi+u_R_phi                                         # (nx,nq,nR)

    # 첫 correction은 dPhi에 -gamma_phi*Phi를 만들며, 아래 Lambda action에
    # +i*gamma_phi*Lambda를 더해 dLambda에 반대 변화를 만든다. 따라서
    # Phi*Lambda*chi의 순간 변화는 점별로 정확히 상쇄된다.
    u_phi, gamma_phi, raw_rate_phi, corrected_rate_phi = (
        remove_local_norm_generator(phi, u_phi, model.dx, axis=0)
    )

    hbo_phi = apply_electronic_hamiltonian(phi, model)
    epsilon_1 = (
        np.sum(np.conj(phi)*(hbo_phi+u_phi), axis=0)*model.dx
    ).real                                                          # (nq,nR)

    # ----- 두 번째 factorization: 양성자와 바깥 무거운 핵 -----
    base_result = proton_base_operator(
        lam, a, b, alpha, chi_phase_R, chi_logamp_R, mask_lam, model,
        return_unmasked=include_unmasked,
    )                                                               # (nq,nR)
    if include_unmasked:
        base_lam_raw, base_lam_unmasked_raw = base_result
    else:
        base_lam_raw = base_result
    hpr_lam_raw = (
        base_lam_raw+epsilon_1*lam+1j*gamma_phi*lam
    )
    hpr_lam, gamma_lam, raw_rate_lam, corrected_rate_lam = (
        remove_local_norm_generator(lam, hpr_lam_raw, model.dq, axis=0)
    )
    # gamma_lam correction으로 dLambda에 -gamma_lam*Lambda가 생긴 만큼
    # outer dchi에는 +gamma_lam*chi를 더한다(coupled_rhs 참조).
    base_lam = hpr_lam-epsilon_1*lam
    epsilon_2 = (
        np.sum(np.conj(lam)*hpr_lam, axis=0)*model.dq
    ).real                                                          # (nR,)

    result = dict(
        a=a, b=b, alpha=alpha, epsilon_1=epsilon_1,
        epsilon_2=epsilon_2, u_phi=u_phi, base_lam=base_lam,
        hbo_phi=hbo_phi, hpr_lam=hpr_lam,
        gamma_phi=gamma_phi, gamma_lam=gamma_lam,
        support_gamma_phi=mask_phi*gamma_phi,
        support_gamma_lam=mask_lam*gamma_lam,
        raw_rate_phi=raw_rate_phi, raw_rate_lam=raw_rate_lam,
        corrected_rate_phi=corrected_rate_phi,
        corrected_rate_lam=corrected_rate_lam,
        mask_phi=mask_phi, mask_lam=mask_lam,
        suppressed_probability_phi=np.asarray(suppressed_probability(
            rho_qR, mask_phi, model.dq, model.dR
        )),
        suppressed_probability_lam=np.asarray(suppressed_probability(
            rho_R, mask_lam, model.dR
        )),
        raw_logamp_phi=np.maximum(
            np.abs(lam_logamp_q),
            np.abs(lam_logamp_R)+np.abs(chi_logamp_R)[None, :],
        ),
        effective_logamp_phi=np.maximum(
            np.abs(mask_phi*lam_logamp_q),
            np.abs(mask_phi*(lam_logamp_R+chi_logamp_R[None, :])),
        ),
        raw_logamp_lam=np.abs(chi_logamp_R),
        effective_logamp_lam=np.abs(mask_lam*chi_logamp_R),
    )
    if include_unmasked:
        coefficient_q_unmasked = (
            lam_phase_q+a-1j*lam_logamp_q
        )
        coefficient_R_unmasked = (
            chi_phase_R[None, :]+lam_phase_R+b
            -1j*(chi_logamp_R[None, :]+lam_logamp_R)
        )
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
            phi, u_phi_unmasked_raw, model.dx, axis=0
        )
        epsilon_1_unmasked = (
            np.sum(
                np.conj(phi)*(hbo_phi+u_phi_unmasked), axis=0
            )*model.dx
        ).real
        hpr_lam_unmasked_raw = (
            base_lam_unmasked_raw+epsilon_1_unmasked*lam
            +1j*gamma_phi_unmasked*lam
        )
        (
            hpr_lam_unmasked, gamma_lam_unmasked, _, _
        ) = remove_local_norm_generator(
            lam, hpr_lam_unmasked_raw, model.dq, axis=0
        )
        epsilon_2_unmasked = (
            np.sum(
                np.conj(lam)*hpr_lam_unmasked, axis=0
            )*model.dq
        ).real
        result.update(
            u_phi_unmasked=u_phi_unmasked,
            hpr_lam_unmasked=hpr_lam_unmasked,
            epsilon_1_unmasked=epsilon_1_unmasked,
            epsilon_2_unmasked=epsilon_2_unmasked,
            gamma_lam_unmasked=gamma_lam_unmasked,
        )
    return result


def nested_factorize(psi, model: Model, floor=1.0e-14):
    """full ``Psi``를 양의 marginal gauge에서 chi, Lambda, Phi로 분해한다.

    이 함수는 reference TDSE를 사후 분석할 때 사용한다. 두 PNC가 정의되지
    않는 density node에서는 작은 floor로 나눗셈만 안정화한다.
    """
    rho_R = np.sum(np.abs(psi)**2, axis=(0, 1))*model.dx*model.dq    # (nR,)
    chi = np.sqrt(np.maximum(rho_R, 0.0)).astype(complex)           # (nR,)
    safe_chi = np.where(np.abs(chi) > floor, chi, 1.0)
    gamma = psi/safe_chi[None, None, :]                             # (nx,nq,nR)

    rho_q_given_R = np.sum(np.abs(gamma)**2, axis=0)*model.dx       # (nq,nR)
    lam = np.sqrt(np.maximum(rho_q_given_R, 0.0)).astype(complex)
    safe_lam = np.where(np.abs(lam) > floor, lam, 1.0)
    phi = gamma/safe_lam[None, :, :]
    return pnc_project(phi, lam, chi, model)[:3]


def reduced_densities(phi, lam, chi, model: Model):
    """3D wavefunction을 그림으로 볼 수 있는 1D/2D density로 줄인다."""
    phi2 = np.abs(phi)**2
    lam2 = np.abs(lam)**2
    chi2 = np.abs(chi)**2
    # R이 주어졌을 때 양성자 확률로 평균한 조건부 전자 밀도 (x,R)
    electron_given_R = np.sum(
        phi2*lam2[None, :, :], axis=1
    )*model.dq
    # full Psi에서 q를 적분한 전자-무거운 핵 joint density (x,R)
    electron_heavy = electron_given_R*chi2[None, :]
    # 전자를 적분한 양성자-무거운 핵 joint density (q,R)
    proton_heavy = lam2*chi2[None, :]
    return dict(
        electron_given_R=electron_given_R,
        electron_heavy=electron_heavy,
        proton_heavy=proton_heavy,
        proton_given_R=lam2,
        heavy=chi2,
    )


def add_model_arguments(parser):
    """direct/reference 명령행에서 공유하는 model option을 등록한다."""
    grid = parser.add_argument_group("실공간 격자")
    grid.add_argument("--nx", type=int, default=174, help="전자 interior 격자점 수")
    grid.add_argument("--nq", type=int, default=87, help="양성자 격자점 수")
    grid.add_argument("--nR", type=int, default=30, help="무거운 핵 격자점 수")
    grid.add_argument(
        "--x-max", type=float, default=8.0,
        help="전자 hard-wall 오른쪽 끝; 왼쪽 끝은 --left-position",
    )
    grid.add_argument("--q-min", type=float, default=-3.36)
    grid.add_argument("--q-max", type=float, default=3.6)
    grid.add_argument("--R-min", type=float, default=3.0)
    grid.add_argument("--R-max", type=float, default=5.4)
    grid.add_argument(
        "--fft-workers", type=int, default=-1,
        help="전자 DST worker 수; -1은 사용 가능한 CPU core를 모두 사용",
    )

    particle = parser.add_argument_group("입자와 초기상태")
    particle.add_argument("--proton-mass", type=float, default=1836.0)
    particle.add_argument("--heavy-mass", type=float, default=12000.0)
    particle.add_argument("--q0", type=float, default=2.0)
    particle.add_argument("--R0", type=float, default=4.2)
    particle.add_argument("--proton-momentum", type=float, default=0.0)
    particle.add_argument("--heavy-momentum", type=float, default=0.0)
    particle.add_argument(
        "--proton-force-constant", type=float, default=0.0,
        help="0이면 local BO surface의 d2E/dq2, 양수면 해당 값을 직접 사용",
    )
    particle.add_argument(
        "--heavy-force-constant", type=float, default=0.0,
        help="0이면 local BO surface의 d2E/dR2, 양수면 해당 값을 직접 사용",
    )
    particle.add_argument(
        "--electron-excitation", type=int, default=0,
        help="초기 local H_BO 전자상태의 0-based index",
    )

    potential = parser.add_argument_group("soft-Coulomb potential")
    potential.add_argument(
        "--left-position", type=float, default=-6.0,
        help="전자 hard wall 및 왼쪽 고정 중심의 위치",
    )
    potential.add_argument(
        "--left-charge", type=float, default=0.0,
        help="왼쪽 고정 중심의 전하 Z_L; 기본값 0이면 Coulomb 항은 꺼짐",
    )
    potential.add_argument("--heavy-charge", type=float, default=1.0)
    potential.add_argument("--soft-e-left", type=float, default=1.0)
    potential.add_argument("--soft-e-proton", type=float, default=0.8)
    potential.add_argument("--soft-e-heavy", type=float, default=1.0)
    potential.add_argument("--soft-p-left", type=float, default=0.8)
    potential.add_argument("--soft-p-heavy", type=float, default=0.8)
    potential.add_argument("--soft-left-heavy", type=float, default=0.8)
    return parser
