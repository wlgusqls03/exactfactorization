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

    # q와 R은 density가 경계에서 충분히 작다는 전제 아래 주기 격자를 쓴다.
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


def local_surface_curvature(surface, model: Model, q0: float, R0: float):
    """BO surface의 ``(q0,R0)`` 주변 3x3 점을 이차식으로 fit한다.

    ``E = c + gq*dq + gR*dR + kq*dq^2/2 + kqR*dq*dR
          + kR*dR^2/2``

    의 계수를 least squares로 구한다. 반환값은 두 gradient, 두 대각 force
    constant와 혼합 curvature다. 세 curvature 모두 결합 nuclear harmonic
    state를 만드는 데 사용한다.
    """
    if not model.q[0] <= q0 <= model.q[-1]:
        raise ValueError(f"q0={q0}가 q grid 밖에 있습니다.")
    if not model.R[0] <= R0 <= model.R[-1]:
        raise ValueError(f"R0={R0}가 R grid 밖에 있습니다.")
    iq = np.sort(np.argsort(np.abs(model.q-q0))[:3])
    iR = np.sort(np.argsort(np.abs(model.R-R0))[:3])
    design, values = [], []
    for jq in iq:
        for jR in iR:
            dq = model.q[jq]-q0
            dR = model.R[jR]-R0
            design.append([1.0, dq, dR, 0.5*dq**2, dq*dR, 0.5*dR**2])
            values.append(surface[jq, jR])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return dict(
        gradient_q=float(coefficients[1]),
        gradient_R=float(coefficients[2]),
        k_q=float(coefficients[3]),
        k_qR=float(coefficients[4]),
        k_R=float(coefficients[5]),
    )


def coupled_harmonic_state(model: Model, args, k_q, k_qR, k_R):
    """Full 2x2 Hessian으로 상관된 nuclear harmonic state를 만든다.

    ``y=(q-q0,R-R0)``와 ``M=diag(m_p,M_H)``에 대해

        D = M^(-1/2) K M^(-1/2),   Omega = sqrt(D)
        Xi(y) = N exp[-y^T M^(1/2) Omega M^(1/2) y/2 + i p^T y]

    이다. ``|Xi|^2``의 covariance는
    ``Sigma=(1/2)[M^(1/2) Omega M^(1/2)]^(-1)``이다. 즉 혼합곡률은
    marginal 폭뿐 아니라 q-R correlation과 조건부 중심 이동도 결정한다.

    반환 shape은 ``Xi(nq,nR)``, ``Sigma(2,2)``, ``omega(2,)``이다.
    """
    masses = np.array([model.proton_mass, model.heavy_mass], float)
    if np.any(masses <= 0.0):
        raise ValueError("두 핵 질량은 양수여야 합니다.")

    force = np.array([[k_q, k_qR], [k_qR, k_R]], float)
    inv_sqrt_mass = np.diag(1.0/np.sqrt(masses))
    sqrt_mass = np.diag(np.sqrt(masses))
    mass_weighted = inv_sqrt_mass@force@inv_sqrt_mass
    omega_squared, normal_modes = np.linalg.eigh(mass_weighted)
    if np.min(omega_squared) <= 0.0:
        raise ValueError(
            "혼합곡률까지 포함한 mass-weighted Hessian이 positive definite가 "
            f"아닙니다. omega^2={omega_squared}. 안정한 중심/전자상태를 "
            "선택하거나 force constant들을 직접 지정하세요."
        )

    frequencies = np.sqrt(omega_squared)
    omega_matrix = normal_modes@np.diag(frequencies)@normal_modes.T
    exponent_matrix = sqrt_mass@omega_matrix@sqrt_mass
    covariance = 0.5*np.linalg.inv(exponent_matrix)

    delta_q = model.q[:, None]-args.q0                         # (nq,1)
    delta_R = model.R[None, :]-args.R0                         # (1,nR)
    quadratic = (
        exponent_matrix[0, 0]*delta_q**2
        +2.0*exponent_matrix[0, 1]*delta_q*delta_R
        +exponent_matrix[1, 1]*delta_R**2
    )                                                           # (nq,nR)
    phase = args.proton_momentum*delta_q+args.heavy_momentum*delta_R
    xi = np.exp(-0.5*quadratic+1j*phase)                         # (nq,nR)
    xi /= np.sqrt(np.sum(np.abs(xi)**2)*model.dq*model.dR)
    return xi, covariance, frequencies


def initial_factors(model: Model, args):
    """Local BO 전자상태와 결합 harmonic nuclear Gaussian으로 초기화."""
    excitation = int(args.electron_excitation)
    if excitation < 0:
        raise ValueError("--electron-excitation은 0 이상이어야 합니다.")

    # ------------------------------------------------------------------
    # 1. 모든 (q,R)에서 local H_BO를 풀어 전자 초기상태/PES를 얻는다.
    # ------------------------------------------------------------------
    energies, electronic_states = local_electronic_basis(model, excitation+1)
    phi = electronic_states[excitation]                             # (nx,nq,nR)
    curvature = local_surface_curvature(
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
    kqR = (
        curvature["k_qR"]
        if args.cross_force_constant is None
        else args.cross_force_constant
    )

    # 사용자가 정한 (q0,R0)는 Gaussian의 전체 중심으로 그대로 유지한다.
    # local gradient는 중심을 옮기는 데 쓰지 않고 진단값으로만 저장한다.
    # 반면 Hessian의 혼합항은 물리적 q-R correlation에 반드시 포함한다.
    xi, covariance, frequencies = coupled_harmonic_state(
        model, args, kq, kqR, kR
    )                                                               # (nq,nR)
    proton_sigma = float(np.sqrt(covariance[0, 0]))
    heavy_sigma = float(np.sqrt(covariance[1, 1]))
    correlation = float(
        covariance[0, 1]/np.sqrt(covariance[0, 0]*covariance[1, 1])
    )
    conditional_slope = float(covariance[0, 1]/covariance[1, 1])
    conditional_sigma = float(np.sqrt(
        covariance[0, 0]-covariance[0, 1]**2/covariance[1, 1]
    ))

    # 계산된 초기 parameter도 NPZ의 args metadata에 함께 남긴다.
    args.electron_initial_state = "local-eigenstate"
    args.proton_sigma = proton_sigma
    args.heavy_sigma = heavy_sigma
    args.initial_proton_force_constant = kq
    args.initial_heavy_force_constant = kR
    args.initial_cross_curvature = kqR
    args.initial_gradient_q = curvature["gradient_q"]
    args.initial_gradient_R = curvature["gradient_R"]
    args.initial_covariance_qR = covariance
    args.initial_correlation_qR = correlation
    args.initial_conditional_center_slope = conditional_slope
    args.initial_conditional_proton_sigma = conditional_sigma
    args.initial_normal_frequencies = frequencies

    for name, sigma, spacing in (
        ("conditional proton", conditional_sigma, model.dq),
        ("heavy", heavy_sigma, model.dR),
    ):
        if sigma < 1.5*spacing:
            warnings.warn(
                f"{name} sigma={sigma:.4g}가 grid spacing={spacing:.4g}에 "
                "비해 좁습니다. 해당 grid 점 수를 늘려 convergence를 확인하세요.",
                RuntimeWarning,
            )

    # ------------------------------------------------------------------
    # 2. 결합 nuclear state Xi(q,R)를 nested factorization한다.
    # ------------------------------------------------------------------
    # chi에는 heavy momentum phase를 두고, Lambda에는 proton momentum과
    # Hessian이 결정한 조건부 q-R correlation을 남기는 gauge를 택한다.
    chi_amplitude = np.sqrt(np.sum(np.abs(xi)**2, axis=0)*model.dq) # (nR,)
    chi = chi_amplitude*np.exp(
        1j*args.heavy_momentum*(model.R-args.R0)
    )                                                               # (nR,)
    safe_chi = np.where(chi_amplitude > 1.0e-300, chi, 1.0+0.0j)
    lam = xi/safe_chi[None, :]                                      # (nq,nR)
    # Underflow로 marginal이 정확히 0인 열에서도 PNC를 명시적으로 유지한다.
    zero_columns = chi_amplitude <= 1.0e-300
    if np.any(zero_columns):
        fallback = normalized_gaussian(
            model.q, model.dq, args.q0, conditional_sigma,
            args.proton_momentum,
        )
        lam[:, zero_columns] = fallback[:, None]
    return phi.astype(complex), lam.astype(complex), chi.astype(complex)


def derivative(values, spacing, axis, order=1):
    """주기 격자 중앙 유한차분.

    같은 함수가 phi, Lambda, chi에 모두 쓰이지만 ``axis``가 다르다.
    예를 들어 phi에서 q 미분은 axis=1, R 미분은 axis=2이다.
    """
    if order == 1:
        return (
            np.roll(values, -1, axis=axis)-np.roll(values, 1, axis=axis)
        )/(2.0*spacing)
    if order == 2:
        return (
            np.roll(values, -1, axis=axis)-2.0*values
            +np.roll(values, 1, axis=axis)
        )/spacing**2
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
    transformed = dst(values, type=1, axis=0, norm="ortho")
    phase = np.exp(-1j*tau*electronic_kinetic_energies(model))
    phase = phase.reshape((-1,)+(1,)*(values.ndim-1))
    return idst(transformed*phase, type=1, axis=0, norm="ortho")


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


def _minus_covariant(field, vector, spacing, axis):
    """``(-i d - vector) field``."""
    return momentum(field, spacing, axis)-vector*field


def _plus_covariant(field, vector, spacing, axis):
    """``(-i d + vector) field``."""
    return momentum(field, spacing, axis)+vector*field


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


def proton_base_operator(lam, chi, a, b, alpha, model: Model, floor):
    """``H_pr`` 중 epsilon_1을 제외한 부분을 Lambda에 적용한다.

    반환값은 proton kinetic과 두 번째 factorization의 coupling을 합한
    ``base_lambda(nq,nR)``이다.
    """
    # [(-i d_q+a)^2/(2m_p)] Lambda
    dplus_q = _plus_covariant(lam, a, model.dq, axis=0)
    proton_kinetic = _plus_covariant(dplus_q, a, model.dq, axis=0)
    proton_kinetic *= 0.5/model.proton_mass

    # U_{p,n}^coup Lambda: R 방향에서 b-alpha가 vector field 역할을 한다.
    vector_R = b-alpha[None, :]
    dplus_R = _plus_covariant(lam, vector_R, model.dR, axis=1)
    dplus_R2 = _plus_covariant(dplus_R, vector_R, model.dR, axis=1)
    ratio_chi_R = regularized_ratio(
        momentum(chi, model.dR, axis=0), chi, floor
    )                                                               # (nR,)
    proton_nuclear_coupling = (
        0.5*dplus_R2+(ratio_chi_R+alpha)[None, :]*dplus_R
    )/model.heavy_mass
    return proton_kinetic+proton_nuclear_coupling


def instantaneous_functionals(phi, lam, chi, model: Model, floor=1.0e-10):
    """현재 세 factor에서 모든 EF scalar/vector potential을 계산한다.

    두 scalar gauge는 parallel-transport gauge를 사용한다:

        epsilon_1 = <Phi|H_el|Phi>_x
        epsilon_2 = <Lambda|H_pr|Lambda>_q

    반환 dictionary의 shape은 주석에 적힌 것과 같다.
    """
    a, b, alpha = geometric_fields(phi, lam, model)

    # ----- 첫 번째 factorization: 전자 coupling U_e,pn -----
    # 양성자 좌표 q 방향 항
    dminus_q = _minus_covariant(phi, a[None, :, :], model.dq, axis=1)
    dminus_q2 = _minus_covariant(dminus_q, a[None, :, :], model.dq, axis=1)
    ratio_lam_q = regularized_ratio(
        momentum(lam, model.dq, axis=0), lam, floor
    )                                                               # (nq,nR)
    u_q_phi = (
        0.5*dminus_q2+(ratio_lam_q+a)[None, :, :]*dminus_q
    )/model.proton_mass

    # 무거운 핵 좌표 R 방향 항. chi와 Lambda의 R 변화가 모두 들어간다.
    dminus_R = _minus_covariant(phi, b[None, :, :], model.dR, axis=2)
    dminus_R2 = _minus_covariant(dminus_R, b[None, :, :], model.dR, axis=2)
    ratio_chi_R = regularized_ratio(
        momentum(chi, model.dR, axis=0), chi, floor
    )                                                               # (nR,)
    ratio_lam_R = regularized_ratio(
        momentum(lam, model.dR, axis=1), lam, floor
    )                                                               # (nq,nR)
    u_R_phi = (
        0.5*dminus_R2
        +(ratio_chi_R[None, :]+ratio_lam_R+b)[None, :, :]*dminus_R
    )/model.heavy_mass
    u_phi = u_q_phi+u_R_phi                                         # (nx,nq,nR)

    hbo_phi = apply_electronic_hamiltonian(phi, model)
    epsilon_1 = (
        np.sum(np.conj(phi)*(hbo_phi+u_phi), axis=0)*model.dx
    ).real                                                          # (nq,nR)

    # ----- 두 번째 factorization: 양성자와 바깥 무거운 핵 -----
    base_lam = proton_base_operator(
        lam, chi, a, b, alpha, model, floor
    )                                                               # (nq,nR)
    hpr_lam = base_lam+epsilon_1*lam
    epsilon_2 = (
        np.sum(np.conj(lam)*hpr_lam, axis=0)*model.dq
    ).real                                                          # (nR,)

    return dict(
        a=a, b=b, alpha=alpha, epsilon_1=epsilon_1,
        epsilon_2=epsilon_2, u_phi=u_phi, base_lam=base_lam,
        hbo_phi=hbo_phi, hpr_lam=hpr_lam,
    )


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
    grid.add_argument("--nx", type=int, default=139, help="전자 interior 격자점 수")
    grid.add_argument("--nq", type=int, default=70, help="양성자 격자점 수")
    grid.add_argument("--nR", type=int, default=30, help="무거운 핵 격자점 수")
    grid.add_argument(
        "--x-max", type=float, default=8.0,
        help="전자 hard-wall 오른쪽 끝; 왼쪽 끝은 --left-position",
    )
    grid.add_argument("--q-min", type=float, default=-3.4)
    grid.add_argument("--q-max", type=float, default=3.6)
    grid.add_argument("--R-min", type=float, default=2.8)
    grid.add_argument("--R-max", type=float, default=5.8)

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
        "--cross-force-constant", type=float, default=None,
        help="생략하면 local BO surface의 d2E/(dq dR), 값 지정 시 override",
    )
    particle.add_argument(
        "--electron-excitation", type=int, default=0,
        help="초기 local H_BO 전자상태의 0-based index",
    )

    potential = parser.add_argument_group("soft-Coulomb potential")
    potential.add_argument("--left-position", type=float, default=-6.0)
    potential.add_argument("--left-charge", type=float, default=1.0)
    potential.add_argument("--heavy-charge", type=float, default=1.0)
    potential.add_argument("--soft-e-left", type=float, default=1.0)
    potential.add_argument("--soft-e-proton", type=float, default=0.8)
    potential.add_argument("--soft-e-heavy", type=float, default=1.0)
    potential.add_argument("--soft-p-left", type=float, default=0.8)
    potential.add_argument("--soft-p-heavy", type=float, default=0.8)
    potential.add_argument("--soft-left-heavy", type=float, default=0.8)
    return parser
