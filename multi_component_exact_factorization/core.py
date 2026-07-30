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
    proton_mass: float
    heavy_mass: float
    potential: np.ndarray         # (nx,nq,nR)


def soft_inverse(distance: np.ndarray, softening: float) -> np.ndarray:
    """1D Coulomb 특이점을 완화한 ``1/sqrt(d^2+a^2)``."""
    return 1.0 / np.sqrt(distance**2 + softening**2)


def build_model(args) -> Model:
    """왼쪽 고정점-전자-양성자-무거운 핵의 1D model을 만든다.

    왼쪽 중심은 움직이지 않으므로 파동함수 좌표축을 갖지 않는다.
    그 영향은 potential에만 포함된다. q와 R 범위를 서로 떨어뜨려 두면
    기본 계산에서 양성자와 무거운 핵의 위치 순서가 자연스럽게 유지된다.
    """
    # FFT 전파와 맞추기 위해 endpoint=False인 균일 주기 격자를 사용한다.
    x = np.linspace(args.x_min, args.x_max, args.nx, endpoint=False)
    q = np.linspace(args.q_min, args.q_max, args.nq, endpoint=False)
    R = np.linspace(args.R_min, args.R_max, args.nR, endpoint=False)
    dx = float(x[1] - x[0])
    dq = float(q[1] - q[0])
    dR = float(R[1] - R[0])

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
    # 실제 전파에서 T_x는 FFT로 적용한다. 초기 고유상태도 같은 discrete
    # Hamiltonian에서 얻어야 하므로, 단위행렬의 각 열에 FFT kinetic을
    # 적용하여 그 연산자의 (nx,nx) matrix 표현을 구성한다.
    kx = 2*np.pi*np.fft.fftfreq(nx, d=model.dx)
    identity = np.eye(nx, dtype=complex)
    kinetic = np.fft.ifft(
        np.fft.fft(identity, axis=0)*(0.5*kx**2)[:, None], axis=0
    )
    # Roundoff로 생기는 작은 anti-Hermitian/imaginary 성분을 제거한다.
    kinetic = 0.5*(kinetic+kinetic.conj().T)

    energies = np.empty((n_states, nq, nR), float)                 # (state,nq,nR)
    states = np.empty((n_states, nx, nq, nR), complex)             # (state,nx,nq,nR)

    # ------------------------------------------------------------------
    # 2. 모든 (q_j,R_k)에서 작은 nx x nx Hermitian 문제를 독립적으로 푼다.
    # ------------------------------------------------------------------
    for iR in range(nR):
        for iq in range(nq):
            hamiltonian = kinetic+np.diag(model.potential[:, iq, iR])
            values, vectors = np.linalg.eigh(hamiltonian)

            # np.linalg.eigh의 열벡터는 sum_x |v_x|^2=1이다. 연속 PNC
            # sum_x |phi_x|^2 dx=1에 맞추기 위해 sqrt(dx)로 나눈다.
            chosen = vectors[:, :n_states].T.astype(complex)/np.sqrt(model.dx)

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
            energies[:, iq, iR] = values[:n_states]
            states[:, :, iq, iR] = chosen
    return energies, states


def local_surface_curvature(surface, model: Model, q0: float, R0: float):
    """BO surface의 ``(q0,R0)`` 주변 3x3 점을 이차식으로 fit한다.

    ``E = c + gq*dq + gR*dR + kq*dq^2/2 + kqR*dq*dR
          + kR*dR^2/2``

    의 계수를 least squares로 구한다. 반환값은 두 gradient, 두 대각 force
    constant와 혼합 curvature다. 초기 nuclear Gaussian을 product 형태로
    유지하기 위해 폭에는 대각 ``kq``, ``kR``만 사용하고 혼합항은 진단용으로
    저장한다.
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


def harmonic_density_sigma(mass: float, force_constant: float):
    """Harmonic ground-state 확률밀도의 표준편차.

    ``omega=sqrt(k/m)``이고 ``Var(x)=1/(2*m*omega)``이므로
    ``sigma=(1/(4*m*k))^(1/4)``이다.
    """
    if mass <= 0.0 or force_constant <= 0.0:
        raise ValueError("mass와 force constant는 양수여야 합니다.")
    return (1.0/(4.0*mass*force_constant))**0.25


def initial_factors(model: Model, args):
    """Local BO 전자상태와 고정 중심 nuclear Gaussian으로 초기화한다."""
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
    if kq <= 0.0 or kR <= 0.0:
        raise ValueError(
            "초기 폭에 사용할 대각 force constant가 양수가 아닙니다: "
            f"k_q={kq:.6g}, k_R={kR:.6g} "
            f"(BO 곡률: {curvature['k_q']:.6g}, {curvature['k_R']:.6g}). "
            "양의 곡률 위치를 선택하거나 --proton-force-constant와 "
            "--heavy-force-constant에 양수를 직접 지정하세요."
        )
    proton_sigma = harmonic_density_sigma(model.proton_mass, kq)
    heavy_sigma = harmonic_density_sigma(model.heavy_mass, kR)

    # 계산된 초기 parameter도 NPZ의 args metadata에 함께 남긴다.
    args.electron_initial_state = "local-eigenstate"
    args.proton_sigma = proton_sigma
    args.heavy_sigma = heavy_sigma
    args.initial_proton_force_constant = kq
    args.initial_heavy_force_constant = kR
    args.initial_cross_curvature = curvature["k_qR"]
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
    # 2. 가장 바깥 factor: heavy-nucleus marginal chi(R,0), shape (nR,)
    # ------------------------------------------------------------------
    chi = normalized_gaussian(
        model.R, model.dR, args.R0, heavy_sigma, args.heavy_momentum
    )                                                               # (nR,)

    # ------------------------------------------------------------------
    # 3. Proton 중심은 모든 R에서 정확히 q0로 고정한다.
    # ------------------------------------------------------------------
    proton_line = normalized_gaussian(
        model.q, model.dq, args.q0, proton_sigma, args.proton_momentum
    )                                                               # (nq,)
    lam = np.repeat(proton_line[:, None], len(model.R), axis=1)     # (nq,nR)
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


def apply_electronic_hamiltonian(phi, model: Model):
    """고유상태 계산 없이 ``[-d_x^2/2+V(x,q,R)] Phi``를 적용한다."""
    kx = 2*np.pi*np.fft.fftfreq(len(model.x), d=model.dx)
    kinetic = np.fft.ifft(
        np.fft.fft(phi, axis=0)*(0.5*kx**2)[:, None, None], axis=0
    )
    return kinetic + model.potential*phi


def electronic_split_step(phi, tau, model: Model):
    """전자 ``T_x/2 -> V -> T_x/2`` split-operator 한 번."""
    kx = 2*np.pi*np.fft.fftfreq(len(model.x), d=model.dx)
    half_t = np.exp(-0.25j*tau*kx**2)[:, None, None]
    phi = np.fft.ifft(np.fft.fft(phi, axis=0)*half_t, axis=0)
    phi *= np.exp(-1j*tau*model.potential)
    phi = np.fft.ifft(np.fft.fft(phi, axis=0)*half_t, axis=0)
    return phi


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
    grid.add_argument("--nx", type=int, default=48, help="전자 격자점 수")
    grid.add_argument("--nq", type=int, default=64, help="양성자 격자점 수")
    grid.add_argument("--nR", type=int, default=72, help="무거운 핵 격자점 수")
    grid.add_argument("--x-min", type=float, default=-12.0)
    grid.add_argument("--x-max", type=float, default=12.0)
    grid.add_argument("--q-min", type=float, default=-3.5)
    grid.add_argument("--q-max", type=float, default=3.0)
    grid.add_argument("--R-min", type=float, default=3.4)
    grid.add_argument("--R-max", type=float, default=8.0)

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
