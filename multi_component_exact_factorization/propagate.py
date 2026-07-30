#!/usr/bin/env python3
"""세 coupled equation을 직접 푸는 1D multi-component EF 전파기.

이 프로그램은 full Psi를 시간 전파한 뒤 분해하는 reference가 아니다.
초기 ``Phi``는 local BO 고유상태로, ``Lambda``와 ``chi``는 full nuclear
Hessian으로 만든 상관 harmonic Gaussian으로 초기화한 뒤 다음 세 식을 함께
적분한다. 전자는 왼쪽 고정점에서 0이 되는 hard-wall 경계를 사용한다.

    i d_t Phi    = (H_el - epsilon_1) Phi
    i d_t Lambda = (H_pr - epsilon_2) Lambda
    i d_t chi    = [(-i d_R+alpha)^2/(2M)+epsilon_2] chi

전자 H_BO 부분은 split operator, 나머지 coupled 부분은 RK4를 사용한다.
Exact factorization의 'exact'는 분해와 연속 방정식을 뜻하며, 이 코드는
유한 격자/시간 간격을 사용하므로 반드시 convergence 검사가 필요하다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import (
    AU_PER_FS,
    add_model_arguments,
    build_model,
    derivative,
    electronic_split_step,
    initial_factors,
    instantaneous_functionals,
    pnc_project,
    reconstruct_psi,
)


def coupled_rhs(phi, lam, chi, model, args):
    """H_BO를 제외한 전자 RHS와 양성자/무거운 핵 RHS를 동시에 계산."""
    fields = instantaneous_functionals(
        phi, lam, chi, model, floor=args.density_threshold
    )

    # 전자 H_BO Phi는 split step에서 처리하므로 여기에는 U_coup-epsilon_1만 둔다.
    dphi = -1j*(
        fields["u_phi"]-fields["epsilon_1"][None, :, :]*phi
    )                                                               # (nx,nq,nR)

    # H_pr Lambda = base_lambda + epsilon_1 Lambda
    dlam = -1j*(
        fields["hpr_lam"]-fields["epsilon_2"][None, :]*lam
    )                                                               # (nq,nR)

    # 바깥 핵 방정식: [(-i d_R+alpha)^2/(2M)+epsilon_2] chi
    alpha = fields["alpha"]
    pchi = -1j*derivative(chi, model.dR, axis=0)+alpha*chi
    p2chi = -1j*derivative(pchi, model.dR, axis=0)+alpha*pchi
    dchi = -1j*(
        0.5*p2chi/model.heavy_mass+fields["epsilon_2"]*chi
    )                                                               # (nR,)
    return dphi, dlam, dchi, fields


def rk4_coupled_step(phi, lam, chi, dt, model, args):
    """세 factor의 coupled subflow를 고전적인 RK4로 한 스텝 전파."""
    k1p, k1l, k1c, _ = coupled_rhs(phi, lam, chi, model, args)
    k2p, k2l, k2c, _ = coupled_rhs(
        phi+0.5*dt*k1p, lam+0.5*dt*k1l, chi+0.5*dt*k1c, model, args
    )
    k3p, k3l, k3c, _ = coupled_rhs(
        phi+0.5*dt*k2p, lam+0.5*dt*k2l, chi+0.5*dt*k2c, model, args
    )
    k4p, k4l, k4c, _ = coupled_rhs(
        phi+dt*k3p, lam+dt*k3l, chi+dt*k3c, model, args
    )
    phi = phi+dt*(k1p+2*k2p+2*k3p+k4p)/6.0
    lam = lam+dt*(k1l+2*k2l+2*k3l+k4l)/6.0
    chi = chi+dt*(k1c+2*k2c+2*k3c+k4c)/6.0
    return pnc_project(phi, lam, chi, model)


def full_step(phi, lam, chi, dt, model, args):
    """``H_BO 반 -> coupled full -> H_BO 반`` 대칭 한 time step."""
    phi = electronic_split_step(phi, 0.5*dt, model)
    phi, lam, chi, pnc_error = rk4_coupled_step(
        phi, lam, chi, dt, model, args
    )
    phi = electronic_split_step(phi, 0.5*dt, model)
    # Split-transform roundoff까지 제거하되 세 factor의 곱 Psi는 보존한다.
    phi, lam, chi, final_error = pnc_project(phi, lam, chi, model)
    return phi, lam, chi, max(pnc_error, final_error)


def output_gauge(phi, lam, chi, fields, time_au, model, args):
    """PDF의 두 gauge 변환을 저장 representation에 명시적으로 적용한다.

    내부 전파는 두 time-Berry connection을 0으로 둔 parallel-transport
    gauge에서 수행한다. 사용자가 지정한 선형 ``theta_1(q,R,t)``와
    ``theta_2(R,t)``는 저장 직전에만 적용하므로 full Psi와 dynamics는
    바뀌지 않는다. 선형형은 공간/시간 미분을 정확히 알 수 있어 scalar와
    vector potential의 gauge 변환을 유한차분 오차 없이 확인하기 좋다.

    적용하는 factor 변환은 PDF의 식과 동일하다.

        Phi'    = exp(+i theta_1) Phi
        Lambda' = exp(-i theta_1+i theta_2) Lambda
        chi'    = exp(-i theta_2) chi

    반환 shape:
        phi_out(nx,nq,nR), lam_out(nq,nR), chi_out(nR)
        transformed: a/b/epsilon_1(nq,nR), alpha/epsilon_2(nR)
        theta_1(nq,nR), theta_2(nR)
    """
    q_offset = model.q[:, None]-args.q0                            # (nq,1)
    R_offset = model.R-args.R0                                     # (nR,)

    # theta_1과 theta_2를 기준 configuration (q0,R0,t=0)에서 0으로 둔다.
    # frequency는 d theta/dt이므로 scalar-potential의 상수 energy shift다.
    theta_1 = (
        args.theta1_q_gradient*q_offset
        +args.theta1_R_gradient*R_offset[None, :]
        +args.theta1_frequency*time_au
    )                                                               # (nq,nR)
    theta_2 = (
        args.theta2_R_gradient*R_offset
        +args.theta2_frequency*time_au
    )                                                               # (nR,)

    # 세 phase factor의 곱은 정확히 1이므로 Psi=Phi*Lambda*chi는 점별 보존된다.
    phase_phi = np.exp(1j*theta_1)                                  # (nq,nR)
    phase_lam = np.exp(-1j*theta_1+1j*theta_2[None, :])             # (nq,nR)
    phase_chi = np.exp(-1j*theta_2)                                 # (nR,)
    phi_out = phase_phi[None, :, :]*phi                             # (nx,nq,nR)
    lam_out = phase_lam*lam                                         # (nq,nR)
    chi_out = phase_chi*chi                                         # (nR,)

    # 선형 theta의 해석적 미분을 PDF의 potential 변환식에 적용한다.
    # a'=a+d_q theta_1, b'=b+d_R theta_1, alpha'=alpha+d_R theta_2
    # epsilon_k'=epsilon_k+d_t theta_k
    transformed = dict(
        a=fields["a"]+args.theta1_q_gradient,
        b=fields["b"]+args.theta1_R_gradient,
        alpha=fields["alpha"]+args.theta2_R_gradient,
        epsilon_1=fields["epsilon_1"]+args.theta1_frequency,
        epsilon_2=fields["epsilon_2"]+args.theta2_frequency,
    )
    return phi_out, lam_out, chi_out, transformed, theta_1, theta_2


def run(args):
    """초기화, direct nested EF 전파, NPZ 저장을 수행한다."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = build_model(args)
    phi, lam, chi = initial_factors(model, args)
    print(
        "초기 배열: "
        f"Phi={phi.shape}, Lambda={lam.shape}, chi={chi.shape}"
    )
    print(
        "격자 간격/경계: "
        f"dx={model.dx:.6f} (hard wall {model.x_left:.3f}..{model.x_right:.3f}), "
        f"dq={model.dq:.6f}, dR={model.dR:.6f}"
    )
    print(
        "초기 correlated nuclear Gaussian: "
        f"(q0,R0)=({args.q0:.4f},{args.R0:.4f}), "
        f"sigma_q={args.proton_sigma:.6f}, "
        f"k_q={args.initial_proton_force_constant:.6f}; "
        f"sigma_R={args.heavy_sigma:.6f}, "
        f"k_R={args.initial_heavy_force_constant:.6f}; "
        f"k_qR={args.initial_cross_curvature:.6f}, "
        f"rho_qR={args.initial_correlation_qR:.6f}; "
        f"p_q={args.proton_momentum:.4f}, p_R={args.heavy_momentum:.4f}"
    )

    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    save_steps = list(range(0, n_steps+1, max(1, args.save_every)))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    nt = len(save_steps)

    # 시간축이 앞에 오는 저장 배열들이다. 이 표를 읽으면 NPZ shape도 같다.
    times_fs = np.empty(nt)                                         # (nt,)
    phis = np.empty((nt, args.nx, args.nq, args.nR), complex)       # (nt,nx,nq,nR)
    lams = np.empty((nt, args.nq, args.nR), complex)                # (nt,nq,nR)
    chis = np.empty((nt, args.nR), complex)                         # (nt,nR)
    avec = np.empty((nt, args.nq, args.nR))                         # first a(q,R)
    bvec = np.empty_like(avec)                                      # first b(q,R)
    alpha = np.empty((nt, args.nR))                                 # second alpha(R)
    eps1 = np.empty_like(avec)                                      # epsilon_1(q,R)
    eps2 = np.empty((nt, args.nR))                                  # epsilon_2(R)
    theta1 = np.empty_like(avec)                                    # theta_1(q,R,t)
    theta2 = np.empty((nt, args.nR))                                # theta_2(R,t)
    norm = np.empty(nt)
    pnc = np.empty(nt)                                               # 저장 factor의 실제 PNC 잔차
    projection_correction = np.empty(nt)                             # substep 투영 전 최대 이탈
    psis = []                                                       # 선택 저장 (nt,nx,nq,nR)

    def save_frame(frame, step, step_projection_correction=0.0):
        """현재 base-gauge factor를 선택 gauge로 바꾸어 한 frame 저장."""
        # 1) 현재 factor로부터 base-gauge의 모든 EF potential을 계산한다.
        fields = instantaneous_functionals(
            phi, lam, chi, model, floor=args.density_threshold
        )
        time_au = step*args.dt_au

        # 2) 사용자가 요청한 theta_1/theta_2를 factor와 potential에 함께 적용한다.
        #    내부 전파 변수 phi/lam/chi 자체는 건드리지 않고 출력 복사본만 만든다.
        phi_out, lam_out, chi_out, saved_fields, th1, th2 = output_gauge(
            phi, lam, chi, fields, time_au, model, args
        )

        # 3) Gauge 변환 후에도 full Psi가 같다는 representation으로 norm을 잰다.
        psi = reconstruct_psi(phi_out, lam_out, chi_out)
        times_fs[frame] = time_au/AU_PER_FS
        phis[frame], lams[frame], chis[frame] = phi_out, lam_out, chi_out
        avec[frame], bvec[frame] = saved_fields["a"], saved_fields["b"]
        alpha[frame] = saved_fields["alpha"]
        eps1[frame] = saved_fields["epsilon_1"]
        eps2[frame] = saved_fields["epsilon_2"]
        theta1[frame], theta2[frame] = th1, th2
        norm[frame] = (
            np.sum(np.abs(psi)**2)*model.dx*model.dq*model.dR
        )
        # 4) 두 partial normalization condition의 최악 오차를 기록한다.
        phi_err = np.max(
            np.abs(np.sum(np.abs(phi)**2, axis=0)*model.dx-1.0)
        )
        lam_err = np.max(
            np.abs(np.sum(np.abs(lam)**2, axis=0)*model.dq-1.0)
        )
        pnc[frame] = max(float(phi_err), float(lam_err))
        projection_correction[frame] = step_projection_correction
        if args.save_psi:
            psis.append(psi.copy())

    save_frame(0, 0)
    frame = 1
    last_pnc = 0.0
    for step in range(1, n_steps+1):
        phi, lam, chi, last_pnc = full_step(
            phi, lam, chi, args.dt_au, model, args
        )
        if not (
            np.all(np.isfinite(phi))
            and np.all(np.isfinite(lam))
            and np.all(np.isfinite(chi))
        ):
            raise FloatingPointError(
                f"step {step}에서 non-finite 값이 발생했습니다. "
                "dt를 줄이거나 density-threshold를 키우세요."
            )
        if frame < nt and step == save_steps[frame]:
            save_frame(frame, step, last_pnc)
            frame += 1
        if step % max(1, args.progress_every) == 0 or step == n_steps:
            print(
                f"step {step:6d}/{n_steps}  "
                f"t={step*args.dt_au/AU_PER_FS:8.4f} fs"
            )

    # PDF Eqs. (52)--(54)의 gauge-dependent time connection을 저장 frame의
    # 중앙차분으로 진단한다. 기본 parallel-transport gauge에서는 둘 다
    # 0에 가까워야 하고, 선형 gauge에서는 각각 지정한 frequency가 된다.
    if nt >= 2:
        times_au = times_fs*AU_PER_FS
        edge_order = 2 if nt >= 3 else 1
        dphi_dt = np.gradient(phis, times_au, axis=0, edge_order=edge_order)
        dlam_dt = np.gradient(lams, times_au, axis=0, edge_order=edge_order)
        # epsilon_GD^(1)=<Phi|-i d_t Phi>_x, shape (nt,nq,nR)
        epsilon_gd_1 = (
            np.sum(np.conj(phis)*(-1j*dphi_dt), axis=1)*model.dx
        ).real                                                          # (nt,nq,nR)

        # 먼저 <Lambda|-i d_t Lambda>_q를 계산한 뒤 PDF Eq. (54)
        # epsilon_GD^(2)=Lambda connection+<Lambda|epsilon_GD^(1)|Lambda>_q
        # 를 조립한다. 최종 shape은 (nt,nR)이다.
        lambda_gd = (
            np.sum(np.conj(lams)*(-1j*dlam_dt), axis=1)*model.dq
        ).real                                                          # (nt,nR)
        epsilon_gd_2 = lambda_gd+np.sum(
            np.abs(lams)**2*epsilon_gd_1, axis=1
        )*model.dq                                                       # (nt,nR)
    else:
        epsilon_gd_1 = np.zeros_like(eps1)
        epsilon_gd_2 = np.zeros_like(eps2)

    gauge_coefficients = (
        args.theta1_q_gradient, args.theta1_R_gradient, args.theta1_frequency,
        args.theta2_R_gradient, args.theta2_frequency,
    )
    gauge_name = (
        "parallel_transport_two_level"
        if all(value == 0.0 for value in gauge_coefficients)
        else "linear_transform_of_parallel_transport_two_level"
    )
    payload = dict(
        kind=np.array("direct_multi_component_exact_factorization"),
        representation=np.array(
            "nested_realspace_correlated_harmonic_hardwall_electron"
        ),
        gauge=np.array(gauge_name),
        base_gauge=np.array("parallel_transport_two_level"),
        x=model.x, q=model.q, R=model.R, times_fs=times_fs,
        phi=phis, lambda_wavefunction=lams, chi=chis,
        a=avec, b=bvec, alpha=alpha,
        epsilon_1=eps1, epsilon_2=eps2,
        theta_1=theta1, theta_2=theta2,
        epsilon_gd_1=epsilon_gd_1, epsilon_gd_2=epsilon_gd_2,
        norm=norm, pnc_error=pnc,
        pnc_projection_correction=projection_correction,
        args=np.array([vars(args)], dtype=object),
    )
    if args.save_psi:
        payload["psi"] = np.asarray(psis)
    path = outdir/"multi_component_direct_ef.npz"
    np.savez_compressed(path, **payload)
    print(f"저장 완료: {path}")
    print(f"최대 norm 오차: {np.max(np.abs(norm-1.0)):.3e}")
    print(f"최대 저장 PNC 오차:       {np.max(pnc):.3e}")
    print(f"최대 PNC projection 보정: {np.max(projection_correction):.3e}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default="results/multi_component_exact_factorization/direct"
    )
    parser.add_argument("--dt-au", type=float, default=0.005)
    parser.add_argument("--t-final-fs", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--density-threshold", type=float, default=1.0e-9,
        help="chi/Lambda node에서 logarithmic derivative를 안정화하는 상대 floor",
    )
    parser.add_argument(
        "--save-psi", action="store_true",
        help="큰 full Psi 배열도 archive에 저장한다",
    )
    gauge = parser.add_argument_group("두 단계 gauge (parallel-transport 기준)")
    gauge.add_argument(
        "--theta1-q-gradient", type=float, default=0.0,
        help="d theta_1/dq; a(q,R,t)에 더해지는 상수",
    )
    gauge.add_argument(
        "--theta1-R-gradient", type=float, default=0.0,
        help="d theta_1/dR; b(q,R,t)에 더해지는 상수",
    )
    gauge.add_argument(
        "--theta1-frequency", type=float, default=0.0,
        help="d theta_1/dt; atomic-unit energy shift",
    )
    gauge.add_argument(
        "--theta2-R-gradient", type=float, default=0.0,
        help="d theta_2/dR; alpha(R,t)에 더해지는 상수",
    )
    gauge.add_argument(
        "--theta2-frequency", type=float, default=0.0,
        help="d theta_2/dt; atomic-unit energy shift",
    )
    add_model_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
