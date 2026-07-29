#!/usr/bin/env python3
"""3차원 full TDSE reference와 사후 nested exact factorization.

이 경로는 direct multi-component EF와 완전히 독립적이다. 먼저
``Psi(x,q,R,t)``를 3D split operator로 전파하고, 저장된 각 frame을

    Psi = Phi_{R,q} Lambda_R chi

로 분해한다. 따라서 direct coupled equation의 정확도를 검증하는 기준으로
쓸 수 있다. Reference 결과가 direct 전파의 다음 step에 공급되지는 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import (
    AU_PER_FS,
    add_model_arguments,
    build_model,
    geometric_fields,
    initial_factors,
    nested_factorize,
    proton_base_operator,
    reconstruct_psi,
    regularized_ratio,
)


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = build_model(args)
    phi0, lam0, chi0 = initial_factors(model, args)
    psi = reconstruct_psi(phi0, lam0, chi0)                         # (nx,nq,nR)
    print(f"full Psi 초기 shape: {psi.shape}")

    # 세 좌표의 운동에너지는 Fourier 공간에서 대각이므로 한꺼번에 적용한다.
    kx = 2*np.pi*np.fft.fftfreq(args.nx, d=model.dx)
    kq = 2*np.pi*np.fft.fftfreq(args.nq, d=model.dq)
    kR = 2*np.pi*np.fft.fftfreq(args.nR, d=model.dR)
    kinetic = (
        0.5*kx[:, None, None]**2
        +0.5*kq[None, :, None]**2/model.proton_mass
        +0.5*kR[None, None, :]**2/model.heavy_mass
    )
    half_t = np.exp(-0.5j*args.dt_au*kinetic)
    full_v = np.exp(-1j*args.dt_au*model.potential)

    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    save_steps = list(range(0, n_steps+1, max(1, args.save_every)))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)

    psi_frames = []
    times_au = []
    norms = []

    def save_frame(step):
        psi_frames.append(psi.copy())
        times_au.append(step*args.dt_au)
        norms.append(
            np.sum(np.abs(psi)**2)*model.dx*model.dq*model.dR
        )

    save_frame(0)
    frame = 1
    for step in range(1, n_steps+1):
        psi = np.fft.ifftn(np.fft.fftn(psi)*half_t)
        psi *= full_v
        psi = np.fft.ifftn(np.fft.fftn(psi)*half_t)
        if frame < len(save_steps) and step == save_steps[frame]:
            save_frame(step)
            frame += 1
        if step % max(1, args.progress_every) == 0 or step == n_steps:
            print(
                f"reference step {step:6d}/{n_steps}  "
                f"t={step*args.dt_au/AU_PER_FS:8.4f} fs"
            )

    psi_frames = np.asarray(psi_frames)                             # (nt,nx,nq,nR)
    times_au = np.asarray(times_au)                                 # (nt,)
    nt = len(times_au)

    # 매 frame에서 두 번의 exact factorization을 순서대로 수행한다.
    phis = np.empty_like(psi_frames)
    lams = np.empty((nt, args.nq, args.nR), complex)
    chis = np.empty((nt, args.nR), complex)
    avec = np.empty((nt, args.nq, args.nR))
    bvec = np.empty_like(avec)
    alpha = np.empty((nt, args.nR))
    for it in range(nt):
        phis[it], lams[it], chis[it] = nested_factorize(
            psi_frames[it], model
        )
        avec[it], bvec[it], alpha[it] = geometric_fields(
            phis[it], lams[it], model
        )

    # 양의 marginal gauge에서 scalar potential은 시간미분을 포함한 각
    # conditional equation을 역으로 풀어 추출한다.
    edge_order = 2 if nt >= 3 else 1
    dchi_dt = np.gradient(chis, times_au, axis=0, edge_order=edge_order)
    dlam_dt = np.gradient(lams, times_au, axis=0, edge_order=edge_order)
    dphi_dt = np.gradient(phis, times_au, axis=0, edge_order=edge_order)

    # PDF Eqs. (52)--(54)의 gauge-dependent time connection도 reference와
    # 함께 저장한다. Reference factorization은 chi와 Lambda를 양의 실수로
    # 고르는 positive-marginal gauge이므로 direct parallel-transport 결과와
    # 이 값 자체를 바로 비교하면 안 된다.
    # epsilon_GD^(1)=<Phi|-i d_t Phi>_x, shape (nt,nq,nR)
    epsilon_gd_1 = (
        np.sum(np.conj(phis)*(-1j*dphi_dt), axis=1)*model.dx
    ).real                                                           # (nt,nq,nR)

    # epsilon_GD^(2)=<Gamma_R|-i d_t Gamma_R>_{p,e}, shape (nt,nR)
    # Gamma_R=Lambda_R Phi에 product rule을 적용한 PDF Eq. (54)를 사용한다.
    lambda_gd = (
        np.sum(np.conj(lams)*(-1j*dlam_dt), axis=1)*model.dq
    ).real                                                           # (nt,nR)
    epsilon_gd_2 = lambda_gd+np.sum(
        np.abs(lams)**2*epsilon_gd_1, axis=1
    )*model.dq
    epsilon_2 = np.empty((nt, args.nR))
    epsilon_1 = np.empty((nt, args.nq, args.nR))
    for it in range(nt):
        pchi = (
            -1j*np.gradient(chis[it], model.dR, edge_order=2)
            +alpha[it]*chis[it]
        )
        p2chi = (
            -1j*np.gradient(pchi, model.dR, edge_order=2)
            +alpha[it]*pchi
        )
        nuclear_kinetic = 0.5*p2chi/model.heavy_mass
        eps2_complex = regularized_ratio(
            1j*dchi_dt[it]-nuclear_kinetic,
            chis[it], args.density_threshold,
        )
        epsilon_2[it] = eps2_complex.real

        base_lam = proton_base_operator(
            lams[it], chis[it], avec[it], bvec[it], alpha[it], model,
            args.density_threshold,
        )
        eps1_complex = regularized_ratio(
            1j*dlam_dt[it]-base_lam+epsilon_2[it][None, :]*lams[it],
            lams[it], args.density_threshold,
        )
        epsilon_1[it] = eps1_complex.real

    payload = dict(
        kind=np.array("full_tdse_multi_component_ef_reference"),
        representation=np.array("nested_factorization_of_full_psi"),
        gauge=np.array("positive_marginal_nested_gauge"),
        x=model.x, q=model.q, R=model.R, times_fs=times_au/AU_PER_FS,
        phi=phis, lambda_wavefunction=lams, chi=chis,
        a=avec, b=bvec, alpha=alpha,
        epsilon_1=epsilon_1, epsilon_2=epsilon_2,
        theta_1=np.zeros_like(avec), theta_2=np.zeros_like(alpha),
        epsilon_gd_1=epsilon_gd_1, epsilon_gd_2=epsilon_gd_2,
        norm=np.asarray(norms),
        args=np.array([vars(args)], dtype=object),
    )
    if not args.compact:
        payload["psi"] = psi_frames
    path = outdir/"multi_component_reference.npz"
    np.savez_compressed(path, **payload)
    print(f"저장 완료: {path}")
    print(f"최대 full-TDSE norm 오차: {np.max(np.abs(np.asarray(norms)-1)):.3e}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default="results/multi_component_exact_factorization/reference"
    )
    parser.add_argument("--dt-au", type=float, default=0.005)
    parser.add_argument("--t-final-fs", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--density-threshold", type=float, default=1.0e-9)
    parser.add_argument(
        "--compact", action="store_true",
        help="재구성 가능한 full Psi를 archive에서 생략한다",
    )
    add_model_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
