#!/usr/bin/env python3
"""CPU-double과 GPU single/mixed archive의 gauge-invariant 오차를 비교한다."""

from __future__ import annotations

import argparse

import numpy as np

from multi_component_exact_factorization.compare import psi_from_archive


def run(args):
    reference = np.load(args.reference, allow_pickle=True)
    candidate = np.load(args.candidate, allow_pickle=True)
    for key in ("x", "q", "R", "times_fs"):
        if reference[key].shape != candidate[key].shape or not np.allclose(
            reference[key], candidate[key], atol=1.0e-12, rtol=0.0
        ):
            raise ValueError(f"{key} grid/time이 서로 다릅니다.")

    x, q, R = reference["x"], reference["q"], reference["R"]
    dx, dq, dR = x[1]-x[0], q[1]-q[0], R[1]-R[0]
    dv = dx*dq*dR
    fidelity_loss, electron_l1, proton_l1, heavy_l1 = [], [], [], []
    mean_x_error, mean_q_error, mean_R_error = [], [], []

    for frame in range(len(reference["times_fs"])):
        psi_ref = psi_from_archive(reference, frame).astype(np.complex128)
        psi_test = psi_from_archive(candidate, frame).astype(np.complex128)
        norm_ref = np.sum(np.abs(psi_ref)**2)*dv
        norm_test = np.sum(np.abs(psi_test)**2)*dv
        overlap = np.sum(np.conj(psi_ref)*psi_test)*dv
        fidelity_loss.append(max(
            0.0, 1.0-abs(overlap)**2/(norm_ref*norm_test)
        ))

        rho_ref, rho_test = np.abs(psi_ref)**2/norm_ref, np.abs(psi_test)**2/norm_test
        marginals_ref = (
            np.sum(rho_ref, axis=(1, 2))*dq*dR,
            np.sum(rho_ref, axis=(0, 2))*dx*dR,
            np.sum(rho_ref, axis=(0, 1))*dx*dq,
        )
        marginals_test = (
            np.sum(rho_test, axis=(1, 2))*dq*dR,
            np.sum(rho_test, axis=(0, 2))*dx*dR,
            np.sum(rho_test, axis=(0, 1))*dx*dq,
        )
        spacings = (dx, dq, dR)
        grids = (x, q, R)
        errors = (electron_l1, proton_l1, heavy_l1)
        mean_errors = (mean_x_error, mean_q_error, mean_R_error)
        for grid, spacing, ref_rho, test_rho, l1_values, mean_values in zip(
            grids, spacings, marginals_ref, marginals_test, errors, mean_errors
        ):
            l1_values.append(np.sum(np.abs(ref_rho-test_rho))*spacing)
            ref_mean = np.sum(grid*ref_rho)*spacing
            test_mean = np.sum(grid*test_rho)*spacing
            mean_values.append(abs(test_mean-ref_mean))

    print(f"비교 frame 수:              {len(fidelity_loss)}")
    print(f"최대 1-fidelity:            {max(fidelity_loss):.3e}")
    print(f"최대 electron marginal L1: {max(electron_l1):.3e}")
    print(f"최대 proton marginal L1:   {max(proton_l1):.3e}")
    print(f"최대 heavy marginal L1:    {max(heavy_l1):.3e}")
    print(f"최대 electron mean 오차:   {max(mean_x_error):.3e} a0")
    print(f"최대 proton mean 오차:     {max(mean_q_error):.3e} a0")
    print(f"최대 heavy mean 오차:      {max(mean_R_error):.3e} a0")
    print("판정할 때 precision 오차가 dt/grid convergence 오차보다 작은지 확인하세요.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="CPU complex128 archive")
    parser.add_argument("candidate", help="GPU single/mixed archive")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
