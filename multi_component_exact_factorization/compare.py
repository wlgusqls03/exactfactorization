#!/usr/bin/env python3
"""Full-TDSE reference와 direct nested EF의 gauge-invariant 결과 비교."""

from __future__ import annotations

import argparse

import numpy as np


def psi_from_archive(data, frame):
    """저장된 Psi가 없으면 세 factor의 곱으로 재구성한다."""
    if "psi" in data:
        return data["psi"][frame]
    return (
        data["phi"][frame]
        *data["lambda_wavefunction"][frame][None, :, :]
        *data["chi"][frame][None, None, :]
    )


def run(args):
    ref = np.load(args.reference, allow_pickle=True)
    direct = np.load(args.direct, allow_pickle=True)
    for key in ("x", "q", "R"):
        if ref[key].shape != direct[key].shape or not np.allclose(ref[key], direct[key]):
            raise ValueError(f"{key} 격자가 서로 다릅니다.")

    dx = float(ref["x"][1]-ref["x"][0])
    dq = float(ref["q"][1]-ref["q"][0])
    dR = float(ref["R"][1]-ref["R"][0])
    dv = dx*dq*dR
    fidelities, heavy_l1, proton_heavy_l1, electron_heavy_l1 = [], [], [], []

    for it, time in enumerate(direct["times_fs"]):
        jt = int(np.argmin(np.abs(ref["times_fs"]-time)))
        if abs(float(ref["times_fs"][jt]-time)) > args.time_tolerance_fs:
            continue
        psi_d = psi_from_archive(direct, it)
        psi_r = psi_from_archive(ref, jt)
        overlap = np.sum(np.conj(psi_r)*psi_d)*dv
        norm_d = np.sum(np.abs(psi_d)**2)*dv
        norm_r = np.sum(np.abs(psi_r)**2)*dv
        fidelities.append(abs(overlap)**2/(norm_d*norm_r))

        rho_d = np.abs(psi_d)**2
        rho_r = np.abs(psi_r)**2
        heavy_d = np.sum(rho_d, axis=(0, 1))*dx*dq
        heavy_r = np.sum(rho_r, axis=(0, 1))*dx*dq
        proton_heavy_d = np.sum(rho_d, axis=0)*dx
        proton_heavy_r = np.sum(rho_r, axis=0)*dx
        electron_heavy_d = np.sum(rho_d, axis=1)*dq
        electron_heavy_r = np.sum(rho_r, axis=1)*dq
        heavy_l1.append(np.sum(np.abs(heavy_d-heavy_r))*dR)
        proton_heavy_l1.append(
            np.sum(np.abs(proton_heavy_d-proton_heavy_r))*dq*dR
        )
        electron_heavy_l1.append(
            np.sum(np.abs(electron_heavy_d-electron_heavy_r))*dx*dR
        )

    if not fidelities:
        raise ValueError("비교할 수 있는 공통 저장 시간이 없습니다.")
    print(f"비교 frame 수:                 {len(fidelities)}")
    print(f"최소 full-Psi fidelity:        {min(fidelities):.10f}")
    print(f"최대 heavy density L1 오차:    {max(heavy_l1):.3e}")
    print(f"최대 proton-heavy L1 오차:     {max(proton_heavy_l1):.3e}")
    print(f"최대 electron-heavy L1 오차:   {max(electron_heavy_l1):.3e}")
    print("주의: scalar/vector potential은 gauge가 다르면 직접 비교하지 않습니다.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("direct")
    parser.add_argument("--time-tolerance-fs", type=float, default=1.0e-10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
