"""Compare compact physical outputs of reference and fused BO-link runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _resolve(value):
    path = Path(value).expanduser()
    if path.is_file():
        return path
    matches = list(path.glob("**/multi_component_born_huang_ef_gpu.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Born--Huang archive를 하나만 지정해야 합니다: {path}"
        )
    return matches[0]


def _common_frames(reference, candidate, tolerance):
    pairs = []
    for left, time in enumerate(reference):
        right = int(np.argmin(np.abs(candidate-time)))
        if abs(float(candidate[right]-time)) <= tolerance:
            pairs.append((left, right))
    if not pairs:
        raise ValueError("일치하는 저장 시간이 없습니다.")
    return np.asarray(pairs, dtype=int)


def run(args):
    reference_path = _resolve(args.reference)
    fused_path = _resolve(args.fused)
    with np.load(reference_path, allow_pickle=True) as reference, np.load(
        fused_path, allow_pickle=True
    ) as fused:
        pairs = _common_frames(
            reference["times_fs"], fused["times_fs"], args.time_tolerance_fs
        )
        left, right = pairs[:, 0], pairs[:, 1]
        kernel_reference = str(reference.get("bo_link_kernel", "legacy"))
        kernel_fused = str(fused.get("bo_link_kernel", "legacy"))
        norm_difference = float(np.max(np.abs(
            reference["norm"][left]-fused["norm"][right]
        )))
        population_difference = float(np.max(np.abs(
            reference["bo_populations"][left]-fused["bo_populations"][right]
        )))
        rho_q_reference = np.sum(
            reference["bo_state_density_q"][left], axis=1
        )
        rho_q_fused = np.sum(fused["bo_state_density_q"][right], axis=1)
        rho_R_reference = np.sum(
            reference["bo_state_density_R"][left], axis=1
        )
        rho_R_fused = np.sum(fused["bo_state_density_R"][right], axis=1)
        dq = float(reference["q"][1]-reference["q"][0])
        dR = float(reference["R"][1]-reference["R"][0])
        q_l1 = float(np.max(np.sum(
            np.abs(rho_q_reference-rho_q_fused), axis=1
        )*dq))
        R_l1 = float(np.max(np.sum(
            np.abs(rho_R_reference-rho_R_fused), axis=1
        )*dR))
        electron_l1 = np.nan
        if "electron_density" in reference.files and "electron_density" in fused.files:
            electron_reference = reference["electron_density"][left]
            electron_fused = fused["electron_density"][right]
            dx = float(reference["x"][1]-reference["x"][0])
            electron_l1 = float(np.max(np.sum(
                np.abs(electron_reference-electron_fused), axis=1
            )*dx))
        reference_seconds = float(reference["wall_seconds"])
        fused_seconds = float(fused["wall_seconds"])
        print(f"reference: {reference_path} ({kernel_reference})")
        print(f"fused:     {fused_path} ({kernel_fused})")
        print(f"common frames: {len(pairs)}")
        print(f"max |norm_ref-norm_fused|: {norm_difference:.6e}")
        print(f"max BO population difference: {population_difference:.6e}")
        print(f"max proton-density L1: {q_l1:.6e}")
        print(f"max heavy-density L1: {R_l1:.6e}")
        print(
            "max electron-density L1: "
            +(f"{electron_l1:.6e}" if np.isfinite(electron_l1) else "not saved")
        )
        print(
            f"wall: reference={reference_seconds:.3f} s, "
            f"fused={fused_seconds:.3f} s, "
            f"speedup={reference_seconds/fused_seconds:.3f}x"
        )
        passed = (
            norm_difference <= args.norm_tolerance
            and population_difference <= args.population_tolerance
            and q_l1 <= args.density_tolerance
            and R_l1 <= args.density_tolerance
            and (
                not np.isfinite(electron_l1)
                or electron_l1 <= args.density_tolerance
            )
        )
        print("end-to-end kernel comparison: " + ("PASS" if passed else "FAIL"))
        if not passed:
            raise SystemExit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("fused")
    parser.add_argument("--time-tolerance-fs", type=float, default=1.0e-9)
    parser.add_argument("--norm-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--population-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-8)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
