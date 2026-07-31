#!/usr/bin/env python3
"""Compare physical observables from two completed MCEF analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def resolve_observables(value):
    """Accept a dynamics NPZ directly or a completed run directory."""
    path = Path(value).expanduser()
    if path.is_file():
        return path
    direct = path/"dynamics_analysis"/"dynamics_observables.npz"
    if direct.is_file():
        return direct
    matches = list(path.glob("**/dynamics_observables.npz")) if path.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"dynamics_observables.npz를 찾지 못했습니다: {path}")
    raise RuntimeError(f"비교 파일이 여러 개입니다: {matches}")


def common_frame_indices(reference_times, candidate_times, tolerance):
    """Match saved frames by physical time rather than array index."""
    pairs = []
    for i, time in enumerate(reference_times):
        j = int(np.argmin(np.abs(candidate_times-time)))
        if abs(float(candidate_times[j]-time)) <= tolerance:
            pairs.append((i, j))
    if not pairs:
        raise ValueError("허용오차 안에서 일치하는 저장 시간이 없습니다.")
    return np.asarray(pairs, dtype=int)


def _spacing(data, grid_key, density_key):
    if grid_key in data.files:
        grid = data[grid_key]
        return float(grid[1]-grid[0])
    # 이전 analysis archive에는 grid가 없었다. 정규화된 첫 density에서 복원한다.
    return 1.0/float(np.sum(data[density_key][0]))


def compare(reference, candidate, time_tolerance_fs=1.0e-9):
    """Return maximum discrepancies at common saved physical times."""
    ref = np.load(resolve_observables(reference))
    test = np.load(resolve_observables(candidate))
    pairs = common_frame_indices(
        ref["times_fs"], test["times_fs"], time_tolerance_fs
    )
    ir, it = pairs[:, 0], pairs[:, 1]
    results = {"common_frames": len(pairs)}
    for key in (
        "electron_mean", "proton_mean", "heavy_mean",
        "electron_width", "proton_width", "heavy_width",
        "electron_left_population", "electron_right_population",
        "electron_rearranged_density", "state_basis_residual",
    ):
        if key in ref.files and key in test.files:
            results[f"max_abs_{key}"] = float(np.max(np.abs(ref[key][ir]-test[key][it])))

    for grid_key, density_key in (
        ("x", "electron_density"), ("q", "proton_density"),
        ("R", "heavy_density"),
    ):
        if ref[density_key].shape[1:] != test[density_key].shape[1:]:
            results[f"max_l1_{density_key}"] = np.nan
            continue
        spacing = _spacing(ref, grid_key, density_key)
        l1 = np.sum(
            np.abs(ref[density_key][ir]-test[density_key][it]), axis=1
        )*spacing
        results[f"max_l1_{density_key}"] = float(np.max(l1))

    if "state_populations" in ref.files and "state_populations" in test.files:
        states = min(ref["state_populations"].shape[1], test["state_populations"].shape[1])
        difference = np.abs(
            ref["state_populations"][ir, :states]
            -test["state_populations"][it, :states]
        )
        for state in range(states):
            results[f"max_abs_population_{state}"] = float(np.max(difference[:, state]))
    return results


def run(args):
    results = compare(args.reference, args.candidate, args.time_tolerance_fs)
    print(f"공통 저장 frame: {results['common_frames']}")
    for key, value in results.items():
        if key == "common_frames":
            continue
        rendered = "grid shape 다름" if np.isnan(value) else f"{value:.6e}"
        print(f"{key}: {rendered}")
    print("norm 보존과 별개로 density, mean/width, BO population이 함께 수렴해야 합니다.")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--time-tolerance-fs", type=float, default=1.0e-9)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
