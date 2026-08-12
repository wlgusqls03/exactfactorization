#!/usr/bin/env python3
"""Print a compact stability audit for a completed discrete-MCEF run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from multi_component_exact_factorization.render_all import (
    find_archive,
    resolve_run_input,
)


def _series(archive, name):
    if name not in archive.files:
        return None
    value = np.asarray(archive[name], float)
    return value if value.ndim == 1 else None


def _finite_max_final(values):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan")
    final = values[-1] if np.isfinite(values[-1]) else finite[-1]
    return float(np.max(np.abs(finite))), float(final)


def _show(archive, name, label=None):
    values = _series(archive, name)
    if values is None:
        print(f"  {(label or name):42s}: not saved")
        return None
    maximum, final = _finite_max_final(values)
    print(f"  {(label or name):42s}: max={maximum:.6e}, final={final:.6e}")
    return maximum


def run(path):
    resolved = resolve_run_input(path)
    archive_path, _ = find_archive(resolved)
    with np.load(archive_path, allow_pickle=False) as archive:
        kind = str(np.asarray(archive.get("kind", "")).item())
        if not kind.startswith("discrete_born_huang"):
            raise ValueError(f"discrete MCEF archive가 아닙니다: {kind}")
        times = np.asarray(archive["times_fs"], float)
        completed = bool(np.asarray(archive["propagation_completed"]).item())
        requested = float(np.asarray(archive["requested_final_time_fs"]).item())
        reason = str(np.asarray(archive["failure_reason"]).item())
        print(f"archive: {archive_path}")
        print(
            f"status: {'COMPLETED' if completed else 'FAILED'}; "
            f"frames={len(times)}; reached={times[-1]:.9f}/{requested:g} fs"
        )
        if reason:
            print(f"failure: {reason}")
        norm = np.asarray(archive["norm"], float)
        norm_error = np.abs(norm-1.0)
        print("\n[physical/constraint]")
        print(
            f"  {'full norm error':42s}: max={np.max(norm_error):.6e}, "
            f"final={norm_error[-1]:.6e}"
        )
        pnc = _show(archive, "pnc_error", "saved PNC error")
        pnc_change = _show(
            archive, "pnc_product_change_l2", "PNC full-product change L2"
        )
        outer_q = _show(archive, "outer_probability_q", "q outer probability")
        outer_R = _show(archive, "outer_probability_R", "R outer probability")
        _show(
            archive, "fixed_center_crossing_q",
            "q probability beyond fixed centers",
        )
        _show(
            archive, "fixed_center_crossing_R",
            "R probability beyond fixed centers",
        )
        _show(archive, "fixed_center_crossing_q_left", "q crossing left center")
        _show(archive, "fixed_center_crossing_q_right", "q crossing right center")
        _show(archive, "fixed_center_crossing_R_left", "R crossing left center")
        _show(archive, "fixed_center_crossing_R_right", "R crossing right center")

        print("\n[spatial discrete identity]")
        spatial = _show(
            archive, "relative_unexplained_residual",
            "unexplained recombination / H_hY",
        )
        _show(archive, "recombination_residual_l2", "total recombination residual L2")
        _show(archive, "predicted_mask_residual_l2", "predicted mask residual L2")
        _show(archive, "epsilon_1_imaginary_defect", "Im E1 defect")
        _show(archive, "epsilon_2_imaginary_defect", "Im E2 defect")

        print("\n[time integration]")
        temporal = _show(
            archive, "rk_product_local_defect_relative",
            "RK4 local product defect relative",
        )
        _show(archive, "rk_product_local_defect_l2", "RK4 local product defect L2")
        _show(archive, "full_norm_rate", "instantaneous full-norm rate")

        print("\n[regularization/geometry]")
        _show(archive, "suppressed_probability_phi", "suppressed probability F")
        _show(archive, "suppressed_probability_lam", "suppressed probability chi")
        _show(archive, "max_abs_regularized_F_ratio", "max regularized F-neighbor ratio")
        _show(archive, "max_abs_regularized_chi_ratio", "max regularized chi-neighbor ratio")
        _show(archive, "weighted_link_defect_phi_q", "weighted q-link magnitude defect")
        _show(archive, "weighted_link_defect_phi_R", "weighted R-link magnitude defect")
        _show(archive, "weighted_link_defect_gamma_R", "weighted outer-link magnitude defect")

        passed = completed and np.all(np.isfinite(norm))
        passed &= spatial is not None and np.isfinite(spatial) and spatial < 1.0e-9
        passed &= pnc_change is not None and np.isfinite(pnc_change) and pnc_change < 1.0e-9
        if pnc is not None:
            passed &= np.isfinite(pnc) and pnc < 1.0e-5
        if outer_q is not None and outer_R is not None:
            passed &= max(outer_q, outer_R) < 1.0e-4
        print("\nassessment: " + ("STRUCTURAL CHECKS PASS" if passed else "REVIEW REQUIRED"))
        if temporal is not None:
            print(
                "note: RK4 temporal defect는 dt convergence 값입니다. "
                "dt를 절반으로 했을 때 강하게 감소하는지 함께 비교하세요."
            )
    return passed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run folder 또는 discrete MCEF NPZ")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if not run(args.run):
        raise SystemExit(2)
