#!/usr/bin/env python3
"""Low-memory support-resolved PNC/factor-norm audit for BO MCEF runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from multi_component_exact_factorization.render_all import resolve_run_input


ARCHIVE_NAME = "multi_component_born_huang_ef_gpu.npz"
LARGE_KEYS = ("electronic_coefficients", "lambda_wavefunction", "chi")


def _resolve(value):
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    path = resolve_run_input(path)
    direct = path/ARCHIVE_NAME
    if direct.is_file():
        return direct.resolve()
    matches = list(path.glob(f"**/{ARCHIVE_NAME}")) if path.is_dir() else []
    if len(matches) != 1:
        raise RuntimeError(f"BO MCEF archive를 하나로 결정할 수 없습니다: {matches}")
    return matches[0].resolve()


def _finite_peak_time(times, values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).any():
        return float("nan"), float("nan")
    index = int(np.nanargmax(np.abs(values)))
    return float(values[index]), float(times[index])


def _finite_min_time(times, values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).any():
        return float("nan"), float("nan")
    index = int(np.nanargmin(values))
    return float(values[index]), float(times[index])


def _region_stats(norm, mask):
    values = np.asarray(norm[mask], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    safe = np.maximum(values, np.finfo(np.float64).tiny)
    return (
        float(np.max(values)),
        float(np.min(values)),
        float(np.quantile(values, 0.99)),
        float(np.quantile(values, 0.01)),
        float(np.max(np.abs(values*values-1.0))),
        float(np.max(np.abs(np.log10(safe)))),
    )


def _append_region(records, prefix, norm, mask):
    maximum, minimum, percentile99, percentile01, pnc_error, log_excursion = (
        _region_stats(norm, mask)
    )
    records[f"max_{prefix}_norm"].append(maximum)
    records[f"min_{prefix}_norm"].append(minimum)
    records[f"p99_{prefix}_norm"].append(percentile99)
    records[f"p01_{prefix}_norm"].append(percentile01)
    records[f"max_{prefix}_pnc_error"].append(pnc_error)
    records[f"max_abs_log10_{prefix}_norm"].append(log_excursion)


def _extract_large_arrays(path, root):
    directory = Path(root)/"factor_arrays"
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = {entry.filename: entry for entry in archive.infolist()}
        missing = [key for key in LARGE_KEYS if f"{key}.npy" not in members]
        if missing:
            raise ValueError(
                "BO factor archive가 아니거나 필요한 배열이 없습니다: "
                +", ".join(missing)
            )
        required = sum(members[f"{key}.npy"].file_size for key in LARGE_KEYS)
        free = shutil.disk_usage(directory).free
        if free < required:
            raise OSError(
                f"임시 추출에 {required/1024**3:.2f} GiB가 필요하지만 "
                f"{free/1024**3:.2f} GiB만 비어 있습니다"
            )
        print(
            f"factor 배열 임시 추출: {required/1024**3:.2f} GiB "
            f"-> {directory}", flush=True,
        )
        for key in LARGE_KEYS:
            archive.extract(f"{key}.npy", path=directory)
    return {
        key: np.load(directory/f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in LARGE_KEYS
    }


def diagnose(value, *, tempdir=None, output=None, progress_every=10,
             threshold=None):
    path = _resolve(value)
    with np.load(path, allow_pickle=False) as archive:
        times = np.asarray(archive["times_fs"], dtype=np.float64)
        q = np.asarray(archive["q"], dtype=np.float64)
        R = np.asarray(archive["R"], dtype=np.float64)
        stored_threshold = float(np.asarray(
            archive.get("deep_tail_zero_threshold", 1.0e-12)
        ).item())
    threshold = stored_threshold if threshold is None else float(threshold)
    if threshold <= 0.0:
        raise ValueError("support 진단 threshold는 양수여야 합니다")
    lower = threshold/10.0
    upper = threshold*10.0
    dq = float(q[1]-q[0])
    dR = float(R[1]-R[0])
    record_names = []
    for factor in ("c", "lambda"):
        for region in ("tail", "transition", "support"):
            prefix = f"{region}_{factor}"
            record_names.extend((
                f"max_{prefix}_norm", f"min_{prefix}_norm",
                f"p99_{prefix}_norm", f"p01_{prefix}_norm",
                f"max_{prefix}_pnc_error",
                f"max_abs_log10_{prefix}_norm",
            ))
    records = {name: [] for name in record_names}
    records.update(
        times_fs=[], tail_probability_phi=[], tail_probability_lam=[],
        transition_probability_phi=[], transition_probability_lam=[],
        tail_fraction_phi=[], tail_fraction_lam=[],
    )
    temp_root = Path(tempdir).expanduser().resolve() if tempdir else path.parent
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="factor-norm-audit-", dir=temp_root) as work:
        arrays = _extract_large_arrays(path, work)
        c_all = arrays["electronic_coefficients"]
        lam_all = arrays["lambda_wavefunction"]
        chi_all = arrays["chi"]
        if not (len(c_all) == len(lam_all) == len(chi_all) == len(times)):
            raise ValueError("factor history와 times_fs frame 수가 다릅니다")
        for iframe, current_time in enumerate(times):
            c = np.asarray(c_all[iframe])
            lam = np.asarray(lam_all[iframe])
            chi = np.asarray(chi_all[iframe])
            c_norm2 = np.sum(np.abs(c)**2, axis=0, dtype=np.float64)
            c_norm = np.sqrt(c_norm2)
            F_density = np.abs(lam*chi[None, :])**2
            rho_qR = c_norm2*F_density
            rho_peak = max(float(np.max(rho_qR)), np.finfo(np.float64).tiny)
            relative_phi = rho_qR/rho_peak
            rho_R = np.sum(rho_qR, axis=0, dtype=np.float64)*dq
            rho_R_peak = max(float(np.max(rho_R)), np.finfo(np.float64).tiny)
            relative_lam = rho_R/rho_R_peak
            lam_norm2 = np.sum(np.abs(lam)**2, axis=0, dtype=np.float64)*dq
            lam_norm = np.sqrt(lam_norm2)
            regions_phi = {
                "tail": relative_phi <= lower,
                "transition": (relative_phi > lower) & (relative_phi < upper),
                "support": relative_phi >= upper,
            }
            regions_lam = {
                "tail": relative_lam <= lower,
                "transition": (relative_lam > lower) & (relative_lam < upper),
                "support": relative_lam >= upper,
            }
            for region, mask in regions_phi.items():
                _append_region(records, f"{region}_c", c_norm, mask)
            for region, mask in regions_lam.items():
                _append_region(records, f"{region}_lambda", lam_norm, mask)
            total = max(float(np.sum(rho_qR)*dq*dR), np.finfo(np.float64).tiny)
            heavy_total = max(float(np.sum(rho_R)*dR), np.finfo(np.float64).tiny)
            records["times_fs"].append(float(current_time))
            records["tail_probability_phi"].append(
                float(np.sum(rho_qR[regions_phi["tail"]])*dq*dR/total)
            )
            records["transition_probability_phi"].append(
                float(np.sum(rho_qR[regions_phi["transition"]])*dq*dR/total)
            )
            records["tail_probability_lam"].append(
                float(np.sum(rho_R[regions_lam["tail"]])*dR/heavy_total)
            )
            records["transition_probability_lam"].append(
                float(np.sum(rho_R[regions_lam["transition"]])*dR/heavy_total)
            )
            records["tail_fraction_phi"].append(float(np.mean(regions_phi["tail"])))
            records["tail_fraction_lam"].append(float(np.mean(regions_lam["tail"])))
            if progress_every and ((iframe+1) % progress_every == 0 or iframe+1 == len(times)):
                print(
                    f"factor audit {iframe+1}/{len(times)}; "
                    f"t={current_time:.4f} fs; "
                    f"tail max ||C||={records['max_tail_c_norm'][-1]:.3e}; "
                    f"support PNC(C)={records['max_support_c_pnc_error'][-1]:.3e}",
                    flush=True,
                )
        arrays.clear()
    result = {key: np.asarray(values) for key, values in records.items()}
    # Newer runs save exact pre-retraction reductions.  They are small arrays,
    # so copy them into the audit without touching the large factor tensors.
    with np.load(path, allow_pickle=False) as archive:
        pre_keys = (
            "max_raw_pnc_phi_error", "max_raw_pnc_lam_error",
            *(
                key for factor in ("c", "lam") for key in (
                    f"max_pre_pnc_{factor}_norm",
                    f"max_inverse_pre_pnc_{factor}_norm",
                    f"max_pre_pnc_tail_{factor}_norm",
                    f"max_inverse_pre_pnc_tail_{factor}_norm",
                    f"max_pre_pnc_support_{factor}_norm",
                    f"max_inverse_pre_pnc_support_{factor}_norm",
                    *(
                        item
                        for threshold_name in (
                            "lt_1e_4", "lt_1e_2", "lt_1e_1",
                            "gt_1e1", "gt_1e2", "gt_1e4",
                        )
                        for item in (
                            f"count_pre_pnc_{factor}_norm_{threshold_name}",
                            f"fraction_pre_pnc_{factor}_norm_{threshold_name}",
                        )
                    ),
                )
            ),
        )
        for key in pre_keys:
            if key in archive.files:
                result[key] = np.asarray(archive[key])
    result.update(
        source_archive=np.array(str(path)),
        deep_tail_zero_threshold=np.array(threshold),
        tail_relative_upper=np.array(lower),
        support_relative_lower=np.array(upper),
    )
    if output is None:
        output = path.parent/"factor_norm_support_audit.npz"
    output = Path(output).expanduser().resolve()
    np.savez_compressed(output, **result)
    print(f"audit saved: {output}")
    print(
        f"regions: tail r<={lower:.3e}; transition {lower:.3e}<r<{upper:.3e}; "
        f"support r>={upper:.3e}"
    )
    for key, label in (
        ("max_tail_c_norm", "max tail ||C||"),
        ("max_abs_log10_tail_c_norm", "max tail |log10||C|||"),
        ("max_support_c_pnc_error", "max occupied C PNC error"),
        ("max_tail_lambda_norm", "max tail ||Lambda||"),
        ("max_abs_log10_tail_lambda_norm", "max tail |log10||Lambda|||"),
        ("max_support_lambda_pnc_error", "max occupied Lambda PNC error"),
        ("tail_probability_phi", "max physical mass in C-gate-off tail"),
        ("tail_probability_lam", "max physical mass in Lambda-gate-off tail"),
    ):
        peak, peak_time = _finite_peak_time(times, result[key])
        print(f"  {label:43s}: {peak:.6e} at {peak_time:.6f} fs")
    for key, label in (
        ("min_tail_c_norm", "min tail ||C||"),
        ("min_support_c_norm", "min occupied ||C||"),
        ("min_tail_lambda_norm", "min tail ||Lambda||"),
        ("min_support_lambda_norm", "min occupied ||Lambda||"),
    ):
        minimum, minimum_time = _finite_min_time(times, result[key])
        print(f"  {label:43s}: {minimum:.6e} at {minimum_time:.6f} fs")
    print("[pre-retraction diagnostics stored by propagation]")
    for factor, label in (("c", "C"), ("lam", "Lambda")):
        max_key = f"max_pre_pnc_{factor}_norm"
        inverse_key = f"max_inverse_pre_pnc_{factor}_norm"
        if max_key not in result or inverse_key not in result:
            fallback = "max_raw_pnc_phi_error" if factor == "c" else "max_raw_pnc_lam_error"
            if fallback in result:
                peak, peak_time = _finite_peak_time(times, result[fallback])
                print(
                    f"  {label}: exact min/max not saved; max raw PNC error="
                    f"{peak:.6e} at {peak_time:.6f} fs"
                )
            else:
                print(f"  {label}: not saved; exact pre-retraction values require a newer run")
            continue
        maximum, maximum_time = _finite_peak_time(times, result[max_key])
        inverse_peak, minimum_time = _finite_peak_time(times, result[inverse_key])
        minimum = 1.0/inverse_peak if inverse_peak > 0.0 else np.nan
        print(
            f"  {label}: global max={maximum:.6e} at {maximum_time:.6f} fs; "
            f"global min={minimum:.6e} at {minimum_time:.6f} fs"
        )
        for threshold_name in (
            "lt_1e_4", "lt_1e_2", "lt_1e_1",
            "gt_1e1", "gt_1e2", "gt_1e4",
        ):
            count_key = f"count_pre_pnc_{factor}_norm_{threshold_name}"
            fraction_key = f"fraction_pre_pnc_{factor}_norm_{threshold_name}"
            count, count_time = _finite_peak_time(times, result[count_key])
            fraction, _ = _finite_peak_time(times, result[fraction_key])
            print(
                f"    {threshold_name:8s}: max count={int(round(count))}, "
                f"fraction={fraction:.6e} at {count_time:.6f} fs"
            )
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="BO MCEF run directory or NPZ archive")
    parser.add_argument(
        "--tempdir", default=None,
        help="large arrays를 잠시 풀 충분한 여유 공간; 생략하면 run 폴더",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--deep-tail-zero-threshold", type=float, default=None,
        help="생략하면 archive에 저장된 실제 propagation 값을 사용",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return diagnose(
        args.run, tempdir=args.tempdir, output=args.output,
        progress_every=args.progress_every,
        threshold=args.deep_tail_zero_threshold,
    )


if __name__ == "__main__":
    main()
