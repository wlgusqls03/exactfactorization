#!/usr/bin/env python3
"""Compare direct BO-TDSE Y with a discrete-MCEF factor trajectory."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


REFERENCE_KEYS = ("times_fs", "q", "R", "tdse_coefficients")
MCEF_KEYS = (
    "times_fs", "q", "R", "electronic_coefficients",
    "lambda_wavefunction", "chi",
)


def _resolve(value, filename):
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    direct = path/filename
    if direct.is_file():
        return direct
    matches = list(path.glob(f"**/{filename}")) if path.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"{filename}을 찾지 못했습니다: {path}")
    raise RuntimeError(f"비교 archive가 여러 개입니다: {matches}")


@contextmanager
def _extract_arrays(path, keys, root, label):
    directory = Path(root)/label
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = {entry.filename: entry for entry in archive.infolist()}
        missing = [key for key in keys if f"{key}.npy" not in members]
        if missing:
            raise ValueError(f"{path.name} missing keys: {', '.join(missing)}")
        required = sum(members[f"{key}.npy"].file_size for key in keys)
        if shutil.disk_usage(directory).free < required:
            raise OSError(
                f"{label} archive 추출에 {required/1024**3:.2f} GiB 필요"
            )
        print(f"{label} archive 추출: {required/1024**3:.2f} GiB", flush=True)
        for key in keys:
            archive.extract(f"{key}.npy", path=directory)
    arrays = {
        key: np.load(directory/f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in keys
    }
    try:
        yield arrays
    finally:
        arrays.clear()


def compare(reference, mcef, *, tolerance_fs=1.0e-8, tempdir=None,
            output=None, progress_every=10):
    reference = _resolve(reference, "multi_component_discrete_tdse_gpu.npz")
    mcef = _resolve(mcef, "multi_component_born_huang_ef_gpu.npz")
    temp_root = Path(tempdir).resolve() if tempdir else Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="discrete-tdse-compare-", dir=temp_root) as work:
        with _extract_arrays(reference, REFERENCE_KEYS, work, "tdse") as ref:
            with _extract_arrays(mcef, MCEF_KEYS, work, "mcef") as fac:
                for key in ("q", "R"):
                    if ref[key].shape != fac[key].shape or not np.allclose(
                        ref[key], fac[key], rtol=0.0, atol=1.0e-13
                    ):
                        raise ValueError(f"{key} grids differ")
                if ref["tdse_coefficients"].shape[1:] != fac[
                    "electronic_coefficients"
                ].shape[1:]:
                    raise ValueError("BO state count or nuclear grid differs")
                dq = float(ref["q"][1]-ref["q"][0])
                dR = float(ref["R"][1]-ref["R"][0])
                records = {
                    "times_fs": [], "fidelity": [], "joint_density_l1": [],
                    "proton_density_l1": [], "heavy_density_l1": [],
                    "max_bo_population_difference": [],
                    "tdse_norm": [], "mcef_norm": [],
                }
                for ir, current_time in enumerate(ref["times_fs"]):
                    im = int(np.argmin(np.abs(fac["times_fs"]-current_time)))
                    if abs(float(fac["times_fs"][im]-current_time)) > tolerance_fs:
                        continue
                    y_ref = np.asarray(ref["tdse_coefficients"][ir])
                    F = (
                        np.asarray(fac["lambda_wavefunction"][im])
                        *np.asarray(fac["chi"][im])[None, :]
                    )
                    y_mcef = np.asarray(fac["electronic_coefficients"][im])*F[None]
                    norm_ref = float(np.sum(np.abs(y_ref)**2)*dq*dR)
                    norm_mcef = float(np.sum(np.abs(y_mcef)**2)*dq*dR)
                    overlap = np.sum(np.conj(y_ref)*y_mcef)*dq*dR
                    fidelity = float(
                        abs(overlap)**2/max(norm_ref*norm_mcef, 1.0e-300)
                    )
                    rho_ref = np.sum(np.abs(y_ref)**2, axis=0)/max(norm_ref, 1e-300)
                    rho_mcef = np.sum(np.abs(y_mcef)**2, axis=0)/max(norm_mcef, 1e-300)
                    q_ref = np.sum(rho_ref, axis=1)*dR
                    q_mcef = np.sum(rho_mcef, axis=1)*dR
                    R_ref = np.sum(rho_ref, axis=0)*dq
                    R_mcef = np.sum(rho_mcef, axis=0)*dq
                    pop_ref = np.sum(np.abs(y_ref)**2, axis=(1, 2))*dq*dR/norm_ref
                    pop_mcef = np.sum(np.abs(y_mcef)**2, axis=(1, 2))*dq*dR/norm_mcef
                    records["times_fs"].append(float(current_time))
                    records["fidelity"].append(fidelity)
                    records["joint_density_l1"].append(
                        float(np.sum(np.abs(rho_ref-rho_mcef))*dq*dR)
                    )
                    records["proton_density_l1"].append(
                        float(np.sum(np.abs(q_ref-q_mcef))*dq)
                    )
                    records["heavy_density_l1"].append(
                        float(np.sum(np.abs(R_ref-R_mcef))*dR)
                    )
                    records["max_bo_population_difference"].append(
                        float(np.max(np.abs(pop_ref-pop_mcef)))
                    )
                    records["tdse_norm"].append(norm_ref)
                    records["mcef_norm"].append(norm_mcef)
                    if progress_every and len(records["times_fs"]) % progress_every == 0:
                        print(
                            f"compare {len(records['times_fs'])} frames; "
                            f"t={float(current_time):.4f} fs; F={fidelity:.10f}",
                            flush=True,
                        )
    if not records["times_fs"]:
        raise ValueError("허용오차 안에서 공통 저장 시각이 없습니다")
    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays.update(
        reference_path=np.array(str(reference)), mcef_path=np.array(str(mcef))
    )
    if output is None:
        output = mcef.parent/"tdse_mcef_comparison.npz"
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"comparison saved: {output}")
    print(f"common frames: {len(arrays['times_fs'])}")
    print(f"minimum fidelity: {np.min(arrays['fidelity']):.12f}")
    print(f"final fidelity:   {arrays['fidelity'][-1]:.12f}")
    print(f"maximum 1-F:      {np.max(1.0-arrays['fidelity']):.6e}")
    print(f"max joint L1:     {np.max(arrays['joint_density_l1']):.6e}")
    print(f"max proton L1:    {np.max(arrays['proton_density_l1']):.6e}")
    print(f"max heavy L1:     {np.max(arrays['heavy_density_l1']):.6e}")
    print(
        "max BO pop diff:  "
        f"{np.max(arrays['max_bo_population_difference']):.6e}"
    )
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("mcef")
    parser.add_argument("--time-tolerance-fs", type=float, default=1.0e-8)
    parser.add_argument("--tempdir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return compare(
        args.reference, args.mcef,
        tolerance_fs=args.time_tolerance_fs,
        tempdir=args.tempdir, output=args.output,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
