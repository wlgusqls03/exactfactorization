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


class _NpyFrameReader:
    """Read the leading axis of one compressed NPZ member sequentially.

    NumPy's normal NPZ interface materializes an entire compressed member.
    A trajectory member is an ordinary C-order NPY stream inside the ZIP, so
    reading exactly one leading-axis slab at a time gives identical arrays
    without a many-GiB extraction directory.
    """

    def __init__(self, archive, key):
        self.key = key
        self.stream = archive.open(f"{key}.npy", "r")
        version = np.lib.format.read_magic(self.stream)
        # _read_array_header is the version-dispatching implementation used by
        # NumPy's public read_array routine (NPY 1.0/2.0/3.0).
        shape, fortran_order, dtype = np.lib.format._read_array_header(
            self.stream, version
        )
        if not shape:
            raise ValueError(f"{key} must have a leading frame axis")
        if fortran_order:
            raise ValueError(f"streaming comparison requires C-order {key}")
        if dtype.hasobject:
            raise ValueError(f"object dtype is not allowed for {key}")
        self.shape = tuple(shape)
        self.frame_shape = self.shape[1:]
        self.dtype = np.dtype(dtype)
        self.frame_bytes = int(np.prod(self.frame_shape))*self.dtype.itemsize
        self.next_index = 0
        self._discard_buffer = bytearray(min(self.frame_bytes, 8*1024**2))

    def close(self):
        self.stream.close()

    def _read_exact_into(self, destination):
        view = memoryview(destination).cast("B")
        offset = 0
        while offset < len(view):
            count = self.stream.readinto(view[offset:])
            if not count:
                raise EOFError(
                    f"unexpected EOF in {self.key} frame {self.next_index}"
                )
            offset += count

    def _discard_one(self):
        remaining = self.frame_bytes
        view = memoryview(self._discard_buffer)
        while remaining:
            count = self.stream.readinto(view[:min(remaining, len(view))])
            if not count:
                raise EOFError(
                    f"unexpected EOF while skipping {self.key} "
                    f"frame {self.next_index}"
                )
            remaining -= count
        self.next_index += 1

    def read(self, index):
        index = int(index)
        if index < self.next_index:
            raise ValueError(
                f"{self.key} streaming indices must increase: "
                f"{index} < {self.next_index}"
            )
        if index >= self.shape[0]:
            raise IndexError(f"{self.key} frame {index} out of range")
        while self.next_index < index:
            self._discard_one()
        frame = np.empty(self.frame_shape, dtype=self.dtype, order="C")
        self._read_exact_into(frame)
        self.next_index += 1
        return frame


@contextmanager
def _stream_arrays(path, keys):
    archive = zipfile.ZipFile(path)
    readers = {}
    try:
        members = {entry.filename for entry in archive.infolist()}
        missing = [key for key in keys if f"{key}.npy" not in members]
        if missing:
            raise ValueError(f"{path.name} missing keys: {', '.join(missing)}")
        readers = {key: _NpyFrameReader(archive, key) for key in keys}
        yield readers
    finally:
        for reader in readers.values():
            reader.close()
        archive.close()


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


def _common_frame_pairs(reference_times, mcef_times, tolerance_fs):
    """Return monotone frame pairs at the same saved physical time."""
    reference_times = np.asarray(reference_times, dtype=np.float64)
    mcef_times = np.asarray(mcef_times, dtype=np.float64)
    pairs = []
    ir = im = 0
    while ir < len(reference_times) and im < len(mcef_times):
        difference = float(reference_times[ir]-mcef_times[im])
        if abs(difference) <= tolerance_fs:
            pairs.append((ir, im))
            ir += 1
            im += 1
        elif difference < 0.0:
            ir += 1
        else:
            im += 1
    return pairs


def _new_records():
    return {
        "times_fs": [], "fidelity": [], "joint_density_l1": [],
        "proton_density_l1": [], "heavy_density_l1": [],
        "max_bo_population_difference": [],
        "tdse_norm": [], "mcef_norm": [],
    }


def _append_metrics(records, current_time, y_ref, coefficients, lam, chi,
                    dq, dR):
    F = lam*chi[None, :]
    y_mcef = coefficients*F[None]
    norm_ref = float(np.sum(np.abs(y_ref)**2)*dq*dR)
    norm_mcef = float(np.sum(np.abs(y_mcef)**2)*dq*dR)
    overlap = np.sum(np.conj(y_ref)*y_mcef)*dq*dR
    fidelity = float(abs(overlap)**2/max(norm_ref*norm_mcef, 1.0e-300))
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
    return fidelity


def _save_and_report(records, reference, mcef, output):
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
                records = _new_records()
                for ir, current_time in enumerate(ref["times_fs"]):
                    im = int(np.argmin(np.abs(fac["times_fs"]-current_time)))
                    if abs(float(fac["times_fs"][im]-current_time)) > tolerance_fs:
                        continue
                    y_ref = np.asarray(ref["tdse_coefficients"][ir])
                    fidelity = _append_metrics(
                        records, current_time, y_ref,
                        np.asarray(fac["electronic_coefficients"][im]),
                        np.asarray(fac["lambda_wavefunction"][im]),
                        np.asarray(fac["chi"][im]), dq, dR,
                    )
                    if progress_every and len(records["times_fs"]) % progress_every == 0:
                        print(
                            f"compare {len(records['times_fs'])} frames; "
                            f"t={float(current_time):.4f} fs; F={fidelity:.10f}",
                            flush=True,
                        )
    return _save_and_report(records, reference, mcef, output)


def compare_streaming(reference, mcef, *, tolerance_fs=1.0e-8, output=None,
                      progress_every=10):
    """Compare full dynamics with O(one frame) RAM and no extracted arrays."""
    reference = _resolve(reference, "multi_component_discrete_tdse_gpu.npz")
    mcef = _resolve(mcef, "multi_component_born_huang_ef_gpu.npz")
    with np.load(reference, allow_pickle=False) as archive:
        reference_times = np.asarray(archive["times_fs"], dtype=np.float64)
        q_reference = np.asarray(archive["q"], dtype=np.float64)
        R_reference = np.asarray(archive["R"], dtype=np.float64)
    with np.load(mcef, allow_pickle=False) as archive:
        mcef_times = np.asarray(archive["times_fs"], dtype=np.float64)
        q_mcef = np.asarray(archive["q"], dtype=np.float64)
        R_mcef = np.asarray(archive["R"], dtype=np.float64)
    for name, reference_grid, mcef_grid in (
        ("q", q_reference, q_mcef), ("R", R_reference, R_mcef),
    ):
        if reference_grid.shape != mcef_grid.shape or not np.allclose(
            reference_grid, mcef_grid, rtol=0.0, atol=1.0e-13
        ):
            raise ValueError(f"{name} grids differ")
    pairs = _common_frame_pairs(
        reference_times, mcef_times, tolerance_fs
    )
    if not pairs:
        raise ValueError("허용오차 안에서 공통 저장 시각이 없습니다")
    dq = float(q_reference[1]-q_reference[0])
    dR = float(R_reference[1]-R_reference[0])
    records = _new_records()
    print(
        "low-disk streaming comparison: compressed NPZ를 순차 해제하며 "
        "공통 frame 하나씩만 RAM에 유지합니다.", flush=True,
    )
    with _stream_arrays(reference, ("tdse_coefficients",)) as ref:
        with _stream_arrays(
            mcef,
            ("electronic_coefficients", "lambda_wavefunction", "chi"),
        ) as fac:
            if ref["tdse_coefficients"].frame_shape != fac[
                "electronic_coefficients"
            ].frame_shape:
                raise ValueError("BO state count or nuclear grid differs")
            expected_lam_shape = ref["tdse_coefficients"].frame_shape[1:]
            expected_chi_shape = (ref["tdse_coefficients"].frame_shape[2],)
            if fac["lambda_wavefunction"].frame_shape != expected_lam_shape:
                raise ValueError("Lambda shape differs from TDSE nuclear grid")
            if fac["chi"].frame_shape != expected_chi_shape:
                raise ValueError("chi shape differs from TDSE nuclear grid")
            for ir, im in pairs:
                y_ref = ref["tdse_coefficients"].read(ir)
                coefficients = fac["electronic_coefficients"].read(im)
                lam = fac["lambda_wavefunction"].read(im)
                chi = fac["chi"].read(im)
                fidelity = _append_metrics(
                    records, reference_times[ir], y_ref,
                    coefficients, lam, chi, dq, dR,
                )
                if progress_every and len(records["times_fs"]) % progress_every == 0:
                    print(
                        f"compare {len(records['times_fs'])}/{len(pairs)} frames; "
                        f"t={reference_times[ir]:.4f} fs; F={fidelity:.10f}",
                        flush=True,
                    )
    return _save_and_report(records, reference, mcef, output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("mcef")
    parser.add_argument("--time-tolerance-fs", type=float, default=1.0e-8)
    parser.add_argument("--tempdir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--low-disk", action="store_true",
        help=(
            "NPZ를 frame-by-frame 순차 해제하여 임시 추출 공간 없이 비교; "
            "RAM은 frame 약 2개만 사용"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.low_disk:
        return compare_streaming(
            args.reference, args.mcef,
            tolerance_fs=args.time_tolerance_fs,
            output=args.output, progress_every=args.progress_every,
        )
    return compare(
        args.reference, args.mcef,
        tolerance_fs=args.time_tolerance_fs,
        tempdir=args.tempdir, output=args.output,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
