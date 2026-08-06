#!/usr/bin/env python3
"""Full-TDSE reference와 direct nested EF의 gauge-invariant 결과 비교."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


ARCHIVE_BASE_KEYS = ("x", "q", "R", "times_fs")


def _array_keys(member_names):
    """비교에 필요한 최소 archive key 목록."""
    if "psi.npy" in member_names:
        return (*ARCHIVE_BASE_KEYS, "psi")
    if "electronic_coefficients.npy" in member_names:
        required = (
            "electronic_coefficients", "bo_basis_states",
            "lambda_wavefunction", "chi",
        )
        missing = [key for key in required if f"{key}.npy" not in member_names]
        if missing:
            raise ValueError(
                "Born--Huang 비교에는 --bo-save-basis-states로 저장한 "
                f"archive가 필요합니다: {', '.join(missing)}"
            )
        return (*ARCHIVE_BASE_KEYS, *required)
    required = ("phi", "lambda_wavefunction", "chi")
    missing = [key for key in required if f"{key}.npy" not in member_names]
    if missing:
        raise ValueError(f"archive에 필요한 배열이 없습니다: {', '.join(missing)}")
    return (*ARCHIVE_BASE_KEYS, *required)


@contextmanager
def open_archive_arrays(path, *, in_memory=False, tempdir=None):
    """NPZ를 한 번만 풀어 배열 dict를 반환한다.

    기본 모드는 압축 member를 임시 ``.npy``로 한 번 추출하고 mmap한다.
    따라서 frame마다 거대한 ``phi.npy``를 다시 inflate하지 않는다.
    """
    path = Path(path).resolve()
    if in_memory:
        with np.load(path, allow_pickle=False) as archive:
            keys = _array_keys({f"{key}.npy" for key in archive.files})
            yield {key: np.asarray(archive[key]) for key in keys}
        return

    with zipfile.ZipFile(path) as archive:
        members = {info.filename: info for info in archive.infolist()}
        keys = _array_keys(set(members))
        required_bytes = sum(members[f"{key}.npy"].file_size for key in keys)
        root = Path(tempdir).resolve() if tempdir else Path(tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < required_bytes:
            raise OSError(
                f"{path.name} 추출에 {required_bytes/1024**3:.2f} GiB가 필요하지만 "
                f"{root}의 여유 공간이 부족합니다. --tempdir를 지정하세요."
            )
        with tempfile.TemporaryDirectory(prefix="mcef-compare-", dir=root) as work:
            work = Path(work)
            print(
                f"archive 한 번 추출: {path} "
                f"({required_bytes/1024**3:.2f} GiB)",
                flush=True,
            )
            for key in keys:
                archive.extract(f"{key}.npy", path=work)
            yield {
                key: np.load(work/f"{key}.npy", mmap_mode="r", allow_pickle=False)
                for key in keys
            }


def psi_from_archive(data, frame):
    """저장된 Psi가 없으면 세 factor의 곱으로 재구성한다."""
    if "psi" in data:
        return data["psi"][frame]
    if "electronic_coefficients" in data:
        phi = np.einsum(
            "jqR,jxqR->xqR",
            data["electronic_coefficients"][frame],
            data["bo_basis_states"], optimize=True,
        )
        return (
            phi*data["lambda_wavefunction"][frame][None, :, :]
            *data["chi"][frame][None, None, :]
        )
    return (
        data["phi"][frame]
        *data["lambda_wavefunction"][frame][None, :, :]
        *data["chi"][frame][None, None, :]
    )


def run(args):
    with ExitStack() as stack:
        ref = stack.enter_context(open_archive_arrays(
            args.reference, in_memory=getattr(args, "in_memory", False),
            tempdir=getattr(args, "tempdir", None),
        ))
        direct = stack.enter_context(open_archive_arrays(
            args.direct, in_memory=getattr(args, "in_memory", False),
            tempdir=getattr(args, "tempdir", None),
        ))
        return compare_arrays(ref, direct, args)


def compare_arrays(ref, direct, args):
    """이미 연 배열을 frame 단위로 비교한다."""
    for key in ("x", "q", "R"):
        if ref[key].shape != direct[key].shape or not np.allclose(
            ref[key], direct[key]
        ):
            raise ValueError(f"{key} 격자가 서로 다릅니다.")

    dx = float(ref["x"][1]-ref["x"][0])
    dq = float(ref["q"][1]-ref["q"][0])
    dR = float(ref["R"][1]-ref["R"][0])
    dv = dx*dq*dR
    fidelities, heavy_l1, proton_heavy_l1, electron_heavy_l1 = [], [], [], []

    total_direct_frames = len(direct["times_fs"])
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
        progress_every = getattr(args, "progress_every", 0)
        if progress_every and (
            len(fidelities) % progress_every == 0
            or it == total_direct_frames-1
        ):
            print(
                f"비교 진행: direct frame {it+1}/{total_direct_frames}, "
                f"공통 frame {len(fidelities)}",
                flush=True,
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
    parser.add_argument(
        "--in-memory", action="store_true",
        help="충분한 RAM이 있을 때 필요한 archive 배열을 한 번에 메모리에 적재",
    )
    parser.add_argument(
        "--tempdir", default=None,
        help="기본 mmap 모드에서 압축 배열을 한 번 풀 임시 디렉터리",
    )
    parser.add_argument(
        "--progress-every", type=int, default=10,
        help="비교 진행을 출력할 공통 frame 간격; 0이면 진행 출력 생략",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
