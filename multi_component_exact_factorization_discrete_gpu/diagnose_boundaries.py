#!/usr/bin/env python3
"""Audit fixed-center crossing and numerical-boundary probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from multi_component_exact_factorization.render_all import resolve_run_input


ARCHIVE_NAMES = (
    "multi_component_discrete_tdse_gpu.npz",
    "multi_component_born_huang_ef_gpu.npz",
    "multi_component_direct_ef_gpu.npz",
)
THRESHOLDS = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3)


def _resolve(value):
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    path = resolve_run_input(path)
    matches = [path/name for name in ARCHIVE_NAMES if (path/name).is_file()]
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        recursive = []
        for name in ARCHIVE_NAMES:
            recursive.extend(path.glob(f"**/{name}"))
        matches = recursive
    if len(matches) != 1:
        raise RuntimeError(f"archive를 하나로 결정할 수 없습니다: {matches}")
    return matches[0].resolve()


def _show_thresholds(times, values):
    for threshold in THRESHOLDS:
        indices = np.flatnonzero(values >= threshold)
        reached = f"{times[indices[0]]:.6f} fs" if indices.size else "not reached"
        print(f"      first >= {threshold:.0e}: {reached}")


def _show_series(archive, times, key, label):
    if key not in archive.files:
        print(f"  {label}: not saved (rerun with the newer code)")
        return
    values = np.asarray(archive[key], dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        print(f"  {label}: no finite samples")
        return
    peak = int(np.nanargmax(values))
    print(
        f"  {label}: max={values[peak]:.6e} at {times[peak]:.6f} fs; "
        f"final={values[-1]:.6e}"
    )
    _show_thresholds(times, values)


def run(value):
    path = _resolve(value)
    with np.load(path, allow_pickle=False) as archive:
        times = np.asarray(archive["times_fs"], dtype=np.float64)
        print(f"archive: {path}")
        print(f"frames={len(times)}; final={times[-1]:.9f} fs")
        print("[fixed-center crossing: outside X_L..X_R]")
        _show_series(archive, times, "fixed_center_crossing_q", "q total")
        _show_series(archive, times, "fixed_center_crossing_q_left", "q left")
        _show_series(archive, times, "fixed_center_crossing_q_right", "q right")
        _show_series(archive, times, "fixed_center_crossing_R", "R total")
        _show_series(archive, times, "fixed_center_crossing_R_left", "R left")
        _show_series(archive, times, "fixed_center_crossing_R_right", "R right")
        print("[numerical boundary: outer five grid cells]")
        _show_series(archive, times, "outer_probability_q", "q PBC edge")
        _show_series(archive, times, "outer_probability_R", "R PBC edge")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directory or NPZ archive")
    args = parser.parse_args(argv)
    for index, value in enumerate(args.runs):
        if index:
            print("\n" + "="*79)
        run(value)


if __name__ == "__main__":
    main()
