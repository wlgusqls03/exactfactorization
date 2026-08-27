#!/usr/bin/env python3
"""Render only selected standalone-TDSE movies without touching other reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from .render_all import find_archive, resolve_run_input
from .tdse_report import (
    _load_ef_fields,
    calculate_observables,
    load_observables,
    make_bo_surface_dynamics_animation,
    make_dynamics_animation,
)


def run(args):
    archive, _ = find_archive(resolve_run_input(args.run))
    obs = calculate_observables(load_observables(archive))
    ef = _load_ef_fields(obs)
    if (
        obs.get("electron_density") is None
        and ef is not None
        and "electron_density" in ef
    ):
        obs["electron_density"] = ef["electron_density"]
        obs["x"] = ef.get("x", obs.get("x"))

    output = Path(args.outdir) if args.outdir else archive.parent/"report"
    output.mkdir(parents=True, exist_ok=True)
    if args.overview:
        make_dynamics_animation(
            obs, output, args.fps, args.max_frames,
            args.animation_dpi, args.format,
        )
    if args.bo_surface:
        if ef is None:
            raise FileNotFoundError(
                "tdse_exact_factorization_fields.npz가 없습니다. "
                "postprocess_tdse_ef를 먼저 실행하세요."
            )
        make_bo_surface_dynamics_animation(
            obs, ef, output, args.fps, args.max_frames,
            args.animation_dpi, args.format,
            surface_count=args.surface_count,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="TDSE 계산 폴더 또는 archive")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--animation-dpi", type=int, default=110)
    parser.add_argument("--surface-count", type=int, default=5)
    parser.set_defaults(overview=True, bo_surface=True)
    parser.add_argument(
        "--no-overview", action="store_false", dest="overview",
        help="overview 영상은 만들지 않음",
    )
    parser.add_argument(
        "--no-bo-surface", action="store_false", dest="bo_surface",
        help="BO-surface/channel-packet 영상은 만들지 않음",
    )
    args = parser.parse_args(argv)
    if not args.overview and not args.bo_surface:
        parser.error("적어도 한 영상은 선택해야 합니다")
    if args.fps <= 0 or args.max_frames <= 0 or args.animation_dpi <= 0:
        parser.error("fps/max-frames/animation-dpi는 양수여야 합니다")
    if args.surface_count <= 0:
        parser.error("--surface-count는 양수여야 합니다")
    return args


if __name__ == "__main__":
    run(parse_args())
