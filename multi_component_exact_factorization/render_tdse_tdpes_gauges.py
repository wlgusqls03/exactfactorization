#!/usr/bin/env python3
"""Regenerate only the dual-gauge TDSE TDPES GI/GD figure and movie."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from .render_all import find_archive, resolve_run_input
from .tdse_report import (
    _gauge_directory_name,
    _load_ef_fields,
    calculate_observables,
    load_observables,
    make_tdpes_decomposition_animation,
    plot_tdpes_decomposition,
    transform_to_zero_potential_gauge,
)


def _selected_gauges(value):
    return ("positive", "zero") if value == "both" else (value,)


def run(args):
    archive, run_dir = find_archive(resolve_run_input(args.run))
    obs = calculate_observables(load_observables(archive))
    cache = run_dir/"tdse_exact_factorization_fields.npz"
    if not cache.is_file():
        raise FileNotFoundError(
            f"{cache}가 없습니다. postprocess_tdse_ef를 먼저 실행하세요."
        )

    output_root = Path(args.outdir) if args.outdir else run_dir/"report"
    products = []
    for gauge in _selected_gauges(args.gauge):
        # Load each gauge independently and release it before the next one so
        # large production field caches do not coexist in RAM.
        ef = _load_ef_fields(obs)
        if ef is None or "epsilon_1_gi" not in ef or "epsilon_2_gi" not in ef:
            raise KeyError(
                "GI scalar fields가 없는 이전 cache입니다. "
                "postprocess_tdse_ef --overwrite를 먼저 실행하세요."
            )
        if gauge == "zero":
            transform_to_zero_potential_gauge(obs, ef)

        outdir = output_root/_gauge_directory_name(gauge)
        outdir.mkdir(parents=True, exist_ok=True)
        products.append(plot_tdpes_decomposition(
            obs, ef, outdir, args.dpi,
            surface_count=args.surface_count,
        ))
        if not args.no_animation:
            products.append(make_tdpes_decomposition_animation(
                obs, ef, outdir, args.fps, args.max_frames,
                args.animation_dpi, args.format,
                surface_count=args.surface_count,
            ))
        del ef
        gc.collect()

    print(
        "TDSE TDPES gauge 전용 재렌더 완료: "
        f"{output_root} ({', '.join(_selected_gauges(args.gauge))})"
    )
    return [product for product in products if product is not None]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="TDSE 계산 폴더 또는 archive")
    parser.add_argument(
        "--gauge", choices=("both", "positive", "zero"), default="both",
        help="재생성할 gauge (기본값: both)",
    )
    parser.add_argument(
        "--outdir", default="",
        help="출력 report root; 기본값은 계산 폴더/report",
    )
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--animation-dpi", type=int, default=110)
    parser.add_argument("--surface-count", type=int, default=2)
    parser.add_argument(
        "--no-animation", action="store_true",
        help="6-panel 정적 PNG만 재생성",
    )
    args = parser.parse_args(argv)
    for name in ("fps", "max_frames", "dpi", "animation_dpi", "surface_count"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')}는 양수여야 합니다")
    return args


if __name__ == "__main__":
    run(parse_args())
