"""Render proton--heavy collision diagnostics from an existing TDSE run."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import tdse_collision_report, tdse_report


def _archive(run):
    path = Path(run).expanduser().resolve()
    if path.is_file():
        return path, path.parent
    candidate = path/"multi_component_discrete_tdse_gpu.npz"
    if not candidate.is_file():
        raise FileNotFoundError(f"TDSE archive를 찾을 수 없습니다: {candidate}")
    return candidate, path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="TDSE run directory or archive path")
    parser.add_argument("--outdir", help="default: RUN/report")
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--animation-dpi", type=int, default=110)
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--decades", type=float, default=6.0)
    parser.add_argument("--no-animation", action="store_true")
    return parser.parse_args(argv)


def run(args):
    archive, run_dir = _archive(args.run)
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else run_dir/"report"
    data = tdse_report.load_observables(archive)
    obs = tdse_report.calculate_observables(data)
    print(
        f"TDSE collision report: archive={archive}; frames={len(obs['times_fs'])}; "
        f"joint={obs['joint_density'].nbytes/1024**3:.2f} GiB"
    )
    return tdse_collision_report.run(
        obs, outdir, dpi=args.dpi, no_animation=args.no_animation,
        fps=args.fps, max_frames=args.max_frames,
        animation_dpi=args.animation_dpi, fmt=args.format,
        snapshot_count=args.snapshots, decades=args.decades,
    )


if __name__ == "__main__":
    run(parse_args())
