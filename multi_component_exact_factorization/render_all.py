#!/usr/bin/env python3
"""Render every standard analysis product from one completed MCEF run.

The positional argument may be an archive, a run-directory path, or only the
run-directory name. A bare name is searched below dated ``results/YYYYMMDD``
directories and the newest matching run is selected.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np

from result_paths import dated_results_dir

from . import dynamics_analysis, excited_state_analysis, visualize, visualize_3d


ARCHIVE_NAMES = (
    "multi_component_direct_ef_gpu.npz",
    "multi_component_direct_ef.npz",
    "multi_component_reference.npz",
)


def _relative_run_name(value: Path) -> Path:
    """Remove a leading relative ``results`` component for dated searches."""
    parts = value.parts
    if parts and parts[0] == "results":
        parts = parts[1:]
    if parts and len(parts[0]) == 8 and parts[0].isdigit():
        parts = parts[1:]
    return Path(*parts) if parts else Path(value.name)


def resolve_run_input(value: str | Path, results_root: str | Path = "results") -> Path:
    """Resolve an archive or run directory, searching newest dated runs."""
    supplied = Path(value).expanduser()
    if supplied.is_file() or supplied.is_dir():
        return supplied.resolve()
    if supplied.is_absolute():
        raise FileNotFoundError(f"계산 결과 경로를 찾을 수 없습니다: {supplied}")

    root = Path(results_root)
    relative = _relative_run_name(supplied)
    today = datetime.now().astimezone().strftime("%Y%m%d")
    candidates = []

    today_candidate = root/today/relative
    if today_candidate.is_file() or today_candidate.is_dir():
        candidates.append(today_candidate)

    candidates.extend(
        path for path in root.glob(f"????????/{relative}")
        if path.is_file() or path.is_dir()
    )
    if len(relative.parts) == 1:
        candidates.extend(
            path for path in root.glob(f"????????/**/{relative.name}")
            if path.is_file() or path.is_dir()
        )

    unique = {path.resolve(): path.resolve() for path in candidates}
    if not unique:
        raise FileNotFoundError(
            f"{supplied!s} 또는 {root}/YYYYMMDD 아래에서 계산 결과를 찾지 못했습니다."
        )
    def newest_key(path: Path):
        run_date = ""
        for index, part in enumerate(path.parts):
            if (
                len(part) == 8 and part.isdigit()
                and index > 0 and path.parts[index-1] == "results"
            ):
                run_date = part
                break
        return run_date, path.stat().st_mtime, str(path)

    return max(unique.values(), key=newest_key)


def find_archive(run_input: Path) -> tuple[Path, Path]:
    """Return ``(archive, run_directory)`` for a resolved input path."""
    if run_input.is_file():
        if run_input.suffix != ".npz":
            raise ValueError(f"NPZ archive가 아닙니다: {run_input}")
        return run_input, run_input.parent

    matches = [run_input/name for name in ARCHIVE_NAMES if (run_input/name).is_file()]
    if not matches:
        raise FileNotFoundError(
            f"{run_input}에서 MCEF archive를 찾지 못했습니다. 예상 파일: "
            + ", ".join(ARCHIVE_NAMES)
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"계산 폴더에 archive가 여러 개 있습니다: {names}. 사용할 NPZ를 직접 입력하세요."
        )
    return matches[0], run_input


def archive_state(archive: Path) -> tuple[int, str]:
    """Read the initial local-electronic state from archive metadata."""
    with np.load(archive, allow_pickle=True) as data:
        options = visualize.archive_arguments(data)
    excitation = int(options.get("electron_excitation", 0))
    label = "ground state" if excitation == 0 else f"excited state n={excitation}"
    return excitation, label


def run(args):
    resolved = resolve_run_input(args.run)
    archive, run_dir = find_archive(resolved)
    excitation, state_label = archive_state(archive)
    n_states = args.n_states or max(3, excitation+2)
    output_root = dated_results_dir(run_dir)

    print(f"계산 폴더: {run_dir}")
    print(f"선택 archive: {archive.name}")
    print(f"초기 전자상태: {state_label}; 분석 BO 상태 수={n_states}")
    print(f"통합 출력 위치: {output_root}")

    common_animation = dict(
        dpi=args.dpi,
        animation_dpi=args.animation_dpi,
        fps=args.fps,
        max_frames=args.max_frames,
        format=args.format,
        no_animation=args.no_animation,
    )

    print("[1/4] 기본 factor/density/potential 그림 생성")
    visualize.run(Namespace(
        archive=str(archive),
        outdir=str(output_root/"figures"),
        snapshots=args.snapshots,
        profile_frame=-1,
        animation_style="all",
        **common_animation,
    ))

    print("[2/4] 실제 marginal 및 nonadiabatic dynamics 분석")
    dynamics_analysis.run(Namespace(
        archive=str(archive),
        outdir=str(output_root/"dynamics_analysis"),
        dpi=args.dpi,
        n_states=n_states,
        frame=-1,
        electron_divider=None,
        no_bo=False,
    ))

    print("[3/4] local electronic-state population 분석")
    excited_state_analysis.run(Namespace(
        archive=str(archive),
        outdir=str(output_root/"excited_state_analysis"),
        n_states=n_states,
        **common_animation,
    ))

    if args.no_3d:
        print("[4/4] 3D HTML 생략 (--no-3d)")
    else:
        print("[4/4] interactive 3D configuration-density HTML 생성")
        visualize_3d.run(Namespace(
            archive=str(archive),
            outdir=str(output_root/"visualization_3d"),
            max_axis_points=args.max_axis_points,
            max_frames=args.max_3d_frames,
            surface_count=args.surface_count,
            isomin_fraction=0.025,
            opacity=0.32,
            colorscale="Viridis",
            fps=float(args.fps),
            width=1050,
            height=820,
        ))

    print(f"모든 분석 완료: {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        help="계산 폴더, 결과 NPZ 경로, 또는 results 아래의 계산 폴더 이름",
    )
    parser.add_argument(
        "--n-states", type=int, default=0,
        help="0이면 초기 excitation에서 분석 상태 수를 자동 선택",
    )
    parser.add_argument("--snapshots", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--animation-dpi", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--max-axis-points", type=int, default=24)
    parser.add_argument("--max-3d-frames", type=int, default=80)
    parser.add_argument("--surface-count", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
