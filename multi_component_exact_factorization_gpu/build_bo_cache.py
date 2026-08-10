"""Build a static Born--Huang basis cache without running dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

from multi_component_exact_factorization.born_huang import (
    load_or_build_born_huang_basis,
)
from multi_component_exact_factorization.core import (
    add_model_arguments,
    build_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MCEF용 static BO energy/state/NAC cache만 미리 생성"
    )
    add_model_arguments(parser)
    parser.add_argument("--bo-states", type=int, default=8)
    parser.add_argument(
        "--bo-basis-cache-dir", default="results/bo_basis_cache"
    )
    parser.add_argument("--rebuild-bo-basis-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.bo_states < 1:
        raise ValueError("--bo-states는 1 이상이어야 합니다.")
    model = build_model(args)
    basis, info = load_or_build_born_huang_basis(
        model, args.bo_states,
        cache_dir=Path(args.bo_basis_cache_dir),
        rebuild=args.rebuild_bo_basis_cache,
    )
    state = "HIT" if info["hit"] else "BUILT"
    print(
        f"BO cache {state}: requested={args.bo_states}, "
        f"stored={info['stored_states']}, seconds={info['seconds']:.2f}"
    )
    print(f"path: {info['path']}")
    print(
        "shapes: "
        f"energies={basis.energies.shape}, states={basis.states.shape}, "
        f"NAC={basis.d_q.shape}"
    )
    return info["path"]


if __name__ == "__main__":
    main()
