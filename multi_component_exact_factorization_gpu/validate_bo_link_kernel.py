"""Validate and microbenchmark the fused BO overlap-link CUDA kernel."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np

from multi_component_exact_factorization.born_huang import (
    forward_overlap_links,
)

from .gpu_born_huang import (
    _fused_covariant_from_transports,
    _launch_fused_transport,
    projected_link_derivatives,
    to_gpu_basis,
)
from .gpu_core import cp


def _random_basis(rng, nstate, nx, nq, nR, complex_states=False):
    dtype = complex if complex_states else float
    states = np.empty((nstate, nx, nq, nR), dtype=dtype)
    for iq in range(nq):
        for iR in range(nR):
            values = rng.normal(size=(nx, nstate))
            if complex_states:
                values = values+1j*rng.normal(size=(nx, nstate))
            orthogonal, _ = np.linalg.qr(values)
            states[:, :, iq, iR] = orthogonal.T
    zeros = np.zeros((nstate, nq, nR), dtype=float)
    return SimpleNamespace(
        energies=zeros,
        link_q1=forward_overlap_links(states, 2, 1, 1.0),
        link_q2=forward_overlap_links(states, 2, 2, 1.0),
        link_R1=forward_overlap_links(states, 3, 1, 1.0),
        link_R2=forward_overlap_links(states, 3, 2, 1.0),
    )


def _relative_error(reference, candidate):
    absolute = float(cp.max(cp.abs(reference-candidate)).get())
    scale = max(float(cp.max(cp.abs(reference)).get()), 1.0e-300)
    return absolute, absolute/scale


def _adjoint_defects(values_u, values_v, basis, spacing, axis, vector):
    d1u, d2u = projected_link_derivatives(
        values_u, basis, spacing, axis, vector=vector
    )
    d1v, d2v = projected_link_derivatives(
        values_v, basis, spacing, axis, vector=vector
    )
    left1 = cp.vdot(values_u, d1v)
    right1 = cp.vdot(d1u, values_v)
    left2 = cp.vdot(values_u, d2v)
    right2 = cp.vdot(d2u, values_v)
    scale1 = cp.maximum(cp.abs(left1)+cp.abs(right1), 1.0e-300)
    scale2 = cp.maximum(cp.abs(left2)+cp.abs(right2), 1.0e-300)
    return (
        float((cp.abs(left1+right1)/scale1).get()),
        float((cp.abs(left2-right2)/scale2).get()),
    )


def _timed(milliseconds_repeats, callback):
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    callback()
    cp.cuda.get_current_stream().synchronize()
    start.record()
    for _ in range(milliseconds_repeats):
        callback()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop)/milliseconds_repeats


def run(args):
    cp.cuda.Device(args.device).use()
    rng = np.random.default_rng(args.seed)
    cpu_basis = _random_basis(
        rng, args.states, max(args.states+2, 8), args.nq, args.nR,
        complex_states=args.complex_links,
    )
    model = SimpleNamespace(real_dtype=cp.float64, complex_dtype=cp.complex128)
    reference = to_gpu_basis(cpu_basis, model, "reference")
    fused = to_gpu_basis(cpu_basis, model, "fused")
    shape = (args.states, args.nq, args.nR)
    coefficients = cp.asarray(
        rng.normal(size=shape)+1j*rng.normal(size=shape), dtype=cp.complex128
    )
    other = cp.asarray(
        rng.normal(size=shape)+1j*rng.normal(size=shape), dtype=cp.complex128
    )
    vector = cp.asarray(
        rng.normal(scale=0.1, size=(args.nq, args.nR)), dtype=cp.float64
    )
    worst_relative = 0.0
    worst_adjoint = 0.0
    print(
        f"shape={shape}, links={'complex128' if args.complex_links else 'float64'}, "
        f"repeats={args.repeats}"
    )
    for axis, spacing, label in ((1, 0.04, "q"), (2, 0.02, "R")):
        ref_plain = projected_link_derivatives(
            coefficients, reference, spacing, axis
        )
        ref_cov = projected_link_derivatives(
            coefficients, reference, spacing, axis, vector=vector
        )
        fused_plain_first, fused_plain_second = _launch_fused_transport(
            coefficients, fused, spacing, axis,
            write_transports=True, write_first=True, write_second=True,
        )
        # Copy only for validation because the reusable workspace is mutable.
        fused_plain = (fused_plain_first.copy(), fused_plain_second.copy())
        fused_cov = _fused_covariant_from_transports(
            coefficients, fused, vector, spacing, axis
        )
        comparisons = (
            ("plain_D1", ref_plain[0], fused_plain[0]),
            ("plain_D2", ref_plain[1], fused_plain[1]),
            ("covariant_D1", ref_cov[0], fused_cov[0]),
            ("covariant_D2", ref_cov[1], fused_cov[1]),
        )
        for name, expected, actual in comparisons:
            absolute, relative = _relative_error(expected, actual)
            worst_relative = max(worst_relative, relative)
            print(
                f"{label} {name}: max_abs={absolute:.3e}, "
                f"max_relative={relative:.3e}"
            )
        plain_anti_h, plain_herm = _adjoint_defects(
            coefficients, other, fused, spacing, axis, None
        )
        cov_anti_h, cov_herm = _adjoint_defects(
            coefficients, other, fused, spacing, axis, vector
        )
        worst_adjoint = max(
            worst_adjoint, plain_anti_h, plain_herm, cov_anti_h, cov_herm
        )
        print(
            f"{label} plain adjoint: D1 anti-H={plain_anti_h:.3e}, "
            f"D2 Hermitian={plain_herm:.3e}"
        )
        print(
            f"{label} covariant adjoint: D1 anti-H={cov_anti_h:.3e}, "
            f"D2 Hermitian={cov_herm:.3e}"
        )

        def reference_pair():
            projected_link_derivatives(coefficients, reference, spacing, axis)
            projected_link_derivatives(
                coefficients, reference, spacing, axis, vector=vector
            )

        def fused_pair():
            _launch_fused_transport(
                coefficients, fused, spacing, axis,
                write_transports=True, write_first=True, write_second=False,
            )
            _fused_covariant_from_transports(
                coefficients, fused, vector, spacing, axis
            )

        reference_ms = _timed(args.repeats, reference_pair)
        fused_ms = _timed(args.repeats, fused_pair)
        print(
            f"{label} plain+covariant: reference={reference_ms:.3f} ms, "
            f"fused={fused_ms:.3f} ms, speedup={reference_ms/fused_ms:.3f}x"
        )

    passed = (
        worst_relative <= args.relative_tolerance
        and worst_adjoint <= args.adjoint_tolerance
    )
    print(
        f"worst_relative={worst_relative:.3e} "
        f"(limit {args.relative_tolerance:.1e})"
    )
    print(
        f"worst_adjoint={worst_adjoint:.3e} "
        f"(limit {args.adjoint_tolerance:.1e})"
    )
    print("BO fused link validation: " + ("PASS" if passed else "FAIL"))
    if not passed:
        raise SystemExit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--nq", type=int, default=37)
    parser.add_argument("--nR", type=int, default=41)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--complex-links", action="store_true")
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-12)
    parser.add_argument("--adjoint-tolerance", type=float, default=2.0e-12)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
