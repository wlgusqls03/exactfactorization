#!/usr/bin/env python3
"""Validate the CUDA discrete-MCEF RHS against the NumPy algebraic oracle."""

from __future__ import annotations

import argparse

import numpy as np

from multi_component_exact_factorization.born_huang import (
    build_born_huang_basis,
    forward_overlap_links,
)
from multi_component_exact_factorization.core import (
    add_model_arguments,
    build_model,
)
from multi_component_exact_factorization_discrete.core import (
    discrete_born_huang_rhs,
    discrete_tdse_action,
    full_step_discrete_tdse,
    reconstruct_coefficient_wavefunction,
)
from multi_component_exact_factorization_gpu.gpu_born_huang import (
    to_gpu_basis,
)
from multi_component_exact_factorization_gpu.gpu_core import cp

from .gpu_core import (
    discrete_rhs_gpu,
    discrete_tdse_action_gpu,
    full_step_discrete_bh,
    full_step_discrete_tdse_gpu,
    make_discrete_gpu_model,
)


def _normalized_problem(model, states, seed):
    rng = np.random.default_rng(seed)
    shape = (states, len(model.q), len(model.R))
    coefficients = rng.normal(size=shape)+1j*rng.normal(size=shape)
    coefficients /= np.sqrt(
        np.sum(np.abs(coefficients)**2, axis=0)
    )[None, :, :]
    lam = rng.normal(size=shape[1:])+1j*rng.normal(size=shape[1:])
    lam /= np.sqrt(np.sum(np.abs(lam)**2, axis=0)*model.dq)[None, :]
    chi = rng.normal(size=shape[2])+1j*rng.normal(size=shape[2])
    chi /= np.sqrt(np.sum(np.abs(chi)**2)*model.dR)
    # Exercise the generalized inverse through an exactly empty conditional
    # site as well as a smooth low-density row.
    lam[0, 1] = 0.0
    lam[1] *= 1.0e-4
    chi[-2] *= 1.0e-5
    return coefficients, lam, chi


def _relative_error(reference, candidate):
    absolute = float(np.max(np.abs(candidate-reference)))
    scale = max(float(np.max(np.abs(reference))), 1.0)
    return absolute, absolute/scale


def run(args):
    cp.cuda.Device(args.device).use()
    # Keep this validation deliberately small; it checks the entire RHS, not
    # production throughput.  The defaults still use the physical Shin--Metiu
    # potential and nontrivial BO overlap links.
    args.nx = args.validation_nx
    args.nq = args.validation_nq
    args.nR = args.validation_nR
    model = build_model(args)
    basis = build_born_huang_basis(model, args.states)
    basis.link_q1 = forward_overlap_links(basis.states, 2, 1, model.dx)
    basis.link_q2 = forward_overlap_links(basis.states, 2, 2, model.dx)
    basis.link_R1 = forward_overlap_links(basis.states, 3, 1, model.dx)
    basis.link_R2 = forward_overlap_links(basis.states, 3, 2, model.dx)
    coefficients, lam, chi = _normalized_problem(model, args.states, args.seed)
    model.coupling_mask_backend = "flat_top"
    model.flat_top_on_phi = args.flat_top_on_phi
    model.flat_top_on_lam = args.flat_top_on_lam
    model.flat_top_transition_decades = args.flat_top_transition_decades

    reference = discrete_born_huang_rhs(
        coefficients, lam, chi, model, basis,
        flat_top_on_phi=args.flat_top_on_phi,
        flat_top_on_lam=args.flat_top_on_lam,
        transition_decades=args.flat_top_transition_decades,
    )
    gpu_model = make_discrete_gpu_model(model)
    c_gpu = cp.ascontiguousarray(cp.asarray(coefficients, dtype=cp.complex128))
    lam_gpu = cp.asarray(lam, dtype=cp.complex128)
    chi_gpu = cp.asarray(chi, dtype=cp.complex128)
    worst = 0.0
    gpu_bases = {}
    y = reconstruct_coefficient_wavefunction(coefficients, lam, chi)
    tdse_action_reference = discrete_tdse_action(y, model, basis)
    tdse_step_reference = full_step_discrete_tdse(
        y, args.step_dt, model, basis
    )
    y_gpu = cp.ascontiguousarray(cp.asarray(y, dtype=cp.complex128))
    for backend in ("reference", "fused"):
        gpu_basis = to_gpu_basis(basis, gpu_model, backend)
        gpu_bases[backend] = gpu_basis
        result = discrete_rhs_gpu(
            c_gpu, lam_gpu, chi_gpu, gpu_model, gpu_basis,
            collect_diagnostics=True,
        )
        cp.cuda.get_current_stream().synchronize()
        print(f"[{backend}]")
        for name, expected, actual in (
            ("dC", reference.dc, cp.asnumpy(result.dc)),
            ("dLambda", reference.dlam, cp.asnumpy(result.dlam)),
            ("dChi", reference.dchi, cp.asnumpy(result.dchi)),
            ("epsilon_1", reference.fields["epsilon_1"],
             cp.asnumpy(result.fields["epsilon_1"])),
            ("epsilon_2", reference.fields["epsilon_2"],
             cp.asnumpy(result.fields["epsilon_2"])),
        ):
            absolute, relative = _relative_error(expected, actual)
            worst = max(worst, relative)
            print(
                f"  {name:10s}: max_abs={absolute:.6e}, "
                f"max_relative={relative:.6e}"
            )
        unexplained = float(
            result.diagnostics["relative_unexplained_residual"].get()
        )
        worst = max(worst, unexplained)
        print(f"  recombination unexplained relative={unexplained:.6e}")
        tdse_action = cp.asnumpy(discrete_tdse_action_gpu(
            y_gpu, gpu_model, gpu_basis
        ))
        absolute, relative = _relative_error(
            tdse_action_reference, tdse_action
        )
        worst = max(worst, relative)
        print(
            f"  TDSE H_hY : max_abs={absolute:.6e}, "
            f"max_relative={relative:.6e}"
        )
        tdse_step = cp.asnumpy(full_step_discrete_tdse_gpu(
            y_gpu, args.step_dt, gpu_model, gpu_basis
        ))
        absolute, relative = _relative_error(tdse_step_reference, tdse_step)
        worst = max(worst, relative)
        print(
            f"  TDSE RK4  : max_abs={absolute:.6e}, "
            f"max_relative={relative:.6e}"
        )
    stepped = {}
    print("[one RK4 step + support-aware PNC]")
    for backend, gpu_basis in gpu_bases.items():
        stepped[backend] = full_step_discrete_bh(
            c_gpu.copy(), lam_gpu.copy(), chi_gpu.copy(), args.step_dt,
            gpu_model, gpu_basis, collect_step_diagnostics=True,
        )
        product_change = float(
            stepped[backend][4]["pnc_product_change_l2"].get()
        )
        print(f"  {backend:9s}: PNC product change L2={product_change:.6e}")
        worst = max(worst, product_change)
    for index, name in enumerate(("C", "Lambda", "chi")):
        expected = cp.asnumpy(stepped["reference"][index])
        actual = cp.asnumpy(stepped["fused"][index])
        absolute, relative = _relative_error(expected, actual)
        worst = max(worst, relative)
        print(
            f"  step {name:6s}: max_abs={absolute:.6e}, "
            f"max_relative={relative:.6e}"
        )
    print(f"worst_relative={worst:.6e}; limit={args.tolerance:.1e}")
    if not np.isfinite(worst) or worst > args.tolerance:
        raise SystemExit("Discrete MCEF GPU validation: FAIL")
    print("Discrete MCEF GPU validation: PASS")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--validation-nx", type=int, default=32)
    parser.add_argument("--validation-nq", type=int, default=7)
    parser.add_argument("--validation-nR", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9281)
    parser.add_argument("--tolerance", type=float, default=2.0e-11)
    parser.add_argument("--step-dt", type=float, default=1.0e-3)
    parser.add_argument("--flat-top-on-phi", type=float, default=1.0e-3)
    parser.add_argument("--flat-top-on-lam", type=float, default=1.0e-3)
    parser.add_argument("--flat-top-transition-decades", type=float, default=3.0)
    add_model_arguments(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
