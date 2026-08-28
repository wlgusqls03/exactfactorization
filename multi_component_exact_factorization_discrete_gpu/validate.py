#!/usr/bin/env python3
"""Validate the CUDA discrete-MCEF RHS against the NumPy algebraic oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from multi_component_exact_factorization.born_huang import (
    build_born_huang_basis,
    forward_overlap_links,
)
from multi_component_exact_factorization.core import (
    add_model_arguments,
    build_model,
)
from multi_component_exact_factorization.spectral_tdse import (
    spectral_action_numpy,
    spectral_energy_numpy,
    split_step_numpy,
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
from .checkpoint import load_checkpoint, write_checkpoint_atomic
from .spectral_tdse import SpectralTDSEGPU, complex_inner_product_gpu


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
    print("[TDSE density-gauge scalar GI/GD decomposition]")
    rho = np.sum(np.abs(y)**2, axis=0)
    rho_R = np.sum(rho, axis=0)*model.dq
    F = np.sqrt(np.maximum(rho, 0.0))
    chi_density = np.sqrt(np.maximum(rho_R, 0.0))
    c_density = np.divide(
        y, F[None, :, :], out=np.zeros_like(y),
        where=F[None, :, :] > 0.0,
    )
    lam_density = np.divide(
        F, chi_density[None, :], out=np.zeros_like(F),
        where=chi_density[None, :] > 0.0,
    )
    density_reference = discrete_born_huang_rhs(
        c_density, lam_density, chi_density, model, basis,
        horizontal_correction=True,
    )
    from .postprocess_tdse_ef import _frame_fields
    decomposed = _frame_fields(
        y, gpu_model, gpu_bases["fused"],
        action_cpu=tdse_action_reference,
    )
    dy = -1j*tdse_action_reference
    drho = 2.0*np.real(np.sum(np.conj(y)*dy, axis=0))
    dF = np.divide(
        drho, 2.0*F, out=np.zeros_like(F), where=F > 0.0,
    )
    drho_R = np.sum(drho, axis=0)*model.dq
    dchi = np.divide(
        drho_R, 2.0*chi_density, out=np.zeros_like(chi_density),
        where=chi_density > 0.0,
    )
    dc = np.divide(
        dy-c_density*dF[None, :, :], F[None, :, :],
        out=np.zeros_like(y), where=F[None, :, :] > 0.0,
    )
    dlam = np.divide(
        dF-lam_density*dchi[None, :], chi_density[None, :],
        out=np.zeros_like(F), where=chi_density[None, :] > 0.0,
    )
    c_norm = np.maximum(np.sum(np.abs(c_density)**2, axis=0), 1.0e-300)
    lam_norm = np.maximum(
        np.sum(np.abs(lam_density)**2, axis=0)*model.dq, 1.0e-300,
    )
    temporal_1 = -1j*np.sum(np.conj(c_density)*dc, axis=0)/c_norm
    temporal_2 = (
        -1j*np.sum(np.conj(lam_density)*dlam, axis=0)
        *model.dq/lam_norm
    )
    epsilon_1_expected = density_reference.fields["epsilon_1"]+temporal_1.real
    epsilon_2_expected = (
        density_reference.fields["epsilon_2"]
        +(
            np.sum(np.abs(lam_density)**2*temporal_1, axis=0)
            *model.dq/lam_norm+temporal_2
        ).real
    )
    for name, expected, actual in (
        ("epsilon_1_gi", density_reference.fields["epsilon_1"],
         decomposed["epsilon_1_gi"]),
        ("epsilon_2_gi", density_reference.fields["epsilon_2"],
         decomposed["epsilon_2_gi"]),
        ("epsilon_1_total", epsilon_1_expected,
         decomposed["epsilon_1"]),
        ("epsilon_2_total", epsilon_2_expected,
         decomposed["epsilon_2"]),
    ):
        absolute, relative = _relative_error(expected, actual)
        worst = max(worst, relative)
        print(
            f"  {name:17s}: max_abs={absolute:.6e}, "
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
    print("[full-grid spectral TDSE one step]")
    rng = np.random.default_rng(args.seed+17)
    psi = rng.normal(
        size=(len(model.x), len(model.q), len(model.R))
    )+1j*rng.normal(size=(len(model.x), len(model.q), len(model.R)))
    psi /= np.sqrt(
        np.sum(np.abs(psi)**2)*model.dx*model.dq*model.dR
    )
    expected_split = split_step_numpy(psi, args.step_dt, model)
    split_solver = SpectralTDSEGPU(
        model, q_block_R=3, R_block_x=4, x_block_R=2
    )
    psi_gpu = cp.ascontiguousarray(cp.asarray(psi, dtype=cp.complex128))
    split_solver.step(psi_gpu, args.step_dt)
    actual_split = cp.asnumpy(psi_gpu)
    absolute, relative = _relative_error(expected_split, actual_split)
    worst = max(worst, relative)
    norm_error = abs(
        np.sum(np.abs(actual_split)**2)*model.dx*model.dq*model.dR-1.0
    )
    worst = max(worst, norm_error)
    print(
        f"  CPU/GPU step: max_abs={absolute:.6e}, "
        f"max_relative={relative:.6e}"
    )
    print(f"  one-step norm error={norm_error:.6e}")
    separate_solver = SpectralTDSEGPU(
        model, q_block_R=3, R_block_x=4, x_block_R=2,
        nuclear_fft_mode="separate",
    )
    separate_gpu = cp.ascontiguousarray(
        cp.asarray(psi, dtype=cp.complex128)
    )
    separate_solver.step(separate_gpu, args.step_dt)
    absolute, separate_relative = _relative_error(
        expected_split, cp.asnumpy(separate_gpu)
    )
    worst = max(worst, separate_relative)
    print(
        "  separate nuclear FFT fallback: "
        f"max_relative={separate_relative:.6e}"
    )
    expected_two_steps = split_step_numpy(
        expected_split, args.step_dt, model
    )
    staggered_gpu = cp.ascontiguousarray(
        cp.asarray(psi, dtype=cp.complex128)
    )
    split_solver.stage_for_next_step(staggered_gpu, args.step_dt)
    split_solver.advance_staggered(
        staggered_gpu, args.step_dt, return_physical=False
    )
    split_solver.advance_staggered(
        staggered_gpu, args.step_dt, return_physical=True
    )
    absolute, staggered_relative = _relative_error(
        expected_two_steps, cp.asnumpy(staggered_gpu)
    )
    worst = max(worst, staggered_relative)
    print(
        "  merged adjacent potential half-kicks: "
        f"max_relative={staggered_relative:.6e}"
    )
    expected_action = spectral_action_numpy(psi, model)
    action_input_gpu = cp.ascontiguousarray(
        cp.asarray(psi, dtype=cp.complex128)
    )
    actual_action_gpu, fused_energy_gpu = split_solver.action_and_energy(
        action_input_gpu
    )
    actual_action = cp.asnumpy(actual_action_gpu)
    absolute, relative = _relative_error(expected_action, actual_action)
    worst = max(worst, relative)
    print(
        f"  H_spectral Psi: max_abs={absolute:.6e}, "
        f"max_relative={relative:.6e}"
    )
    expected_inner = np.vdot(psi, expected_action)
    actual_inner = complex(
        complex_inner_product_gpu(action_input_gpu, actual_action_gpu).get()
    )
    inner_scale = max(abs(expected_inner), 1.0)
    inner_relative = abs(actual_inner-expected_inner)/inner_scale
    worst = max(worst, inner_relative)
    print(
        "  bounded <Psi|H|Psi> reduction: "
        f"relative={inner_relative:.6e}"
    )
    expected_energy = spectral_energy_numpy(psi, model)
    actual_energy_gpu = split_solver.energy(
        cp.ascontiguousarray(cp.asarray(psi, dtype=cp.complex128))
    )
    energy_errors = []
    fused_energy_errors = []
    for name in (
        "norm", "kinetic_x", "kinetic_q", "kinetic_R",
        "potential", "energy",
    ):
        actual = float(actual_energy_gpu[name].get())
        fused = float(fused_energy_gpu[name].get())
        scale = max(abs(expected_energy[name]), 1.0)
        energy_errors.append(abs(actual-expected_energy[name])/scale)
        fused_energy_errors.append(abs(fused-expected_energy[name])/scale)
    energy_relative = max(energy_errors)
    fused_energy_relative = max(fused_energy_errors)
    worst = max(worst, energy_relative, fused_energy_relative)
    print(f"  spectral energy: max_relative={energy_relative:.6e}")
    print(
        "  fused H_spectral Psi/energy: "
        f"max_relative={fused_energy_relative:.6e}"
    )
    print("[checkpoint round-trip + resumed RK4 step]")
    fused_step = stepped["fused"]
    checkpoint_metadata = {
        "validation": "discrete-mcef-gpu",
        "dt_au": float(args.step_dt),
        "shape": list(coefficients.shape),
    }
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_path = Path(temporary)/"checkpoint.npz"
        write_checkpoint_atomic(
            checkpoint_path,
            completed_step=1,
            coefficients=fused_step[0],
            lam=fused_step[1],
            chi=fused_step[2],
            metadata=checkpoint_metadata,
        )
        loaded = load_checkpoint(
            checkpoint_path, expected_metadata=checkpoint_metadata
        )
        loaded_gpu = (
            cp.ascontiguousarray(cp.asarray(
                loaded["electronic_coefficients"], dtype=cp.complex128
            )),
            cp.asarray(loaded["lambda_wavefunction"], dtype=cp.complex128),
            cp.asarray(loaded["chi"], dtype=cp.complex128),
        )
        for index, name in enumerate(("C", "Lambda", "chi")):
            expected = cp.asnumpy(fused_step[index])
            actual = cp.asnumpy(loaded_gpu[index])
            absolute, relative = _relative_error(expected, actual)
            worst = max(worst, relative)
            print(
                f"  state {name:6s}: max_abs={absolute:.6e}, "
                f"max_relative={relative:.6e}"
            )
        direct_next = full_step_discrete_bh(
            fused_step[0], fused_step[1], fused_step[2], args.step_dt,
            gpu_model, gpu_bases["fused"],
        )
        resumed_next = full_step_discrete_bh(
            loaded_gpu[0], loaded_gpu[1], loaded_gpu[2], args.step_dt,
            gpu_model, gpu_bases["fused"],
        )
        for index, name in enumerate(("C", "Lambda", "chi")):
            expected = cp.asnumpy(direct_next[index])
            actual = cp.asnumpy(resumed_next[index])
            absolute, relative = _relative_error(expected, actual)
            worst = max(worst, relative)
            print(
                f"  next {name:7s}: max_abs={absolute:.6e}, "
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
    # Structural validation has self-contained harmless values; production
    # propagation still requires both physical parameters explicitly.
    parser.set_defaults(heavy_trap_alpha=0.005, erf_r_qR=5.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
