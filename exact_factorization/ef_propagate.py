#!/usr/bin/env python3
"""Direct self-consistent propagation of the 1D exact-factorization equations.

The conditional electronic state is represented in a finite BO basis,
Phi_R(r,t) = sum_s C_s(R,t) phi_s(r;R). Increasing --n-states provides the
electronic-basis convergence test. No surface hopping or stochastic step is
used: chi and all C_s are propagated as coupled quantum fields.

Array shapes in this file are:
    chi                 (nR,)
    coefficients C_s   (ns,nR)
    Phi_R(r)            (nr,nR)
    A, epsilon, mask    (nR,)
Saved arrays add a leading time-frame axis ``nt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shin_metiu_1d import AU_PER_FS

from .core import (
    build_model,
    coefficients_to_phi,
    coupling_from_phi,
    derivative,
    enforce_partial_normalization,
    normalize_initial_chi,
    phase_of_chi,
    project_electronic,
)


def electronic_nuclear_rhs(coefficients, chi, model, args):
    """Evaluate both coupled EF time derivatives at one RK stage.

    This is the central routine of the direct solver. It maps the current
    quantum state

    ``coefficients(ns,nR), chi(nR)``

    to

    ``dcoeff/dt(ns,nR), dchi/dt(nR)``

    while also returning the instantaneous physical/coupling fields
    ``A(nR)``, ``epsilon(nR)``, ``U_en coefficients(ns,nR)``, and
    ``mask(nR)``.
    """
    # Conditional electronic fields are generally not periodic as functions
    # of their parameter R, even when the full Psi uses a periodic FFT box.
    derivative_boundary = (
        "periodic" if args.derivative_scheme == "spectral" else "dirichlet"
    )

    # 1) Reconstruct Phi_R(r) from its BO coefficients at every R grid point.
    #    C(ns,nR) x phi_BO(ns,nR,nr) -> Phi(nr,nR).
    phi = coefficients_to_phi(coefficients, model)

    # 2) Evaluate the exact electron--nuclear coupling operator and A(R).
    A, u_phi, mask, _ = coupling_from_phi(
        phi, chi, model, args.density_threshold, derivative_boundary,
        norm_correction=not args.no_norm_correction,
    )

    # 3) Project U_en Phi back into the same finite BO basis. H_BO is diagonal
    #    in this basis, so its action is simply E_s(R) C_s(R).
    u_coeff = project_electronic(u_phi, model)
    h_coeff = model.bo_energies * coefficients + u_coeff

    # epsilon=<Phi|H_BO+U_en|Phi> in the epsilon_GD=0 (temporal) gauge.
    # Subtracting epsilon in the electronic equation fixes the redundant
    # R-dependent phase shared by Phi and chi.
    epsilon = np.sum(coefficients.conj() * h_coeff, axis=0).real
    dcoeff = -1j * (h_coeff - epsilon[None, :] * coefficients)

    # 4) Nuclear equation: i d_t chi = [(-i d_R+A)^2/(2M)+epsilon] chi.
    #    Apply the covariant momentum twice rather than expanding derivatives
    #    of A by hand.
    momentum_chi = -1j * derivative(
        chi, model.dR, boundary=derivative_boundary
    ) + A * chi
    kinetic_chi = -1j * derivative(
        momentum_chi, model.dR, boundary=derivative_boundary
    ) + A * momentum_chi
    dchi = -1j * (0.5 * kinetic_chi / model.mass + epsilon * chi)
    return dcoeff, dchi, A, epsilon, u_coeff, mask


def rk4_step(coefficients, chi, dt, model, args):
    """Advance ``C(ns,nR)`` and ``chi(nR)`` by one coupled RK4 step.

    Couplings are recomputed at all four internal RK stages. Freezing A,
    epsilon, or U_en for the whole step would decouple the two equations and
    reduce the temporal accuracy.
    """
    k1c, k1x, *_ = electronic_nuclear_rhs(coefficients, chi, model, args)
    k2c, k2x, *_ = electronic_nuclear_rhs(
        coefficients + 0.5 * dt * k1c, chi + 0.5 * dt * k1x, model, args
    )
    k3c, k3x, *_ = electronic_nuclear_rhs(
        coefficients + 0.5 * dt * k2c, chi + 0.5 * dt * k2x, model, args
    )
    k4c, k4x, *_ = electronic_nuclear_rhs(
        coefficients + dt * k3c, chi + dt * k3x, model, args
    )
    coefficients = coefficients + dt * (k1c + 2*k2c + 2*k3c + k4c) / 6.0
    chi = chi + dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
    if args.boundary == "dirichlet":
        # The isolated-box boundary condition is imposed only after completing
        # the RK combination; intermediate stages still use one-sided stencils.
        chi[[0, -1]] = 0.0
    if not args.no_pnc_projection:
        # Numerical RK errors can violate sum_s |C_s(R)|^2=1. The correction
        # transfers the local norm to chi, leaving every product C_s*chi and
        # therefore the reconstructed molecular wavefunction unchanged.
        coefficients, chi, pnc_before = enforce_partial_normalization(coefficients, chi)
    else:
        pnc_before = float(np.max(np.abs(np.sum(np.abs(coefficients)**2, axis=0)-1)))
    return coefficients, chi, pnc_before


def run(args):
    """Set the initial state, propagate it, and write the direct-EF archive."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("Building grids and BO basis...")
    model = build_model(args)
    if not 0 <= args.initial_state < args.n_states:
        raise ValueError("--initial-state must be smaller than --n-states")

    # Initial factorization:
    #   chi(R,0) = normalized Gaussian nuclear packet
    #   Phi_R(r,0) = selected BO state, i.e. C_initial(R,0)=1 for all R.
    sigma = args.sigma if args.sigma > 0 else 1.0 / np.sqrt(2.85)
    chi = np.exp(
        -0.5 * ((model.R - args.R0) / sigma) ** 2
        + 1j * args.P0 * (model.R - args.R0)
    ).astype(complex)
    if args.boundary == "dirichlet":
        chi[[0, -1]] = 0.0
    chi = normalize_initial_chi(chi, model.dR)
    coefficients = np.zeros((args.n_states, args.nR), dtype=complex)
    coefficients[args.initial_state] = 1.0

    n_steps = int(round(args.t_final_fs * AU_PER_FS / args.dt_au))
    save_steps = list(range(0, n_steps + 1, max(1, args.save_every)))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    nframes = len(save_steps)

    # Allocate saved trajectories. ``nt`` below means nframes, not n_steps.
    times = np.empty(nframes)                                      # (nt,)
    chis = np.empty((nframes, args.nR), complex)                   # (nt,nR)
    coeffs = np.empty((nframes, args.n_states, args.nR), complex) # (nt,ns,nR)
    Avec = np.empty((nframes, args.nR))                            # (nt,nR)
    epsilon = np.empty((nframes, args.nR))                         # (nt,nR)
    u_coeffs = np.empty(
        (nframes, args.n_states, args.nR), complex
    )                                                              # (nt,ns,nR)
    masks = np.empty((nframes, args.nR))                           # (nt,nR)
    populations = np.empty((nframes, args.n_states))              # (nt,ns)
    norm = np.empty(nframes)                                      # (nt,)
    pnc_error = np.empty(nframes)                                 # (nt,)
    # Full real-space fields are large, so lists are filled only when the
    # user explicitly requests --save-fields. Each frame is (nr,nR).
    phis = []
    psis = []
    uphis = []

    def save_frame(frame, step, pnc_before=0.0):
        """Measure one state without changing the propagated variables."""
        _, _, A, eps, uc, mask = electronic_nuclear_rhs(coefficients, chi, model, args)
        times[frame] = step * args.dt_au / AU_PER_FS
        chis[frame] = chi
        coeffs[frame] = coefficients
        Avec[frame] = A
        epsilon[frame] = eps
        u_coeffs[frame] = uc
        masks[frame] = mask
        # F_s(R,t)=C_s(R,t) chi(R,t), hence P_s=int dR |C_s chi|^2.
        populations[frame] = np.sum(
            np.abs(coefficients * chi[None, :]) ** 2, axis=1
        ) * model.dR
        norm[frame] = np.sum(np.abs(chi) ** 2) * model.dR
        pnc_error[frame] = max(
            pnc_before,
            float(np.max(np.abs(np.sum(np.abs(coefficients)**2, axis=0)-1.0))),
        )
        if args.save_fields:
            phi = coefficients_to_phi(coefficients, model)
            _, uphi, _, _ = coupling_from_phi(
                phi, chi, model, args.density_threshold,
                "periodic" if args.derivative_scheme == "spectral" else "dirichlet",
                norm_correction=not args.no_norm_correction,
            )
            phis.append(phi.copy())
            # Psi(r,R)=Phi_R(r) chi(R); broadcasting chi across the r axis.
            psis.append((phi * chi[None, :]).copy())
            uphis.append(uphi.copy())

    save_frame(0, 0)
    frame = 1
    last_pnc = 0.0
    for step in range(1, n_steps + 1):
        coefficients, chi, last_pnc = rk4_step(
            coefficients, chi, args.dt_au, model, args
        )
        if not np.all(np.isfinite(chi)) or not np.all(np.isfinite(coefficients)):
            raise FloatingPointError(f"EF propagation became non-finite at step {step}")
        if frame < nframes and step == save_steps[frame]:
            save_frame(frame, step, last_pnc)
            frame += 1
        if step % max(1, args.progress_every) == 0:
            print(
                f"step {step:6d}/{n_steps}  t={step*args.dt_au/AU_PER_FS:7.3f} fs"
                f"  norm={np.sum(np.abs(chi)**2)*model.dR:.10f}"
                f"  pnc(pre)={last_pnc:.2e}"
            )

    # NPZ shape map:
    #   scalar grids/basis: r(nr), R(nR), E_BO(ns,nR), phi_BO(ns,nR,nr)
    #   trajectories: chi(nt,nR), coefficients(nt,ns,nR), A/epsilon(nt,nR)
    #   diagnostics: populations(nt,ns), norm(nt), pnc_error(nt)
    payload = dict(
        kind=np.array("direct_ef"), gauge=np.array("epsilon_gd_zero"),
        r=model.r, R=model.R, times_fs=times,
        chi=chis, coefficients=coeffs, A=Avec, epsilon=epsilon,
        u_coefficients=u_coeffs, mask=masks,
        phase_S=np.array([phase_of_chi(x, m) for x, m in zip(chis, masks)]),
        populations=populations, norm=norm, pnc_error=pnc_error,
        bo_energies=model.bo_energies, bo_states=model.bo_states,
        args=np.array([vars(args)], dtype=object),
    )
    if args.save_fields:
        payload.update(phi=np.asarray(phis), psi=np.asarray(psis), u_phi=np.asarray(uphis))
    path = outdir / "shin_metiu_direct_ef.npz"
    np.savez_compressed(path, **payload)
    print(f"Saved {path}")
    print(
        "Saved shapes: "
        f"chi={chis.shape}, coefficients={coeffs.shape}, A={Avec.shape}, "
        f"epsilon={epsilon.shape}, populations={populations.shape}"
    )
    if args.save_fields:
        print(
            "Full-field shapes: "
            f"phi={np.asarray(phis).shape}, psi={np.asarray(psis).shape}, "
            f"u_phi={np.asarray(uphis).shape}"
        )
    print("Final populations:", " ".join(f"P{i+1}={p:.6f}" for i,p in enumerate(populations[-1])))
    print(f"Maximum nuclear norm error: {np.max(np.abs(norm-1)):.3e}")
    print(f"Maximum pre-projection PNC error: {np.max(pnc_error):.3e}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results/exact_factorization/direct")
    p.add_argument("--nr", type=int, default=128)
    p.add_argument("--nR", type=int, default=512)
    p.add_argument("--r-min", type=float, default=-15.0)
    p.add_argument("--r-max", type=float, default=15.0)
    p.add_argument("--R-min", type=float, default=-8.0)
    p.add_argument("--R-max", type=float, default=8.0)
    p.add_argument("--dt-au", type=float, default=0.05)
    p.add_argument("--t-final-fs", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--initial-state", type=int, default=1)
    p.add_argument("--mass", type=float, default=1836.0)
    p.add_argument("--L", type=float, default=19.0)
    p.add_argument("--Rf", type=float, default=5.0)
    p.add_argument("--Rl", type=float, default=3.1)
    p.add_argument("--Rr", type=float, default=4.0)
    p.add_argument("--R0", type=float, default=-4.0)
    p.add_argument("--P0", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--boundary", choices=("dirichlet", "periodic"), default="dirichlet")
    p.add_argument("--derivative-scheme", choices=("finite-difference", "spectral"),
                   default="finite-difference",
                   help="Use finite differences for non-periodic conditional electronic fields.")
    p.add_argument("--density-threshold", type=float, default=1.0e-10)
    p.add_argument("--save-fields", action="store_true",
                   help="Also save full Phi(r,R), Psi(r,R), and U_en Phi snapshots.")
    p.add_argument("--no-pnc-projection", action="store_true")
    p.add_argument("--no-norm-correction", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
