#!/usr/bin/env python3
"""BO-free real-space propagation of the 1D exact-factorization equations.

This module never computes, stores, or projects onto BO eigenstates. The two
independent propagated fields are

    Phi_R(r,t)   shape (nr,nR)
    chi(R,t)     shape (nR,)

and the molecular wavefunction is reconstructed as Psi=Phi*chi. The symbol
H_BO in the exact-factorization equation is used only as the real-space
operator -d_r^2/2 + V(r,R); no electronic eigenvalue problem is solved.

The stiff electronic H_BO part is advanced with a unitary split operator in
r. The remaining nonlinear EF coupling and the nuclear equation are advanced
together with RK4. This Strang/partitioned integrator is a research prototype;
time-step and grid convergence must be checked against the accompanying full
TDSE reference.
"""

from __future__ import annotations

import argparse

import numpy as np

from shin_metiu_1d import AU_PER_FS
from result_paths import dated_results_dir

from .core import (
    apply_hbo,
    build_grid_model,
    coupling_from_phi,
    derivative,
    gaussian_conditional_electronic_state,
    gaussian_nuclear_state,
    phase_of_chi,
)


def derivative_boundary(args) -> str:
    """Translate the user-facing derivative choice to the core convention."""
    return "periodic" if args.derivative_scheme == "spectral" else "dirichlet"


def enforce_realspace_pnc(phi, chi, dr):
    """Restore int dr |Phi_R|^2=1 while preserving Psi=Phi_R*chi.

    Parameters/returns have shapes ``phi(nr,nR)`` and ``chi(nR)``. Moving the
    local norm from Phi into chi keeps their product unchanged pointwise.
    """
    local_norm = np.sqrt(np.sum(np.abs(phi) ** 2, axis=0) * dr)  # (nR,)
    error = float(np.max(np.abs(local_norm**2 - 1.0)))
    safe = np.where(local_norm > 1.0e-14, local_norm, 1.0)
    return phi / safe[None, :], chi * safe, error


def ef_functionals(phi, chi, model, args):
    """Return instantaneous A, epsilon, U_en Phi, and tail mask.

    Shapes are ``A(nR)``, ``epsilon(nR)``, ``u_phi(nr,nR)``, and ``mask(nR)``.
    In the epsilon_GD=0 gauge,

        epsilon(R)=<Phi_R|H_BO+U_en|Phi_R>_r.
    """
    A, u_phi, mask, _ = coupling_from_phi(
        phi, chi, model, args.density_threshold, derivative_boundary(args),
        norm_correction=not args.no_norm_correction,
    )
    h_phi = apply_hbo(phi, model) + u_phi                    # (nr,nR)
    epsilon = (np.sum(phi.conj() * h_phi, axis=0) * model.dr).real
    return A, epsilon, u_phi, mask


def coupling_nuclear_rhs(phi, chi, model, args):
    """Evaluate the non-H_BO electronic RHS and the full nuclear RHS.

    ``H_BO Phi`` is intentionally absent from ``dphi`` because that stiff
    subflow is handled by :func:`hbo_split_step`. The sum of both subflows is
    the complete exact-factorization equation.
    """
    A, epsilon, u_phi, mask = ef_functionals(phi, chi, model, args)

    # Electronic coupling subflow:
    # i d_t Phi = (U_en - epsilon) Phi.
    dphi = -1j * (u_phi - epsilon[None, :] * phi)            # (nr,nR)

    # Nuclear equation:
    # i d_t chi = [(-i d_R+A)^2/(2M)+epsilon] chi.
    mode = derivative_boundary(args)
    momentum_chi = -1j * derivative(chi, model.dR, boundary=mode) + A * chi
    kinetic_chi = (
        -1j * derivative(momentum_chi, model.dR, boundary=mode)
        + A * momentum_chi
    )
    dchi = -1j * (0.5 * kinetic_chi / model.mass + epsilon * chi)
    return dphi, dchi, A, epsilon, u_phi, mask


def rk4_coupling_step(phi, chi, dt, model, args):
    """Advance the coupled EF/nuclear subflow through one RK4 step."""
    k1p, k1x, *_ = coupling_nuclear_rhs(phi, chi, model, args)
    k2p, k2x, *_ = coupling_nuclear_rhs(
        phi + 0.5*dt*k1p, chi + 0.5*dt*k1x, model, args
    )
    k3p, k3x, *_ = coupling_nuclear_rhs(
        phi + 0.5*dt*k2p, chi + 0.5*dt*k2x, model, args
    )
    k4p, k4x, *_ = coupling_nuclear_rhs(phi + dt*k3p, chi + dt*k3x, model, args)
    phi = phi + dt*(k1p + 2*k2p + 2*k3p + k4p)/6.0
    chi = chi + dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
    if args.boundary == "dirichlet":
        chi[[0, -1]] = 0.0
    if not args.no_pnc_projection:
        phi, chi, pnc_before = enforce_realspace_pnc(phi, chi, model.dr)
    else:
        pnc_before = float(
            np.max(np.abs(np.sum(np.abs(phi)**2, axis=0)*model.dr - 1.0))
        )
    return phi, chi, pnc_before


def hbo_split_step(phi, tau, model):
    """Apply ``exp(-i H_BO tau)`` without diagonalizing H_BO.

    A second-order ``T_r/2 -> V -> T_r/2`` split is performed independently
    for every R column of ``phi(nr,nR)``. The operation is unitary in r and
    therefore preserves the electronic PNC up to roundoff.
    """
    kr = 2.0*np.pi*np.fft.fftfreq(len(model.r), d=model.dr)   # (nr,)
    half_kinetic = np.exp(-0.5j*tau*(0.5*kr**2))             # exp(-i T tau/2)
    phi = np.fft.ifft(
        np.fft.fft(phi, axis=0) * half_kinetic[:, None], axis=0
    )
    phi *= np.exp(-1j*tau*model.potential)                   # (nr,nR)
    phi = np.fft.ifft(
        np.fft.fft(phi, axis=0) * half_kinetic[:, None], axis=0
    )
    return phi


def full_step(phi, chi, dt, model, args):
    """One symmetric BO-free EF step.

    The decomposition is ``H_BO half -> coupled EF full -> H_BO half``.
    ``H_BO`` here is an operator application, not a BO-state calculation.
    """
    phi = hbo_split_step(phi, 0.5*dt, model)
    phi, chi, pnc_before = rk4_coupling_step(phi, chi, dt, model, args)
    phi = hbo_split_step(phi, 0.5*dt, model)
    # The H_BO subflow is theoretically unitary; a final projection removes
    # only accumulated FFT roundoff and still preserves Phi*chi.
    if not args.no_pnc_projection:
        phi, chi, pnc_after_hbo = enforce_realspace_pnc(phi, chi, model.dr)
        pnc_before = max(pnc_before, pnc_after_hbo)
    return phi, chi, pnc_before


def run(args):
    """Initialize BO-free fields, propagate them, and save the trajectory."""
    outdir = dated_results_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = build_grid_model(args)
    print("Built pure real-space model (no BO eigenproblem).")

    nuclear_sigma = args.sigma if args.sigma > 0 else 1.0/np.sqrt(2.85)
    chi = gaussian_nuclear_state(
        model.R, model.dR, args.R0, nuclear_sigma, args.P0
    )                                                               # (nR,)
    if args.boundary == "dirichlet":
        chi[[0, -1]] = 0.0
        chi /= np.sqrt(np.sum(np.abs(chi)**2)*model.dR)
    phi = gaussian_conditional_electronic_state(
        model, args.electron_center, args.electron_sigma,
        args.electron_momentum, args.electron_follow_nucleus,
    )                                                               # (nr,nR)

    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    save_steps = list(range(0, n_steps+1, max(1, args.save_every)))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    nt = len(save_steps)

    times = np.empty(nt)                                            # (nt,)
    phis = np.empty((nt, args.nr, args.nR), complex)                # (nt,nr,nR)
    chis = np.empty((nt, args.nR), complex)                         # (nt,nR)
    Avec = np.empty((nt, args.nR))                                  # (nt,nR)
    epsilon = np.empty((nt, args.nR))                               # (nt,nR)
    masks = np.empty((nt, args.nR))                                 # (nt,nR)
    norm = np.empty(nt)                                             # (nt,)
    pnc_error = np.empty(nt)                                        # (nt,)
    psis = []                                                       # optional (nt,nr,nR)
    uphis = []                                                      # optional (nt,nr,nR)

    def save_frame(frame, step, pnc_before=0.0):
        A, eps, uphi, mask = ef_functionals(phi, chi, model, args)
        times[frame] = step*args.dt_au/AU_PER_FS
        phis[frame] = phi
        chis[frame] = chi
        Avec[frame] = A
        epsilon[frame] = eps
        masks[frame] = mask
        psi = phi*chi[None, :]                                     # (nr,nR)
        norm[frame] = np.sum(np.abs(psi)**2)*model.dr*model.dR
        pnc_error[frame] = max(
            pnc_before,
            float(np.max(np.abs(np.sum(np.abs(phi)**2, axis=0)*model.dr-1))),
        )
        if args.save_psi:
            psis.append(psi.copy())
        if args.save_u_phi:
            uphis.append(uphi.copy())

    save_frame(0, 0)
    frame = 1
    last_pnc = 0.0
    for step in range(1, n_steps+1):
        phi, chi, last_pnc = full_step(phi, chi, args.dt_au, model, args)
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(chi)):
            raise FloatingPointError(f"Real-space EF became non-finite at step {step}")
        if frame < nt and step == save_steps[frame]:
            save_frame(frame, step, last_pnc)
            frame += 1
        if step % max(1, args.progress_every) == 0:
            current_norm = np.sum(np.abs(phi*chi[None,:])**2)*model.dr*model.dR
            print(
                f"step {step:6d}/{n_steps}  t={step*args.dt_au/AU_PER_FS:7.3f} fs"
                f"  norm={current_norm:.10f}  pnc(pre)={last_pnc:.2e}"
            )

    payload = dict(
        kind=np.array("direct_ef_realspace"),
        representation=np.array("realspace_no_bo"),
        gauge=np.array("epsilon_gd_zero"),
        r=model.r, R=model.R, times_fs=times,
        phi=phis, chi=chis, A=Avec, epsilon=epsilon, mask=masks,
        phase_S=np.array([phase_of_chi(x,m) for x,m in zip(chis,masks)]),
        norm=norm, pnc_error=pnc_error,
        args=np.array([vars(args)], dtype=object),
    )
    if args.save_psi:
        payload["psi"] = np.asarray(psis)
    if args.save_u_phi:
        payload["u_phi"] = np.asarray(uphis)
    path = outdir/"shin_metiu_direct_ef_realspace.npz"
    np.savez_compressed(path, **payload)
    print(f"Saved {path}")
    print(
        f"Saved shapes: phi={phis.shape}, chi={chis.shape}, A={Avec.shape}, "
        f"epsilon={epsilon.shape}, norm={norm.shape}"
    )
    print(f"Maximum molecular norm error: {np.max(np.abs(norm-1)):.3e}")
    print(f"Maximum pre-projection PNC error: {np.max(pnc_error):.3e}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results/exact_factorization/realspace_direct")
    p.add_argument("--nr", type=int, default=128)
    p.add_argument("--nR", type=int, default=256)
    p.add_argument("--r-min", type=float, default=-15.0)
    p.add_argument("--r-max", type=float, default=15.0)
    p.add_argument("--R-min", type=float, default=-8.0)
    p.add_argument("--R-max", type=float, default=8.0)
    p.add_argument("--dt-au", type=float, default=0.02)
    p.add_argument("--t-final-fs", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--mass", type=float, default=1836.0)
    p.add_argument("--L", type=float, default=19.0)
    p.add_argument("--Rf", type=float, default=5.0)
    p.add_argument("--Rl", type=float, default=3.1)
    p.add_argument("--Rr", type=float, default=4.0)
    p.add_argument("--R0", type=float, default=-4.0)
    p.add_argument("--P0", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Nuclear Gaussian width; default is 1/sqrt(2.85).")
    p.add_argument("--electron-center", type=float, default=-4.0,
                   help="Fixed center, or offset from R with --electron-follow-nucleus.")
    p.add_argument("--electron-sigma", type=float, default=1.0)
    p.add_argument("--electron-momentum", type=float, default=0.0)
    p.add_argument("--electron-follow-nucleus", action="store_true")
    p.add_argument("--boundary", choices=("dirichlet","periodic"), default="dirichlet")
    p.add_argument("--derivative-scheme", choices=("finite-difference","spectral"),
                   default="finite-difference")
    p.add_argument("--density-threshold", type=float, default=1e-10)
    p.add_argument("--save-psi", action="store_true",
                   help="Store reconstructible Psi=Phi*chi frames.")
    p.add_argument("--save-u-phi", action="store_true",
                   help="Store the large differential action U_en Phi.")
    p.add_argument("--no-pnc-projection", action="store_true")
    p.add_argument("--no-norm-correction", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
