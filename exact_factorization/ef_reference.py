#!/usr/bin/env python3
"""Generate a full-TDSE reference and extract exact-factorization fields.

The full Psi(r,R,t) is propagated independently. It is then factorized in the
one-dimensional A=0 gauge using the exact nuclear density and current. This is
the benchmark against which a direct EF propagation should be compared.

Shape convention:
    psi, phi, U_en phi     (nt,nr,nR)
    chi, A, epsilon, S     (nt,nR)
    BO coefficients        (nt,ns,nR)
Here ``nt`` counts saved frames; the full TDSE is still advanced through many
smaller integration steps between two frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shin_metiu_1d import AU_PER_FS

from .core import (
    apply_hbo,
    build_model,
    coupling_from_phi,
    derivative,
    normalize_initial_chi,
    phase_of_chi,
    project_electronic,
)


def integrate_phase_gradient(gradient, R, weight):
    """Integrate a ``(nR,)`` phase gradient outwards from maximum density.

    Starting in the occupied region avoids accumulating arbitrary phase noise
    from the nearly empty left tail before reaching the wavepacket.
    """
    anchor = int(np.argmax(weight))
    phase = np.zeros_like(gradient, dtype=float)
    for j in range(anchor + 1, len(R)):
        phase[j] = phase[j-1] + 0.5 * (gradient[j-1] + gradient[j]) * (R[j]-R[j-1])
    for j in range(anchor - 1, -1, -1):
        phase[j] = phase[j+1] - 0.5 * (gradient[j+1] + gradient[j]) * (R[j+1]-R[j])
    return phase


def factorize_A_zero(psi_frames, times_au, model, threshold, boundary):
    """Factorize saved ``Psi(nt,nr,nR)`` frames in the one-dimensional A=0 gauge.

    Returns a dictionary of EF trajectories. The independent factorization
    fields are ``chi(nt,nR)`` and ``phi(nt,nr,nR)``. All other arrays are
    derived from them and are documented next to their allocation below.
    """
    nframes = len(times_au)
    chi = np.empty((nframes, len(model.R)), complex) # (nt,nR)
    phi = np.empty_like(psi_frames)                  # (nt,nr,nR)
    mask = np.empty((nframes, len(model.R)))         # (nt,nR)
    current = np.empty((nframes, len(model.R)))      # (nt,nR)

    for n, psi in enumerate(psi_frames):
        # Nuclear marginal density rho_n(R)=int dr |Psi(r,R)|^2; shape (nR,).
        density = np.sum(np.abs(psi)**2, axis=0) * model.dr
        floor = max(threshold * float(density.max()), np.finfo(float).tiny)
        mask[n] = density / (density + floor)
        # Exact nuclear probability current from the full molecular field.
        dpsi = derivative(psi, model.dR, axis=1, boundary=boundary)
        current[n] = (
            np.sum((psi.conj() * dpsi).imag, axis=0) * model.dr / model.mass
        )
        # Start from a real positive chi and iteratively gauge-transform the
        # conditional field until its Berry connection vanishes under the
        # same finite-difference operator used for EF post-processing.
        # Any factorization may start with |chi|=sqrt(rho_n). Its phase is then
        # fixed by gauge. Division by chi gives the conditional Phi_R.
        chi_n = np.sqrt(density).astype(complex)                    # (nR,)
        phi_n = psi * chi_n.conj()[None, :] / (density[None, :] + floor) # (nr,nR)
        for _ in range(4):
            # Gauge iteration: Phi->exp(i theta)Phi and
            # chi->exp(-i theta)chi preserve Psi=Phi*chi while driving A to 0.
            dphi = derivative(phi_n, model.dR, axis=1, boundary="dirichlet")
            residual_A = (
                np.sum(phi_n.conj()*(-1j*dphi), axis=0)*model.dr
            ).real * mask[n]
            theta = integrate_phase_gradient(-residual_A, model.R, density)
            phi_n *= np.exp(1j*theta)[None, :]
            chi_n *= np.exp(-1j*theta)
        chi[n] = chi_n
        phi[n] = phi_n

    # epsilon inversion requires d_t chi, so at least three saved frames are
    # needed. Fine --save-every improves this temporal finite difference.
    dchi_dt = (
        np.gradient(chi, times_au, axis=0, edge_order=2)
        if nframes >= 3 else np.zeros_like(chi)
    )                                                       # (nt,nR)
    epsilon = np.empty((nframes, len(model.R)))             # (nt,nR), total
    epsilon_imag = np.empty_like(epsilon)                   # inversion error
    epsilon_gi = np.empty_like(epsilon)                     # (nt,nR)
    epsilon_gd = np.empty_like(epsilon)                     # (nt,nR)
    A = np.empty_like(epsilon)                              # (nt,nR)
    u_coeff = np.empty(
        (nframes, len(model.bo_energies), len(model.R)), complex
    )                                                       # (nt,ns,nR)
    coefficients = np.empty_like(u_coeff)                   # (nt,ns,nR)
    u_action = np.empty_like(phi)                           # (nt,nr,nR)

    for n in range(nframes):
        # Evaluate the same EF functionals used by the direct solver, but on
        # factors extracted from the independently propagated exact Psi.
        A_n, uphi, _, _ = coupling_from_phi(
            phi[n], chi[n], model, threshold, "dirichlet", norm_correction=False
        )
        # The phase was reconstructed in the A=0 gauge. Residual A measures
        # spatial/time discretization error and is retained as a diagnostic.
        A[n] = A_n
        u_action[n] = uphi
        coefficients[n] = project_electronic(phi[n], model)
        u_coeff[n] = project_electronic(uphi, model)
        # epsilon_GI(R)=<Phi_R|H_BO+U_en|Phi_R>_r.
        hphi = apply_hbo(phi[n], model) + uphi
        epsilon_gi[n] = (
            np.sum(phi[n].conj() * hphi, axis=0) * model.dr
        ).real
        if nframes >= 3:
            # In A=0 gauge the nuclear equation can be inverted pointwise:
            # epsilon=(i d_t chi + d_R^2 chi/(2M))/chi. The imaginary part
            # should vanish in the occupied region and is stored as an error
            # diagnostic instead of being silently discarded.
            kinetic_chi = derivative(
                derivative(chi[n], model.dR, boundary="dirichlet"),
                model.dR, boundary="dirichlet",
            )
            inverted = (1j*dchi_dt[n] + 0.5*kinetic_chi/model.mass)
            inverted = inverted * chi[n].conj() / (
                np.abs(chi[n])**2 + threshold*np.max(np.abs(chi[n])**2)
            )
            epsilon[n] = inverted.real
            epsilon_imag[n] = inverted.imag
            epsilon_gd[n] = epsilon[n] - epsilon_gi[n]
        else:
            epsilon[n] = np.nan
            epsilon_imag[n] = np.nan
            epsilon_gd[n] = np.nan
    return dict(
        chi=chi, phi=phi, mask=mask, nuclear_current=current, A=A,
        epsilon=epsilon, epsilon_imag=epsilon_imag,
        epsilon_gi=epsilon_gi, epsilon_gd=epsilon_gd,
        coefficients=coefficients, u_coefficients=u_coeff, u_phi=u_action,
        phase_S=np.array([phase_of_chi(x, m) for x,m in zip(chi,mask)]),
    )


def run(args):
    """Propagate full ``Psi(nr,nR)``, factorize its frames, and save them."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.boundary != "periodic":
        raise ValueError("The FFT full-TDSE reference currently requires --boundary periodic")
    print("Building grids and BO basis...")
    model = build_model(args)
    sigma = args.sigma if args.sigma > 0 else 1.0 / np.sqrt(2.85)
    chi0 = np.exp(
        -0.5*((model.R-args.R0)/sigma)**2
        + 1j*args.P0*(model.R-args.R0)
    )
    chi0 = normalize_initial_chi(chi0, model.dR)
    # Psi(r,R,0)=phi_initial(r;R) chi(R,0), shape (nr,nR).
    psi = model.bo_states[args.initial_state].T * chi0[None, :]

    # The 2D kinetic operator is diagonal in joint (k_r,k_R) Fourier space.
    kr = 2*np.pi*np.fft.fftfreq(args.nr, d=model.dr)
    kR = 2*np.pi*np.fft.fftfreq(args.nR, d=model.dR)
    kinetic = kr[:,None]**2/2 + kR[None,:]**2/(2*model.mass)
    half_T = np.exp(-0.5j*args.dt_au*kinetic)
    full_V = np.exp(-1j*args.dt_au*model.potential)
    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    save_steps = list(range(0, n_steps+1, max(1,args.save_every)))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    frames = []
    times_au = []
    norms = []

    def save(step):
        """Append one complex Psi(nr,nR) frame and its full norm."""
        frames.append(psi.copy())
        times_au.append(step*args.dt_au)
        norms.append(np.sum(np.abs(psi)**2)*model.dr*model.dR)

    save(0)
    frame = 1
    for step in range(1,n_steps+1):
        # Second-order Strang step: exp(-iT dt/2) exp(-iV dt) exp(-iT dt/2).
        psi = np.fft.ifftn(np.fft.fftn(psi)*half_T)
        psi *= full_V
        psi = np.fft.ifftn(np.fft.fftn(psi)*half_T)
        if frame < len(save_steps) and step == save_steps[frame]:
            save(step)
            frame += 1
        if step % max(1,args.progress_every) == 0:
            print(f"step {step:6d}/{n_steps}  t={step*args.dt_au/AU_PER_FS:7.3f} fs")

    psi_frames = np.asarray(frames)
    times_au = np.asarray(times_au)
    print("Extracting A=0-gauge exact-factorization fields...")
    ef = factorize_A_zero(
        psi_frames, times_au, model, args.density_threshold, args.boundary
    )
    # F_s(R,t)=<phi_s(R)|Psi(t)>_r=C_s(R,t)chi(R,t).
    # Integration over R gives arrays P_s(t) with shape (nt,ns).
    populations = np.sum(
        np.abs(ef["coefficients"]*ef["chi"][:,None,:])**2,
        axis=2,
    )*model.dR
    payload = dict(
        kind=np.array("full_tdse_ef_reference"), gauge=np.array("A_zero"),
        r=model.r, R=model.R,
        times_fs=times_au/AU_PER_FS, psi=psi_frames,
        populations=populations, norm=np.asarray(norms),
        bo_energies=model.bo_energies, bo_states=model.bo_states,
        args=np.array([vars(args)],dtype=object), **ef,
    )
    if args.compact:
        payload.pop("phi")
        payload.pop("u_phi")
    path = outdir/"shin_metiu_ef_reference.npz"
    np.savez_compressed(path,**payload)
    print(f"Saved {path}")
    print(
        "Computed reference shapes: "
        f"psi={psi_frames.shape}, chi={ef['chi'].shape}, "
        f"phi={ef['phi'].shape}, A={ef['A'].shape}, "
        f"epsilon={ef['epsilon'].shape}, "
        f"coefficients={ef['coefficients'].shape}, "
        f"populations={populations.shape}"
    )
    if args.compact:
        print("Compact archive: phi and u_phi were computed but omitted from NPZ.")
    print(f"Maximum full-TDSE norm error: {np.max(np.abs(np.asarray(norms)-1)):.3e}")
    occupied = ef["mask"] > 0.9999
    residual_A = np.max(np.abs(ef["A"])[occupied]) if np.any(occupied) else np.nan
    print(f"Maximum high-density |A| in nominal A=0 gauge: {residual_A:.3e}")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir",default="results/exact_factorization/reference")
    p.add_argument("--nr",type=int,default=128)
    p.add_argument("--nR",type=int,default=256)
    p.add_argument("--r-min",type=float,default=-15.0)
    p.add_argument("--r-max",type=float,default=15.0)
    p.add_argument("--R-min",type=float,default=-8.0)
    p.add_argument("--R-max",type=float,default=8.0)
    p.add_argument("--dt-au",type=float,default=0.1)
    p.add_argument("--t-final-fs",type=float,default=0.1)
    p.add_argument("--save-every",type=int,default=5,
                   help="Fine time sampling is needed for epsilon inversion.")
    p.add_argument("--progress-every",type=int,default=100)
    p.add_argument("--n-states",type=int,default=2)
    p.add_argument("--initial-state",type=int,default=1)
    p.add_argument("--mass",type=float,default=1836.0)
    p.add_argument("--L",type=float,default=19.0)
    p.add_argument("--Rf",type=float,default=5.0)
    p.add_argument("--Rl",type=float,default=3.1)
    p.add_argument("--Rr",type=float,default=4.0)
    p.add_argument("--R0",type=float,default=-4.0)
    p.add_argument("--P0",type=float,default=0.0)
    p.add_argument("--sigma",type=float,default=0.0)
    p.add_argument("--boundary",choices=("periodic",),default="periodic")
    p.add_argument("--density-threshold",type=float,default=1e-10)
    p.add_argument("--compact",action="store_true",
                   help="Omit reconstructible Phi and U_en Phi arrays.")
    return p.parse_args()


if __name__=="__main__":
    run(parse_args())
