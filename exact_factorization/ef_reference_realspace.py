#!/usr/bin/env python3
"""BO-free full-TDSE reference for the real-space direct EF solver.

The initial conditional electronic field is an explicitly normalized Gaussian,
not a BO eigenstate. The molecular field Psi(r,R,t) is propagated with the
full two-dimensional TDSE and subsequently factorized into chi and Phi in the
A=0 gauge. No BO eigenproblem or BO projection is performed anywhere.
"""

from __future__ import annotations

import argparse

import numpy as np

from shin_metiu_1d import AU_PER_FS
from result_paths import dated_results_dir

from .core import (
    build_grid_model,
    gaussian_conditional_electronic_state,
    gaussian_nuclear_state,
)
from .ef_reference import factorize_A_zero


def run(args):
    """Propagate full ``Psi(nr,nR)`` and extract BO-free EF reference fields."""
    if args.boundary != "periodic":
        raise ValueError("The FFT reference requires --boundary periodic")
    outdir = dated_results_dir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = build_grid_model(args)
    print("Built pure real-space reference model (no BO eigenproblem).")

    nuclear_sigma = args.sigma if args.sigma > 0 else 1.0/np.sqrt(2.85)
    chi0 = gaussian_nuclear_state(
        model.R, model.dR, args.R0, nuclear_sigma, args.P0
    )                                                               # (nR,)
    phi0 = gaussian_conditional_electronic_state(
        model, args.electron_center, args.electron_sigma,
        args.electron_momentum, args.electron_follow_nucleus,
    )                                                               # (nr,nR)
    psi = phi0*chi0[None, :]                                        # (nr,nR)

    # Full molecular split operator in the joint (r,R) Fourier grid.
    kr = 2*np.pi*np.fft.fftfreq(args.nr, d=model.dr)                 # (nr,)
    kR = 2*np.pi*np.fft.fftfreq(args.nR, d=model.dR)                 # (nR,)
    kinetic = kr[:,None]**2/2 + kR[None,:]**2/(2*model.mass)        # (nr,nR)
    half_T = np.exp(-0.5j*args.dt_au*kinetic)
    full_V = np.exp(-1j*args.dt_au*model.potential)

    n_steps = int(round(args.t_final_fs*AU_PER_FS/args.dt_au))
    save_steps = list(range(0,n_steps+1,max(1,args.save_every)))
    if save_steps[-1] != n_steps:
        save_steps.append(n_steps)
    psi_frames = []                                                 # frames of (nr,nR)
    times_au = []
    norm = []

    def save_frame(step):
        psi_frames.append(psi.copy())
        times_au.append(step*args.dt_au)
        norm.append(np.sum(np.abs(psi)**2)*model.dr*model.dR)

    save_frame(0)
    frame = 1
    for step in range(1,n_steps+1):
        psi = np.fft.ifftn(np.fft.fftn(psi)*half_T)
        psi *= full_V
        psi = np.fft.ifftn(np.fft.fftn(psi)*half_T)
        if frame < len(save_steps) and step == save_steps[frame]:
            save_frame(step)
            frame += 1
        if step % max(1,args.progress_every) == 0:
            print(f"step {step:6d}/{n_steps}  t={step*args.dt_au/AU_PER_FS:7.3f} fs")

    psi_frames = np.asarray(psi_frames)                             # (nt,nr,nR)
    times_au = np.asarray(times_au)                                 # (nt,)
    print("Extracting pure real-space A=0-gauge EF fields...")
    ef = factorize_A_zero(
        psi_frames, times_au, model, args.density_threshold, args.boundary
    )
    # With an empty BO basis the helper returns empty coefficient arrays.
    # They have no physical role here and are removed from this archive.
    ef.pop("coefficients")
    ef.pop("u_coefficients")
    if args.compact:
        ef.pop("u_phi")

    payload = dict(
        kind=np.array("full_tdse_ef_reference_realspace"),
        representation=np.array("realspace_no_bo"),
        gauge=np.array("A_zero"),
        r=model.r, R=model.R, times_fs=times_au/AU_PER_FS,
        psi=psi_frames, norm=np.asarray(norm),
        args=np.array([vars(args)],dtype=object), **ef,
    )
    path = outdir/"shin_metiu_ef_reference_realspace.npz"
    np.savez_compressed(path, **payload)
    print(f"Saved {path}")
    print(
        f"Saved shapes: psi={psi_frames.shape}, phi={ef['phi'].shape}, "
        f"chi={ef['chi'].shape}, A={ef['A'].shape}, epsilon={ef['epsilon'].shape}"
    )
    print(f"Maximum full-TDSE norm error: {np.max(np.abs(np.asarray(norm)-1)):.3e}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir",default="results/exact_factorization/realspace_reference")
    p.add_argument("--nr",type=int,default=128)
    p.add_argument("--nR",type=int,default=256)
    p.add_argument("--r-min",type=float,default=-15.0)
    p.add_argument("--r-max",type=float,default=15.0)
    p.add_argument("--R-min",type=float,default=-8.0)
    p.add_argument("--R-max",type=float,default=8.0)
    p.add_argument("--dt-au",type=float,default=0.02)
    p.add_argument("--t-final-fs",type=float,default=0.1)
    p.add_argument("--save-every",type=int,default=5,
                   help="Fine saved times improve epsilon inversion.")
    p.add_argument("--progress-every",type=int,default=100)
    p.add_argument("--mass",type=float,default=1836.0)
    p.add_argument("--L",type=float,default=19.0)
    p.add_argument("--Rf",type=float,default=5.0)
    p.add_argument("--Rl",type=float,default=3.1)
    p.add_argument("--Rr",type=float,default=4.0)
    p.add_argument("--R0",type=float,default=-4.0)
    p.add_argument("--P0",type=float,default=0.0)
    p.add_argument("--sigma",type=float,default=0.0)
    p.add_argument("--electron-center",type=float,default=-4.0)
    p.add_argument("--electron-sigma",type=float,default=1.0)
    p.add_argument("--electron-momentum",type=float,default=0.0)
    p.add_argument("--electron-follow-nucleus",action="store_true")
    p.add_argument("--boundary",choices=("periodic",),default="periodic")
    p.add_argument("--density-threshold",type=float,default=1e-10)
    p.add_argument("--compact",action="store_true",
                   help="Omit U_en Phi; full Psi and extracted Phi remain saved.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
