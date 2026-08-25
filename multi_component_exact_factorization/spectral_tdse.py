"""Reference second-order spectral split operator for the full 3D TDSE.

The electronic coordinate uses the existing hard-wall interior grid and an
orthonormal DST-I.  The q and R coordinates retain the existing periodic
grids and use orthonormal FFTs.  Consequently this module changes neither the
stored grids nor the coupled MCEF finite-difference backends; it is the
independent full-TDSE reference requested for Feit--Fleck--Steiger-style
propagation.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dst, fft, fftfreq, idst, ifft


def spectral_kinetic_energies(model):
    """Return continuum-spectral ``(T_x,T_q,T_R)`` eigenvalue arrays."""
    nx = len(model.x)
    length_x = float(model.x_right-model.x_left)
    mode_x = np.arange(1, nx+1, dtype=float)
    kinetic_x = 0.5*(np.pi*mode_x/length_x)**2
    wave_q = 2.0*np.pi*fftfreq(len(model.q), d=model.dq)
    wave_R = 2.0*np.pi*fftfreq(len(model.R), d=model.dR)
    kinetic_q = wave_q**2/(2.0*model.proton_mass)
    kinetic_R = wave_R**2/(2.0*model.heavy_mass)
    return kinetic_x, kinetic_q, kinetic_R


def apply_spectral_kinetic_numpy(wavefunction, tau, model):
    """Apply ``exp(-i*tau*(T_x+T_q+T_R))`` without a finite stencil."""
    kinetic_x, kinetic_q, kinetic_R = spectral_kinetic_energies(model)
    values = dst(wavefunction, type=1, axis=0, norm="ortho")
    values *= np.exp(-1j*tau*kinetic_x)[:, None, None]
    values = idst(values, type=1, axis=0, norm="ortho")
    values = fft(values, axis=1, norm="ortho")
    values *= np.exp(-1j*tau*kinetic_q)[None, :, None]
    values = ifft(values, axis=1, norm="ortho")
    values = fft(values, axis=2, norm="ortho")
    values *= np.exp(-1j*tau*kinetic_R)[None, None, :]
    return ifft(values, axis=2, norm="ortho")


def split_step_numpy(wavefunction, dt, model):
    """One time-reversible second-order Strang/Feit split step.

    ``exp(-i V dt/2) exp(-i T dt) exp(-i V dt/2)`` has local error
    ``O(dt**3)`` and global error ``O(dt**2)``.  Each factor is unitary to
    roundoff for a real potential.
    """
    half = np.exp(-0.5j*dt*model.potential)
    values = half*wavefunction
    values = apply_spectral_kinetic_numpy(values, dt, model)
    return half*values


def spectral_action_numpy(wavefunction, model):
    """Apply the full continuum-spectral Hamiltonian ``H Psi``."""
    kinetic_x, kinetic_q, kinetic_R = spectral_kinetic_energies(model)
    x_modes = dst(wavefunction, type=1, axis=0, norm="ortho")
    x_action = idst(
        x_modes*kinetic_x[:, None, None], type=1, axis=0, norm="ortho"
    )
    q_modes = fft(wavefunction, axis=1, norm="ortho")
    q_action = ifft(
        q_modes*kinetic_q[None, :, None], axis=1, norm="ortho"
    )
    R_modes = fft(wavefunction, axis=2, norm="ortho")
    R_action = ifft(
        R_modes*kinetic_R[None, None, :], axis=2, norm="ortho"
    )
    return model.potential*wavefunction+x_action+q_action+R_action


def spectral_energy_numpy(wavefunction, model):
    """Return the real spectral energy decomposition and full norm."""
    density = np.real(wavefunction*np.conj(wavefunction))
    cell = model.dx*model.dq*model.dR
    norm = float(np.sum(density, dtype=np.float64)*cell)
    kinetic_x, kinetic_q, kinetic_R = spectral_kinetic_energies(model)
    x_modes = dst(wavefunction, type=1, axis=0, norm="ortho")
    q_modes = fft(wavefunction, axis=1, norm="ortho")
    R_modes = fft(wavefunction, axis=2, norm="ortho")
    tx = float(np.sum(np.abs(x_modes)**2*kinetic_x[:, None, None])*cell)
    tq = float(np.sum(np.abs(q_modes)**2*kinetic_q[None, :, None])*cell)
    tR = float(np.sum(np.abs(R_modes)**2*kinetic_R[None, None, :])*cell)
    potential = float(np.sum(density*model.potential, dtype=np.float64)*cell)
    return {
        "norm": norm,
        "kinetic_x": tx,
        "kinetic_q": tq,
        "kinetic_R": tR,
        "potential": potential,
        "energy": tx+tq+tR+potential,
    }
