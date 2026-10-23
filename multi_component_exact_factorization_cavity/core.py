"""Frozen-heavy molecular model and untruncated real-space initial states.

The molecular potential is exactly the ``erf_shin_metiu`` branch of historical
``multi_component_exact_factorization/core.py`` with its old heavy coordinate
fixed.  The old proton ``q`` is called ``R`` here; ``Q=sqrt(omega)*q_c`` is new.
All charges are unit charges and the electronic mass is one (atomic units).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dst

from multi_component_exact_factorization.core import bare_inverse, erf_inverse


HISTORICAL_ARCHIVE = (
    "results/20260916/"
    "erf_fullgrid_BOcurvature_compact_qR_analysisBO2_100fs/"
    "multi_component_discrete_tdse_gpu.npz"
)

# Values read from HISTORICAL_ARCHIVE['args'], not from a toy-model paper.
# core.gaussian_wavepacket uses exp(-(R-R0)^2/(4*sigma^2)): sigma is density SD.
DEFAULTS = {
    "nx": 359, "nR": 750, "nQ": 128,
    "x_left": -22.0, "x_right": 30.0,
    "R_left": -12.012, "R_right": 11.988, "Q_max": 10.0,
    "mass": 1836.0, "heavy_fixed": 9.5, "left_fixed": -9.5,
    "heavy_trap_alpha": 0.03, "heavy_trap_center": 9.5,
    "erf_r_lx": 3.1, "erf_r_qx": 5.0,
    "erf_r_Rx": 4.0, "erf_r_qR": 1.0,
    "R0": -4.0, "sigma": 0.3208418789467482, "p0": 0.0,
    "n_bo": 4, "n_vib": 6, "n_fock_analysis": 8,
    "n_pol_plot": 4, "weak_splitting_fraction": 0.01,
    "strong_splitting_fraction": 0.10, "detuning": 0.0, "dse": True,
    "vsc_analysis_model": "projected_full",
    # User-confirmed continuation: use the actual inherited single-well v=1,
    # not an artificial R=0 preparation wall or a claimed double-well state.
    "local_state_method": "single_well",
}


def configuration(config=None):
    """Return explicit editable parameters; ERF and cavity strengths differ."""
    values = dict(DEFAULTS)
    values.update(config or {})
    for name in ("nx", "nR", "nQ", "n_bo", "n_vib", "n_fock_analysis"):
        values[name] = int(values[name])
        if values[name] < 2:
            raise ValueError(f"{name} must be at least two")
    if values["n_bo"] > values["nx"]:
        raise ValueError("n_bo exceeds real electronic grid size")
    for name in ("mass", "sigma", "Q_max", "erf_r_lx", "erf_r_qx",
                 "erf_r_Rx", "erf_r_qR"):
        if not np.isfinite(values[name]) or values[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not values["x_left"] < values["x_right"]:
        raise ValueError("x_left must be smaller than x_right")
    if not values["R_left"] < values["R0"] < values["R_right"]:
        raise ValueError("historical nuclear Gaussian center is outside R box")
    if values["detuning"] <= -1.0:
        raise ValueError("fractional detuning must exceed -1")
    return values


def sine_kinetic(n, spacing, mass=1.0):
    """Dense DST-I kinetic and its eigenvalues, identical to spectral TDSE."""
    length = (int(n) + 1) * float(spacing)
    energy = 0.5 * (np.pi * np.arange(1, n + 1) / length)**2 / mass
    transform = dst(np.eye(n), type=1, norm="ortho", axis=0)
    return (transform * energy[None, :]) @ transform.T, energy


def frozen_potential(x, R, config):
    """Historical potential at fixed heavy, including the ion-ion constant.

    The left proton-ion Coulomb pole is *not* smoothed or clipped.  Sampling a
    pole is an invalid grid and raises, just as the historical model does.
    The fixed left/right sites contribute once each to the total dipole.
    """
    c = configuration(config)
    xx = np.asarray(x)[None, :]
    rr = np.asarray(R)[:, None]
    left, heavy = c["left_fixed"], c["heavy_fixed"]
    return (
        bare_inverse(rr-left, label="left-fixed/proton")
        - erf_inverse(xx-left, c["erf_r_lx"])
        + bare_inverse(np.asarray(heavy-left), label="left-fixed/heavy")
        - erf_inverse(rr-xx, c["erf_r_qx"])
        - erf_inverse(heavy-xx, c["erf_r_Rx"])
        + erf_inverse(rr-heavy, c["erf_r_qR"])
        + c["heavy_trap_alpha"]*(heavy-c["heavy_trap_center"])**2
    )


def build_molecular(config=None):
    """Create electronic Dirichlet / nuclear periodic grids and bare model."""
    c = configuration(config)
    dx = (c["x_right"]-c["x_left"])/(c["nx"]+1)
    dR = (c["R_right"]-c["R_left"])/c["nR"]
    x = c["x_left"] + dx*np.arange(1, c["nx"]+1)
    R = c["R_left"] + dR*np.arange(c["nR"])
    Tx, kinetic_x = sine_kinetic(c["nx"], dx)
    return dict(
        config=c, x=x, R=R, dx=dx, dR=dR, mass=float(c["mass"]),
        Tx=Tx, kinetic_x=kinetic_x, V=frozen_potential(x, R, c),
        mu=R[:, None]-x[None, :]+c["left_fixed"]+c["heavy_fixed"],
    )


def photon_grid(config):
    c = configuration(config)
    return np.linspace(-c["Q_max"], c["Q_max"], c["nQ"], endpoint=False)


def gaussian(R, dR, center, sigma, momentum=0.0):
    """Historical density-standard-deviation convention, normalized on grid."""
    value = np.exp(-0.25*((R-center)/sigma)**2 + 1j*momentum*(R-center))
    return value/np.sqrt(np.sum(np.abs(value)**2)*dR)


def vacuum(Q):
    """Normalized photon oscillator vacuum in dimensionless Q representation."""
    Q = np.asarray(Q)
    wave = np.pi**(-0.25)*np.exp(-0.5*Q**2)
    return wave/np.sqrt(np.sum(wave**2)*(Q[1]-Q[0]))


def initial_wave(packet, Q, campaign):
    """Bare electronic eigenstate × chosen nucleus × vacuum; no BO TDSE."""
    if campaign not in ("pen", "vsc"):
        raise ValueError("campaign must be pen or vsc")
    state = 1 if campaign == "pen" else 0
    chi = packet["chi_pen" if campaign == "pen" else "chi_vsc"]
    wave = packet["bo_phi"][:, :, state, None]*chi[:, None, None]*vacuum(Q)[None, None, :]
    return np.asarray(wave, dtype=np.complex128)
