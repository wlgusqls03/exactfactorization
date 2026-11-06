"""Fewest-switches events on declared internal electron--bridge surfaces.

This is the adiabatic FSSH baseline of Tully, JCP 93, 1061 (1990), Eq. (19),
https://doi.org/10.1063/1.459170, not a new hopping law derived from the Gaussian
projection. The real-coupling limit also agrees with ``shopping.f90:hopping``
(TSHFS) in the pinned G-CTMQC source. We do NOT copy that engine's optional
forced hops, energy sharing, or decoherence corrections. Default policy: no
decoherence, no amplitude collapse, frustrated hops keep the old momentum.

Conventions: A[I,n], d[I,n,m] = <u_n|partial_R u_m>, d^dagger = -d,
and A_dot = -i E A - v d A. Thus J[n,m] = 2 Re(A[n]* v d[n,m] A[m])
is population leaving n for m, and p_dot[n] = -sum_m J[n,m]. Only this native
NAC flux enters hops: population changes caused by Gaussian compression must
be diagnosed separately by the caller. All units are atomic units.

The caller integrates the continuous equations up to an event boundary and
passes a sufficiently short interval dt. Probabilities exceeding one raise
an error requiring a smaller event/electronic step; they are never clipped.
Randomness is exclusively through a caller-owned numpy.random.Generator;
checkpoint its bit_generator.state along with the active-state array.
"""
from dataclasses import dataclass

import numpy as np


class HoppingStepError(ValueError):
    """Invalid event interval or an unsupported vanishing active population."""


def _coefficients(A):
    A = np.asarray(A, dtype=np.complex128)
    if A.ndim != 2 or min(A.shape) == 0 or not np.all(np.isfinite(A)):
        raise ValueError('A must be a finite nonempty (trajectory, state) array')
    if np.any(np.sum(abs(A)**2, axis=1) <= 0):
        raise ValueError('A has a zero-norm trajectory')
    return A


def _couplings(A, d, velocity):
    d = np.asarray(d, dtype=np.complex128)
    velocity = np.broadcast_to(np.asarray(velocity, dtype=float), (len(A),))
    if d.shape != A.shape + (A.shape[1],):
        raise ValueError('d must have shape (trajectory, state, state)')
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(velocity)):
        raise ValueError('Nonfinite coupling or velocity')
    defect = np.max(abs(d + d.conj().swapaxes(-1, -2)))
    if defect > 1e-10 * max(1., float(np.max(abs(d)))):
        raise ValueError('Derivative coupling must be anti-Hermitian')
    return d, velocity


def _indices(indices, ntraj, nstates, name):
    raw = np.asarray(indices)
    if raw.shape != (ntraj,) or raw.dtype.kind not in 'iu':
        raise ValueError(name + ' must be an integer array with one entry per trajectory')
    if np.any(raw < 0) or np.any(raw >= nstates):
        raise ValueError(name + ' contains an out-of-range channel')
    return raw.astype(np.int64, copy=False)


def coefficient_rhs(A, energies, d, velocity):
    """Native, phase-coherent adiabatic amplitudes; no Gaussian/no-decoherence term."""
    A = _coefficients(A)
    d, velocity = _couplings(A, d, velocity)
    energies = np.asarray(energies, dtype=float)
    if energies.shape != A.shape or not np.all(np.isfinite(energies)):
        raise ValueError('energies must be finite with shape A.shape')
    return -1j*energies*A - velocity[:, None]*np.einsum('Inm,Im->In', d, A)


def pair_population_flux(A, d, velocity):
    """J[I,n,m] leaves n and enters m; invariant under channel phase changes.

    A' = exp(-i theta) A and d' = exp(i(theta_m-theta_n)) d off diagonal.
    The diagonal Berry connection contributes phase only, not hopping flux.
    """
    A = _coefficients(A)
    d, velocity = _couplings(A, d, velocity)
    flux = 2*np.real(A.conj()[:, :, None]*d*A[:, None, :])*velocity[:, None, None]
    diagonal = np.arange(A.shape[1])
    flux[:, diagonal, diagonal] = 0.
    return flux


def sample_active_states(A, rng):
    """Draw initial active surfaces with probabilities |A_n|^2 / sum_m |A_m|^2.

    This normalizes the *sampling distribution* only; amplitudes are unchanged.
    The propagation driver must separately record/check their raw norm defect.
    """
    A = _coefficients(A)
    if not isinstance(rng, np.random.Generator):
        raise TypeError('Use an explicit numpy.random.Generator')
    population = abs(A)**2
    cumulative = np.cumsum(population/population.sum(axis=1, keepdims=True), axis=1)
    cumulative[:, -1] = 1.  # exact distribution endpoint, not state renormalization
    return np.sum(rng.random(len(A))[:, None] >= cumulative, axis=1).astype(np.int64)


def hopping_probabilities(A, d, velocity, active, dt, *, population_tolerance=1e-14):
    """Tully Eq. (19), positive outgoing native flux from the active surface.

    Return [trajectory, target] with a zero diagonal/self-hop probability.
    This API intentionally accepts no compressed-state finite population change.
    Timestep convergence remains required even when every row sum is below one.
    """
    A = _coefficients(A)
    active = _indices(active, *A.shape, 'active')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if not np.isfinite(population_tolerance) or population_tolerance < 0:
        raise ValueError('population_tolerance must be finite and nonnegative')
    population = abs(A)**2
    rows = np.arange(len(A))
    active_population = population[rows, active]
    # Use a relative criterion so an overall coefficient scale cannot alter hops.
    bad = active_population <= population_tolerance*population.sum(axis=1)
    if np.any(bad):
        raise HoppingStepError('Vanishing active-state population at trajectories %s; '
                               'refine the event step, do not replace the denominator'
                               % np.flatnonzero(bad).tolist())
    flux = pair_population_flux(A, d, velocity)
    probabilities = dt*np.maximum(flux[rows, active], 0.)/active_population[:, None]
    probabilities[rows, active] = 0.
    sums = probabilities.sum(axis=1)
    if not np.all(np.isfinite(sums)) or np.any(sums > 1.):
        raise HoppingStepError('FSSH probability sum exceeds one (max=%g); '
                               'reduce dt / increase electronic-event substeps, no clipping'
                               % np.max(sums))
    return probabilities


def draw_hop_targets(probabilities, active, rng):
    """One stochastic event per trajectory; return old active index for no hop."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or min(probabilities.shape) == 0:
        raise ValueError('probabilities must have shape (trajectory, state)')
    active = _indices(active, *probabilities.shape, 'active')
    if not isinstance(rng, np.random.Generator):
        raise TypeError('Use an explicit numpy.random.Generator')
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
            or np.any(probabilities.sum(axis=1) > 1.)
            or np.any(probabilities[np.arange(len(active)), active] != 0)):
        raise HoppingStepError('Invalid probabilities: negative, nonfinite, self hop, or sum > 1')
    cumulative = np.cumsum(probabilities, axis=1)
    draw = rng.random(len(active))
    proposed = np.sum(draw[:, None] >= cumulative, axis=1)
    return np.where(proposed < probabilities.shape[1], proposed, active).astype(np.int64)


@dataclass(frozen=True)
class HopResult:
    """Event bookkeeping; energy_residual uses the active-surface classical energy."""
    momentum: np.ndarray
    active: np.ndarray
    attempted: np.ndarray
    accepted: np.ndarray
    frustrated: np.ndarray
    energy_residual: np.ndarray


def apply_hops(momentum, active, targets, energies, mass, *, frustrated_policy='keep'):
    """1D energy-conserving rescale, preserving the sign of incoming momentum.

    P_new^2 = P_old^2 - 2 M (E_target-E_active). If there is insufficient
    kinetic energy the hop is rejected; 'keep' (default) or 'reverse' selects
    the explicitly declared frustrated-hop momentum policy. No amplitudes are
    changed: no collapse or decoherence is part of this baseline.

    A forced downhill event at exactly zero momentum has no specified direction
    and is rejected as an API error. Native NAC flux is zero at v=0, so the
    standard probability path never proposes such an event.
    """
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 2 or min(energies.shape) == 0 or not np.all(np.isfinite(energies)):
        raise ValueError('energies must be finite with shape (trajectory, state)')
    active = _indices(active, *energies.shape, 'active')
    targets = _indices(targets, *energies.shape, 'targets')
    momentum = np.asarray(momentum, dtype=float)
    if momentum.shape != (len(active),) or not np.all(np.isfinite(momentum)):
        raise ValueError('momentum must be finite with one value per trajectory')
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('mass must be finite and positive')
    if frustrated_policy not in ('keep', 'reverse'):
        raise ValueError("frustrated_policy must be 'keep' or 'reverse'")
    rows = np.arange(len(active))
    delta = energies[rows, targets] - energies[rows, active]
    attempted = targets != active
    squared = momentum**2 - 2*mass*delta
    if np.any(attempted & (momentum == 0) & (delta < 0)):
        raise HoppingStepError('Forced downhill hop at zero momentum has no declared direction')
    frustrated = attempted & (squared < 0)
    accepted = attempted & ~frustrated
    result_P = momentum.copy()
    result_P[accepted] = np.sign(momentum[accepted])*np.sqrt(squared[accepted])
    if frustrated_policy == 'reverse':
        result_P[frustrated] *= -1
    result_active = np.where(accepted, targets, active)
    before = momentum**2/(2*mass)+energies[rows, active]
    after = result_P**2/(2*mass)+energies[rows, result_active]
    return HopResult(result_P, result_active, attempted, accepted, frustrated, after-before)
