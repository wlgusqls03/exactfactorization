"""Ensemble coupling in CTMQC, atomic units.

Independent implementation of Villaseco Arribas et al., Molecules 27, 4002
(2022), Eqs. (16)--(18), (26)--(27), DOI 10.3390/molecules27134002.
Q=-grad(log density)/2, NOT a canonical/physical trajectory momentum.
Fixed Gaussian density standard deviations are called bandwidths here.
No G-CTMQC source code is incorporated.
"""
import numpy as np


def density_quantum_momentum(positions, bandwidths, block=128):
    """Exact Gaussian-mixture log derivative, blockwise O(N^2), bounded RAM.

    Intended for small reference tests, NOT 10,000-trajectory production.
    The default corrected-pair algorithm below needs no N*N kernel matrix.
    """
    x = np.asarray(positions, dtype=float)
    h = np.asarray(bandwidths, dtype=float)
    if x.ndim != 2 or h.shape != (x.shape[1],) or np.any(h <= 0):
        raise ValueError('Need positions[N,D] and positive density bandwidths[D]')
    result = np.empty_like(x)
    for start in range(0, len(x), block):
        delta = x[start:start+block, None, :]-x[None, :, :]
        w = np.exp(-.5*np.sum((delta/h)**2, axis=-1))
        result[start:start+block] = np.sum(w[:, :, None]*delta, axis=1)/(2*h*h*w.sum(axis=1)[:, None])
    return result


def coupled_terms(x, populations, integrated_forces, masses, bandwidths,
                  threshold=1e-3, method='corrected', intercept_limit_sigma=10.):
    """Return real coefficient growth rates and CT force, not Ehrenfest force.

    Each state pair uses its own Q_lk. With df=f_l-f_k,
      rate_l += p_k sum_d Q_lk,d df_d/M_d; rate_k -= p_l (...)
      F_CT += 2 p_l p_k df * sum_d Q_lk,d df_d/M_d.
    Thus sum_l p_l rate_l=0 trajectory by trajectory. Corrected Q also makes
    the ensemble CT-only population derivative zero, pairwise and per DOF.

    An ill-conditioned signed denominator or a center outside the ensemble
    bounding interval + intercept_limit_sigma * bandwidth disables that
    pair/direction and is
    counted, rather than introducing an unbounded intercept or silently using
    an uncorrected Q. This explicit safety policy differs from G-CTMQC's
    distance/M-parameter fallback. Converge threshold and trajectory count.
    """
    n, states = populations.shape
    dof = x.shape[1]
    rate = np.zeros((n, states))
    force = np.zeros((n, dof))
    q2 = np.zeros((n, dof))
    attempted, disabled, distant = 0, 0, 0
    raw = density_quantum_momentum(x, bandwidths) if method == 'density' else None
    if method not in ('corrected', 'density'):
        raise ValueError('Unknown quantum momentum prescription')
    for l in range(states):
        for k in range(l):
            df = integrated_forces[:, l]-integrated_forces[:, k]
            active = (populations[:, l] > threshold) & (populations[:, k] > threshold)
            weight = populations[:, l]*populations[:, k]*active
            q = np.zeros((n, dof))
            for d in range(dof):
                w = weight*df[:, d]
                size = np.sum(abs(w))
                if size < 1e-30:
                    continue
                attempted += 1
                if raw is not None:
                    q[:, d] = raw[:, d]
                    continue
                denominator = w.sum()
                if abs(denominator) <= 1e-10*size:
                    disabled += 1
                    continue
                # Subtract a local origin to avoid catastrophic cancellation.
                origin = x[:, d].mean()
                center = origin+np.dot(x[:, d]-origin, w)/denominator
                # A near-cancelled signed denominator need not be machine-small
                # to send the center thousands of bohr away. Disable the WHOLE
                # pair/direction, preserving its no-net-transfer constraint.
                # Never clip Q per trajectory, which would break that identity.
                margin = intercept_limit_sigma*bandwidths[d]
                if center < x[:, d].min()-margin or center > x[:, d].max()+margin:
                    disabled += 1
                    distant += 1
                    continue
                q[:, d] = (x[:, d]-center)/(2*bandwidths[d]**2)
            q *= active[:, None]
            coupling = np.sum(q*df/masses, axis=1)
            rate[:, l] += populations[:, k]*coupling
            rate[:, k] -= populations[:, l]*coupling
            force += 2*weight[:, None]*coupling[:, None]*df
            q2 += weight[:, None]*q*q
    info = dict(q_pair_weighted_rms=np.sqrt(np.mean(q2, axis=0)),
                q_pair_weighted_local_rms=np.sqrt(q2),
                q_disabled_directions=disabled, q_distant_directions=distant, q_attempted_directions=attempted,
                ct_population_rate=2*np.mean(populations*rate, axis=0),
                coefficient_norm_rate_max=float(np.max(abs(2*np.sum(populations*rate, axis=1)))))
    return rate, force, info
