"""Heavy-density estimators and explicitly separate quantum momentum (Q1--Q10).

These arrays are not classical P_R, EF connections a/alpha, or a reconstructed
complex chi. No quantum-potential force is implied by a density estimate.
"""
from dataclasses import dataclass

import numpy as np


class DensityFailure(ValueError):
    """An occupied density component cannot be reconstructed as declared."""


class InterceptCancellationError(DensityFailure):
    """Q10 has a cancelling denominator; no implicit cap/EDI repair is used."""


@dataclass
class HeavyDensity:
    evaluation_R: np.ndarray
    density: np.ndarray
    log_density: np.ndarray
    quantum_momentum: np.ndarray
    posterior: np.ndarray
    channel_weights: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    audit: dict


def _ensemble(R, eta, populations=None, normalization_tolerance=1.e-10):
    if any(np.iscomplexobj(a) for a in (R, eta) if a is not None):
        raise ValueError("R and eta must be real")
    R, eta = np.asarray(R, dtype=float), np.asarray(eta, dtype=float)
    if R.ndim != 1 or len(R) == 0 or eta.shape != R.shape:
        raise ValueError("R and eta must be matching nonempty vectors")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(eta)) or np.any(eta < 0):
        raise ValueError("Finite positions and nonnegative trajectory weights required")
    if not np.isfinite(normalization_tolerance) or normalization_tolerance < 0:
        raise ValueError("normalization_tolerance must be nonnegative")
    if abs(eta.sum() - 1.) > normalization_tolerance:
        raise ValueError("eta must be normalized; no hidden normalization")
    if populations is None:
        return R, eta
    if np.iscomplexobj(populations):
        raise ValueError("Channel populations must be real")
    p = np.asarray(populations, dtype=float)
    if p.ndim != 2 or p.shape[0] != len(R) or p.shape[1] == 0:
        raise ValueError("populations must have shape (Ntraj, Nchannel)")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Channel populations must be finite and nonnegative")
    if np.max(np.abs(p.sum(axis=1) - 1.)) > normalization_tolerance:
        raise ValueError("Channel populations must sum to one at each trajectory")
    return R, eta, p


def _gaussian_mixture(evaluation_R, weights, centers, widths, audit):
    grid = np.asarray(evaluation_R, dtype=float)
    if grid.ndim != 1 or not np.all(np.isfinite(grid)):
        raise ValueError("evaluation_R must be a finite vector")
    active = weights > 0
    x = grid[:, None] - centers[None, active]
    var = widths[active] ** 2
    if np.any(~np.isfinite(var)) or np.any(var <= 0):
        raise DensityFailure("Channel variance underflows/overflows; choose an explicit width policy")
    logs = np.log(weights[active])[None, :] - .5 * np.log(2. * np.pi * var)[None, :] - .5 * x ** 2 / var
    maximum = np.max(logs, axis=1)
    if not np.all(np.isfinite(maximum)):
        raise DensityFailure("Density log evaluation overflow; rescale coordinates/widths")
    relative = np.exp(logs - maximum[:, None])
    posterior_active = relative / relative.sum(axis=1, keepdims=True)
    log_density = maximum + np.log(relative.sum(axis=1))
    raw = .5 * np.sum(posterior_active * x / var, axis=1)
    if not np.all(np.isfinite(raw)):
        raise DensityFailure("Nonfinite quantum momentum")
    posterior = np.zeros((len(grid), len(weights)))
    posterior[:, active] = posterior_active
    density = np.exp(log_density)
    audit = dict(audit, underflow_points=int(np.count_nonzero(density == 0)),
                 active_channels=np.flatnonzero(active).tolist(),
                 log_derivative_uses_posterior=True, normalization=float(weights.sum()))
    return HeavyDensity(grid.copy(), density, log_density, raw, posterior,
                        weights.copy(), centers.copy(), widths.copy(), audit)


def reconstruct_heavy_density(R, eta, populations, evaluation_R=None, *, channel_basis,
                              width_floor=None, normalization_tolerance=1.e-10):
    """Q1--Q5: one moment-matched Gaussian per declared orthonormal channel.

    Exactly unoccupied channels are omitted without dividing by their weight.
    Every positive channel is retained. Zero/narrow widths fail by default;
    optional ``width_floor`` explicitly changes the estimator and is audited.
    Widths mean standard deviations, not the 2016 exponential-width parameter.
    """
    if not isinstance(channel_basis, str) or not channel_basis.strip():
        raise ValueError("Declare the orthonormal channel basis")
    R, eta, p = _ensemble(R, eta, populations, normalization_tolerance)
    if width_floor is not None and (not np.isfinite(width_floor) or width_floor <= 0):
        raise ValueError("width_floor must be finite and positive")
    weighted = eta[:, None] * p
    total = weighted.sum(axis=0)
    active = total > 0
    centers = np.full(p.shape[1], np.nan)
    widths = np.full(p.shape[1], np.nan)
    normalized = weighted[:, active] / total[None, active]
    centers[active] = np.sum(normalized * R[:, None], axis=0)
    widths[active] = np.sqrt(np.sum(normalized * (R[:, None] - centers[None, active]) ** 2, axis=0))
    original_widths = widths.copy()
    changed = np.zeros_like(active)
    if width_floor is not None:
        changed = active & (widths < width_floor)
        widths[changed] = width_floor
    if np.any(~np.isfinite(widths[active])) or np.any(widths[active] <= 0):
        raise DensityFailure("Occupied channel has zero/nonfinite width; choose an explicit width policy")
    audit = {"mode": "state_resolved_gaussian", "channel_basis": channel_basis,
             "width_convention": "standard_deviation", "width_floor": width_floor,
             "regularized_channels": np.flatnonzero(changed).tolist(),
             "original_widths": original_widths, "omitted_population": 0.0,
             "zero_population_channels": np.flatnonzero(~active).tolist()}
    return _gaussian_mixture(R if evaluation_R is None else evaluation_R, total, centers, widths, audit)


def reconstruct_kde_density(R, eta, evaluation_R=None, *, bandwidth, normalization_tolerance=1.e-10):
    """Q6: trajectory KDE, explicitly NOT the state-resolved estimator."""
    R, eta = _ensemble(R, eta, normalization_tolerance=normalization_tolerance)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("KDE bandwidth must be positive and is distinct from MLS/bridge widths")
    return _gaussian_mixture(R if evaluation_R is None else evaluation_R, eta, R,
                             np.full_like(R, bandwidth),
                             {"mode": "trajectory_kde", "bandwidth": float(bandwidth)})


def two_channel_intercept_correction(R, eta, populations, force_history, *,
                                     raw_quantum_momentum, slope, uncoupled_channels=False,
                                     channel_basis, mass=1., cancellation_rtol=1.e-12):
    """Q8--Q10 for a declared genuinely uncoupled two-channel test.

    This is not a universal CTMQC engine, and zero nuclear NAC alone is not an
    uncoupled photon/electron system. A cancelling denominator raises; no EDI,
    clipping, pair-to-scalar substitution, or hidden denominator floor is used.
    Returns raw/corrected arrays separately. If the entire source is zero the
    correction is inactive and raw is retained, rather than inventing R_star.
    """
    if uncoupled_channels is not True:
        raise ValueError("Q10 requires explicitly declared genuinely uncoupled channels")
    if not isinstance(channel_basis, str) or not channel_basis.strip():
        raise ValueError("Declare the force-history/population channel basis")
    R, eta, p = _ensemble(R, eta, populations)
    f = np.asarray(force_history, dtype=float)
    raw = np.asarray(raw_quantum_momentum, dtype=float)
    if p.shape[1] != 2 or f.shape != p.shape or raw.shape != R.shape:
        raise ValueError("Q10 supports two channels with histories (N,2) and raw momentum (N,)")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(raw)):
        raise ValueError("Nonfinite force history/raw momentum")
    if np.ndim(slope) != 0 or not np.isfinite(slope):
        raise ValueError("Q9 requires one finite common slope")
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(cancellation_rtol) or not 0 <= cancellation_rtol < 1:
        raise ValueError("Invalid mass or cancellation tolerance")
    source = eta * (f[:, 0] - f[:, 1]) * p[:, 0] * p[:, 1]
    denominator = float(np.sum(source))
    magnitude = float(np.sum(np.abs(source)))
    if not np.isfinite(denominator) or not np.isfinite(magnitude):
        raise DensityFailure("Nonfinite Q10 source; rescale/check force histories")
    active = magnitude > 0 and slope != 0
    if active and abs(denominator) <= cancellation_rtol * magnitude:
        raise InterceptCancellationError("Q10 denominator cancels; select/derive a regularization explicitly")
    intercept = float(np.dot(R, source) / denominator) if active else None
    corrected = (float(slope) * (R - intercept) if active else
                 np.zeros_like(raw) if slope == 0 else raw.copy())
    return {"raw": raw.copy(), "corrected": corrected, "intercept": intercept,
            "active": active, "slope": float(slope), "denominator": denominator,
            "inactive_reason": (None if active else "zero_common_slope" if slope == 0 else "zero_population_source"),
            "denominator_absolute_sum": magnitude, "channel_basis": channel_basis,
            "uncoupled_channels": True, "mode": "two_channel_common_slope_intercept",
            "raw_ensemble_population_rate": float(2. / mass * np.dot(raw, source)),
            "corrected_ensemble_population_rate": float(2. / mass * np.dot(corrected, source))}
