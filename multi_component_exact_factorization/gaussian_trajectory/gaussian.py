"""Positive Gaussian density and complex-rate projection on a common q grid.

Equation labels refer to ``paper/MCEF_Gaussian_Trajectory_Codex_Reference.tex``:
S3--S5, P3--P18 and N1. This module supplies no force, gauge, or heavy-method
closure. A caller supplies the complex material ``G_rhs_eom`` evaluated at the
CURRENT state. Arrays describe one trajectory: (nq,), (nq, ne), (nq, 3*K-1).
Only the bridge is positive; phase remains in C. All arithmetic is float64 /
complex128. No population clipping, coefficient renormalization, density floor,
matrix ridge, or automatic Gaussian merging is performed.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np


class GaussianRankError(ValueError):
    """The selected Gaussian tangent representation is numerically singular."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


@dataclass
class GaussianBridge:
    q: np.ndarray
    quadrature: np.ndarray
    parameters: np.ndarray
    coordinates: str
    weights: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    kernels: np.ndarray
    rho: np.ndarray
    amplitude: np.ndarray
    jacobian: np.ndarray
    audit: Dict


@dataclass
class AmplitudeProjection:
    parameter_dot: np.ndarray
    Lambda_dot: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
    singular_values: np.ndarray
    support: np.ndarray
    audit: Dict


@dataclass
class ProjectedRate:
    C_dot: np.ndarray
    parameter_dot: np.ndarray
    Lambda_dot: np.ndarray
    G_dot_actual: np.ndarray
    amplitude_rate_requested: np.ndarray
    projection: AmplitudeProjection
    support: np.ndarray
    audit: Dict


def _real_vector(values, name):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(name + " must be real")
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError(name + " must be a finite one-dimensional array")
    return a


def _components(components):
    if isinstance(components, bool) or int(components) != components or components < 1:
        raise ValueError("components must be a positive integer")
    return int(components)


def pack_parameters(weights, centers, widths, *, coordinates="physical"):
    """Pack P5 or independent logits / centers / log-widths.

    Transformed weights use last logit = 0, hence exactly K-1 parameters.
    Zero weights are legal in physical coordinates, but usually imply a rank
    failure during projection. Transformed coordinates require positive weights.
    """
    w = _real_vector(weights, "weights")
    Q = _real_vector(centers, "centers")
    s = _real_vector(widths, "widths")
    if not len(w) or w.shape != Q.shape or w.shape != s.shape:
        raise ValueError("weights, centers and widths must have equal nonzero length")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-13):
        raise ValueError("mixture weights must be nonnegative and sum to one")
    if np.any(s <= 0):
        raise ValueError("Gaussian standard deviations must be positive")
    if coordinates == "physical":
        return np.concatenate((w[:-1], Q, s))
    if coordinates == "transformed":
        if np.any(w <= 0):
            raise ValueError("independent logits require strictly positive weights")
        return np.concatenate((np.log(w[:-1]) - np.log(w[-1]), Q, np.log(s)))
    raise ValueError("coordinates must be 'physical' or 'transformed'")


def unpack_parameters(parameters, components, *, coordinates="physical"):
    """Return physical (weights, centers, standard deviations), without clipping."""
    K = _components(components)
    z = _real_vector(parameters, "parameters")
    if z.size != 3 * K - 1:
        raise ValueError("Gaussian representation needs 3*K-1 independent parameters")
    Q = z[K - 1:2 * K - 1].copy()
    if coordinates == "physical":
        w = np.r_[z[:K - 1], 1.0 - z[:K - 1].sum()]
        s = z[2 * K - 1:].copy()
    elif coordinates == "transformed":
        logits = np.r_[z[:K - 1], 0.0]
        w = np.exp(logits - np.max(logits))
        w /= w.sum()
        with np.errstate(over="ignore", under="ignore"):
            s = np.exp(z[2 * K - 1:])
    else:
        raise ValueError("coordinates must be 'physical' or 'transformed'")
    if np.any(w < 0) or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("invalid weights or widths; reduce step or change parameterization")
    return w, Q, s


def build_bridge(q, quadrature, parameters, *, components, coordinates="physical",
                 normalize_discrete=True):
    """Evaluate S3, P11--P13 and the exact finite-domain correction N1.

    ``normalize_discrete=True`` is a declared numerical policy: each kernel is
    normalized with the supplied quadrature, and its normalization derivative
    is subtracted from its score. False retains real-line Gaussian kernels and
    reports their finite-grid norm defects. Kernels and mixtures use shifted
    exponentials / log-sum-exp to avoid underflow in intermediate normalizers.
    """
    q = _real_vector(q, "q")
    omega = _real_vector(quadrature, "quadrature")
    if len(q) < 2 or np.any(np.diff(q) <= 0):
        raise ValueError("q must contain at least two strictly increasing points")
    if omega.shape != q.shape or np.any(omega <= 0):
        raise ValueError("positive quadrature weights must match q")
    w, Q, s = unpack_parameters(parameters, components, coordinates=coordinates)
    K = len(w)
    u = (q[:, None] - Q) / s
    log_kernel = -0.5 * u * u - np.log(s) - 0.5 * np.log(2 * np.pi)
    if not np.all(np.isfinite(log_kernel)):
        raise ValueError("unresolved/overflowing Gaussian parameters on this q grid")
    log_Z = np.logaddexp.reduce(log_kernel + np.log(omega[:, None]), axis=0)
    log_used = log_kernel - log_Z if normalize_discrete else log_kernel
    kernels = np.exp(log_used)
    score_Q = u / s
    score_s = (u * u - 1.0) / s
    if normalize_discrete:
        score_Q -= np.sum(omega[:, None] * kernels * score_Q, axis=0)
        score_s -= np.sum(omega[:, None] * kernels * score_s, axis=0)
    # Zero component weights are represented by -inf, not epsilon clipping.
    log_w = np.full(K, -np.inf)
    np.log(w, out=log_w, where=w > 0)
    log_rho = np.logaddexp.reduce(log_used + log_w, axis=1)
    rho = np.exp(log_rho)
    amplitude = np.exp(0.5 * log_rho)
    if coordinates == "physical":
        Jw = kernels[:, :-1] - kernels[:, [-1]]
        Js = w * kernels * score_s
    else:
        Jw = w[:-1] * (kernels[:, :-1] - rho[:, None])
        Js = w * kernels * score_s * s
    JQ = w * kernels * score_Q
    J = np.concatenate((Jw, JQ, Js), axis=1)
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(rho)):
        raise ValueError("Gaussian/Jacobian evaluation overflowed")
    grid_norm = float(omega @ rho)
    audit = {
        "components": K, "coordinates": coordinates,
        "normalization": "discrete_N1" if normalize_discrete else "real_line",
        "grid_norm": grid_norm, "grid_norm_error": abs(grid_norm - 1.0),
        "kernel_log_grid_norm_before_normalization": log_Z.tolist(),
        "kernel_grid_norm_before_normalization": np.exp(log_Z).tolist(),
        "max_integrated_jacobian": float(np.max(np.abs(omega @ J))),
        "density_underflow_points": int(np.sum(rho == 0)),
        "min_weight": float(w.min()), "min_width": float(s.min()),
    }
    return GaussianBridge(q.copy(), omega.copy(), np.asarray(parameters).copy(),
                          coordinates, w, Q, s, kernels, rho, amplitude, J, audit)


def select_support(bridge, *, support_relative=0.0, support_absolute=0.0,
                   max_excluded_mass=1e-10):
    """Explicit support policy. No density is raised to a floor.

    A masked C field is frozen only when ``split_complex_rate`` is explicitly
    given this mask; discarded RHS and probability are reported. Default mass
    budget prevents suppressing appreciably occupied points. Set the budget
    deliberately when testing threshold sensitivity.
    """
    values = (support_relative, support_absolute, max_excluded_mass)
    if any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("support thresholds and excluded mass budget must be finite >= 0")
    if support_relative >= 1:
        raise ValueError("support_relative must be smaller than one")
    threshold = max(support_absolute, support_relative * float(bridge.rho.max()))
    mask = (bridge.rho > threshold) & (bridge.amplitude > 0)
    mass = float(bridge.quadrature[~mask] @ bridge.rho[~mask])
    if not np.any(mask):
        raise ValueError("no representable supported Gaussian density")
    if mass > max_excluded_mass:
        raise ValueError("excluded Gaussian mass %.6g exceeds budget %.6g" %
                         (mass, max_excluded_mass))
    audit = {"support_relative": float(support_relative),
             "support_absolute": float(support_absolute),
             "excluded_mass": mass, "excluded_points": int(np.sum(~mask)),
             "max_excluded_mass": float(max_excluded_mass),
             "min_supported_amplitude": float(bridge.amplitude[mask].min())}
    return mask, audit


def split_complex_rate(C, G_rhs_eom, amplitude, *, support=None,
                       pnc_tolerance=1e-8):
    """P3: return (C_dot, real requested amplitude rate, audit).

    The *real*, unprojected contraction is essential; using Lambda_dot here
    destroys electronic PNC in general. A supplied support mask is an explicit
    freeze of C outside support, not a claim of valid fields at density nodes.
    With no mask, zero amplitudes cause an error. No extra time step appears.
    """
    C = np.asarray(C, dtype=np.complex128)
    G_rhs_eom = np.asarray(G_rhs_eom, dtype=np.complex128)
    amplitude = _real_vector(amplitude, "amplitude")
    if C.ndim != 2 or C.shape != G_rhs_eom.shape or C.shape[0] != len(amplitude):
        raise ValueError("C and G_rhs_eom must have matching (nq, ne) shapes")
    if C.shape[1] == 0 or not np.all(np.isfinite(C)) or not np.all(np.isfinite(G_rhs_eom)):
        raise ValueError("complex coefficients and EOM rate must be finite and nonempty")
    if np.any(amplitude < 0):
        raise ValueError("bridge amplitude cannot be negative")
    mask = np.ones(amplitude.shape, dtype=bool) if support is None else np.asarray(support, bool)
    if mask.shape != amplitude.shape or not np.any(mask) or np.any(amplitude[mask] <= 0):
        raise ValueError("support must contain positive, nonzero bridge amplitudes")
    if not np.isfinite(pnc_tolerance) or pnc_tolerance < 0:
        raise ValueError("pnc_tolerance must be finite and nonnegative")
    pnc_error = float(np.max(np.abs(np.sum(abs(C[mask]) ** 2, axis=1) - 1)))
    if pnc_error > pnc_tolerance:
        raise ValueError("raw electronic PNC error %.6g exceeds tolerance %.6g" %
                         (pnc_error, pnc_tolerance))
    r = np.real(np.sum(C.conj() * G_rhs_eom, axis=1))
    C_dot = np.zeros_like(C)
    C_dot[mask] = (G_rhs_eom[mask] - C[mask] * r[mask, None]) / amplitude[mask, None]
    if not np.all(np.isfinite(C_dot)):
        raise FloatingPointError("P3 overflow: refine grid/support rather than flooring Lambda")
    pnc_rate = 2 * np.real(np.sum(C.conj() * C_dot, axis=1))
    return C_dot, r, {"max_raw_pnc_error": pnc_error,
                      "max_pnc_rate_error": float(np.max(np.abs(pnc_rate[mask]))),
                      "outside_support_policy": "freeze_C"}


def project_amplitude_rate(bridge, amplitude_rate_requested, *, support=None,
                           rank_rtol=1e-12, rank_policy="raise"):
    """P8--P14 weighted SVD solve, returning actual Lambda_dot and diagnostics.

    ``rank_policy='truncate'`` explicitly accepts a truncated SVD/minimum-norm
    solution. Default raises ``GaussianRankError`` carrying rank diagnostics.
    Singular values refer to L, not to its condition-number-squared Gram matrix.
    """
    r = _real_vector(amplitude_rate_requested, "amplitude_rate_requested")
    if r.shape != bridge.rho.shape:
        raise ValueError("amplitude rate must match bridge q grid")
    if rank_policy not in ("raise", "truncate"):
        raise ValueError("rank_policy must be 'raise' or explicit 'truncate'")
    if not np.isfinite(rank_rtol) or not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must lie strictly between zero and one")
    mask = np.ones(r.shape, bool) if support is None else np.asarray(support, bool)
    if mask.shape != r.shape or not np.any(mask) or np.any(bridge.amplitude[mask] <= 0):
        raise ValueError("explicit support is required at amplitude nodes")
    sqrt_w = np.sqrt(bridge.quadrature[mask])
    L = sqrt_w[:, None] * bridge.jacobian[mask] / bridge.amplitude[mask, None]
    y = 2 * sqrt_w * r[mask]
    parameter_dot, _, rank, singular_values = np.linalg.lstsq(L, y, rcond=rank_rtol)
    full_rank = bridge.jacobian.shape[1]
    condition = float(singular_values[0] / singular_values[-1]) if rank == full_rank else float("inf")
    audit = {"rank": int(rank), "parameter_count": full_rank,
             "rank_rtol": float(rank_rtol), "condition_L": condition,
             "rank_policy": rank_policy, "discarded_directions": full_rank - int(rank),
             "regularization": "none" if rank == full_rank else "explicit_truncated_SVD"}
    if rank != full_rank and rank_policy == "raise":
        raise GaussianRankError("Gaussian tangent rank %d < %d; duplicate/zero-weight or unresolved components" %
                                (rank, full_rank), audit)
    Lambda_dot = np.zeros_like(r)
    # Projection is supported locally, but the parameter velocity changes the
    # same globally normalized density. Evaluate it at every representable point.
    representable = bridge.amplitude > 0
    Lambda_dot[representable] = (bridge.jacobian[representable] @ parameter_dot) / (2 * bridge.amplitude[representable])
    residual = Lambda_dot - r
    audit.update({
        "projection_residual_l2": float(np.sqrt(bridge.quadrature[mask] @ residual[mask] ** 2)),
        "projection_residual_l2_full": float(np.sqrt(bridge.quadrature @ residual ** 2)),
        "normal_equation_residual": float(np.linalg.norm(L.T @ (L @ parameter_dot - y))),
        "requested_norm_rate": float(2 * bridge.quadrature @ (bridge.amplitude * r)),
        "projected_norm_rate": float(bridge.quadrature @ (bridge.jacobian @ parameter_dot)),
    })
    return AmplitudeProjection(parameter_dot, Lambda_dot, L.T @ L, L.T @ y,
                               singular_values, mask.copy(), audit)


def split_and_project(G_rhs_eom, C, bridge, *, support_relative=0.0,
                      support_absolute=0.0, max_excluded_mass=1e-10,
                      rank_rtol=1e-12, rank_policy="raise", pnc_tolerance=1e-8):
    """One synchronized P3/P8 evaluation; P15 uses the ACTUAL projected rate.

    Outside selected support C is frozen while parameters change globally;
    therefore P15 is checked on supported points only. The excluded RHS norm
    and full state-change norm are retained rather than claiming exactness there.
    """
    mask, audit = select_support(bridge, support_relative=support_relative,
                                 support_absolute=support_absolute,
                                 max_excluded_mass=max_excluded_mass)
    C = np.asarray(C, dtype=np.complex128)
    G_rhs_eom = np.asarray(G_rhs_eom, dtype=np.complex128)
    C_dot, r, split_audit = split_complex_rate(C, G_rhs_eom, bridge.amplitude,
                                              support=mask, pnc_tolerance=pnc_tolerance)
    p = project_amplitude_rate(bridge, r, support=mask, rank_rtol=rank_rtol,
                              rank_policy=rank_policy)
    actual = bridge.amplitude[:, None] * C_dot + p.Lambda_dot[:, None] * C
    rhs_P15 = G_rhs_eom + (p.Lambda_dot - r)[:, None] * C
    audit.update(split_audit)
    audit.update(p.audit)
    audit.update({
        "P15_max_error_supported": float(np.max(np.abs(actual[mask] - rhs_P15[mask]))),
        "discarded_rhs_l2": float(np.sqrt(bridge.quadrature[~mask] @
                                           np.sum(abs(G_rhs_eom[~mask]) ** 2, axis=1))),
        "actual_vs_eom_rate_l2": float(np.sqrt(bridge.quadrature @
                                                np.sum(abs(actual - G_rhs_eom) ** 2, axis=1))),
    })
    return ProjectedRate(C_dot, p.parameter_dot, p.Lambda_dot, actual, r, p, mask, audit)
