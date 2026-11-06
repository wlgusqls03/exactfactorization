"""Current-position local R derivatives, reference equations D1--D2.

This is a reconstruction, not a new heavy-force or CTMQC closure. Values at
different trajectories must already have a common electronic basis and outer
gauge. No phase alignment is performed here: it could erase physical currents.
"""
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


class MLSFailure(ValueError):
    """The declared stencil cannot support the requested reconstruction."""


@dataclass(frozen=True)
class MLSConfig:
    degree: int = 2
    neighbors: int = 7
    kernel: str = "gaussian"  # gaussian, uniform, tricube
    radius: Optional[float] = None  # hard physical-distance cutoff
    scale_policy: str = "local_max"  # local_max, fixed
    fixed_scale: Optional[float] = None
    scale_factor: float = 1.0
    balance: str = "prefer"  # prefer equal left/right, or nearest
    boundary: str = "one_sided"  # one_sided, require_two_sided
    condition_limit: float = 1.e10
    rcond: float = 1.e-12
    duplicate_tolerance: float = 1.e-12
    multistream_tolerance: float = 1.e-6
    multistream_policy: str = "raise"  # raise, report
    failure_policy: str = "raise"  # raise, nan (explicit diagnostic holes)

    def validate(self):
        if not isinstance(self.degree, int) or self.degree < 2:
            raise ValueError("MLS degree must be an integer >= 2")
        if not isinstance(self.neighbors, int) or self.neighbors < self.degree + 1:
            raise ValueError("neighbors must be >= degree + 1")
        choices = ((self.kernel, ("gaussian", "uniform", "tricube")),
                   (self.scale_policy, ("local_max", "fixed")),
                   (self.balance, ("prefer", "nearest")),
                   (self.boundary, ("one_sided", "require_two_sided")),
                   (self.multistream_policy, ("raise", "report")),
                   (self.failure_policy, ("raise", "nan")))
        if any(v not in allowed for v, allowed in choices):
            raise ValueError("Unknown MLS policy")
        for name in ("scale_factor", "condition_limit", "rcond"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("duplicate_tolerance", "multistream_tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if self.radius is not None and (not np.isfinite(self.radius) or self.radius <= 0):
            raise ValueError("radius must be positive")
        if self.scale_policy == "fixed" and (
                self.fixed_scale is None or not np.isfinite(self.fixed_scale)
                or self.fixed_scale <= 0):
            raise ValueError("fixed scale policy requires a positive fixed_scale")


@dataclass
class MLSOperator:
    positions: np.ndarray
    config: MLSConfig
    stencils: list
    duplicate_groups: list
    geometry_audit: dict

    def apply(self, values, *, basis_convention, gauge_convention):
        """Return d/dR, d²/dR² and audit; values shape (Ntraj, ...).

        Convention strings are mandatory provenance declarations, not an
        automatic proof that the caller transported a moving basis correctly.
        A constant unitary change of common basis commutes with this operation.
        R-dependent basis changes require external transport/connection terms.
        """
        if not isinstance(basis_convention, str) or not basis_convention.strip():
            raise ValueError("Declare the common electronic basis convention")
        if not isinstance(gauge_convention, str) or not gauge_convention.strip():
            raise ValueError("Declare the common outer gauge convention")
        data = np.asarray(values)
        if data.ndim < 1 or data.shape[0] != len(self.positions):
            raise ValueError("values first axis must match trajectory positions")
        if not np.all(np.isfinite(data)):
            raise ValueError("MLS values contain NaN/Inf")
        data = data.astype(np.complex128 if np.iscomplexobj(data) else np.float64)
        flat = data.reshape(len(data), -1)
        if flat.shape[1] == 0:
            raise ValueError("MLS cannot reconstruct empty values")
        duplicate_scatter = []
        for group in self.duplicate_groups:
            local = flat[group]
            scatter = float(np.sqrt(np.mean(np.abs(local - local.mean(axis=0)) ** 2)))
            duplicate_scatter.append({"indices": group.tolist(), "rms": scatter})
            if scatter > self.config.multistream_tolerance and self.config.multistream_policy == "raise":
                raise MLSFailure("Multistream values at coincident R: scatter=" + str(scatter))
        first = np.full_like(flat, np.nan)
        second = np.full_like(flat, np.nan)
        residual = np.full(len(data), np.nan)
        for i, stencil in enumerate(self.stencils):
            if stencil is None:
                continue
            indices, coefficient_map, design, sqrt_w, h = stencil
            beta = coefficient_map @ flat[indices]
            first[i] = beta[1] / h
            second[i] = beta[2] / h ** 2
            defect = sqrt_w[:, None] * (design @ beta - flat[indices])
            residual[i] = np.sqrt(np.sum(np.abs(defect) ** 2) /
                                  (np.sum(sqrt_w ** 2) * flat.shape[1]))
        audit = dict(self.geometry_audit)
        audit.update(basis_convention=basis_convention, gauge_convention=gauge_convention,
                     duplicate_scatter=duplicate_scatter, local_fit_rms=residual,
                     phase_alignment_performed=False,
                     reconstruction="single_field_local_polynomial")
        return first.reshape(data.shape), second.reshape(data.shape), audit


def _candidate_indices(positions, order, rank, i, config):
    """Bound search by neighbor count; avoid a full N-by-N distance matrix."""
    n = config.neighbors
    k = rank[i]
    candidates = order[max(0, k - n):min(len(order), k + n + 1)]
    delta = positions[candidates] - positions[i]
    if config.radius is not None:
        candidates = candidates[np.abs(delta) <= config.radius]
    candidates = candidates[np.argsort(np.abs(positions[candidates] - positions[i]), kind="stable")]
    if config.balance == "nearest":
        return candidates[:n]
    tol = config.duplicate_tolerance
    left = candidates[positions[candidates] < positions[i] - tol]
    right = candidates[positions[candidates] > positions[i] + tol]
    # Include the trajectory itself; extra coincident histories remain candidates
    # and are independently audited rather than phase-aligned or averaged away.
    selected = [i]
    half = (n - 1) // 2
    selected.extend(left[:half].tolist())
    selected.extend(right[:half].tolist())
    remaining = [j for j in candidates if j not in selected]
    selected.extend(remaining[:max(0, n - len(selected))])
    return np.asarray(selected, dtype=int)


def build_mls_operator(R, config=None):
    """Build reusable geometry weights for *current* R, equations D1--D2.

    Degree is never silently reduced. ``failure_policy='nan'`` keeps explicit
    holes with error strings for visualization only; propagation should use
    the default 'raise'. Neighbor geometry must be rebuilt at every stage.
    """
    config = MLSConfig() if config is None else config
    config.validate()
    if np.iscomplexobj(R):
        raise ValueError("R positions must be real")
    positions = np.asarray(R, dtype=np.float64)
    if positions.ndim != 1 or len(positions) == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("R must be a nonempty finite one-dimensional array")
    order = np.argsort(positions, kind="stable")
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    groups = []
    begin = 0
    for stop in range(1, len(order) + 1):
        if stop == len(order) or positions[order[stop]] - positions[order[begin]] > config.duplicate_tolerance:
            if stop - begin > 1:
                groups.append(order[begin:stop])
            begin = stop
    condition = np.full(len(positions), np.inf)
    scale = np.full(len(positions), np.nan)
    left_right = np.zeros((len(positions), 2), dtype=int)
    distinct = np.zeros(len(positions), dtype=int)
    stencils, errors, all_indices = [], [], []
    for i in range(len(positions)):
        idx = _candidate_indices(positions, order, rank, i, config)
        all_indices.append(idx.tolist())
        dx = positions[idx] - positions[i]
        distinct[i] = 1 + np.count_nonzero(np.diff(np.sort(dx)) > config.duplicate_tolerance)
        left_right[i] = [np.count_nonzero(dx < -config.duplicate_tolerance),
                         np.count_nonzero(dx > config.duplicate_tolerance)]
        try:
            if distinct[i] < config.degree + 1:
                raise MLSFailure("Too few distinct R positions")
            if config.boundary == "require_two_sided" and np.any(left_right[i] == 0):
                raise MLSFailure("Two-sided stencil required at boundary")
            h = config.fixed_scale if config.scale_policy == "fixed" else np.max(np.abs(dx))
            h = float(h * config.scale_factor)
            if not np.isfinite(h) or h <= 0:
                raise MLSFailure("Invalid local length scale")
            xi = dx / h
            design = np.column_stack([xi ** j / math.factorial(j) for j in range(config.degree + 1)])
            weights = (np.exp(-.5 * xi ** 2) if config.kernel == "gaussian" else
                       np.maximum(0., 1. - np.abs(xi) ** 3) ** 3 if config.kernel == "tricube" else
                       np.ones_like(xi))
            sqrt_w = np.sqrt(weights)
            u, singular, vh = np.linalg.svd(sqrt_w[:, None] * design, full_matrices=False)
            good = singular > config.rcond * singular[0]
            condition[i] = singular[0] / singular[-1] if singular[-1] > 0 else np.inf
            scale[i] = h
            if np.count_nonzero(good) != config.degree + 1 or condition[i] > config.condition_limit:
                raise MLSFailure("Ill-conditioned/rank-deficient stencil (condition=%g)" % condition[i])
            coefficient_map = ((vh.T / singular) @ u.T) * sqrt_w[None, :]
            stencils.append((idx, coefficient_map, design, sqrt_w, h))
            errors.append(None)
        except MLSFailure as exc:
            message = "trajectory %d: %s" % (i, exc)
            if config.failure_policy == "raise":
                raise MLSFailure(message) from exc
            stencils.append(None)
            errors.append(message)
    audit = {"condition": condition, "scale": scale, "left_right_count": left_right,
             "distinct_count": distinct, "indices": all_indices, "errors": errors,
             "degree": config.degree, "boundary": config.boundary, "kernel": config.kernel}
    return MLSOperator(positions.copy(), config, stencils, groups, audit)


def reconstruct_R_derivatives(R, values, *, basis_convention, gauge_convention, config=None):
    """Convenience current-geometry build and application."""
    return build_mls_operator(R, config).apply(
        values, basis_convention=basis_convention, gauge_convention=gauge_convention)
