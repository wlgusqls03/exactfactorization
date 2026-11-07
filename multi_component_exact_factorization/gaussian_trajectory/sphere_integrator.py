"""Opt-in fourth-order integration of P3 in homogeneous sphere coordinates.

The physical coefficients obey pointwise ``C^dagger C = 1``.  Introduce only
within a step an auxiliary vector Y, with C = Y / ||Y|| and
Y_dot = ||Y|| * C_dot(C, z, R, ...).  For a real-tangent P3 rate,
Re(C^dagger C_dot) = 0, differentiating the coordinate map gives exactly P3.
This removes the arbitrary radial extension of P3 away from its constraint
surface.  The Gaussian parameter rate, force and histories are evaluated on
the SAME physical stage and advanced jointly by classical RK4.

At the end of a step Y is rebased to its physical representative C.  This is
an explicit integrator/coordinate choice, NOT a fix for a bad physical G norm,
a population clipping rule, a density mask, or a new force/closure.  A large
auxiliary norm defect is rejected and reported, not hidden by the map.  PNC
preservation alone does not establish time-step, grid or Gaussian convergence.
The method is experimental and does not change legacy RK4/DOP853 defaults.
"""
import numpy as np


class SphereStepError(FloatingPointError):
    """A homogeneous-coordinate step failed its explicit numerical guards."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = dict(audit)


def sphere_rk4_step(state, dt, evaluate, *, max_auxiliary_defect=0.05,
                    tangent_tolerance=1e-10, initial_pnc_tolerance=1e-10):
    """Advance one complete ensemble; return ``(physical_state, audit)``.

    ``state['C']`` is complex, with electronic components on its last axis.
    Every other entry is an ordinary real/complex ODE variable.  ``evaluate``
    returns the complete derivative dictionary, optionally paired with its
    own diagnostics.  All inputs and returned derivatives are copied.

    The auxiliary guard uses max abs(||Y||^2 - 1), including RK stages AND the
    final candidate.  The tangent guard divides the raw rate defect
    abs(2 Re C^dagger C_dot) by max(1, 2 ||C_dot||), so it is meaningful even
    for large but accurately tangent rates.  Neither defect is subtracted
    from the supplied derivative.  A rejected step must not advance a TSH
    random stream, active surface, or force-history checkpoint.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Sphere RK4 requires a positive finite time step')
    if (not np.isfinite(max_auxiliary_defect) or not 0 < max_auxiliary_defect < 1
            or not np.isfinite(tangent_tolerance) or tangent_tolerance <= 0
            or not np.isfinite(initial_pnc_tolerance) or initial_pnc_tolerance <= 0):
        raise ValueError('Need 0 < auxiliary guard < 1 and positive finite tolerances')
    if not isinstance(state, dict) or 'C' not in state:
        raise ValueError('Sphere integrator requires a coefficient-state dictionary')
    initial = {key: np.array(value, copy=True) for key, value in state.items()}
    C0 = initial['C']
    if C0.ndim < 2 or C0.shape[-1] < 1 or not np.iscomplexobj(C0) or C0.size == 0:
        raise ValueError('C must be a nonempty complex array with a component axis')
    if any(not np.all(np.isfinite(value)) for value in initial.values()):
        raise ValueError('Initial sphere state must be finite')
    initial_error = float(np.max(abs(np.sum(abs(C0)**2, axis=-1)-1)))
    audit = dict(integrator='sphere-rk4', rhs_evaluations=0,
                 initial_pnc_error=initial_error,
                 max_auxiliary_stage_norm_error=0.,
                 candidate_auxiliary_norm_error=0.,
                 max_raw_tangent_rate_error=0.,
                 max_relative_tangent_rate_error=0.,
                 max_auxiliary_defect=float(max_auxiliary_defect),
                 tangent_tolerance=float(tangent_tolerance),
                 coordinate_policy='Y homogeneous lift; C=Y/||Y||; per-step rebase',
                 density_floor_or_mask_added=False)
    if initial_error > initial_pnc_tolerance:
        raise SphereStepError('Initial physical C violates PNC; refusing to normalize it', audit)

    def physical(auxiliary, *, candidate=False):
        if any(not np.all(np.isfinite(value)) for value in auxiliary.values()):
            raise SphereStepError('Nonfinite sphere RK4 proposal; reduce dt', audit)
        norm2 = np.sum(abs(auxiliary['C'])**2, axis=-1)
        defect = float(np.max(abs(norm2-1)))
        if candidate:
            audit['candidate_auxiliary_norm_error'] = defect
        else:
            audit['max_auxiliary_stage_norm_error'] = max(
                audit['max_auxiliary_stage_norm_error'], defect)
        if defect > max_auxiliary_defect or np.any(norm2 <= 0):
            raise SphereStepError('Auxiliary sphere norm defect %.6g exceeds %.6g; '
                                  'reduce dt, do not infer accuracy from mapped PNC' %
                                  (defect, max_auxiliary_defect), audit)
        norm = np.sqrt(norm2)
        result = {key: value.copy() for key, value in auxiliary.items()}
        result['C'] = auxiliary['C']/norm[..., None]
        return result, norm

    def lifted_rate(auxiliary):
        current, norm = physical(auxiliary)
        returned = evaluate(current)
        raw = returned[0] if isinstance(returned, tuple) else returned
        if not isinstance(raw, dict) or set(raw) != set(initial):
            raise ValueError('Sphere RHS must return exactly the complete state keys')
        rate = {key: np.array(value, copy=True) for key, value in raw.items()}
        if any(rate[key].shape != initial[key].shape for key in initial):
            raise ValueError('Sphere RHS shapes must match state shapes')
        if any(not np.all(np.isfinite(value)) for value in rate.values()):
            raise SphereStepError('Nonfinite sphere RHS', audit)
        defect = abs(2*np.real(np.sum(current['C'].conj()*rate['C'], axis=-1)))
        scale = np.maximum(1., 2*np.sqrt(np.sum(abs(rate['C'])**2, axis=-1)))
        raw_error, relative_error = float(defect.max()), float((defect/scale).max())
        audit['rhs_evaluations'] += 1
        audit['max_raw_tangent_rate_error'] = max(audit['max_raw_tangent_rate_error'], raw_error)
        audit['max_relative_tangent_rate_error'] = max(audit['max_relative_tangent_rate_error'], relative_error)
        if relative_error > tangent_tolerance:
            raise SphereStepError('Supplied C_dot is not real-tangent to PNC; '
                                  'refusing to remove a physical radial RHS', audit)
        rate['C'] *= norm[..., None]
        return rate

    def shift(rate, fraction):
        return {key: initial[key]+fraction*dt*rate[key] for key in initial}

    k1 = lifted_rate(initial)
    k2 = lifted_rate(shift(k1, .5))
    k3 = lifted_rate(shift(k2, .5))
    k4 = lifted_rate(shift(k3, 1.))
    proposal = {key: initial[key]+dt*(k1[key]+2*k2[key]+2*k3[key]+k4[key])/6
                for key in initial}
    result, _ = physical(proposal, candidate=True)
    audit['output_pnc_error'] = float(np.max(abs(np.sum(abs(result['C'])**2, axis=-1)-1)))
    return result, audit
