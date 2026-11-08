"""Optional step-doubling error control, without changing any physical RHS.

Each outer interval is transactional: a failure leaves its input unchanged.
All C entries (including tails), transformed parameters, and passive budget
integrals participate in the tolerance. No density mask/energy reset is used.
"""
import numpy as np
from ..gaussian_trajectory.sphere_integrator import sphere_rk4_step,SphereStepError
from ..gaussian_trajectory.gaussian import GaussianRankError
from .projection import EnergyConstraintError


# These are numerical trial-domain failures emitted by the shared Gaussian
# evaluator. A smaller RK trial may remain in the domain. Configuration/shape
# ValueErrors are deliberately NOT caught, and initial states are still checked
# before this integrator is entered. No weights/widths are clipped or repaired.
RETRIABLE_GAUSSIAN_DOMAIN_ERRORS = frozenset({
    'invalid weights or widths; reduce step or change parameterization',
    'unresolved/overflowing Gaussian parameters on this q grid',
    'Gaussian/Jacobian evaluation overflowed',
    'Full representable support required; no automatic floor or mask',
})


class IntegrationResolutionError(FloatingPointError):
    def __init__(self,message,audit):
        super().__init__(message);self.audit=dict(audit)


def adaptive_interval(state,dt,evaluate,*,rtol=1e-8,atol=1e-10,max_halvings=16,
                      max_attempts=2048,max_auxiliary_defect=.05):
    if not (np.isfinite(dt) and dt>0 and np.isfinite(rtol) and np.isfinite(atol) and rtol>0 and atol>0):
        raise ValueError('Positive finite adaptive step/tolerances required')
    if max_halvings<1 or max_attempts<1:raise ValueError('Positive adaptive budgets required')
    current={k:np.array(v,copy=True) for k,v in state.items()}
    elapsed=0.;h=dt;minimum=dt/2**max_halvings
    audit=dict(accepted_substeps=0,rejected_trials=0,minimum_step_au=dt,last_scaled_error=0.,
               max_halvings=max_halvings,max_attempts=max_attempts,
               method='sphere RK4 step doubling; accept two half steps, no Richardson state repair')
    def step(s,h):return sphere_rk4_step(s,h,evaluate,max_auxiliary_defect=max_auxiliary_defect)[0]
    for attempt in range(max_attempts):
        h=min(h,dt-elapsed)
        try:
            coarse=step(current,h);half=step(current,h/2);fine=step(half,h/2)
            # RK4 full-vs-two-half difference = 15 times leading local error
            # of the two-half solution. General complex C is NOT restricted.
            err=max(float(np.max(abs(fine[k]-coarse[k])/
                     (15*(atol+rtol*np.maximum(abs(fine[k]),abs(current[k])))))) for k in current)
            audit['last_scaled_error']=err
            if not np.isfinite(err):err=float('inf')
            if err<=1:
                # The final candidate can leave the Gaussian domain even when
                # all stage states were valid. Reject it here, not after the
                # whole outer interval, so step-size reduction remains possible.
                evaluate(fine)
                current=fine;elapsed+=h;audit['accepted_substeps']+=1
                audit['minimum_step_au']=min(audit['minimum_step_au'],h/2)
                if dt-elapsed<=8*np.finfo(float).eps*dt:return current,audit
                h*=min(2.,max(.2,.8*err**(-.2))) if err>0 else 2.
                continue
            factor=max(.2,min(.8,.8*err**(-.2)))
        except (SphereStepError,GaussianRankError,EnergyConstraintError) as exc:
            audit['last_rejection']=repr(exc);audit['last_rejection_audit']=getattr(exc,'audit',{})
            factor=.5
        except ValueError as exc:
            if str(exc) not in RETRIABLE_GAUSSIAN_DOMAIN_ERRORS:
                raise
            audit['last_rejection']=repr(exc)
            audit['last_rejection_audit']={'kind':'Gaussian trial outside representable domain'}
            factor=.5
        audit['rejected_trials']+=1;h*=factor
        if h<minimum:
            raise IntegrationResolutionError('Adaptive local error/domain guard could not resolve interval; no fallback',audit)
    raise IntegrationResolutionError('Adaptive interval exceeded trial budget; no fallback',audit)
