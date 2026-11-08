"""Fixed, time-independent Hermitian-H energy-constrained amplitude projection.

Let A=sqrt(w)*dLambda/dz, y=sqrt(w)*r and
h=sqrt(w)*Re sum_k C_k^*[(H-E)G]_k. Minimize ||A v-y|| subject
to h.T A v=h.T y. P3 for C is unchanged. With A=U S V.T and u=S V.T v,
the problem is Euclidean projection of U.T y onto c.T u=h.T y,
c=U.T h. This is the requested KKT problem without squaring the condition
number. No state rescaling, reference trajectory, or post-step energy repair.
"""
import numpy as np

from ..gaussian_trajectory.gaussian import (
    AmplitudeProjection, ProjectedRate, GaussianRankError, split_complex_rate)


class EnergyConstraintError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = dict(audit)


def constrained_least_squares(A, y, h, *, rank_rtol=1e-12,
                              constraint_rtol=1e-10, power_atol=1e-12):
    """Full-rank SVD solve; fail on unresolved infeasible energy direction.

    h is already quadrature weighted. power_atol bounds |2*h.T(A v-y)|,
    with h scaled by state norm by the caller. An already-satisfied constraint
    is explicitly reported; it is not a hidden fallback after failed solving.
    """
    if any(np.iscomplexobj(x) for x in (A,y,h)):
        raise ValueError('Amplitude least squares and energy covector must be real')
    A,y,h = (np.asarray(x,dtype=float) for x in (A,y,h))
    if (A.ndim!=2 or A.shape[1]<1 or y.shape!=(A.shape[0],) or h.shape!=y.shape
            or any(not np.all(np.isfinite(x)) for x in (A,y,h))):
        raise ValueError('Finite A[nq,nparam], y[nq], h[nq] required')
    if not (0<rank_rtol<1 and 0<constraint_rtol<1 and np.isfinite(power_atol) and power_atol>=0):
        raise ValueError('Invalid SVD/energy-constraint tolerances')
    U,s,Vh=np.linalg.svd(A,full_matrices=False)
    rank=int(np.count_nonzero(s>s[0]*rank_rtol)) if len(s) and s[0]>0 else 0
    audit=dict(rank=rank,parameter_count=A.shape[1],singular_values=s.tolist(),
               rank_rtol=rank_rtol,condition_A=float(s[0]/s[-1]) if rank==A.shape[1] else float('inf'))
    if rank!=A.shape[1]:
        raise GaussianRankError('Energy projection: Gaussian tangent rank %d < %d'%(rank,A.shape[1]),audit)
    u0=U.T@y; c=U.T@h; target=float(h@y); missing=float(target-c@u0)
    cnorm=float(np.linalg.norm(c)); hnorm=float(np.linalg.norm(h))
    audit.update(unconstrained_power=float(-2*missing),constraint_covector_norm=cnorm,
        energy_covector_norm=hnorm,constraint_resolvability=cnorm/hnorm if hnorm else 0.,
        power_atol=power_atol,constraint_rtol=constraint_rtol)
    if abs(2*missing)<=power_atol:
        u=u0.copy();audit['constraint_status']='already_satisfied_within_tolerance'
    elif cnorm<=constraint_rtol*hnorm or cnorm==0:
        raise EnergyConstraintError('Energy constraint not resolved by current Gaussian tangent',audit)
    else:
        direction=c/cnorm
        u=u0+direction*(missing/cnorm)
        audit['constraint_status']='enforced'
    velocity=Vh.T@(u/s)
    residual=A@velocity-y
    power=float(2*h@residual)
    # Strict physical tolerance plus an explicitly recorded floating-point
    # dot-product bound; no arbitrary cutoff or denominator regularization.
    roundoff=128*np.finfo(float).eps*hnorm*(np.linalg.norm(A@velocity)+np.linalg.norm(y))
    audit.update(constrained_power=power,roundoff_power_bound=float(roundoff),
        correction_amplitude_rate_norm=float(np.linalg.norm(u-u0)),
        parameter_rate_norm=float(np.linalg.norm(velocity)),
        amplitude_residual=float(np.linalg.norm(residual)),
        least_squares_residual=float(np.linalg.norm(U@u0-y)))
    if not np.all(np.isfinite(velocity)) or abs(power)>power_atol+roundoff:
        raise EnergyConstraintError('Energy constraint solve failed its residual check',audit)
    return velocity, s, audit


def energy_constrained_rate(HG, C, bridge, *, rank_rtol=1e-12,
                            constraint_rtol=1e-10, power_atol=1e-12):
    """Only fixed-R, time-independent Hermitian H. No support masks accepted.

    A moving-R/backend RHS must NOT be passed as HG. This experimental API
    takes the physical Hamiltonian action, constructs F=-iHG internally, and
    checks its expectation is real. The driver independently checks H itself.
    """
    C=np.asarray(C,dtype=complex); HG=np.asarray(HG,dtype=complex)
    if np.any(bridge.amplitude<=0) or np.any(bridge.rho<=0):
        raise ValueError('Full representable support required; no automatic floor or mask')
    F=-1j*HG
    Cdot,r,split_audit=split_complex_rate(C,F,bridge.amplitude,pnc_tolerance=1e-10)
    G=bridge.amplitude[:,None]*C; w=bridge.quadrature
    norm=float(w@np.sum(abs(G)**2,axis=1))
    expectation=np.einsum('q,qk,qk->',w,G.conj(),HG)/norm
    if abs(expectation.imag)>1e-11*max(1.,abs(expectation.real)):
        raise ValueError('Hamiltonian expectation not real; energy constraint is not applicable')
    E=float(expectation.real)
    A=np.sqrt(w)[:,None]*bridge.jacobian/(2*bridge.amplitude[:,None])
    y=np.sqrt(w)*r
    h=np.sqrt(w)*np.real(np.sum(C.conj()*(HG-E*G),axis=1))/norm
    velocity,s,audit=constrained_least_squares(A,y,h,rank_rtol=rank_rtol,
        constraint_rtol=constraint_rtol,power_atol=power_atol)
    lamdot=bridge.jacobian@velocity/(2*bridge.amplitude)
    actual=bridge.amplitude[:,None]*Cdot+lamdot[:,None]*C
    delta=actual-F
    power=2*np.real(np.einsum('q,qk,qk->',w,delta.conj(),HG-E*G))/norm
    audit.update(split_audit)
    audit.update(method='experimental_fixed_R_energy_constraint',energy_Ha=E,
        power_actual=float(power),norm_rate=float(2*np.real(np.einsum('q,qk,qk->',w,G.conj(),actual))),
        projection_residual_l2=float(np.sqrt(w@(lamdot-r)**2)),
        actual_vs_eom_rate_l2=float(np.sqrt(w@np.sum(abs(delta)**2,axis=1))),
        P15_max_error=float(np.max(abs(delta-(lamdot-r)[:,None]*C))),
        condition_L=audit['condition_A'],rank_policy='raise',excluded_mass=0.,
        stationary_H_only=True,C_phase_ansatz_added=False)
    p=AmplitudeProjection(velocity,lamdot,4*A.T@A,4*A.T@y,2*s,
                          np.ones(len(w),bool),audit.copy())
    return ProjectedRate(Cdot,velocity,lamdot,actual,r,p,p.support,audit)
