"""Distinct fixed-K amplitude fits using only the physical initial density."""
import numpy as np
from scipy.optimize import least_squares
from ..gaussian_trajectory.gaussian import build_bridge,pack_parameters,GaussianRankError


def fit_distinct(q,w,density,K,*,separation=.65,max_nfev=3000,rank_rtol=1e-12,optimize=True,refine_single=False):
    """Hermite-node seed + non-overlapping center bounds; no duplicate kernels.

    seeding represents a Gaussian convolution by quadrature, but the final
    mixture is fit in the amplitude metric. Actual initial errors and tangent
    conditioning are reported, not silently replaced by the physical target.
    """
    if isinstance(K,bool) or int(K)!=K or K<1 or 3*K-1>len(q):
        raise ValueError('Need positive K with 3*K-1 <= nq')
    if not 0<separation<.95 or max_nfev<1 or not 0<rank_rtol<1:
        raise ValueError('Invalid initialization/fit controls')
    rho=np.asarray(density,float)
    if rho.shape!=np.asarray(q).shape or np.any(rho<0) or not np.all(np.isfinite(rho)) or w@rho<=0:
        raise ValueError('Finite nonnegative initial density required')
    rho=rho/(w@rho);target=np.sqrt(rho)
    mean=w@(q*rho);sigma=np.sqrt(w@((q-mean)**2*rho))
    if sigma<=0: raise ValueError('Unresolved initial width')
    nodes,weights=np.polynomial.hermite_e.hermegauss(K);weights/=np.sqrt(2*np.pi)
    centers=mean+separation*sigma*nodes
    widths=np.full(K,sigma*np.sqrt(1-separation**2) if K>1 else sigma)
    z=pack_parameters(weights,centers,widths,coordinates='transformed')
    def bridge(z):return build_bridge(q,w,z,components=K,coordinates='transformed')
    def residual(z):return np.sqrt(w)*(bridge(z).amplitude-target)
    def jac(z):
        b=bridge(z)
        if np.any(b.amplitude==0):raise ValueError('Unrepresentable fit tails; change grid explicitly')
        return np.sqrt(w)[:,None]*b.jacobian/(2*b.amplitude[:,None])
    success=True;message='K=1 moment initialization' if K==1 else 'Unfitted Hermite convolution seed';nfev=0
    if optimize and (K>1 or refine_single):
        half=.2*np.min(np.diff(centers)) if K>1 else 2*sigma
        lower=np.r_[np.full(K-1,-10.),centers-half,np.full(K,np.log(.2*sigma))]
        upper=np.r_[np.full(K-1,10.),centers+half,np.full(K,np.log(2*sigma))]
        fit=least_squares(residual,z,jac=jac,bounds=(lower,upper),max_nfev=max_nfev,
                          ftol=1e-10,xtol=1e-10,gtol=1e-10)
        z=fit.x;success=bool(fit.success);message=fit.message;nfev=fit.nfev
    b=bridge(z);s=np.linalg.svd(jac(z),compute_uv=False)
    rank=int(np.count_nonzero(s>s[0]*rank_rtol))
    audit=dict(success=success,message=message,nfev=nfev,components=K,
        amplitude_l2_error=float(np.linalg.norm(residual(z))),density_l1_error=float(w@abs(b.rho-rho)),
        rank=rank,parameter_count=len(z),condition_A=float(s[0]/s[-1]),singular_values=s.tolist(),
        separation=separation,min_weight=float(min(b.weights)),centers=b.centers.tolist(),
        widths=b.widths.tolist(),weights=b.weights.tolist(),reference_feedback=False,
        initialization=('single-Gaussian moments (not amplitude minimization)' if K==1 and not refine_single else
            'distinct Hermite seed; constrained amplitude-metric fit' if optimize else 'Hermite convolution seed; no amplitude fitting'))
    if not success:
        exc=ValueError('Initial amplitude fit did not converge: '+message);exc.audit=audit;raise exc
    if rank<len(z):raise GaussianRankError('Initial Gaussian tangent is rank deficient',audit)
    return b,audit
