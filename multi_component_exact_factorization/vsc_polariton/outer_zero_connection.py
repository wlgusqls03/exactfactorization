"""Fixed-anchor outer gauge; no unwrap, moving anchors, or node bridging.

Phi'=g1 Phi, Lambda'=g1* g2 Lambda, chi'=g2* chi.
Here g1=1, g2=exp(i eta), eta_R=-alpha, kappa=eta_t.
Atomic units. Only the component containing a fixed grid anchor is defined.
Other components are deliberately undefined, not independently height-aligned.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from .phase7_support import budget_support,weighted_rms
from .phase7_outer_gauge import derivative_interior

LIMITS=dict(alpha_relative=1e-3,current_momentum_relative=1e-3,force_absolute=1e-5,
            reconstruction=1e-10,rate_reconstruction=1e-9)


def integrate_anchor(R,value,rho,anchor,budget=1e-8):
    """(NR,) values; fixed anchor index. Return spline/trapezoid integrals.

    Both use exactly the same connected probability-budget support.
    Neither extends across an excluded gap. No automatic anchor relocation.
    """
    dr=float(R[1]-R[0]);mask,meta,labels=budget_support(rho,dr,budget)
    result=np.full(len(R),np.nan);trap=result.copy()
    if not mask[anchor]:return result,trap,dict(meta,anchor_occupied=False),np.zeros(len(R),bool)
    support=labels==labels[anchor];ids=np.flatnonzero(support)
    if len(ids)<7 or not np.isfinite(value[ids]).all():
        return result,trap,dict(meta,anchor_occupied=True,too_short_or_nonfinite=True),np.zeros(len(R),bool)
    anti=CubicSpline(R[ids],value[ids]).antiderivative()
    result[ids]=anti(R[ids])-anti(R[anchor])
    temp=cumulative_trapezoid(value[ids],R[ids],initial=0)
    trap[ids]=temp-temp[np.flatnonzero(ids==anchor)[0]]
    return result,trap,dict(meta,anchor_occupied=True,
        retained_probability=float(np.sum(rho[ids])/np.sum(rho))),support


def fixed_outer_fields(R,rho,alpha,alpha_t,epsilon,force,anchor,budget=1e-8):
    """Snapshot gauge and instantaneous rate, fixed reference eta=kappa=0.

    Arrays (NR,), a.u.; alpha_t must come independently from H Psi.
    Force-work curve is separate and NOT used to construct epsilon_A0.
    """
    eta,eta_trap,meta,s=integrate_anchor(R,-alpha,rho,anchor,budget)
    k,kt,_,_=integrate_anchor(R,-alpha_t,rho,anchor,budget)
    work,wt,_,_=integrate_anchor(R,-force,rho,anchor,budget)
    eps=epsilon+k;aligned=eps-eps[anchor]
    g=np.exp(1j*eta)
    return dict(eta=eta,kappa=k,g=g,epsilon_A0=eps,epsilon_A0_aligned=aligned,
        force_work=work,support=s,eta_trapezoid=eta_trap,kappa_trapezoid=kt,
        work_trapezoid=wt),meta


def validate_wave(u,ut,R,volume,mass,outer,gauge):
    """Native saved wave (NR, internal...), ut=-i H u; finite-difference checks.

    Re-evaluate transformed conditional connection and current; never set it
    to zero by substitution. FD4/6 convergence is reported, not assumed.
    Reconstruct both u and ut with transformed factors and their rates.
    """
    ids=np.flatnonzero(gauge['support']);dr=float(R[1]-R[0])
    if len(ids)<7:return dict(status='UNDEFINED',pass_gate=False)
    rho=outer['rho_R'][ids];C=np.sqrt(rho)
    flat=u.reshape(len(u),-1);flat_t=ut.reshape(len(u),-1)
    rt=np.zeros(len(ids))
    for start in range(0,flat.shape[1],128):
        z=flat[ids,start:start+128];zt=flat_t[ids,start:start+128]
        rt+=2*np.sum((z.conj()*zt).real,axis=1)*volume
    Ct=rt/(2*C)
    g=gauge['g'][ids];k=gauge['kappa'][ids]
    cp=C*g.conj();cp_t=g.conj()*(Ct-1j*k*C)
    ap_fields={order:np.zeros(len(ids)-order) for order in (4,6)}
    rec_error=0.;rate_error=0.
    for start in range(0,flat.shape[1],128):
        z=flat[ids,start:start+128];zt=flat_t[ids,start:start+128]
        gamma=z/C[:,None];gt=(zt-gamma*Ct[:,None])/C[:,None]
        gammap=gamma*g[:,None];gammap_t=g[:,None]*(gt+1j*k[:,None]*gamma)
        rec_error+=float(np.sum(abs(gammap*cp[:,None]-z)**2)*volume*dr)
        rate_error+=float(np.sum(abs(gammap_t*cp[:,None]+gammap*cp_t[:,None]-zt)**2)*volume*dr)
        for order in (4,6):
            h=order//2;dg=derivative_interior(gammap,dr,order)
            ap_fields[order]+=np.sum((gammap[h:-h].conj()*dg).imag,axis=1)*volume
    report=dict(reconstruction=np.sqrt(rec_error),rate_reconstruction=np.sqrt(rate_error))
    for order in (4,6):
        h=order//2;inner=ids[h:-h];w=rho[h:-h]*dr
        ap=ap_fields[order]
        dc=derivative_interior(cp,dr,order)
        current=(cp[h:-h].conj()*dc).imag/mass+rho[h:-h]*ap/mass
        f=-derivative_interior(gauge['epsilon_A0'][ids],dr,order)
        rms=lambda x:float(np.sqrt(np.average(abs(x)**2,weights=w)))
        scale=max(1.,rms(outer['alpha'][inner]))
        report['fd'+str(order)]=dict(alpha_relative=rms(ap)/scale,
            current_momentum_relative=rms((current-rho[h:-h]*outer['alpha'][inner]/mass)*mass/rho[h:-h])/scale,
            force_absolute=rms(f-outer['EF_force'][inner]))
    check=report['fd6'];report['limits']=LIMITS
    report['pass_gate']=all(check[k]<LIMITS[k] for k in check) and all(report[k]<LIMITS[k] for k in ('reconstruction','rate_reconstruction'))
    report['status']='SNAPSHOT_PASS' if report['pass_gate'] else 'SNAPSHOT_FAIL'
    report['scope']='Fixed-anchor snapshot only; no assertion of full temporal/basis convergence.'
    return report


def link_jet(S,St,min_overlap=1e-8):
    """Open fixed 1D chain; S=<Phi_i|Phi_i+1>, instantaneous St. No regularization."""
    if not np.isfinite(S).all() or not np.isfinite(St).all() or np.any(abs(S)<min_overlap):
        raise ValueError('Unreliable link: split support or refine, never bridge')
    g=np.r_[1.+0j,np.cumprod(S.conj()/abs(S))]
    k=np.r_[0.,-np.cumsum(np.imag(St/S))]
    return g,k
