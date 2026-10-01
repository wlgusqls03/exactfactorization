"""QHJ diagnostics from SAVED full waves, not derivatives of masked fields.

Positive outer marginal: epsilon2=(sqrt(rho)''/sqrt(rho)-alpha**2)/(2M).
Qn=-sqrt(rho)''/(2M sqrt(rho)); F_EF=-epsilon2_R+alpha_t.
F_EF - Qn_R = alpha_t + (alpha/M) alpha_R.
The RHS is M times material acceleration of marginal probability flow,
not acceleration of every particle or photon-conditioned configuration.
All units atomic. Original spectral R derivatives and instantaneous H Psi
are reused; no coarse-frame temporal finite difference, smoothing or fill.
"""
import numpy as np
from .real_grid_mcef_fields import derivative,divide


def outer_qhj(u,Hu,dr,internal_volume,mass):
    """u,Hu:(NR,internal...) physical wave and Hamiltonian action.

    internal_volume=dx*dq or dx*dQ (or dx for n=0 sector).
    Returns (NR,) arrays, including independent wave-jet density derivatives.
    Identity residuals test algebra, NOT independent grid convergence.
    """
    axes=tuple(range(1,u.ndim));w=internal_volume;M=mass
    inner=lambda v:np.sum(u.conj()*v,axis=axes)*w
    ur=derivative(u,dr,0);urr=derivative(u,dr,0,2);ut=-1j*Hu
    rho=np.sum(abs(u)**2,axis=axes)*w
    r1=2*inner(ur).real
    r2=2*(inner(urr).real+np.sum(abs(ur)**2,axis=axes)*w)
    r3=2*inner(derivative(u,dr,0,3)).real+6*np.sum((ur.conj()*urr).real,axis=axes)*w
    rt=2*inner(ut).real
    alpha=divide(inner(ur).imag,rho)
    ar=divide(inner(urr).imag,rho)-alpha*divide(r1,rho)
    at=divide(np.sum((ut.conj()*ur+u.conj()*derivative(ut,dr,0)).imag,axis=axes)*w,rho)-alpha*divide(rt,rho)
    curvature=divide(r2,2*rho)-divide(r1**2,4*rho**2)
    curvature_R=divide(r3,2*rho)-divide(r1*r2,rho**2)+divide(r1**3,2*rho**3)
    quantum=-curvature/(2*M);quantum_force=curvature_R/(2*M)
    eps=(curvature-alpha**2)/(2*M)
    scalar_force=-curvature_R/(2*M)+alpha*ar/M
    ef_force=scalar_force+at
    material=at+alpha*ar/M
    return dict(rho_R=rho,alpha=alpha,alpha_R=ar,alpha_t=at,
        Q_nuclear=quantum,epsilon2=eps,scalar_force=scalar_force,
        EF_force=ef_force,quantum_force=quantum_force,
        net_flow_force=ef_force+quantum_force,material_momentum_derivative=material,
        balance_residual=ef_force+quantum_force-material,
        QHJ_residual=eps+quantum+alpha**2/(2*M))


def joint_qhj(u,Hu,dr,dx,dq,mass,block=4):
    """Physical Psi_q:(NR,Nx,Nq). First-level positive F=sqrt(rho_qR).

    B=partial_q b-partial_R a. Fq=-epsilon1_q+a_t-(b/M)B;
    FR=-epsilon1_R+b_t+a B. Add -grad Qjoint to get material momenta
    derivatives. Scalar gradients use analytic wave jets / QHJ inversion,
    NOT FFT of masked scalars; agreement is an algebraic diagnostic.
    """
    M=mass;ur=derivative(u,dr,0);urr=derivative(u,dr,0,2)
    urrr=derivative(u,dr,0,3);ut=-1j*Hu;utr=derivative(ut,dr,0)
    out={}
    for start in range(0,len(u),block):
        s=slice(start,start+block);z=u[s];zr=ur[s];zrr=urr[s];zt=ut[s]
        zq=derivative(z,dq,2);zqq=derivative(z,dq,2,2);zqqq=derivative(z,dq,2,3)
        zrq=derivative(zr,dq,2);zrrq=derivative(zrr,dq,2);zrqq=derivative(zr,dq,2,2)
        I=lambda a,b:np.sum(a.conj()*b,axis=1)*dx
        rho=I(z,z).real;rq=2*I(z,zq).real;rr=2*I(z,zr).real;rt=2*I(z,zt).real
        rqq=2*(I(z,zqq)+I(zq,zq)).real;rrr=2*(I(z,zrr)+I(zr,zr)).real
        rrq=2*(I(z,zrq)+I(zr,zq)).real
        rqqq=2*I(z,zqqq).real+6*I(zq,zqq).real
        rrrr=2*I(z,urrr[s]).real+6*I(zr,zrr).real
        rqqr=2*(I(zr,zqq)+I(z,zrqq)).real+4*I(zq,zrq).real
        rrrq=2*(I(zq,zrr)+I(z,zrrq)).real+4*I(zr,zrq).real
        curvature=lambda ri,rii:divide(rii,2*rho)-divide(ri**2,4*rho**2)
        gradcurv=lambda ri,rii,rk,rik,riik:divide(riik,2*rho)-divide(rii*rk+ri*rik,2*rho**2)+divide(ri**2*rk,2*rho**3)
        Qj=-curvature(rq,rqq)/2-curvature(rr,rrr)/(2*M)
        FQq=gradcurv(rq,rqq,rq,rqq,rqqq)/2+gradcurv(rr,rrr,rq,rrq,rrrq)/(2*M)
        FQr=gradcurv(rq,rqq,rr,rrq,rqqr)/2+gradcurv(rr,rrr,rr,rrr,rrrr)/(2*M)
        a=divide(I(z,zq).imag,rho);b=divide(I(z,zr).imag,rho)
        aq=divide(I(z,zqq).imag,rho)-a*divide(rq,rho)
        br=divide(I(z,zrr).imag,rho)-b*divide(rr,rho)
        ar=divide((I(zr,zq)+I(z,zrq)).imag,rho)-a*divide(rr,rho)
        bq=divide((I(zq,zr)+I(z,zrq)).imag,rho)-b*divide(rq,rho)
        at=divide((I(zt,zq)+I(z,derivative(zt,dq,2))).imag,rho)-a*divide(rt,rho)
        bt=divide((I(zt,zr)+I(z,utr[s])).imag,rho)-b*divide(rt,rho)
        B=bq-ar
        sq=-FQq+a*aq+b*bq/M;sr=-FQr+a*ar+b*br/M
        v=dict(rho_qR=rho,a=a,b=b,berry=B,Q_joint=Qj,epsilon1_QHJ=-Qj-a*a/2-b*b/(2*M),
            photon_scalar_force=sq,photon_time_force=at,photon_berry_force=-b*B/M,
            photon_quantum_force=FQq,photon_net_force=sq+at-b*B/M+FQq,
            photon_material=at+a*aq+b*ar/M,
            nuclear_scalar_force=sr,nuclear_time_force=bt,nuclear_berry_force=a*B,
            nuclear_quantum_force=FQr,nuclear_net_force=sr+bt+a*B+FQr,
            nuclear_material=bt+a*bq+b*br/M)
        for key,value in v.items():
            if key not in out:out[key]=np.empty((len(u),)+value.shape[1:])
            out[key][s]=value
    return out
