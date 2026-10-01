"""Exact n=0-sector nested EF, positive marginals, atomic units.

Psi(R,x,q)=u(R,x)*vacuum(q); no BO elimination and no propagation.
Reuse native FFT derivatives from real_grid_mcef_fields. Photon factors and
their derivatives are analytic, avoiding a large redundant 3D allocation.
epsilon1=epsilon2+omega*(Q**2-1)/2 in this gauge; a=0, b=alpha.
Independent expectation and inversion routes remain available for epsilon2.
"""
import numpy as np
from .real_grid_mcef_fields import derivative,divide


def analyze_vacuum(u,packet,Q):
    """u:(NR,Nx), Q:(NQ,); returns physical-q fields (NR,NQ)/(NR,).

    Provenance: same PF operator and gauge as real_grid_mcef_fields.analyze,
    restricted exactly to g=0,n=0. Full H u supplies instantaneous dt u.
    """
    if float(packet['g_chi'])!=0:raise ValueError('Only exact uncoupled vacuum sector')
    M,dr,dx,w=(float(packet[k]) for k in ('mass','dR','dx','omega'))
    if u.shape!=(len(packet['R']),len(packet['x'])):raise ValueError('Wrong wave shape')
    ur=derivative(u,dr,0);urr=derivative(u,dr,0,2)
    hel=-derivative(u,dx,1,2)/2+packet['potential']*u
    ut=-1j*(-urr/(2*M)+hel+w*u/2)
    inner=lambda v:np.sum(u.conj()*v,axis=1)*dx
    rho=np.sum(abs(u)**2,axis=1)*dx
    r1=2*inner(ur).real;r2=2*(inner(urr).real+np.sum(abs(ur)**2,axis=1)*dx)
    rt=2*inner(ut).real;alpha=divide(inner(ur).imag,rho)
    ar=divide(inner(urr).imag,rho)-alpha*divide(r1,rho)
    logR=divide(r1,2*rho);curv=divide(r2,2*rho)-logR**2
    geo=(divide(np.sum(abs(ur)**2,axis=1)*dx,rho)-logR**2-alpha**2)/(2*M)
    cond=divide(inner(hel),rho)+w/2
    gd=-1j*(divide(inner(ut),rho)-divide(rt,2*rho))
    ea=cond+geo+gd
    eb=(curv-alpha**2)/(2*M)+1j*(divide(rt,2*rho)+(ar+2*alpha*logR)/(2*M))
    utr=derivative(ut,dr,0)
    at=divide(np.sum((ut.conj()*ur+u.conj()*utr).imag,axis=1)*dx,rho)-alpha*divide(rt,rho)
    r3=2*inner(derivative(u,dr,0,3)).real+6*np.sum((ur.conj()*urr).real,axis=1)*dx
    cr=divide(r3,2*rho)-divide(r1*r2,rho**2)+.5*divide(r1**3,rho**3)
    er=cr/(2*M)-alpha*ar/M
    q=Q/np.sqrt(w);vac=np.sqrt(w/np.pi)*np.exp(-Q**2)
    lam=np.broadcast_to(vac,(len(u),len(Q))).copy();joint=rho[:,None]*vac
    phi=divide(u,np.sqrt(rho)[:,None]);rebuilt=phi*np.sqrt(rho)[:,None]
    occupied=rho>0
    return dict(R=packet['R'],q=q,Q=Q,rho_R=rho,rho_qR=joint,lambda_density=lam,
        a=np.zeros_like(joint),b=np.broadcast_to(alpha[:,None],joint.shape).copy(),alpha=alpha,
        current=rho*alpha/M,alpha_t=at,epsilon2_R=er,force=-er+at,
        epsilon2_A=ea,epsilon2_B=eb,epsilon2_cond=cond,epsilon2_geo=geo,epsilon2_GD=gd,
        epsilon1_A=ea[:,None]+w*(Q[None,:]**2-1)/2,
        epsilon1_B=eb[:,None]+w*(Q[None,:]**2-1)/2,
        conditional_q=np.zeros(len(u)),
        electronic_PNC_error=abs(np.sum(abs(phi)**2,axis=1)*dx-1),
        photon_PNC_error=abs(lam.sum(axis=1)*(q[1]-q[0])-1),
        reconstruction_L2=float(np.sqrt(np.sum(abs(rebuilt[occupied]-u[occupied])**2)*dx*dr)))
