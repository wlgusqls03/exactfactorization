"""Positive-real F=chi*Lambda nested EF for a two-component electronic spinor.

Derivatives are native FFT derivatives of Psi, time derivative is -i H Psi.
Quotient differentiation is continuum algebra evaluated on spectral samples;
identities alone do NOT certify derivative/basis convergence. No phase unwrap.
Conventions: a=Im<Phi|Dq Phi>, b=Im<Phi|DR Phi>, alpha=Im<Xi|DR Xi>.
Scalar shifts as epsilon'=epsilon+theta_t; outer force=-epsilon2_R+alpha_t.
"""
import numpy as np
from .model import derivative


def ratio(a,b):
    """Unfloored division; exactly zero support stays undefined (NaN)."""
    a,b=np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.full(a.shape,np.nan,dtype=np.result_type(a,float)),where=b>0)


@np.errstate(over='ignore',invalid='ignore',divide='ignore')
def analyze(u,p):
    """Return native fields [NR,Nq] or [NR], all atomic units.

    Epsilon1 routes: expectation cond+geo+GD vs F-equation inversion.
    Epsilon2 routes: direct Xi expectation vs chi inversion vs nested integral.
    Derivative ratios from Psi avoid FFT across masked conditional states.
    """
    c=p.c;dr,dq=p.dr,p.dq;M=c.mass;R,q=c.grids()
    ur=derivative(u,dr,0);urr=derivative(u,dr,0,2)
    uq=derivative(u,dq,1);uqq=derivative(u,dq,1,2)
    ut=-1j*p.action(u);utr=derivative(ut,dr,0)
    dot=lambda v:np.sum(u.conj()*v,axis=-1)
    joint=np.sum(abs(u)**2,axis=-1);rho=joint.sum(axis=1)*dq
    fq=ratio(dot(uq).real,joint);fr=ratio(dot(ur).real,joint)
    ft=ratio(dot(ut).real,joint)
    fqq=ratio(dot(uqq).real+np.sum(abs(uq)**2,axis=-1),joint)-fq*fq
    frr=ratio(dot(urr).real+np.sum(abs(ur)**2,axis=-1),joint)-fr*fr
    a=ratio(dot(uq).imag,joint);b=ratio(dot(ur).imag,joint)
    aq=ratio(dot(uqq).imag,joint)-2*a*fq
    br=ratio(dot(urr).imag,joint)-2*b*fr
    gq=(ratio(np.sum(abs(uq)**2,axis=-1),joint)-fq*fq-a*a)/2
    gr=(ratio(np.sum(abs(ur)**2,axis=-1),joint)-fr*fr-b*b)/(2*M)
    cond=ratio(dot(np.einsum('...ij,...j->...i',p.v,u)),joint)
    gd=-1j*(ratio(dot(ut),joint)-ft)
    e1=cond+gq+gr+gd
    inv1=.5*(fqq-a*a)+(frr-b*b)/(2*M)+1j*(ft+a*fq+aq/2+(b*fr+br/2)/M)
    inner=lambda v:dot(v).sum(axis=1)*dq
    r1=2*inner(ur).real
    r2=2*(inner(urr).real+np.sum(abs(ur)**2,axis=(1,2))*dq)
    rt=2*inner(ut).real
    logR=ratio(r1,2*rho);curv=ratio(r2,2*rho)-logR**2
    alpha=ratio(inner(ur).imag,rho)
    ar=ratio(inner(urr).imag,rho)-2*alpha*logR
    at=ratio(np.sum((ut.conj()*ur+u.conj()*utr).imag,axis=(1,2))*dq,rho)-alpha*ratio(rt,rho)
    g2=(ratio(np.sum(abs(ur)**2,axis=(1,2))*dq,rho)-logR**2-alpha**2)/(2*M)
    cond2=ratio(1j*inner(ut)+inner(urr)/(2*M),rho)
    gd2=-1j*(ratio(inner(ut),rho)-ratio(rt,2*rho))
    e2=cond2+g2+gd2
    inv2=(curv-alpha**2)/(2*M)+1j*(ratio(rt,2*rho)+(ar+2*alpha*logR)/(2*M))
    urrr=derivative(u,dr,0,3)
    r3=2*inner(urrr).real+6*np.sum((ur.conj()*urr).real,axis=(1,2))*dq
    cr=ratio(r3,2*rho)-ratio(r1*r2,rho**2)+.5*ratio(r1**3,rho**3)
    e2r=cr/(2*M)-alpha*ar/M
    F=np.sqrt(joint);chi=np.sqrt(rho);lam=ratio(F,chi[:,None]);phi=ratio(u,F[...,None])
    lr=lam*(fr-logR[:,None]);lq=lam*fq
    lt=lam*(ft-ratio(rt,2*rho)[:,None])
    integrand=.5*(lq*lq+a*a*lam*lam)+lam*lam*e1+(lr*lr+(b-alpha[:,None])**2*lam*lam)/(2*M)-1j*lam*lt
    # Undefined values at exact nodes have zero integration weight, not a filled phase.
    nested=np.sum(np.where(joint>0,integrand,0),axis=1)*dq
    pnc=np.sum(abs(phi)**2,axis=-1)-1
    rebuilt=chi[:,None,None]*lam[...,None]*phi
    rec=np.sqrt(np.sum(np.where(joint[...,None]>0,abs(rebuilt-u)**2,0))*dr*dq)
    jR=dot(ur).imag/M;jq=dot(uq).imag
    continuity=2*dot(ut).real+derivative(jR,dr,0).real+derivative(jq,dq,1).real
    return dict(R=R,q=q,rho_qR=joint,rho_R=rho,lambda_density=lam**2,
        a=a,b=b,alpha=alpha,alpha_t=at,epsilon1=e1,epsilon1_inv=inv1,
        cond=cond,geo_q=gq,geo_R=gr,GI=cond+gq+gr,GD=gd,
        epsilon2=e2,epsilon2_inv=inv2,epsilon2_nested=nested,
        epsilon2_cond=cond2,epsilon2_geo=g2,epsilon2_GD=gd2,
        force=-e2r+at,current_R=jR.sum(axis=1)*dq,current_qR=jR,current_qq=jq,
        berry=2*ratio(np.sum((uq.conj()*ur).imag,axis=-1),joint)-2*b*fq+2*a*fr,
        continuity=continuity,electronic_PNC=pnc,photon_PNC=(lam*lam).sum(axis=1)*dq-1,
        alpha_nested=np.sum(np.where(joint>0,lam*lam*b,0),axis=1)*dq,
        reconstruction_L2=rec)


def diagnostics(f):
    """Occupied relative thresholds 1e-4/1e-6/1e-8; include excluded mass."""
    out={'reconstruction_L2':float(f['reconstruction_L2']),'supports':{}}
    for eta in (1e-4,1e-6,1e-8):
        item={}
        for density,fields in [(f['rho_qR'],dict(epsilon1_route=f['epsilon1']-f['epsilon1_inv'],
                epsilon1_imag=f['epsilon1'].imag,PNC_e=f['electronic_PNC'],continuity=f['continuity'])),
                (f['rho_R'],dict(epsilon2_route=f['epsilon2']-f['epsilon2_inv'],
                epsilon2_nested=f['epsilon2']-f['epsilon2_nested'],epsilon2_imag=f['epsilon2'].imag,
                PNC_q=f['photon_PNC'],alpha_nested=f['alpha']-f['alpha_nested']))]:
            mask=density>eta*density.max();weight=density[mask]
            for name,v in fields.items():
                item[name]=float(np.sqrt(np.sum(weight*abs(v[mask])**2)/weight.sum()))
            item['excluded_'+str(density.ndim)+'D']=float(density[~mask].sum()/density.sum())
        out['supports'][str(eta)]=item
    return out


def compact(f,stride=2):
    """Native-grid display decimation ONLY, after full-resolution EF calculation.

    Exact paper cuts retained at native resolution. Do not differentiate the
    decimated movie arrays or call them a new propagated wavefunction.
    """
    maps=['rho_qR','epsilon1','cond','geo_q','geo_R','GI','GD','a','b','berry']
    lines=['rho_R','epsilon2','epsilon2_cond','epsilon2_geo','epsilon2_GD','alpha','alpha_t','force','current_R']
    out={k:f[k].real[::stride,::stride] for k in maps}
    out.update({k:f[k].real for k in lines});out['R_line']=f['R']
    out['R']=f['R'][::stride];out['q']=f['q'][::stride]
    for axis,targets in [('q',[0.,1.5]),('R',[2.,4.])]:
        for n,target in enumerate(targets):
            idx=int(np.argmin(abs(f[axis]-target)))
            out[f'cut_{axis}{n}_at']=f[axis][idx]
            for k in ['rho_qR','epsilon1','cond','GI','GD','geo_q','geo_R']:
                out[f'cut_{axis}{n}_{k}']=(f[k][:,idx] if axis=='q' else f[k][idx]).real
    out['q_line']=f['q']
    return out
