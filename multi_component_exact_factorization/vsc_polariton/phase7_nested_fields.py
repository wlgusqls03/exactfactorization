"""Positive-marginal nested EF from full Psi and native instantaneous actions.

Native arrays (R,x,n), displayed fields (R,q), units atomic. Quotient
derivatives follow the continuously reconstructed wave, not finite differences
of floor-filled conditional factors. No phase or normalization repair.
"""
import numpy as np
from .phase7_native import R_derivative,internal_action,bare_action
from .phase7_fock_to_q import transform,backproject


def ratio(a,b):
    out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan,dtype=np.result_type(a,b,float))
    return np.divide(a,b,out=out,where=np.broadcast_to(b,np.shape(out))>0)


def outer_fields(u,p):
    dx,dr,M=map(float,(p['dx'],p['dR'],p['mass']))
    ur=R_derivative(u,dr);urr=R_derivative(u,dr,2)
    inside=internal_action(u,p);ut=-1j*(inside-urr/(2*M))
    inner=lambda v:np.sum(u.conj()*v,axis=(1,2))*dx
    rho=np.sum(abs(u)**2,axis=(1,2))*dx
    rho1=2*inner(ur).real
    rho2=2*(inner(urr).real+np.sum(abs(ur)**2,axis=(1,2))*dx)
    rhot=2*inner(ut).real
    alpha=ratio(inner(ur).imag,rho)
    ar=ratio(inner(urr).imag,rho)-alpha*ratio(rho1,rho)
    log1=ratio(rho1,2*rho)
    r1=ratio(rho1,rho);r2=ratio(rho2,rho)
    curvature=r2/2-r1*r1/4
    geo=(ratio(np.sum(abs(ur)**2,axis=(1,2))*dx,rho)-log1**2-alpha**2)/(2*M)
    gd=-1j*(ratio(inner(ut),rho)-ratio(rhot,2*rho))
    cond=ratio(inner(inside),rho)
    ea=cond+geo+gd
    eb=(curvature-alpha**2)/(2*M)+1j*(ratio(rhot,2*rho)+(ar+2*alpha*log1)/(2*M))
    utr=R_derivative(ut,dr)
    at=ratio(np.sum((ut.conj()*ur+u.conj()*utr).imag,axis=(1,2))*dx,rho)-alpha*ratio(rhot,rho)
    del utr
    urrr=R_derivative(u,dr,3)
    rho3=2*inner(urrr).real+6*np.sum((ur.conj()*urr).real,axis=(1,2))*dx
    del urrr
    curvature_r=ratio(rho3,2*rho)-r1*r2+r1*r1*r1/2
    epsilon_r=curvature_r/(2*M)-alpha*ar/M
    force=-epsilon_r+at
    return dict(rho_R=rho,chi=np.sqrt(rho),alpha=alpha,alpha_R=ar,alpha_t=at,
        rho_R_t=rhot,chi_log_R=log1,epsilon2_A=ea,epsilon2_B=eb,
        epsilon2_cond=cond,epsilon2_geo=geo,epsilon2_GD=gd,force=force,
        epsilon2_R=epsilon_r),ur,urr,ut


def nested_fields(u,p,grid,outer,ur,urr,ut,block=8):
    nr,nx,nf=u.shape;nq=len(grid['q'])
    dx,dr,M,w=map(float,(p['dx'],p['dR'],p['mass'],p['omega']))
    q=grid['q'];weights=grid['weights'];out={}
    recon2=back2=0.;pnc=0.;lambda_pnc=0.;moments=np.zeros(4)
    for start in range(0,nr,block):
        sl=slice(start,min(start+block,nr))
        psi=transform(u[sl],grid);dq=transform(u[sl],grid,1);dqq=transform(u[sl],grid,2)
        dR=transform(ur[sl],grid);dRR=transform(urr[sl],grid);dt=transform(ut[sl],grid)
        inner=lambda v:np.sum(psi.conj()*v,axis=1)*dx
        rho=np.sum(abs(psi)**2,axis=1)*dx;F=np.sqrt(rho)
        rhoq=2*inner(dq).real;rhoR=2*inner(dR).real;rhot=2*inner(dt).real
        rhoqq=2*(inner(dqq).real+np.sum(abs(dq)**2,axis=1)*dx)
        rhoRR=2*(inner(dRR).real+np.sum(abs(dR)**2,axis=1)*dx)
        fq=ratio(rhoq,2*rho);fr=ratio(rhoR,2*rho);ft=ratio(rhot,2*rho)
        fqq=ratio(rhoqq,2*rho)-fq**2;frr=ratio(rhoRR,2*rho)-fr**2
        phi=ratio(psi,F[:,None,:])
        phiq=ratio(dq-psi*fq[:,None,:],F[:,None,:])
        phiR=ratio(dR-psi*fr[:,None,:],F[:,None,:])
        phit=ratio(dt-psi*ft[:,None,:],F[:,None,:])
        a=ratio(inner(dq).imag,rho);b=ratio(inner(dR).imag,rho)
        aq=ratio(inner(dqq).imag,rho)-a*ratio(rhoq,rho)
        bR=ratio(inner(dRR).imag,rho)-b*ratio(rhoR,rho)
        geo_q=(np.sum(abs(phiq)**2,axis=1)*dx-a*a)/2
        geo_R=(np.sum(abs(phiR)**2,axis=1)*dx-b*b)/(2*M)
        gd=-1j*np.sum(phi.conj()*phit,axis=1)*dx
        packet_block=dict(p,tx=p['tx'],potential=p['potential'][sl])
        bare=transform(bare_action(u[sl],packet_block),grid)
        harmonic=.5*w*w*q*q
        light=np.sqrt(2*w)*float(p['g_chi'])*p['mu'][sl,:,None]*q
        hpsi=bare+(harmonic[None,None,:]+light+p['dse'][sl,:,None])*psi
        cond=ratio(inner(hpsi),rho)
        ea=cond+geo_q+geo_R+gd
        eb=.5*(fqq-a*a)+(frr-b*b)/(2*M)+1j*(ft+a*fq+aq/2+(b*fr+bR/2)/M)
        # Independent conditional electronic EOM action, not GD by subtraction.
        coupling=np.zeros_like(phi)
        for ds,dss,phids,fs,fss,A,As,mass in (
            (dq,dqq,phiq,fq,fqq,a,aq,1.),(dR,dRR,phiR,fr,frr,b,bR,M)):
            phi2=ratio(dss,F[:,None,:])-2*phids*fs[:,None,:]-phi*fss[:,None,:]
            D=-1j*phids-A[:,None,:]*phi
            D2=-phi2+2j*A[:,None,:]*phids+(1j*As+A*A)[:,None,:]*phi
            coupling+=(D2/2+(-1j*fs+A)[:,None,:]*D)/mass
        eom=ratio(hpsi,F[:,None,:])+coupling-ea[:,None,:]*phi-1j*phit
        defect=1j*dt+.5*dqq+dRR/(2*M)-hpsi
        berry=2*ratio(np.sum((dq.conj()*dR).imag,axis=1)*dx,rho)-2*b*fq+2*a*fr
        chi=outer['chi'][sl,None]
        lam=ratio(F,chi)
        lamq=lam*fq;lamR=lam*(fr-outer['chi_log_R'][sl,None])
        lamt=lam*(ft-ratio(outer['rho_R_t'][sl],2*outer['rho_R'][sl])[:,None])
        lam2=lam*lam;al=outer['alpha'][sl,None]
        e2_integrand=.5*(lamq**2+a*a*lam2)+lam2*ea+(lamR**2+(b-al)**2*lam2)/(2*M)-1j*lam*lamt
        e2_nested=np.sum(e2_integrand*weights,axis=1)
        pnc_values=np.sum(abs(phi)**2,axis=1)*dx
        valid=rho>0
        pnc=max(pnc,float(np.max(abs(pnc_values[valid]-1))))
        lambda_pnc=max(lambda_pnc,float(np.max(abs(np.sum(lam2*weights,axis=1)-1))))
        rebuilt=phi*lam[:,None,:]*chi[:,None,:]
        recon2+=float(np.sum(np.nan_to_num(abs(rebuilt-psi)**2)*weights)*dx*dr)
        back=backproject(psi,grid)
        back2+=float(np.sum(abs(back-u[sl])**2)*dx*dr)
        normq=np.sum(rho*weights)*dr
        qmoment=np.sum(rho*weights*q)*dr
        pmoment=np.sum(inner(dq).imag*weights)*dr
        occupation=(np.sum(np.sum(abs(dq)**2,axis=1)*dx*weights)/2+
                    np.sum(rho*harmonic*weights))/w*dr-normq/2
        moments+=np.array([normq,qmoment,pmoment,occupation])
        # Project the unnormalized wave, avoiding 0 * undefined conditional
        # character in empty tails when computing physical BO populations.
        bo=np.einsum('rxj,rxq->rjq',p['phi'][sl].conj(),psi)*dx
        bo_density=np.moveaxis(abs(bo)**2,1,2)
        fields=dict(rho_qR=rho,lambda_density=lam2,a=a,b=b,berry=berry,
            epsilon1_A=ea,epsilon1_B=eb,epsilon1_cond=cond,epsilon1_GD=gd,
            epsilon1_qgeo=geo_q,epsilon1_Rgeo=geo_R,
            epsilon1_bare=ratio(inner(bare),rho),epsilon1_harmonic=np.broadcast_to(harmonic,rho.shape),
            epsilon1_LM=ratio(inner(light*psi),rho),epsilon1_DSE=ratio(inner(p['dse'][sl,:,None]*psi),rho),
            electronic_eom_norm=np.sqrt(np.sum(abs(eom)**2,axis=1)*dx),
            projection_defect_norm=np.sqrt(np.sum(abs(defect)**2,axis=1)*dx),
            epsilon2_nested=e2_nested,alpha_nested=np.sum(lam2*b*weights,axis=1),
            rho_R_q=np.sum(rho*weights,axis=1),BO_density=bo_density,
            BO_character=ratio(bo_density,rho[:,:,None]),
            electronic_PNC_error=abs(pnc_values-1),
            photon_PNC_error=abs(np.sum(lam2*weights,axis=1)-1),
            connection_a_check=np.sum((phi.conj()*phiq).imag,axis=1)*dx-a,
            connection_b_check=np.sum((phi.conj()*phiR).imag,axis=1)*dx-b,
            conditional_q=np.sum(lam2*weights*q,axis=1))
        for name,value in fields.items():
            if name not in out:out[name]=np.empty((nr,)+value.shape[1:],dtype=value.dtype)
            out[name][sl]=value
    checks=dict(reconstruction_L2=np.sqrt(recon2),backprojection_L2=np.sqrt(back2),
                electronic_PNC=pnc,photon_PNC=lambda_pnc,moments=tuple(moments))
    return out,checks
