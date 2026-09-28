"""Memory-bounded exact Fourier-wave postprocessing (R,x,n), atomic units.

Caches bilinears on 2N: |u|², Im<u|u_R>, d_t|u|², d_t Im<u|u_R>.
Nonlinear divisions and gauge integration are evaluated only on support.
"""
import numpy as np
from scipy.fft import fft,ifft
from scipy.interpolate import CubicSpline
from .phase7_support import budget_support,weighted_rms
from .phase7_outer_gauge import derivative_interior


def evaluate(coeff,n,order=0,length=1.,shift=0.):
    """FFT coefficients to n>=N samples; preserve negative Nyquist exactly."""
    old=len(coeff)
    if n<old:raise ValueError('No spectral truncation allowed')
    modes=np.rint(np.fft.fftfreq(old)*old).astype(int)
    k=2*np.pi*modes/length
    weight=(1j*k)**order*np.exp(1j*k*shift)
    result=np.zeros((n,)+coeff.shape[1:],dtype=complex)
    result[modes % n]=coeff*weight.reshape((old,)+(1,)*(coeff.ndim-1))*n/old
    return ifft(result,axis=0)


def bilinears(u,ut,dx,dr,block=128):
    """Unaliased contractions, same interpolated u and instantaneous ut."""
    n=len(u);length=n*dr;out=np.zeros((2*n,4))
    flat=u.reshape(n,-1);ft=ut.reshape(n,-1)
    for start in range(0,flat.shape[1],block):
        c=fft(flat[:,start:start+block],axis=0);ct=fft(ft[:,start:start+block],axis=0)
        v=evaluate(c,2*n);vr=evaluate(c,2*n,1,length);vt=evaluate(ct,2*n)
        vtr=evaluate(ct,2*n,1,length)
        out[:,0]+=np.sum(abs(v)**2,axis=1)*dx
        out[:,1]+=np.sum((v.conj()*vr).imag,axis=1)*dx
        out[:,2]+=2*np.sum((v.conj()*vt).real,axis=1)*dx
        out[:,3]+=np.sum((vt.conj()*vr+v.conj()*vtr).imag,axis=1)*dx
    return fft(out,axis=0)


def shifted_overlap(u,dx,dr,shift,block=128):
    """K_s(R)=integral u*(R)u(R+s) dx dn; Fourier cache on 2N."""
    n=len(u);flat=u.reshape(n,-1);out=np.zeros(2*n,complex)
    for start in range(0,flat.shape[1],block):
        c=fft(flat[:,start:start+block],axis=0)
        v=evaluate(c,2*n);vs=evaluate(c,2*n,length=n*dr,shift=shift)
        out+=np.sum(v.conj()*vs,axis=1)*dx
    return fft(out)


def probe(cache,R,mass,factor,budget,links=None):
    n=len(R)*factor;length=len(R)*(R[1]-R[0]);h=length/n
    r=R[0]+np.arange(n)*h
    raw=evaluate(cache,n,length=length).real
    d=evaluate(cache,n,1,length).real;dd=evaluate(cache,n,2,length).real
    ddd=evaluate(cache,n,3,length).real
    rho,m,rhot,mt=raw.T
    negative=float(-np.minimum(rho,0).sum()*h)
    if negative>1e-13:raise ValueError('Non-roundoff negative density')
    mask,meta,labels=budget_support(np.maximum(rho,0),h,budget)
    with np.errstate(divide='ignore',invalid='ignore'):
        a=m/rho;ar=d[:,1]/rho-a*d[:,0]/rho;at=mt/rho-a*rhot/rho
        r1=d[:,0]/rho;r2=dd[:,0]/rho;r3=ddd[:,0]/rho
        eps=(r2/2-r1*r1/4-a*a)/(2*mass)
        force=-(r3/2-r1*r2+r1**3/2-2*a*ar)/(2*mass)+at
        imag=rhot/(2*rho)+(ar+a*r1)/(2*mass)
    eta=np.full(n,np.nan);etat=eta.copy();eg=eta.copy()
    force4=eta.copy();force6=eta.copy();a6=eta.copy();j6=eta.copy()
    for comp in range(1,meta['components']+1):
        ids=np.flatnonzero(labels==comp)
        if len(ids)<7:continue
        xx=r[ids]
        for value,target in ((a,eta),(at,etat)):
            anti=CubicSpline(xx,value[ids]).antiderivative()
            target[ids]=-(anti(xx)-anti(xx[0]))
        eg[ids]=eps[ids]+etat[ids]
        for order,target in ((4,force4),(6,force6)):
            k=order//2;target[ids[k:-k]]=-derivative_interior(eg[ids],h,order)
        interior=ids[3:-3]
        if links is not None:
            total=np.zeros(len(interior),complex)
            for s,c in zip((-3,-2,-1,1,2,3),(-1,9,-45,45,-9,1)):
                kernel=links[s][interior]
                total+=c*kernel*np.exp(1j*(eta[interior+s]-eta[interior]))/np.sqrt(rho[interior]*rho[interior+s])
            a6[interior]=total.imag/(60*h)
            chi=np.sqrt(rho[ids])*np.exp(-1j*eta[ids])
            j6[interior]=((chi[3:-3].conj()*derivative_interior(chi,h,6)).imag+rho[interior]*a6[interior])/mass
    valid=np.isfinite(force6)&mask;weight=np.maximum(rho,0)*h
    rms=lambda v:weighted_rms(v,weight,valid)
    force_error=rms(force6-force);scale=max(1.,rms(a) or 0.)
    report=dict(factor=factor,budget=budget,support=meta,negative_roundoff_mass=negative,
        excluded_stencil_mass=float(weight[~valid].sum()/weight.sum()),
        force_RMS=rms(force),force_fd6_error=force_error,force_fd4_fd6=rms(force4-force6),
        scalar_imag_RMS=rms(imag),
        force_pass=force_error is not None and force_error<1e-5,
        gauge_checked=links is not None)
    if links is not None:
        ae=rms(a6);je=rms((j6-m/mass)*mass/np.where(rho>0,rho,np.nan))
        report.update(alpha_fd6_RMS=ae,current_momentum_RMS=je,alpha_scale=scale,
                      gauge_pass=ae is not None and je is not None and ae/scale<1e-3 and je/scale<1e-3)
    return dict(R=r,rho=rho,alpha=a,alpha_t=at,epsilon2=eps,force=force,
                eta=eta,eta_t=etat,epsilon2_A0=eg,force_A0_fd6=force6,alpha_A0_fd6=a6,mask=mask),report
