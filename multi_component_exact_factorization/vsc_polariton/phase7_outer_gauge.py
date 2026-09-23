"""Connected-support outer alpha=0 construction and independent FD checks.

Only the outer gauge is changed. a and b are NOT set to zero. Each snapshot
uses a frozen left anchor per component, with independent time-only constants.
No phase interpolation across unoccupied gaps.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from .phase7_support import budget_support,weighted_rms


def derivative_interior(value,spacing,order):
    coefficients={4:np.array([1,-8,0,8,-1])/12,
                  6:np.array([-1,9,-45,0,45,-9,1])/60}[order]
    half=len(coefficients)//2
    result=np.zeros_like(value[half:-half])
    for j,c in enumerate(coefficients):result+=c*value[j:len(value)-2*half+j]
    return result/spacing


def outer_gauge(u,packet,outer,budget):
    R=packet['R'];dr=float(packet['dR']);dx=float(packet['dx']);M=float(packet['mass'])
    mask,meta,labels=budget_support(outer['rho_R'],dr,budget)
    fields={k:np.full(len(R),np.nan) for k in
            ('eta','eta_t','epsilon2_A0','alpha_A0_fd4','alpha_A0_fd6',
             'current_A0_fd4','current_A0_fd6','force_A0_fd4','force_A0_fd6')}
    reconstruction=0.
    for component in range(1,meta['components']+1):
        ids=np.flatnonzero(labels==component)
        if len(ids)<7:continue
        r=R[ids];a=outer['alpha'][ids];at=outer['alpha_t'][ids]
        anti=CubicSpline(r,a).antiderivative();anti_t=CubicSpline(r,at).antiderivative()
        eta=-(anti(r)-anti(r[0]));eta_t=-(anti_t(r)-anti_t(r[0]))
        chi=outer['chi'][ids];rho=outer['rho_R'][ids]
        gamma=u[ids]/chi[:,None,None]
        transformed=gamma*np.exp(1j*eta)[:,None,None]
        chig=chi*np.exp(-1j*eta)
        reconstruction+=float(np.sum(abs(transformed*chig[:,None,None]-u[ids])**2)*dx*dr)
        e=outer['epsilon2_B'][ids].real+eta_t
        fields['eta'][ids]=eta;fields['eta_t'][ids]=eta_t;fields['epsilon2_A0'][ids]=e
        for order in (4,6):
            half=order//2;interior=ids[half:-half]
            dg=derivative_interior(transformed,dr,order)
            ag=np.sum((transformed[half:-half].conj()*dg).imag,axis=(1,2))*dx
            dchi=derivative_interior(chig,dr,order)
            current=((chig[half:-half].conj()*dchi).imag+rho[half:-half]*ag)/M
            fields['alpha_A0_fd'+str(order)][interior]=ag
            fields['current_A0_fd'+str(order)][interior]=current
            fields['force_A0_fd'+str(order)][interior]=-derivative_interior(e,dr,order)
    valid=np.isfinite(fields['alpha_A0_fd6'])
    mass=outer['rho_R']*dr
    alpha_scale=max(1.,weighted_rms(outer['alpha'],mass,valid) or 0.)
    alpha_error=weighted_rms(fields['alpha_A0_fd6'],mass,valid)
    force_error=weighted_rms(fields['force_A0_fd6']-outer['force'],mass,valid)
    current_momentum_error=weighted_rms(
        (fields['current_A0_fd6']-outer['rho_R']*outer['alpha']/M)*M/outer['rho_R'],mass,valid)
    report=dict(support=meta,reconstruction_L2=np.sqrt(reconstruction),
        alpha_fd6_RMS=alpha_error,alpha_scale=alpha_scale,
        force_fd6_RMS=force_error,current_momentum_RMS=current_momentum_error,
        force_fd4_vs_fd6_RMS=weighted_rms(fields['force_A0_fd4']-fields['force_A0_fd6'],mass,valid),
        excluded_for_stencil=float(mass[~valid].sum()/mass.sum()),
        gauge_pass=bool(alpha_error is not None and alpha_error/alpha_scale<1e-3 and
                        current_momentum_error/alpha_scale<1e-3),
        force_derivative_pass=bool(force_error is not None and force_error<1e-5),
        scope='Snapshot component-local gauge; no inter-frame phase tracking')
    return fields,report
