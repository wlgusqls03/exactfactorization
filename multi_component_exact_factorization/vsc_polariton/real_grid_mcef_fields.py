"""Read-only, instantaneous nested-EF diagnostics for saved full real-grid TDSE.

Input Psi_Q(R,x,Q), Q=sqrt(omega)*q; output physical q quantities in a.u.
Positive chi and Lambda gauge. Photon kinetic mass is ONE, not 1/omega.
The FFT derivatives are those of photon_real_grid.RealGridPF. H Psi provides
the time derivative; no interpolation between sparse saved frames is used.
Formula provenance: PHASE7_NESTED_EF_DERIVATION, with quadrature dq in place
of the Fock contraction. Continuum quotient derivatives do not certify
finite-grid product rules. Route agreement is NOT spatial convergence.
Only small (R,q) fields are saved, never the full conditional electronic wave.
"""
import numpy as np
from scipy.fft import fft, ifft
from .phase7_support import budget_support, weighted_rms


def derivative(u, spacing, axis, order=1):
    """Native periodic spectral derivative, same kinetic symbol as propagation."""
    shape = [1] * u.ndim
    shape[axis] = u.shape[axis]
    k = 2*np.pi*np.fft.fftfreq(u.shape[axis], spacing)
    f = fft(u, axis=axis)
    f *= ((1j*k)**order).reshape(shape)
    return ifft(f, axis=axis, overwrite_x=True)


def divide(a, b):
    """No density floor: undefined quotients remain NaN at exact zeros."""
    out = np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan,
                  dtype=np.result_type(a, b, float))
    return np.divide(a, b, out=out, where=np.broadcast_to(b, out.shape)>0)


def action(u, packet, q, block=4):
    """Full PF action on physical Psi_q(R,x,q), Ha*wave; full (R-x)^2 DSE."""
    dr, dx, M = (float(packet[k]) for k in ('dR', 'dx', 'mass'))
    w, g = (float(packet[k]) for k in ('omega', 'g_chi'))
    out = -derivative(u, dr, 0, 2)/(2*M)
    for first in range(0, len(u), block):
        s = slice(first, first+block)
        mu = packet['R'][s,None]-packet['x'][None,:]
        v = packet['potential'][s,:,None]+.5*w*w*(q+np.sqrt(2/w**3)*g*mu[:,:,None])**2
        out[s] += -.5*derivative(u[s], dx, 1, 2)
        out[s] += -.5*derivative(u[s], q[1]-q[0], 2, 2)+v*u[s]
    return out


def analyze(psi_Q, packet, Q, block=4):
    """Return natural-gauge fields (R)/(R,q) and instantaneous residual arrays.

    Peak storage is several full waves on HOST RAM; no GPU needed. phi0..2
    are diagnostics only: propagation is not projected onto three BO states.
    eps1/eps2 A: independent expectation with -i<conditional|dt conditional>.
    B: marginal-equation inversion with analytic quotient derivatives.
    """
    w, M, dr, dx = (float(packet[k]) for k in ('omega','mass','dR','dx'))
    q = Q/np.sqrt(w)
    dq = float(q[1]-q[0])
    u = psi_Q*w**.25
    ur = derivative(u, dr, 0)
    urr = derivative(u, dr, 0, 2)
    ut = -1j*action(u, packet, q, block)
    inner = lambda v: np.sum(u.conj()*v, axis=(1,2))*dx*dq
    rho = np.sum(abs(u)**2, axis=(1,2))*dx*dq
    r1 = 2*inner(ur).real
    r2 = 2*(inner(urr).real+np.sum(abs(ur)**2,axis=(1,2))*dx*dq)
    rt = 2*inner(ut).real
    alpha = divide(inner(ur).imag, rho)
    ar = divide(inner(urr).imag, rho)-alpha*divide(r1,rho)
    logR = divide(r1,2*rho)
    curv = divide(r2,2*rho)-logR**2
    geo = (divide(np.sum(abs(ur)**2,axis=(1,2))*dx*dq,rho)-logR**2-alpha**2)/(2*M)
    cond = divide(1j*inner(ut)+inner(urr)/(2*M),rho)
    gd = -1j*(divide(inner(ut),rho)-divide(rt,2*rho))
    e2A = cond+geo+gd
    e2B = (curv-alpha**2)/(2*M)+1j*(divide(rt,2*rho)+(ar+2*alpha*logR)/(2*M))
    utr = derivative(ut,dr,0)
    at = divide(np.sum((ut.conj()*ur+u.conj()*utr).imag,axis=(1,2))*dx*dq,rho)-alpha*divide(rt,rho)
    del utr
    urrr = derivative(u,dr,0,3)
    r3 = 2*inner(urrr).real+6*np.sum((ur.conj()*urr).real,axis=(1,2))*dx*dq
    del urrr
    cr = divide(r3,2*rho)-divide(r1*r2,rho**2)+.5*divide(r1**3,rho**3)
    e2r = cr/(2*M)-alpha*ar/M
    out = dict(R=packet['R'],q=q,Q=Q,rho_R=rho,current=rho*alpha/M,
               alpha=alpha,alpha_R=ar,alpha_t=at,rho_R_t=rt,
               epsilon2_A=e2A,epsilon2_B=e2B,epsilon2_cond=cond,
               epsilon2_geo=geo,epsilon2_GD=gd,epsilon2_R=e2r,
               force=-e2r+at)
    reconstruction = 0.
    for first in range(0,len(u),block):
        s = slice(first,first+block)
        z=u[s];zr=ur[s];zrr=urr[s];zt=ut[s]
        zq=derivative(z,dq,2);zqq=derivative(z,dq,2,2)
        ix=lambda v: np.sum(z.conj()*v,axis=1)*dx
        joint=np.sum(abs(z)**2,axis=1)*dx
        F=np.sqrt(joint)
        fq=divide(ix(zq).real,joint);fr=divide(ix(zr).real,joint)
        ft=divide(ix(zt).real,joint)
        fqq=divide(ix(zqq).real+np.sum(abs(zq)**2,axis=1)*dx,joint)-fq**2
        frr=divide(ix(zrr).real+np.sum(abs(zr)**2,axis=1)*dx,joint)-fr**2
        phi=divide(z,F[:,None,:])
        phiq=divide(zq-z*fq[:,None,:],F[:,None,:])
        phir=divide(zr-z*fr[:,None,:],F[:,None,:])
        phit=divide(zt-z*ft[:,None,:],F[:,None,:])
        a=divide(ix(zq).imag,joint);b=divide(ix(zr).imag,joint)
        aq=divide(ix(zqq).imag,joint)-2*a*fq
        br=divide(ix(zrr).imag,joint)-2*b*fr
        gq=(np.sum(abs(phiq)**2,axis=1)*dx-a*a)/2
        gr=(np.sum(abs(phir)**2,axis=1)*dx-b*b)/(2*M)
        gd1=-1j*np.sum(phi.conj()*phit,axis=1)*dx
        bare=-.5*derivative(z,dx,1,2)+packet['potential'][s,:,None]*z
        mu=packet['R'][s,None]-packet['x'][None,:]
        lm=np.sqrt(2*w)*float(packet['g_chi'])*mu[:,:,None]*q
        dse=float(packet['g_chi'])**2*mu**2/w
        harmonic=.5*w*w*q*q
        hz=bare+(lm+dse[:,:,None]+harmonic)*z
        c1=divide(ix(hz),joint)
        e1A=c1+gq+gr+gd1
        e1B=.5*(fqq-a*a)+(frr-b*b)/(2*M)+1j*(ft+a*fq+aq/2+(b*fr+br/2)/M)
        chi=np.sqrt(rho[s,None]);lam=divide(F,chi)
        lamR=lam*(fr-logR[s,None]);lamq=lam*fq
        lamt=lam*(ft-divide(rt[s],2*rho[s])[:,None])
        e2nested=np.sum((.5*(lamq**2+a*a*lam**2)+lam**2*e1A+
                        (lamR**2+(b-alpha[s,None])**2*lam**2)/(2*M)-1j*lam*lamt),axis=1)*dq
        rebuilt=phi*lam[:,None,:]*chi[:,None,:]
        # Only exact zero-density cells may be omitted, never finite low tails.
        occupied=np.broadcast_to(joint[:,None,:]>0,z.shape)
        reconstruction+=float(np.sum(abs(rebuilt[occupied]-z[occupied])**2)*dx*dq*dr)
        bo=np.einsum('rxj,rxq->rjq',packet['phi'][s].conj(),z)*dx
        bod=np.moveaxis(abs(bo)**2,1,2)
        fields=dict(rho_qR=joint,lambda_density=lam**2,a=a,b=b,
            berry=2*divide(np.sum((zq.conj()*zr).imag,axis=1)*dx,joint)-2*b*fq+2*a*fr,
            epsilon1_A=e1A,epsilon1_B=e1B,epsilon1_cond=c1,
            epsilon1_qgeo=gq,epsilon1_Rgeo=gr,epsilon1_GD=gd1,
            epsilon1_bare=divide(ix(bare),joint),epsilon1_LM=divide(ix(lm*z),joint),
            epsilon1_DSE=divide(ix(dse[:,:,None]*z),joint),
            epsilon2_nested=e2nested,alpha_nested=np.sum(lam**2*b,axis=1)*dq,
            BO_character=divide(bod,joint[:,:,None]),BO_density=bod,
            electronic_PNC_error=abs(np.sum(abs(phi)**2,axis=1)*dx-1),
            photon_PNC_error=abs(np.sum(lam**2,axis=1)*dq-1),
            conditional_q=np.sum(lam**2*q,axis=1)*dq,
            full_EOM_residual=np.sqrt(np.sum(abs(1j*zt+.5*zqq+zrr/(2*M)-hz)**2,axis=1)*dx))
        for k,v in fields.items():
            if k not in out:out[k]=np.empty((len(u),)+v.shape[1:],dtype=v.dtype)
            out[k][s]=v
    out['reconstruction_L2']=np.sqrt(reconstruction)
    return out


def diagnostics(fields, budgets=(1e-6,1e-8,1e-10)):
    """Probability-weighted errors; no Phase7 PASS inferred from identities."""
    f=fields;dr=float(f['R'][1]-f['R'][0]);dq=float(f['q'][1]-f['q'][0])
    report=dict(phase7_pass=False,scope='Instantaneous natural-gauge diagnostics, NOT full field convergence',
                norm=float(f['rho_R'].sum()*dr),reconstruction_L2=float(f['reconstruction_L2']),budgets={})
    for budget in budgets:
        mr,ir,_=budget_support(f['rho_R'],dr,budget)
        mj,ij,_=budget_support(f['rho_qR'],dr*dq,budget)
        vals={}
        for k,v in dict(epsilon1_route=f['epsilon1_A']-f['epsilon1_B'],
                       epsilon1_imag=f['epsilon1_A'].imag,
                       electronic_PNC=f['electronic_PNC_error']).items():
            vals[k]=weighted_rms(v,f['rho_qR']*dr*dq,mj)
        for k,v in dict(epsilon2_route=f['epsilon2_A']-f['epsilon2_B'],
                       epsilon2_nested=f['epsilon2_A']-f['epsilon2_nested'],
                       epsilon2_imag=f['epsilon2_A'].imag,
                       photon_PNC=f['photon_PNC_error'],
                       alpha_nested=f['alpha']-f['alpha_nested']).items():
            vals[k]=weighted_rms(v,f['rho_R']*dr,mr)
        report['budgets'][str(budget)]=dict(outer_support=ir,joint_support=ij,weighted_errors=vals)
    return report
