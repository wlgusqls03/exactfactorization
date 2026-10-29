"""Positive-marginal nested EF of a finite-Fock wavefunction.

Uses the legacy model_a/factorization.py quotient identities, but with analytic
oscillator derivatives and the ACTUAL projected H action. Photon truncation
is a nonlocal q operator. Its local quotient has an imaginary part and adds
S_projection to conditional continuity. Never hide it as numerical error.
Outer EF is an ordinary exact R-marginal equation. No phase unwrapping.
"""
import numpy as np
from .model import AU_FS, cbo, derivative, electronic, oscillator


def divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=np.result_type(a, float)), where=b > 0)


@np.errstate(invalid='ignore', divide='ignore', over='ignore')
def analyze(u, p, t_fs=0., nq=160, qmax=10.):
    """Reconstruct on a DISPLAY q grid; all R derivatives use native coefficients.

    epsilon1 is the real effective scalar including the projection correction;
    the extra nonlocal imaginary source is separately stored, not discarded.
    q quadrature errors in bridge checks are retained and reported.
    """
    c, R, dr, M = p.c, p.R, p.dr, p.c.mass
    q = np.linspace(-qmax, qmax, nq, endpoint=False)
    dq = q[1]-q[0]
    h, dh, ddh = oscillator(c, q)
    expand = lambda z, basis=h: np.einsum('ren,nq->rqe', z, basis)
    ur, urr, urrr = [derivative(u, dr, n) for n in (1, 2, 3)]
    ut = -1j*p.action(u)
    utr = derivative(ut, dr)
    z, zr, zrr, zt = map(expand, (u, ur, urr, ut))
    zq, zqq = expand(u, dh), expand(u, ddh)
    dot = lambda y: np.sum(z.conj()*y, axis=-1)
    inner = lambda y: np.sum(u.conj()*y, axis=(1, 2))
    joint = np.sum(abs(z)**2, axis=-1)
    rho = np.sum(abs(u)**2, axis=(1, 2))
    rt = 2*inner(ut).real
    rr = 2*inner(ur).real
    rrr = 2*(inner(urr).real+np.sum(abs(ur)**2, axis=(1, 2)))
    logR, logt = divide(rr, 2*rho), divide(rt, 2*rho)
    curvature = divide(rrr, 2*rho)-logR**2
    alpha = divide(inner(ur).imag, rho)
    alpha_r = divide(inner(urr).imag, rho)-2*alpha*logR
    alpha_t = divide(np.sum((ut.conj()*ur+u.conj()*utr).imag, axis=(1, 2)), rho)-alpha*divide(rt, rho)
    cond2 = divide(inner(1j*ut+urr/(2*M)), rho).real
    geo2 = (divide(np.sum(abs(ur)**2, axis=(1, 2)), rho)-logR**2-alpha**2)/(2*M)
    gd2 = divide(inner(ut).imag, rho)
    e2 = cond2+geo2+gd2
    inv2 = (curvature-alpha**2)/(2*M)
    rho3 = 2*inner(urrr).real+6*np.sum((ur.conj()*urr).real, axis=(1, 2))
    curvature_r = divide(rho3, 2*rho)-divide(rr*rrr, rho**2)+.5*divide(rr**3, rho**3)
    e2_r = curvature_r/(2*M)-alpha*alpha_r/M
    fq, fr, ft = (divide(dot(v).real, joint) for v in (zq, zr, zt))
    a, b = divide(dot(zq).imag, joint), divide(dot(zr).imag, joint)
    aq = divide(dot(zqq).imag, joint)-2*a*fq
    br = divide(dot(zrr).imag, joint)-2*b*fr
    gq = (divide(np.sum(abs(zq)**2, axis=-1), joint)-fq*fq-a*a)/2
    gr = (divide(np.sum(abs(zr)**2, axis=-1), joint)-fr*fr-b*b)/(2*M)
    gd1 = divide(dot(zt).imag, joint)
    # H_Fock = T_R + T_q + V_CBO + nonlocal truncation correction in q.
    bare, dip = electronic(c, R)
    v = bare[:, None]+c.omega*c.g*q[None, :, None, None]*dip[:, None]
    v += .5*c.omega**2*q[None, :, None, None]**2*np.eye(2)
    if c.dse:
        v += .5*c.g**2*(dip@dip)[:, None]
    vz = np.einsum('rqij,rqj->rqi', v, z)
    correction = 1j*zt+zrr/(2*M)+zqq/2-vz
    projection = divide(dot(correction), joint)
    cond1 = divide(dot(vz), joint).real
    e1_local = cond1+gq+gr+gd1
    e1 = e1_local+projection.real
    lam2 = divide(joint, rho[:, None])
    lam = np.sqrt(lam2)
    lr, lq = lam*(fr-logR[:, None]), lam*fq
    density_dt = 2*lam2*(ft-logt[:, None])
    sq = -lam2*(aq+2*a*fq)
    sadv = -2*alpha[:, None]*lam*lr/M
    srel = -lam2*(2*fr*(b-alpha[:, None])+br-alpha_r[:, None])/M
    # Derived from the independently applied nonlocal H correction, not fitted
    # as 'target minus sum'. Its conditional q integral vanishes in continuum.
    sproj = 2*lam2*projection.imag
    integrate = lambda z: np.sum(np.where(joint > 0, z, 0), axis=1)*dq
    mean_e1 = integrate(lam2*e1)
    kq = integrate((lq*lq+a*a*lam2)/2)
    gl = integrate((lr*lr+(b-alpha[:, None])**2*lam2)/(2*M))
    ce, cv = np.linalg.eigh(v)
    cbo_density = abs(np.einsum('rqek,rqe->rqk', cv, z))**2
    # Independent spectral reconstructions, not aliases of the expectations.
    wcbo = divide(np.sum(cbo_density*ce, axis=-1), joint)
    pol_coeff = np.einsum('rik,ri->rk', p.pol_vectors, u.reshape(c.nr, -1))
    wpol = divide(np.sum(abs(pol_coeff)**2*p.pol_energy, axis=-1), rho)
    # Resolve the weighted polaritonic energy without double counting it.
    # All terms are computed in native Fock space (no q quadrature error).
    native_expect = lambda y: divide(inner(y).real, rho)
    bo2 = native_expect(np.einsum('rij,rjn->rin', bare, u))
    pk2 = native_expect(np.einsum('nm,rem->ren', p.ops['p2']/2, u))
    pv2 = native_expect(np.einsum('nm,rem->ren', c.omega**2*p.ops['q2']/2, u))
    lm2 = native_expect(c.omega*c.g*np.einsum('rij,nm,rjm->rin', dip, p.ops['q'], u))
    dse2 = (native_expect(.5*c.g**2*np.einsum('rij,rjn->rin', dip@dip, u))
            if c.dse else np.zeros_like(rho))
    f = dict(R=R, q=q, omega=c.omega, g_cavity=c.g, mass=M, t_fs=t_fs,
             system='pen_2026_fock', gauge='positive_marginals', dse=c.dse,
             scalar_kind='finite-Fock effective scalar; nonlocal source retained',
             rho_qR=joint, rho_R=rho, lambda_density=lam2,
             a=a, b=b, alpha=alpha, alpha_t=alpha_t, delta_b=b-alpha[:, None],
             epsilon1=e1, epsilon1_local=e1_local, epsilon1_projection=projection.real,
             epsilon1_GI=cond1+gq+gr+projection.real, epsilon1_GD=gd1,
             epsilon1_cond=cond1, epsilon1_qgeo=gq, epsilon1_Rgeo=gr,
             epsilon1_wCBO=wcbo,
             epsilon2=e2, epsilon2_GI=cond2+geo2, epsilon2_GD=gd2,
             epsilon2_cond=cond2, epsilon2_geo=geo2,
             epsilon2_wPol=wpol, epsilon2_wBO=bo2, epsilon2_pkin=pk2,
             epsilon2_ppot=pv2, epsilon2_LM=lm2, epsilon2_DSE=dse2,
             epsilon1_GI_residual=cond1+gq+gr+projection.real-(wcbo+gq+gr+projection.real),
             epsilon2_GI_residual=cond2+geo2-(wpol+geo2),
             epsilon2_wPol_residual=wpol-bo2-pk2-pv2-lm2-dse2,
             epsilon2_inversion=inv2, force=-e2_r+alpha_t,
             alpha_reconstructed=integrate(lam2*b), mean_e1=mean_e1,
             Kq=kq, G_R_lambda=gl, Ct=np.zeros_like(rho),
             epsilon2_reconstructed=mean_e1+kq+gl,
             density_dt=density_dt, S_q=sq, S_adv=sadv, S_rel=srel, S_projection=sproj,
             continuity_residual=density_dt-sq-sadv-srel-sproj,
             cbo_population=np.sum(cbo_density, axis=(0, 1))*dr*dq,
             q_integral_error=float(np.sum(joint)*dr*dq-np.sum(rho)*dr))
    return f


def diagnostics(f):
    """Weighted errors on occupied support, not a convergence certificate."""
    def rms(value, density):
        mask = (density > density.max()*1e-5) & np.isfinite(value)
        if not mask.any():
            return None
        return float(np.sqrt(np.sum(density[mask]*abs(value[mask])**2)/density[mask].sum()))
    return dict(alpha_bridge_rms=rms(f['alpha']-f['alpha_reconstructed'], f['rho_R']),
                epsilon2_bridge_rms=rms(f['epsilon2']-f['epsilon2_reconstructed'], f['rho_R']),
                epsilon2_inversion_rms=rms(f['epsilon2']-f['epsilon2_inversion'], f['rho_R']),
                continuity_rms=rms(f['continuity_residual'], f['rho_qR']),
                projection_source_rms=rms(f['S_projection'], f['rho_qR']),
                epsilon1_GI_rms=rms(f['epsilon1_GI_residual'], f['rho_qR']),
                epsilon2_GI_rms=rms(f['epsilon2_GI_residual'], f['rho_R']),
                weighted_pol_budget_rms=rms(f['epsilon2_wPol_residual'], f['rho_R']),
                q_integral_error=f['q_integral_error'])
