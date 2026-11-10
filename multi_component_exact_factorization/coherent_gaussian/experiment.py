"""Reproducible fixed-R K/basis/time tests, never a full MCEF trajectory claim."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import eigh

from ..pen_perspectives_2026.model import AU_FS
from .core import State, overlap, gaussian, tangent_rate, evaluate, hermitian_inverse
from .models import Pen, RealPen, McefLocal


def quadrature(sigma, origin, count):
    x, w = hermgauss(count)
    q = origin+np.sqrt(2)*sigma*x
    # Physical integration weights. Only used in reference/postprocessing.
    w = np.sqrt(2)*sigma*w*np.exp(x*x)
    return q, w


def oscillator_values(q, sigma, origin, nf):
    """Stable normalized Hermite recurrence and derivative, reference only."""
    x = (q-origin)/(np.sqrt(2)*sigma)
    f = np.empty((len(q), nf+1))
    f[:, 0] = (2*np.pi*sigma*sigma)**(-.25)*np.exp(-x*x/2)
    if nf:
        f[:, 1] = np.sqrt(2)*x*f[:, 0]
    for n in range(1, nf):
        f[:, n+1] = np.sqrt(2/(n+1))*x*f[:, n]-np.sqrt(n/(n+1))*f[:, n-1]
    df = np.empty((len(q), nf))
    for n in range(nf):
        df[:, n] = ((np.sqrt(n)*f[:, n-1] if n else 0)-np.sqrt(n+1)*f[:, n+1])/(2*sigma)
    return f[:, :nf], df


def initial_state(model, K, nquad=80):
    if K < 1:
        raise ValueError('K must be positive')
    z = np.zeros(K, complex)
    if K > 1:
        radius = max(.8, .55*np.sqrt(K-1))
        z[1:] = radius*np.exp(2j*np.pi*np.arange(K-1)/(K-1))
    # Independently projected COMMON physical initial state; measure mismatch.
    q, w = quadrature(model.sigma, model.origin, nquad)
    U, _ = model.basis(q)
    target = model.initial_physical(q)
    channel = np.einsum('nxk,nx->nk', U.conj(), target)
    g = gaussian(q, z, model.sigma, model.origin)[0]
    rhs = np.einsum('ni,nk,n->ki', g.conj(), channel, w)
    inv, _ = hermitian_inverse(overlap(z))
    d = rhs@inv.T
    norm = np.einsum('ki,ij,kj', d.conj(), overlap(z), d).real
    # Initial normalization only. No normalization during propagation.
    d /= np.sqrt(norm)
    state = State(z, d)
    G = evaluate(state, q, model.sigma, model.origin, (U, model.basis(q)[1]))[0]
    error = np.sqrt(np.sum(w[:, None]*abs(G-target)**2))
    return state, float(error)


class Reference:
    """Independent uncompressed Hermite reference; not called by Gaussian RHS."""
    def __init__(self, model, nf=32, count=96):
        self.model, self.nf = model, nf
        self.q, self.w = quadrature(model.sigma, model.origin, max(count, 2*nf+8))
        self.f, self.df = oscillator_values(self.q, model.sigma, model.origin, nf)
        n = model.nprimitive
        kinetic = np.einsum('ni,nj,n->ij', self.df, self.df, self.w)/(2*model.mass)
        V = model.potential(self.q)
        H = np.einsum('ni,nkl,nj,n->kilj', self.f, V, self.f, self.w)
        H += np.einsum('kl,ij->kilj', np.eye(n), kinetic)
        self.H = H.reshape(n*nf, n*nf)
        self.E, self.U = eigh(self.H, check_finite=False)
        self.initial = self.project(model.initial_physical(self.q))
        self.initial /= np.linalg.norm(self.initial)
        self.coeff = self.U.conj().T@self.initial

    def project(self, G):
        return np.einsum('ni,nk,n->ki', self.f, G, self.w).ravel()

    def state(self, t, initial=None):
        c = self.coeff if initial is None else self.U.conj().T@initial
        return self.U@(np.exp(-1j*(self.E-self.model.energy_shift)*t)*c)

    def values(self, v):
        return self.f@v.reshape(self.model.nprimitive, self.nf).T


def metrics(model, state, ref, t, matched_initial, *, shape_regularization=1e-7,
            energy_constraint=False):
    G, Gq, lam, C = evaluate(state, ref.q, model.sigma, model.origin, model.basis(ref.q))
    v = ref.project(G)
    vr = ref.state(t)
    vm = ref.state(t, matched_initial)
    norm = float(np.sum(abs(G)**2*ref.w[:, None]))
    err = lambda b: float(np.linalg.norm(v-b))
    phase_err = lambda b: float(np.sqrt(max(0., np.vdot(v,v).real+np.vdot(b,b).real-2*abs(np.vdot(b,v)))))
    H00, H10 = model.matrices(state)
    rate, a = tangent_rate(state, H00, H10, conserve=energy_constraint,
                           shape_regularization=shape_regularization)
    # Density error is not a substitute for phase-sensitive state error.
    density_ref = np.sum(abs(ref.values(vr))**2, axis=1)
    density_l1 = float(np.sum(abs(lam**2-density_ref)*ref.w))
    kinetic = float(np.sum(abs(Gq)**2*ref.w[:, None])/(2*model.mass*norm))
    qq = float(np.sum((ref.q-model.origin)**2*lam**2*ref.w)/norm)
    result = dict(norm=a['norm'], energy=a['energy']+model.energy_shift,
        reference_energy=float(np.vdot(ref.initial, ref.H@ref.initial).real),
        state_l2=err(vr), phase_aligned_l2=phase_err(vr), matched_initial_l2=err(vm),
        density_l1=density_l1, q_mean=float(np.sum(ref.q*lam**2*ref.w)/norm),
        q_kinetic=kinetic, reference_projection_missing_norm=float(a['norm']-np.vdot(v,v).real),
        coefficient_rank=a['coefficient_metric']['rank'], shape_rank=a['shape_metric']['rank'],
        raw_energy_rate=a['raw_energy_rate'], raw_norm_rate=a['raw_norm_rate'],
        constraint_correction_norm=a['constraint_correction_norm'],
        shape_equation_defect=a['shape_equation_defect'])
    if isinstance(model, Pen):
        result['photon_number'] = kinetic/model.c.omega+.5*model.c.omega*qq-.5
        result['reference_photon_number'] = float(np.sum(abs(vr.reshape(model.nprimitive,ref.nf))**2*np.arange(ref.nf)))
        bo = G@model.Ubo.conj()
        br = ref.values(vr)@model.Ubo.conj()
        result['excited_population'] = float(np.sum(abs(bo[:,1])**2*ref.w)/norm)
        result['reference_excited_population'] = float(np.sum(abs(br[:,1])**2*ref.w))
    else:
        good = abs(ref.q-model.origin) <= model.radius
        result['outside_local_fit_mass'] = float(np.sum(lam[~good]**2*ref.w[~good])/norm)
        # Only compare on the pole-free declared local domain, never integrate
        # an unrestricted Gaussian against the singular original potential.
        actual = model.original_potential(ref.q[good])
        fitted = np.diagonal(model.potential(ref.q[good])-model.Tx, axis1=1, axis2=2)
        result['local_potential_rms_Ha'] = float(np.sqrt(np.sum(
            abs(G[good])**2*(actual-fitted)**2*ref.w[good,None])/norm))
    # Gdot is reconstructed from analytic Gaussian tangents, never a q FD.
    g = gaussian(ref.q, state.z, model.sigma, model.origin)[0]
    gd = g*((ref.q[:,None]-model.origin)/model.sigma-state.z[None])*rate.z[None]
    gd -= g*np.real(state.z.conj()*rate.z)[None]
    chdot = g@rate.d.T+gd@state.d.T
    U = model.basis(ref.q)[0]
    Gdot = np.einsum('nkl,nl->nk', U, chdot)
    vd = ref.project(Gdot)
    result['schrodinger_residual'] = float(np.linalg.norm(vd+1j*(ref.H@v-model.energy_shift*v)))
    return result, v


def quadrature_check(model,state,tolerance=1e-5):
    """Independent n vs 2n test; detect inaccurate CBO derivative integrals.

    This is a necessary local check, not a mathematical convergence proof.
    It deliberately refuses underresolved real-electron CBO calculations.
    """
    if isinstance(model,Pen) and model.representation!='cbo':return 0.
    n=model.nquad
    A=model.matrices(state)
    try:
        model.nquad=2*n;B=model.matrices(state)
    finally:model.nquad=n
    defect=max(float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(b))) for a,b in zip(A,B))
    if defect>tolerance:
        raise ValueError(f'Unresolved local quadrature n={n} vs {2*n}: relative matrix defect {defect:g}; increase --nquad, no reliable propagation yet')
    return defect


def run(model, K, out, end_fs=1., every_fs=.1, rtol=1e-7, atol=1e-9,
        max_step=.5, energy_constraint=False, nf_reference=32, max_nfev=200000,
        shape_regularization=1e-7):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists():
        raise FileExistsError('Choose a fresh output directory: '+str(out))
    start = time.perf_counter()
    cfg = dict(model=model.name, representation=model.representation, K=K,
        ne=model.ne, R=model.R, sigma=model.sigma, end_fs=end_fs,
        every_fs=every_fs, rtol=rtol, atol=atol, max_step_au=max_step,
        energy_constraint=energy_constraint, nquad=model.nquad,
        shape_regularization=shape_regularization,
        nf_reference=nf_reference, max_nfev=max_nfev, no_propagated_q_grid=True,
        complex_propagated_variables=K*(model.ne+1),
        physics=asdict(model.c) if isinstance(model,Pen) else dict(nx=model.nprimitive,
                    degree=model.degree, radius=model.radius, surrogate=True),
        method='fixed-R full-complex single-set coherent Gaussian TDVP',
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('core.py').read_bytes()
                    +Path(__file__).with_name('models.py').read_bytes()).hexdigest())
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    try:
        init, initial_error = initial_state(model, K)
        quadrature_check(model,init)
        ref = Reference(model, nf=nf_reference)
        ref_basis = model.basis(ref.q)
        matched = ref.project(evaluate(init, ref.q, model.sigma, model.origin, ref_basis)[0])
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
        summary=dict(status='FAILED_INITIALIZATION',error=repr(exc),last_saved_fs=0.,
                     wall_seconds=time.perf_counter()-start)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'metrics.json').write_text('[]\n')
        print(str(out),json.dumps(summary),flush=True)
        return summary
    setup = time.perf_counter()-start
    calls = 0
    def rhs(t,y):
        nonlocal calls
        calls += 1
        if calls > max_nfev:
            raise RuntimeError('RHS budget reached; no long-run certification')
        s = State.unpack(y, model.ne,K)
        H0,H1 = model.matrices(s)
        rate,a = tangent_rate(s,H0,H1,conserve=energy_constraint,
                              shape_regularization=shape_regularization)
        if not np.all(np.isfinite(rate.packed())) or max(abs(s.z)) > 40:
            raise FloatingPointError('Nonfinite/divergent Gaussian parameters')
        return rate.packed()
    targets = np.unique(np.r_[np.arange(0,end_fs,every_fs),end_fs])/AU_FS
    rows, saved, projections = [], [], []
    from scipy.integrate import DOP853
    failed = None
    ti = 0
    propagation_seconds = 0.
    try:
        integrator = DOP853(rhs,0,init.packed(),end_fs/AU_FS,rtol=rtol,atol=atol,max_step=max_step)
        while ti < len(targets):
            target = targets[ti]
            while integrator.t < target and integrator.status == 'running':
                clock = time.perf_counter()
                integrator.step()
                propagation_seconds += time.perf_counter()-clock
            if integrator.status == 'failed':
                raise RuntimeError('DOP853 failed before target')
            y = integrator.y if target == integrator.t else integrator.dense_output()(target)
            s = State.unpack(y,model.ne,K)
            row,v = metrics(model,s,ref,target,matched,
                            shape_regularization=shape_regularization,
                            energy_constraint=energy_constraint)
            row['t_fs'] = float(target*AU_FS)
            rows.append(row);saved.append(y.copy());projections.append(v)
            if ti%10==0:print(f'K={K}: t={row["t_fs"]:.3f} fs, RHS calls={calls}',flush=True)
            np.savez_compressed(out/'checkpoint.npz',t_au=integrator.t,y=integrator.y,
                K=K,ne=model.ne)
            if row['norm'] < .5 or row['norm'] > 1.5:
                raise FloatingPointError('Unacceptable norm drift')
            if abs(row['reference_projection_missing_norm'])>1e-3:
                raise ValueError('Reference basis/quadrature no longer resolves the state; increase --nf-reference')
            if row.get('outside_local_fit_mass',0)>1e-3:
                raise ValueError('Packet left declared MCEF local surrogate domain; not a full MCEF validation')
            quadrature_check(model,s)
            ti += 1
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
        failed = repr(exc)
    summary = dict(status='FAILED' if failed else 'COMPLETE_NOT_CERTIFIED',error=failed,
        last_saved_fs=rows[-1]['t_fs'] if rows else 0., initial_state_l2=initial_error,
        setup_seconds=setup,propagation_seconds=propagation_seconds,
        wall_seconds=time.perf_counter()-start,nfev=calls,
        warning='Fixed-R conditional benchmark, not full nuclear CTMQC or full TDSE.',
        max_state_l2=max((r['state_l2'] for r in rows),default=None),
        max_energy_drift=max((abs(r['energy']-rows[0]['energy']) for r in rows),default=None),
        max_norm_error=max((abs(r['norm']-1) for r in rows),default=None))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.savez_compressed(out/'states.npz', t_fs=[r['t_fs'] for r in rows],
        y=np.asarray(saved), projected=np.asarray(projections),reference_initial=ref.initial)
    print(str(out),json.dumps(summary),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',choices=['pen-na','pen-rabi','pen-real','mcef-local'],default='pen-na')
    p.add_argument('--basis',default='bo')
    p.add_argument('--components',type=int,default=4)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--end-fs',type=float,default=1.)
    p.add_argument('--every-fs',type=float,default=.1)
    p.add_argument('--g',type=float,default=.01)
    p.add_argument('--dse',action='store_true')
    p.add_argument('--R',type=float)
    p.add_argument('--nx',type=int,default=33)
    p.add_argument('--nstates',type=int)
    p.add_argument('--degree',type=int,default=6)
    p.add_argument('--nquad',type=int,default=32)
    p.add_argument('--nf-reference',type=int,default=32)
    p.add_argument('--energy-constraint',action='store_true')
    p.add_argument('--rtol',type=float,default=1e-7)
    p.add_argument('--atol',type=float,default=1e-9)
    p.add_argument('--max-step',type=float,default=.5)
    p.add_argument('--max-nfev',type=int,default=200000)
    p.add_argument('--shape-regularization',type=float,default=1e-7)
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--movies',action='store_true')
    a=p.parse_args()
    if a.end_fs<=0 or a.every_fs<=0:
        p.error('Positive end/every times required')
    if a.system=='mcef-local':
        m=McefLocal(R=9.5 if a.R is None else a.R,nx=a.nx,degree=a.degree,
            representation=a.basis,nstates=a.nstates,nquad=a.nquad)
    elif a.system=='pen-real':
        m=RealPen(R=-4. if a.R is None else a.R,nx=a.nx,g=a.g,dse=a.dse,
                  representation=a.basis,nquad=a.nquad)
    else:
        m=Pen(model='rabi' if a.system=='pen-rabi' else 'nonadiabatic',g=a.g,dse=a.dse,
            R=a.R,representation=a.basis,nquad=a.nquad)
    result=run(m,a.components,a.out,a.end_fs,a.every_fs,a.rtol,a.atol,a.max_step,
        a.energy_constraint,a.nf_reference,a.max_nfev,a.shape_regularization)
    if not a.no_render:
        from .render import render_fixed
        render_fixed(a.out,a.movies)
    if result['status'].startswith('FAILED'):raise SystemExit(2)


if __name__=='__main__':
    main()
