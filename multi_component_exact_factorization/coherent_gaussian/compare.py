"""Moving-nucleus PEN comparison: QD, native EL/POL CTMQC, Gaussian-POL.

Gaussian-POL projects the EXISTING native POL CT source onto full-complex
Gaussian tangents and reuses its force/history equations. It is experimental,
not unmodified CTMQC, not exact EF-G/MLS, and not claimed cheaper than POL.
Temporary Fock projection for the CT closure is explicit and tail-guarded.
No q grid is propagated, stored, or differentiated.
"""
import argparse
from dataclasses import replace, asdict
import json
import hashlib
from pathlib import Path
import time
import subprocess
import numpy as np
from scipy.integrate import DOP853
from ..pen_perspectives_2026.model import (
    AU_FS, electronic, Propagator, initial, observables as qd_observables)
from ..pen_perspectives_2026.resonance import paper_config
from ..pen_ctmqc_2026.core import Engine, Settings, observables, hamiltonian, align_bands
from .core import State, fock_coefficients, overlap, tangent_rate
from .models import Pen
from .experiment import initial_state


class GaussianPOL:
    def __init__(self, config, settings, K, regularization=1e-4, tail_tolerance=1e-7):
        if settings.perspective!='pol':
            raise ValueError('Gaussian quantum photon uses POL source, never EL source')
        self.engine=Engine(config,settings)
        self.c,self.s,self.K=config,settings,K
        self.reg,self.tail_tolerance=regularization,tail_tolerance
        self.n= settings.ntraj
        self.width=2+3*K+2*config.nf
        self.max_tail=0.;self.max_norm_rate=0.;self.calls=0
        self.anchor=None
        self.history_resets=0

    def model(self,R):
        model='rabi' if self.c.r1==self.c.r2 else 'nonadiabatic'
        m=Pen(model=model,g=self.c.g,dse=self.c.dse,R=R,
              representation='diabatic',omega=self.c.omega)
        # Keep the same material Schrodinger phase on ALL trajectories.
        # An R-dependent rotating phase without its R derivative is invalid
        # for later MLS connection diagnostics.
        m.energy_shift=0.
        return m

    def initial(self):
        native=self.engine.initial()
        self.anchor=native['anchor'].copy()
        y=np.zeros((self.n,self.width),complex)
        y[:,:2]=np.c_[native['X'][:,0],native['P'][:,0]]
        for i,R in enumerate(y[:,0].real):
            s,_=initial_state(self.model(R),self.K)
            y[i,2:2+3*self.K]=s.packed()
        return y.ravel()

    def native_state(self,y):
        y=y.reshape(self.n,self.width)
        coeff=np.empty((self.n,2*self.c.nf),complex)
        for i in range(self.n):
            s=State.unpack(y[i,2:2+3*self.K],2,self.K)
            f=fock_coefficients(s.z,self.c.nf)
            coeff[i]=(s.d@f.T).ravel()
            norm=np.einsum('ki,ij,kj',s.d.conj(),overlap(s.z),s.d).real
            tail=abs(norm-np.vdot(coeff[i],coeff[i]).real)
            self.max_tail=max(self.max_tail,float(tail))
            if tail>self.tail_tolerance:
                raise ValueError('Gaussian lies outside CT Fock subspace: increase --nf; tail='+str(tail))
        return dict(X=y[:,0:1].real,P=y[:,1:2].real,coeff=coeff,
            integrated_forces=y[:,2+3*self.K:].real[:,:,None])

    def rhs(self,t,y):
        self.calls+=1
        if self.calls>300000:
            raise RuntimeError('Gaussian ensemble RHS budget exceeded')
        state=self.native_state(y)
        deriv,info=self.engine.evaluate(state,self.anchor)
        rows=y.reshape(self.n,self.width)
        out=np.zeros_like(rows)
        out[:,0]=deriv['X'][:,0];out[:,1]=deriv['P'][:,0]
        out[:,2+3*self.K:]=deriv['integrated_forces'][:,:,0]
        # Reuse the CT eigenvectors/rates already evaluated, rather than
        # assembling V a second time and subtracting two large complex RHSs.
        ad=np.einsum('nik,ni->nk',info['vectors'],state['coeff'])
        Fct=np.einsum('nik,nk->ni',info['vectors'],info['rates']*ad)
        for i in range(self.n):
            s=State.unpack(rows[i,2:2+3*self.K],2,self.K)
            m=self.model(state['X'][i,0])
            f=fock_coefficients(s.z,self.c.nf)
            T=np.zeros_like(f);T[1:]=np.sqrt(np.arange(1,self.c.nf))[:,None]*f[:-1]
            fct=Fct[i].reshape(2,self.c.nf)
            bc=fct@f.conj()
            bs=np.sum(s.d.conj()*(fct@T.conj()),axis=0)
            rate,a=tangent_rate(s,*m.matrices(s),extra_rate=(bc,bs),shape_regularization=self.reg)
            out[i,2:2+3*self.K]=rate.packed()
            self.max_norm_rate=max(self.max_norm_rate,abs(a['norm_rate']))
        return out.ravel()

    def accept(self, integrator):
        """Track channels/reset inactive histories after an accepted step.

        Same threshold and projectors as Engine.step; timing follows ADAPTIVE
        accepted steps, not the native fixed RK4 lattice. This distinction is
        recorded and requires step-size convergence, not a claim of identity.
        No coefficient renormalization or energy correction is performed.
        """
        state=self.native_state(integrator.y)
        E,U=np.linalg.eigh(hamiltonian(self.c,'pol',state['X'])[0])
        _,self.anchor=align_bands(E,U,self.anchor)
        pop=abs(np.einsum('nik,ni->nk',self.anchor,state['coeff']))**2
        pop/=pop.sum(axis=1)[:,None]
        active=(pop>self.s.population_threshold)&(pop<1-self.s.population_threshold)
        rows=integrator.y.reshape(self.n,self.width)
        hist=rows[:,2+3*self.K:]
        changed=np.any(hist[~active]!=0)
        self.history_resets+=int(np.count_nonzero(hist[~active]))
        hist*=active
        if changed:
            integrator.f=self.rhs(integrator.t,integrator.y)

    def observable(self,y):
        s=self.native_state(y)
        s['anchor']=self.anchor
        return observables(self.engine,s)


def save(root,name,rows,summary,**arrays):
    out=root/name;out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if arrays:np.savez_compressed(out/'states.npz',**arrays)
    print(name,json.dumps(summary),flush=True)


def run_comparison(out,*,model='nonadiabatic',resonance='on',end_fs=1.,every_fs=.25,
                   ntraj=16,seed=2026,nf=8,nr=128,dt=.05,K=4,g=.01,dse=False,
                   regularization=1e-4,methods=('qd','el','pol','gaussian'),algorithm='ctmqc',
                   gaussian_max_step=.5,tail_tolerance=1e-6):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():raise FileExistsError('Use fresh comparison directory')
    c=paper_config(model,resonance,nf=nf,nr=nr,dt=dt,g=g,dse=dse)
    config=dict(physics=asdict(c),ntraj=ntraj,seed=seed,K=K,end_fs=end_fs,every_fs=every_fs,
        algorithm=algorithm,methods=list(methods),initial='bare S1 x vacuum (EL vacuum Wigner)',
        regularization=regularization,gaussian_max_step=gaussian_max_step,
        tail_tolerance=tail_tolerance,
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest(),
        limitations=['Finite-Fock CT closure/reference requires nf convergence.',
            'Small Wigner ensemble is not sampling-converged.',
            'Gaussian CT projection modifies native CTMQC; energy is NOT constrained.',
            'QD periodic R box differs from open trajectory dynamics; inspect edges.',
            'Native RK4 and Gaussian DOP853 have different step/error controls.',
            'Inactive-history resets follow accepted adaptive steps for Gaussian, fixed steps for native.',
            'Native means repository Python CTMQC, not authors Fortran G-CTMQC.'])
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    targets=np.unique(np.r_[np.arange(0,end_fs,every_fs),end_fs])
    summaries={}
    for name in methods:
        start=time.perf_counter();rows=[];ys=[];error=None;maxraw=0.
        try:
            if name=='gaussian':
                engine=GaussianPOL(c,Settings(perspective='pol',ntraj=ntraj,seed=seed,
                    initial='bare',algorithm=algorithm),K,regularization,tail_tolerance)
                integ=DOP853(engine.rhs,0,engine.initial(),end_fs/AU_FS,
                    rtol=2e-6,atol=2e-8,max_step=gaussian_max_step)
                for target in targets/AU_FS:
                    while integ.t<target and integ.status=='running':
                        integ.step()
                        engine.accept(integ)
                    if integ.status=='failed':raise RuntimeError('Gaussian adaptive integration failed')
                    y=integ.y if target==integ.t else integ.dense_output()(target)
                    row=engine.observable(y);row['t_fs']=float(target*AU_FS)
                    rows.append(row);ys.append(y.copy())
                    print(f'{name}: saved t={row["t_fs"]:.3f} fs, RHS calls={engine.calls}',flush=True)
            elif name in ('el','pol'):
                engine=Engine(c,Settings(perspective=name,ntraj=ntraj,seed=seed,initial='bare',algorithm=algorithm))
                state=engine.initial();step=0
                for target in targets:
                    while step*c.dt*AU_FS<target-1e-12:
                        state,raw=engine.step(state);maxraw=max(maxraw,raw);step+=1
                    row=observables(engine,state);row['t_fs']=step*c.dt*AU_FS
                    rows.append(row)
                    print(f'{name}: saved t={row["t_fs"]:.3f} fs',flush=True)
            elif name=='qd':
                engine=Propagator(c);state=initial(c);step=0
                for target in targets:
                    while step*c.dt*AU_FS<target-1e-12:state=engine.step(state);step+=1
                    row=qd_observables(state,engine);row['t_fs']=step*c.dt*AU_FS
                    rows.append(row)
            else:raise ValueError('Unknown method '+name)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:error=repr(exc)
        summary=dict(status='FAILED' if error else 'COMPLETE_NOT_CERTIFIED',error=error,
            last_saved_fs=rows[-1]['t_fs'] if rows else 0.,wall_seconds=time.perf_counter()-start,
            native_norm_step_error=maxraw,method=name,
            max_energy_drift=max((abs(r['energy']-rows[0]['energy']) for r in rows),default=None))
        if name=='gaussian':
            summary.update(nfev=engine.calls,max_ct_projection_tail=engine.max_tail,
                max_projected_norm_rate=engine.max_norm_rate,
                inactive_history_reset_count=engine.history_resets,
                no_propagated_q_grid=True,
                complex_variables_per_trajectory=engine.width)
        save(out,name,rows,summary,**({'y':np.asarray(ys),'t_fs':[r['t_fs'] for r in rows]} if ys else {}))
        summaries[name]=summary
    (out/'comparison_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['nonadiabatic','rabi'],default='nonadiabatic')
    p.add_argument('--resonance',choices=['on','off'],default='on')
    p.add_argument('--end-fs',type=float,default=1.)
    p.add_argument('--every-fs',type=float,default=.25)
    p.add_argument('--ntraj',type=int,default=16)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--nf',type=int,default=8)
    p.add_argument('--nr',type=int,default=128)
    p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--K',type=int,default=4)
    p.add_argument('--g',type=float,default=.01)
    p.add_argument('--dse',action='store_true')
    p.add_argument('--regularization',type=float,default=1e-4)
    p.add_argument('--gaussian-max-step',type=float,default=.5)
    p.add_argument('--tail-tolerance',type=float,default=1e-6)
    p.add_argument('--methods',nargs='+',choices=['qd','el','pol','gaussian'],default=['qd','el','pol','gaussian'])
    p.add_argument('--algorithm',choices=['ctmqc','ehrenfest'],default='ctmqc')
    p.add_argument('--no-render',action='store_true')
    a=p.parse_args();kw=vars(a).copy();kw.pop('no_render');result=run_comparison(**kw)
    if not a.no_render:
        from .render import render_comparison
        from .benchmark import audit_comparison
        render_comparison(a.out)
        if (a.out/'qd/metrics.json').exists():audit_comparison(a.out)
    if any(r['status']=='FAILED' for r in result.values()):raise SystemExit(2)


if __name__=='__main__':main()
