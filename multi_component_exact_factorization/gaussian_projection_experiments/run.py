"""Opt-in fixed-R diagnostic driver; never modifies legacy results/defaults.

Example: python -m multi_component_exact_factorization.gaussian_projection_experiments.run
  --out results/YYYYMMDD/fixed_R_energy --components 2 --mode energy --end-fs .4
All reference propagation is diagnostic, optional, and absent from the RHS.
"""
import argparse
from dataclasses import asdict,dataclass,field
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from ..gaussian_trajectory.run import AU_FS,atomic_json,atomic_npz,source_hashes
from ..gaussian_trajectory.sphere_integrator import sphere_rk4_step
from .initialization import fit_distinct
from .fixed_problem import FixedProblem,MatchedReference
from .diagnostics import frame,summarize
from .integration import adaptive_interval


@dataclass
class ExperimentConfig:
    model:dict=field(default_factory=lambda:dict(name='pen-nonadiabatic',nq=64,g=0.,q_operator='fd2'))
    fixed_R:float=None
    components:int=1
    mode:str='standard'
    dt_au:float=.005
    end_fs:float=.4
    every_fs:float=.01
    separation:float=.65
    initialization:str='fit'
    rank_rtol:float=1e-12
    constraint_rtol:float=1e-10
    power_atol:float=1e-12
    reference:bool=True
    reference_max_dimension:int=2048
    max_auxiliary_defect:float=.05
    integrator:str='sphere-rk4'
    rtol:float=1e-8
    atol:float=1e-10
    adaptive_max_halvings:int=16
    adaptive_max_attempts:int=2048

    def validate(self):
        if self.mode not in ('standard','energy'):raise ValueError('Unknown projection mode')
        if self.initialization not in ('fit','hermite'):raise ValueError('Select fit or hermite initial mixture')
        for v in (self.dt_au,self.end_fs,self.every_fs):
            if not np.isfinite(v) or v<=0:raise ValueError('Positive finite times required')
        if int(self.components)!=self.components or self.components<1:raise ValueError('Positive integer K required')
        if not 0<self.separation<.95:raise ValueError('Need 0 < separation < .95')
        if not 0<self.rank_rtol<1 or not 0<self.constraint_rtol<1:raise ValueError('Invalid rank controls')
        if not np.isfinite(self.power_atol) or self.power_atol<0:raise ValueError('Invalid power tolerance')
        if not 0<self.max_auxiliary_defect<1:raise ValueError('Invalid sphere guard')
        if self.reference_max_dimension<1:raise ValueError('Invalid reference size limit')
        if self.integrator not in ('sphere-rk4','adaptive-sphere-rk4'):raise ValueError('Unknown integrator')
        if not np.all(np.isfinite([self.rtol,self.atol])) or min(self.rtol,self.atol)<=0:raise ValueError('Positive finite integration tolerances required')
        for value in (self.adaptive_max_halvings,self.adaptive_max_attempts):
            if isinstance(value,bool) or int(value)!=value or value<1:
                raise ValueError('Positive integer adaptive budgets required')


def fingerprints():
    return dict(legacy=source_hashes(),experiment={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob('*.py'))})


def run(config,out,*,resume=False):
    """Atomic restart is last accepted state. Failed proposal is never saved.

    Resuming permits only end_fs to change (same dt, model, source, method).
    Different dt/K/mode must run in a fresh directory, not silently fork history.
    """
    config.validate();out=Path(out);start=time.monotonic();hashes=fingerprints()
    nsteps=int(np.ceil(config.end_fs/(AU_FS*config.dt_au)-1e-10))
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError('Refusing to overwrite existing experiment: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    cfg=asdict(config);state=None;history=[];step=0;previous_wall=0.
    if resume:
        saved=json.loads((out/'config.json').read_text());check=dict(cfg)
        saved.pop('end_fs');check.pop('end_fs')
        if saved!=check:raise ValueError('Resume configuration differs beyond end_fs')
        metadata=json.loads((out/'metadata.json').read_text())
        if metadata['source_hashes']!=hashes:raise ValueError('Resume source hash mismatch')
        with np.load(out/'checkpoint.npz',allow_pickle=False) as z:
            state={k:z['state_'+k].copy() for k in ('C','z','residual_integral','energy_integral')}
            step=int(z['step']);G0=z['initial_G'].copy()
            keys=[k[8:] for k in z.files if k.startswith('history_')]
            history=[{k:z['history_'+k][j].copy() for k in keys} for j in range(len(z['history_t_au']))]
        previous_wall=json.loads((out/'status.json').read_text()).get('wall_seconds',0.)
        if nsteps<step:raise ValueError('Requested end precedes checkpoint')
    atomic_json(out/'config.json',cfg)
    try:
        problem=FixedProblem(config.model,config.fixed_R)
        if state is None:
            rho=np.sum(abs(problem.physical_G0)**2,axis=1)
            bridge,fit_audit=fit_distinct(problem.q,problem.w,rho,config.components,
                separation=config.separation,rank_rtol=config.rank_rtol,optimize=config.initialization=='fit')
            state=dict(C=problem.C0.copy(),z=bridge.parameters.copy(),residual_integral=np.array(0.),energy_integral=np.array(0.))
            G0=bridge.amplitude[:,None]*state['C']
            atomic_json(out/'initialization.json',fit_audit)
        reference=MatchedReference(problem,G0,config.reference_max_dimension) if config.reference else None
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        status=dict(status='FAILED_INITIALIZATION',error=repr(exc),audit=getattr(exc,'audit',{}),wall_seconds=time.monotonic()-start)
        atomic_json(out/'status.json',status);return status
    if not resume:
        revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False).stdout.strip()
        atomic_json(out/'metadata.json',dict(source_hashes=hashes,git_revision=revision,
            fixed_R=problem.R,hermitian_error=problem.hermitian_error,cached_action_error=problem.cached_action_error,
            method='fixed-R experiment; general complex C(q); positive bridge only',
            reference_feedback=False,energy_post_step_repair=False,moving_R=False,
            photon_discretization=problem.model.photon_discretization_audit(),
            grid_note='Periodic finite-grid Hamiltonian; matched references do not certify continuum convergence',
            constraint_scope='Only fixed R and time-independent Hermitian H; not applied to MTE/TSH/CTMQC forces'))
    trial_audit={}
    def evaluate(s):
        derivative,detail=problem.rate(s,config.components,config.mode,config.rank_rtol,config.constraint_rtol,config.power_atol)
        audit=detail[3].audit
        trial_audit['max_parameter_rate']=max(trial_audit.get('max_parameter_rate',0.),float(np.linalg.norm(derivative['z'])))
        trial_audit['max_condition_A']=max(trial_audit.get('max_condition_A',0.),float(audit['condition_L']))
        if 'constraint_resolvability' in audit:
            trial_audit['minimum_constraint_resolvability']=min(trial_audit.get('minimum_constraint_resolvability',1.),audit['constraint_resolvability'])
            trial_audit['minimum_energy_covector_norm']=min(trial_audit.get('minimum_energy_covector_norm',float('inf')),audit['constraint_covector_norm'])
        return derivative,detail
    def record(detail):
        f=frame(problem,state,detail,step*config.dt_au,reference)
        if history and abs(float(history[-1]['t_au'])-f['t_au'])<1e-14:history[-1]=f
        else:history.append(f)
    def save(status,**extra):
        arrays,summary=summarize(history,status)
        summary.update(last_saved_fs=step*config.dt_au*AU_FS,step=step,
            wall_seconds=previous_wall+time.monotonic()-start,**extra)
        # One atomic checkpoint contains its matching diagnostics history.
        atomic_npz(out/'checkpoint.npz',step=np.array(step),initial_G=G0,
            **{'state_'+k:v for k,v in state.items()},**{'history_'+k:v for k,v in arrays.items()})
        atomic_npz(out/'diagnostics.npz',q=problem.q,w=problem.w,kq=problem.model.kq,**arrays)
        atomic_json(out/'status.json',summary)
        return summary
    interval=max(1,int(round(config.every_fs/(AU_FS*config.dt_au))))
    detail=None;lastlog=time.monotonic()
    try:
        _,detail=evaluate(state);record(detail);save('RUNNING')
        print('RUN %s K=%d %s nq=%d g=%g dt=%g au -> %g fs'%(out,config.components,config.mode,len(problem.q),problem.model.config.g,config.dt_au,config.end_fs),flush=True)
        while step<nsteps:
            trial_audit.clear()
            if config.integrator=='adaptive-sphere-rk4':
                proposed,audit=adaptive_interval(state,config.dt_au,evaluate,rtol=config.rtol,atol=config.atol,
                                               max_auxiliary_defect=config.max_auxiliary_defect,
                                               max_halvings=config.adaptive_max_halvings,
                                               max_attempts=config.adaptive_max_attempts)
            else:
                proposed,audit=sphere_rk4_step(state,config.dt_au,evaluate,max_auxiliary_defect=config.max_auxiliary_defect)
            # Reject nonfinite/rank-deficient final physical states too, not
            # just the four RK stages. This adds an explicit final-state guard.
            _,new_detail=evaluate(proposed)
            state=proposed;step+=1;detail=new_detail
            if step%interval==0 or step==nsteps:record(detail);save('RUNNING')
            if time.monotonic()-lastlog>15:
                print('  t=%.6f fs E=%.9g residual=%.3g'%(step*config.dt_au*AU_FS,frame(problem,state,detail,step*config.dt_au)['energy'],detail[-1]),flush=True)
                lastlog=time.monotonic()
        result=save('COMPLETE_NOT_CERTIFIED')
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        failure=dict(error=repr(exc),attempted_step=step+1,audit=getattr(exc,'audit',{}),
                     last_interval_rhs_diagnostics=trial_audit.copy(),
                     instruction='Last accepted checkpoint only. No fallback, mask or energy repair was applied.')
        atomic_json(out/'failure.json',failure)
        if detail is not None:
            record(detail);result=save('FAILED',**failure)
        else:
            result=dict(status='FAILED_INITIAL_RHS',**failure);atomic_json(out/'status.json',result)
    print('%s %s last=%.6f fs'%(result['status'],out,step*config.dt_au*AU_FS),flush=True)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--model',choices=['pen-nonadiabatic','pen-rabi','mcef'],default='pen-nonadiabatic')
    p.add_argument('--components',type=int,default=1);p.add_argument('--mode',choices=['standard','energy'],default='standard')
    p.add_argument('--q-operator',choices=['fd2','fd4','fft'],default='fd2')
    p.add_argument('--nq',type=int,default=64);p.add_argument('--qmin',type=float,default=-14);p.add_argument('--qmax',type=float,default=14)
    p.add_argument('--nx',type=int,default=33);p.add_argument('--g',type=float,default=0.)
    p.add_argument('--omega',type=float);p.add_argument('--dse',action='store_true');p.add_argument('--fixed-R',type=float)
    p.add_argument('--dt-au',type=float,default=.005);p.add_argument('--end-fs',type=float,default=.4)
    p.add_argument('--every-fs',type=float,default=.01);p.add_argument('--separation',type=float,default=.65)
    p.add_argument('--initialization',choices=['fit','hermite'],default='fit',help='fit: optimize initial amplitude; hermite: better-separated convolution seed with explicitly reported initial error')
    p.add_argument('--rank-rtol',type=float,default=1e-12);p.add_argument('--constraint-rtol',type=float,default=1e-10)
    p.add_argument('--power-atol',type=float,default=1e-12);p.add_argument('--no-reference',action='store_true')
    p.add_argument('--reference-max-dimension',type=int,default=2048)
    p.add_argument('--integrator',choices=['sphere-rk4','adaptive-sphere-rk4'],default='sphere-rk4')
    p.add_argument('--rtol',type=float,default=1e-8);p.add_argument('--atol',type=float,default=1e-10)
    p.add_argument('--adaptive-max-halvings',type=int,default=16)
    p.add_argument('--adaptive-max-attempts',type=int,default=2048)
    p.add_argument('--render',action='store_true');p.add_argument('--movies',action='store_true')
    return p


def config_from_args(a):
    return ExperimentConfig(model=dict(name=a.model,nq=a.nq,qmin=a.qmin,qmax=a.qmax,nx=a.nx,
        g=a.g,dse=a.dse,omega=a.omega,q_operator=a.q_operator),fixed_R=a.fixed_R,
        components=a.components,mode=a.mode,dt_au=a.dt_au,end_fs=a.end_fs,every_fs=a.every_fs,
        separation=a.separation,initialization=a.initialization,rank_rtol=a.rank_rtol,constraint_rtol=a.constraint_rtol,
        power_atol=a.power_atol,reference=not a.no_reference,reference_max_dimension=a.reference_max_dimension,
        integrator=a.integrator,rtol=a.rtol,atol=a.atol,
        adaptive_max_halvings=a.adaptive_max_halvings,adaptive_max_attempts=a.adaptive_max_attempts)


def main():
    a=parser().parse_args();result=run(config_from_args(a),a.out,resume=a.resume)
    if (a.render or a.movies) and (a.out/'diagnostics.npz').exists():
        from .render import render_case
        render_case(a.out,movies=a.movies)
    if result['status']!='COMPLETE_NOT_CERTIFIED':raise SystemExit(2)


if __name__=='__main__':main()
