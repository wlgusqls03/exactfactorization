"""Predeclared Model A reproduction/convergence campaign, not a parameter search."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .model import Config,Propagator
from .factorization import analyze
from .run import simulate,atomic_json
from .plot import render

OBS_LIMITS={'P_S1':2e-4,'P_right':2e-4,'J_R4':2e-6,'n_ph':2e-3,'mean_R':2e-4}
FIELD_LIMITS={'rho_L1':1e-3,'epsilon1_rms':1e-3,'epsilon2_rms':1e-3,'force_rms':1e-3}


def settings():
    """Paper grid plus one-factor spacing/dt and fixed-spacing larger boxes."""
    c=Config()
    return {'coupled':c,'free':replace(c,g=0),
            'coupled_grid':replace(c,nr=600,nq=600),
            'coupled_dt':replace(c,dt=.025),
            # Same dR=.016 and same R=4 node (integer number of extra cells).
            'coupled_box':replace(c,rmin=-.992,rmax=8.992,nr=624,qmin=-16,qmax=16,nq=800),
            'free_grid':replace(c,g=0,nr=600,nq=600)}


def compare(root,left,right):
    """Observable histories + six full-wave EF events; NOT solver residual only."""
    root=Path(root);a=json.loads((root/left/'observables.json').read_text());b=json.loads((root/right/'observables.json').read_text())
    if [v['time_au'] for v in a]!=[v['time_au'] for v in b]:raise ValueError('Unmatched times')
    errors={k:max(abs(x[k]-y[k]) for x,y in zip(a,b)) for k in OBS_LIMITS}
    ca=Config(**json.loads((root/left/'identity.json').read_text())['config'])
    cb=Config(**json.loads((root/right/'identity.json').read_text())['config'])
    pa,pb=Propagator(ca),Propagator(cb);events=[]
    for path in sorted((root/left/'waves').glob('wave_*.npz')):
        other=root/right/'waves'/path.name
        if not other.exists():raise FileNotFoundError(other)
        with np.load(path) as z:fa=analyze(z['psi'],pa);t=float(z['time_au'])
        with np.load(other) as z:fb=analyze(z['psi'],pb)
        rr,qq=np.meshgrid(fa['R'],fa['q'],indexing='ij');pts=np.stack([rr,qq],axis=-1)
        interp=lambda k:RegularGridInterpolator((fb['R'],fb['q']),fb[k].real,bounds_error=False,fill_value=np.nan)(pts)
        rhoB=interp('rho_qR');mask=(fa['rho_qR']>1e-6*fa['rho_qR'].max())&(rhoB>1e-6*np.nanmax(rhoB))
        weight=fa['rho_qR'][mask];rms=lambda v:float(np.sqrt(np.sum(weight*v[mask]**2)/weight.sum()))
        er=rms(fa['epsilon1'].real-interp('epsilon1'))
        rmask=(fa['rho_R']>1e-6*fa['rho_R'].max())
        valid=np.isfinite(rhoB)
        entry=dict(time_au=t,rho_L1=float(np.sum(abs(fa['rho_qR'][valid]-rhoB[valid]))*pa.dr*pa.dq),epsilon1_rms=er)
        entry['outside_common_grid_mass']=float(fa['rho_qR'][~valid].sum()*pa.dr*pa.dq)
        entry['joint_comparison_mass']=float(weight.sum()*pa.dr*pa.dq)
        entry['rho_L1']+=entry['outside_common_grid_mass']
        for k in ['epsilon2','force']:
            vals=np.interp(fa['R'],fb['R'],fb[k].real)
            entry[k+'_rms']=float(np.sqrt(np.average((fa[k].real[rmask]-vals[rmask])**2,weights=fa['rho_R'][rmask])))
        events.append(entry)
    field_errors={k:max(e[k] for e in events) for k in FIELD_LIMITS}
    passed=all(np.isfinite(v) and v<OBS_LIMITS[k] for k,v in errors.items()) and all(np.isfinite(v) and v<FIELD_LIMITS[k] for k,v in field_errors.items())
    return dict(left=left,right=right,status='PASS' if passed else 'FAIL',observable_errors=errors,
                field_errors=field_errors,events=events,comparison='Native EF first, linear interpolation for grid comparison only; no scalar offset fitting')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--backend',choices=['cpu','gpu'],default='cpu');ap.add_argument('--end',type=float,default=1250)
    ap.add_argument('--no-movies',action='store_true');ap.add_argument('--plan',action='store_true')
    a=ap.parse_args();plan=dict(cases={k:asdict(v) for k,v in settings().items()},end_au=a.end,every_au=2.5,
        observation_limits=OBS_LIMITS,field_limits=FIELD_LIMITS,fields_cases=['coupled','free'],
        full_wave_events=[0,250,500,750,1000,1250],estimated_disk_upper_GiB=7,
        gpu_note='Two-component 2D wave, not x*q*R. Float64/complex128; sequential cases.')
    if a.plan:print(json.dumps(plan,indent=2));return
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'plan.json').exists() and json.loads((a.out/'plan.json').read_text())!=plan:raise ValueError('Campaign plan differs; use a new output folder')
    atomic_json(a.out/'plan.json',plan)
    statuses={}
    for name,c in settings().items():
        statuses[name]=simulate(c,a.out/name,end=a.end,backend=a.backend,fields=name in ['coupled','free'],resume=True)
    comparisons=[compare(a.out,'coupled',name) for name in ['coupled_grid','coupled_dt','coupled_box']]+[compare(a.out,'free','free_grid')]
    passed=a.end==1250 and all(s['status']=='PASS_INTERNAL' for s in statuses.values()) and all(c['status']=='PASS' for c in comparisons)
    report=dict(status='PASS_NUMERICAL' if passed else 'NOT_CERTIFIED',full_interval=a.end==1250,
                paper_reproduction_certified=False,comparisons=comparisons,internal=statuses,
                note='Paper caption vs full-PG scalar ambiguity remains; resemblance is not proof. No DSE. No new physical tuning.')
    atomic_json(a.out/'campaign_validation.json',report)
    from .photon_number import model_series,draw
    if not (a.out/'photon_comparison').exists():
        draw([model_series(a.out/name) for name in ['coupled','free']],a.out/'photon_comparison')
    for name in ['coupled','free']:
        target=a.out/(name+'_plots')
        if not (target/'manifest.json').exists():
            # Never overwrite interrupted rendering; choose explicit new folder on retry.
            if target.exists() and any(target.iterdir()):
                print(f'Partial render preserved: {target}; use plot --out NEW_DIRECTORY',flush=True)
            else:render(a.out/name,target,not a.no_movies)
    print(json.dumps(dict(status=report['status'],out=str(a.out.resolve())),indent=2))


if __name__=='__main__':main()
