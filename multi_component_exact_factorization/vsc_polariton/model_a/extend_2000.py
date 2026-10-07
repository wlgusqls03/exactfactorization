"""Model A only: new 0--2000 au state movies, read-only 1250 au references.

Primary coupled/free runs replay from zero because old compact EF fields do not
contain electronic channel densities. Four numerical controls continue existing
1250 au waves. Same Hamiltonian, grid, dt and initial preparation; NO DSE.
This is diagnostic propagation, never an automatic extension of certification.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import tarfile
import numpy as np
from .model import Config,Propagator,initial,observables
from .factorization import analyze,compact,diagnostics
from .run import atomic_json,atomic_npz,backend_check,LIMITS
from .event_audit import sha,populations,fields,compare_fields
from .state_movies import Channels

SOURCES=('model.py','factorization.py','run.py','event_audit.py','state_movies.py','extend_2000.py')


def check_source(folder):
    """Reject a different propagation kernel, but allow new plotting modules."""
    ident=json.loads((folder/'identity.json').read_text())
    for name in ['model.py','factorization.py']:
        if ident['source'][name]!=sha(Path(__file__).parent/name):raise ValueError(f'Old kernel mismatch: {name}')
    return Config(**ident['config'])


def evolve(c,out,end=2000.,every=2.5,backend='cpu',source=None,dense=True,resume=False):
    """Serial streaming, sparse wave events, separate atomic restart. [R,q,2].

    Conserved energy is checked against the ORIGINAL t=0 record for continuation.
    Primary replay is checked against original wave_1250, including its phase.
    """
    out=Path(out);source=Path(source) if source else None
    for v in [end/c.dt,every/c.dt]:
        if abs(v-round(v))>1e-8:raise ValueError('Time must lie on timestep grid')
    start_time=0. if dense or source is None else 1250.
    if end<start_time:raise ValueError('End before starting wave')
    if out.exists() and not resume:raise FileExistsError(out)
    ident=dict(config=asdict(c),end_au=end,every_au=every,dense=dense,start_au=start_time,
        source={n:sha(Path(__file__).parent/n) for n in SOURCES},
        reference_wave_sha256=sha(source/'waves/wave_1250.npz') if source else None)
    if out.exists() and (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=ident:raise ValueError('Restart identity mismatch; use a new output folder')
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'identity.json',ident)
    for name in ['waves']+(['fields','states'] if dense else []):(out/name).mkdir(exist_ok=True)
    xp=np
    if backend=='gpu':
        import cupy as xp
        check=backend_check(c);atomic_json(out/'backend_check.json',check)
        if check['status']!='PASS':raise RuntimeError('GPU equivalence failed')
    prop=Propagator(c,xp);host=prop if backend=='cpu' else Propagator(c)
    channels=Channels(c) if dense else None;checkpoint=out/'restart.npz'
    rows=[];ef=[];start=int(round(start_time/c.dt));comparison=None
    if checkpoint.exists() and resume:
        with np.load(checkpoint,allow_pickle=False) as z:
            u=xp.asarray(z['psi']);start=int(z['step']);history=json.loads(str(z['history']))
        rows=history['rows'];ef=history['ef'];comparison=history['replay']
    elif start_time:
        with np.load(source/'waves/wave_1250.npz',allow_pickle=False) as z:
            if float(z['time_au'])!=1250 or z['psi'].shape!=(c.nr,c.nq,2):raise ValueError('Wrong continuation wave')
            u=xp.asarray(z['psi'])
        rows=[r for r in json.loads((source/'observables.json').read_text()) if r['time_au']<1250]
        ef=[r for r in json.loads((source/'ef_diagnostics.json').read_text()) if r['time_au']<1250]
    else:u=xp.asarray(initial(c))
    last=round(end/c.dt);cadence=round(every/c.dt);timer=time.perf_counter()
    for step in range(start,last+1):
        if step>start:u=prop.step(u)
        if step%cadence and step!=last:continue
        if rows and rows[-1]['step']==step:continue
        t=step*c.dt;row=observables(u,prop);row.update(step=step,time_au=t,time_fs=t*.024188843265857)
        event=abs(t/250-round(t/250))<1e-9 or step==last
        wave=np.asarray(u) if backend=='cpu' else xp.asnumpy(u)
        row['P_right_spectral']=populations(wave,c)['spectral_wave2R']
        rows.append(row)
        if dense or event:
            f=analyze(wave,host);d=diagnostics(f);d.update(step=step,time_au=t);ef.append(d)
            if dense:
                state=channels.project(wave)
                if np.max(abs(state['BO_global']-np.array([row['P_S0'],row['P_S1']])))>1e-9:raise ValueError('BO population inconsistent')
                atomic_npz(out/'states'/f'states_{step:07d}.npz',**state,time_au=t)
                atomic_npz(out/'fields'/f'fields_{step:07d}.npz',**compact(f),time_au=t,time_fs=row['time_fs'])
            if event:
                filename=f'wave_{round(t):04d}.npz' if abs(t-round(t))<1e-9 else f'wave_step_{step:07d}.npz'
                atomic_npz(out/'waves'/filename,psi=wave,time_au=t)
        if dense and source and abs(t-1250)<1e-9:
            with np.load(source/'waves/wave_1250.npz') as z:original=z['psi']
            comparison=float(np.linalg.norm(wave-original)*np.sqrt(prop.dr*prop.dq))
            if comparison>1e-8:raise ValueError(f'Replay differs from old wave at 1250 au: {comparison}')
        if step==start or step==last or abs(t/25-round(t/25))<1e-9:
            atomic_npz(checkpoint,psi=wave,step=step,history=json.dumps(dict(rows=rows,ef=ef,replay=comparison)))
            atomic_json(out/'observables.json',rows);atomic_json(out/'ef_diagnostics.json',ef)
            print(f'{out.name}: {t:g}/{end:g} au | P(S1)={row["P_S1"]:.6f} | N={row["n_ph"]:.6f}',flush=True)
    errors=dict(norm=max(abs(r['norm']-1) for r in rows),energy_drift=max(abs(r['energy']-rows[0]['energy']) for r in rows),
        R_edge=max(r['R_edge'] for r in rows),q_edge=max(r['q_edge'] for r in rows),
        reconstruction_L2=max(r['reconstruction_L2'] for r in ef),
        continuity=max(v['continuity'] for r in ef for v in r['supports'].values()),
        ef_route=max(v[k] for r in ef for v in r['supports'].values() for k in ['epsilon1_route','epsilon2_route','epsilon2_nested','epsilon1_imag','epsilon2_imag']))
    status=dict(status='PASS_INTERNAL' if all(np.isfinite(v) and v<LIMITS[k] for k,v in errors.items()) else 'FAIL_INTERNAL',
        errors=errors,limits=LIMITS,complete=rows[-1]['step']==last,phase_pass=False,field_convergence_certified=False,
        replay_L2_at_1250=comparison,elapsed_seconds=time.perf_counter()-timer,
        note='Model A only. Newly computed extension; old 1250 au certification is not extended.')
    atomic_json(out/'status.json',status);return status


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--backend',choices=['cpu','gpu'],default='gpu');ap.add_argument('--end',type=float,default=2000.)
    ap.add_argument('--no-movies',action='store_true');ap.add_argument('--plan',action='store_true')
    args=ap.parse_args();source=args.source.resolve();out=args.out.resolve()
    if args.end<=1250:raise ValueError('Extension endpoint must exceed 1250 au')
    if source==out or source in out.parents:raise ValueError('New output must be outside original campaign')
    names=['coupled','free','coupled_grid','coupled_dt','coupled_box','free_grid']
    configs={n:check_source(source/n) for n in names}
    inputs={str(p.relative_to(source)):sha(p) for n in names for p in [source/n/'identity.json',source/n/'observables.json',source/n/'ef_diagnostics.json',source/n/'waves/wave_1250.npz']}
    plan=dict(end_au=args.end,end_fs=args.end*.024188843265857,every_au=2.5,
        cases={n:dict(config=asdict(c),start_au=0 if n in ['coupled','free'] else 1250) for n,c in configs.items()},
        source_hashes=inputs,source_root=str(source),dense_frames=int(round(args.end/2.5))+1,
        required_free_GiB=10,precision='complex128',physical_changes='duration only',
        no_DSE=True,not_certified=True,note='Two primaries replay for new state projections; four controls continue. No VSC propagation.')
    if args.plan:print(json.dumps(plan,indent=2));return
    if not args.no_movies and shutil.which('ffmpeg') is None:raise RuntimeError('ffmpeg required before starting this movie campaign')
    if out.exists() and (out/'extension_plan.json').exists():
        if json.loads((out/'extension_plan.json').read_text())!=plan:raise ValueError('Changed plan; choose new output')
    elif out.exists():raise FileExistsError(out)
    already=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    required=max(1024**3,10*1024**3-already)
    if shutil.disk_usage(out.parent).free<required:raise OSError(f'Require {required/1024**3:.2f} GiB additional free space; preserve original inputs')
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'extension_plan.json',plan)
    statuses={n:evolve(c,out/n,end=args.end,backend=args.backend,source=source/n,dense=n in ['coupled','free'],resume=True) for n,c in configs.items()}
    comparisons=[]
    for t in [1250,1500,1750,2000]:
        if t>args.end:continue
        for left,right in [('coupled','coupled_grid'),('coupled','coupled_dt'),('coupled','coupled_box'),('free','free_grid')]:
            data=[]
            for n in [left,right]:
                with np.load(out/n/'waves'/f'wave_{t:04d}.npz') as z:data.append(fields(z['psi'],configs[n]))
            r=compare_fields(*data);r.update(left=left,right=right,time_au=t);comparisons.append(r)
    atomic_json(out/'extension_validation.json',dict(status='DIAGNOSTIC_COMPLETE_NOT_CERTIFIED',phase_pass=False,internal=statuses,comparisons=comparisons))
    if not args.no_movies:
        from .review_plot import render
        from .state_movies import render as state_render
        from .component_detail import render as component_render
        if not (out/'plots').exists():render(out,out/'plots',families=('vectors','tdpes1','outer'))
        elif not (out/'plots/manifest.json').exists():raise RuntimeError('Partial movie output preserved; choose a fresh render folder')
        for n in ['coupled','free']:
            for sub,renderer in [('state_movies',state_render),('component_movies',component_render)]:
                target=out/n/sub
                if not target.exists():renderer(out/n,target)
                elif not (target/'manifest.json').exists():raise RuntimeError(f'Partial render preserved: {target}')
    for name,h in inputs.items():
        if sha(source/name)!=h:raise RuntimeError('Original input changed: '+name)
    archive=out/'model_a_2000_review.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'x:gz') as tar:
            for p in sorted(out.rglob('*')):
                selected_wave=p.parent.name=='waves' and p.name in ['wave_1500.npz','wave_2000.npz']
                if p.is_file() and (p.suffix in ['.json','.png','.mp4'] or selected_wave or p.parent.name=='states'):
                    tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
    print('SEND: '+str(archive),flush=True)


if __name__=='__main__':main()
