"""Dense true-time VSC movies with bounded disk use; no sparse-wave interpolation.

Replay the SAME validated real-grid split operator on GPU, extracting compact
MCEF fields on CPU every 4 au (~0.097 fs). Save one restart, not hundreds of
3D waves. Old packets, validation, propagation and artifacts stay read-only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace
import numpy as np
from .run_phase6_gpu import sha,save_json,save_npz,load_packet,environment
from .run_photon_real_grid import identity,failures
from .photon_real_grid import RealGridPF,initial_grid
from .real_grid_mcef_fields import analyze,diagnostics

ROOT=Path(__file__).resolve().parents[2]/'results/vsc_polariton/photon_real_grid'


def schedule(end,dt,cadence):
    """Integer-time actual frames; final endpoint is always included."""
    for t in (end,cadence):
        if t<=0 or dt<=0 or abs(round(t/dt)*dt-t)>1e-10:
            raise ValueError('Positive end/cadence must be integer multiples of dt')
    total=round(end/dt);every=round(cadence/dt)
    return sorted(set(range(0,total+1,every))|{total})


def estimate(shape,frames):
    """Conservative output forecast, full float64/complex128, no lossy storage."""
    nr,nx,nq=shape
    wave=nr*nx*nq*16
    fields=nr*nq*8*7+nr*8*20+nq*16
    return dict(frame_count=frames,restart_GiB=wave/2**30,
        compact_fields_upper_GiB=fields*frames/2**30,
        required_free_GiB=(fields*frames+2*wave)/2**30+2,
        note='Includes atomic restart temporary + 2 GiB reserve; MP4 size depends on content. CPU extraction also needs several GiB RAM.')


def compare_observation(row,path):
    """Reproducibility cross-check against original trajectory, not new physics."""
    limits=dict(norm=1e-9,energy=1e-9,product=1e-9,flux=1e-10,nph=1e-8,mean_R=1e-9,P_exc=1e-9)
    if not path.exists():return {'status':'NO_MATCHING_SAVED_OBSERVABLE'}
    with np.load(path,allow_pickle=False) as old:
        errors={k:abs(float(row[k])-float(old[k])) for k in limits}
    bad={k:v for k,v in errors.items() if not np.isfinite(v) or v>limits[k]}
    if bad:raise ValueError('Replay differs from original observables: '+str(bad))
    return dict(status='PASS',errors=errors)


def replay(args,packet,digest,gate,source):
    """Restartable identical GPU propagation + native-FFT instantaneous fields."""
    from .vsc_movie_only import compact_fields
    base=gate['identity']
    cfg=SimpleNamespace(dt=base['dt'],half_Q=base['half_Q'],nq=base['nq'])
    if not gate.get('gpu_propagation_allowed') or identity(cfg,digest)!=base:
        raise ValueError('Need unchanged matching real-grid CPU/GPU validation')
    if source['status']!='COMPLETE_NOT_CERTIFIED':
        raise ValueError('Original full run must be completed; cannot bypass a failed propagation')
    if any(source['identity'].get(k)!=v for k,v in base.items()):
        raise ValueError('Source trajectory identity mismatch')
    if args.end_au>float(source['time_au'])+1e-12:
        raise ValueError('No extension beyond original completed interval')
    frames=schedule(args.end_au,cfg.dt,args.every_au)
    signature=dict(base=base,end_au=args.end_au,every_au=args.every_au,block=args.block,
        sources={n:sha(Path(__file__).with_name(n)) for n in
                 ('run_vsc_dense_movies.py','real_grid_mcef_fields.py','vsc_movie_only.py')},
        source_status_sha256=sha(args.source_run/'status.json'))
    status=args.out/'replay_status.json';checkpoint=args.out/'restart.npz'
    if status.exists():
        old=json.loads(status.read_text())
        if old['identity']!=signature:raise ValueError('Different replay identity; choose NEW --out')
        if old['status']=='COMPLETE_DIAGNOSTIC':return
        if old['status']=='FAILED_DIAGNOSTIC':raise ValueError('Failed attempt preserved; use NEW --out')
    (args.out/'fields').mkdir(exist_ok=True)
    shape=(len(packet['R']),len(packet['x']),cfg.nq)
    plan=estimate(shape,sum(not (args.out/'fields'/f'fields_{s:07d}.npz').exists() for s in frames))
    if shutil.disk_usage(args.out).free/2**30<plan['required_free_GiB']:
        raise OSError('Insufficient free disk: '+str(plan))
    gpu=RealGridPF(packet,cfg.half_Q,cfg.nq,cfg.dt,True,args.device)
    env=environment(gpu.xp)
    if any(env[k]!=gate['environment'][k] for k in ('gpu','cupy','driver','runtime')):
        raise ValueError('GPU environment changed; rerun original real-grid hardware check first')
    encoded=json.dumps(signature,sort_keys=True)
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as z:
            if str(z['identity'])!=encoded:raise ValueError('Restart identity mismatch')
            u=gpu.xp.asarray(z['psi']);start=int(z['step']);e0=float(z['initial_energy'])
    else:
        u0,err=initial_grid(packet,gpu.Q)
        if err['norm_difference']>1e-10 or err['backprojection_L2']>1e-8:
            raise ValueError('Initial transform failed')
        u=gpu.xp.asarray(u0);del u0
        start=0;e0=gpu.observe(u)['energy']
    stop=[False];handlers={}
    for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    total=frames[-1];frame_set=set(frames);ck=round(64/cfg.dt);begin=time.monotonic()
    try:
        for step in range(start,total+1):
            if step in frame_set or stop[0]:
                row=gpu.observe(u);bad=failures(row,e0)
                if bad:raise ValueError('Original propagation limits failed: '+str(bad))
                match=compare_observation(row,args.source_run/f'observable_{step:07d}.npz')
                dest=args.out/'fields'/f'fields_{step:07d}.npz'
                if step in frame_set and not dest.exists():
                    t0=time.monotonic();host=gpu.host(u)
                    with np.errstate(divide='ignore',invalid='ignore',over='ignore',under='ignore'):
                        fields=analyze(host,packet,gpu.Q,args.block)
                    del host
                    fields['time_au']=step*cfg.dt
                    check=diagnostics(fields)
                    # Diagnostics are retained, never promoted into a Phase7 PASS.
                    check.update(replay_observation=match,analysis_seconds=time.monotonic()-t0,
                                 energy_drift=row['energy']-e0,time_au=step*cfg.dt)
                    save_npz(dest,**compact_fields(fields))
                    save_json(dest.with_suffix('.json'),check);del fields
                if step%ck==0 or step==total or stop[0]:
                    host=gpu.host(u)
                    save_npz(checkpoint,psi=host,step=step,initial_energy=e0,identity=encoded)
                    del host
                state='COMPLETE_DIAGNOSTIC' if step==total else 'INTERRUPTED' if stop[0] else 'RUNNING'
                save_json(status,dict(identity=signature,status=state,step=step,time_au=step*cfg.dt,
                          phase7_pass=False,segment_seconds=time.monotonic()-begin,storage=plan))
                print(f'{state} {step}/{total} t={step*cfg.dt*.024188843265857:.3f} fs | compact fields only',flush=True)
                if state!='RUNNING':return
            u=gpu.step(u)
    except Exception as exc:
        save_json(status,dict(identity=signature,status='FAILED_DIAGNOSTIC',step=step,error=str(exc),phase7_pass=False))
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','replay','render'),default='plan')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--source-run',type=Path)
    p.add_argument('--validation',type=Path)
    p.add_argument('--fields',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--end-au',type=float,default=1652.)
    p.add_argument('--every-au',type=float,default=4.)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--block',type=int,default=4)
    p.add_argument('--fps',type=float,default=24)
    p.add_argument('--density-floor',type=float,default=1e-5)
    p.add_argument('--support-budget',type=float,default=1e-8)
    p.add_argument('--families',nargs='+',choices=('state','nuclear','photon'),default=['state','nuclear','photon'])
    p.add_argument('--epsilon1-vmax-ev',type=float,
                   help='Display epsilon1 in a FIXED +/- eV range; tails saturate, raw fields unchanged')
    p.add_argument('--epsilon1-scale',choices=('linear','symlog'),default='symlog')
    p.add_argument('--signed-scale',choices=('linear','symlog'),default='symlog',
                   help='Display scale for connections, force and epsilon2; densities remain log-scaled')
    p.add_argument('--allow-sparse-preview',action='store_true')
    p.add_argument('--no-render',action='store_true')
    a=p.parse_args(argv)
    if a.epsilon1_vmax_ev is not None and (not np.isfinite(a.epsilon1_vmax_ev) or a.epsilon1_vmax_ev<=0):
        p.error('--epsilon1-vmax-ev must be finite and positive')
    a.out=a.out.resolve();a.out.relative_to(ROOT.resolve())
    if a.block<1 or a.fps<=0 or not 0<a.density_floor<1 or not 0<a.support_budget<1:p.error('Invalid settings')
    # Set cache destinations before importing Matplotlib. Scientific outputs stay under results.
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'movie_cache/matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'movie_cache'))
    if a.mode=='render':
        if not a.fields:p.error('--render requires --fields')
        with np.load(a.input,allow_pickle=False) as z:omega=float(z['omega'])
        from .vsc_movie_only import render
        render(sorted(a.fields.glob('fields_*.npz')),a.out,omega,a.fps,a.density_floor,
               a.support_budget,a.allow_sparse_preview,a.families,
               epsilon1_vmax_ev=a.epsilon1_vmax_ev,epsilon1_scale=a.epsilon1_scale,signed_scale=a.signed_scale)
        return
    if not a.validation or not a.source_run:p.error('plan/replay requires --validation and --source-run')
    gate=json.loads(a.validation.read_text());source=json.loads((a.source_run/'status.json').read_text())
    packet,digest=load_packet(a.input)
    frames=schedule(a.end_au,gate['identity']['dt'],a.every_au)
    plan=estimate((len(packet['R']),len(packet['x']),gate['identity']['nq']),len(frames))
    plan.update(cadence_fs=a.every_au*.024188843265857,end_fs=a.end_au*.024188843265857,
                propagation_hours_estimate=gate.get('estimated_39p96fs_propagation_hours'),
                timing_note='Replay estimate excludes CPU MCEF extraction and rendering. Use short probe to measure these.')
    print(json.dumps(plan,indent=2),flush=True)
    if a.mode=='plan':return
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1':p.error('Set OPENBLAS_NUM_THREADS=1')
    a.out.mkdir(parents=True,exist_ok=True)
    lock=a.out/'active.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        replay(a,packet,digest,gate,source)
        status=json.loads((a.out/'replay_status.json').read_text())
        if status['status']!='COMPLETE_DIAGNOSTIC':raise SystemExit(130)
        if not a.no_render and status['status']=='COMPLETE_DIAGNOSTIC' and not (a.out/'movies').exists():
            from .vsc_movie_only import render
            render(sorted((a.out/'fields').glob('fields_*.npz')),a.out/'movies',float(packet['omega']),
                   a.fps,a.density_floor,a.support_budget,False,a.families,
                   epsilon1_vmax_ev=a.epsilon1_vmax_ev,epsilon1_scale=a.epsilon1_scale,signed_scale=a.signed_scale)
    finally:lock.unlink()


if __name__=='__main__':main()
