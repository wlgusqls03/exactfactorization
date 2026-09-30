"""Gated, restartable, low-storage full-coordinate PF GPU runner.

Outputs only under results/vsc_polariton/photon_real_grid. Initial packets
are reused read-only; no new transfer bundle or historical source edit.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
import numpy as np
from .photon_real_grid import RealGridPF, initial_grid
from .run_phase6_gpu import sha,save_json,save_npz,load_packet,environment

ROOT=Path(__file__).resolve().parents[2]/'results/vsc_polariton/photon_real_grid'
EVENT_AU=(0.,160.,608.,672.,1408.,1440.,1652.)
LIMITS=dict(norm=1e-9,energy=1e-6,electron_edge=1e-8,edge=1e-8,
            photon_edge=1e-8,continuity_error=2e-6)


def codes():
    return {name:sha(Path(__file__).with_name(name)) for name in
            ('run_photon_real_grid.py','photon_real_grid.py','run_phase6_gpu.py','phase6_gpu_backend.py')}


def failures(row,e0):
    values={k:abs(row[k]) for k in LIMITS}
    values['norm']=abs(row['norm']-1);values['energy']=abs(row['energy']-e0)
    bad={k:dict(value=float(v),limit=LIMITS[k]) for k,v in values.items()
         if not np.isfinite(v) or v>=LIMITS[k]}
    if row['BO_projection_remainder'] < -1e-9:bad['BO_normalization']=row['BO_projection_remainder']
    return bad


def disk_guard(out,wave_bytes,remaining_waves=0):
    # Existing restart and snapshots occupy space already. Need a pending
    # checkpoint plus room for its first creation and future scientific waves.
    required=(remaining_waves+2)*wave_bytes+2*1024**3
    free=shutil.disk_usage(out).free
    if free<required:raise OSError(f'Need {required/2**30:.2f} GiB free; have {free/2**30:.2f}; nothing deleted')


def identity(args,digest):
    return dict(input_sha256=digest,code_sha256=codes(),dt=args.dt,
                half_Q=args.half_Q,nq=args.nq,representation='Psi_Q(R,x,Q); integral dR dx dQ')


def check(args,packet,digest):
    """8 full-shape CPU/GPU steps + >=128 GPU steps; measured not assumed."""
    if (args.out/'gpu_validation.json').exists():raise FileExistsError('Validation is immutable')
    cpu=RealGridPF(packet,args.half_Q,args.nq,args.dt)
    initial,transform=initial_grid(packet,cpu.Q)
    if transform['norm_difference']>1e-10 or transform['backprojection_L2']>1e-8:
        raise ValueError('Initial transform failed: '+str(transform))
    # Finite input basis is checked, not normalized into apparent agreement.
    gpu=RealGridPF(packet,args.half_Q,args.nq,args.dt,True,args.device);xp=gpu.xp
    u=initial.copy();v=xp.asarray(initial)
    a0=cpu.observe(u);b0=gpu.observe(v);gpu.sync()
    cpu_action=cpu.action(u);gpu_action=gpu.host(gpu.action(v))
    action_error=float(np.linalg.norm(cpu_action-gpu_action)*np.sqrt(cpu.volume))
    del cpu_action,gpu_action
    begin=time.perf_counter()
    for _ in range(args.cpu_steps):u=cpu.step(u)
    cpu_seconds=time.perf_counter()-begin
    gpu.sync();begin=time.perf_counter()
    for _ in range(args.cpu_steps):v=gpu.step(v)
    gpu.sync();gpu_agreement_seconds=time.perf_counter()-begin
    a=cpu.observe(u);b=gpu.observe(v)
    wave_error=float(np.linalg.norm(u-gpu.host(v))*np.sqrt(cpu.volume))
    errors={k:abs(a[k]-b[k]) for k in ('norm','energy','product','flux','P_exc','nph','mean_R')}
    errors.update(wave_L2=wave_error,action_L2=action_error)
    agreement=all(np.isfinite(v) and v<1e-9 for v in errors.values())
    del cpu,u,v;gc.collect()
    v=xp.asarray(initial);del initial
    e0=b0['energy'];bad=failures(b0,e0)
    propagation=0.;observing=0.;rows=[]
    for step in range(1,args.steps+1):
        gpu.sync();begin=time.perf_counter();v=gpu.step(v);gpu.sync()
        propagation+=time.perf_counter()-begin
        if step%16==0 or step==args.steps:
            begin=time.perf_counter();row=gpu.observe(v);gpu.sync();observing+=time.perf_counter()-begin
            f=failures(row,e0)
            if f:bad[str(step)]=f
            rows.append(dict(step=step,energy_drift=row['energy']-e0,norm=row['norm'],
                             photon_edge=row['photon_edge'],failures=f))
            print(f'GPU check {step}/{args.steps}: dE={row["energy"]-e0:.3e}, norm={row["norm"]:.12f}',flush=True)
            if f:break
    ok=agreement and not bad and args.steps>=128 and args.cpu_steps>=8 and step==args.steps
    info=environment(xp)
    sec=propagation/step
    result=dict(status='PASS' if ok else 'FAIL',gpu_propagation_allowed=ok,
        phase6_pass=False,phase7_pass=False,identity=identity(args,digest),environment=info,
        transform=transform,errors=errors,failures=bad,records=rows,
        cpu_steps=args.cpu_steps,cpu_seconds=cpu_seconds,gpu_agreement_seconds=gpu_agreement_seconds,
        gpu_steps=step,gpu_propagation_seconds=propagation,gpu_observation_seconds=observing,
        gpu_seconds_per_step=sec,estimated_39p96fs_propagation_hours=1652/args.dt*sec/3600,
        wave_GiB=float(np.prod(gpu.shape)*16/2**30),
        pool_reserved_GiB=xp.get_default_memory_pool().total_bytes()/2**30,
        note='Reserved pool is not peak VRAM. Forecast excludes I/O/setup/analysis. Hardware agreement is NOT basis or timestep convergence.')
    save_json(args.out/'gpu_validation.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return ok


def propagate(args,packet,digest):
    if args.validation is None:raise ValueError('--validation required')
    gate=json.loads(args.validation.read_text())
    ident=identity(args,digest)
    if gate.get('identity')!=ident or not gate.get('gpu_propagation_allowed'):
        raise ValueError('Need matching CPU/GPU PASS for this exact input/grid/dt/code')
    gpu=RealGridPF(packet,args.half_Q,args.nq,args.dt,True,args.device);xp=gpu.xp
    info=environment(xp)
    if any(info[k]!=gate['environment'][k] for k in ('gpu','cupy','driver','runtime')):
        raise ValueError('Hardware/software differs from validation')
    ident.update(end_au=args.end_au,wave_every_au=args.wave_every_au,wave_policy=args.wave_policy,
                 observe_every_au=args.observe_every_au,checkpoint_every_au=args.checkpoint_every_au)
    encoded=json.dumps(ident,sort_keys=True)
    step0=0;checkpoint=args.out/'restart.npz';statuspath=args.out/'status.json'
    if statuspath.exists():
        state=json.loads(statuspath.read_text())
        if state['identity']!=ident:raise ValueError('Run identity differs; preserve old run and select another --out')
        if state['status']=='COMPLETE_NOT_CERTIFIED':return True
        if state['status']=='FAILED_DIAGNOSTIC':raise ValueError('Failed run preserved; use new output directory')
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            if str(z['identity'])!=encoded:raise ValueError('Restart identity mismatch')
            u=xp.asarray(z['psi']);step0=int(z['step']);e0=float(z['initial_energy'])
    else:
        initial,err=initial_grid(packet,gpu.Q)
        if err['backprojection_L2']>1e-8:raise ValueError('Initial transform failed')
        u=xp.asarray(initial);del initial
        e0=gpu.observe(u)['energy']
    total=round(args.end_au/args.dt);every=round(args.observe_every_au/args.dt)
    ck=round(args.checkpoint_every_au/args.dt)
    wave_steps={0,total}
    if args.wave_policy=='events':wave_steps.update(round(t/args.dt) for t in EVENT_AU if t<=args.end_au)
    if args.wave_every_au>0:wave_steps.update(range(0,total+1,round(args.wave_every_au/args.dt)))
    wave_bytes=int(np.prod(gpu.shape))*16
    remaining=sum(s>=step0 and not (args.out/f'wave_{s:07d}.npz').exists() for s in wave_steps)
    disk_guard(args.out,wave_bytes,remaining)
    stopping=[False];handlers={}
    for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    started=time.perf_counter();prop_seconds=0.;obs_seconds=0.
    try:
        for step in range(step0,total+1):
            stop=stopping[0] or (args.max_steps is not None and step-step0>=args.max_steps)
            if step%every==0 or step in wave_steps or step%ck==0 or stop:
                begin=time.perf_counter();row=gpu.observe(u);gpu.sync();obs_seconds+=time.perf_counter()-begin
                bad=failures(row,e0)
                state='FAILED_DIAGNOSTIC' if bad else 'COMPLETE_NOT_CERTIFIED' if step==total else 'INTERRUPTED' if stop else 'RUNNING'
                frame=args.out/f'observable_{step:07d}.npz'
                if not frame.exists():save_npz(frame,time_au=step*args.dt,**row)
                if step%ck==0 or step in wave_steps or state!='RUNNING':
                    disk_guard(args.out,wave_bytes)
                    host=gpu.host(u)
                    wave=args.out/f'wave_{step:07d}.npz'
                    if (step in wave_steps or bad) and not wave.exists():
                        save_npz(wave,psi=host,R=gpu.R,x=gpu.x,Q=gpu.Q,q=gpu.Q/np.sqrt(gpu.omega),
                                 omega=gpu.omega,mass=gpu.mass,g_chi=gpu.g,dx=gpu.dx,dR=gpu.dr,dQ=gpu.dQ,
                                 time_au=step*args.dt,representation='Psi_Q(R,x,Q)',identity=encoded)
                    save_npz(checkpoint,psi=host,step=step,initial_energy=e0,identity=encoded)
                    del host
                summary=dict(status=state,step=step,time_au=step*args.dt,identity=ident,
                    failures=bad,phase6_pass=False,phase7_pass=False,energy_drift=row['energy']-e0,
                    product=row['product'],photon_edge=row['photon_edge'],environment=info,
                    segment_wall_seconds=time.perf_counter()-started,segment_propagation_seconds=prop_seconds,
                    segment_observation_seconds=obs_seconds)
                save_json(statuspath,summary)
                print(f'{state} {step}/{total} t={step*args.dt*.024188843265857:.4f}fs P={row["product"]:.6g} dE={row["energy"]-e0:.3e}',flush=True)
                if state!='RUNNING':return not bad
            gpu.sync();begin=time.perf_counter();u=gpu.step(u);gpu.sync();prop_seconds+=time.perf_counter()-begin
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['check','propagate'],default='check')
    p.add_argument('--device',type=int,default=0);p.add_argument('--dt',type=float,default=.125)
    p.add_argument('--half-Q',dest='half_Q',type=float,default=24.);p.add_argument('--nq',type=int,default=384)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--cpu-steps',type=int,default=8)
    p.add_argument('--end-au',type=float,default=64.);p.add_argument('--max-steps',type=int)
    p.add_argument('--observe-every-au',type=float,default=4.)
    p.add_argument('--checkpoint-every-au',type=float,default=64.)
    p.add_argument('--wave-every-au',type=float,default=0.,help='0: event snapshots only (low disk usage)')
    p.add_argument('--wave-policy',choices=['events','endpoints'],default='events')
    p.add_argument('--validation',type=Path)
    args=p.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1':p.error('Set OPENBLAS_NUM_THREADS=1')
    try:args.out.resolve().relative_to(ROOT.resolve())
    except ValueError:p.error('--out must be inside '+str(ROOT))
    if args.dt<=0 or args.end_au<=0 or args.steps<1 or args.cpu_steps<1 or args.wave_every_au<0:
        p.error('Invalid time configuration')
    if args.max_steps is not None and args.max_steps<1:p.error('max-steps must be positive')
    for t in (args.end_au,args.observe_every_au,args.checkpoint_every_au):
        if t<=0 or abs(round(t/args.dt)*args.dt-t)>1e-10:p.error('Times must be positive integer multiples of dt')
    if args.wave_every_au and abs(round(args.wave_every_au/args.dt)*args.dt-args.wave_every_au)>1e-10:
        p.error('wave cadence must divide dt exactly')
    args.out.mkdir(parents=True,exist_ok=True)
    lock=args.out/'active.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        packet,digest=load_packet(args.input)
        ok=check(args,packet,digest) if args.mode=='check' else propagate(args,packet,digest)
        return 0 if ok else 2
    except Exception:
        message=traceback.format_exc();save_json(args.out/f'error_{time.time_ns()}.json',dict(error=message))
        print(message,flush=True);return 1
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
