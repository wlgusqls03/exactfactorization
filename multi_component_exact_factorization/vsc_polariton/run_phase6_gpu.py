"""Standalone server CPU/GPU comparison and gated GPU propagation.

Only depends on phase6_gpu_backend.py, NumPy, SciPy and optional CuPy.
No historical simulation, Phase7, or untracked VSC dependencies are imported.
"""
import argparse
import hashlib
import json
import os
import platform
import signal
import time
import traceback
from pathlib import Path
import numpy as np
from .phase6_gpu_backend import PFBackend


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save_json(path,data):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def save_npz(path,**data):
    with path.with_suffix('.pending').open('wb') as f:
        np.savez(f,**data);f.flush();os.fsync(f.fileno())
    path.with_suffix('.pending').replace(path)


def load_packet(path):
    digest=sha(path)
    sidecar=json.loads(path.with_suffix('.json').read_text())
    if digest!=sidecar['sha256']:raise ValueError('Input SHA256 mismatch: transfer NPZ and JSON again')
    with np.load(path,allow_pickle=False) as a:packet={k:a[k].copy() for k in a.files}
    meta=json.loads(str(packet['metadata']))
    if meta.get('format_version')!=1 or meta['legacy_adapter_validation']['status']!='PASS':
        raise ValueError('Packet has no passing historical-CPU adapter validation')
    return packet,digest


def code_id():
    return {p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('phase6_gpu_backend.py'))}


def environment(cp):
    prop=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name=prop['name'];name=name.decode() if isinstance(name,bytes) else name
    return dict(python=platform.python_version(),numpy=np.__version__,cupy=cp.__version__,
        gpu=name,device=int(cp.cuda.Device().id),driver=int(cp.cuda.runtime.driverGetVersion()),
        runtime=int(cp.cuda.runtime.runtimeGetVersion()),total_memory=int(prop['totalGlobalMem']),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),precision='complex128/float64')


def absolute_failures(row,e0):
    checks=dict(norm=(abs(row['norm']-1),1e-9),energy=(abs(row['energy']-e0),1e-6),
        electron_edge=(row['electron_edge'],1e-8),R_edge=(row['edge'],1e-8),
        Fock_tail=(row['top'],1e-8),continuity=(abs(row['continuity_error']),2e-6))
    return {k:dict(value=float(v),limit=lim) for k,(v,lim) in checks.items() if not np.isfinite(v) or v>=lim}


def compare(packet,digest,dt,steps,device,out):
    """Compare same initial Psi at every 4-au sample; synchronized timings exclude warmup."""
    cpu=PFBackend(packet,dt)
    gpu=PFBackend(packet,dt,gpu=True,device=device);cp=gpu.xp
    info=environment(cp)
    u=packet['psi'].copy();begin=time.perf_counter();v=cp.asarray(u);gpu.sync()
    transfer=time.perf_counter()-begin
    begin=time.perf_counter();gpu.step(v.copy());gpu.sync();warmup=time.perf_counter()-begin
    # No warmup evolution is included in the compared trajectory.
    action_error=float(np.linalg.norm(cpu.action(u)-gpu.host(gpu.action(v)))*np.sqrt(cpu.volume))
    timings=dict(cpu_propagation=0.,gpu_propagation=0.,cpu_observables=0.,gpu_observables=0.)
    records=[];bad=[];every=round(4/dt);start=time.perf_counter()
    limits=dict(wave_L2=1e-8,overlap_error=1e-9,norm=1e-9,energy=1e-9,
                product=1e-9,flux=1e-10,P_exc=1e-9,nph=1e-8,mean_R=1e-9)
    maxima={k:0. for k in limits};e0=None
    for step in range(steps+1):
        if step%every==0 or step==steps:
            begin=time.perf_counter();a=cpu.observe(u);timings['cpu_observables']+=time.perf_counter()-begin
            gpu.sync();begin=time.perf_counter();b=gpu.observe(v);gpu.sync()
            timings['gpu_observables']+=time.perf_counter()-begin
            if e0 is None:e0=a['energy']
            host=gpu.host(v)
            err=dict(wave_L2=float(np.linalg.norm(u-host)*np.sqrt(cpu.volume)),
                     overlap_error=float(abs(1-abs(np.vdot(u,host))*cpu.volume/np.sqrt(a['norm']*b['norm']))))
            err.update({k:float(abs(a[k]-b[k])) for k in limits if k not in err})
            for key in limits:maxima[key]=max(maxima[key],err[key]) if np.isfinite(err[key]) else float('inf')
            failures=dict(cpu=absolute_failures(a,e0),gpu=absolute_failures(b,e0))
            if failures['cpu'] or failures['gpu']:bad.append(dict(step=step,failures=failures))
            records.append(dict(step=step,time_au=step*dt,errors=err,
                cpu={k:float(a[k]) for k in ('norm','energy','product','flux','P_exc','nph','mean_R')},
                gpu={k:float(b[k]) for k in ('norm','energy','product','flux','P_exc','nph','mean_R')}))
            print(f'comparison {step}/{steps}: L2={err["wave_L2"]:.3e}, dE={err["energy"]:.3e}',flush=True)
        if step==steps:break
        begin=time.perf_counter();u=cpu.step(u);timings['cpu_propagation']+=time.perf_counter()-begin
        gpu.sync();begin=time.perf_counter();v=gpu.step(v);gpu.sync()
        timings['gpu_propagation']+=time.perf_counter()-begin
    agree=all(maxima[k]<limits[k] for k in limits) and np.isfinite(action_error) and action_error<1e-10
    gpu_seconds=timings['gpu_propagation']+timings['gpu_observables']
    cpu_seconds=timings['cpu_propagation']+timings['cpu_observables']
    result=dict(status='PASS' if agree and not bad else 'FAIL',gpu_propagation_allowed=agree and not bad and steps>=128,
        phase6_pass=False,input_sha256=digest,code_sha256=code_id(),dt=dt,steps=steps,
        environment=info,shape=list(cpu.shape),backend_agreement=agree,action_L2_error=action_error,
        errors=maxima,limits=limits,absolute_failures=bad,records=records,timings=timings,
        initial_transfer_seconds=transfer,gpu_warmup_seconds=warmup,wall_seconds=time.perf_counter()-start,
        propagation_speedup=timings['cpu_propagation']/timings['gpu_propagation'],
        compute_plus_observable_speedup=cpu_seconds/gpu_seconds,
        estimated_gpu_39p96fs_hours=gpu_seconds/steps*(1652/dt)/3600,
        estimated_cpu_39p96fs_hours=cpu_seconds/steps*(1652/dt)/3600,
        gpu_pool_reserved_bytes=int(cp.get_default_memory_pool().total_bytes()),
        note='Short hardware equivalence only. Time estimates exclude checkpoint I/O, basis/setup, '
             'and full-interval convergence. Reserved pool is not peak total VRAM.')
    save_json(out/'gpu_validation.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def propagate(packet,digest,dt,device,out,validation,max_steps):
    """Validated hardware path, checkpointed 1652 au; NEVER Phase6 scientific certification."""
    gate=json.loads(validation.read_text())
    gpu=PFBackend(packet,dt,gpu=True,device=device);info=environment(gpu.xp)
    if (gate['status']!='PASS' or not gate['gpu_propagation_allowed'] or gate['input_sha256']!=digest
        or gate['code_sha256']!=code_id() or gate['dt']!=dt or gate['steps']<128
        or any(gate['environment'][k]!=info[k] for k in ('gpu','driver','runtime','cupy'))):
        raise RuntimeError('Need matching >=128-step CPU/GPU PASS for this input, dt, code and hardware')
    checkpoint=out/'restart.npz';step0=0;u=gpu.xp.asarray(packet['psi']);e0=None
    if checkpoint.exists():
        with np.load(checkpoint) as a:
            if str(a['identity'])!=json.dumps([digest,dt,code_id()],sort_keys=True):
                raise RuntimeError('Restart identity mismatch')
            u=gpu.xp.asarray(a['psi']);step0=int(a['step']);e0=float(a['initial_energy'])
    statuspath=out/'status.json'
    if statuspath.exists() and json.loads(statuspath.read_text())['status'] in ('FAILED_DIAGNOSTIC','COMPLETE_NOT_CERTIFIED'):
        raise RuntimeError('Terminal run is immutable; use a different directory for a new run')
    stopping=[False];handlers={}
    for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    try:
        total=round(1652/dt);every=round(4/dt);waves=round(32/dt)
        for step in range(step0,total+1):
            stop=stopping[0] or (max_steps is not None and step-step0>=max_steps)
            if step%every==0 or step==total or stop:
                row=gpu.observe(u)
                if e0 is None:e0=row['energy']
                failed=absolute_failures(row,e0)
                state='FAILED_DIAGNOSTIC' if failed else 'COMPLETE_NOT_CERTIFIED' if step==total else 'INTERRUPTED' if stop else 'RUNNING'
                frame=out/f'observable_{step:07d}.npz'
                if not frame.exists():save_npz(frame,time_au=step*dt,**row)
                host=gpu.host(u)
                wave=out/f'wave_{step:07d}.npz'
                if (step%waves==0 or step==total or failed) and not wave.exists():
                    save_npz(wave,psi=host,time_au=step*dt,R=packet['R'],x=packet['x'])
                save_npz(checkpoint,psi=host,step=step,initial_energy=e0,
                         identity=json.dumps([digest,dt,code_id()],sort_keys=True))
                status=dict(status=state,step=step,time_au=step*dt,failures=failed,phase6_pass=False,
                            input_sha256=digest,dt=dt,environment=info,code_sha256=code_id())
                save_json(statuspath,status);print(json.dumps(status),flush=True)
                if state!='RUNNING':return status
            u=gpu.step(u)
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['check','propagate'],default='check')
    p.add_argument('--device',type=int,default=0);p.add_argument('--dt',type=float)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--validation',type=Path)
    p.add_argument('--max-steps',type=int)
    args=p.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1':p.error('Set OPENBLAS_NUM_THREADS=1')
    allowed=Path(__file__).resolve().parents[2]/'results/vsc_polariton/phase6_gpu'
    try:args.out.resolve().relative_to(allowed.resolve())
    except ValueError:p.error('--out must be below results/vsc_polariton/phase6_gpu (historical outputs are protected)')
    args.out.mkdir(parents=True,exist_ok=True)
    lock=args.out/'active.lock';fd=os.open(str(lock),os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if args.mode=='check' and (args.out/'gpu_validation.json').exists():
            raise FileExistsError('Preserve old validation; choose a new output directory')
        packet,digest=load_packet(args.input);dt=float(packet['dt']) if args.dt is None else args.dt
        if dt<=0 or abs(round(4/dt)*dt-4)>1e-12 or args.steps<1:
            raise ValueError('Positive dt must divide 4 au, and steps must be positive')
        if args.mode=='check':
            result=compare(packet,digest,dt,args.steps,args.device,args.out)
        else:
            if args.validation is None:raise ValueError('--validation is required')
            result=propagate(packet,digest,dt,args.device,args.out,args.validation,args.max_steps)
        return 0 if result['status'] not in ('FAIL','FAILED_DIAGNOSTIC') else 2
    except Exception:
        message=traceback.format_exc()
        path=args.out/f'error_{time.time_ns()}.json'
        save_json(path,dict(status='ERROR',traceback=message,phase6_pass=False))
        print(message,flush=True);return 1
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
