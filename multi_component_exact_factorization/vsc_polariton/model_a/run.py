"""Restartable CPU/GPU Model A TDSE, isolated outputs and honest validation gates."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .model import Config, Propagator, initial, observables
from .factorization import analyze, diagnostics, compact

LIMITS=dict(norm=1e-8,energy_drift=1e-6,R_edge=1e-8,q_edge=1e-8,
            reconstruction_L2=1e-10,ef_route=1e-7,continuity=1e-7)
EVENTS=(0.,250.,500.,750.,1000.,1250.)


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def atomic_npz(path,**values):
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**values);tmp.replace(path)


def identity(c,end,every,fields,stride):
    """Freeze numerical settings AND source bytes for restart, never silently mix."""
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    return dict(config=asdict(c),end_au=end,every_au=every,fields=fields,stride=stride,source=hashes)


def backend_check(c,steps=16):
    """Same grid/initial state CPU vs CUDA, full wave and observable agreement."""
    import cupy as cp
    cpu=Propagator(c);gpu=Propagator(c,cp)
    a=initial(c);b=cp.asarray(a)
    action_error=float(np.linalg.norm(cpu.action(a)-cp.asnumpy(gpu.action(b)))*np.sqrt(cpu.dr*cpu.dq))
    t=time.perf_counter()
    for _ in range(steps):a=cpu.step(a)
    cpu_s=time.perf_counter()-t;cp.cuda.Stream.null.synchronize();t=time.perf_counter()
    for _ in range(steps):b=gpu.step(b)
    cp.cuda.Stream.null.synchronize();gpu_s=time.perf_counter()-t
    err=float(np.linalg.norm(a-cp.asnumpy(b))*np.sqrt(cpu.dr*cpu.dq))
    oa,ob=observables(a,cpu),observables(b,gpu)
    errors={k:abs(oa[k]-ob[k]) for k in oa}
    passed=action_error<1e-9 and err<1e-9 and max(errors.values())<1e-9
    return dict(status='PASS' if passed else 'FAIL',steps=steps,wave_L2=err,action_L2=action_error,
                observables=errors,cpu_seconds=cpu_s,gpu_seconds=gpu_s,
                estimated_propagation_1250au_seconds=gpu_s/steps*1250/c.dt,
                note='Estimate excludes EF extraction, compression, movies. Not spatial convergence.')


def simulate(c,out,end=1250.,every=2.5,backend='cpu',fields=True,stride=2,resume=False):
    """Stream one frame at a time; checkpoint stores wave+history consistently.

    Full waves saved only at six paper events. All EF uses full native grids;
    display maps are stride-decimated and slices are native. No float32.
    """
    out=Path(out);ident=identity(c,end,every,fields,stride)
    if end<0 or every<=0 or stride<1:raise ValueError('Require end>=0, every>0, stride>=1')
    for value in (end/c.dt,every/c.dt):
        if abs(value-round(value))>1e-8:raise ValueError('end/every must be multiples of dt')
    if out.exists() and any(out.iterdir()) and not resume:raise FileExistsError('Output exists; choose new folder or --resume')
    if resume and out.exists() and (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=ident:raise ValueError('Restart settings/source mismatch')
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'identity.json').exists():atomic_json(out/'identity.json',ident)
    (out/'waves').mkdir(exist_ok=True)
    if fields:(out/'fields').mkdir(exist_ok=True)
    frame_count=int(round(end/every))+1
    estimated=frame_count*((c.nr+stride-1)//stride)*((c.nq+stride-1)//stride)*10*8 if fields else 0
    if shutil.disk_usage(out).free<estimated+512*1024**2:raise OSError('Insufficient free space for conservative uncompressed fields + 512 MiB reserve')
    xp=np
    if backend=='gpu':
        import cupy as xp
        check=backend_check(c);atomic_json(out/'backend_check.json',check)
        if check['status']!='PASS':raise RuntimeError('CPU/GPU check failed')
    prop=Propagator(c,xp);hostprop=prop if backend=='cpu' else Propagator(c)
    checkpoint=out/'restart.npz';start=0;rows=[];efrows=[]
    if checkpoint.exists() and resume:
        with np.load(checkpoint,allow_pickle=False) as z:
            u=xp.asarray(z['psi']);start=int(z['step'])
            history=json.loads(str(z['history']));rows=history['rows'];efrows=history['ef']
    else:u=xp.asarray(initial(c))
    last=int(round(end/c.dt));cadence=int(round(every/c.dt));timer=time.perf_counter()
    for step in range(start,last+1):
        if step>start:u=prop.step(u)
        if step%cadence and step!=last:continue
        if rows and step==start:continue
        t=step*c.dt;row=observables(u,prop);row.update(step=step,time_au=t,time_fs=t*.024188843265857)
        rows.append(row)
        event=any(abs(t-e)<c.dt/3 for e in EVENTS)
        if fields or event:
            wave=np.asarray(u) if backend=='cpu' else xp.asnumpy(u)
            f=analyze(wave,hostprop);report=diagnostics(f);report.update(time_au=t,step=step);efrows.append(report)
            if fields:atomic_npz(out/'fields'/f'fields_{step:07d}.npz',**compact(f,stride),time_au=t,time_fs=row['time_fs'])
            if event:atomic_npz(out/'waves'/f'wave_{round(t):04d}.npz',psi=wave,time_au=t)
        # Atomic restart every 25 au and at end, independent of movie frames.
        if step==0 or step==last or abs(t/25-round(t/25))<1e-8:
            host=np.asarray(u) if backend=='cpu' else xp.asnumpy(u)
            atomic_npz(checkpoint,psi=host,step=step,history=json.dumps(dict(rows=rows,ef=efrows)))
            atomic_json(out/'observables.json',rows);atomic_json(out/'ef_diagnostics.json',efrows)
            print(f'{out.name}: {t:.1f}/{end:g} au | norm {row["norm"]:.12f} | S1 {row["P_S1"]:.6f} | n {row["n_ph"]:.6f}',flush=True)
    if not rows:raise RuntimeError('Empty run')
    errors=dict(norm=max(abs(r['norm']-1) for r in rows),
        energy_drift=max(abs(r['energy']-rows[0]['energy']) for r in rows),
        R_edge=max(r['R_edge'] for r in rows),q_edge=max(r['q_edge'] for r in rows),
        reconstruction_L2=max(r['reconstruction_L2'] for r in efrows))
    errors['ef_route']=max(max(v[k] for k in ['epsilon1_route','epsilon2_route','epsilon2_nested','epsilon1_imag','epsilon2_imag']) for r in efrows for v in r['supports'].values())
    errors['continuity']=max(v['continuity'] for r in efrows for v in r['supports'].values())
    passed=all(np.isfinite(v) and v<LIMITS[k] for k,v in errors.items())
    status=dict(status='PASS_INTERNAL' if passed else 'FAIL_INTERNAL',errors=errors,limits=LIMITS,
                complete=rows[-1]['step']==last,elapsed_segment_seconds=time.perf_counter()-timer,
                paper_reproduction_certified=False,field_convergence_certified=False,
                note='Internal conservation/identities only. Compare separate grid, box, dt runs.')
    atomic_json(out/'status.json',status)
    return status


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    ap.add_argument('--nr',type=int,default=500);ap.add_argument('--nq',type=int,default=500)
    ap.add_argument('--dt',type=float,default=.05);ap.add_argument('--g',type=float,default=.01)
    ap.add_argument('--rmin',type=float,default=0);ap.add_argument('--rmax',type=float,default=8)
    ap.add_argument('--qmin',type=float,default=-10);ap.add_argument('--qmax',type=float,default=10)
    ap.add_argument('--end',type=float,default=1250);ap.add_argument('--every',type=float,default=2.5)
    ap.add_argument('--no-fields',action='store_true');ap.add_argument('--stride',type=int,default=2)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    if a.stride<1:ap.error('stride must be positive')
    c=Config(**{k:getattr(a,k) for k in ['nr','nq','dt','g','rmin','rmax','qmin','qmax']})
    print(json.dumps(simulate(c,a.out,a.end,a.every,a.backend,not a.no_fields,a.stride,a.resume),indent=2))


if __name__=='__main__':main()
