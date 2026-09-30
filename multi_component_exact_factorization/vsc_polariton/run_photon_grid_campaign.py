"""Serial 11-GB-GPU campaign: hardware checks -> short pilot -> full comparison.

No original files deleted. No automatic Phase6/7 certification. The first
physical case is selected by --input (recommend existing barrier_F240 packet).
Only base run keeps event waves; refinements keep endpoints + restart.
"""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import numpy as np
from .run_photon_real_grid import ROOT,codes,EVENT_AU
from .run_phase6_gpu import sha,save_json

SETTINGS=(('base_Q384',24.,384,.125),('spacing_Q512',24.,512,.125),
          ('box28_Q448',28.,448,.125),('dt00625_Q384',24.,384,.0625))
OBS_LIMITS=dict(product=2e-4,flux=2e-6,nph=2e-3,mean_R=2e-4)


def call(arguments,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        with subprocess.Popen(arguments,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,bufsize=1) as proc:
            for line in proc.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            if proc.wait():raise RuntimeError('Run stopped; inspect '+str(log))


def rows(directory):
    values={}
    for path in sorted(directory.glob('observable_*.npz')):
        with np.load(path) as z:
            values[float(z['time_au'])]={k:float(z[k]) for k in OBS_LIMITS}
    return values


def compare(root,end,fock_reference=None):
    """Compare common 4-au observable samples, without interpolation."""
    reference=rows(root/SETTINGS[0][0]/'full')
    required=set(np.arange(0,end+1e-9,4.).tolist())|{end}
    if not required.issubset(reference):raise ValueError('Base samples incomplete')
    state=json.loads((root/SETTINGS[0][0]/'full/status.json').read_text())
    if state['status']!='COMPLETE_NOT_CERTIFIED' or state['failures']:raise ValueError('Incomplete/failed base')
    report=dict(status='PASS',scope='Global observable convergence only; NOT MCEF field/Phase7 certification',
                phase7_pass=False,end_au=end,limits=OBS_LIMITS,comparisons={})
    for name,*_ in SETTINGS[1:]:
        status=json.loads((root/name/'full/status.json').read_text())
        if status['status']!='COMPLETE_NOT_CERTIFIED' or status['failures']:raise ValueError('Incomplete/failed '+name)
        other=rows(root/name/'full')
        if not required.issubset(other):raise ValueError('Missing refinement samples '+name)
        errors={k:max(abs(reference[t][k]-other[t][k]) for t in required) for k in OBS_LIMITS}
        ok=all(np.isfinite(v) and v<OBS_LIMITS[k] for k,v in errors.items())
        report['comparisons'][name]=dict(status='PASS' if ok else 'FAIL',max_errors=errors)
        if not ok:report['status']='FAIL'
    if fock_reference is not None:
        other=rows(fock_reference)
        if not required.issubset(other):raise ValueError('Fock reference missing common observable samples')
        errors={k:max(abs(reference[t][k]-other[t][k]) for t in required) for k in OBS_LIMITS}
        ok=all(np.isfinite(v) and v<OBS_LIMITS[k] for k,v in errors.items())
        report['fock_reference']=dict(path=str(fock_reference),max_errors=errors,status='PASS' if ok else 'FAIL')
        if not ok:report['status']='FAIL'
    save_json(root/'comparison.json',report)
    return report


def pack(root,stage):
    """Small diagnostics only; waves/restarts stay on server, never deleted."""
    target=root/f'{stage}_diagnostics.tar.gz'
    paths=sorted(p for p in root.rglob('*') if p.is_file() and
                 (p.suffix in ('.json','.log') or p.name.startswith('observable_')))
    manifest={str(p.relative_to(root)):sha(p) for p in paths}
    if target.exists():return target
    pending=root/f'{stage}_diagnostics.pending.{os.getpid()}.tar.gz'
    with tarfile.open(pending,'x:gz') as tar:
        for p in paths:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
        raw=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('SHA256_MANIFEST.json');info.size=len(raw)
        tar.addfile(info,io.BytesIO(raw))
    pending.replace(target)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'barrier_v1')
    p.add_argument('--stage',choices=['check','pilot','all'],default='pilot')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--fock-reference',type=Path,help='Read-only F240 full/ directory with status and observables')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1':p.error('Set OPENBLAS_NUM_THREADS=1')
    args.out=args.out.resolve();args.out.relative_to(ROOT.resolve())
    if not args.input.is_file() or not args.input.with_suffix('.json').is_file():
        p.error('Existing Phase6 input NPZ AND its SHA256 sidecar JSON are required')
    digest=sha(args.input)
    if digest!=json.loads(args.input.with_suffix('.json').read_text())['sha256']:raise ValueError('Input SHA mismatch')
    if args.fock_reference is not None:
        previous=json.loads((args.fock_reference/'status.json').read_text())
        if previous.get('input_sha256')!=digest or previous.get('failures') or previous.get('time_au')!=1652.:
            raise ValueError('Fock reference must be completed, failure-free, and use the SAME packet')
    with np.load(args.input) as z:
        nr,nx=len(z['R']),len(z['x']);omega=float(z['omega']);coupling=float(z['g_chi'])
    args.out.mkdir(parents=True,exist_ok=True)
    plan=dict(input=str(args.input.resolve()),input_sha256=digest,codes=codes(),
              campaign_sha256=sha(Path(__file__)),settings=SETTINGS,limits=OBS_LIMITS,
              omega=omega,g_chi=coupling,pilot_au=64.,full_au=1652.,
              base_wave_events_au=EVENT_AU,refinement_waves='initial and final only',
              fock_reference=str(args.fock_reference.resolve()) if args.fock_reference else None,
              note='Q-space wavefunction, not physical q-space amplitude; no Phase7 PASS')
    planpath=args.out/'campaign_plan.json'
    # Tuple/list normalization for reliable restart across JSON round-trips.
    plan=json.loads(json.dumps(plan))
    if planpath.exists() and json.loads(planpath.read_text())!=plan:raise ValueError('Plan changed; select a new --out')
    if not planpath.exists():save_json(planpath,plan)
    total=0
    if args.stage!='check':
        total+=sum(3*nr*nx*nq*16 for _,_,nq,_ in SETTINGS)  # pilot endpoints+restart
    if args.stage=='all':
        total+=sum((8 if i==0 else 3)*nr*nx*nq*16 for i,(_,_,nq,_) in enumerate(SETTINGS))
    total+=max(nr*nx*nq*16 for _,_,nq,_ in SETTINGS)+2*1024**3
    existing=sum(f.stat().st_size for f in args.out.rglob('*.npz'))
    remaining=max(2*1024**3,total-existing)
    free=shutil.disk_usage(args.out).free
    print(f'Preflight: forecast additional reserve {remaining/2**30:.2f} GiB; free {free/2**30:.2f} GiB',flush=True)
    if free<remaining:raise OSError('Insufficient disk; historical data untouched')
    if args.preflight_only:return
    lock=args.out/'campaign.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        command=[sys.executable,'-m','multi_component_exact_factorization.vsc_polariton.run_photon_real_grid']
        for name,half,nq,dt in SETTINGS:
            common=['--input',str(args.input),'--device',str(args.device),'--half-Q',str(half),'--nq',str(nq),'--dt',str(dt)]
            out=args.out/'checks'/name
            gate=out/'gpu_validation.json'
            if not gate.exists():call(command+common+['--mode','check','--out',str(out)],out/'run.log')
            checked=json.loads(gate.read_text())
            expected=dict(input_sha256=digest,code_sha256=codes(),dt=dt,half_Q=half,nq=nq,
                          representation='Psi_Q(R,x,Q); integral dR dx dQ')
            if not checked.get('gpu_propagation_allowed') or checked['identity']!=expected:
                raise RuntimeError('Failed/stale hardware check; old results preserved')
        timing={name:json.loads((args.out/'checks'/name/'gpu_validation.json').read_text())['estimated_39p96fs_propagation_hours']
                for name,*_ in SETTINGS}
        save_json(args.out/'time_estimates.json',dict(hours_by_setting=timing,
                  four_full_runs_propagation_hours=sum(timing.values()),excludes='I/O/setup/postprocessing'))
        if args.stage=='check':
            print('SEND:',pack(args.out,'check'));return
        for label,end in [('pilot',64.)]+([('full',1652.)] if args.stage=='all' else []):
            for name,half,nq,dt in SETTINGS:
                out=args.out/label/name/'full';gate=args.out/'checks'/name/'gpu_validation.json'
                call(command+['--input',str(args.input),'--device',str(args.device),'--half-Q',str(half),
                    '--nq',str(nq),'--dt',str(dt),'--mode','propagate','--end-au',str(end),
                    '--wave-policy','events' if label=='full' and name==SETTINGS[0][0] else 'endpoints',
                    '--validation',str(gate),'--out',str(out)],out/'run.log')
                state=json.loads((out/'status.json').read_text())
                if state['status']!='COMPLETE_NOT_CERTIFIED':raise RuntimeError('Interrupted; rerun same campaign command to resume')
            result=compare(args.out/label,end,args.fock_reference)
            print('SEND:',pack(args.out,label),flush=True)
            if result['status']!='PASS':raise RuntimeError(label+' convergence FAIL; not proceeding further')
    finally:lock.unlink()


if __name__=='__main__':main()
