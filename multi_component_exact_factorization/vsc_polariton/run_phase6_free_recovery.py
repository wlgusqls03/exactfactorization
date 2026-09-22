"""Bounded exact-vacuum free convergence, then remaining completion jobs.

NF=1 only for eta=0. Full electronic x coordinate remains explicit.
No historical file is overwritten; no Phase7 is imported.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from .run_phase6_completion import run, regression, source_hashes, PREFIX
from .run_phase6_gpu import load_packet, save_json
from .phase6_completion_audit import collect, diagnostic, compare

BOXES = (28.8, 36., 48., 72.)


def boundary_only(result):
    """Only the predeclared box gate permits trying the next box."""
    if result['status']!='FAILED_DIAGNOSTIC' or set(result['failures'])!={'electron_edge'}:
        return False
    value=result['failures']['electron_edge'].get('value',float('nan'))
    return math.isfinite(value) and value>=1e-8


def job(packet, out, dt, device):
    """Existing GPU hardware check + immutable/restartable full-time driver."""
    _, digest = load_packet(packet)
    status = out/'full/status.json'; check = out/'check/gpu_validation.json'
    if status.exists():
        result = json.loads(status.read_text())
        if result['input_sha256'] != digest or result['dt'] != dt:
            raise RuntimeError('Existing job identity differs')
        if result['status'] in ('COMPLETE_NOT_CERTIFIED','FAILED_DIAGNOSTIC'):
            return result
    common = ['--input',str(packet),'--dt',str(dt),'--device',str(device)]
    if not check.exists():
        run([sys.executable,'-m',PREFIX+'run_phase6_gpu','--mode','check',
             '--steps','128','--out',str(out/'check')]+common,out/'check/run.log')
    try:
        run([sys.executable,'-m',PREFIX+'run_phase6_gpu','--mode','propagate',
             '--validation',str(check),'--out',str(out/'full')]+common,out/'full/run.log')
    except RuntimeError:
        if not status.exists() or json.loads(status.read_text())['status'] != 'FAILED_DIAGNOSTIC':
            raise
    return json.loads(status.read_text())


def required(packet, out, dt, device):
    result = job(packet,out,dt,device)
    if result['status'] != 'COMPLETE_NOT_CERTIFIED':
        raise RuntimeError('Required run did not complete: '+str(out))
    arrays = collect(out/'full')
    if not diagnostic(arrays)['pass_checks']:
        raise RuntimeError('Full interval diagnostic failure: '+str(out))
    return arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,default=Path('results/vsc_polariton/phase6_gpu/free_recovery_transfer_v2/inputs'))
    parser.add_argument('--completion-inputs',type=Path,default=Path('results/vsc_polariton/phase6_gpu/completion_transfer/inputs'))
    parser.add_argument('--out',type=Path,default=Path('results/vsc_polariton/phase6_gpu/free_recovery_campaign_v1'))
    parser.add_argument('--device',type=int,default=0)
    args = parser.parse_args(); root = args.out
    root.resolve().relative_to(Path('results/vsc_polariton/phase6_gpu').resolve())
    if os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        parser.error('Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1')
    # Largest planned free packets plus two large barrier archives and margin.
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    needed = max(0,18*1024**3-used)+2*1024**3
    if shutil.disk_usage(root.parent).free < needed:
        raise RuntimeError('Need about20 GiB free for a new campaign; no deletion performed')
    for half in BOXES:
        for dx in (.3,.24):
            p,_ = load_packet(args.inputs/f'free_L{half:g}_dx{dx:g}.npz')
            if float(p['g_chi']) != 0 or p['psi'].shape[-1] != 1:
                raise RuntimeError('Not exact eta=0 vacuum sector')
        load_packet(args.inputs/f'free_L{half:g}_dx0.3_R550.npz')
    for case in ('free','barrier','resonant'):
        for nf in (120,160):load_packet(args.completion_inputs/f'{case}_F{nf}.npz')
    root.mkdir(parents=True,exist_ok=True)
    def identities():
        return dict(sources=source_hashes(),inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for folder in (args.inputs,args.completion_inputs) for p in sorted(folder.iterdir()) if p.is_file()})
    lock=root/'batch.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        before=root/'source_baseline.json'
        current=identities()
        if before.exists() and json.loads(before.read_text())!=current:
            raise RuntimeError('Source changed: preserve old campaign')
        if not before.exists():save_json(before,current)
        marker=root/'regression_before_pass.json'
        if not marker.exists():
            print('REGRESSION: /tmp figures are disposable test outputs; scientific results stay under '+str(root),flush=True)
            regression(root/'regression','before')
            run([sys.executable,'-m','unittest',PREFIX+'tests.test_phase6_free_recovery'],root/'regression/before_recovery.log')
            save_json(marker,dict(status='PASS'))
        previous=None;chosen=None;box_reports={}
        for half in BOXES:
            name=f'free_L{half:g}_dx0.3';out=root/name
            result=job(args.inputs/(name+'.npz'),out,.125,args.device)
            box_reports[name]=result
            save_json(root/'free_box_progress.json',box_reports)
            if boundary_only(result):
                previous=None
                continue  # predeclared numerical box refinement, not tolerance tuning
            if result['status']!='COMPLETE_NOT_CERTIFIED':
                raise RuntimeError('Non-box failure, requires diagnosis: '+name)
            arrays=collect(out/'full')
            if not diagnostic(arrays)['pass_checks']:
                raise RuntimeError('Full diagnostics fail: '+name)
            if previous is not None:
                difference=compare(previous[2],arrays)
                if difference['pass_checks']:
                    chosen=previous
                    save_json(root/'free_selection.json',dict(name=previous[0],half_box=previous[1],
                              larger_box=name,box_comparison=difference))
                    break
            previous=(name,half,arrays)
        if chosen is None:
            raise RuntimeError('No certified adjacent box pair up to72 a0; STOP, no Phase7')
        name,half,base=chosen
        fine=required(args.inputs/f'free_L{half:g}_dx0.24.npz',root/'free_xfine',.125,args.device)
        nuclear=required(args.inputs/f'free_L{half:g}_dx0.3_R550.npz',root/'free_Rfine',.125,args.device)
        temporal=required(args.inputs/(name+'.npz'),root/'free_tfine',.0625,args.device)
        checks=dict(x_spacing=compare(base,fine),R_spacing=compare(base,nuclear),time_step=compare(base,temporal))
        save_json(root/'free_refinement.json',checks)
        if not all(v['pass_checks'] for v in checks.values()):
            raise RuntimeError('Free refinement failed; do not promote')
        for nf in (120,160):
            required(args.completion_inputs/f'barrier_F{nf}.npz',root/f'barrier_F{nf}',.125,args.device)
        # Stationary test uses the selected free production box, not old small box.
        from .phase6_completion_controls import stationary
        p,_=load_packet(args.inputs/(name+'.npz'));(root/'controls').mkdir(exist_ok=True)
        stationary(p,root/'controls/stationary.json',args.device)
        run([sys.executable,'-m',PREFIX+'phase6_completion_controls','--inputs',str(args.completion_inputs),
             '--out',str(root/'controls'),'--device',str(args.device)],root/'controls/run.log')
        regression(root/'regression','after')
        run([sys.executable,'-m','unittest',PREFIX+'tests.test_phase6_free_recovery'],root/'regression/after_recovery.log')
        unchanged=identities()==json.loads(before.read_text())
        save_json(root/'source_integrity.json',dict(unchanged=unchanged))
        if not unchanged:raise RuntimeError('Source integrity failed')
        # Audit adapter substitutes the exact free-sector path while retaining
        # existing compact-control comparisons. Relabel legacy keys explicitly.
        from . import phase6_completion_audit as audit
        original_collect=audit.collect
        def mapped(path):
            if path==root/'free_F120/full':return base
            if path==root/'free_F120_dt00625/full':return temporal
            return original_collect(path)
        old_argv=sys.argv
        try:
            audit.collect=mapped
            sys.argv=['audit','--out',str(root),'--inputs',str(args.completion_inputs)]
            audit.main()
        finally:
            audit.collect=original_collect;sys.argv=old_argv
        path=root/'phase6_completion_validation.json';report=json.loads(path.read_text())
        report['free_representation']='Exact eta=0 n=0 sector, NOT F120 propagation; legacy audit aliases only'
        report['runs']['free_N1']=report['runs'].pop('free_F120')
        report['runs']['free_N1_dt00625']=report['runs'].pop('free_F120_dt00625')
        report['free_selection']=json.loads((root/'free_selection.json').read_text())
        report['free_refinement']=checks
        save_json(path,report)
        save_json(root/'recovery_status.json',dict(status=report['status'],phase6_pass=False,
                                                  phase7_allowed=False,reason='Final scientific review pending'))
        print('Return compact free_recovery_campaign_v1 results. Final review pending; no Phase7.')
    except Exception as exc:
        save_json(root/'recovery_status.json',dict(status='BLOCKED',reason=str(exc),phase7_allowed=False))
        raise
    finally:lock.unlink()


if __name__=='__main__':main()
