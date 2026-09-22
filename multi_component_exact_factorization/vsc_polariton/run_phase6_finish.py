"""Finish an immutable completed recovery campaign. Never propagate full 3D.

Old sources/inputs are verified against the server baseline. New controls and
validation outputs live in a separate directory. No Phase7 imports/actions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
from .run_phase6_completion import source_hashes, regression, run, PREFIX
from .run_phase6_gpu import load_packet, save_json
from .phase6_completion_audit import collect, diagnostic, compare
from .phase6_stationary_compat import stationary


def verify_manifest(manifest):
    for name, expected in manifest.items():
        path=Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Protected source/input mismatch: '+name)


def main():
    p=argparse.ArgumentParser()
    base=Path('results/vsc_polariton/phase6_gpu')
    p.add_argument('--previous',type=Path,default=base/'free_recovery_campaign_v1')
    p.add_argument('--out',type=Path,default=base/'finish_campaign_v1')
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args(); old=args.previous; root=args.out
    root.resolve().relative_to(base.resolve())
    if root.resolve()==old.resolve():raise ValueError('Separate output required')
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1':p.error('Set OPENBLAS_NUM_THREADS=1')
    baseline=json.loads((old/'source_baseline.json').read_text())
    verify_manifest(baseline['sources']);verify_manifest(baseline['inputs'])
    selection=json.loads((old/'free_selection.json').read_text())
    inputs=base/'completion_transfer/inputs'
    free_inputs=base/'free_recovery_transfer_v2/inputs'
    names=[selection['name'],selection['larger_box'],'free_xfine','free_Rfine',
           'free_tfine','barrier_F120','barrier_F160']
    packets=[free_inputs/(selection['name']+'.npz'),
        free_inputs/(selection['larger_box']+'.npz'),
        free_inputs/f"free_L{selection['half_box']:g}_dx0.24.npz",
        free_inputs/f"free_L{selection['half_box']:g}_dx0.3_R550.npz",
        free_inputs/(selection['name']+'.npz'),inputs/'barrier_F120.npz',inputs/'barrier_F160.npz']
    arrays={};reports={}
    for name,packet in zip(names,packets):
        status=json.loads((old/name/'full/status.json').read_text())
        _,digest=load_packet(packet)
        if (status['status']!='COMPLETE_NOT_CERTIFIED' or status['input_sha256']!=digest
                or status['dt']!=(.0625 if name=='free_tfine' else .125)):
            raise RuntimeError('Invalid completed run identity: '+name)
        arrays[name]=collect(old/name/'full');reports[name]=diagnostic(arrays[name])
        if not reports[name]['pass_checks']:raise RuntimeError('Diagnostics fail: '+name)
    comparisons={name:compare(arrays[selection['name']],arrays[name]) for name in names[1:5]}
    comparisons['barrier_Fock']=compare(arrays['barrier_F120'],arrays['barrier_F160'])
    if not all(c['pass_checks'] for c in comparisons.values()):raise RuntimeError('Convergence failed')
    if shutil.disk_usage(base).free<2*1024**3:raise RuntimeError('Reserve 2 GiB; no files deleted')
    root.mkdir(parents=True,exist_ok=True)
    # Include old compact output hashes; exclude large immutable scientific waves.
    def identity():
        paths=[f for f in old.rglob('*') if f.is_file() and
               (f.name.startswith('observable_') or f.suffix=='.json')]
        return dict(sources=source_hashes(),old_outputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                    inputs={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in baseline['inputs']})
    initial=identity();stamp=root/'finish_baseline.json'
    if stamp.exists() and json.loads(stamp.read_text())!=initial:raise RuntimeError('Finish identity changed')
    if not stamp.exists():save_json(stamp,initial)
    lock=root/'batch.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        save_json(root/'reused_validation.json',dict(runs=reports,comparisons=comparisons,selection=selection))
        marker=root/'regression_before_pass.json'
        if not marker.exists():
            regression(root/'regression','before')
            run([sys.executable,'-m','unittest',PREFIX+'tests.test_phase6_finish'],root/'regression/before_finish.log')
            save_json(marker,dict(status='PASS'))
        packet,_=load_packet(packets[0]);(root/'controls').mkdir(exist_ok=True)
        stationary(packet,root/'controls/stationary.json',args.device)
        run([sys.executable,'-m',PREFIX+'phase6_completion_controls','--inputs',str(inputs),
             '--out',str(root/'controls'),'--device',str(args.device)],root/'controls/run.log')
        regression(root/'regression','after')
        run([sys.executable,'-m','unittest',PREFIX+'tests.test_phase6_finish'],root/'regression/after_finish.log')
        unchanged=identity()==initial
        save_json(root/'source_integrity.json',dict(unchanged=unchanged))
        if not unchanged:raise RuntimeError('Protected inputs/sources/results changed')
        from . import phase6_completion_audit as audit
        aliases={'free_F120':selection['name'],'free_F120_dt00625':'free_tfine',
                 'barrier_F120':'barrier_F120','barrier_F160':'barrier_F160'}
        def mapped(path):
            for alias,name in aliases.items():
                if path==root/alias/'full':return arrays[name]
            return collect(path)
        with patch.object(audit,'collect',side_effect=mapped),patch.object(sys,'argv',
                ['audit','--out',str(root),'--inputs',str(inputs)]):audit.main()
        report=json.loads((root/'phase6_completion_validation.json').read_text())
        report['free_representation']='Exact eta=0 N_Fock=1; legacy audit labels only, not BO truncation'
        report['reused_campaign']=str(old)
        save_json(root/'phase6_completion_validation.json',report)
        save_json(root/'finish_status.json',dict(status=report['status'],phase6_pass=False,phase7_allowed=False))
        print('Computed gates complete; send finish_campaign_v1 for scientific review. No Phase7.')
    except BaseException as exc:
        save_json(root/'finish_status.json',dict(status='BLOCKED',reason=str(exc),phase7_allowed=False))
        raise
    finally:lock.unlink()


if __name__=='__main__':main()
