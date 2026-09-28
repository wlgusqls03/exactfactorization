"""Serial server F200 refinement; uses unchanged Phase6 GPU engine.

No deletion. Existing completed/failed runs stay immutable. Interrupted runs
resume through the engine checkpoint. This driver never certifies Phase7.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .run_phase6_gpu import load_packet, sha, code_id
from .phase7_f200_compact_gpu import retain_wave

PREFIX='multi_component_exact_factorization.vsc_polariton.'
WAVE_BYTES=352*160*200*16
WAVE_COUNT=sum(retain_wave(s) for s in range(0,13217,256))+1 # final off cadence
RESERVE_BYTES=2*(WAVE_COUNT+1)*WAVE_BYTES+3*1024**3


def command(args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as output:
        with subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
            for line in proc.stdout:
                print(line,end='',flush=True);output.write(line);output.flush()
            if proc.wait():raise RuntimeError('Failed; preserve '+str(log))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parents[2]/'results/vsc_polariton/phase6_gpu'
    p.add_argument('--inputs',type=Path,default=base/'f200_server_transfer_v1/inputs')
    p.add_argument('--out',type=Path,default=base/'phase7_f200_campaign_v1')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();a.out.resolve().relative_to(base.resolve())
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1':p.error('Set OPENBLAS_NUM_THREADS=1')
    identities={}
    for case in ('resonant','barrier'):
        packet,digest=load_packet(a.inputs/f'{case}_F200.npz')
        if packet['psi'].shape!=(352,160,200) or float(packet['dt'])!=.125:raise ValueError('Wrong F200 input')
        meta=json.loads(str(packet['metadata']))
        if meta['orthogonal_rotation_validation']['status']!='PASS':raise ValueError('Orthogonal validation missing')
        identities[case]=digest
        del packet
    existing=sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file()) if a.out.exists() else 0
    remaining=max(0,RESERVE_BYTES-existing)
    free=shutil.disk_usage(base).free
    print(f'Required remaining reserve {remaining/1024**3:.2f} GiB; free {free/1024**3:.2f} GiB',flush=True)
    if free<remaining:raise RuntimeError('Insufficient disk; no old file will be deleted')
    if a.preflight_only:return
    a.out.mkdir(parents=True,exist_ok=True)
    identity=dict(inputs=identities,engine=code_id(),driver=sha(Path(__file__)),storage=sha(Path(__file__).with_name('phase7_f200_compact_gpu.py')))
    baseline=a.out/'campaign_identity.json'
    if baseline.exists():
        if json.loads(baseline.read_text())!=identity:raise RuntimeError('Campaign identity changed')
    else:baseline.write_text(json.dumps(identity,indent=2))
    lock=a.out/'batch.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        for case in ('resonant','barrier'):
            root=a.out/case;status=root/'full/status.json';gate=root/'check/gpu_validation.json'
            if status.exists():
                s=json.loads(status.read_text())
                if s['input_sha256']!=identities[case]:raise RuntimeError('Input identity mismatch')
                if s['status']=='COMPLETE_NOT_CERTIFIED':continue
                if s['status']=='FAILED_DIAGNOSTIC':raise RuntimeError('Preserve failed diagnostic '+str(status))
            common=['--input',str(a.inputs/f'{case}_F200.npz'),'--device',str(a.device),'--dt','0.125']
            if not gate.exists():
                command([sys.executable,'-m',PREFIX+'run_phase6_gpu','--mode','check','--steps','128','--out',str(root/'check')]+common,root/'check/run.log')
            command([sys.executable,'-m',PREFIX+'phase7_f200_compact_gpu','--mode','propagate','--validation',str(gate),'--out',str(root/'full')]+common,root/'full/run.log')
            if json.loads(status.read_text())['status']!='COMPLETE_NOT_CERTIFIED':raise RuntimeError('Interrupted; rerun same command to resume')
        (a.out/'batch_complete.json').write_text(json.dumps(dict(status='COMPLETE_NOT_CERTIFIED',phase7_pass=False,inputs=identities),indent=2))
        print('Both F200 runs complete. NOT Phase7 PASS. Keep all wave/restart/observable files.',flush=True)
        from .phase7_pack_f200_results import pack
        if not (a.out/'phase7_f200_results.tar.gz').exists():pack(a.out)
    finally:lock.unlink()


if __name__=='__main__':main()
