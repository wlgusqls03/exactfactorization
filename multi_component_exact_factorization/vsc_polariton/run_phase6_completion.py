"""One serial server command. No deletion, no Phase7, restartable full runs.

Reserve space before work; keep all four runs' scientific waves. Historical
resonant/spatial convergence is reviewed separately, not rerun by this batch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PREFIX = 'multi_component_exact_factorization.vsc_polariton.'


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write('\nCOMMAND '+repr(command)+'\n'); f.flush()
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True); f.write(line); f.flush()
            if process.wait():
                raise RuntimeError('Command failed; preserve diagnostics: '+str(log))


def source_hashes():
    files = set()
    for name in ('multi_component_exact_factorization', 'multi_component_exact_factorization_discrete',
                 'multi_component_exact_factorization_discrete_gpu', 'tests'):
        files.update(Path(name).rglob('*.py'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def regression(root, label):
    run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], root/f'{label}_mcef.log')
    # Local-only Phase1-6 exporter dependencies are intentionally not needed on
    # the GPU server. Full local VSC regression is separate, not claimed here.
    modules = [PREFIX+'tests.'+name for name in
               ('test_phase6_gpu','test_phase6_orthogonal','test_phase6_completion')]
    run([sys.executable, '-m', 'unittest']+modules, root/f'{label}_portable_vsc.log')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, default=Path('results/vsc_polariton/phase6_gpu/completion_transfer/inputs'))
    p.add_argument('--out', type=Path, default=Path('results/vsc_polariton/phase6_gpu/completion_campaign_v1'))
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        p.error('Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1')
    allowed = Path('results/vsc_polariton/phase6_gpu').resolve()
    a.out.resolve().relative_to(allowed)
    # 4*(53 waves+restart), exact known production array sizes + controls/check/temp margin.
    expected = sum(352*160*n*16*54 for n in (120,120,120,160))
    existing = sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file()) if a.out.exists() else 0
    needed = max(0, expected-existing)+3*1024**3
    free = shutil.disk_usage(allowed).free
    print(f'Remaining reserve {needed/1024**3:.2f} GiB; available {free/1024**3:.2f} GiB', flush=True)
    if free < needed:
        raise RuntimeError('Insufficient disk. No old file will be deleted.')
    from .run_phase6_gpu import load_packet
    for case in ('free','barrier','resonant'):
        for nf in (120,160):
            packet, _ = load_packet(a.inputs/f'{case}_F{nf}.npz')
            if packet['psi'].shape != (352,160,nf):
                raise ValueError('Unexpected packet shape')
    a.out.mkdir(parents=True, exist_ok=True)
    def campaign_hashes():
        return dict(sources=source_hashes(), inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(a.inputs.iterdir()) if p.is_file()})
    lock = a.out/'batch.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        baseline = a.out/'source_baseline.json'
        current = campaign_hashes()
        if baseline.exists():
            old = json.loads(baseline.read_text())
            if old != current:
                raise RuntimeError('Source changed since batch start; do not reuse checkpoints blindly')
        else:
            baseline.write_text(json.dumps(current, indent=2))
        marker = a.out/'regression_before_pass.json'
        if not marker.exists():
            regression(a.out/'regression','before')
            marker.write_text('{"status":"PASS"}\n')
        jobs = [('free_F120','free_F120',.125),
                ('free_F120_dt00625','free_F120',.0625),
                ('barrier_F120','barrier_F120',.125),
                ('barrier_F160','barrier_F160',.125)]
        for name, packet_name, dt in jobs:
            out = a.out/name
            check = out/'check/gpu_validation.json'; status = out/'full/status.json'
            if status.exists():
                state = json.loads(status.read_text())['status']
                if state == 'COMPLETE_NOT_CERTIFIED':
                    continue
                if state == 'FAILED_DIAGNOSTIC':
                    raise RuntimeError('Preserved failed run: '+str(status))
            common = ['--input',str(a.inputs/(packet_name+'.npz')),'--device',str(a.device),'--dt',str(dt)]
            if not check.exists():
                run([sys.executable,'-m',PREFIX+'run_phase6_gpu','--mode','check',
                     '--steps','128','--out',str(out/'check')]+common, out/'check/run.log')
            run([sys.executable,'-m',PREFIX+'run_phase6_gpu','--mode','propagate',
                 '--validation',str(check),'--out',str(out/'full')]+common, out/'full/run.log')
            if json.loads(status.read_text())['status'] != 'COMPLETE_NOT_CERTIFIED':
                raise RuntimeError('Not complete: '+str(status))
        run([sys.executable,'-m',PREFIX+'phase6_completion_controls','--inputs',str(a.inputs),
             '--out',str(a.out/'controls'),'--device',str(a.device)],a.out/'controls/run.log')
        regression(a.out/'regression','after')
        integrity = campaign_hashes() == json.loads(baseline.read_text())
        (a.out/'source_integrity.json').write_text(json.dumps(dict(unchanged=integrity)))
        if not integrity:
            raise RuntimeError('Protected source changed during batch')
        run([sys.executable,'-m',PREFIX+'phase6_completion_audit','--out',str(a.out),
             '--inputs',str(a.inputs)],a.out/'audit.log')
        print('Batch complete. Send compact results for final Phase6 certification. No Phase7 executed.')
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
