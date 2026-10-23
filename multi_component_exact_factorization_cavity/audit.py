"""Read-only preservation audit and logged legacy regression entry point.

The historical Phase 1--7 manifest is checked, never replaced. Pre-existing
differences are reported separately from changes during this new work.
Only the explicitly supplied new audit directory is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MANIFEST = ROOT/'results/vsc_polariton/phase567_baseline.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def historical_check():
    """Missing or outdated historical manifest is never called PASS."""
    if not HISTORICAL_MANIFEST.is_file():
        return dict(status='NOT_AVAILABLE', path=str(HISTORICAL_MANIFEST))
    data = json.loads(HISTORICAL_MANIFEST.read_text())
    expected = {}
    for group in data.get('manifests', {}).values():
        expected.update(group)
    changed = [name for name, old in expected.items()
               if not (ROOT/name).is_file() or digest(ROOT/name) != old]
    return dict(status='PASS' if not changed else 'MISMATCH',
                baseline_head=data.get('head'), changed=changed,
                checked_count=len(expected), manifest_sha256=digest(HISTORICAL_MANIFEST))


def capture(outdir):
    """Capture current historical tracked source bytes without editing them."""
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out/'preservation_before.json'
    if path.exists():
        raise FileExistsError('Refusing to replace preservation baseline: '+str(path))
    files = [p for p in git('ls-files').splitlines()
             if (ROOT/p).is_file()
             and not p.startswith('multi_component_exact_factorization_cavity/')
             and not Path(p).name.startswith('test_cavity_')]
    result = dict(head=git('rev-parse', 'HEAD'), status=git('status', '--short'),
                  diff_names=git('diff', '--name-status'), diff_check=git('diff', '--check'),
                  source_hashes={p:digest(ROOT/p) for p in files},
                  historical=historical_check())
    path.write_text(json.dumps(result, indent=2)+'\n')
    return path


def verify(snapshot, *, authorized_paths=()):
    """Compare starting bytes; report authorized edits without hiding changes.

    authorized_paths must be explicitly supplied by the caller after user
    authorization. They never make legacy_source_unchanged or the historical
    manifest status true when bytes actually changed.
    """
    path = Path(snapshot)
    data = json.loads(path.read_text())
    changed = [p for p, old in data['source_hashes'].items()
               if not (ROOT/p).is_file() or digest(ROOT/p) != old]
    hist = historical_check()
    previous = set(data['historical'].get('changed', []))
    current = set(hist.get('changed', []))
    authorized = set(authorized_paths)
    unexpected = set(changed)-authorized
    new_historical = current-previous
    manifest_unchanged = (hist.get('manifest_sha256') ==
                          data['historical'].get('manifest_sha256'))
    return dict(legacy_source_unchanged=not changed, changed_during_task=changed,
                historical=hist, new_historical_mismatches=sorted(new_historical),
                authorized_paths=sorted(authorized),
                authorized_changes=sorted(set(changed)&authorized),
                unauthorized_changes=sorted(unexpected),
                unexpected_historical_mismatches=sorted(new_historical-authorized),
                historical_manifest_bytes_unchanged=manifest_unchanged,
                preservation_with_authorized_edits=(not unexpected and
                    not (new_historical-authorized) and manifest_unchanged),
                preexisting_historical_mismatches=sorted(previous),
                diff_check=git('diff', '--check'), head=git('rev-parse', 'HEAD'))


def _run(command, log, env):
    started = time.monotonic()
    with log.open('x') as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    text = log.read_text()
    match = re.search(r'Ran (\d+) tests? in', text)
    tests = int(match.group(1)) if match else None
    passed = process.returncode == 0
    if 'unittest' in command:
        passed = passed and tests is not None and tests > 0
    return dict(command=command, returncode=process.returncode,
                status='PASS' if passed else 'FAIL', tests=tests,
                seconds=time.monotonic()-started, log=str(log))


def run_regressions(outdir, *, legacy_smoke=True, suites=('mcef', 'vsc')):
    """Run unchanged suites, including the VSC namespace-package tests.

    VSC's test directory intentionally has no __init__.py. unittest discovery
    with ``-t .`` rejects it, while discovery without ``-t`` loses relative
    imports. Explicit fully-qualified module names preserve namespace imports
    without modifying the historical directory.
    """
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'regression.json').exists():
        raise FileExistsError('Use a new audit output directory for each run')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLCONFIGDIR=str(out/'mpl_cache'),
               XDG_CACHE_HOME=str(out/'cache'), PYTHONWARNINGS='ignore')
    # Keep legacy regression independent of newly developing extension tests.
    # Adding tests/ to PYTHONPATH matches historical discovery's import names;
    # the repository root remains available for VSC namespace-package imports.
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'tests'), str(ROOT),
                                      env.get('PYTHONPATH', '')])
    mcef_modules = [p.stem for p in sorted((ROOT/'tests').glob('test*.py'))
                    if not p.name.startswith('test_cavity_')]
    vsc_modules = ['.'.join(p.relative_to(ROOT).with_suffix('').parts)
                   for p in sorted((ROOT/'multi_component_exact_factorization/vsc_polariton/tests').glob('test*.py'))]
    commands = {
        'mcef': [sys.executable, '-m', 'unittest', '-q', *mcef_modules],
        'vsc': [sys.executable, '-m', 'unittest', '-q', *vsc_modules],
    }
    results = {}
    for name in suites:
        if (name == 'vsc' and not vsc_modules) or (name == 'mcef' and not mcef_modules):
            results[name] = dict(status='NOT_AVAILABLE', reason='No '+name+' tests present')
            continue
        command = commands[name]
        print('Running legacy '+name+' regression; log='+str(out/(name+'.log')), flush=True)
        results[name] = _run(command, out/(name+'.log'), env)
        print(name+': '+results[name]['status'], flush=True)
    if legacy_smoke and all(v['status'] == 'PASS' for v in results.values()):
        command = [sys.executable, '-m', 'multi_component_exact_factorization.reference',
                   '--outdir', str(out/'legacy_reference_smoke'), '--compact',
                   '--nx', '23', '--nq', '24', '--nR', '12',
                   '--q-min', '-6.1', '--q-max', '6.1', '--R-min', '8', '--R-max', '11',
                   '--q0', '-4', '--R0', '9.5', '--electron-excitation', '1',
                   '--erf-r-qr', '1', '--heavy-trap-alpha', '.03',
                   '--proton-force-constant', '.005', '--heavy-force-constant', '.03',
                   '--fft-workers', '1', '--dt-au', '.001', '--t-final-fs', '.0001',
                   '--save-every', '1']
        results['legacy_smoke'] = _run(command, out/'legacy_smoke.log', env)
    elif legacy_smoke:
        results['legacy_smoke'] = dict(status='NOT_RUN', reason='Legacy regression failure gate')
    result = dict(status='PASS' if all(v['status']=='PASS' for v in results.values()) else 'FAIL',
                  checks=results)
    (out/'regression.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', required=True)
    p.add_argument('--suite', choices=['all', 'mcef', 'vsc'], default='all')
    p.add_argument('--authorized-path', action='append', default=[],
                   help='Explicit user-authorized legacy edit; still reported as changed')
    p.add_argument('--no-legacy-smoke', action='store_true')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture', action='store_true')
    mode.add_argument('--verify', type=Path, metavar='SNAPSHOT_JSON')
    mode.add_argument('--regression', action='store_true')
    args = p.parse_args()
    if args.capture:
        print(capture(args.outdir))
    elif args.verify:
        result = verify(args.verify, authorized_paths=args.authorized_path)
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        path = out/'preservation_after.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        if not result['preservation_with_authorized_edits']:
            raise SystemExit(1)
    else:
        result = run_regressions(args.outdir, legacy_smoke=not args.no_legacy_smoke,
                                 suites=('mcef','vsc') if args.suite=='all' else (args.suite,))
        print(json.dumps(result, indent=2))
        if result['status'] != 'PASS':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
