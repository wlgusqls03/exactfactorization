"""Read-only historical-source regression; outputs only to a NEW audit directory."""
import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import unittest
from .run import atomic_json


def source_manifest():
    """Hash every pre-existing Python/config source; exclude this new package only."""
    paths=[]
    for root in ['multi_component_exact_factorization','multi_component_exact_factorization_discrete',
                 'multi_component_exact_factorization_discrete_gpu','multi_component_exact_factorization_gpu','tests']:
        paths.extend(p for p in Path(root).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.toml','.yaml','.sh'] and 'model_a' not in p.parts)
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(out,baseline=None,suite='all'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=source_manifest();atomic_json(out/'protected_sources.json',manifest)
    git={}
    for args in [['rev-parse','HEAD'],['status','--short'],['diff','--name-status'],['diff','--check']]:
        proc=subprocess.run(['git']+args,capture_output=True,text=True);git[' '.join(args)]=dict(code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
    atomic_json(out/'git.json',git)
    result={}
    # Explicit package module imports work with namespace test directories.
    # Run suites serially; stdout goes to logs, not scientific result directories.
    for name,folder in [('mcef','tests'),('vsc','multi_component_exact_factorization/vsc_polariton/tests')]:
        if suite!='all' and suite!=name:continue
        modules=[str(p.with_suffix('')).replace('/','.') for p in sorted(Path(folder).glob('test*.py'))]
        start=time.perf_counter();records=[]
        for module in modules:
            logpath=out/(module+'.log')
            with logpath.open('w') as log:
                proc=subprocess.run([sys.executable,'-m','unittest','-q',module],stdout=log,stderr=subprocess.STDOUT)
            match=re.search(r'Ran (\d+) tests?',logpath.read_text())
            records.append(dict(module=module,returncode=proc.returncode,count=int(match[1]) if match else None))
        result[name]=dict(records=records,tests=sum(r['count'] or 0 for r in records),
            failed_modules=[r for r in records if r['returncode']],elapsed_seconds=time.perf_counter()-start)
        print(name,result[name],flush=True)
    with (out/'smoke.log').open('w') as log:
        proc=subprocess.run([sys.executable,'-m','multi_component_exact_factorization.vsc_polariton.phase567_smoke'],stdout=log,stderr=subprocess.STDOUT)
    result['smoke_returncode']=proc.returncode
    result['sources_unchanged_during_tests']=source_manifest()==manifest
    if baseline:result['matches_baseline']=manifest==json.loads((Path(baseline)/'protected_sources.json').read_text())
    atomic_json(out/'summary.json',result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--baseline')
    ap.add_argument('--suite',choices=['all','mcef','vsc'],default='all')
    a=ap.parse_args();run(a.out,a.baseline,a.suite)
