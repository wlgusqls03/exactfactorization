"""Immutable input/source baseline and prerequisite regression for Phase7 pilot."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from .phase567_integrity import verify
from .phase567_smoke import smoke

INPUT=Path('results/vsc_polariton/phase7/results/phase7_pilot_inputs')
OUTPUT=Path('results/vsc_polariton/phase7/pilot_v1')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    out=OUTPUT/'baseline';out.mkdir(parents=True,exist_ok=False)
    git={key:subprocess.check_output(['git']+args,text=True) for key,args in
         [('head',['rev-parse','HEAD']),('status',['status','--short']),
          ('diff',['diff','--name-status']),('check',['diff','--check'])]}
    historical=verify()
    render='multi_component_exact_factorization/render_final_visualizations.py'
    expected=subprocess.check_output(['git','show','277c45e:'+render])
    if historical['changed'] not in ([],[render]) or Path(render).read_bytes()!=expected:
        raise RuntimeError('Unexpected historical source/artifact change')
    manifest=json.loads((INPUT/'phase7_pilot_manifest.json').read_text())
    for name,sha in manifest['files'].items():
        path=INPUT/name
        path.resolve().relative_to(INPUT.resolve())
        if digest(path)!=sha:raise RuntimeError('Input SHA mismatch '+name)
    sources={str(p):digest(p) for parent in ('multi_component_exact_factorization',
        'multi_component_exact_factorization_discrete','multi_component_exact_factorization_discrete_gpu','tests')
        for p in Path(parent).rglob('*.py')}
    baseline=dict(git=git,historical=historical,inputs=manifest['files'],sources=sources,
        historical_exception='User-requested proton-heavy plotting commit 277c45e, before Phase7',
        phase6_scope='User-approved three core cases 0--1652 au; not a 200 fs or thermal-rate claim')
    (out/'manifest.json').write_text(json.dumps(baseline,indent=2))
    modules=['multi_component_exact_factorization.vsc_polariton.tests.'+p.stem for p in
        sorted(Path('multi_component_exact_factorization/vsc_polariton/tests').glob('test_*.py'))]
    counts={}
    for name,cmd in [('mcef',[sys.executable,'-m','unittest','discover','-s','tests']),
                     ('vsc',[sys.executable,'-m','unittest']+modules)]:
        with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(name+' regression FAIL')
        counts[name]=int(re.search(r'Ran (\d+) tests',(out/(name+'.log')).read_text()).group(1))
    (out/'validation.json').write_text(json.dumps(dict(status='PASS',counts=counts,smoke=smoke()),indent=2))
    print('Phase7 baseline PASS',counts,flush=True)


if __name__=='__main__':main()
