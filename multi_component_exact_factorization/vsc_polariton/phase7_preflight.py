"""Read-only Phase7 prerequisites; never uses failed-run waves as production."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from .phase567_integrity import verify
from .phase567_smoke import smoke


def main():
    root=Path('results/vsc_polariton/phase7/preflight_v2')
    root.mkdir(parents=True,exist_ok=False)
    baseline={}
    for name,args in [('status',['status','--short']),('head',['rev-parse','HEAD']),
                      ('diff',['diff','--name-status']),('diff_check',['diff','--check'])]:
        baseline[name]=subprocess.check_output(['git']+args,text=True)
    baseline['historical_integrity']=verify()
    finish=Path('results/vsc_polariton/phase6_results/phase6_finish_v3_results/finish_campaign_v3')
    prior=json.loads((finish/'finish_baseline.json').read_text())
    baseline['server_sources_changed']=[n for n,h in prior['sources'].items()
        if not Path(n).is_file() or hashlib.sha256(Path(n).read_bytes()).hexdigest()!=h]
    # The user-requested proton-heavy plotting task changed this file BEFORE
    # Phase7. Record that historical evolution, never silently rewrite a hash.
    changed=baseline['historical_integrity']['changed']
    render='multi_component_exact_factorization/render_final_visualizations.py'
    approved=subprocess.check_output(['git','show','277c45e:'+render])
    known=changed==[render] and Path(render).read_bytes()==approved
    baseline['prior_user_requested_change']=(dict(path=render,commit='277c45e',
        reason='Earlier requested proton-heavy plots; not a Phase7 edit') if known else None)
    (root/'baseline.json').write_text(json.dumps(baseline,indent=2))
    if (changed and not known) or baseline['server_sources_changed']:
        raise RuntimeError('Protected manifest mismatch; no calculations allowed')
    modules=['multi_component_exact_factorization.vsc_polariton.tests.'+p.stem
        for p in sorted(Path('multi_component_exact_factorization/vsc_polariton/tests').glob('test_*.py'))]
    for name,command in [('mcef',[sys.executable,'-m','unittest','discover','-s','tests']),
                         ('vsc',[sys.executable,'-m','unittest']+modules)]:
        with (root/(name+'.log')).open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(name+' regression failed')
    (root/'smoke.json').write_text(json.dumps(smoke(),indent=2))
    waves={}
    for p in Path('results/vsc_polariton').rglob('wave_*.npz'):
        status=p.parent/'status.json'
        waves.setdefault(str(p.parent),dict(count=0,status=json.loads(status.read_text()).get('status') if status.exists() else 'NO_STATUS'))['count']+=1
    available=any(v['status']=='COMPLETE_NOT_CERTIFIED' for v in waves.values())
    result=dict(status='INPUT_REVIEW_REQUIRED' if available else 'BLOCKED_MISSING_VALIDATED_WAVEFUNCTIONS',regressions='PASS',
        available_wave_groups=waves,
        reason='Local compact completed archives lack full Psi. Failed historical pilot waves are not production inputs.',
        phase7_started=False,phase7_pass=False)
    (root/'preflight.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
