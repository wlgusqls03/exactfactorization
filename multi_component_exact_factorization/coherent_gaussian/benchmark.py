"""Matched instantaneous RHS timings, plus saved-run error/cost audit.

RHS timings do not equal whole-run speedups: timestep, rank regularization,
eigensolver work, sampling error and Fock truncation must also be compared.
"""
import argparse
import json
import platform
from pathlib import Path
import time
import numpy as np
from ..pen_perspectives_2026.resonance import paper_config
from ..pen_ctmqc_2026.core import Engine, Settings
from .compare import GaussianPOL


def timing(fn,repeats):
    fn();samples=[];cpu=[]
    for _ in range(repeats):
        t=time.perf_counter();tc=time.process_time();fn();samples.append(time.perf_counter()-t);cpu.append(time.process_time()-tc)
    return dict(median_seconds=float(np.median(samples)),min_seconds=min(samples),
                repeats=repeats,median_cpu_seconds=float(np.median(cpu)))


def benchmark(out,ntraj=16,nf=12,components=(1,2,4,8),repeats=12):
    c=paper_config('nonadiabatic','on',nf=nf,nr=64)
    rows=[]
    for perspective in ('el','pol'):
        e=Engine(c,Settings(perspective=perspective,ntraj=ntraj,initial='bare'))
        s=e.initial();r=timing(lambda:e.evaluate(s,s['anchor']),repeats)
        rows.append(dict(method=perspective,ntraj=ntraj,nf=nf,**r))
    for K in components:
        e=GaussianPOL(c,Settings(perspective='pol',ntraj=ntraj,initial='bare'),K)
        y=e.initial();r=timing(lambda:e.rhs(0,y),repeats)
        rows.append(dict(method='gaussian',K=K,ntraj=ntraj,nf=nf,
            state_bytes=y.nbytes,**r))
    result=dict(platform=platform.platform(),numpy=np.__version__,rows=rows,
        warning='Initial-state RHS microbenchmark, NOT a complete-trajectory speed comparison.')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result


def audit_comparison(root):
    root=Path(root)
    qd=json.loads((root/'qd/metrics.json').read_text())
    qt=np.array([x['t_fs'] for x in qd]);result={}
    # All methods use ONE common interval, so a failed 18-fs calculation is
    # never ranked against another method's full 40-fs error.
    common_end=qt[-1]
    for name in ('el','pol','gaussian'):
        p=root/name/'metrics.json'
        if p.exists():
            rr=json.loads(p.read_text())
            if rr:common_end=min(common_end,rr[-1]['t_fs'])
    for name in ('el','pol','gaussian'):
        p=root/name/'metrics.json'
        if not p.exists():continue
        rows=json.loads(p.read_text())
        if not rows:continue
        old_t=np.array([x['t_fs'] for x in rows])
        t=np.linspace(0,common_end,max(2,int(np.ceil(common_end/.05))+1));item={}
        for key in ('photon_number','bo_population','product_R_ge_4'):
            value=lambda r:r[key][1] if key=='bo_population' else r[key]
            a=np.interp(t,old_t,[value(r) for r in rows])
            b=np.interp(t,qt,[value(r) for r in qd])
            item[key]=dict(mae=float(np.mean(abs(a-b))),max_error=float(max(abs(a-b))))
        item['through_fs']=float(t[-1]);item['warning']='No claim beyond common saved interval; finite Ntraj and QD errors remain.'
        result[name]=item
    (root/'accuracy_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def audit_fixed(root):
    """Label limits in earlier exploratory runs, without rewriting their data."""
    root=Path(root);rows=json.loads((root/'metrics.json').read_text());reasons=[]
    first=None
    for r in rows:
        issue=[]
        if abs(r['reference_projection_missing_norm'])>1e-3:issue.append('Reference basis/quadrature unresolved')
        if r.get('outside_local_fit_mass',0)>1e-3:issue.append('Outside local surrogate domain')
        if abs(r['energy']-rows[0]['energy'])>1e-5:issue.append('Fixed-H energy drift requires investigation')
        if issue:
            if first is None:first=r['t_fs']
            reasons.extend(issue)
    result=dict(first_invalid_diagnostic_fs=first,reasons=sorted(set(reasons)),
        certified=False,warning='No state-accuracy interpretation after reference/domain failure; older exploratory receipts retained.')
    (root/'quality_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--ntraj',type=int,default=16)
    p.add_argument('--nf',type=int,default=12);p.add_argument('--components',type=int,nargs='+',default=[1,2,4,8])
    p.add_argument('--repeats',type=int,default=12)
    a=p.parse_args();benchmark(**vars(a))


if __name__=='__main__':main()
