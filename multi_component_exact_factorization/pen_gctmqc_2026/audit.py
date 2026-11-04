"""Read-only time-step comparison; reports measurements, not certification."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(coarse,fine,out):
    coarse,fine,out=map(Path,(coarse,fine,out))
    reports=[]
    for cfg in sorted(coarse.rglob('author_settings.json')):
        rel=cfg.parent.relative_to(coarse)
        other=fine/rel
        if not (other/'observables.json').exists():
            continue
        ca=json.loads((cfg.parent/'config.json').read_text())
        cb=json.loads((other/'config.json').read_text())
        for key in ca:
            if key!='dt' and ca[key]!=cb[key]:
                raise ValueError('Different physics: '+key)
        sa,sb=json.loads(cfg.read_text()),json.loads((other/'author_settings.json').read_text())
        for key in ('perspective','initial','ntraj','seed','m_parameter','algorithm','initial_sample_sha256'):
            if sa[key]!=sb[key]:
                raise ValueError('Different ensemble/algorithm: '+key)
        a=json.loads((cfg.parent/'observables.json').read_text())
        b=json.loads((other/'observables.json').read_text())
        ta,tb=np.array([x['t_fs'] for x in a]),np.array([x['t_fs'] for x in b])
        take=(ta>=tb[0])&(ta<=tb[-1]);t=ta[take]
        if not len(t):
            raise ValueError('No shared times')
        errors={}
        exact_errors={}
        # Separate exact shared timestamps from interpolation uncertainty.
        j=np.argmin(abs(ta[:,None]-tb[None,:]),axis=1)
        exact=abs(ta-tb[j])<1e-10
        for key in ('photon_number','energy','R_mean','product_R_ge_4','basis_population'):
            ya=np.array([x[key] for x in a])[take]
            yb=np.array([x[key] for x in b])
            if yb.ndim==1:
                matched=np.interp(t,tb,yb)
            else:
                matched=np.array([np.interp(t,tb,yb[:,j]) for j in range(yb.shape[1])]).T
            errors[key]=float(np.max(abs(ya-matched)))
            if np.any(exact):
                exact_errors[key]=float(np.max(abs(np.array([x[key] for x in a])[exact]-yb[j[exact]])))
        reports.append(dict(case=str(rel),dt_au=[ca['dt'],cb['dt']],
            common_end_fs=float(t[-1]),max_observable_differences=errors,
            exact_shared_time_count=int(exact.sum()),exact_shared_time_max_differences=exact_errors,
            coarse=json.loads((cfg.parent/'validation.json').read_text()),
            fine=json.loads((other/'validation.json').read_text())))
    if not reports:
        raise ValueError('No matching author cases')
    out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(cases=reports,certified=False,
        note='Only shared saved times; linear interpolation for dt comparison. Not a 40-fs or Ntraj convergence test.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coarse',type=Path,required=True)
    p.add_argument('--fine',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.coarse,a.fine,a.out),indent=2))


if __name__=='__main__':
    main()
