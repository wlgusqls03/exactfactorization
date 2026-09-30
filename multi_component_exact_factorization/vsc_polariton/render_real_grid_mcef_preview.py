"""Re-render saved small real-grid diagnostic fields without TDSE/derivatives.

This is only visualization, never a validation gate. Useful after an unrelated
missing/corrupt refinement file interrupted a previous postprocessing run.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .run_real_grid_mcef_preview import ROOT, digest
from .real_grid_mcef_fields import diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields',required=True,type=Path)
    p.add_argument('--observables',type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--movie',action='store_true')
    p.add_argument('--label',default='Full real-grid 3DOF | MCEF diagnostic')
    args=p.parse_args();out=args.out.resolve();out.relative_to(ROOT.resolve())
    if out.exists():raise FileExistsError('Choose a new --out')
    paths=sorted(args.fields.glob('fields_*.npz'))
    if not paths:raise FileNotFoundError('No saved MCEF fields')
    frames=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:frames.append({k:z[k] for k in z.files})
    out.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(out/'system_cache'))
    from .real_grid_mcef_plotting import style,dynamics,snapshots
    style()
    info=dict(phase7_pass=False,scope='Rendering of existing diagnostic fields, not new field validation',
              field_sha256={str(p):digest(p) for p in paths},
              diagnostics=[diagnostics(f) for f in frames])
    if args.observables:
        times=[];rows=[]
        for path in sorted(args.observables.glob('observable_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                times.append(float(z['time_au']))
                rows.append({k:z[k] for k in ('rho_R','product','flux')})
        if not rows:raise FileNotFoundError('No observables')
        info['dynamics']=dynamics(frames[0]['R'],times,rows,out,args.label)
    info['plotting']=snapshots(frames,out,args.label,movie=args.movie)
    (out/'render_summary.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    print('Saved actual-data diagnostic figures:',out)


if __name__=='__main__':main()
