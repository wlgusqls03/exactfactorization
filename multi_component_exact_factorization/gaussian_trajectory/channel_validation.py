"""Bounded CPU pilot matrix; no production claim or overwrite of old results."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from .channel_run import ChannelConfig, run
from .run import atomic_json


def validate(out, end_fs=.01, movies=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Choose a new pilot root')
    if not 0 < end_fs <= .05:
        raise ValueError('This bounded pilot supports 0 < end_fs <= .05; use channel_run for explicit longer studies')
    out.mkdir(parents=True)
    records = []
    for model in ('mcef','pen-nonadiabatic','pen-rabi'):
        for g in ([None] if model=='mcef' else [0., .01]):
            grid = dict(name=model,nx=9,nq=16,qmin=-6.,qmax=-2.) if model=='mcef' else dict(name=model,nq=32,g=g)
            for method in ('ctmqc','tsh'):
                for compression in ('native','gaussian'):
                    tag = '%s_g%s_%s_%s' % (model,g,method,compression)
                    cfg = ChannelConfig(model=grid,method=method,compression=compression,ntraj=7,
                          dt_au=.01,end_fs=end_fs,every_fs=end_fs/2)
                    records.append(_case(cfg,out/tag))
    # Modified-Q is a separate prescription, not the density-Q default.
    for compression in ('native','gaussian'):
        cfg = ChannelConfig(model=dict(name='pen-nonadiabatic',nq=32,g=.01),method='ctmqc',
              quantum_momentum_mode='pair_intercept',compression=compression,ntraj=7,
              dt_au=.01,end_fs=end_fs,every_fs=end_fs/2)
        records.append(_case(cfg,out/('pair_intercept_'+compression)))
    for record in records:
        path=Path(record['path'])
        if record['status']!='COMPLETE_NOT_CERTIFIED':
            continue
        config=json.loads((path/'config.json').read_text())
        if config['compression']=='gaussian':
            native=path.with_name(path.name.replace('gaussian','native'))
            if (native/'checkpoint.npz').exists():
                a=_last_frame(path); b=_last_frame(native)
                record['native_comparison']={
                    'max_G_l2':float(np.sqrt(np.einsum('q,iqk->i',a['quadrature'],abs(a['G']-b['G'])**2)).max()),
                    'mean_energy_difference_Ha':float(a['eta']@(a['energy']-b['energy'])),
                    'max_P_difference':float(np.max(abs(a['P']-b['P'])))}
    render_paths=[out/('%s_g%s_%s_gaussian' % (model,g,method))
                  for model,g in [('mcef',None),('pen-nonadiabatic',.01)] for method in ('ctmqc','tsh')]
    from .renderer import render as base_render
    from .channel_render import render as channel_render
    for path in render_paths:
        if (path/'checkpoint.npz').exists():
            base_render(path,movies=movies)
            channel_render(path,movies=movies)
    report=dict(cases=records,passed=sum(r['status']=='COMPLETE_NOT_CERTIFIED' for r in records),
                total=len(records),scope='Short CPU implementation pilots, not physical/time/grid/K/ensemble convergence',
                rendered=[str(p) for p in render_paths])
    atomic_json(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('passed','total','scope')},indent=2),flush=True)
    return report


def _last_frame(path):
    with np.load(sorted((path/'frames').glob('frame_*.npz'))[-1]) as f:
        return {k:f[k].copy() for k in f.files}


def _case(config,path):
    start=time.perf_counter()
    try:
        run(config,path,render=False)
        status=json.loads((path/'status.json').read_text())
        first=sorted((path/'frames').glob('frame_*.npz'))[0]
        final=_last_frame(path)
        with np.load(first) as f:
            delta=float(final['eta']@(final['energy']-f['energy']))
        return dict(path=str(path),status=status['status'],t_fs=status['t_fs'],
                    wall_seconds=time.perf_counter()-start,audit=status['numerical_audit'],
                    mean_energy_change_Ha=delta,hops=status['hops'])
    except Exception as exc:
        return dict(path=str(path),status='FAILED',error=repr(exc),wall_seconds=time.perf_counter()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--end-fs',default=.01,type=float)
    p.add_argument('--movies',action='store_true')
    a=p.parse_args()
    report=validate(a.out,a.end_fs,a.movies)
    if report['passed']!=report['total']:
        raise SystemExit('Some pilots failed; see report.json (no failures hidden)')


if __name__=='__main__':
    main()
