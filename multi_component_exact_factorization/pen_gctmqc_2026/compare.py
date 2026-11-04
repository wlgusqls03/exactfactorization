"""Compare direct observables, never invent CTMQC exact-factorization fields."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .render import STYLE,finish


def load(path):
    config=json.loads((path/'config.json').read_text())
    rows=json.loads((path/'observables.json').read_text())
    times=np.array([r['t_fs'] for r in rows])
    if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid time series '+str(path))
    return config,rows,times


def compare(author_root,out,python_root=None,qd_root=None,tmax=None,require=None):
    author_root,out=Path(author_root),Path(out)
    groups=sorted({p.parent.parent.relative_to(author_root) for p in author_root.rglob('author_settings.json')})
    reports=[]
    with plt.rc_context(STYLE):
        for rel in groups:
            paths={f'G-CTMQC {v.upper()}':author_root/rel/v for v in ('el','pol')}
            if python_root:
                paths.update({f'Python {v.upper()}':Path(python_root)/rel/v for v in ('el','pol')})
            if qd_root:
                paths['QD (finite Fock)']=Path(qd_root)/rel
            data={name:load(p) for name,p in paths.items() if (p/'observables.json').exists()}
            if not data:
                continue
            cfg=next(iter(data.values()))[0]
            for name,(c,_,_) in data.items():
                for key in ('mass','omega','g','dse','k','a','b','r1','r2','r3','delta','z','mu12','nf'):
                    if c.get(key)!=cfg.get(key):
                        raise ValueError('Different physical configurations: '+name+' '+key)
            begin=max(item[2][0] for item in data.values())
            end=min(item[2][-1] for item in data.values())
            if tmax is not None:
                end=min(end,tmax)
            report=dict(group=str(rel),common_start_fs=float(begin),common_end_fs=float(end),methods=list(data),
                warnings=['EL(paper) starts in upper CBO; POL/QD in bare S1 x vacuum.',
                          'EL photon number is classical Wigner estimator; POL/QD is quantum finite-Fock.',
                          'Energy and conditional mean energy are not exact TDPES.',
                          'Different trajectory counts/seeds have sampling error; do not infer solver error from mean difference alone.'])
            for name,path in paths.items():
                if (path/'status.json').exists():
                    status=json.loads((path/'status.json').read_text()).get('status')
                    if status not in ('COMPLETE','COMPLETE_NOT_CERTIFIED'):
                        report['warnings'].append(f'{name}: status={status}; saved data only.')
            absent=[name for name in paths if name not in data]
            report['missing_methods']=absent
            if require is not None and (end+1e-8<require or absent):
                report['status']='SKIPPED_INCOMPLETE'
                reports.append(report)
                continue
            fig,axs=plt.subplots(3,2,figsize=(15,12),constrained_layout=True)
            specs=[('bo_population',1,'BO S1 population'),('photon_number',None,'Photon number'),
                   ('product_R_ge_4',None,r'$P(R\geq4)$'),('R_mean',None,'Mean R (a0)'),
                   ('energy',None,'E(t) - E(0) (Ha)'),('pol_population',1,'Pol1 population (quantum only)')]
            for ax,(key,column,label) in zip(axs.flat,specs):
                for j,(name,(_,rows,t)) in enumerate(data.items()):
                    if key not in rows[0]:
                        continue
                    y=np.array([r[key] for r in rows])
                    if column is not None:
                        y=y[:,column]
                    if key=='energy':
                        y=y-y[0]
                    inside=(t>begin)&(t<end)
                    tx=np.r_[begin,t[inside],end]
                    ax.plot(tx,np.interp(tx,t,y),lw=1.9,label=name,ls='--' if name.startswith('Python') else '-')
                finish(ax,ylabel=label)
                ax.set_xlim(begin,end if end>begin else begin+1e-3)
                if key=='product_R_ge_4':
                    # Do not visually magnify 1e-16 roundoff into a reaction.
                    ax.set_ylim(-.02,1.02)
            fig.suptitle(str(rel)+'\nG-CTMQC / independent Python / QD — direct observables')
            target=out/rel
            target.mkdir(parents=True,exist_ok=True)
            fig.savefig(target/'observables_comparison.png',dpi=160)
            plt.close(fig)
            report['status']='PLOTTED_COMMON_TIME_ONLY'
            report['endpoint_interpolation']='Linear at common interval endpoints; no extrapolation'
            reports.append(report)
    out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(reports,indent=2)+'\n')
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--author-root',type=Path,required=True)
    p.add_argument('--python-root',type=Path)
    p.add_argument('--qd-root',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--t-max-fs',type=float)
    p.add_argument('--require-through-fs',type=float)
    a=p.parse_args()
    reports=compare(a.author_root,a.out,a.python_root,a.qd_root,a.t_max_fs,a.require_through_fs)
    for r in reports:
        print(r['status'],r['group'],r['common_end_fs'])
    if not reports or any(r['status']=='SKIPPED_INCOMPLETE' for r in reports):
        raise SystemExit(2)


if __name__=='__main__':
    main()
