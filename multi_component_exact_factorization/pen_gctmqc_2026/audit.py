"""Read-only time-step comparison; reports measurements, not certification."""
import argparse
import json
from pathlib import Path
import numpy as np


def norm_intervals(rows, threshold=1e-5):
    """The saved monitor is an interval maximum, not an instantaneous norm."""
    intervals = []
    for i, row in enumerate(rows):
        if row['raw_norm_error'] > threshold:
            intervals.append([rows[max(0, i-1)]['t_fs'], row['t_fs']])
    return dict(threshold=threshold, intervals=intervals,
                first_interval=intervals[0] if intervals else None,
                count=len(intervals))


def audit(coarse,fine,out,require_through_fs=None,plots=False):
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
        if sa.get('adapter_safety_version') != sb.get('adapter_safety_version'):
            raise ValueError('Different native initialization versions; not a timestep test')
        if sa.get('source_manifest',{}).get('binary_sha256') != sb.get('source_manifest',{}).get('binary_sha256'):
            raise ValueError('Different native executables; not a timestep test')
        a=json.loads((cfg.parent/'observables.json').read_text())
        b=json.loads((other/'observables.json').read_text())
        ta,tb=np.array([x['t_fs'] for x in a]),np.array([x['t_fs'] for x in b])
        if require_through_fs is not None and min(ta[-1],tb[-1])+1e-8 < require_through_fs:
            raise ValueError(f'{rel}: both runs must reach {require_through_fs:g} fs')
        take=(ta>=tb[0])&(ta<=tb[-1]);t=ta[take]
        if not len(t):
            raise ValueError('No shared times')
        errors={}
        exact_errors={}
        # Separate exact shared timestamps from interpolation uncertainty.
        j=np.argmin(abs(ta[:,None]-tb[None,:]),axis=1)
        exact=abs(ta-tb[j])<1e-10
        for key in ('photon_number','energy','R_mean','product_R_ge_4','bo_population','basis_population'):
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
            data_coverage_end_fs=float(min(ta[-1],tb[-1])),
            exact_shared_time_count=int(exact.sum()),exact_shared_time_max_differences=exact_errors,
            first_common_fs=float(t[0]),
            raw_norm_monitor=dict(coarse=norm_intervals(a),fine=norm_intervals(b)),
            coarse=json.loads((cfg.parent/'validation.json').read_text()),
            fine=json.loads((other/'validation.json').read_text())))
        if plots:
            plot_comparison(a,b,ca['dt'],cb['dt'],str(rel),out.parent/'plots'/rel/'timestep_comparison.png')
    if not reports:
        raise ValueError('No matching author cases')
    out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(cases=reports,certified=False,
        requested_through_fs=require_through_fs,
        note='Only shared saved times; exact timestamps and interpolated differences are separate. '
             'Coverage is measured, not assumed. This does not certify Ntraj/M/seed convergence.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    return report


def plot_comparison(a,b,dt_a,dt_b,title,out):
    """Small, fixed-layout numerical audit; no exact TDPES is inferred."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.size':12,'axes.titlesize':14,'axes.labelsize':13,
                         'legend.fontsize':11}):
        fig,axs=plt.subplots(3,2,figsize=(15,12),constrained_layout=True)
        for rows,dt,color in ((a,dt_a,'#0072B2'),(b,dt_b,'#D55E00')):
            t=np.array([r['t_fs'] for r in rows])
            label=rf'$\Delta t={dt:g}$ a.u.'
            series=[([r['photon_number'] for r in rows],'Photon number'),
                    ([r['bo_population'][1] for r in rows],r'$P(S_1)$'),
                    ([r['R_mean'] for r in rows],r'$\langle R\rangle$ (a0)'),
                    ([r['product_R_ge_4'] for r in rows],r'$P(R\geq4)$'),
                    ([r['energy']-rows[0]['energy'] for r in rows],r'$E(t)-E(0)$ (Ha)'),
                    ([r['raw_norm_error'] for r in rows],r'Pre-normalization $|\|C\|^2-1|$')]
            for ax,(y,ylabel) in zip(axs.flat,series):
                ax.plot(t,y,color=color,lw=1.6,label=label)
                ax.set(xlabel='t (fs)',ylabel=ylabel,xlim=(0,min(a[-1]['t_fs'],b[-1]['t_fs'])))
                ax.grid(alpha=.15)
        for i,ax in enumerate(axs.flat):
            ax.set_title(chr(65+i)+'  '+['Photon exchange','Electronic population',
                'Nuclear motion','Right-side probability','Energy diagnostic','Step-error monitor'][i])
        axs[0,0].legend()
        axs[0,1].set_ylim(-.03,1.03)
        axs[1,1].set_ylim(-.03,1.03)
        axs[2,0].axhline(0,color='.4',ls='--',lw=.8)
        axs[2,1].set_yscale('symlog',linthresh=1e-12)
        axs[2,1].set_ylim(bottom=0)
        axs[2,1].axhline(1e-5,color='k',ls='--',lw=1,label='Review threshold $10^{-5}$')
        axs[2,1].legend()
        fig.suptitle(title+'\nTime-step validation; completion and normalized norm are not accuracy tests',fontsize=16)
        fig.savefig(out,dpi=160)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coarse',type=Path,required=True)
    p.add_argument('--fine',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-through-fs',type=float)
    p.add_argument('--plots',action='store_true')
    a=p.parse_args()
    report=audit(a.coarse,a.fine,a.out,a.require_through_fs,a.plots)
    for case in report['cases']:
        print(case['case'], 'data coverage', case['data_coverage_end_fs'], 'fs;',
              'exact shared frames:', case['exact_shared_time_count'],
              'differences:',case['exact_shared_time_max_differences'])
    print('AUDIT',a.out,'(not a convergence certificate)')


if __name__=='__main__':
    main()
