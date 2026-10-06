"""Bare oscillator number from BOTH quadratures; read-only VSC/Model A adapter.

All plotted expectations are divided by the recorded norm. VSC p^2 is recovered
from the independently propagated photon energy, NEVER from density gradients.
This is not an output-photon counting observable in an interacting cavity.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def model_series(run):
    """Read scalar histories; no wave or TDSE needed. q2,p2 are physical a.u."""
    run=Path(run);rows=json.loads((run/'observables.json').read_text())
    c=json.loads((run/'identity.json').read_text())['config']
    norm=np.array([r['norm'] for r in rows])
    data=dict(time_fs=np.array([r['time_fs'] for r in rows]),
              N=np.array([r['n_ph'] for r in rows])/norm,norm=norm)
    if all('n_q' in r and 'n_p' in r for r in rows):
        for k in ['n_q','n_p','q2','p2']:
            data[k]=np.array([r[k] for r in rows])/norm
        if not np.allclose(data['N'],data['n_q']+data['n_p']-.5,atol=1e-10):
            raise ValueError('Inconsistent Model A photon moments')
    return data,dict(source=str(run.resolve()),omega_au=c['omega'],g=c['g'],
        method='Native FFT photon derivative in model.observables; scalar history',
        label=f'Model A g={c["g"]:g}',certified=False)


def vsc_series(run,plan=None):
    """VSC Q=sqrt(omega)q: nq=<Q²>/2, np=Eph/omega-nq.

    Require original status and matching campaign input SHA. No frequency is
    inferred from the very number being checked. energy_parts[3] is Eph in Ha.
    The resulting p2 is 2*omega*np, not a new independent wave-derivative test.
    """
    run=Path(run)
    status=json.loads((run/'status.json').read_text());ident=status['identity']
    if plan is None:
        plan=next((p/'campaign_plan.json' for p in run.parents if (p/'campaign_plan.json').exists()),None)
    if plan is None:raise FileNotFoundError('Need --plan path/to/original/campaign_plan.json')
    config=json.loads(Path(plan).read_text());omega=float(config['omega'])
    if omega<=0 or config.get('input_sha256')!=ident.get('input_sha256'):
        raise ValueError('Plan and run input identities do not match')
    Q=np.linspace(-ident['half_Q'],ident['half_Q'],ident['nq'],endpoint=False)
    dQ=Q[1]-Q[0];rows=[]
    for path in sorted(run.glob('observable_*.npz')):
        with np.load(path,allow_pickle=False) as z:
            norm=float(z['norm']);rho=z['rho_Q'];eph=float(z['energy_parts'][3])
            if rho.shape!=Q.shape or norm<=0:raise ValueError(f'Invalid marginal: {path}')
            if abs(rho.sum()*dQ-norm)>1e-8:raise ValueError(f'Q marginal norm mismatch: {path}')
            nq=float(np.sum(rho*Q**2)*dQ/2/norm);np_=eph/(omega*norm)-nq
            number=nq+np_-.5;saved=float(z['nph'])/norm
            if abs(number-saved)>1e-8 or np_<-1e-10:raise ValueError(f'Photon energy/number mismatch: {path}')
            rows.append([float(z['time_au'])*.024188843265857,number,nq,np_,2*nq/omega,2*omega*np_,norm])
    if not rows:raise ValueError('No observable_*.npz; density-only fields are insufficient')
    a=np.asarray(rows)
    if np.any(np.diff(a[:,0])<=0):raise ValueError('Duplicate/nonmonotonic times')
    return dict(zip(['time_fs','N','n_q','n_p','q2','p2','norm'],a.T)),dict(
        source=str(run.resolve()),plan=str(Path(plan).resolve()),omega_au=omega,
        label='VSC '+run.parent.name,method='Stored native-FFT Eph plus rho_Q; energy_parts[3]',
        check='Algebraic stored nph/Eph consistency, NOT independent basis convergence',
        source_status=status['status'],certified=False)


def draw(series,out):
    """N, change from initial, quadrature budget. Save PNG+NPZ+provenance JSON."""
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a NEW photon plot directory')
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    meta=[]
    for i,(d,m) in enumerate(series):
        t=d['time_fs'];label=m['label'];color=plt.cm.tab10(i%10)
        axes[0].plot(t,d['N'],label=label,color=color)
        axes[1].plot(t,d['N']-d['N'][0],label=label,color=color)
        if 'n_q' in d:
            axes[2].plot(t,d['n_q'],color=color,label=label+r' : $\omega\langle q^2\rangle/2$')
            axes[2].plot(t,d['n_p'],color=color,ls='--',label=label+r' : $\langle p^2\rangle/(2\omega)$')
        np.savez_compressed(out/f'photon_number_{i}.npz',**d)
        meta.append(dict(m,initial_N=float(d['N'][0]),final_N=float(d['N'][-1]),
                         min_N=float(d['N'].min()),max_N=float(d['N'].max())))
    for ax,title in zip(axes,[r'$\langle N\rangle$',r'$\langle N(t)\rangle-\langle N(0)\rangle$',r'Quadrature budget: $N=n_q+n_p-1/2$']):
        ax.set(xlabel='Time (fs)',title=title);ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    axes[2].axhline(.25,color='gray',lw=.7,ls=':',label='Vacuum per quadrature')
    fig.suptitle('Bare cavity occupation | '+', '.join(f'omega={m["omega_au"]:.8g} Ha' for _,m in series))
    fig.savefig(out/'photon_number.png',dpi=160);plt.close(fig)
    (out/'photon_number.json').write_text(json.dumps(dict(series=meta,
        normalization='All moments / recorded norm; no clipping of negative roundoff',
        interpretation='Bare oscillator occupation, not emitted/detected photon count; diagnostic only'),indent=2)+'\n')
    return meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-run',type=Path,action='append',default=[])
    ap.add_argument('--vsc-run',type=Path,action='append',default=[])
    ap.add_argument('--plan',type=Path);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if not a.model_run and not a.vsc_run:ap.error('Supply --model-run or --vsc-run')
    series=[model_series(p) for p in a.model_run]+[vsc_series(p,a.plan) for p in a.vsc_run]
    print(json.dumps(draw(series,a.out),indent=2))


if __name__=='__main__':main()
