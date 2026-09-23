"""Provisional matched-scale snapshot figures, not certified Phase7 results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .phase7_pilot_baseline import OUTPUT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis',type=Path,default=OUTPUT/'analysis_v2')
    parser.add_argument('--out',type=Path,default=OUTPUT/'figures_v1')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':13})
    cases=['free','resonant','barrier'];data={}
    for case in cases:
        paths=list(args.analysis.glob(case+'_t1652_Nq*.npz'))
        path=max(paths,key=lambda p:int(p.stem.split('Nq')[-1]))
        with np.load(path) as z:data[case]={k:z[k] for k in z.files}
    # Fixed cross-case scale on declared support; no percentile clipping.
    maximum=max(np.max(abs(d['epsilon1_B'].real[d['mask_1e-08']])) for d in data.values())
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    images=[]
    qmax=max(np.max(abs(d['q'])) for d in data.values())
    for ax,(case,d) in zip(axes,data.items()):
        val=np.ma.masked_where(~d['mask_1e-08'],d['epsilon1_B'].real)
        im=ax.pcolormesh(d['q'],d['R'],val,cmap='RdBu_r',vmin=-maximum,vmax=maximum,shading='auto')
        images.append(im)
        levels=np.max(d['rho_qR'])*np.array([.01,.1,.5])
        ax.contour(d['q'],d['R'],d['rho_qR'],levels=levels,colors='0.2',linewidths=.6)
        ax.set(title=case,xlabel=r'$q_c$ (a.u.)',ylabel=r'$R$ ($a_0$)')
        ax.set_xlim(-qmax,qmax);ax.set_ylim(-4.4,4.4)
    fig.colorbar(im,ax=axes,label=r'Re $\epsilon^{(1)}$ (Ha), positive-marginal gauge')
    fig.suptitle('Phase 7 pilot: 39.96 fs — scalar convergence NOT certified')
    for ext in ('png','pdf'):fig.savefig(args.out/('epsilon1_pilot.'+ext),dpi=180)
    for im in images:im.set_clim(-.1,.1)
    fig.suptitle('Phase 7 pilot — fixed colour zoom ±0.1 Ha; outliers saturated; NOT certified')
    for ext in ('png','pdf'):fig.savefig(args.out/('epsilon1_pilot_zoom.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,layout='constrained')
    for case,d in data.items():
        mask=d['outer_mask_1e-08'];r=d['R']
        axes[0].plot(r,d['outer_rho_R'],label=case)
        axes[1].plot(r,np.where(mask,d['outer_epsilon2_B'].real,np.nan))
        axes[2].plot(r,np.where(mask,d['outer_force'],np.nan))
    axes[0].set_ylabel(r'$\rho_R$ ($a_0^{-1}$)')
    axes[0].legend(loc='upper left',bbox_to_anchor=(1.01,1))
    axes[1].set_ylabel(r'$\epsilon^{(2)}_{PG}$ (Ha)')
    axes[2].set(ylabel=r'$-\partial_R\epsilon^{(2)}+\partial_t\alpha$ (Ha/$a_0$)',xlabel=r'$R$ ($a_0$)')
    for ax in axes:ax.axvline(0,color='.5',linestyle=':',linewidth=.8)
    axes[2].axhline(0,color='.5',linestyle=':',linewidth=.8)
    fig.suptitle('Phase 7 pilot at 39.96 fs — raw positive-marginal gauge; force provisional')
    for ext in ('png','pdf'):fig.savefig(args.out/('outer_fields_pilot.'+ext),dpi=180)
    plt.close(fig)
    (args.out/'README.txt').write_text('Provisional two-time pilot. No interpolation/movie. '
        'epsilon1 routes fail the declared tolerance for coupled final frames. '
        'Contours: 1%, 10%, 50% of each joint density peak. '
        'Scalar positive-marginal gauge, no subtracted energy reference. '
        'Force is gauge-invariant -epsilon_R+alpha_t, not hydrodynamic acceleration.\n')


if __name__=='__main__':main()
