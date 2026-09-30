"""Render photon-box and FFT-spacing evidence from the read-only audit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('results/vsc_polariton/phase7/photon_grid_audit_v1')
    r=json.loads((root/'summary.json').read_text())
    with np.load(root/'photon_densities.npz') as z:
        data={k:z[k] for k in z.files}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    Q=data['Q']
    colors=plt.cm.viridis(np.linspace(.08,.88,5))
    for color,f in zip(colors,[f for f in r['frames'] if f['case']=='barrier_F240']):
        key=f['case']+'_'+Path(f['path']).stem
        axes[0,0].semilogy(Q,np.maximum(data[key],1e-30),color=color,
                         label=f"{f['time_fs']:.2f} fs")
    axes[0,0].set(xlim=(-25,25),ylim=(1e-14,1),xlabel=r'$Q=\sqrt{\omega_c}\,q_c$',
                  ylabel=r'$\rho_Q$',title='A  |  Actual photon distribution: barrier case')
    axes[0,0].legend(fontsize=9,ncol=2)
    for L in (-24,24): axes[0,0].axvline(L,color='#bf5139',ls='--',lw=1)
    widths=np.array([4,6,8,10,12,16,20,24,28])
    for case,color in [('barrier_F240','#087d8e'),('resonant_F200','#dc7833')]:
        frames=[f for f in r['frames'] if f['case']==case]
        tails=[max(f['tails'][str(L)]['probability'] for f in frames) for L in widths]
        axes[0,1].semilogy(widths,tails,'o-',color=color,label=case)
    axes[0,1].axhline(1e-8,color='0.3',ls='--',label=r'$10^{-8}$ probability')
    axes[0,1].set(ylim=(1e-30,1),xlabel=r'Box half-width $L_Q$',ylabel=r'$P(|Q|>L_Q)$',
                  title='B  |  Largest tail among available snapshots')
    axes[0,1].legend(fontsize=9)
    grids=r['frames'][0]['grids']
    ids=[i for i,g in enumerate(grids) if g['half_Q']==24]
    for metric,label,color in [('backprojection_L2','Wave back-projection','#087d8e'),
                  ('derivative2_relative_L2',r'$\partial_Q^2\Psi$: relative $L^2$','#dc7833')]:
        values=[max(f['grids'][i][metric] for f in r['frames']) for i in ids]
        axes[1,0].semilogy([grids[i]['NQ'] for i in ids],values,'o-',label=label,color=color)
    axes[1,0].set(xlabel=r'$N_Q$ at $Q\in[-24,24)$',ylabel='Largest representation error',
                  title='C  |  Spacing matters even when norm looks correct')
    axes[1,0].legend(fontsize=9)
    axes[1,1].axis('off')
    axes[1,1].text(0,1,
        'STARTING GRID, NOT A PRODUCTION PASS\n\n'
        r'$Q\in[-24,24),\quad N_Q=384,\quad\Delta Q=0.125$'+'\n\n'
        'Refinement: same box with 512 points\n'
        'Box check: [-28,28) with 448 points\n\n'
        'At 170.6 meV: q half-box = 303.11 a.u.\n'
        'At 161.77 meV: q half-box = 311.27 a.u.\n\n'
        '9 saved snapshots; two coupled cases.\n'
        'No new TDSE was propagated.\n'
        'No claim of local TDPES or full-time convergence.',
        va='top',linespacing=1.5)
    for ax in axes.ravel()[:3]:ax.grid(alpha=.15)
    fig.suptitle('Photon coordinate-grid audit | Same finite-Fock states, independent FFT derivatives',fontsize=14)
    for suffix in ('png','pdf'):
        target=root/f'photon_grid_comparison.{suffix}'
        if target.exists():raise FileExistsError(target)
        fig.savefig(target,dpi=190)
    plt.close(fig)


if __name__=='__main__':
    main()
