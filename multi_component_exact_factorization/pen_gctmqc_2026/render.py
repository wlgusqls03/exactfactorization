"""Direct author-trajectory diagnostics, explicitly NOT exact MCEF fields."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from ..pen_ctmqc_2026.core import hamiltonian
from .data import load_case, read_frame, frame_observables, analyze_case

STYLE={'font.size':13,'axes.labelsize':14,'axes.titlesize':15,'figure.titlesize':17,
       'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False}
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']


def finish(ax, xlabel='t (fs)', ylabel='', title=''):
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.grid(alpha=.15)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best',framealpha=.85)


def summary(root, c, meta, rows):
    t=np.array([r['t_fs'] for r in rows])
    value=lambda k:np.array([r[k] for r in rows])
    fig,axs=plt.subplots(3,2,figsize=(15,13),constrained_layout=True)
    for j in range(2):
        axs[0,0].plot(t,value('bo_population')[:,j],color=COLORS[j],label=rf'$S_{j}$')
    finish(axs[0,0],ylabel='Population',title='A  Bare electronic populations')
    pop=value('basis_population')
    name='Pol' if meta['perspective']=='pol' else 'CBO'
    for j in range(pop.shape[1]):
        axs[0,1].plot(t,pop[:,j],color=COLORS[j],label=rf'$\mathrm{{{name}}}_{{{j}}}$')
    finish(axs[0,1],ylabel='Population',title='B  Conditional-state populations')
    for ax in axs[0]:
        ax.set_ylim(-.03,1.03)
    axs[1,0].plot(t,value('photon_number'),color=COLORS[0],label=r'$\langle N\rangle$')
    axs[1,0].fill_between(t,value('photon_number')-value('photon_number_se'),
                         value('photon_number')+value('photon_number_se'),alpha=.18)
    finish(axs[1,0],ylabel='Photon number',title='C  Photon number'+(' (EL classical estimator)' if name=='CBO' else ''))
    axs[1,1].plot(t,value('energy')-rows[0]['energy'],color=COLORS[1],label=r'$E(t)-E(0)$')
    axs[1,1].axhline(0,color='.5',ls='--',lw=.8)
    finish(axs[1,1],ylabel='Energy change (Ha)',title='D  Energy diagnostic (not guaranteed conserved)')
    axs[2,0].plot(t,value('product_R_ge_4'),color=COLORS[2],label=r'$P(R\geq4)$')
    finish(axs[2,0],ylabel='Probability',title='E  Right-side probability')
    axs[2,0].set_ylim(-.03,1.03)
    axs[2,1].plot(t,value('raw_norm_error'),color=COLORS[1],label='Before normalization')
    axs[2,1].plot(t,value('coefficient_norm_max_error'),color=COLORS[0],label='Saved normalized state')
    axs[2,1].set_yscale('symlog',linthresh=1e-15)
    finish(axs[2,1],ylabel=r'$|\|C\|^2-1|$',title='F  Numerical monitoring')
    fig.suptitle(f"Author G-CTMQC / {meta['perspective'].upper()} / "
                 f"g={c.g:g}, omega={c.omega:g} Ha, N={meta['ntraj']}\n"
                 'Short/long completion is not a convergence certificate')
    fig.savefig(root/'plots/dynamics_summary.png',dpi=160)
    plt.close(fig)


def trajectory_figure(c,meta,frame,rows,limits):
    fig,axs=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    X,P=frame['X'],frame['P']
    R=np.linspace(*limits['R'],300)
    h=c.sigma_r/np.sqrt(2)
    den=np.exp(-.5*((R[:,None]-X[None,:,0])/h)**2).mean(axis=1)/(np.sqrt(2*np.pi)*h)
    axs[0,0].fill_between(R,den,color='.65',alpha=.7)
    axs[0,0].axvline(4,color='k',ls='--',lw=1)
    finish(axs[0,0],'R (a0)','Density','A  Nuclear ensemble (Gaussian-smoothed)')
    axs[0,0].set(xlim=limits['R'],ylim=(0,limits['density']))
    v=hamiltonian(c,meta['perspective'],X)[0]
    local=np.einsum('ni,nij,nj->n',frame['coeff'].conj(),v,frame['coeff']).real
    axs[0,1].scatter(X[:,0],local,s=12,alpha=.5,color=COLORS[0],label='Conditional mean energy')
    coords=R[:,None] if meta['perspective']=='pol' else np.c_[R,np.zeros_like(R)]
    energies=np.linalg.eigvalsh(hamiltonian(c,meta['perspective'],coords)[0])
    for j in range(energies.shape[1]):
        label=rf'$\mathrm{{Pol}}_{j}$' if meta['perspective']=='pol' else rf'$\mathrm{{CBO}}_{j}(q_c=0)$'
        axs[0,1].plot(R,energies[:,j],'--',color=COLORS[j],lw=1.2,label=label)
    finish(axs[0,1],'R (a0)','Energy (Ha)','B  Energy proxy — NOT exact TDPES')
    axs[0,1].set(xlim=limits['R'],ylim=limits['energy'])
    t=np.array([r['t_fs'] for r in rows])
    axs[1,0].plot(t,[r['photon_number'] for r in rows],color=COLORS[0])
    finish(axs[1,0],ylabel='Photon number',title='C  Photon exchange')
    key='pol_population' if meta['perspective']=='pol' else 'cbo_population'
    pop=np.array([r[key] for r in rows])
    for j in range(pop.shape[1]):
        axs[1,1].plot(t,pop[:,j],color=COLORS[j],label=f'{"Pol" if key.startswith("pol") else "CBO"}{j}')
    finish(axs[1,1],ylabel='Population',title='D  Conditional-state populations')
    axs[1,1].set_ylim(-.03,1.03)
    for ax in axs[1]:
        ax.axvline(frame['t_fs'],color='k',ls='--')
        ax.set_xlim(t[0],max(t[-1],1e-6))
    fig.suptitle(f"Author G-CTMQC / {meta['perspective'].upper()} / t={frame['t_fs']:.3f} fs / g={c.g:g}")
    return fig


def render_case(root,movies=False):
    root=Path(root)
    c,meta,files=load_case(root)
    if not (root/'observables.json').exists():
        analyze_case(root)
    rows=json.loads((root/'observables.json').read_text())
    (root/'plots').mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        summary(root,c,meta,rows)
        # Range fixed for this entire video; actual observed ensemble determines
        # the window, not a density-dependent y-axis that can hide differences.
        minimum,maximum=np.inf,-np.inf
        lo,hi=np.inf,-np.inf
        for f in files:
            z=read_frame(f,c,meta['perspective'])
            minimum=min(minimum,z['X'][:,0].min())
            maximum=max(maximum,z['X'][:,0].max())
            lo=min(lo,z['energies'].min()); hi=max(hi,z['energies'].max())
        pad=max(.3,.1*(maximum-minimum))
        limits=dict(R=(minimum-pad,maximum+pad),energy=(lo-.05,hi+.05),
                    density=1.05/(np.sqrt(2*np.pi)*c.sigma_r/np.sqrt(2)))
        selected=set([0,len(files)//2,len(files)-1])
        for i in sorted(selected):
            f=trajectory_figure(c,meta,read_frame(files[i],c,meta['perspective']),rows,limits)
            f.savefig(root/f'plots/trajectory_{i:05d}.png',dpi=160)
            plt.close(f)
        if movies:
            if not FFMpegWriter.isAvailable():
                (root/'plots/movie_unavailable.txt').write_text('ffmpeg unavailable; PNG figures remain valid.\n')
            else:
                # One stable canvas per frame, dimensions are constant. No fake
                # temporal interpolation of trajectory dynamics is inserted.
                writer=FFMpegWriter(fps=12,codec='libx264',extra_args=['-pix_fmt','yuv420p'])
                first=trajectory_figure(c,meta,read_frame(files[0],c,meta['perspective']),rows,limits)
                with writer.saving(first,root/'plots/trajectory_diagnostics.mp4',100):
                    for path in files:
                        frame=trajectory_figure(c,meta,read_frame(path,c,meta['perspective']),rows,limits)
                        writer.fig=frame
                        writer.grab_frame()
                        plt.close(frame)
                plt.close(first)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--movies',action='store_true')
    a=p.parse_args()
    cases=[a.root] if (a.root/'author_settings.json').exists() else [f.parent for f in sorted(a.root.rglob('author_settings.json'))]
    if not cases:
        raise ValueError('No author cases found')
    for root in cases:
        render_case(root,a.movies)


if __name__=='__main__':
    main()
