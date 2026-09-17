"""Render six coupling actions, their sums and Hamiltonian action diagnostics."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from .report_plot_style import MASK_COLOR, SIGNED_CMAP
from .coupled_actions import NAMES

TITLES = (
 r'$\|(-i\partial_q-a)^2\Phi/(2m_p)\|_x$',
 r'$\|(-i\partial_qF/F+a)(-i\partial_q-a)\Phi/m_p\|_x$',
 r'$\|(-i\partial_R-b)^2\Phi/(2M)\|_x$',
 r'$\|(-i\partial_RF/F+b)(-i\partial_R-b)\Phi/M\|_x$',
 r'$|(-i\partial_R+b-\alpha)^2\Lambda/(2M\Lambda)|$',
 r'$|(-i\partial_R\chi/\chi+\alpha)(-i\partial_R+b-\alpha)\Lambda/(M\Lambda)|$',
 r'$\|U^{coup}_{e,pn}\Phi\|_x$', r'$|U^{coup}_{p,n}\Lambda/\Lambda|$', r'$b-\alpha$ (signed)',
)

# Share the renderer and time selection, but keep each physical question on
# its own page instead of squeezing nine fields into one movie.
PANEL_GROUPS = (
    ('electronic_coupling_terms', 'Electronic coupling: four contributions', (0,1,2,3), (2,2)),
    ('proton_heavy_coupling_terms', 'Proton–heavy coupling and relative connection', (4,5,8), (1,3)),
    ('coupling_action_totals', 'Coherent total coupling actions', (6,7), (1,2)),
)


def render(path, output, args):
    from .render_final_visualizations import _save_figure, _save_analysis_movie, _save_individual_frames, _absolute_overlay
    with np.load(path) as data:
        # The signed expectation maps are available for follow-up TDPES work;
        # this magnitude report needs only their compact time summaries.
        d={k:data[k] for k in data.files if k not in ('expectation_real','expectation_imag')}
    if tuple(d['names']) != NAMES:
        raise ValueError('Unknown coupled-action field ordering')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    times=d['times_fs'];magnitudes=d['magnitude']
    obs=dict(q=d['q'],R=d['R'],joint_density=d['rho'])
    finite=d['valid'] & (d['rho'] >= float(d['density_floor']))
    bounds=[]
    for group in ([0,1,2,3,6],[4,5,7]):
        bound=1e-12
        for f in range(len(times)):
            for j in group:
                v=magnitudes[f,j][finite[f]&np.isfinite(magnitudes[f,j])]
                if v.size:bound=max(bound,float(np.quantile(v,.995)))
        bounds.append(bound)
    vals=abs(d['delta'][finite & np.isfinite(d['delta'])])
    delta_bound=max(float(np.quantile(vals,.995)),1e-12) if vals.size else 1.
    source=str(d['source'])
    def build(first, group):
        _, title, indices, shape = group
        fig,axes=plt.subplots(*shape,figsize=(7*shape[1],5.5*shape[0]),
                              squeeze=False,constrained_layout=True)
        images=[]
        for j,ax in zip(indices,axes.flat):
            cmap=plt.get_cmap(SIGNED_CMAP if j==8 else 'magma').copy();cmap.set_bad(MASK_COLOR)
            bound=delta_bound if j==8 else bounds[1 if j in (4,5,7) else 0]
            image=ax.imshow(np.zeros((2,2)),origin='lower',aspect='auto',interpolation='nearest',
                            cmap=cmap,norm=Normalize(-bound if j==8 else 0,bound))
            images.append(image)
            formatter=ScalarFormatter(useMathText=True);formatter.set_powerlimits((-3,3))
            fig.colorbar(image,ax=ax,pad=.025,shrink=.8,format=formatter,
                         label='momentum (a.u.)' if j==8 else 'Ha',extend='both')
            ax.set(title=TITLES[j],xlabel=r'$q$ ($a_0$)',ylabel=r'$R$ ($a_0$)')
            ax.title.set_fontsize(13)
            ax.tick_params(labelsize=11)
        heading=fig.suptitle('',fontsize=16)
        fig.supxlabel('PG; differential central5 diagnostic, not native discrete action. '
                      'Electronic x-norm / proton local action ratio. '+source,fontsize=10)
        def update(f):
            mask=finite[f]
            foundq=np.flatnonzero(mask.any(1));foundR=np.flatnonzero(mask.any(0))
            limits=[]
            for coord,found in ((d['q'],foundq),(d['R'],foundR)):
                if not found.size:found=np.arange(len(coord))
                pad=max(2,int(.08*(found[-1]-found[0]+1)))
                limits.append((coord[max(0,found[0]-pad)],coord[min(len(coord)-1,found[-1]+pad)]))
            for j,ax,image in zip(indices,axes.flat,images):
                value=d['delta'][f] if j==8 else magnitudes[f,j]
                image.set_data(np.ma.masked_where(~mask | ~np.isfinite(value),value).T)
                image.set_extent((d['q'][0],d['q'][-1],d['R'][0],d['R'][-1]))
                ax.set(xlim=limits[0],ylim=limits[1]);_absolute_overlay(ax,obs,f)
            heading.set_text(f'{title} | t={times[f]:.4f} fs')
            return images
        update(first)
        return fig,update
    snapshots=np.unique(np.rint(np.linspace(0,len(times)-1,min(args.snapshot_count,len(times)))).astype(int))
    products=[]
    frames=np.unique(np.rint(np.linspace(0,len(times)-1,min(args.max_frames,len(times)))).astype(int))
    for group in PANEL_GROUPS:
        stem=group[0]
        products.extend(_save_individual_frames(lambda f:build(f,group)[0],snapshots,times,
                                                output/(stem+'_frames'),stem,args.dpi))
        if not args.no_animation:
            fig,update=build(int(frames[0]),group)
            anim=FuncAnimation(fig,lambda i:update(int(frames[i])),frames=len(frames),blit=False)
            products.append(_save_analysis_movie(anim,fig,output,stem+'_movie',args))
    # Magnitudes must be combined as complex actions before taking a norm.
    fig,axes=plt.subplots(4,1,figsize=(13,15),constrained_layout=True)
    for group,ax in zip(([0,1,2,3,6],[4,5,7],[6,8,9,7,10,11,12]),axes[:3]):
        for j in group:ax.plot(times,d['rms'][:,j],marker='.',label=NAMES[j])
        ax.set(ylabel='Occupied-support RMS (Ha)')
        ax.legend(ncol=4,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False)
    for j in range(6):axes[3].plot(times,d['mean_real'][:,j],marker='.',label=NAMES[j])
    axes[3].set(ylabel='Signed real expectation (Ha)',xlabel='time (fs)')
    axes[3].legend(ncol=3,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False)
    for ax in axes:
        ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
    fig.suptitle('PG action strengths and conditional-expectation contributions\n'+source,fontsize=15)
    path=output/'six_coupled_actions_time_summary.png';_save_figure(fig,path,args.dpi);products.append(path)
    return products


def main():
    from .render_final_visualizations import parse_args
    args=parse_args()
    path=Path(args.run)
    if path.is_dir():
        direct=path/'coupled_action_diagnostics.npz'
        path=direct if direct.exists() else path/'coupled_action_analysis/coupled_action_diagnostics.npz'
    directory=path.parent.parent if path.parent.name=='coupled_action_analysis' else path.parent
    output=Path(args.outdir) if args.outdir else directory/'report/final_visualizations'
    render(path,output,args)


if __name__=='__main__':
    main()
