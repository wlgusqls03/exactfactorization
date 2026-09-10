"""TDPES versus effective TDPES, using already-loaded PG scalar arrays."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

from .external_potential import harmonic_potential
from .report_plot_style import SIGNED_CMAP, MASK_COLOR


def render_external_comparison(obs, ef, output, args, snapshots):
    from .render_final_visualizations import (
        _frame_focus, _movie_frames, _total_source, _save_individual_frames,
        _save_figure, _save_analysis_movie, _absolute_overlay,
    )
    t, q, R = obs['times_fs'], obs['q'], obs['R']
    V = harmonic_potential(R, obs['options'])
    e1, e2 = _total_source(ef, 1), _total_source(ef, 2)
    frames = _movie_frames(obs, args.max_frames)
    bound, low, high = 1e-6, np.inf, -np.inf
    for f in np.unique(np.r_[frames, snapshots]):
        mask, _, _ = _frame_focus(obs, f, args.analysis_focus_floor)
        for a in (e1[f], np.broadcast_to(V, e1[f].shape), e1[f]+V):
            if np.any(mask):
                bound = max(bound, float(np.max(np.abs(a[mask]))))
        support = obs['heavy_density'][f] >= args.analysis_focus_floor*obs['heavy_density'][f].max()
        for a in (e2[f], V, e2[f]+V):
            low, high = min(low,float(a[support].min())), max(high,float(a[support].max()))
    padding = max(.08*(high-low),1e-6)
    cmap = plt.get_cmap(SIGNED_CMAP).copy(); cmap.set_bad(MASK_COLOR)

    def build(frame):
        fig = plt.figure(figsize=(15,9), constrained_layout=True)
        grid = fig.add_gridspec(2,3, height_ratios=(1, .9))
        axes = [fig.add_subplot(grid[0,i]) for i in range(3)]
        bottom = fig.add_subplot(grid[1,:])
        images = []
        for axis, title in zip(axes, ('TDPES1 (trap excluded)', 'External harmonic', 'Effective TDPES1 = TDPES1 + external')):
            im = axis.imshow(np.zeros((2,2)), origin='lower', aspect='auto',
                             cmap=cmap, norm=Normalize(-bound,bound))
            axis.set(title=title, xlabel=r'proton $q$ ($a_0$)', ylabel=r'heavy $R$ ($a_0$)')
            axis.set_facecolor(MASK_COLOR); images.append(im)
        fig.colorbar(images[0], ax=axes, shrink=.85, pad=.02, label='energy (Hartree)')
        lines = [bottom.plot(R,np.zeros_like(R),style,color=color,lw=width,label=label)[0]
                 for style,color,width,label in (
                     ('--','tab:blue',1.8,'TDPES2 (trap excluded)'),
                     (':','tab:orange',2.,'External harmonic'),
                     ('-','black',2.4,'Effective TDPES2 = TDPES2 + external'))]
        bottom.set(xlabel=r'heavy $R$ ($a_0$)', ylabel='energy (Hartree)', ylim=(low-padding,high+padding))
        bottom.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,fontsize=11)
        bottom.grid(alpha=.15)
        title = fig.suptitle('', fontsize=15)

        def update(f):
            mask,(iq,iR),limits = _frame_focus(obs,f,args.analysis_focus_floor)
            for im,axis,a in zip(images,axes,(e1[f],np.broadcast_to(V,e1[f].shape),e1[f]+V)):
                im.set_data(np.ma.masked_where(~mask,a)[np.ix_(iq,iR)].T)
                im.set_extent((*limits[0],*limits[1])); axis.set(xlim=limits[0],ylim=limits[1])
                _absolute_overlay(axis, obs, f)
            for line,a in zip(lines,(e2[f],V,e2[f]+V)):
                line.set_ydata(a)
            occupied = obs['heavy_density'][f]>=args.analysis_focus_floor*obs['heavy_density'][f].max()
            found = np.flatnonzero(occupied)
            if found.size:
                bottom.set_xlim(R[max(0,found[0]-3)],R[min(len(R)-1,found[-1]+3)])
            title.set_text(f'TDPES / external confinement / effective TDPES | t={t[f]:.4f} fs\n'
                           'Positive gauge; raw energies; fixed colour and energy scales')
            return (*images,*lines,title)
        update(frame)
        return fig, update

    output = Path(output)
    products = _save_individual_frames(lambda f:build(f)[0], snapshots, t,
                output/'external_harmonic_comparison_frames', 'external_harmonic_comparison', args.dpi)
    # Reuse exactly the saved frame artwork in a compact 8-time montage.
    fig, axes = plt.subplots(2,4,figsize=(24,14),constrained_layout=True)
    for axis in axes.flat: axis.axis('off')
    for axis,path in zip(axes.flat,products): axis.imshow(plt.imread(path))
    summary = output/'external_harmonic_comparison_snapshots.png'
    _save_figure(fig,summary,args.dpi); products.append(summary)
    if not args.no_animation:
        fig, update = build(int(frames[0]))
        animation = FuncAnimation(fig,lambda i:update(int(frames[i])),frames=len(frames),blit=False)
        products.append(_save_analysis_movie(animation,fig,output,'external_harmonic_comparison_movie',args))
    return products
