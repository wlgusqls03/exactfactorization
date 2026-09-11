"""Local BO character, distinguished from physical channel density."""
import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

from .report_plot_style import MASK_COLOR


def population_frame(obs, ef, frame):
    """No renormalization to the two displayed channels: preserve omitted mass."""
    rho = np.asarray(obs['joint_density'][frame], float)
    channels = np.asarray(ef['bo_channel_density_qR'][frame, :2], float)
    if channels.shape != (2, *rho.shape):
        raise ValueError('Local BO population requires stored channels j=0 and j=1')
    local = np.divide(channels, rho[None], out=np.full_like(channels, np.nan), where=rho[None]>0)
    denominator = rho.sum(axis=0)
    conditional = np.divide(channels.sum(axis=1), denominator[None],
                            out=np.full((2, rho.shape[1]), np.nan), where=denominator[None]>0)
    return local, conditional


def render_bo_local_population(obs, ef, output, args, snapshots):
    from .render_final_visualizations import (
        _bo3d_preparation, _movie_frames, _save_individual_frames, _save_figure,
        _save_analysis_movie, _absolute_overlay,
    )
    local_args = copy.copy(args)
    local_args.surface_count = 2
    prep = _bo3d_preparation(obs, ef, local_args)
    if prep['n_states'] != 2:
        raise ValueError('Two BO energies and channel densities are required')
    times, q, R = obs['times_fs'], obs['q'], obs['R']
    frames = _movie_frames(obs, args.max_frames)
    # Keep the same camera/domain/energy zero throughout the movie.
    qi, ri = prep['q_indices'], prep['R_indices']
    qi = qi[np.linspace(0, len(qi)-1, min(len(qi), args.movie_bo3d_q_points), dtype=int)]
    ri = ri[np.linspace(0, len(ri)-1, min(len(ri), args.movie_bo3d_R_points), dtype=int)]
    Q, RR = np.meshgrid(q[qi], R[ri], indexing='ij')
    energy = prep['energies'][:2][:, qi][:, :, ri]
    finite = energy[np.isfinite(energy)]
    lo, hi = float(finite.min()), float(finite.max())
    pad = max(.05*(hi-lo), .001)
    norm = Normalize(0, 100)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(MASK_COLOR)

    def build(first):
        fig = plt.figure(figsize=(16.5, 11.8))
        grid = fig.add_gridspec(2, 2, left=.07, right=.86, bottom=.13,
                               top=.89, hspace=.32, wspace=.27)
        surfaces = [fig.add_subplot(grid[0, j], projection='3d') for j in range(2)]
        plane = fig.add_subplot(grid[1, 0])
        lines_axis = fig.add_subplot(grid[1, 1])
        for j, ax in enumerate(surfaces):
            ax.plot_wireframe(Q, RR, energy[j], rstride=max(1,len(qi)//12),
                              cstride=max(1,len(ri)//12), color='0.6', alpha=.35, linewidth=.45)
            ax.set(xlim=prep['q_limits'], ylim=prep['R_limits'], zlim=(lo-pad, hi+pad),
                   xlabel=r'proton $q$ ($a_0$)', ylabel=r'heavy $R$ ($a_0$)', zlabel='BO energy (Ha)')
            ax.view_init(elev=29, azim=-132)
            ax.set_box_aspect((1.3, 1, .7))
            ax.tick_params(labelsize=10)
            ax.set_title(('Ground' if j==0 else 'First excited')+rf' BO surface | $p_{j}(q,R,t)$',
                         fontsize=14, pad=12)
        image = plane.imshow(np.zeros((len(R),len(q))), origin='lower', aspect='auto',
            extent=(q[0],q[-1],R[0],R[-1]), cmap=cmap, norm=norm, interpolation='nearest')
        plane.set_facecolor(MASK_COLOR)
        plane.set(xlim=prep['q_limits'], ylim=prep['R_limits'],
                  xlabel=r'proton $q$ ($a_0$)', ylabel=r'heavy $R$ ($a_0$)')
        plane.set_title(r'Local ground character $p_0=\rho_0/\rho_{qR}$', fontsize=14, pad=12)
        lines = [lines_axis.plot(R, np.zeros_like(R), lw=2.3, color=color, label=label)[0]
                 for color,label in [('tab:blue',r'Ground $P_0(R,t)$'),
                                     ('tab:orange',r'Excited $P_1(R,t)$'),
                                     ('0.45','Other / unrepresented')]]
        lines[2].set_linestyle(':')
        # Small fixed padding keeps exactly 0%/100% curves off the axis spines.
        lines_axis.set(xlim=prep['R_limits'], ylim=(-2,102), yticks=[0,20,40,60,80,100], xlabel=r'heavy $R$ ($a_0$)',
                       ylabel='Proton-averaged BO population')
        lines_axis.yaxis.set_major_formatter(PercentFormatter(100))
        lines_axis.set_title(r'$P_j(R,t)=\int dq\,|\Lambda_R|^2p_j$', fontsize=14, pad=12)
        lines_axis.grid(alpha=.18)
        silhouette, = lines_axis.plot(R, np.zeros_like(R), color='forestgreen', alpha=.35,
                                       lw=1.2, label='Heavy density (scaled guide)')
        for ax in (plane, lines_axis):
            ax.tick_params(labelsize=11)
            ax.xaxis.label.set_size(12)
            ax.yaxis.label.set_size(12)
        cax = fig.add_axes((.91,.30,.016,.47))
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                           ticks=[0,20,40,60,80,100], format=PercentFormatter(100))
        bar.set_label('Local BO population (fixed scale)', fontsize=12)
        bar.ax.tick_params(labelsize=11)
        lines_axis.legend(loc='upper center', bbox_to_anchor=(.5,-.21), ncol=2,
                          frameon=False, fontsize=10)
        heading = fig.suptitle('', fontsize=17, fontweight='bold')
        fig.text(.47,.945, r'Color = local BO character; height = fixed BO energy (trap excluded)',
                 ha='center', fontsize=12)
        fig.text(.07,.045, r'Colored support: $\rho_{qR}\geq10^{-3}\,a_0^{-2}$; grey wireframe = BO landscape.'
                 '\nLocal populations, not transition rates; density contours use absolute values.', fontsize=11)
        colored = []

        def update(frame):
            local, conditional = population_frame(obs, ef, frame)
            rho = obs['joint_density'][frame]
            active = np.isfinite(rho) & (rho>=1e-3)
            for surface in colored:
                surface.remove()
            colored.clear()
            for j, ax in enumerate(surfaces):
                occupied = active[np.ix_(qi,ri)] & np.isfinite(local[j][np.ix_(qi,ri)])
                colors = cmap(norm(100*local[j][np.ix_(qi,ri)]))
                colors[...,3] = occupied.astype(float)
                colored.append(ax.plot_surface(Q, RR, np.where(occupied, energy[j], np.nan),
                    facecolors=colors, shade=False, linewidth=0, antialiased=False,
                    rcount=len(qi), ccount=len(ri)))
            image.set_data(np.ma.masked_where(~active, 100*local[0]).T)
            _absolute_overlay(plane, obs, frame)
            occupied_R = np.any(active, axis=0)
            for j in range(2):
                lines[j].set_ydata(np.where(occupied_R,100*conditional[j],np.nan))
            remaining = 1-conditional.sum(axis=0)
            lines[2].set_ydata(np.where(occupied_R,100*remaining,np.nan))
            lines[2].set_visible(bool(np.any(remaining[occupied_R]>1e-6)))
            heavy = obs['heavy_density'][frame]
            silhouette.set_ydata(15*heavy/max(float(heavy.max()),1e-300))
            heading.set_text(f'Configuration-resolved BO character | t={times[frame]:.4f} fs')
            return (image, *lines, silhouette, heading, *colored)
        update(int(first))
        return fig, update

    output = Path(output)
    stem = 'bo_local_population'
    products = _save_individual_frames(lambda f:build(f)[0], snapshots, times,
                                       output/(stem+'_frames'), stem, args.dpi)
    fig, axes = plt.subplots(2,4,figsize=(26,19),constrained_layout=True)
    for ax in axes.flat:
        ax.axis('off')
    for ax,path in zip(axes.flat,products):
        ax.imshow(plt.imread(path))
    path=output/(stem+'_snapshots.png')
    _save_figure(fig,path,args.dpi)
    products.append(path)
    if not args.no_animation:
        fig,update=build(frames[0])
        animation=FuncAnimation(fig,lambda i:update(int(frames[i])),frames=len(frames),blit=False)
        products.append(_save_analysis_movie(animation,fig,output,stem+'_movie',args))
    return products
