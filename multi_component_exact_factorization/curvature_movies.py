"""PG amplitude-ratio decomposition: saved totals versus continuum diagnostics.

Reuses the audit's unmasked five-point derivatives and bond-to-site momenta.
Only one derived frame is retained; no new full-resolution field cache is made.
"""
import json
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np

from .audit_pg_curvature import curvature_terms, heavy_curvature_terms
from .external_potential import harmonic_potential
from .report_plot_style import MASK_COLOR, SIGNED_CMAP

TDPES1_ZOOM_BOUND_HA = 0.05


def render_curvature_movies(obs, ef, output, args, snapshots):
    from .render_final_visualizations import (
        _frame_focus, _movie_frames, _total_source, _save_individual_frames,
        _save_figure, _save_analysis_movie,
    )
    times, R = obs['times_fs'], obs['R']
    mq = float(obs['options'].get('proton_mass', 1836.15267343))
    mR = float(obs['options'].get('heavy_mass', 1836.15267343))
    V = harmonic_potential(R, obs['options'])
    total1, total2 = _total_source(ef, 1), _total_source(ef, 2)
    frames = _movie_frames(obs, args.max_frames)
    selected = np.unique(np.r_[frames, snapshots])
    floor = args.analysis_focus_floor

    @lru_cache(maxsize=1)
    def values(f):
        terms1 = curvature_terms(obs['joint_density'][f], ef['a'][f], ef['b'][f],
                                 obs['dq'], obs['dR'], mq, mR)
        terms2 = heavy_curvature_terms(obs['heavy_density'][f], ef['alpha'][f], obs['dR'], mR)
        return ((total1[f], *terms1, np.broadcast_to(-V, total1[f].shape)),
                (total2[f], *terms2, -V))

    # Fixed, shared linear scale per movie, not re-normalised each frame.
    bounds = [1e-12, 1e-12]
    records = []
    for f in selected:
        rho1, rho2 = obs['joint_density'][f], obs['heavy_density'][f]
        row = {'time_fs': float(times[f])}
        support1 = (rho1 >= floor*rho1.max()) & (rho1 > 0)
        row['first_level_phase_branch_bonds'] = int(np.count_nonzero(support1 & (
            (np.abs(ef['a'][f]*obs['dq']) > .9*np.pi)
            | (np.abs(ef['b'][f]*obs['dR']) > .9*np.pi))))
        for level, (arrays, rho) in enumerate(zip(values(int(f)), (rho1, rho2))):
            support = (rho >= floor*rho.max()) & (rho > 0)
            for a in arrays:
                finite = np.abs(a[support & np.isfinite(a)])
                if finite.size:
                    bounds[level] = max(bounds[level], float(np.quantile(finite, .995)))
            residual = arrays[0]-sum(arrays[1:])
            valid = support & np.isfinite(residual)
            weight = rho[valid]
            row[f'tdpes{level+1}_residual_rms_Ha'] = (
                float(np.sqrt(np.sum(weight*residual[valid]**2)/weight.sum()))
                if weight.sum() else None)
        records.append(row)

    titles = (
        (r'TDPES1 $\epsilon^{(1)}_{\rm PG}$' '\n(trap excluded; not effective TDPES)',
         r'$\partial_q^2 F/(2m_p F)$', r'$\partial_R^2 F/(2MF)$',
         r'$-a_{\rm site}^2/(2m_p)$', r'$-b_{\rm site}^2/(2M)$',
         r'$-V_{\rm ext}^{R}$' '\nTrap subtraction'),
        (r'TDPES2 $\epsilon^{(2)}_{\rm PG}$' '\n(trap excluded; not effective TDPES)',
         r'$\partial_R^2\chi/(2M\chi)$', r'$-\alpha_{\rm site}^2/(2M)$',
         r'$-V_{\rm ext}^{R}$' '\nTrap subtraction'),
    )
    cmap = plt.get_cmap(SIGNED_CMAP).copy()
    cmap.set_bad(MASK_COLOR)

    def build(level, first, zoom=False):
        columns = 3 if level == 0 else 2
        fig, axes = plt.subplots(2, columns, figsize=(16 if level == 0 else 13, 9),
                                 constrained_layout=True)
        artists = []
        bound = TDPES1_ZOOM_BOUND_HA if zoom else bounds[level]
        for panel, (ax, title) in enumerate(zip(axes.flat, titles[level])):
            ax.set_title(title, fontsize=14, pad=10)
            ax.tick_params(labelsize=11)
            if level == 0:
                artists.append(ax.imshow(np.zeros((2, 2)), origin='lower', aspect='auto',
                                         cmap=cmap, norm=Normalize(-bound, bound), interpolation='nearest'))
                ax.set(xlabel=r'proton $q$ ($a_0$)', ylabel=r'heavy $R$ ($a_0$)')
                ax.set_facecolor(MASK_COLOR)
            else:
                color = ('black', 'tab:green', 'tab:purple', 'tab:orange')[panel]
                artists.append(ax.plot(R, np.zeros_like(R), color=color, lw=2.2 if panel == 0 else 1.8)[0])
                ax.set(xlabel=r'heavy $R$ ($a_0$)', ylabel='energy (Hartree)',
                       ylim=(-1.05*bound, 1.05*bound))
                ax.grid(alpha=.15)
        if level == 0:
            fig.colorbar(artists[0], ax=list(axes.flat), pad=.02, shrink=.88,
                         label='energy (Hartree)', extend='both')
        heading = fig.suptitle('', fontsize=16)
        detail = (r'Fixed colour zoom: $\pm 0.05$ Ha; larger magnitudes saturate, not removed.'
                  if zoom else 'Unmasked amplitude derivatives; bond momenta averaged onto sites; fixed display scales.')
        fig.supxlabel('PG; raw trap-excluded TDPES. Continuum diagnostic, not an exact finite-link identity.\n'
                      +detail, fontsize=11)

        def update(f):
            arrays = values(int(f))[level]
            if level == 0:
                support, (iq, iR), limits = _frame_focus(obs, f, floor)
                for ax, artist, a in zip(axes.flat, artists, arrays):
                    artist.set_data(np.ma.masked_where(~support | ~np.isfinite(a), a)[np.ix_(iq, iR)].T)
                    artist.set_extent((*limits[0], *limits[1]))
                    ax.set(xlim=limits[0], ylim=limits[1])
            else:
                rho = obs['heavy_density'][f]
                support = (rho >= floor*rho.max()) & (rho > 0)
                occupied = np.flatnonzero(support)
                lo, hi = (max(0, occupied[0]-3), min(len(R)-1, occupied[-1]+3)) if occupied.size else (0, len(R)-1)
                for ax, artist, a in zip(axes.flat, artists, arrays):
                    artist.set_ydata(np.where(support & np.isfinite(a), a, np.nan))
                    ax.set_xlim(R[lo], R[hi])
            qualifier = ' [fixed colour zoom]' if zoom else ''
            heading.set_text(f'TDPES{level+1}: amplitude curvature and momentum balance{qualifier} | t={times[f]:.4f} fs')
            return (*artists, heading)
        update(first)
        return fig, update

    products = []
    # Retain both original products. The additional view shares all raw arrays,
    # masks and frame selection; only its Normalize limits differ.
    for level, zoom in ((0, False), (1, False), (0, True)):
        stem = f'tdpes{level+1}_pg_curvature'+('_zoom' if zoom else '')
        images = _save_individual_frames(lambda f: build(level, f, zoom)[0], snapshots, times,
                                         output/(stem+'_frames'), stem, args.dpi)
        products.extend(images)
        fig, axes = plt.subplots(2, 4, figsize=(24, 14), constrained_layout=True)
        for ax in axes.flat:
            ax.axis('off')
        for ax, path in zip(axes.flat, images):
            ax.imshow(plt.imread(path))
        path = output/(stem+'_snapshots.png')
        _save_figure(fig, path, args.dpi)
        products.append(path)
        if not args.no_animation:
            fig, update = build(level, int(frames[0]), zoom)
            animation = FuncAnimation(fig, lambda i: update(int(frames[i])), frames=len(frames), blit=False)
            products.append(_save_analysis_movie(animation, fig, output, stem+'_movie', args))

    # Report saturation rather than quietly claiming all extrema are visible.
    for f, row in zip(selected, records):
        for level, (arrays, rho) in enumerate(zip(values(int(f)), (obs['joint_density'][f], obs['heavy_density'][f]))):
            support = (rho >= floor*rho.max()) & (rho > 0)
            row[f'tdpes{level+1}_clipped_fraction'] = [float(np.mean(np.abs(a[support])>bounds[level])) for a in arrays]
            if level == 0:
                row['tdpes1_zoom_clipped_fraction'] = [float(np.mean(np.abs(a[support])>TDPES1_ZOOM_BOUND_HA)) for a in arrays]
    path = output/'pg_curvature_movies_diagnostics.json'
    path.write_text(json.dumps({'convention': 'external excluded; continuum versus saved TDPES',
                               'density_floor': floor, 'fixed_bounds_Ha': bounds,
                               'tdpes1_zoom_bound_Ha': TDPES1_ZOOM_BOUND_HA,
                               'records': records}, indent=2))
    for level in (1, 2):
        errors = [row[f'tdpes{level}_residual_rms_Ha'] for row in records]
        finite = [error for error in errors if error is not None]
        if finite:
            print(f'TDPES{level} PG curvature: max displayed-frame support-weighted '
                  f'residual RMS={max(finite):.6g} Ha (saved total minus signed terms)', flush=True)
    products.append(path)
    return products
