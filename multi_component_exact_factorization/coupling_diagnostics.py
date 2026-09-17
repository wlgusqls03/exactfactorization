"""PG proton--heavy coupling diagnostics, not a native finite-link identity.

Only the p,n action is reconstructed here. Electronic actions require Phi and
are deliberately not inferred from scalar geometry or two BO populations.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter

from .core import derivative, covariant_square
from .report_plot_style import MASK_COLOR, SIGNED_CMAP


def coupling_frame(rho, heavy, b, alpha, dq, dR, mass):
    """Return b-alpha and three complex local action ratios in Hartree.

Forward-bond momenta are averaged to sites, as in curvature diagnostics.
Derivatives are taken BEFORE the display mask. Ratios are meaningful only on
occupied support. D=(-i partial_R+b_site-alpha_site).
    """
    chi = np.sqrt(np.maximum(heavy, 0))
    lam = np.divide(np.sqrt(np.maximum(rho, 0)), chi[None, :],
                    out=np.full_like(rho, np.nan), where=chi[None, :] > 0)
    b_site = (b+np.roll(b, 1, axis=1))/2
    alpha_site = (alpha+np.roll(alpha, 1))/2
    delta = b_site-alpha_site[None, :]
    dlam = -1j*derivative(lam, dR, axis=1)+delta*lam
    logchi = np.divide(derivative(chi, dR, axis=0), chi,
                       out=np.full_like(chi, np.nan), where=chi > 0)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        actions = (covariant_square(lam, delta, dR, axis=1, sign=1)/(2*mass),
                   alpha_site[None, :]*dlam/mass,
                   -1j*logchi[None, :]*dlam/mass)
        ratios = tuple(np.divide(a, lam, out=np.full(a.shape, np.nan+0j),
                                 where=lam > 0) for a in actions)
    return delta, ratios


def summarize(rho, delta, ratios, floor=1e-3):
    valid = (rho >= floor) & np.isfinite(delta)
    for value in ratios:
        valid &= np.isfinite(value)
    weight = np.where(valid, rho, 0.)
    mass = weight.sum()
    if mass <= 0:
        return dict(support_fraction=0., delta_rms=None, rms=[None]*4,
                    real_mean=[None]*4)
    weight /= mass
    fields = (*ratios, sum(ratios))
    return dict(support_fraction=float(mass/rho.sum()),
                delta_rms=float(np.sqrt(np.sum(weight*np.where(valid, delta, 0)**2))),
                rms=[float(np.sqrt(np.sum(weight*np.abs(np.where(valid, a, 0))**2)))
                     for a in fields],
                real_mean=[float(np.sum(weight*np.where(valid, a.real, 0))) for a in fields])


def render_coupling_diagnostics(obs, ef, output, args, snapshots):
    from .render_final_visualizations import (
        _movie_frames, _frame_focus, _absolute_overlay, _save_individual_frames,
        _save_figure, _save_analysis_movie,
    )
    if 'zero' in str(ef.get('gauge', '')):
        raise ValueError('Coupling diagnostics require positive-density gauge')
    times = obs['times_fs']
    mass = float(obs['options']['heavy_mass'])
    def values(f):
        return coupling_frame(obs['joint_density'][f], obs['heavy_density'][f],
                              ef['b'][f], ef['alpha'][f], obs['dq'], obs['dR'], mass)
    records, bounds = [], np.full(4, 1e-12)
    for f, time in enumerate(times):
        delta, ratios = values(f)
        rho = obs['joint_density'][f]
        row = summarize(rho, delta, ratios)
        row['time_fs'] = float(time)
        row['phase_branch_sites'] = int(np.count_nonzero(
            (rho >= 1e-3) & (np.abs(ef['b'][f]*obs['dR']) > .9*np.pi)))
        records.append(row)
        for j, a in enumerate((delta, *ratios)):
            sample = np.abs(a[(rho >= 1e-3) & np.isfinite(a)])
            if sample.size:
                bounds[j] = max(bounds[j], float(np.quantile(sample, .995)))
    # Use one scale for the three action maps for a fair magnitude comparison.
    bounds[1:] = max(bounds[1:])
    path = output/'proton_heavy_coupling_diagnostics.json'
    path.write_text(json.dumps(dict(
        convention='PG; bond-to-site central5 continuum diagnostic; not native link action',
        electronic_actions='not reconstructed: require coherent Phi and its spatial derivatives',
        density_floor=1e-3, fields=['quadratic', 'alpha_transport', 'chi_log_amplitude', 'total'],
        statistics='rho-weighted, renormalized within common finite occupied support',
        records=records), indent=2, allow_nan=False))
    products = [path]
    labels = ('quadratic', 'alpha transport', 'chi amplitude', 'total')
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(times, [r['delta_rms'] for r in records])
    axes[0].set(ylabel=r'RMS $(b-\alpha)_{site}$ (a.u.)')
    for j, label in enumerate(labels):
        axes[1].plot(times, [r['rms'][j] for r in records], label=label)
        axes[2].plot(times, [r['real_mean'][j] for r in records], label=label)
    axes[1].set(ylabel='Coupling action ratio RMS (Ha)')
    axes[2].set(ylabel='Signed real mean (Ha)', xlabel='time (fs)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
    axes[1].legend(ncol=2, loc='lower left', bbox_to_anchor=(0, 1.01), frameon=False)
    axes[2].axhline(0, color='gray', ls='--', lw=.7)
    fig.suptitle('Proton–heavy coupling: occupied-support diagnostics (PG)', fontsize=15)
    path = output/'proton_heavy_coupling_time_summary.png'
    _save_figure(fig, path, args.dpi)
    products.append(path)
    titles = (r'$(b-\alpha)_{site}$',
              r'$|D_R^2\Lambda/(2M\Lambda)|$',
              r'$|\alpha D_R\Lambda/(M\Lambda)|$',
              r'$|-i(\partial_R\chi/\chi)D_R\Lambda/(M\Lambda)|$')
    def build(first):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        images = []
        for j, ax in enumerate(axes.flat):
            cmap = plt.get_cmap(SIGNED_CMAP if j == 0 else 'magma').copy()
            cmap.set_bad(MASK_COLOR)
            im = ax.imshow(np.zeros((2, 2)), origin='lower', aspect='auto',
                           cmap=cmap, norm=Normalize(-bounds[j] if j == 0 else 0, bounds[j]))
            images.append(im)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            fig.colorbar(im, ax=ax, pad=.025, shrink=.85, format=formatter,
                         label='momentum (a.u.)' if j == 0 else 'Ha', extend='both')
            ax.set(title=titles[j], xlabel=r'$q$ ($a_0$)', ylabel=r'$R$ ($a_0$)')
        heading = fig.suptitle('', fontsize=15)
        fig.supxlabel('PG; central5 continuum diagnostic, not exact finite-link action. '
                      'Action maps show magnitude, not a signed TDPES.', fontsize=10)
        def update(f):
            delta, ratios = values(int(f))
            support, (iq, iR), limits = _frame_focus(obs, f, args.analysis_focus_floor)
            for ax, im, a in zip(axes.flat, images, (delta, *map(np.abs, ratios))):
                im.set_data(np.ma.masked_where(~support | ~np.isfinite(a), a)[np.ix_(iq,iR)].T)
                im.set_extent((*limits[0], *limits[1]))
                ax.set(xlim=limits[0], ylim=limits[1])
                _absolute_overlay(ax, obs, f)
            heading.set_text(f'Proton–heavy coupling | t={times[f]:.4f} fs')
            return images
        update(first)
        return fig, update
    images = _save_individual_frames(lambda f: build(f)[0], snapshots, times,
                    output/'proton_heavy_coupling_frames', 'proton_heavy_coupling', args.dpi)
    products.extend(images)
    columns = min(4, len(images))
    rows = int(np.ceil(len(images)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6*columns, 4.5*rows),
                             squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.axis('off')
    for ax, image in zip(axes.flat, images):
        ax.imshow(plt.imread(image))
    path = output/'proton_heavy_coupling_snapshots.png'
    _save_figure(fig, path, args.dpi)
    products.append(path)
    if not args.no_animation:
        frames = _movie_frames(obs, args.max_frames)
        fig, update = build(int(frames[0]))
        animation = FuncAnimation(fig, lambda i: update(int(frames[i])), frames=len(frames), blit=False)
        products.append(_save_analysis_movie(animation, fig, output, 'proton_heavy_coupling_movie', args))
    return products
