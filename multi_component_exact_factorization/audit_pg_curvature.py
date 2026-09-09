"""Streaming positive-gauge amplitude-curvature diagnostic (no propagation).

The saved BO Hamiltonian includes the trap.  Consequently the continuum
comparison for the stored total is Q_q+Q_R-a_site**2/(2m)-b_site**2/(2M),
without subtracting the trap again.  This is NOT an exact finite-link identity.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .core import derivative
from . import tdse_report
from .render_all import find_archive, resolve_run_input
from .render_final_visualizations import _frame_focus, _save_figure, _time_tag
from .report_plot_style import MASK_COLOR, SIGNED_CMAP
from .visualize import selected_frames
from multi_component_exact_factorization_discrete_gpu.compare_tdse import _stream_arrays


def curvature_terms(rho, a_bond, b_bond, dq, dR, mass_q, mass_R):
    """Differentiate unmasked amplitude, then evaluate only where nonzero.

    Average neighboring forward-bond connections onto sites. Principal-phase
    branch artifacts are not unwrapped or silently repaired by this diagnostic.
    """
    F = np.sqrt(np.maximum(rho, 0.0))
    terms = []
    for axis, spacing, mass in ((0, dq, mass_q), (1, dR, mass_R)):
        lap = derivative(F, spacing, axis, order=2)
        terms.append(np.divide(lap, 2*mass*F, out=np.zeros_like(F), where=F>0))
    a = (a_bond+np.roll(a_bond, 1, axis=0))/2
    b = (b_bond+np.roll(b_bond, 1, axis=1))/2
    return (*terms, -a*a/(2*mass_q), -b*b/(2*mass_R))


def render_summary(records, output, dpi):
    t = [r['time_fs'] for r in records]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for name in ('total','Qq','QR','Kq','KR'):
        axes[0].plot(t, [r[name+'_mean'] for r in records], label=name)
    axes[0].set(ylabel='Support-weighted mean (Hartree)', title='Signed contributions (occupied support only)')
    axes[0].legend(ncol=5, loc='lower center', bbox_to_anchor=(.5, 1.10), frameon=False)
    for name in ('total','Qq','QR','Kq','KR','residual'):
        axes[1].semilogy(t, np.maximum([r[name+'_rms'] for r in records], 1e-300), label=name)
    upper = max(r[k+'_rms'] for r in records for k in ('total','Qq','QR','Kq','KR','residual'))
    axes[1].set(xlabel='time (fs)', ylabel='Support-weighted RMS (Hartree)',
                ylim=(1e-7, max(.1, 1.5*upper)))
    axes[1].legend(ncol=6, loc='lower center', bbox_to_anchor=(.5, 1.01), frameon=False)
    axes[1].text(.01,.03, r'Values below $10^{-7}$ Ha are outside this display; retained in JSON.',
                 transform=axes[1].transAxes, fontsize=9)
    _save_figure(fig, Path(output)/'pg_curvature_time_summary.png', dpi)


def run(args):
    archive, directory = find_archive(resolve_run_input(args.run))
    cache = directory/'tdse_exact_factorization_fields.npz'
    output = Path(args.outdir) if args.outdir else directory/'report/final_visualizations/pg_curvature_audit'
    output.mkdir(parents=True, exist_ok=True)
    with np.load(archive, allow_pickle=True) as data:
        options = tdse_report._options(data)
        t, q, R = (data[k] for k in ('times_fs', 'q', 'R'))
    with np.load(cache) as data:
        if 'positive_density' not in str(data['gauge']):
            raise ValueError('This diagnostic requires positive-density gauge fields')
        if not np.allclose(data['times_fs'], t, rtol=0, atol=1e-10):
            raise ValueError('Archive/cache time grids differ')
    mq, mR = float(options['proton_mass']), float(options['heavy_mass'])
    dq, dR = float(q[1]-q[0]), float(R[1]-R[0])
    snapshots = set(selected_frames(len(t), min(args.snapshot_count, len(t))))
    keys = ['a', 'b', 'tdpes1_total']
    records, saved = [], []
    # Only one frame per NPZ member lives in RAM. No large extraction/cache.
    with _stream_arrays(archive, ['joint_density']) as density, _stream_arrays(cache, keys) as fields:
        for frame, time in enumerate(t):
            rho = density['joint_density'].read(frame)
            values = {key: fields[key].read(frame) for key in keys}
            Qq, QR, Kq, KR = curvature_terms(rho, values['a'], values['b'], dq, dR, mq, mR)
            total = values['tdpes1_total']
            recon = Qq+QR+Kq+KR
            residual = total-recon
            support = rho >= args.density_floor*rho.max()
            weight = np.where(support, rho, 0.0)
            weight /= weight.sum()
            row = {'time_fs': float(time), 'support_mass_fraction': float(rho[support].sum()/rho.sum())}
            for name, value in zip(('total', 'Qq', 'QR', 'Kq', 'KR', 'reconstruction', 'residual'),
                                   (total, Qq, QR, Kq, KR, recon, residual)):
                row[name+'_mean'] = float(np.sum(weight*value))
                row[name+'_rms'] = float(np.sqrt(np.sum(weight*value**2)))
            row['phase_branch_bonds'] = int(np.count_nonzero(support &
                ((np.abs(values['a']*dq)>.9*np.pi) | (np.abs(values['b']*dR)>.9*np.pi))))
            records.append(row)
            if frame in snapshots:
                saved.append((frame, rho.copy(), (total, Qq, QR, Kq, KR, recon, residual)))
            if frame % 50 == 0:
                print(f'curvature audit {frame+1}/{len(t)}; residual RMS={row["residual_rms"]:.4g} Ha', flush=True)
    # Fixed common scale across selected snapshots. Out-of-scale values remain
    # visible as saturated colours; report clipping fractions explicitly.
    scale = max(float(np.quantile(np.abs(v[rho>=args.density_floor*rho.max()]), .99))
                for _, rho, arrays in saved for v in arrays)
    scale = max(scale, 1e-12)
    titles = (r'Saved total $\epsilon^{(1)}_{\rm PG}$',
              r'$Q_q=(\partial_q^2 F)/(2m_p F)$', r'$Q_R=(\partial_R^2 F)/(2MF)$',
              r'$-a_{\rm site}^2/(2m_p)$', r'$-b_{\rm site}^2/(2M)$',
              r'$Q_q+Q_R-a_{\rm site}^2/(2m_p)-b_{\rm site}^2/(2M)$',
              'Residual: saved total minus continuum diagnostic')
    for frame, rho, arrays in saved:
        obs = {'joint_density': [rho], 'q': q, 'R': R}
        active, (iq, iR), limits = _frame_focus(obs, 0, args.density_floor)
        fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
        cmap = plt.get_cmap(SIGNED_CMAP).copy(); cmap.set_bad(MASK_COLOR)
        clipped = []
        for ax, value, title in zip(axes.flat, arrays, titles):
            clipped.append(float(np.mean(np.abs(value[active])>scale)))
            shown = np.ma.masked_where(~active, value)[np.ix_(iq, iR)]
            im = ax.imshow(shown.T, origin='lower', aspect='auto', cmap=cmap,
                           norm=Normalize(-scale, scale), extent=(*limits[0], *limits[1]))
            ax.set(title=title, xlabel=r'proton $q$ ($a_0$)', ylabel=r'heavy $R$ ($a_0$)')
            ax.set_facecolor(MASK_COLOR)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.02, .9, 'Continuum diagnostic, not a finite-link identity.\n\n'
                          'Trap is already inside the saved BO Hamiltonian.\n'
                          'Do not subtract it again.\n\n'
                          'Amplitude differentiated before density masking.\n'
                          'Bond connections averaged onto sites.\n\n'
                          f'Displayed support: density >= {args.density_floor:g} peak\n'
                          f'Fixed colour range: +/- {scale:.4f} Hartree', va='top', fontsize=11)
        fig.colorbar(im, ax=list(axes.flat[:7]), shrink=.8, pad=.015, label='Hartree', extend='both')
        fig.suptitle(f'Positive-gauge curvature / momentum balance | t={t[frame]:.4f} fs', fontsize=16)
        _save_figure(fig, output/f'pg_curvature_{_time_tag(t[frame])}.png', args.dpi)
        records[frame]['snapshot_clipped_fraction'] = dict(zip(('total','Qq','QR','Kq','KR','reconstruction','residual'), clipped))
    render_summary(records, output, args.dpi)
    report = {'convention': 'trap included; continuum comparison, not exact finite-link identity',
              'density_floor': args.density_floor, 'fixed_color_bound_Ha': scale, 'records': records}
    (output/'pg_curvature_audit.json').write_text(json.dumps(report, indent=2))
    print(f'Curvature audit saved: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--outdir')
    parser.add_argument('--density-floor', type=float, default=1e-3)
    parser.add_argument('--snapshot-count', type=int, default=8)
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()
    if not 0 < args.density_floor <= 1 or args.snapshot_count < 1:
        parser.error('density-floor must be in (0,1]; snapshot-count must be positive')
    run(args)


if __name__ == '__main__':
    main()
