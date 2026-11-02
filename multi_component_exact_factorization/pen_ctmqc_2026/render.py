"""Direct CTMQC PNG/MP4 diagnostics with explicit approximation labels.

No a/b/alpha or exact epsilon/GI/GD is synthesized. Trajectory-population-
weighted CBO/Pol energies are named as such; they omit EF geometric terms.
All line/color scales are fixed over the rendered time range. Only 2D cameras
follow density support. Mixed density matrices are never given a wave phase.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import MaxNLocator
from ..pen_perspectives_2026.model import Config
from ..pen_perspectives_2026.render import png_indices
from ..pen_perspectives_2026.workflow import atomic_json, load_npz
from .core import Settings
from .diagnostics import prepare_frame

COLORS = ['#161616', '#0072B2', '#D55E00', '#009E73', '#8E44AD', '#E69F00']
STYLE = {'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
         'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 11,
         'figure.titlesize': 21, 'axes.spines.top': False, 'axes.spines.right': False}


def limits(values, symmetric=False):
    values = np.concatenate([np.asarray(v).ravel() for v in values])
    values = values[np.isfinite(values)]
    if not len(values):
        return (-1., 1.)
    lo, hi = float(values.min()), float(values.max())
    if symmetric:
        hi = max(abs(lo), abs(hi), 1e-12)
        lo = -hi
    span = max(hi-lo, abs(hi)*.02, 1e-8)
    ticks = MaxNLocator(5).tick_values(lo-.06*span, hi+.06*span)
    return float(ticks[0]), float(ticks[-1])


def decorate(ax, xlabel, ylabel, title):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(alpha=.16)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=min(4, len(handles)), frameon=False)


def silhouette(ax, x, y):
    twin = ax.twinx()
    finite = np.where(np.isfinite(y), y, 0)
    twin.fill_between(x, 0, finite, color='.64', alpha=.35)
    twin.set_ylim(0, max(float(np.max(finite)), 1e-30)*1.4)
    twin.set_yticks([])
    twin.set_zorder(ax.get_zorder()-1)
    ax.patch.set_alpha(0)
    for spine in twin.spines.values():
        spine.set_visible(False)


def new_figure(c, s, f, title, rows=2):
    fig, axes = plt.subplots(rows, 2, figsize=(18, 5.4*rows))
    fig.subplots_adjust(left=.075, right=.97, bottom=.13, top=.87 if rows == 2 else .9, hspace=.69, wspace=.28)
    fig.suptitle(f'{s.algorithm.upper()} {s.perspective.upper()} | {title}\n'
                 +fr'$t={f["t_fs"]:.3f}$ fs; $\omega_c={c.omega:g}$; $g={c.g:g}$; '
                 +f'DSE {"on" if c.dse else "off"}; Ntraj={s.ntraj}; direct ensemble diagnostics', y=.987)
    for i, ax in enumerate(axes.flat):
        ax.text(-.1, 1.05, chr(65+i), transform=ax.transAxes, fontweight='bold', fontsize=18)
    return fig, axes


def map_panel(fig, ax, f, key, scale, window, title, label, signed=False):
    im = ax.pcolormesh(f['R'], f['q'], f[key].T, shading='auto', cmap='RdBu_r' if signed else 'viridis', vmin=scale[0], vmax=scale[1], rasterized=True)
    relative = f['joint']/max(float(f['joint'].max()), 1e-300)
    ax.contour(f['R'], f['q'], relative.T, levels=np.r_[np.geomspace(1e-4, .08, 8), np.arange(.1, 1., .075)], colors='black', linewidths=.7)
    ax.set(xlim=window[:2], ylim=window[2:], xlabel=r'$R$ ($a_0$)', ylabel=r'$q_c$ (a.u.)', title=title)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    bar = fig.colorbar(im, ax=ax, pad=.025)
    bar.set_label(label, fontsize=14)
    bar.ax.tick_params(labelsize=12)
    bar.locator = MaxNLocator(5)
    bar.update_ticks()


def trace(ax, data, key, labels=None, ylabel='', title=''):
    values = np.asarray(data[key])
    if values.ndim == 1:
        ax.plot(data['t_fs'], values, color=COLORS[1], lw=2, label=labels[0] if labels else None)
    else:
        for i in range(values.shape[1]):
            ax.plot(data['t_fs'], values[:, i], color=COLORS[i % len(COLORS)], lw=2, label=labels[i] if labels else str(i))
    decorate(ax, '$t$ (fs)', ylabel, title)


def overview(c, s, f, data, scales, window):
    fig, axes = new_figure(c, s, f, 'Dynamics overview', 3)
    for ax, x, density, label, title in ((axes[0, 0], f['R'], f['rho_R'], '$R$ ($a_0$)', 'Nuclear ensemble density'),
                                        (axes[0, 1], f['q'], f['rho_q'], '$q_c$ (a.u.)', 'Photon-coordinate probability')):
        ax.fill_between(x, density, color='.65', alpha=.45)
        ax.plot(x, density, color=COLORS[1], lw=2)
        ax.set_xlim(scales['R_range'] if label.startswith('$R') else scales['q_range'])
        ax.set_ylim(scales['rhoR' if label.startswith('$R') else 'rhoq'])
        decorate(ax, label, 'Density', title)
    trace(axes[1, 0], data, 'bo_population', ['$S_0$', '$S_1$'], 'Population', 'Bare BO populations')
    key = 'pol_population' if s.perspective == 'pol' else 'cbo_population'
    labels = [fr'$|\mathrm{{Pol}}_{i}\rangle$' for i in range(2*c.nf)] if s.perspective == 'pol' else ['$\mathrm{CBO}_0$', '$\mathrm{CBO}_1$']
    trace(axes[1, 1], data, key, labels, 'Population', 'Quantum-subsystem populations')
    for ax in axes[1]:
        ax.set_ylim(-.025, 1.025)
    trace(axes[2, 0], data, 'photon_number', None, r'$\langle N\rangle$', 'Photon number' + (' (Wigner estimator)' if s.perspective == 'el' else ' (quantum)'))
    values, error = np.array(data['photon_number']), np.array(data['photon_number_se'])
    axes[2, 0].fill_between(data['t_fs'], values-error, values+error, alpha=.18, color=COLORS[1])
    trace(axes[2, 1], data, 'kinetic_R', None, 'Energy (Ha)', 'Nuclear kinetic energy')
    for ax in axes[1:].flat:
        ax.axvline(f['t_fs'], color='.3', ls='--', lw=1)
    return fig


def fields(c, s, f, data, scales, window):
    fig, axes = new_figure(c, s, f, 'Direct fields (not exact TDPES)', 3)
    i = int(np.argmin(abs(f['R']-c.r2)))
    map_panel(fig, axes[0, 0], f, 'joint', scales['joint'], window, 'Joint probability', 'Density')
    map_panel(fig, axes[1, 0], f, 'local_CBO_energy', scales['local_energy'], window, 'Conditional mean CBO energy', 'Energy (Ha)')
    shown = dict(f, upper_cbo=f['local_CBO_population'][:, :, 1])
    map_panel(fig, axes[2, 0], shown, 'upper_cbo', (0., 1.), window, 'Local upper CBO population', 'Population')
    for ax in axes[:, 0]:
        ax.axvline(f['R'][i], color='black', ls='--', lw=1.1)
    mask = f['rho_R'] > 1e-4*f['rho_R'].max()
    ax = axes[0, 1]
    silhouette(ax, f['R'], f['rho_R'])
    ax.plot(f['R'], np.where(mask, f['mean_energy_R'], np.nan), color='black', lw=2.5, label='Weighted subsystem energy')
    if s.perspective == 'pol':
        for k in range(min(4, 2*c.nf)):
            ax.plot(f['R'], f['pol_energy'][:, k], '--', color=COLORS[k+1], lw=1.5, label=fr'$|\mathrm{{Pol}}_{k}\rangle$')
    ax.set_ylim(scales['outer_energy'])
    decorate(ax, '$R$ ($a_0$)', 'Energy (Ha)', 'Population-weighted energy (geometric terms omitted)')
    ax = axes[1, 1]
    silhouette(ax, f['R'], f['rho_R'])
    for y, label, color in ((f['mf_force_R'], 'Mean-field', COLORS[1]), (f['ct_force_R'], 'CT correction', COLORS[2]),
                            (f['mf_force_R']+f['ct_force_R'], 'Total', 'black')):
        ax.plot(f['R'], np.where(mask, y, np.nan), color=color, lw=2, label=label)
    ax.axhline(0, color='.4', ls='--', lw=1)
    ax.set_ylim(scales['force'])
    decorate(ax, '$R$ ($a_0$)', 'Force (a.u.)', 'Force actually driving nuclear trajectories')
    ax = axes[2, 1]
    for k, label in enumerate(f['state_labels']):
        ax.plot(f['R'], f['state_density'][:, k], lw=2, color=COLORS[k % len(COLORS)], label=label)
    ax.set_ylim(scales['rhoR'])
    decorate(ax, '$R$ ($a_0$)', 'Density', 'State-resolved nuclear ensemble')
    for ax in axes[:, 1]:
        ax.set_xlim(scales['R_range'])
    return fig


def slices(c, s, f, data, scales, window):
    fig, axes = new_figure(c, s, f, 'CBO slice diagnostics; mixed ensemble averages')
    i, j = int(np.argmax(f['rho_R'])), int(np.argmin(abs(f['q'])))
    for col, (x, select, density, xlabel, desc) in enumerate((
            (f['R'], (slice(None), j), f['joint'][:, j], '$R$ ($a_0$)', fr'$q_c={f["q"][j]:.2f}$'),
            (f['q'], (i, slice(None)), f['conditional_probability'][i], '$q_c$ (a.u.)', fr'$R_{{peak}}={f["R"][i]:.2f}\,a_0$'))):
        valid = density > max(float(np.max(density)), 1e-300)*1e-4
        ax = axes[0, col]
        silhouette(ax, x, density)
        ax.plot(x, np.where(valid, f['local_CBO_energy'][select], np.nan), color='black', lw=2.5, label='Conditional mean CBO energy')
        for k in range(2):
            ax.plot(x, f['cbo'][select][:, k], '--', color=COLORS[k+1], lw=1.8, label=fr'$E^{{CBO}}_{k}$')
        ax.set_ylim(scales['slice_energy'])
        decorate(ax, xlabel, 'Energy (Ha)', 'Slice at '+desc)
        if col == 0:
            # q's harmonic energy can dominate a shared energy axis. Preserve
            # that comparable axis but add an explicitly labelled R-energy zoom.
            inset = ax.inset_axes([.55, .47, .42, .46])
            inset.plot(x, np.where(valid, f['local_CBO_energy'][select], np.nan), color='black', lw=1.5)
            for k in range(2):
                inset.plot(x, f['cbo'][select][:, k], '--', color=COLORS[k+1], lw=1.1)
            inset.set_xlim(scales['R_range'])
            inset.set_ylim(scales['R_slice_energy'])
            inset.set_title('R energy zoom (Ha)', fontsize=11)
            inset.tick_params(labelsize=10)
            inset.xaxis.set_major_locator(MaxNLocator(3))
            inset.yaxis.set_major_locator(MaxNLocator(3))
        ax = axes[1, col]
        silhouette(ax, x, density)
        for k in range(2):
            ax.plot(x, np.where(valid, f['local_CBO_population'][select][:, k], np.nan), color=COLORS[k+1], lw=2, label=fr'$\mathrm{{CBO}}_{k}$')
        ax.set_ylim(-.025, 1.025)
        decorate(ax, xlabel, 'Conditional population', 'State character at '+desc)
        for ax in axes[:, col]:
            ax.set_xlim(scales['R_range'] if col == 0 else scales['q_range'])
    return fig


def corrections(c, s, f, data, scales, window):
    fig, axes = new_figure(c, s, f, 'Coupled-trajectory corrections')
    mask = f['rho_R'] > 1e-4*f['rho_R'].max()
    ax = axes[0, 0]
    silhouette(ax, f['R'], f['rho_R'])
    ax.plot(f['R'], np.where(mask, f['q_pair_R'], np.nan), color=COLORS[1], lw=2)
    ax.set_ylim(scales['q_pair'])
    decorate(ax, '$R$ ($a_0$)', 'Momentum (a.u.)', 'Pair-weighted quantum momentum RMS (not alpha)')
    trace(axes[0, 1], data, 'ct_force_rms', ['$R$'] if s.perspective == 'pol' else ['$R$', '$q_c$'], 'Force (a.u.)', 'CT force RMS over trajectories')
    trace(axes[1, 0], data, 'ct_population_rate', [str(i) for i in range(len(data['ct_population_rate'][0]))], r'Population / $t_{au}$', 'Ensemble CT-only population derivative')
    axes[1, 0].axhline(0, color='.4', ls='--')
    drift = np.array(data['energy'])-data['energy'][0]
    axes[1, 1].plot(data['t_fs'], drift, color=COLORS[2], lw=2)
    axes[1, 1].axhline(0, color='.4', ls='--')
    decorate(axes[1, 1], '$t$ (fs)', 'Energy change (Ha)', 'Ensemble energy drift (not forced to zero)')
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvline(f['t_fs'], color='.3', ls='--', lw=1)
    return fig


def numerical_checks(c, s, f, data):
    """Checks are diagnostics, not a claim of trajectory/threshold convergence."""
    fig, axes = new_figure(c, s, f, 'Numerical checks')
    ax = axes[0, 0]
    ax.plot(data['t_fs'], data['coefficient_norm_max_error'], label='After normalization', lw=2)
    if 'max_norm_step_error_so_far' in data:
        ax.plot(data['t_fs'], data['max_norm_step_error_so_far'], label='Largest raw step error', lw=2)
    decorate(ax, '$t$ (fs)', 'Absolute error', 'Coefficient norm (raw error retained)')
    ax = axes[0, 1]
    ax.plot(data['t_fs'], np.asarray(data['energy'])-data['energy'][0], lw=2, color=COLORS[2])
    ax.axhline(0, color='.4', ls='--')
    decorate(ax, '$t$ (fs)', 'Energy change (Ha)', 'Energy drift; CTMQC is not energy conserving')
    ax = axes[1, 0]
    if 'guard_counts_so_far' in data:
        counts = np.asarray(data['guard_counts_so_far'], float)
        fraction = np.divide(counts[:, 1], counts[:, 0], out=np.zeros(len(counts)), where=counts[:, 0] > 0)
        ax.plot(data['t_fs'], fraction, lw=2, color=COLORS[3])
    else:
        ax.text(.05, .5, 'Cumulative counts unavailable in this earlier cache', transform=ax.transAxes)
    ax.set_ylim(-.025, 1.025)
    decorate(ax, '$t$ (fs)', 'Disabled / attempted', 'Quantum-momentum guard fraction (RK stages)')
    ax = axes[1, 1]
    ax.plot(data['t_fs'], np.max(abs(np.asarray(data['ct_population_rate'])), axis=1), lw=2)
    decorate(ax, '$t$ (fs)', r'Population / $t_{au}$', 'No-net-transfer constraint: largest residual')
    return fig


def render_case(case, movies=True, movie_stride=1, png_every_fs=5., dpi=85, t_max_fs=None):
    started = time.perf_counter()
    case = Path(case)
    c = Config(**json.loads((case/'config.json').read_text()))
    s = Settings(**json.loads((case/'trajectory_config.json').read_text()))
    paths = sorted((case/'frames').glob('trajectories_*.npz'))
    if t_max_fs is not None:
        paths = [p for p in paths if float(load_npz(p)['t_fs']) <= t_max_fs+1e-9]
    if not paths:
        raise ValueError('No CTMQC frames in selected range')
    rows = json.loads((case/'observables.json').read_text())
    rows = [r for r in rows if r['t_fs'] <= float(load_npz(paths[-1])['t_fs'])+1e-9]
    data = {key: [r[key] for r in rows] for key in rows[0]}
    out = case/('plots' if t_max_fs is None else f'plots_to_{t_max_fs:g}fs')
    out.mkdir(exist_ok=True)
    samples = {k: [] for k in ('rhoR', 'rhoq', 'joint', 'local_energy', 'outer_energy', 'slice_energy', 'R_slice_energy', 'force', 'q_pair')}
    windows, times, checks = [], [], []
    for path in paths:
        f = prepare_frame(c, s, load_npz(path))
        support = f['joint'] > 1e-4*f['joint'].max()
        occupied = f['rho_R'] > 1e-4*f['rho_R'].max()
        ir, iq = np.where(support)
        if not len(ir):
            raise ValueError('No ensemble density on display grid; check escaped-trajectory fraction and extend the display range')
        window = (max(c.rmin, f['R'][ir.min()]-.25), min(c.rmax, f['R'][ir.max()]+.25),
                  max(f['q'][0], f['q'][iq.min()]-.5), min(f['q'][-1], f['q'][iq.max()]+.5))
        # Expand immediately, contract slowly, so density cannot leave the camera.
        if windows:
            prev = windows[-1]
            window = tuple(min(v, .85*p+.15*v) if k % 2 == 0 else max(v, .85*p+.15*v)
                           for k, (v, p) in enumerate(zip(window, prev)))
        windows.append(window)
        times.append(f['t_fs'])
        checks.append(dict(t_fs=f['t_fs'], displayed_density_mass=f['display_mass']))
        for key, value in dict(rhoR=f['rho_R'], rhoq=f['rho_q'], joint=f['joint'][support],
                local_energy=f['local_CBO_energy'][support], outer_energy=f['mean_energy_R'][occupied],
                slice_energy=f['cbo'][support].ravel(),
                R_slice_energy=f['cbo'][occupied, np.argmin(abs(f['q']))].ravel(),
                force=np.r_[f['mf_force_R'][occupied], f['ct_force_R'][occupied], (f['mf_force_R']+f['ct_force_R'])[occupied]],
                q_pair=f['q_pair_R'][occupied]).items():
            samples[key].append(np.asarray(value).ravel()[::max(1, np.size(value)//1000)])
        if s.perspective == 'pol':
            samples['outer_energy'].append(f['pol_energy'][occupied].ravel())
    scales = {key: limits(v, symmetric=key == 'force') for key, v in samples.items()}
    for key in ('rhoR', 'rhoq', 'joint', 'q_pair'):
        scales[key] = (0., scales[key][1])
    scales['R_range'] = (min(w[0] for w in windows), max(w[1] for w in windows))
    scales['q_range'] = (min(w[2] for w in windows), max(w[3] for w in windows))
    selected = png_indices(times, png_every_fs)
    movie_indices = sorted(set(range(0, len(paths), movie_stride)) | {len(paths)-1}) if movies else []
    families = dict(dynamics_overview=overview, CTMQC_fields=fields, slice_diagnostics=slices, force_decomposition=corrections)
    with plt.rc_context(STYLE), ExitStack() as stack:
        writers = {}
        for i in sorted(set(selected)|set(movie_indices)):
            f = prepare_frame(c, s, load_npz(paths[i]))
            for name, factory in families.items():
                folder = out/name
                folder.mkdir(exist_ok=True)
                fig = factory(c, s, f, data, scales, windows[i])
                if i in selected:
                    fig.savefig(folder/f'frame_{i:05d}.png', dpi=max(dpi, 125))
                if i in movie_indices:
                    if name not in writers:
                        writer = FFMpegWriter(fps=20, codec='libx264', extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', '-threads', '1', '-crf', '22'])
                        stack.enter_context(writer.saving(fig, folder/f'{name}.mp4', dpi=dpi))
                        writers[name] = writer
                    writers[name].fig = fig
                    writers[name].grab_frame()
                plt.close(fig)
        fig = numerical_checks(c, s, f, data)
        fig.savefig(out/'numerical_checks.png', dpi=max(dpi, 125))
        plt.close(fig)
    report = dict(wall_seconds=time.perf_counter()-started, frames=len(paths), movies=movies,
        png_indices=selected, png_times_fs=[times[i] for i in selected], scales=scales,
        density_checks=checks, output_families=list(families),
        limitations='DIRECT CTMQC quantities only. No reconstructed wave, no exact epsilon/a/b/alpha, no exact continuity identity.',
        density_method='Fixed Gaussian smoothing of ensemble histograms; POL uses local mixed quantum density matrices.')
    atomic_json(out/'render_report.json', report)
    print(f'RENDERED {case}: {len(paths)} frames; {report["wall_seconds"]:.1f}s', flush=True)
    return report


def comparison(root, tdse_root=None, t_max_fs=None):
    """Compare EL/POL directly, optional matched-parameter TDSE read-only overlay."""
    root = Path(root)
    reports = []
    for group in sorted(root.glob('*/*/dse_*/*')):
        paths = [group/view/'observables.json' for view in ('el', 'pol')]
        if not any(p.exists() for p in paths):
            continue
        series = []
        configs = []
        for path, label in zip(paths, ('EL', 'POL')):
            if path.exists():
                rows = json.loads(path.read_text())
                rows = [r for r in rows if t_max_fs is None or r['t_fs'] <= t_max_fs+1e-9]
                if rows:
                    settings = json.loads(path.with_name('trajectory_config.json').read_text())
                    series.append((settings['algorithm'].upper()+' '+label+' ('+settings['initial']+')', rows))
                    configs.append(json.loads(path.with_name('config.json').read_text()))
        note = 'CTMQC perspectives: no exact EF wavefunction reconstruction.'
        if tdse_root is not None and configs:
            tdse_case = Path(tdse_root)/group.relative_to(root)
            if (tdse_case/'observables.json').exists():
                other = json.loads((tdse_case/'config.json').read_text())
                physical = ('mass', 'omega', 'g', 'dse', 'k', 'a', 'b', 'r1', 'r2', 'r3', 'delta', 'z', 'mu12', 'nf')
                if all(other[k] == cfg[k] for cfg in configs for k in physical):
                    rows = json.loads((tdse_case/'observables.json').read_text())
                    rows = [r for r in rows if t_max_fs is None or r['t_fs'] <= t_max_fs+1e-9]
                    if rows:
                        series.append(('TDSE (bare initial; finite Fock)', rows))
                    note += ' EL paper preparation differs from bare TDSE; EL photon space is continuous classical.'
                else:
                    note += ' TDSE overlay SKIPPED: Hamiltonian/Fock mismatch.'
        if not series:
            continue
        out = group/('perspective_comparison' if t_max_fs is None else f'perspective_comparison_to_{t_max_fs:g}fs')
        out.mkdir(exist_ok=True)
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(2, 2, figsize=(17, 11))
            fig.subplots_adjust(top=.87, bottom=.18, hspace=.55, wspace=.3)
            fig.suptitle(f'EL / POL dynamics | {group.relative_to(root)}\n'+r'CTMQC approximations; same model, different quantum/classical partitions', fontsize=20)
            for index, (label, rows) in enumerate(series):
                t = [r['t_fs'] for r in rows]
                for ax, values in zip(axes.flat, ([r['photon_number'] for r in rows], [r['bo_population'][1] for r in rows],
                                                 [r['product_R_ge_4'] for r in rows], [r['signed_flux_R4'] for r in rows])):
                    ax.plot(t, values, lw=2, label=label, color=COLORS[index+1])
            for ax, title, ylabel in zip(axes.flat, ('Photon number', 'Bare excited BO population', 'Probability in R >= 4', 'Signed flux at R = 4 (KDE for CTMQC)'),
                                        (r'$\langle N\rangle$', 'Population', 'Probability', r'Probability / $t_{au}$')):
                decorate(ax, '$t$ (fs)', ylabel, title)
            if any(label.startswith('TDSE') for label, _ in series):
                fig.text(.08, .025, 'TDSE uses bare S1 × vacuum. EL(paper) uses local CBO preparation and classical photon trajectories.', fontsize=12)
            fig.savefig(out/'EL_POL_comparison.png', dpi=140)
            plt.close(fig)
        report = dict(group=str(group), series=[s[0] for s in series], note=note)
        atomic_json(out/'comparison.json', report)
        reports.append(report)
    return reports
