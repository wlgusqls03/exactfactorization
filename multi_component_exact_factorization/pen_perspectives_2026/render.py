"""Streaming reports from compact Fock coefficients; no TDSE and no field archive.

Reuses the existing cavity renderer's map, contour, silhouette, A and slice
layout helpers. Overrides source/consistency semantics for finite Fock space.
The unprojected local CBO reference is not confused with a polaritonic PES.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import time
import resource
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from multi_component_exact_factorization_cavity import render as style
from .fields import analyze, diagnostics
from .model import Config, Propagator, cbo, oscillator
from .workflow import atomic_json, load_npz

COLORS = ['#161616', '#0072B2', '#D55E00', '#009E73', '#8E44AD']


def frame_paths(case, t_max_fs=None):
    paths = sorted((Path(case)/'coefficients').glob('wave_*.npz'))
    if t_max_fs is not None and (not np.isfinite(t_max_fs) or t_max_fs < 0):
        raise ValueError('t-max-fs must be finite and nonnegative')
    paths = [p for p in paths if t_max_fs is None or float(load_npz(p)['t_fs']) <= t_max_fs+1e-9]
    if not paths:
        raise ValueError(f'No coefficient frames in {case} for requested time range')
    return paths


def prepare(paths, p, nq):
    """Two-pass rendering: retain only scales, cameras and small time traces."""
    samples = {k: [] for k in ('epsilon1', 'a', 'b', 'energy1', 'energy2', 'momentum', 'source',
                               'alpha_direct', 'e2_direct', 'alpha_residual', 'e2_residual', 'bridge_terms')}
    windows, checks, cbop = [], [], []
    previous = None
    first = None
    for path in paths:
        z = load_npz(path)
        f = analyze(z['u'], p, float(z['t_fs']), nq)
        if first is None:
            first = f
        support = f['rho_qR'] > 1e-5*f['rho_qR'].max()
        outer = f['rho_R'] > 1e-5*f['rho_R'].max()
        def collect(name, values):
            v = np.asarray(values).real.ravel()
            samples[name].append(v[::max(1, len(v)//1200)])
        for key in ('epsilon1', 'a', 'b'):
            collect(key, f[key][support])
        for key in ('b', 'delta_b'):
            collect('momentum', f[key][support])
        collect('momentum', f['alpha'][outer])
        for key in ('density_dt', 'S_q', 'S_adv', 'S_rel', 'S_projection'):
            collect('source', f[key][support])
        for level, mask in ((1, support), (2, outer)):
            for suffix in ('', '_GI', '_GD'):
                collect(f'energy{level}', f[f'epsilon{level}{suffix}'][mask])
        for key, short in (('alpha', 'alpha'), ('epsilon2', 'e2')):
            collect(short+'_direct', f[key][outer])
            collect(short+'_residual', (f[key]-f[key+'_reconstructed'])[outer])
        for key in ('mean_e1', 'Kq', 'G_R_lambda'):
            collect('bridge_terms', f[key][outer])
        previous = style._window(f, previous)
        windows.append(previous)
        checks.append(dict(t_fs=float(z['t_fs']), **diagnostics(f)))
        cbop.append(f['cbo_population'].tolist())
    scales = {key: style._range(values, symmetric=key not in ('epsilon1', 'energy1', 'energy2', 'bridge_terms', 'e2_direct'))
              for key, values in samples.items()}
    q = first['q']
    refs = dict(R=p.R, q=q, cbo=cbo(p.c, p.R, q)[0], pol_energy=p.pol_energy)
    rlim = (min(w[0] for w in windows), max(w[1] for w in windows))
    qlim = (min(w[2] for w in windows), max(w[3] for w in windows))
    # Keep reference tails drawn, but don't let empty-box large energies crush
    # the occupied structure. Limits stay fixed throughout the movie.
    rmask = (p.R >= rlim[0]) & (p.R <= rlim[1])
    qmask = (q >= qlim[0]) & (q <= qlim[1])
    for level, values in ((1, refs['cbo'][rmask][:, qmask]), (2, p.pol_energy[rmask])):
        lo, hi = scales[f'energy{level}']
        ticks = style.MaxNLocator(5).tick_values(min(lo, values.min()), max(hi, values.max()))
        scales[f'energy{level}'] = (float(ticks[0]), float(ticks[-1]))
    display = dict(scales=scales, windows=windows, i=int(np.argmin(abs(p.R-p.c.r2))),
                   j=int(np.argmin(abs(q))), line_xlim_R=rlim, line_xlim_q=qlim,
                   slices_R=[int(np.argmin(abs(p.R-r))) for r in (p.c.r2, 4.)],
                   slices_q=[int(np.argmin(abs(q-v))) for v in (0., 1.5)])
    return display, refs, checks, cbop


def stamp(fig, f, title=None):
    if title is None:
        title = fig._suptitle.get_text().splitlines()[0].split('    ')[0]
        while title.startswith('PEN perspectives 2026 | '):
            title = title[len('PEN perspectives 2026 | '):]
    fig.suptitle(f'PEN perspectives 2026 | {title}\n'
                 +fr'$t={f["t_fs"]:.2f}$ fs; $g={f["g_cavity"]:g}$; DSE {"on" if f["dse"] else "off"}; '
                 +'2 electronic states × finite Fock basis; positive marginal gauge', fontsize=21, y=.992)


def sources(f, display, index, refs):
    fig = style.source_figure(f, display, index, refs)
    ax = [a for a in fig.axes if a.get_title().startswith('Source terms')][0]
    i = display['i']
    mask = style._line_support(f, 1, i)
    ax.plot(f['q'], np.where(mask, f['S_projection'][i], np.nan), color=COLORS[4], lw=2,
            label=r'$S_{\mathrm{projection}}$')
    style._legend(ax, 3)
    stamp(fig, f, r'$\partial_t|\Lambda_R|^2=S_q+S_{\rm adv}+S_{\rm rel}+S_{\rm projection}$')
    return fig


def consistency(f, display, index, refs):
    fig = style.consistency_figure(f, display, index, refs)
    ax = [a for a in fig.axes if a.get_title().startswith('Continuity reconstruction')][0]
    i = display['i']
    support = style._line_support(f, 1, i)
    for line in ax.lines:
        if line.get_label() == 'sum of sources':
            line.set_ydata(np.where(support, sum(f[k][i] for k in ('S_q', 'S_adv', 'S_rel', 'S_projection')), np.nan))
            line.set_label('sum including projection source')
    style._legend(ax, 2)
    stamp(fig, f, 'Bridge consistency (including finite-Fock correction)')
    return fig


def gi_components(f, display, refs):
    fig, axes = plt.subplots(2, 2, figsize=(21, 13))
    fig.subplots_adjust(left=.08, right=.97, top=.87, bottom=.12, hspace=.7, wspace=.28)
    i = display['i']
    for col, level in enumerate((1, 2)):
        x = f['q'] if level == 1 else f['R']
        sel = i if level == 1 else slice(None)
        mask = style._line_support(f, level, i if level == 1 else None)
        density = f['lambda_density'][i] if level == 1 else f['rho_R']
        ax = axes[0, col]
        style._silhouette(ax, x, density)
        ax.plot(x, np.where(mask, f[f'epsilon{level}_GI'][sel], np.nan), lw=2.6, label=fr'$\epsilon^{{({level})}}_{{GI}}$')
        style._references(ax, refs, f, level=level, fixed_index=i)
        relevant = f[f'epsilon{level}_GI'][sel][mask]
        if relevant.size:
            # GI focus excludes total/GD: stable snapshot-specific zoom, stated title.
            ref = refs['cbo'][i] if level == 1 else refs['pol_energy']
            vals = np.r_[relevant, ref[mask].ravel()]
            limits = style._range([vals])
            ax.set_ylim(limits)
        ax.set_title(fr'$\epsilon^{{({level})}}_{{GI}}$ and reference energies (zoom)')
        ax = axes[1, col]
        keys = (('epsilon1_cond', r'$\langle H_{BO}+V_{ph}+H_{LM}+H_{DSE}\rangle_\Phi$'),
                ('epsilon1_qgeo', r'$\epsilon^{(1)}_{geo,q_c}$'), ('epsilon1_Rgeo', r'$\epsilon^{(1)}_{geo,R}$'),
                ('epsilon1_projection', 'Fock-projection correction')) if level == 1 else (
                ('epsilon2_cond', r'$\langle H-T_R\rangle_\Gamma$'), ('epsilon2_geo', r'$\epsilon^{(2)}_{geo,R}$'))
        for k, (key, label) in enumerate(keys):
            ax.plot(x, np.where(mask, f[key][sel], np.nan), lw=2, color=COLORS[k+1], label=label)
        ax.set_title('Gauge-invariant components')
        for ax in axes[:, col]:
            ax.set_xlim(display['line_xlim_q' if level == 1 else 'line_xlim_R'])
            style._nice(ax, r'$q_c$ (a.u.)' if level == 1 else r'$R$ ($a_0$)', 'Energy (Ha)')
            style._legend(ax, 2)
    stamp(fig, f, 'GI decomposition / CBO and polaritonic reference comparison')
    return fig


def overview(f, p, data, u):
    fig, axes = plt.subplots(2, 2, figsize=(19, 12))
    fig.subplots_adjust(left=.08, right=.96, bottom=.12, top=.87, hspace=.58, wspace=.28)
    ax = axes[0, 0]
    ax.fill_between(p.R, f['rho_R'], color='.65', alpha=.65)
    ax.plot(p.R, f['rho_R'], color=COLORS[1], lw=2)
    ax.set(xlim=(p.c.rmin, p.c.rmax), ylim=(0, 1.08/(np.sqrt(np.pi)*p.c.sigma_r)), title='Nuclear probability density')
    style._nice(ax, r'$R$ ($a_0$)', r'$|\chi|^2$')
    ax = axes[0, 1]
    ph = f['rho_qR'].sum(axis=0)*p.dr
    ax.fill_between(f['q'], ph, color='.65', alpha=.65)
    ax.plot(f['q'], ph, color=COLORS[2], lw=2)
    ax.set(xlim=(-10, 10), ylim=(0, .6), title='Photon quadrature probability density')
    style._nice(ax, r'$q_c$ (a.u.)', 'Density')
    for ax, key, title, prefix in ((axes[1, 0], 'bo_population', 'Bare BO populations', 'S'),
                                    (axes[1, 1], 'pol_population', 'Polaritonic populations', 'Pol')):
        values = np.asarray(data[key])
        for k in range(values.shape[1]):
            ax.plot(data['t_fs'], values[:, k], lw=2, label=fr'$|{prefix}_{k}\rangle$')
        ax.axvline(f['t_fs'], ls=':', color='.4')
        ax.set(xlim=(0, max(data['t_fs'][-1], .01)), ylim=(-.02, 1.02), title=title)
        style._nice(ax, 't (fs)', 'Probability')
        style._legend(ax)
    stamp(fig, f, 'Dynamics overview')
    return fig


def outer(f, p, data, u, display):
    fig, axes = plt.subplots(2, 2, figsize=(21, 13))
    fig.subplots_adjust(left=.08, right=.97, top=.87, bottom=.12, hspace=.65, wspace=.28)
    ax = axes[0, 0]
    support = style._line_support(f, 2)
    style._silhouette(ax, p.R, f['rho_R'])
    style._scalar_lines(ax, p.R, f, 2)
    style._references(ax, {'R': p.R, 'pol_energy': p.pol_energy}, f, level=2)
    ax.set_title(r'Outer nuclear effective field: $\epsilon^{(2)}$')
    ax.set_xlim(display['line_xlim_R'])
    style._nice(ax, r'$R$ ($a_0$)', 'Energy (Ha)', display['scales']['energy2'])
    style._legend(ax, 4)
    ax = axes[0, 1]
    ax.plot(p.R, np.where(support, f['force'], np.nan), lw=2, label=r'$-\partial_R\epsilon^{(2)}+\partial_t\alpha$')
    ax.axhline(0, color='.4', ls='--')
    ax.set_title('Gauge-invariant outer EF force')
    ax.set_xlim(display['line_xlim_R'])
    style._nice(ax, r'$R$ ($a_0$)', 'Force (a.u.)', display['force_limits'])
    style._legend(ax)
    ax = axes[1, 0]
    coeff = np.einsum('rik,ri->rk', p.pol_vectors, u.reshape(p.c.nr, -1))
    for k in range(min(4, coeff.shape[1])):
        color = style.REF_COLORS[k]
        ax.plot(p.R, p.pol_energy[:, k], color=color, ls='--', lw=1.5, label=fr'$|Pol_{k}\rangle$')
        # SAME factor for every state/time, not individual normalization.
        ax.fill_between(p.R, p.pol_energy[:, k], p.pol_energy[:, k]+.04*abs(coeff[:, k])**2, color=color, alpha=.38)
    ax.set_title(r'State-resolved nuclear density on $|Pol_k\rangle$')
    ax.set_xlim(display['line_xlim_R'])
    style._nice(ax, r'$R$ ($a_0$)', 'Energy (Ha)', display['wavepacket_limits'])
    ax.text(.01, .98, 'Fill height = 0.04 × state density (same scale for all states)', transform=ax.transAxes, va='top', fontsize=10)
    style._legend(ax, 4)
    ax = axes[1, 1]
    for k, values in enumerate(np.asarray(data['pol_population']).T):
        ax.plot(data['t_fs'], values, lw=2, color=style.REF_COLORS[k % 4], label=fr'$|Pol_{k}\rangle$')
    ax.axvline(f['t_fs'], ls=':', color='.4')
    ax.set(xlim=(0, max(data['t_fs'][-1], .01)), ylim=(-.02, 1.02), title='Polaritonic populations')
    style._nice(ax, 't (fs)', 'Probability')
    style._legend(ax, 4)
    stamp(fig, f, 'Outer dynamics and polaritonic wavepacket')
    return fig


def state_composition(p):
    """Actual fixed-R eigenvector weights in the bare adiabatic x Fock basis."""
    c = p.c
    vectors = p.pol_vectors.reshape(c.nr, 2, c.nf, 2*c.nf)
    coeff = np.einsum('rej,renk->rjnk', p.bo_vectors, vectors)
    fig, axes = plt.subplots(2, 2, figsize=(19, 12), sharex=True, sharey=True, constrained_layout=True)
    for k, ax in enumerate(axes.flat):
        for j in range(2):
            for n in range(min(2, c.nf)):
                ax.plot(p.R, abs(coeff[:, j, n, k])**2, lw=2,
                        color=style.REF_COLORS[2*j+n], label=fr'$|S_{j},{n}\rangle$')
        if c.nf > 2:
            ax.plot(p.R, (abs(coeff[:, :, 2:, k])**2).sum(axis=(1, 2)),
                    label='Higher Fock components (see coefficients)', color='.5')
        ax.set(xlim=(0, 8), ylim=(-.02, 1.02), title=fr'Composition of $|Pol_{k}\rangle$')
        style._nice(ax, r'$R$ ($a_0$)', r'$|C_{S_j,n}^{(k)}|^2$')
        ax.legend(fontsize=12)
    fig.suptitle(f'PEN perspectives 2026 | g={c.g:g}, DSE {"on" if c.dse else "off"}\n'
                 +'Energy-ordered Pol labels; crossings/degeneracies can change individual eigenvectors')
    return fig


def render_case(case, movies=True, movie_stride=1, nq=160, dpi=85, t_max_fs=None):
    started = time.perf_counter()
    case = Path(case)
    c = Config(**json.loads((case/'config.json').read_text()))
    p = Propagator(c)
    paths = frame_paths(case, t_max_fs)
    print(f'RENDER {case}: {len(paths)} coefficient frames; reconstructing fields, no TDSE', flush=True)
    display, refs, checks, cbop = prepare(paths, p, nq)
    rows = json.loads((case/'observables.json').read_text())
    rows = [r for r in rows if r['t_fs'] <= float(load_npz(paths[-1])['t_fs'])+1e-9]
    data = {key: [r[key] for r in rows] for key in rows[0]}
    display['slice_observables'] = dict(t_fs=[v['t_fs'] for v in checks],
                                       photon_number=[r['photon_number'] for r in rows], cbo_population=cbop)
    out = case/('plots' if t_max_fs is None else f'plots_to_{t_max_fs:g}fs')
    out.mkdir(exist_ok=True)
    selected = sorted(set([0, len(paths)//2, len(paths)-1]))
    indices = sorted(set(range(0, len(paths), movie_stride)) | {len(paths)-1}) if movies else selected
    # Stable force and packet scales, using all saved times (not just movie stride).
    force_values = []
    for path in paths:
        z = load_npz(path)
        f = analyze(z['u'], p, float(z['t_fs']), nq)
        mask = f['rho_R'] > 1e-5*f['rho_R'].max()
        force_values.append(f['force'][mask])
    display['force_limits'] = style._range(force_values, symmetric=True)
    rmask = (p.R >= display['line_xlim_R'][0]) & (p.R <= display['line_xlim_R'][1])
    display['wavepacket_limits'] = (float(p.pol_energy[rmask].min()-.02), float(p.pol_energy[rmask].max()+.13))
    factories = dict(A_diagnostics_clean=lambda f, i, u: style.bridge_figure(f, display, i, refs),
                     B_diagnostics_clean=lambda f, i, u: sources(f, display, i, refs),
                     epsilon1_slice_analysis=lambda f, i, u: style.slice_figure(f, display, i, refs),
                     dynamics_overview=lambda f, i, u: overview(f, p, data, u),
                     outer_nuclear_effective=lambda f, i, u: outer(f, p, data, u, display))
    with plt.rc_context(style.STYLE), ExitStack() as stack:
        fig = state_composition(p)
        fig.savefig(out/'polaritonic_composition.png', dpi=max(140, dpi))
        plt.close(fig)
        writers, dimensions = {}, {}
        for index in sorted(set(indices)|set(selected)):
            z = load_npz(paths[index])
            f = analyze(z['u'], p, float(z['t_fs']), nq)
            for name, factory in factories.items():
                fig = factory(f, index, z['u'])
                stamp(fig, f)
                folder = out/name
                folder.mkdir(exist_ok=True)
                if index in selected:
                    fig.savefig(folder/f'frame_{index:05d}.png', dpi=max(140, dpi))
                if movies and index in indices:
                    if name not in writers:
                        writer = FFMpegWriter(fps=20, codec='libx264', extra_args=[
                            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                            '-pix_fmt', 'yuv420p', '-crf', '22', '-threads', '1'])
                        stack.enter_context(writer.saving(fig, folder/f'{name}.mp4', dpi=dpi))
                        writers[name] = writer
                        dimensions[name] = list(fig.get_size_inches())
                    # Matplotlib writer retains first fig; swap same-size figure explicitly.
                    writers[name].fig = fig
                    writers[name].grab_frame()
                plt.close(fig)
            if index in selected:
                for name, fig in (('bridge_consistency', consistency(f, display, index, refs)),
                                  ('GI_components', gi_components(f, display, refs))):
                    folder = out/name
                    folder.mkdir(exist_ok=True)
                    fig.savefig(folder/f'frame_{index:05d}.png', dpi=max(140, dpi))
                    plt.close(fig)
    atomic_json(out/'consistency.json', checks)
    report = dict(wall_seconds=time.perf_counter()-started, cache_frames=len(paths),
                  movie_frames=len(indices) if movies else 0, movie_stride=movie_stride,
                  nq_display=nq, r_points=c.nr, t_fs=[checks[0]['t_fs'], checks[-1]['t_fs']],
                  CBO_population=cbop, scales=display['scales'],
                  note='Continuous q reconstruction of finite-Fock wave; S_projection included. Not the grid-photon TDSE.',
                  peak_process_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  dimensions_inches=dimensions)
    atomic_json(out/'render_report.json', report)
    return report


def comparison(root, t_max_fs=None):
    root = Path(root)
    datasets = []
    for path in sorted(root.glob('dse_*/*/config.json')):
        c = Config(**json.loads(path.read_text()))
        rows = json.loads(path.with_name('observables.json').read_text())
        rows = [r for r in rows if t_max_fs is None or r['t_fs'] <= t_max_fs+1e-9]
        if rows:
            datasets.append((path.parent, c, rows))
    if not datasets:
        return
    out = root/('comparison' if t_max_fs is None else f'comparison_to_{t_max_fs:g}fs')
    out.mkdir(exist_ok=True)
    with plt.rc_context(style.STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(23, 12), constrained_layout=True)
        quantities = [('photon_number', 'Bare photon number'), ('bo_population', 'Bare excited BO population'),
                      ('product_R_ge_4', r'Probability at $R\geq4$'), ('signed_flux_R4', r'Signed flux at $R=4$'),
                      ('energy', 'Energy drift (Ha)'), ('top_fock_population', 'Highest retained Fock population')]
        for folder, c, rows in datasets:
            color = dict(zero=COLORS[0], weak=COLORS[1], strong=COLORS[2])[folder.name]
            t = [r['t_fs'] for r in rows]
            label = f'{folder.name}: g={c.g:g}, DSE {"on" if c.dse else "off"}'
            for ax, (key, title) in zip(axes.flat, quantities):
                y = np.array([r[key] for r in rows])
                if key == 'bo_population':
                    y = y[:, 1]
                if key == 'energy':
                    y = y-y[0]
                ax.plot(t, y, color=color, ls='--' if c.dse else '-', lw=1.8, label=label)
                ax.set_title(title)
                style._nice(ax, 't (fs)', 'Flux (a.u.)' if key == 'signed_flux_R4' else 'Ha' if key == 'energy' else 'Dimensionless')
                ax.axhline(0, ls=':', color='.5', lw=.7)
        axes[0, 0].legend(fontsize=10)
        fig.suptitle('PEN perspectives 2026 | finite-Fock DSE/coupling comparison')
        fig.savefig(out/'six_case_dynamics.png', dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharex=True, sharey=True, constrained_layout=True)
        for folder, c, rows in datasets:
            row, col = int(c.dse), ['zero', 'weak', 'strong'].index(folder.name)
            ax = axes[row, col]
            p = Propagator(c)
            for k in range(min(4, 2*c.nf)):
                ax.plot(p.R, p.pol_energy[:, k], color=style.REF_COLORS[k], lw=2, label=fr'$|Pol_{k}\rangle$')
            for e in range(2):
                for n in range(2):
                    ax.plot(p.R, p.bo_energy[:, e]+c.omega*(n+.5), color='.55', ls=':', lw=.8)
            ax.set(title=f'{folder.name}: g={c.g:g}, DSE {"on" if c.dse else "off"}', xlim=(0, 8), ylim=(-.02, .65))
            style._nice(ax, r'$R$ ($a_0$)', 'Energy (Ha)')
            if row == 0 and col == 0:
                ax.legend()
        fig.suptitle('Four polaritonic surfaces; gray dots: uncoupled bare product energies')
        fig.savefig(out/'polaritonic_surfaces.png', dpi=160)
        plt.close(fig)
