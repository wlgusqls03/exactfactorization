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
GI1_TERMS = [('epsilon1_wCBO', r'$\epsilon^{(1)}_{\mathrm{wCBO}}$'),
             ('epsilon1_qgeo', r'$\epsilon^{(1)}_{q_c,\mathrm{geo}}$'),
             ('epsilon1_Rgeo', r'$\epsilon^{(1)}_{R,\mathrm{geo}}$'),
             ('epsilon1_projection', r'$\epsilon^{(1)}_{\mathrm{proj}}$')]
GI2_TERMS = [('epsilon2_wBO', r'$\epsilon^{(2)}_{\mathrm{wBO}}$'),
             ('epsilon2_pkin', r'$\epsilon^{(2)}_{p,\mathrm{kin}}$'),
             ('epsilon2_ppot', r'$\epsilon^{(2)}_{p,\mathrm{pot}}$'),
             ('epsilon2_LM', r'$\epsilon^{(2)}_{\mathrm{LM}}$'),
             ('epsilon2_DSE', r'$\epsilon^{(2)}_{\mathrm{DSE}}$'),
             ('epsilon2_geo', r'$\epsilon^{(2)}_{R,\mathrm{geo}}$')]
TERM_COLORS = ['#0072B2', '#D55E00', '#009E73', '#8E44AD', '#A65628', '#CC79A7']


def weighted_focus(values, weights, symmetric=False):
    """Time-global density-weighted 0.5--99.5% focus; no data modification."""
    if not values:
        return (-1e-6, 1e-6)
    v, w = np.concatenate(values), np.concatenate(weights)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[valid], w[valid]
    if not len(v):
        return (-1e-6, 1e-6)
    order = np.argsort(v)
    cumulative = np.cumsum(w[order])/np.sum(w)
    lo, hi = np.interp([.005, .995], cumulative, v[order])
    return style._range([np.array([lo, hi])], symmetric=symmetric)


def clipped_markers(ax, x, y, limits, color):
    """Explicitly mark real features outside a focus window instead of hiding them."""
    span = limits[1]-limits[0]
    for mask, marker, edge in ((y > limits[1], '^', limits[1]-.018*span),
                               (y < limits[0], 'v', limits[0]+.018*span)):
        indices = np.flatnonzero(mask)
        indices = indices[::max(1, len(indices)//12)]
        ax.plot(x[indices], np.full(len(indices), edge), linestyle='none', marker=marker,
                color=color, markersize=4, clip_on=True)


def frame_paths(case, t_max_fs=None):
    paths = sorted((Path(case)/'coefficients').glob('wave_*.npz'))
    if t_max_fs is not None and (not np.isfinite(t_max_fs) or t_max_fs < 0):
        raise ValueError('t-max-fs must be finite and nonnegative')
    paths = [p for p in paths if t_max_fs is None or float(load_npz(p)['t_fs']) <= t_max_fs+1e-9]
    if not paths:
        raise ValueError(f'No coefficient frames in {case} for requested time range')
    return paths


def png_indices(times_fs, every_fs=None):
    """Nearest saved frames to 0, interval, 2*interval, ...; include endpoints.

    No interpolation/repropagation. Ties choose the earlier saved frame.
    Without an interval preserve the legacy first/middle/last selection.
    """
    times = np.asarray(times_fs, dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('PNG selection needs finite, strictly increasing frame times')
    if every_fs is None:
        return sorted({0, len(times)//2, len(times)-1})
    if not np.isfinite(every_fs) or every_fs <= 0:
        raise ValueError('png-every-fs must be finite and positive')
    if len(times) == 1 or every_fs <= np.min(np.diff(times)):
        return list(range(len(times)))
    # Work in frame space, so a tiny requested interval cannot allocate a huge
    # target array. A frame is selected iff its nearest-neighbour cell contains
    # an integer multiple of every_fs. Midpoint ties belong to the earlier frame.
    midpoints = (times[:-1]+times[1:])/2
    lower = np.floor(midpoints[:-1]/every_fs)+1
    upper = np.floor(midpoints[1:]/every_fs)
    selected = np.flatnonzero(lower <= upper)+1
    return sorted({0, len(times)-1, *selected.tolist()})


def prepare(paths, p, nq):
    """Two-pass rendering: retain only scales, cameras and small time traces."""
    samples = {k: [] for k in ('epsilon1', 'a', 'b', 'energy1', 'energy2', 'momentum', 'source',
                               'alpha_direct', 'e2_direct', 'alpha_residual', 'e2_residual', 'bridge_terms',
                               'gi1', 'gi2', 'budget1', 'budget2', 'small1', 'small2', 'force')}
    windows, checks, cbop, peak_indices = [], [], [], []
    focus_weights = {k: [] for k in ('budget1', 'budget2', 'small1', 'small2')}
    packet_peak = 0.
    moments = []
    previous = None
    first = None
    for path in paths:
        z = load_npz(path)
        f = analyze(z['u'], p, float(z['t_fs']), nq)
        if first is None:
            first = f
        support = f['rho_qR'] > 1e-5*f['rho_qR'].max()
        outer = f['rho_R'] > 1e-5*f['rho_R'].max()
        peak_indices.append(int(np.argmax(f['rho_R'])))
        density = f['rho_R']/f['rho_R'].sum()
        mean = float(np.sum(density*p.R))
        moments.append((mean, float(np.sqrt(np.sum(density*(p.R-mean)**2)))))
        coeff = np.einsum('rik,ri->rk', p.pol_vectors, z['u'].reshape(p.c.nr, -1))
        packet_peak = max(packet_peak, float(np.max(abs(coeff)**2)))
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
        collect('force', f['force'][outer])
        i = int(np.argmin(abs(p.R-p.c.r2)))
        j = int(np.argmin(abs(f['q'])))
        for level, selector in ((1, i), (1, (slice(None), j)), (2, slice(None))):
            mask = style._line_support(f, level, selector if level == 1 else None)
            density = (f['rho_qR'] if level == 1 else f['rho_R'])[selector][mask]
            # Normalize each slice before accumulating, so the q/R cuts and
            # different frames have equal weight in the global camera choice.
            weights = density/max(float(density.sum()), 1e-300)
            collect(f'gi{level}', f[f'epsilon{level}_GI'][selector][mask])
            collect(f'budget{level}', f[f'epsilon{level}_GI'][selector][mask])
            focus_weights[f'budget{level}'].append(weights[::max(1, len(weights)//1200)])
            for key, _ in (GI1_TERMS if level == 1 else GI2_TERMS):
                collect(f'budget{level}', f[key][selector][mask])
                focus_weights[f'budget{level}'].append(weights[::max(1, len(weights)//1200)])
                if key not in ('epsilon1_wCBO', 'epsilon2_wBO', 'epsilon2_pkin', 'epsilon2_ppot'):
                    collect(f'small{level}', f[key][selector][mask])
                    focus_weights[f'small{level}'].append(weights[::max(1, len(weights)//1200)])
        previous = style._window(f, previous)
        windows.append(previous)
        checks.append(dict(t_fs=float(z['t_fs']), **diagnostics(f)))
        cbop.append(f['cbo_population'].tolist())
    scales = {key: style._range(values, symmetric=key in ('a', 'b', 'momentum', 'source', 'force',
                                                       'alpha_direct', 'alpha_residual', 'e2_residual', 'small1', 'small2'))
              for key, values in samples.items()}
    for key, weights in focus_weights.items():
        scales[key] = weighted_focus(samples[key], weights, symmetric=key.startswith('small'))
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
                   slices_q=[int(np.argmin(abs(q-v))) for v in (0., 1.5)],
                   peak_indices=peak_indices, packet_peak=max(packet_peak, 1e-30),
                   force_limits=scales['force'], nuclear_moments=moments)
    # Static references are drawn across the full x extent. Their values on
    # the occupied time-union inform fixed GI-only limits (never total/GD).
    i, j = display['i'], display['j']
    scales['gi1'] = style._range([scales['gi1'], refs['cbo'][i, qmask], refs['cbo'][rmask, j]])
    scales['gi2'] = style._range([scales['gi2'], refs['pol_energy'][rmask]])
    return display, refs, checks, cbop


def stamp(fig, f, title=None):
    if title is None:
        title = fig._suptitle.get_text().splitlines()[0].split('    ')[0]
        while title.startswith('PEN perspectives 2026 | '):
            title = title[len('PEN perspectives 2026 | '):]
    frequency = fr'$\omega_c={f["omega"]:.6g}$ a.u.; ' if 'omega' in f else ''
    scenario = f.get('scenario_label', '')
    fig.suptitle(f'PEN perspectives 2026 | {title}\n'
                 +(scenario+'; ' if scenario else '')
                 +fr'$t={f["t_fs"]:.2f}$ fs; $g={f["g_cavity"]:g}$; DSE {"on" if f["dse"] else "off"}; '
                 +frequency+('positive marginal gauge' if scenario else '2 electronic states × finite Fock basis; positive marginal gauge'), fontsize=21, y=.992)


def source_slice(ax, f, display, i, title):
    """Eulerian partial-time sources sampled at a slice, not a moving derivative."""
    mask = style._line_support(f, 1, i)
    style._silhouette(ax, f['q'], f['lambda_density'][i])
    for key, label, color in zip(('density_dt', 'S_q', 'S_adv', 'S_rel', 'S_projection'),
            (r'$\partial_t|\Lambda_R|^2$', r'$S_q$', r'$S_{\mathrm{adv}}$',
             r'$S_{\mathrm{rel}}$', r'$S_{\mathrm{projection}}$'), COLORS):
        ax.plot(f['q'], np.where(mask, f[key][i], np.nan), color=color, lw=2, label=label)
    ax.axhline(0, color='.4', ls='--', lw=1)
    ax.set_xlim(display['line_xlim_q'])
    ax.set_title(title+fr': $R={f["R"][i]:.3f}\,a_0$')
    style._nice(ax, r'$q_c$ (a.u.)', 'Conditional density / a.u. time', display['scales']['source'])
    style._unsupported_note(ax, mask)
    style._legend(ax, 3)


def sources(f, display, index, refs):
    """Four maps, fixed-R pair, then a pair at argmax_R |chi|^2."""
    fig, axes = plt.subplots(4, 2, figsize=(23, 24))
    fig.subplots_adjust(left=.07, right=.96, bottom=.10, top=.92, hspace=.72, wspace=.30)
    i, peak = display['i'], int(np.argmax(f['rho_R']))
    formulas = (r'(A) $\partial_t|\Lambda_R|^2$',
                r'(B) $S_q=-\partial_{q_c}(a|\Lambda_R|^2)$',
                r'(C) $S_{\mathrm{adv}}=-\frac{\alpha}{M}\partial_R|\Lambda_R|^2$',
                r'(D) $S_{\mathrm{rel}}=-\frac{1}{M|\chi|^2}\partial_R[|\chi|^2|\Lambda_R|^2(b-\alpha)]$')
    for ax, key, title in zip(axes[:2].flat, ('density_dt', 'S_q', 'S_adv', 'S_rel'), formulas):
        style._map(ax, f, key, display['scales']['source'], display['windows'][index], f['R'][i], title)
        ax.set_title(title, fontsize=16 if key == 'S_rel' else 18)
        ax.axvline(f['R'][peak], color='#AA3377', ls=':', lw=2)
    source_slice(axes[2, 0], f, display, i, '(E) Fixed-R source slice')
    source_slice(axes[3, 0], f, display, peak, '(G) Density-peak source slice')
    for row, cut, label in ((2, i, '(F) Fixed R'), (3, peak, '(H) Density peak')):
        style._momentum(axes[row, 1], f, cut, display['scales']['momentum'], display['line_xlim_q'])
        axes[row, 1].set_title(label+fr': momentum at $R={f["R"][cut]:.3f}\,a_0$')
    fig.text(.5, .016, r'Black dashed: fixed R; magenta dotted: $R_{\rm peak}=\arg\max_R|\chi|^2$. '
             'G/H sample the same instantaneous fields; no moving-frame derivative.', ha='center', fontsize=14)
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
    """Six panels: two first-level directions and outer spectral/energy budgets.

    wCBO/wPol are independently reconstructed from local spectral populations.
    Photon kinetic energy is already inside wPol: the expanded lower-right
    budget REPLACES wPol, never adds a second photon kinetic term to it.
    """
    fig, axes = plt.subplots(2, 3, figsize=(30, 17))
    fig.subplots_adjust(left=.055, right=.98, top=.88, bottom=.20, hspace=.72, wspace=.30)
    i, j = display['i'], display['j']
    specs = ((1, 'q', i, f['lambda_density'][i], fr'$q_c$-slice: $R={f["R"][i]:.3f}\,a_0$'),
             (1, 'R', (slice(None), j), f['rho_qR'][:, j], fr'$R$-slice: $q_c={f["q"][j]:.2f}$'),
             (2, 'R', slice(None), f['rho_R'], r'Outer nuclear $\epsilon^{(2)}_{\mathrm{GI}}$'))
    for col, (level, direction, sel, density, title) in enumerate(specs):
        x = f[direction]
        mask = style._line_support(f, level, sel if level == 1 else None)
        ax = axes[0, col]
        style._silhouette(ax, x, density)
        total_label = fr'$\epsilon^{{({level})}}_{{\mathrm{{GI}}}}$'
        ax.plot(x, np.where(mask, f[f'epsilon{level}_GI'][sel], np.nan), color=COLORS[0], lw=2.8, label=total_label)
        clipped_markers(ax, x, np.where(mask, f[f'epsilon{level}_GI'][sel], np.nan),
                        display['scales'][f'gi{level}'], COLORS[0])
        weighted_key, weighted_label = ('epsilon1_wCBO', r'$\epsilon^{(1)}_{\mathrm{wCBO}}$') if level == 1 else (
            'epsilon2_wPol', r'$\epsilon^{(2)}_{\mathrm{wPol}}$')
        ax.plot(x, np.where(mask, f[weighted_key][sel], np.nan), color=COLORS[1], ls='-.', lw=2, label=weighted_label)
        style._references(ax, refs, f, level=level, direction=direction, fixed_index=i if direction == 'q' else j)
        ax.set_title(f'({"ABC"[col]}) '+title)
        ax = axes[1, col]
        style._silhouette(ax, x, density)
        ax.plot(x, np.where(mask, f[f'epsilon{level}_GI'][sel], np.nan), color=COLORS[0], lw=2.8, label=total_label)
        clipped_markers(ax, x, np.where(mask, f[f'epsilon{level}_GI'][sel], np.nan),
                        display['scales'][f'budget{level}'], COLORS[0])
        keys = GI1_TERMS if level == 1 else [term for term in GI2_TERMS if f['dse'] or term[0] != 'epsilon2_DSE']
        inset = ax.inset_axes([.54, .51, .44, .43])
        for k, (key, label) in enumerate(keys):
            color = TERM_COLORS[(GI1_TERMS if level == 1 else GI2_TERMS).index((key, label))]
            values = np.where(mask, f[key][sel], np.nan)
            ax.plot(x, values, lw=2, color=color, label=label)
            clipped_markers(ax, x, values, display['scales'][f'budget{level}'], color)
            if key not in ('epsilon1_wCBO', 'epsilon2_wBO', 'epsilon2_pkin', 'epsilon2_ppot'):
                inset.plot(x, values, lw=1.4, color=color)
                clipped_markers(inset, x, values, display['scales'][f'small{level}'], color)
        inset.set_title('Small contributions (same colors)', fontsize=10)
        inset.set_xlim(display['line_xlim_'+direction])
        style._nice(inset, limits=display['scales'][f'small{level}'])
        inset.tick_params(labelsize=9)
        inset.yaxis.get_offset_text().set_fontsize(9)
        inset.axhline(0, ls=':', color='.5', lw=.7)
        ax.set_title(f'({"DEF"[col]}) '+('GI components: '+direction+'-slice' if level == 1 else 'Expanded weighted-Pol energy + geometry'))
        for row, ax in enumerate(axes[:, col]):
            ax.set_xlim(display['line_xlim_'+direction])
            style._nice(ax, r'$q_c$ (a.u.)' if direction == 'q' else r'$R$ ($a_0$)', 'Energy (Ha)',
                        display['scales'][f'{"gi" if row == 0 else "budget"}{level}'])
            style._unsupported_note(ax, mask)
            style._legend(ax, 3)
    fig.text(.5, .018, r'$\epsilon^{(2)}_{\mathrm{GI}}=\epsilon^{(2)}_{\mathrm{wPol}}+\epsilon^{(2)}_{R,\mathrm{geo}}$; '
             r'$\epsilon^{(2)}_{\mathrm{wPol}}=\epsilon^{(2)}_{\mathrm{wBO}}+\epsilon^{(2)}_{p,\mathrm{kin}}+'
             r'\epsilon^{(2)}_{p,\mathrm{pot}}+\epsilon^{(2)}_{\mathrm{LM}}+\epsilon^{(2)}_{\mathrm{DSE}}$'
             '\nPhoton kinetic energy is included once. No separate external field in this model.'
             '\nBudget/inset focus: density-weighted 0.5–99.5%; triangles mark off-scale values. Limits fixed in time.',
             ha='center', fontsize=14)
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


def packet_height(density, peak, mode='sqrt'):
    """One monotone display transform for EVERY branch and frame.

    Sqrt is a visibility transform only: filled area is not population.
    Actual probabilities are retained unchanged in the neighbouring panel.
    """
    normalized = np.maximum(density, 0)/max(peak, 1e-30)
    if mode == 'sqrt':
        normalized = np.sqrt(normalized)
    elif mode != 'linear':
        raise ValueError('packet-scale must be sqrt or linear')
    return .10*normalized


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
        height = packet_height(abs(coeff[:, k])**2, display['packet_peak'], display['packet_scale'])
        ax.fill_between(p.R, p.pol_energy[:, k], p.pol_energy[:, k]+height, color=color, alpha=.48)
        ax.plot(p.R, p.pol_energy[:, k]+height, color=color, lw=1.7)
    ax.set_title(r'State-resolved nuclear density on $|Pol_k\rangle$')
    ax.set_xlim(display['line_xlim_R'])
    style._nice(ax, r'$R$ ($a_0$)', 'Energy (Ha)', display['wavepacket_limits'])
    transform = 'sqrt(density / shared peak)' if display['packet_scale'] == 'sqrt' else 'density / shared peak'
    ax.text(.01, .98, 'Fill height = 0.10 Ha × '+transform+'\nShared scale across states/time; area is not population.',
            transform=ax.transAxes, va='top', fontsize=11, bbox=dict(facecolor='white', alpha=.82, edgecolor='none'))
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
    from .resonance import scenario_report
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
    fig.suptitle(scenario_report(c)['label']+f' | omega={c.omega:g} au, g={c.g:g}, DSE {"on" if c.dse else "off"}\n'
                 +'Energy-ordered Pol labels; crossings/degeneracies can change individual eigenvectors')
    return fig


def render_case(case, movies=True, movie_stride=1, nq=160, dpi=85, t_max_fs=None, packet_scale='sqrt', png_every_fs=None):
    from .resonance import scenario_report
    from .paper_plots import exchange_summary
    started = time.perf_counter()
    case = Path(case)
    c = Config(**json.loads((case/'config.json').read_text()))
    scenario = scenario_report(c)
    p = Propagator(c)
    paths = frame_paths(case, t_max_fs)
    print(f'RENDER {case}: {len(paths)} coefficient frames; reconstructing fields, no TDSE', flush=True)
    display, refs, checks, cbop = prepare(paths, p, nq)
    if packet_scale not in ('linear', 'sqrt'):
        raise ValueError('packet-scale must be sqrt or linear')
    display['packet_scale'] = packet_scale
    rows = json.loads((case/'observables.json').read_text())
    rows = [r for r in rows if r['t_fs'] <= float(load_npz(paths[-1])['t_fs'])+1e-9]
    data = {key: [r[key] for r in rows] for key in rows[0]}
    display['slice_observables'] = dict(t_fs=[v['t_fs'] for v in checks],
                                       photon_number=[r['photon_number'] for r in rows], cbo_population=cbop)
    out = case/('plots' if t_max_fs is None else f'plots_to_{t_max_fs:g}fs')
    out.mkdir(exist_ok=True)
    exchange_summary(out, c, data, checks, cbop, display['nuclear_moments'])
    selected = png_indices([v['t_fs'] for v in checks], png_every_fs)
    print(f'PNG: {len(selected)} saved frames; interval={png_every_fs} fs '
          '(None = first/middle/last); endpoints included, no interpolation', flush=True)
    indices = sorted(set(range(0, len(paths), movie_stride)) | {len(paths)-1}) if movies else selected
    # Force, GI and packet scales already collected in the first streaming pass.
    rmask = (p.R >= display['line_xlim_R'][0]) & (p.R <= display['line_xlim_R'][1])
    display['wavepacket_limits'] = (float(p.pol_energy[rmask].min()-.02), float(p.pol_energy[rmask].max()+.13))
    factories = dict(A_diagnostics_clean=lambda f, i, u: style.bridge_figure(f, display, i, refs),
                     B_diagnostics_clean=lambda f, i, u: sources(f, display, i, refs),
                     epsilon1_slice_analysis=lambda f, i, u: style.slice_figure(f, display, i, refs),
                     GI_components=lambda f, i, u: gi_components(f, display, refs),
                     dynamics_overview=lambda f, i, u: overview(f, p, data, u),
                     outer_nuclear_effective=lambda f, i, u: outer(f, p, data, u, display))
    quadrature_checks = []
    with plt.rc_context(style.STYLE), ExitStack() as stack:
        fig = state_composition(p)
        fig.savefig(out/'polaritonic_composition.png', dpi=max(140, dpi))
        plt.close(fig)
        writers, dimensions = {}, {}
        for index in sorted(set(indices)|set(selected)):
            z = load_npz(paths[index])
            f = analyze(z['u'], p, float(z['t_fs']), nq)
            f['scenario_label'] = scenario['label']
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
                # Independent enlarged quadrature, NOT an altered TDSE box.
                # Off-resonance HO functions have larger tails at |q|=10.
                wide = analyze(z['u'], p, float(z['t_fs']), max(320, 2*nq), qmax=16.)
                quadrature_checks.append(dict(t_fs=float(z['t_fs']),
                    qmax10=diagnostics(f), qmax16=diagnostics(wide)))
                del wide
                for name, fig in (('bridge_consistency', consistency(f, display, index, refs)),):
                    folder = out/name
                    folder.mkdir(exist_ok=True)
                    fig.savefig(folder/f'frame_{index:05d}.png', dpi=max(140, dpi))
                    plt.close(fig)
    atomic_json(out/'consistency.json', checks)
    atomic_json(out/'quadrature_checks.json', quadrature_checks)
    png_selection = dict(png_every_fs=png_every_fs,
        policy='Nearest saved frame to integer multiples of interval; endpoints included; no interpolation. None: first/middle/last.',
        frames=[dict(index=i, t_fs=float(checks[i]['t_fs']), source=str(paths[i]),
                     filename=f'frame_{i:05d}.png') for i in selected],
        existing_outputs='Matching PNGs overwritten; other existing PNGs and, with movies=False, all MP4s retained.')
    atomic_json(out/'png_selection.json', png_selection)
    report = dict(wall_seconds=time.perf_counter()-started, cache_frames=len(paths),
                  scenario=scenario,
                  png_every_fs=png_every_fs, png_frame_indices=selected,
                  png_times_fs=[float(checks[i]['t_fs']) for i in selected],
                  movie_frames=len(indices) if movies else 0, movie_stride=movie_stride,
                  nq_display=nq, r_points=c.nr, t_fs=[checks[0]['t_fs'], checks[-1]['t_fs']],
                  CBO_population=cbop, scales=display['scales'],
                  tracked_R_peak=[float(p.R[j]) for j in display['peak_indices']],
                  frame_times_fs=[v['t_fs'] for v in checks],
                  tracking_definition='argmax of nuclear marginal |chi|^2; G/H are instantaneous fixed-coordinate sources sampled there',
                  packet_scale=packet_scale, packet_density_shared_peak=display['packet_peak'],
                  gi_focus='budget/inset: density-weighted 0.5--99.5 percentiles; off-scale triangle markers; time-global scales',
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
