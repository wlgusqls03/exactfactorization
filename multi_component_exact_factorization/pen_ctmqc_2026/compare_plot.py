"""Three-way analysis figures: same observable first, interpretation second.

Inspired by PCCP 2026 Figs5/8/9/10 (N and S1) and Figs6/7 (branching).
No copied graphics. Conditional mean static energy is NEVER called a TDPES.
All signed forces are distinguished from the quantum-amplitude contribution.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ..pen_perspectives_2026.model import local_hamiltonian
from .compare_data import LABELS
from .render import STYLE, silhouette, limits

COLOR = {'qd': '#151515', 'el': '#0072B2', 'pol': '#D55E00'}
BRANCH = ['#0072B2', '#D55E00', '#009E73', '#8E44AD']


def label(a):
    return LABELS[a.method] if a.method == 'qd' else a.settings.algorithm.upper()+' '+a.method.upper()


def layout(title, rows=2, cols=2, subtitle=''):
    fig, axes = plt.subplots(rows, cols, figsize=(9*cols, 6*rows), squeeze=False)
    bottom = .27 if rows == 1 else (.19 if rows == 2 else .15)
    fig.subplots_adjust(left=.08, right=.97, bottom=bottom, top=.87, wspace=.28, hspace=.75)
    fig.suptitle(title, y=.985, fontsize=23)
    fig.text(.5, .935, subtitle, ha='center', fontsize=12)
    for i, ax in enumerate(axes.flat):
        ax.text(-.11, 1.04, chr(65+i), transform=ax.transAxes, fontsize=20, weight='bold')
    return fig, axes


def finish(ax, title, x, y, xlim=None, ylim=None, zero=False):
    ax.set(title=title, xlabel=x, ylabel=y)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=x.startswith(('$R$', '$q_c$'))))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(alpha=.15)
    if zero:
        ax.axhline(0, color='.45', ls=':', lw=1, zorder=0)
    handles, names = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(names, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper center', bbox_to_anchor=(.5, -.22),
                  ncol=min(3, len(unique)), fontsize=11, frameon=False)


def note(ax, message):
    ax.text(.5, .5, message, transform=ax.transAxes, ha='center', va='center', fontsize=13, wrap=True)


def footer(fig, text):
    fig.text(.08, .02, text, fontsize=11, color='.3')


def time_line(ax, a, value, interval, name=None):
    """Clip connected scalar samples to overlap; never extrapolate endpoints.

    Include interpolated boundary points so a coarser CT trace does not vanish
    when the QD comparison window ends before the next CT saved sample.
    Wavefunctions/trajectories are never interpolated by this plotting helper.
    """
    t = a.values('t_fs')
    display = np.unique(np.r_[interval[0], t[(t > interval[0]) & (t < interval[1])], interval[1]])
    y = np.interp(display, t, value)
    ax.plot(display, y, color=COLOR[a.method], lw=2.4, label=name or label(a),
            marker='.' if len(display) == 1 else None)


def trace(ax, archives, key, interval, column=None, transform=None):
    for method, a in archives.items():
        value = a.values(key)
        if value is None:
            continue
        if column is not None:
            if value.ndim < 2 or column >= value.shape[1]:
                continue
            value = value[:, column]
        if transform is not None:
            value = transform(value, a)
        time_line(ax, a, value, interval)


def observations(archives, interval, title, warning):
    fig, axs = layout(title+' | Common observables', 3, subtitle='QD vs CTMQC EL vs CTMQC POL; native saved observables')
    items = [('photon_number', None, r'$\langle N\rangle$', 'Photon number'),
             ('bo_population', 1, r'$P_{S_1}$', 'Bare excited-state population'),
             ('kinetic_R', None, 'Energy (Ha)', 'Nuclear kinetic energy'),
             ('product_R_ge_4', None, 'Probability', r'Region probability: $R\geq4\,a_0$'),
             ('signed_flux_R4', None, r'Probability / $t_{au}$', r'Signed flux at $R=4\,a_0$')]
    for ax, (key, col, ylabel, heading) in zip(axs.flat, items):
        trace(ax, archives, key, interval, col)
        finish(ax, heading, '$t$ (fs)', ylabel, interval, zero=key == 'signed_flux_R4')
    for method, a in archives.items():
        n, p = a.values('photon_number'), a.values('bo_population')
        if n is None or p is None:
            continue
        time_line(axs[2, 1], a, n-n[0]+p[:, 1]-p[0, 1], interval)
    finish(axs[2, 1], 'Excitation-balance diagnostic (not a conservation law)', '$t$ (fs)',
           r'$\Delta\langle N\rangle+\Delta P_{S_1}$', interval, zero=True)
    footer(fig, warning+'\nR>=4 is a region diagnostic, not a reaction yield for Rabi; CT flux is a KDE estimate.')
    return fig


def photon_budget(archives, interval, title, warning):
    fig, axs = layout(title+' | Photon number and quadratures', subtitle=r'$N=\omega_c\langle q_c^2\rangle/2+\langle p_c^2\rangle/(2\omega_c)-1/2$')
    trace(axs[0, 0], archives, 'photon_number', interval)
    trace(axs[0, 1], archives, 'q2', interval, transform=lambda v, a: .5*a.config.omega*v)
    trace(axs[1, 0], archives, 'p2', interval, transform=lambda v, a: v/(2*a.config.omega))
    for method, a in archives.items():
        n, q2, p2 = (a.values(k) for k in ('photon_number', 'q2', 'p2'))
        if any(v is None for v in (n, q2, p2)):
            continue
        residual = .5*a.config.omega*q2+p2/(2*a.config.omega)-.5-n
        time_line(axs[1, 1], a, residual, interval)
    for ax, name, ylabel in zip(axs.flat,
            ('Photon number', 'Position-quadrature contribution', 'Momentum-quadrature contribution', 'Number identity residual'),
            (r'$\langle N\rangle$', r'$\omega_c\langle q_c^2\rangle/2$', r'$\langle p_c^2\rangle/(2\omega_c)$', 'Residual')):
        finish(ax, name, '$t$ (fs)', ylabel, interval, zero=True)
    footer(fig, warning+'\nEL uses a Wigner estimator, not a Fock population; negative sampling estimates are not clipped.')
    return fig


def pol_populations(archives, interval, title, warning):
    fig, axs = layout(title+' | Polaritonic populations', subtitle='Only QD and POL have a quantum electron-photon coefficient vector')
    for k, ax in enumerate(axs.flat):
        available = {m: a for m, a in archives.items() if m != 'el'}
        trace(ax, available, 'pol_population', interval, k)
        if not ax.lines:
            note(ax, 'Polaritonic population unavailable')
        finish(ax, fr'$|\mathrm{{Pol}}_{k}\rangle$', '$t$ (fs)', 'Population', interval, (-.03, 1.03))
    footer(fig, warning+'\nEL Pol populations are not reconstructed. For NF>2, these four branches need not exhaust the population.')
    return fig


def accuracy(archives, interval, errors, title, warning):
    fig, axs = layout(title+' | Differences and numerical checks', subtitle='Differences are relative to our finite-Fock QD, not to digitized paper data')
    for method, fields in errors.items():
        for ax, key in zip(axs[0], ('photon_number', 'bo_population')):
            if key in fields:
                z = fields[key]
                ax.plot(z['t_fs'], z['error'], color=COLOR[method], lw=2.4, label=LABELS[method]+' - QD')
    for method, a in archives.items():
        e = a.values('energy')
        if e is not None:
            time_line(axs[1, 0], a, e-e[0], interval)
        edge = a.values('edge_probability')
        if edge is not None:
            time_line(axs[1, 1], a, edge, interval, label(a)+(' edge strip' if method == 'qd' else ' outside box'))
    for ax, name, ylabel in zip(axs.flat, ('Photon-number difference', 'S1 population difference', 'Energy drift', 'Boundary warning (different definitions)'),
                                (r'$\Delta\langle N\rangle$', r'$\Delta P_{S_1}$', r'$E(t)-E(0)$ (Ha)', 'Probability')):
        finish(ax, name, '$t$ (fs)', ylabel, interval, zero=True)
    footer(fig, warning+'\nScalar traces are linearly aligned without extrapolation; QD and CT boundary diagnostics are not the same quantity.')
    return fig


def snapshot_heading(frames, title, reader):
    times = '; '.join(f'{LABELS[k]} {v["t_fs"]:.3f} fs' for k, v in frames.items())
    return title, times+fr'; $\omega_c={reader.c.omega:g}$; $g={reader.c.g:g}$; DSE {"on" if reader.c.dse else "off"}'


def density(frames, reader, scales, title):
    title, sub = snapshot_heading(frames, title+' | State-resolved densities', reader)
    fig, axs = layout(title, subtitle=sub)
    for method, f in frames.items():
        for ax, x, y in ((axs[0, 0], f['R'], f['rho_R']), (axs[0, 1], f['q'], f['rho_q']),
                         (axs[1, 0], f['R'], f['bo_density'][:, 0]), (axs[1, 1], f['R'], f['bo_density'][:, 1])):
            ax.plot(x, y, color=COLOR[method], lw=2.4, label=LABELS[method])
    for ax, name, x, xr, yr in zip(axs.flat,
            ('Nuclear density', 'Photon-coordinate probability', 'Nuclear density on bare S0', 'Nuclear density on bare S1'),
            ('$R$ ($a_0$)', '$q_c$ (a.u.)', '$R$ ($a_0$)', '$R$ ($a_0$)'),
            (scales['R_range'], scales['q_range'], scales['R_range'], scales['R_range']),
            (scales['density_R'], scales['density_q'], scales['density_R'], scales['density_R'])):
        finish(ax, name, x, 'Probability density', xr, yr)
    footer(fig, fr'Matched Gaussian display widths: $h_R={reader.widths[0]:.3g}$, $h_q={reader.widths[1]:.3g}$; no mass renormalization.'
           +'\nState densities integrate to their population; each branch is NOT normalized separately.')
    return fig


def guidance(frames, reader, scales, title):
    title, sub = snapshot_heading(frames, title+' | Polaritonic branch guidance', reader)
    fig, axs = layout(title, rows=1, subtitle=sub)
    energies = np.linalg.eigvalsh(local_hamiltonian(reader.c, reader.R))
    for ax, method in zip(axs.flat, ('qd', 'pol')):
        f = frames.get(method)
        if f is None:
            note(ax, LABELS[method]+' unavailable')
        else:
            for k in range(min(4, energies.shape[1])):
                baseline = energies[:, k]
                ax.plot(reader.R, baseline, '--', lw=1.5, color=BRANCH[k])
                lift = scales['branch_lift']*f['pol_density'][:, k]
                ax.fill_between(reader.R, baseline, baseline+lift, color=BRANCH[k], alpha=.4)
                ax.plot(reader.R, baseline+lift, color=BRANCH[k], lw=1.8, label=fr'$|\mathrm{{Pol}}_{k}\rangle$')
        finish(ax, LABELS[method], '$R$ ($a_0$)', 'Reference energy (Ha) + scaled density', scales['R_range'], scales['branch_energy'])
    footer(fig, 'Density lifted onto static Pol surfaces: vertical packet height is NOT kinetic energy or a trajectory energy.'
           +'\nSame density multiplier in both panels/all frames; no quantum Pol populations are assigned to EL.')
    return fig


def potential(frames, reader, scales, title, slice_R=None):
    title, sub = snapshot_heading(frames, title+' | TDPES and static-energy context', reader)
    fig, axs = layout(title, subtitle=sub)
    reference = frames.get('qd', next(iter(frames.values())))
    i = int(np.argmax(reference['rho_R'])) if slice_R is None else int(np.argmin(abs(reader.R-slice_R)))
    R = float(reader.R[i])
    native = frames['qd']['native'] if 'qd' in frames else None
    if native is not None:
        mask = native['rho_R'] > native['rho_R'].max()*1e-4
        for key, name, color in (('epsilon2', r'$\epsilon^{(2)}$', 'black'), ('epsilon2_GI', r'$\epsilon^{(2)}_{GI}$', '#0072B2'), ('epsilon2_GD', r'$\epsilon^{(2)}_{GD}$', '#D55E00')):
            axs[0, 0].plot(native['R'], np.where(mask, native[key], np.nan), color=color, lw=2.3, label=name)
        silhouette(axs[0, 0], native['R'], native['rho_R'])
        j = int(np.argmin(abs(native['R']-R)))
        jd = native['rho_qR'][j]
        mask = (jd > jd.max()*1e-4) & (native['rho_R'][j] > native['rho_R'].max()*1e-4)
        for key, name, color in (('epsilon1', r'$\epsilon^{(1)}$', 'black'), ('epsilon1_GI', r'$\epsilon^{(1)}_{GI}$', '#0072B2'), ('epsilon1_GD', r'$\epsilon^{(1)}_{GD}$', '#D55E00')):
            axs[1, 0].plot(native['q'], np.where(mask, native[key][j], np.nan), color=color, lw=2.3, label=name)
        silhouette(axs[1, 0], native['q'], jd)
    else:
        note(axs[0, 0], 'QD wavefunction unavailable: no TDPES')
        note(axs[1, 0], 'CTMQC TDPES is not reconstructed')
    for method, f in frames.items():
        support = f['rho_R'] > f['rho_R'].max()*1e-4
        axs[0, 1].plot(reader.R, np.where(support, f['outer_mean_energy'], np.nan), color=COLOR[method], lw=2.4, label=LABELS[method])
        support_q = (f['joint'][i] > f['joint'][i].max()*1e-4) & support[i]
        axs[1, 1].plot(reader.q, np.where(support_q, f['mean_CBO_energy'][i], np.nan), color=COLOR[method], lw=2.4, label=LABELS[method])
    energies = np.linalg.eigvalsh(local_hamiltonian(reader.c, reader.R))
    for k in range(min(4, energies.shape[1])):
        axs[0, 1].plot(reader.R, energies[:, k], '--', lw=1.3, color=BRANCH[k], alpha=.7)
    for k in range(2):
        axs[1, 1].plot(reader.q, reader.ce[i, :, k], '--', lw=1.5, color=('#009E73', '#8E44AD')[k], label=fr'$E_{{{k}}}^{{CBO}}$')
    silhouette(axs[0, 1], reader.R, reference['rho_R'])
    silhouette(axs[1, 1], reader.q, reference['joint'][i])
    specs = [('QD only: outer TDPES (positive gauge)', '$R$ ($a_0$)', 'R_range', 'epsilon2'),
             ('Mean electron-photon energy (not TDPES)', '$R$ ($a_0$)', 'R_range', 'outer_energy'),
             (fr'QD only: $\epsilon^{{(1)}}$ at $R={R:.2f}\,a_0$', '$q_c$ (a.u.)', 'q_range', 'epsilon1'),
             (fr'Mean CBO energy: $R={R:.2f}\,a_0$ (not TDPES)', '$q_c$ (a.u.)', 'q_range', 'cbo_energy')]
    for ax, (heading, x, xr, yr) in zip(axs.flat, specs):
        finish(ax, heading, x, 'Energy (Ha)', scales[xr], scales[yr])
    footer(fig, 'Left: native QD fields, GI includes geometric/projection terms. Right: smoothed conditional static-energy averages only.'
           +'\nEL outer energy includes photon kinetic energy; dashed outer curves: Pol0-3 in energy order. Support threshold: 1e-4.')
    return fig


def forces(frames, reader, scales, title):
    title, sub = snapshot_heading(frames, title+' | Forces and nuclear current', reader)
    fig, axs = layout(title, subtitle=sub)
    native = frames['qd']['native'] if 'qd' in frames else None
    if native is not None:
        support = native['rho_R'] > native['rho_R'].max()*1e-4
        for key, name, color in [('force', r'$-\partial_R\epsilon^{(2)}+\partial_t\alpha$', 'black'),
                ('quantum_amplitude_force', r'$-\partial_R Q_R$', '#009E73'),
                ('hydrodynamic_force', 'QD hydrodynamic total', '#8E44AD')]:
            axs[0, 0].plot(native['R'], np.where(support, native[key], np.nan), color=color, lw=2.2, label=name)
        silhouette(axs[0, 0], native['R'], native['rho_R'])
    else:
        note(axs[0, 0], 'QD coefficients unavailable')
    for method, f in frames.items():
        support = f['rho_R'] > f['rho_R'].max()*1e-4
        name = 'QD conditional interaction force' if method == 'qd' else LABELS[method]+' MF+CT'
        axs[0, 1].plot(reader.R, np.where(support, f['mf_force']+f['ct_force'], np.nan), color=COLOR[method], lw=2.3, label=name)
        axs[1, 0].plot(reader.R, f['current_R'], color=COLOR[method], lw=2.3, label=LABELS[method])
        axs[1, 1].plot(reader.R, f['rho_R'], color=COLOR[method], lw=2.3, label=LABELS[method])
    for ax, name, ylabel, scale in zip(axs.flat,
            ('QD effective field and quantum-amplitude force', 'Interaction / trajectory forces (not panel A)', 'Nuclear probability current', 'Nuclear density context'),
            ('Force (a.u.)', 'Force (a.u.)', r'Probability / $t_{au}$', 'Probability density'),
            ('force_qd', 'force_ct', 'current', 'density_R')):
        finish(ax, name, '$R$ ($a_0$)', ylabel, scales['R_range'], scales[scale], zero=scale != 'density_R')
    footer(fig, 'QD hydrodynamic force includes the marginal quantum-amplitude potential; CTMQC trajectory forces need not equal it.'
           +'\nQD conditional interaction force is the local mean of -d(H-T_R)/dR. Current/density use matched smoothing.')
    return fig


def paper_overview(groups, model, title):
    fig, axs = layout(title, subtitle='On/off resonance comparison inspired by PCCP 2026 Figs. 5, 8, 9, 10')
    for j, resonance in enumerate(('on', 'off')):
        entry = groups.get(resonance)
        if entry is None:
            for ax in axs[:, j]: note(ax, resonance+'-resonance unavailable')
            continue
        archives, interval = entry
        trace(axs[0, j], archives, 'photon_number', interval)
        trace(axs[1, j], archives, 'bo_population', interval, 1)
        finish(axs[0, j], resonance+'-resonance', '$t$ (fs)', r'$\langle N\rangle$', interval)
        finish(axs[1, j], resonance+'-resonance', '$t$ (fs)', r'$P_{S_1}$', interval)
    # Rows compare identical observables, so use exactly the same y limits.
    for row in axs:
        ylim = (min(ax.get_ylim()[0] for ax in row), max(ax.get_ylim()[1] for ax in row))
        for ax in row: ax.set_ylim(ylim)
    footer(fig, 'Common observables; finite-Fock QD. Check matching.json before attributing differences solely to the CTMQC approximation.'
           +'\nEL(paper) and bare QD/POL initial preparations differ; on/off time windows may differ.')
    return fig
