"""Paper-facing on/off and Rabi summaries, separate from existing EF panels.

CBO populations and bare BO populations are different projectors. Plot both
with explicit names; do not silently compare one with a differently defined
paper population. No schematic Rabi sine curve is substituted for TDSE data.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import Config
from .resonance import scenario_report, PAPER_FIGURE_WINDOWS_FS
from .workflow import atomic_json


def exchange_summary(out, c, data, checks, cbop, moments):
    """Small traces from cached waves; no propagation or giant field archives."""
    out = Path(out)
    times = np.array([r['t_fs'] for r in checks])
    summary = dict(scenario=scenario_report(c), t_fs=times.tolist(),
                   photon_number=list(data['photon_number']),
                   bare_excited_population=np.asarray(data['bo_population'])[:, 1].tolist(),
                   cbo_excited_population=np.asarray(cbop)[:, 1].tolist(),
                   R_mean=[v[0] for v in moments], R_std=[v[1] for v in moments],
                   population_note='Bare BO and q-dependent CBO projections differ at finite coupling.',
                   norm_error=float(np.max(abs(np.array(data['norm'])-1))))
    atomic_json(out/'exchange_summary.json', summary)
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    axes[0].plot(times, summary['photon_number'], color='#0072B2', lw=2.2)
    axes[0].set(title='Photon excitation', ylabel=r'$\langle N\rangle$')
    axes[1].plot(times, summary['cbo_excited_population'], color='#D55E00', lw=2.2, label='CBO excited population')
    axes[1].plot(times, summary['bare_excited_population'], color='#009E73', ls='--', lw=2, label='Bare BO excited population')
    axes[1].set(title='Electronic excitation: two explicitly different projections', ylabel='Population')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=12)
    axes[2].plot(times, np.array(summary['R_mean'])-summary['R_mean'][0], lw=2, color='#CC79A7', label=r'$\langle R\rangle-\langle R\rangle_0$')
    axes[2].plot(times, np.array(summary['R_std'])-summary['R_std'][0], lw=2, ls='--', color='#333333', label=r'$\sigma_R-\sigma_{R,0}$')
    axes[2].set(title='Nuclear motion (not a reaction yield for the Rabi model)', ylabel=r'Change ($a_0$)', xlabel='t (fs)')
    axes[2].legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, fontsize=12)
    for ax in axes:
        ax.axhline(0, ls=':', lw=.8, color='.5')
        ax.set_xlim(0, max(float(times[-1]), .01))
        ax.tick_params(labelsize=12)
    fig.suptitle(summary['scenario']['label']+f' | omega={c.omega:g} au, g={c.g:g}, DSE {"on" if c.dse else "off"}', fontsize=16)
    fig.savefig(out/'excitation_exchange.png', dpi=150)
    plt.close(fig)
    return summary


def resonance_comparison(model_root, t_max_fs=None):
    """Load cached traces only; one on/off comparison per DSE setting."""
    model_root = Path(model_root)
    plots_name = 'plots' if t_max_fs is None else f'plots_to_{t_max_fs:g}fs'
    output = model_root/('resonance_comparison' if t_max_fs is None else f'resonance_comparison_to_{t_max_fs:g}fs')
    output.mkdir(exist_ok=True)
    colors = dict(zero='#555555', weak='#0072B2', strong='#D55E00')
    products = []
    for dse in ('off', 'on'):
        records = []
        for resonance in ('on', 'off'):
            for case in ('zero', 'weak', 'strong'):
                folder = model_root/resonance/f'dse_{dse}'/case
                path = folder/plots_name/'exchange_summary.json'
                if path.exists():
                    records.append((resonance, case, json.loads(path.read_text()),
                                    Config(**json.loads((folder/'config.json').read_text()))))
        if not records:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex='col', constrained_layout=True)
        handles, labels = [], []
        present = set()
        seen_cases = set()
        last_times = {}
        for resonance, case, data, c in records:
            col = int(resonance == 'off')
            present.add(col)
            t = data['t_fs']
            last_times[col] = max(last_times.get(col, 0), t[-1])
            color = colors[case]
            line, = axes[0, col].plot(t, data['photon_number'], color=color, lw=2)
            if case not in seen_cases:
                handles.append(line)
                labels.append(f'{case}: g={c.g:g}')
                seen_cases.add(case)
            axes[0, col].set_title(f'{resonance}-resonance: omega={c.omega:g} au', fontsize=17)
            axes[1, col].plot(t, data['cbo_excited_population'], color=color, lw=2)
            axes[1, col].plot(t, data['bare_excited_population'], color=color, lw=1.6, ls='--')
        # Zoom off-resonance amplitudes automatically, but keep explicit numeric
        # ticks. These columns intentionally have independent y scales.
        for col in (0, 1):
            if col not in present:
                for ax in axes[:, col]:
                    ax.text(.5, .5, 'Not calculated', ha='center', transform=ax.transAxes)
            axes[0, col].set_ylabel(r'$\langle N\rangle$', fontsize=16)
            axes[1, col].set(ylabel='Excited population', xlabel='t (fs)')
            for ax in axes[:, col]:
                ax.tick_params(labelsize=13)
                ax.grid(alpha=.18)
                ax.ticklabel_format(axis='y', useOffset=False)
        title = 'Rabi oscillations' if model_root.name == 'rabi' else 'Nonadiabatic process'
        fig.suptitle(title+f' | DSE {dse}\nSolid: CBO population; dashed: bare BO population. Independent y scales.', fontsize=17)
        axes[0, 0].legend(handles, labels, loc='best', fontsize=12)
        fig.set_constrained_layout_pads(h_pad=.15, w_pad=.1, hspace=.12)
        path = output/f'on_off_dse_{dse}.png'
        fig.savefig(path, dpi=160)
        # Same calculated traces, now cropped to the axes in Figs.5/8/9/10.
        # In particular Fig.10 shows only 3.5 fs, not the 30 fs of Fig.9.
        for col, resonance in enumerate(('on', 'off')):
            if col in present:
                limit = PAPER_FIGURE_WINDOWS_FS[model_root.name][resonance]
                axes[1, col].set_xlim(0, max(min(last_times[col], limit), .01))
        fig.suptitle(title+f' | DSE {dse} | paper figure time windows\n'
                     'Solid: CBO population; dashed: bare BO population. Independent x/y scales.', fontsize=17)
        cropped = output/f'on_off_paper_window_dse_{dse}.png'
        fig.savefig(cropped, dpi=160)
        plt.close(fig)
        products.append(str(path))
    return products
