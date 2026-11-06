"""Cached internal-channel diagnostics, not reconstructed exact EF potentials.

Channel identity is fixed by the driver's declared adiabatic ordering. Selected
channels and all movie axes are fixed over the saved interval. Marker *area*
on the internal energy curves encodes a trajectory's channel population; it is
not a wavepacket amplitude or a reconstructed quantum nuclear wavefunction.
Only small cached diagnostic arrays are loaded; no G/checkpoint wave is read.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .renderer import STYLE, COLORS, _axis, _legend, _nice_limit


KEYS = ('t_fs', 'R', 'P', 'eta', 'channel_population', 'channel_energy',
        'active_state', 'Q_raw', 'ct_force_correction', 'hops_attempted',
        'hops_accepted', 'hops_frustrated', 'projection_residual', 'energy',
        'active_surface_energy', 'photon_number')


def load_frame(path):
    with np.load(path, allow_pickle=False) as archive:
        frame = {key: archive[key].copy() for key in KEYS if key in archive}
    for key in ('t_fs', 'R', 'P', 'eta', 'channel_population', 'channel_energy'):
        if key not in frame:
            raise ValueError('%s lacks %s' % (path, key))
    frame['t_fs'] = float(frame['t_fs'])
    return frame


def _mean(frame, key):
    value = np.asarray(frame.get(key, np.nan))
    return float(value) if value.ndim == 0 else np.einsum('i,i...->...', frame['eta'], value)


def prepare(root):
    """Streaming prepass with stable channel selection, limits and small traces."""
    root = Path(root)
    files = sorted((root/'frames').glob('frame_*.npz'))
    if not files:
        raise FileNotFoundError('No channel frames under '+str(root/'frames'))
    traces = {key: [] for key in ('t_fs', 'channel_population', 'energy',
                                  'active_surface_energy', 'photon_number',
                                  'Q_raw', 'Q_raw_rms', 'ct_force_correction',
                                  'ct_force_correction_rms', 'projection_residual',
                                  'hops_attempted', 'hops_accepted', 'hops_frustrated')}
    position, momentum = [], []
    for path in files:
        frame = load_frame(path)
        for key in traces:
            if key.endswith('_rms'):
                source = np.asarray(frame.get(key[:-4], np.nan))
                value = np.sqrt(np.sum(frame['eta']*source**2))
            else:
                value = _mean(frame, key)
            traces[key].append(value)
        position.extend(frame['R'].tolist())
        momentum.extend(frame['P'].tolist())
    traces = {key: np.asarray(value) for key, value in traces.items()}
    population = traces['channel_population']
    if population.ndim != 2:
        raise ValueError('Channel count must remain fixed for one movie')
    rank = np.argsort(-population.mean(axis=0), kind='stable')
    selected = np.sort(rank[:4])
    energies = []
    for path in files:
        energies.extend(load_frame(path)['channel_energy'][:, selected].ravel().tolist())
    config_path, metadata_path = root/'config.json', root/'metadata.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    label = metadata.get('method_label')
    if label is None:
        label = 'FSSH' if config.get('method') == 'tsh' else 'Experimental CTMQC'
        label += (' + Gaussian bridge projection' if config.get('compression') == 'gaussian'
                  else ' internal-channel reference')
    return dict(root=root, files=files, traces=traces, selected=selected,
                colors={int(channel): COLORS[i] for i, channel in enumerate(rank[:6])},
                summary_selected=np.sort(rank[:6]), method_label=label,
                limits=dict(R=_nice_limit(position), P=_nice_limit(momentum),
                            energy=_nice_limit(energies), t=_nice_limit(traces['t_fs'])))


def _population_lines(ax, data, selected):
    t, population = data['traces']['t_fs'], data['traces']['channel_population']
    for channel in selected:
        ax.plot(t, population[:, channel], color=data['colors'][int(channel)], label='Channel %d' % channel)
    if len(selected) < population.shape[1]:
        other = population.sum(axis=1)-population[:, selected].sum(axis=1)
        ax.plot(t, other, '--', color='.35', label='Other retained channels')
    ax.set_ylim(-.025, max(1.025, 1.025*float(population.sum(axis=1).max())))
    ax.set_xlim(data['limits']['t'])
    _axis(ax, 't (fs)', 'Population', 'Internal-channel populations')
    _legend(ax)


def plot_summary(data):
    tr, t = data['traces'], data['traces']['t_fs']
    fig, axes = plt.subplots(3, 2, figsize=(16, 16), layout='constrained')
    _population_lines(axes[0, 0], data, data['summary_selected'])
    axes[0, 1].plot(t, tr['photon_number'], color=COLORS[1])
    _axis(axes[0, 1], 't (fs)', r'$\langle n_{\rm ph}\rangle$', 'Photon number')
    if not np.any(np.isfinite(tr['photon_number'])):
        axes[0, 1].text(.5, .5, 'Not applicable / unavailable', ha='center', transform=axes[0, 1].transAxes)
    axes[1, 0].plot(t, tr['energy']-tr['energy'][0], color=COLORS[0],
                    label='Quantum expectation + heavy kinetic')
    if np.any(np.isfinite(tr['active_surface_energy'])):
        value = tr['active_surface_energy']
        axes[1, 0].plot(t, value-value[0], color=COLORS[1], ls='--',
                       label='FSSH active-surface energy + heavy kinetic')
    axes[1, 0].axhline(0, color='.4', ls='--', lw=1)
    _axis(axes[1, 0], 't (fs)', 'Energy change (Ha)', 'Energy diagnostics (distinct definitions)')
    _legend(axes[1, 0])
    axes[1, 1].plot(t, tr['projection_residual'], color=COLORS[3])
    _axis(axes[1, 1], 't (fs)', 'Amplitude-rate residual (a.u.)', 'Gaussian projection: ensemble mean')
    axes[2, 0].plot(t, tr['Q_raw'], color=COLORS[0], label='Raw quantum momentum')
    axes[2, 0].plot(t, tr['Q_raw_rms'], color=COLORS[0], ls=':', label='Raw momentum RMS')
    _axis(axes[2, 0], 't (fs)', 'Momentum (a.u.)', 'Ensemble-mean CT diagnostics')
    twin = axes[2, 0].twinx()
    twin.plot(t, tr['ct_force_correction'], color=COLORS[1], ls='--', label='CT force correction')
    twin.plot(t, tr['ct_force_correction_rms'], color=COLORS[3], ls=':', label='CT force RMS')
    twin.set_ylabel('Force correction (Ha/a0)', color=COLORS[1])
    axes[2, 0].tick_params(axis='y', labelcolor=COLORS[0])
    twin.tick_params(axis='y', labelcolor=COLORS[1])
    lines = axes[2, 0].lines+twin.lines
    axes[2, 0].legend(lines, [line.get_label() for line in lines], fontsize=10, ncol=2)
    for key, label, color in zip(('hops_attempted', 'hops_accepted', 'hops_frustrated'),
                                ('Attempted', 'Accepted', 'Frustrated'), COLORS):
        axes[2, 1].step(t, tr[key], where='post', label=label, color=color)
    _axis(axes[2, 1], 't (fs)', 'Cumulative event count', 'Surface-hop events (TSH only)')
    _legend(axes[2, 1])
    for ax in axes.ravel():
        ax.set_xlim(data['limits']['t'])
    fig.suptitle(data['method_label']+'\nInternal-channel diagnostics; not exact EF scalars')
    return fig


def plot_snapshot(data, frame):
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), layout='constrained')
    R, P, population = frame['R'], frame['P'], frame['channel_population']
    order, selected = np.argsort(R), data['selected']
    for channel in selected:
        color = data['colors'][int(channel)]
        energy = frame['channel_energy'][:, channel]
        axes[0, 0].plot(R[order], energy[order], '--', color=color, alpha=.8,
                       label='Channel %d' % channel)
        axes[0, 0].scatter(R, energy, s=120*population[:, channel],
                          color=color, alpha=.6, edgecolors='none')
        axes[1, 0].scatter(R, population[:, channel], s=18, color=color, alpha=.65,
                          label='Channel %d' % channel)
    _axis(axes[0, 0], 'R (a0)', 'Energy (Ha)', 'Internal surfaces; marker area = population')
    axes[0, 0].set_ylim(data['limits']['energy'])
    _legend(axes[0, 0])
    active = np.asarray(frame.get('active_state', np.full(len(R), -1)))
    if np.any(active >= 0):
        for channel in selected:
            keep = active == channel
            axes[0, 1].scatter(R[keep], P[keep], s=22, color=data['colors'][int(channel)],
                               label='Active %d' % channel)
        other = ~np.isin(active, selected)
        axes[0, 1].scatter(R[other], P[other], s=22, color='.5', label='Other active')
        _legend(axes[0, 1])
    else:
        axes[0, 1].scatter(R, P, s=22, color=COLORS[0], alpha=.7)
    _axis(axes[0, 1], 'R (a0)', 'Momentum (a.u.)', 'Heavy phase-space trajectories')
    axes[0, 1].set_ylim(data['limits']['P'])
    _axis(axes[1, 0], 'R (a0)', 'Population', 'Trajectory-resolved channel populations')
    axes[1, 0].set_ylim(-.025, 1.025)
    _population_lines(axes[1, 1], data, selected)
    axes[1, 1].axvline(frame['t_fs'], color='black', ls='--', lw=1.4)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlim(data['limits']['R'])
    fig.suptitle('%s\nt = %.5g fs; approximate internal-channel analysis' %
                 (data['method_label'], frame['t_fs']))
    return fig


def render(root, *, movies=False, fps=12, dpi=160):
    """Save summary/representative PNGs; optional temporary-frame MP4 encoding."""
    if fps <= 0 or dpi <= 0:
        raise ValueError('fps and dpi must be positive')
    data = prepare(root)
    out = data['root']/'channels'
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig = plot_summary(data)
        fig.savefig(out/'summary.png', dpi=dpi)
        plt.close(fig)
        for index in sorted({0, len(data['files'])//2, len(data['files'])-1}):
            fig = plot_snapshot(data, load_frame(data['files'][index]))
            fig.savefig(out/('snapshot_%06d.png' % index), dpi=dpi)
            plt.close(fig)
        if movies:
            ffmpeg = shutil.which('ffmpeg')
            if ffmpeg is None:
                warnings.warn('ffmpeg unavailable; channel PNGs were saved, MP4 skipped')
                return out
            with tempfile.TemporaryDirectory(prefix='mcef_channel_movie_') as tmp:
                for index, path in enumerate(data['files']):
                    fig = plot_snapshot(data, load_frame(path))
                    fig.savefig(Path(tmp)/('frame_%06d.png' % index), dpi=100)
                    plt.close(fig)
                subprocess.run([ffmpeg, '-y', '-v', 'error', '-framerate', str(fps),
                                '-i', str(Path(tmp)/'frame_%06d.png'), '-c:v', 'libx264',
                                '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                                str(out/'channels.mp4')], check=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--movies', action='store_true')
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args(argv)
    print(render(args.root, movies=args.movies, fps=args.fps))


if __name__ == '__main__':
    main()
