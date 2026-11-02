"""Read saved QD + CTMQC EL/POL archives and make three-way diagnostics.

Example: python -m multi_component_exact_factorization.pen_ctmqc_2026.compare
  --ctmqc-root CT_ROOT --qd-root QD_ROOT --out COMPARISON_ROOT --every-fs 5
No dynamics rerun, input writes, or full reconstructed-field cache.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import shutil
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from ..pen_perspectives_2026.model import local_hamiltonian
from ..pen_perspectives_2026.workflow import atomic_json
from .compare_data import (load_group, overlap, aligned_errors, nearest_frame,
                           choose_snapshots, FrameReader, snapshot_metrics)
from . import compare_plot as plot
from .render import STYLE, limits


def get_frames(reader, t, tolerance):
    frames, omitted = {}, []
    for method, a in reader.archives.items():
        index = nearest_frame(a, t, tolerance)
        if index is None:
            omitted.append(method)
        else:
            frames[method] = reader.read(method, index)
    return frames, omitted


def scan(reader, times, tolerance, slice_R):
    """Time-global scales from saved frames; only short scale samples retained."""
    names = ('density_R', 'density_q', 'epsilon1', 'epsilon2', 'outer_energy',
             'cbo_energy', 'force_qd', 'force_ct', 'current')
    values = {k: [] for k in names}
    rmins, rmaxs, qmins, qmaxs = [], [], [], []
    def add(key, value):
        v = np.asarray(value).ravel()
        values[key].append(v[::max(1, len(v)//3000)])
    for t in times:
        frames, _ = get_frames(reader, t, tolerance)
        if not frames:
            continue
        reference = frames.get('qd', next(iter(frames.values())))
        i = np.argmax(reference['rho_R']) if slice_R is None else np.argmin(abs(reader.R-slice_R))
        for f in frames.values():
            for x, d, lo, hi in ((reader.R, f['rho_R'], rmins, rmaxs), (reader.q, f['rho_q'], qmins, qmaxs)):
                idx = np.flatnonzero(d > d.max()*1e-4)
                if len(idx):
                    lo.append(float(x[idx[0]])); hi.append(float(x[idx[-1]]))
            mask = f['rho_R'] > f['rho_R'].max()*1e-4
            jointmask = f['joint'][i] > f['joint'][i].max()*1e-4
            add('density_R', f['rho_R']); add('density_q', f['rho_q'])
            add('outer_energy', f['outer_mean_energy'][mask])
            add('cbo_energy', f['mean_CBO_energy'][i, jointmask])
            add('cbo_energy', reader.ce[i, jointmask])
            add('force_ct', (f['mf_force']+f['ct_force'])[mask])
            add('current', f['current_R'])
            native = f['native']
            if native is not None:
                use = native['rho_R'] > native['rho_R'].max()*1e-4
                for k in ('epsilon2', 'epsilon2_GI', 'epsilon2_GD'):
                    add('epsilon2', native[k][use])
                j = np.argmin(abs(native['R']-reader.R[i]))
                useq = (native['rho_qR'][j] > native['rho_qR'][j].max()*1e-4) & use[j]
                for k in ('epsilon1', 'epsilon1_GI', 'epsilon1_GD'):
                    add('epsilon1', native[k][j, useq])
                for k in ('force', 'quantum_amplitude_force', 'hydrodynamic_force'):
                    add('force_qd', native[k][use])
    scales = {k: limits(v if v else [np.array([0.])], symmetric=k in ('force_ct', 'force_qd', 'current')) for k, v in values.items()}
    for k in ('density_R', 'density_q'):
        scales[k] = (0., max(scales[k][1], 1e-10))
    force_range = (min(scales['force_qd'][0], scales['force_ct'][0]),
                   max(scales['force_qd'][1], scales['force_ct'][1]))
    scales['force_qd'] = scales['force_ct'] = force_range
    scales['R_range'] = (max(reader.R[0], min(rmins)-.25), min(reader.R[-1], max(rmaxs)+.25)) if rmins else (reader.R[0], reader.R[-1])
    scales['q_range'] = (max(reader.q[0], min(qmins)-.5), min(reader.q[-1], max(qmaxs)+.5)) if qmins else (reader.q[0], reader.q[-1])
    mask = (reader.R >= scales['R_range'][0]) & (reader.R <= scales['R_range'][1])
    energy = np.linalg.eigvalsh(local_hamiltonian(reader.c, reader.R))[:, :4]
    lo, hi = limits([energy[mask]])
    scales['branch_lift'] = .16*(hi-lo)/scales['density_R'][1]
    scales['branch_energy'] = (lo, hi+.2*(hi-lo))
    # Static references in the outer-energy panel must remain visible.
    scales['outer_energy'] = limits([np.array(scales['outer_energy']), energy[mask]])
    return scales


def render_group(archives, warnings, relative, out, args):
    out.mkdir(parents=True, exist_ok=True)
    title = str(relative).replace('/', ' | ')
    interval = overlap(archives, args.t_max_fs)
    report = dict(group=str(relative), methods=list(archives), warnings=warnings,
        sources={k: str(a.path.resolve()) for k, a in archives.items()}, common_time_fs=interval,
        source_status={k: a.status.get('status', 'unknown') for k, a in archives.items()},
        source_numerical_checks={k: a.status for k, a in archives.items()},
        trajectory_settings={k: vars(a.settings) for k, a in archives.items() if a.settings is not None},
        initial_preparation={k: 'bare S1 x vacuum' if k == 'qd' else a.settings.initial for k, a in archives.items()},
        limitations=['No CTMQC full wavefunction reconstruction; CTMQC exact EF fields unavailable.',
                     'QD is the repository finite-Fock TDSE, not digitized/published paper QD.',
                     'Same Hamiltonian parameters do not remove EL classical-space/preparation differences.'])
    atomic_json(out/'matching.json', report)
    if interval is None:
        report['skipped'] = 'No common observable time interval'
        atomic_json(out/'comparison.json', report)
        return report, None
    warn = ('EL(paper) preparation differs from bare QD/POL.' if any('different preparations' in w for w in warnings)
            else 'Common bare preparation where available; EL photon space remains classical.')
    if any('SKIPPED' in w for w in warnings):
        warn += ' A mismatched archive was excluded; see matching.json.'
    errors = aligned_errors(archives, interval)
    report['observable_errors'] = errors
    report['photon_number_identity_max_abs'] = {}
    report['basis_checks'] = {}
    report['initial_common_observables'] = {}
    for method, a in archives.items():
        report['initial_common_observables'][method] = {
            k: a.rows[0][k] for k in ('t_fs', 'photon_number', 'photon_number_se', 'bo_population', 'pol_population', 'cbo_population', 'kinetic_R')
            if k in a.rows[0]}
        report['basis_checks'][method] = {
            k: float(np.max(a.values(k))) for k in ('top_fock_population', 'omitted_coupling_norm_Ha')
            if a.values(k) is not None}
        number, q2, p2 = (a.values(k) for k in ('photon_number', 'q2', 'p2'))
        if all(v is not None for v in (number, q2, p2)):
            report['photon_number_identity_max_abs'][method] = float(np.max(abs(
                a.config.omega*q2/2+p2/(2*a.config.omega)-.5-number)))
    report['interpolation'] = 'Scalar observables only, linear within overlap; no wave/trajectory interpolation.'
    with plt.rc_context(STYLE):
        for name, factory in (('common_observables', plot.observations), ('photon_budget', plot.photon_budget),
                              ('polaritonic_populations', plot.pol_populations)):
            fig = factory(archives, interval, title, warn)
            fig.savefig(out/(name+'.png'), dpi=args.dpi)
            plt.close(fig)
        fig = plot.accuracy(archives, interval, errors, title, warn)
        fig.savefig(out/'differences_and_checks.png', dpi=args.dpi)
        plt.close(fig)
    frame_archives = {k: a for k, a in archives.items() if len(a.paths)}
    frame_interval = overlap(frame_archives, args.t_max_fs, frames=True)
    if not frame_archives or frame_interval is None:
        report['frame_warning'] = 'No common saved frames; time traces only. No rerun was launched.'
        atomic_json(out/'comparison.json', report)
        return report, interval
    reader = FrameReader(frame_archives, args.density_width_scale)
    if args.slice_R is not None and not reader.R[0] <= args.slice_R <= reader.R[-1]:
        raise ValueError('slice-R is outside comparison display grid')
    snapshots = choose_snapshots(frame_archives, frame_interval, args.every_fs, args.snapshot_times)
    movies = choose_snapshots(frame_archives, frame_interval, args.movie_every_fs) if args.movies else []
    times = sorted(set(snapshots+movies))
    print(f'COMPARE {relative}: {len(times)} saved-frame targets, methods={list(archives)}, no propagation', flush=True)
    scales = scan(reader, times, args.time_tolerance_fs, args.slice_R)
    report.update(scales=scales, density_widths=reader.widths.tolist(), snapshot_times_requested=args.snapshot_times,
                  frame_interval_fs=frame_interval, snapshots=[],
                  smoothing='QD and CTMQC densities at matched Gaussian resolution; static-energy numerator/denominator smoothed separately; native QD EF fields unsmoothed.')
    families = dict(density_comparison=plot.density, polaritonic_guidance=plot.guidance,
                    potential_context=plot.potential, force_and_current=plot.forces)
    with plt.rc_context(STYLE), ExitStack() as stack:
        writers = {}
        for i, t in enumerate(times):
            frames, missing = get_frames(reader, t, args.time_tolerance_fs)
            if not frames:
                continue
            metric = snapshot_metrics(frames)
            metric.update(target_time_fs=t, missing_or_outside_tolerance=missing, png=t in snapshots)
            report['snapshots'].append(metric)
            for name, factory in families.items():
                folder = out/name
                folder.mkdir(exist_ok=True)
                extra = dict(slice_R=args.slice_R) if name == 'potential_context' else {}
                fig = factory(frames, reader, scales, title, **extra)
                if t in snapshots:
                    fig.savefig(folder/f'frame_{i:05d}_t{t:09.4f}fs.png', dpi=args.dpi)
                if t in movies:
                    if name not in writers:
                        writer = FFMpegWriter(fps=20, codec='libx264', extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', '-threads', '1', '-crf', '22'])
                        stack.enter_context(writer.saving(fig, folder/(name+'.mp4'), dpi=args.movie_dpi))
                        writers[name] = writer
                    writers[name].fig = fig
                    writers[name].grab_frame()
                plt.close(fig)
    report['outputs'] = ['common_observables.png', 'photon_budget.png', 'polaritonic_populations.png', 'differences_and_checks.png']+list(families)
    atomic_json(out/'comparison.json', report)
    return report, interval


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ctmqc-root', type=Path, required=True)
    p.add_argument('--qd-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', choices=['all', 'nonadiabatic', 'rabi'], default='all')
    p.add_argument('--resonance', choices=['both', 'on', 'off'], default='both')
    p.add_argument('--case', choices=['all', 'zero', 'weak', 'strong'], default='all')
    p.add_argument('--dse', choices=['both', 'on', 'off'], default='both')
    p.add_argument('--every-fs', type=float, default=5., help='Snapshot PNG cadence, include endpoints')
    p.add_argument('--snapshot-times', type=float, nargs='+', help='Explicit fs times, nearest actual frames')
    p.add_argument('--t-max-fs', type=float)
    p.add_argument('--time-tolerance-fs', type=float, help='Otherwise nearest within 0.51*largest saved-frame interval; actual times shown')
    p.add_argument('--density-width-scale', type=float, default=1., help='Display KDE only, never dynamics quantum momentum')
    p.add_argument('--slice-R', type=float, help='Fixed R slice; default follows QD density peak')
    p.add_argument('--movies', action='store_true', help='Four optional MP4 families; PNGs/JSON by default')
    p.add_argument('--movie-every-fs', type=float, default=.25)
    p.add_argument('--dpi', type=int, default=140)
    p.add_argument('--movie-dpi', type=int, default=85)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    nums = [args.every_fs, args.movie_every_fs, args.density_width_scale]
    if not np.all(np.isfinite(nums)) or min(nums) <= 0:
        p.error('Positive finite cadence and density-width-scale required')
    for value in (args.t_max_fs, args.time_tolerance_fs):
        if value is not None and (not np.isfinite(value) or value < 0):
            p.error('Time limits/tolerance must be finite and nonnegative')
    if args.snapshot_times is not None and (not np.all(np.isfinite(args.snapshot_times)) or min(args.snapshot_times) < 0):
        p.error('Snapshot times must be finite and nonnegative')
    if args.slice_R is not None and not np.isfinite(args.slice_R):
        p.error('slice-R must be finite')
    if min(args.dpi, args.movie_dpi) < 50:
        p.error('dpi>=50 required')
    if args.movies and shutil.which('ffmpeg') is None:
        p.error('ffmpeg required for --movies; omit it for PNGs')
    if args.out.resolve() in (args.ctmqc_root.resolve(), args.qd_root.resolve()):
        p.error('Use a separate comparison output directory, not an input root')
    groups = sorted({path.relative_to(root) for root in (args.ctmqc_root, args.qd_root)
                     for path in root.glob('*/*/dse_*/*') if path.is_dir() and path.name in ('zero', 'weak', 'strong')})
    groups = [g for g in groups if (args.model == 'all' or g.parts[0] == args.model)
              and (args.resonance == 'both' or g.parts[1] == args.resonance)
              and (args.dse == 'both' or g.parts[2] == 'dse_'+args.dse)
              and (args.case == 'all' or g.parts[3] == args.case)]
    if not groups:
        p.error('No scenario folders found. Roots must contain model/on|off/dse_off|on/case/')
    args.out.mkdir(parents=True, exist_ok=True)
    started, reports, summary = time.perf_counter(), [], {}
    for relative in groups:
        archives, warnings = load_group(args.ctmqc_root, args.qd_root, relative)
        for warning in warnings:
            print(f'{relative}: {warning}', flush=True)
        if not archives:
            reports.append(dict(group=str(relative), skipped='No readable archive', warnings=warnings))
            continue
        # Failures in derived figures must not alter/restart any simulation.
        try:
            report, interval = render_group(archives, warnings, relative, args.out/relative, args)
            reports.append(report)
            if interval is not None:
                key = (relative.parts[0], relative.parts[2], relative.parts[3])
                summary.setdefault(key, {})[relative.parts[1]] = (archives, interval)
        except (OSError, ValueError, KeyError) as error:
            reports.append(dict(group=str(relative), error=repr(error), warnings=warnings))
            print(f'COMPARISON FAILED {relative}: {error}', flush=True)
    with plt.rc_context(STYLE):
        for (model, dse, case), data in summary.items():
            fig = plot.paper_overview(data, model, f'{model} | {dse} | {case} | QD / EL / POL')
            fig.savefig(args.out/f'paper_observables_{model}_{dse}_{case}.png', dpi=args.dpi)
            plt.close(fig)
    atomic_json(args.out/'comparison_report.json', dict(wall_seconds=time.perf_counter()-started,
        groups=reports, source_ctmqc=str(args.ctmqc_root.resolve()), source_qd=str(args.qd_root.resolve()),
        no_propagation=True, no_input_writes=True, paper='https://doi.org/10.1039/D6CP02020H'))
    if any('error' in report for report in reports):
        raise SystemExit('Some comparison groups failed; see comparison_report.json. Input calculations unchanged.')


if __name__ == '__main__':
    main()
