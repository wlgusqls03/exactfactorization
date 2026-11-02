"""Separate CTMQC CLI: 2 models x 2 resonances x 2 perspectives by default.

TDSE archives are read only for optional comparison, never used to propagate
CTMQC or to manufacture its wavefunction. No previous output is overwritten.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from ..pen_perspectives_2026.model import AU_FS, Config
from ..pen_perspectives_2026.resonance import paper_config, scenario_report, PAPER_FIGURE_WINDOWS_FS
from ..pen_perspectives_2026.workflow import atomic_json, atomic_npz, load_npz, source_record
from .core import Engine, Settings, observables


def identity(c, s, every_fs):
    names = [Path(__file__).with_name(n) for n in ('core.py', 'quantum_momentum.py')]
    names.append(Path(__file__).parents[1]/'pen_perspectives_2026'/'model.py')
    return dict(schema=1, molecular_config=asdict(c), trajectory_config=asdict(s), every_fs=every_fs,
                dynamics_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in names})


def capabilities():
    return dict(direct=['trajectory positions/momenta', 'quantum coefficients in fixed basis',
        'BO populations', 'CBO populations (EL)', 'polaritonic populations (POL)',
        'bare oscillator number (quantum POL / Wigner estimator EL)',
        'mean-field and CT forces', 'pair quantum momentum', 'population growth from CT term',
        'ensemble density and conditional probability', 'population-weighted static energy'],
        unavailable=['unique full complex Psi', 'exact a,b,alpha', 'exact epsilon1,epsilon2 including GI/GD',
            'exact Lambda wavefunction', 'exact S_q,S_adv,S_rel identity'],
        policy='No wavefunction reconstruction requested. Do not label KDE densities as exact EF amplitudes, '
               'weighted energies as full GI/TDPES, or CT forces as exact EF fields.')


def simulate(c, s, out, end_fs, every_fs=.1, resume=False):
    engine = Engine(c, s)
    if not np.all(np.isfinite([end_fs, every_fs])) or end_fs < 0 or every_fs <= 0:
        raise ValueError('Need finite end_fs>=0 and every_fs>0')
    out = Path(out)
    ident = identity(c, s, every_fs)
    total = int(math.ceil(end_fs/(AU_FS*c.dt)))
    cadence = max(1, int(round(every_fs/(AU_FS*c.dt))))
    start = time.perf_counter()
    old_wall, records, max_norm_step = 0., [], 0.
    guard_counts = np.zeros(3, dtype=np.int64)
    if resume:
        z = load_npz(out/'restart.npz')
        if json.loads(str(z['identity'].item())) != ident:
            raise ValueError('Resume rejected: physics/method/seed/width/dt/source/cadence changed')
        step, old_wall, max_norm_step = int(z['step']), float(z['wall_seconds']), float(z['max_norm_step_error'])
        guard_counts = z['guard_counts'].astype(np.int64)
        if total < step:
            raise ValueError('Requested end precedes restart')
        state = {k: z[k] for k in ('X', 'P', 'coeff', 'integrated_forces', 'anchor')}
        records = [r for r in json.loads((out/'observables.json').read_text()) if r['step'] <= step]
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'{out}: choose new directory or --resume')
        out.mkdir(parents=True, exist_ok=True)
        state, step = engine.initial(), 0
        atomic_json(out/'config.json', asdict(c))
        atomic_json(out/'trajectory_config.json', asdict(s))
        atomic_json(out/'scenario.json', scenario_report(c))
        atomic_json(out/'capabilities.json', capabilities())
        provenance = dict(identity=ident, source=source_record(),
            ctmqc_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            references=['https://doi.org/10.1039/D6CP02020H', 'https://doi.org/10.3390/molecules27134002'],
            equations='CT coefficient and force: Molecules 2022 Eqs16-18; pair Q: Eqs26-27. Standard 2/M force prefactor.',
            author_source_audited='G-CTMQC b7a2a6d95656e4e0535e744b102670d85b07fb88; independently implemented, not a port.',
            reproduction_status='NOT an author-code reproduction certificate. Paper inputs for Gaussian kernels/integrator '
                                'and fallback are not fully specified; see README.txt.',
            preparation=('EL: upper local CBO at (R,q), Eq25' if s.perspective == 'el' and s.initial == 'paper'
                         else 'bare S1(R), Eq27; POL includes exact Fock vacuum'),
            sampling='Random Wigner ensemble, not Bohmian trajectories. R/P draws shared across perspectives for same seed.',
            gaussian_density_sd=engine.bandwidths.tolist(), integrator='Coupled classical RK4, all ensemble terms reevaluated at each stage',
            boundary='Open analytic model; no wrapping/reflection; out-of-TDSE-box fraction reported',
            photon_number='Bare Pauli-Fierz oscillator N; EL estimator may be slightly negative from sampling; not clipped.',
            force_integral='Relative-to-mean integrated adiabatic force; inactive states reset at population_threshold.',
            singular_intercept_policy='Disable entire ill-conditioned/distant pair/direction; record counts. '
                                      'Center must lie inside ensemble bbox + intercept_limit_sigma * bandwidth. '
                                      'No silent uncorrected fallback or per-trajectory Q clipping.',
            requested_analysis=capabilities())
        atomic_json(out/'provenance.json', provenance)
    (out/'frames').mkdir(exist_ok=True)
    if shutil.disk_usage(out).free < 100*1024**2:
        raise OSError('Less than 100 MiB free')
    first_step = step

    def save():
        row = dict(step=step, t_fs=step*c.dt*AU_FS, max_norm_step_error_so_far=max_norm_step,
                   guard_counts_so_far=guard_counts.tolist(), **observables(engine, state))
        if not all(np.all(np.isfinite(v)) for v in row.values()):
            raise FloatingPointError('Nonfinite CTMQC observable')
        if records and records[-1]['step'] == step:
            records[-1] = row
        else:
            records.append(row)
        _, detail = engine.evaluate(state, state['anchor'])
        # Frames omit force-integral history and eigenvector anchors. Only the
        # latest atomic restart needs those large arrays for continuation.
        atomic_npz(out/'frames'/f'trajectories_{step:08d}.npz',
            X=state['X'], P=state['P'], coeff=state['coeff'], step=step, t_fs=row['t_fs'],
            mean_energy=detail['mean_energy'], mf_force=detail['mf_force'], ct_force=detail['ct_force'],
            q_pair_rms=detail['q_pair_weighted_local_rms'], ct_rates=detail['rates'])
        atomic_json(out/'observables.json', records)
        atomic_npz(out/'restart.npz', **state, step=step, identity=json.dumps(ident, sort_keys=True),
            wall_seconds=old_wall+time.perf_counter()-start, max_norm_step_error=max_norm_step, guard_counts=guard_counts)
        atomic_json(out/'status.json', dict(status='RUNNING', step=step, total_steps=total, t_fs=row['t_fs']))

    if not resume:
        save()
    try:
        while step < total:
            state, norm_error = engine.step(state)
            guard_counts += engine.last_step_guard_counts
            max_norm_step = max(max_norm_step, norm_error)
            step += 1
            if step % cadence == 0 or step == total:
                save()
                print(f'{s.perspective.upper()} {out.parent.name}: {step}/{total}, t={step*c.dt*AU_FS:.4f} fs', flush=True)
    except BaseException as error:
        atomic_json(out/'failure.json', dict(error=repr(error), attempted_step=step,
            instruction='Last atomic restart remains valid; diagnose failure before resume.'))
        atomic_json(out/'status.json', dict(status='FAILED', attempted_step=step,
                    last_checkpoint_step=records[-1]['step'] if records else None, error=repr(error)))
        raise
    elapsed = time.perf_counter()-start
    energy = np.array([r['energy'] for r in records])
    status = dict(status='COMPLETE_NOT_CERTIFIED', method=s.algorithm, perspective=s.perspective,
        step=step, total_steps=total, t_fs=step*c.dt*AU_FS, wall_seconds=old_wall+elapsed,
        latest_wall_seconds=elapsed, latest_steps=step-first_step, ntraj=s.ntraj,
        max_norm_step_error=max_norm_step,
        quantum_momentum_guard_counts=dict(zip(('attempted', 'disabled', 'distant'), guard_counts.tolist())),
        max_norm_error=max(r['coefficient_norm_max_error'] for r in records),
        max_energy_drift_Ha=float(np.max(abs(energy-energy[0]))),
        max_ct_only_ensemble_population_rate=max(float(np.max(abs(np.asarray(r['ct_population_rate'])))) for r in records),
        max_edge_probability=max(r['edge_probability'] for r in records),
        warning='CTMQC approximation, not exact TDSE. Norm conservation is not convergence. '
                'Original CTMQC is not exactly energy conserving. Converge dt, Ntraj, bandwidth, threshold and seeds.')
    atomic_json(out/'status.json', status)
    return status


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'run', 'render', 'compare'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', choices=['nonadiabatic', 'rabi', 'all'], default='all')
    p.add_argument('--resonance', choices=['on', 'off', 'both'], default='both')
    p.add_argument('--perspective', choices=['el', 'pol', 'both'], default='both')
    p.add_argument('--case', choices=['zero', 'weak', 'strong', 'all'], default='strong')
    p.add_argument('--dse', choices=['off', 'on', 'both'], default='off')
    p.add_argument('--weak-g', type=float, default=.001)
    p.add_argument('--strong-g', type=float, default=.01)
    p.add_argument('--ntraj', type=int, default=10000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--initial', choices=['paper', 'bare'], default='paper')
    p.add_argument('--algorithm', choices=['ctmqc', 'ehrenfest'], default='ctmqc')
    p.add_argument('--quantum-momentum', choices=['corrected', 'density'], default='corrected')
    p.add_argument('--bandwidth-scale', type=float, default=1.)
    p.add_argument('--population-threshold', type=float, default=.001)
    p.add_argument('--intercept-limit-sigma', type=float, default=10., help='Pair-center guard beyond ensemble box in density kernel widths; converge this numerical choice')
    p.add_argument('--dt', type=float, default=.1, help='Atomic units; coupled RK4, not TDSE splitting')
    p.add_argument('--nf', type=int, default=2, help='POL quantum Fock basis only; EL q is continuous classical')
    p.add_argument('--end-fs', type=float, help='Default: paper windows 30/40/30/3.5 fs')
    p.add_argument('--every-fs', type=float, default=.1)
    p.add_argument('--jobs', type=int, choices=[1, 2, 3, 4], default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--no-movies', action='store_true')
    p.add_argument('--movie-stride', type=int, default=1)
    p.add_argument('--png-every-fs', type=float, default=5.)
    p.add_argument('--dpi', type=int, default=85)
    p.add_argument('--t-max-fs', type=float, help='Render/compare only, no propagation')
    p.add_argument('--tdse-root', type=Path, help='Optional read-only paper_campaign root for matched overlays')
    return p


def build_tasks(args):
    models = ['nonadiabatic', 'rabi'] if args.model == 'all' else [args.model]
    resonances = ['on', 'off'] if args.resonance == 'both' else [args.resonance]
    views = ['el', 'pol'] if args.perspective == 'both' else [args.perspective]
    cases = ['zero', 'weak', 'strong'] if args.case == 'all' else [args.case]
    dses = ['off', 'on'] if args.dse == 'both' else [args.dse]
    g = dict(zero=0., weak=args.weak_g, strong=args.strong_g)
    tasks = []
    for model in models:
        for resonance in resonances:
            for dse in dses:
                for case in cases:
                    c = paper_config(model, resonance, dt=args.dt, nf=args.nf, g=g[case], dse=dse == 'on')
                    for perspective in views:
                        s = Settings(perspective, args.ntraj, args.seed, args.algorithm, args.initial,
                            args.quantum_momentum, args.bandwidth_scale, args.population_threshold,
                            intercept_limit_sigma=args.intercept_limit_sigma)
                        s.validate()
                        options = dict(end_fs=args.end_fs if args.end_fs is not None else PAPER_FIGURE_WINDOWS_FS[model][resonance],
                            every_fs=args.every_fs, resume=args.resume, render=not args.no_render,
                            movies=not args.no_movies, movie_stride=args.movie_stride,
                            png_every_fs=args.png_every_fs, dpi=args.dpi, t_max_fs=args.t_max_fs)
                        tasks.append((c, s, args.out/model/resonance/f'dse_{dse}'/case/perspective, options))
    return tasks


def preflight(tasks, command):
    for c, s, out, opts in tasks:
        if command in ('render', 'compare'):
            if not (out/'trajectory_config.json').exists() or not list((out/'frames').glob('trajectories_*.npz')):
                raise FileNotFoundError(f'Missing CTMQC archive: {out}')
        elif opts['resume']:
            z = load_npz(out/'restart.npz')
            if json.loads(str(z['identity'].item())) != identity(c, s, opts['every_fs']):
                raise ValueError(f'Resume mismatch: {out}')
            if int(np.ceil(opts['end_fs']/(c.dt*AU_FS))) < int(z['step']):
                raise ValueError(f'Resume end precedes checkpoint: {out}')
        elif out.exists() and any(out.iterdir()):
            raise FileExistsError(f'{out}: choose new root or --resume')


def run_task(task):
    c, s, out, opts = task
    print(f'RUN {out}: {s.algorithm} {s.perspective}, Ntraj={s.ntraj}, omega={c.omega}, g={c.g}', flush=True)
    status = simulate(c, s, out, opts['end_fs'], opts['every_fs'], opts['resume'])
    if opts['render']:
        try:
            render_task(task)
        except Exception as error:
            atomic_json(Path(out)/'render_failure.json', dict(error=repr(error),
                instruction='Dynamics/restart are retained. Use render after resolving the plotting error; no rerun needed.'))
            raise
    return status


def render_task(task):
    from .render import render_case
    _, _, out, opts = task
    return render_case(out, movies=opts['movies'], movie_stride=opts['movie_stride'],
                       png_every_fs=opts['png_every_fs'], dpi=opts['dpi'], t_max_fs=opts['t_max_fs'])


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    numbers = [args.every_fs, args.weak_g, args.strong_g, args.png_every_fs]
    if args.end_fs is not None:
        numbers.append(args.end_fs)
    if not np.all(np.isfinite(numbers)) or min(args.weak_g, args.strong_g) < 0 or args.every_fs <= 0 or args.png_every_fs <= 0:
        p.error('Finite parameters, nonnegative g and positive intervals required')
    if args.end_fs is not None and args.end_fs < 0:
        p.error('end-fs must be nonnegative')
    if args.movie_stride < 1 or args.dpi < 40:
        p.error('movie-stride>=1 and dpi>=40 required')
    if args.t_max_fs is not None and (args.command not in ('render', 'compare') or not np.isfinite(args.t_max_fs) or args.t_max_fs < 0):
        p.error('t-max-fs is nonnegative and render/compare-only')
    tasks = build_tasks(args)
    plan = []
    for c, s, out, opts in tasks:
        dof, states = (2, 2) if s.perspective == 'el' else (1, 2*c.nf)
        frames = int(np.ceil(opts['end_fs']/opts['every_fs']))+1
        # Upper estimate before ZIP compression, excluding PNG/MP4.
        frame_bytes = s.ntraj*8*(5*dof+1+3*states)
        plan.append(dict(path=str(out), perspective=s.perspective, model=asdict(c), settings=asdict(s),
                    end_fs=opts['end_fs'], approximate_frame_count=frames,
                    uncompressed_frames_MiB=frames*frame_bytes/1024**2))
    if args.command == 'plan':
        print(json.dumps(dict(cases=plan, count=len(plan), note='No TDSE propagation or writes; sizes exclude plots/restart.'), indent=2))
        return
    preflight(tasks, args.command)
    if args.command in ('run', 'render') and not args.no_render and not args.no_movies and shutil.which('ffmpeg') is None:
        p.error('ffmpeg missing; use --no-movies')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command in ('run', 'render'):
        action = run_task if args.command == 'run' else render_task
        if args.jobs == 1:
            for task in tasks:
                action(task)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(action, tasks))
    if args.command != 'run' or not args.no_render:
        from .render import comparison
        comparison(args.out, tdse_root=args.tdse_root, t_max_fs=args.t_max_fs)
    report = []
    for _, _, out, _ in tasks:
        status = json.loads((out/'status.json').read_text())
        report.append(dict(path=str(out), **status,
            bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())))
    atomic_json(args.out/'ctmqc_campaign_report.json', dict(cases=report,
        total_bytes=sum(r['bytes'] for r in report),
        note='Measured local/server hostname in provenance. No claim of author-code or sampling convergence.'))


if __name__ == '__main__':
    main()
