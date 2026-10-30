"""Explicit Table-2 on/off campaigns; legacy bare-gap workflow stays unchanged.

Output: ROOT/{nonadiabatic,rabi}/{on,off}/dse_{off,on}/{zero,weak,strong}.
One process pool for the entire selection: --jobs never multiplies per model.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .model import AU_FS, Config
from .resonance import paper_config, scenario_report
from .workflow import atomic_json, identity, load_npz, run_case, render_task, cost_report


def build_tasks(args):
    """Pure planning; all physics is explicit in each saved Config."""
    models = ['nonadiabatic', 'rabi'] if args.model == 'all' else [args.model]
    resonances = ['on', 'off'] if args.resonance == 'both' else [args.resonance]
    cases = ['zero', 'weak', 'strong'] if args.case == 'all' else [args.case]
    dses = ['off', 'on'] if args.dse == 'both' else [args.dse]
    gs = dict(zero=0., weak=args.weak_g, strong=args.strong_g)
    tasks = []
    for model in models:
        for resonance in resonances:
            for dse in dses:
                for case in cases:
                    c = paper_config(model, resonance, nr=args.nr, dt=args.dt, nf=args.nf,
                                     g=gs[case], dse=dse == 'on')
                    options = dict(end_fs=args.end_fs, every_fs=args.every_fs, resume=args.resume,
                                   render=not args.no_render, movies=not args.no_movies,
                                   movie_stride=args.movie_stride, nq=args.nq_display, dpi=args.dpi,
                                   packet_scale=args.packet_scale, resonance='paper-'+resonance,
                                   png_every_fs=args.png_every_fs,
                                   scenario=scenario_report(c, model, resonance))
                    tasks.append((c, args.out/model/resonance/f'dse_{dse}'/case, options))
    return tasks


def preflight(tasks, command):
    """Fail BEFORE any run if one target would overwrite/mis-resume another."""
    for c, out, options in tasks:
        if command == 'render':
            if not (out/'config.json').exists() or not list((out/'coefficients').glob('wave_*.npz')):
                raise FileNotFoundError(f'{out}: config or coefficient archive missing')
            # Rendering always reads the SAVED config, not new CLI defaults.
        elif options['resume']:
            saved = load_npz(out/'restart.npz')
            if json.loads(str(saved['identity'].item())) != identity(c, options['every_fs']):
                raise ValueError(f'{out}: resume Hamiltonian/grid/cadence mismatch')
            if int(np.ceil(options['end_fs']/(c.dt*AU_FS))) < int(saved['step']):
                raise ValueError(f'{out}: requested end precedes restart')
        elif out.exists() and any(out.iterdir()):
            raise FileExistsError(f'{out}: choose a new root or explicitly --resume')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'run', 'render'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', choices=['nonadiabatic', 'rabi', 'all'], default='nonadiabatic')
    p.add_argument('--resonance', choices=['on', 'off', 'both'], default='on')
    p.add_argument('--case', choices=['zero', 'weak', 'strong', 'all'], default='strong')
    p.add_argument('--dse', choices=['off', 'on', 'both'], default='off')
    p.add_argument('--weak-g', type=float, default=.001)
    p.add_argument('--strong-g', type=float, default=.01)
    p.add_argument('--nr', type=int, default=500, help='Convergence override; paper: 500')
    p.add_argument('--nf', type=int, default=2, help='Finite-Fock propagation size, NOT display grid')
    p.add_argument('--dt', type=float, default=.1, help='Atomic units; paper: 0.1')
    p.add_argument('--end-fs', type=float, default=40.)
    p.add_argument('--every-fs', type=float, default=.1, help='Dense traces for fast off-resonance oscillations')
    p.add_argument('--jobs', type=int, choices=[1, 2, 3, 4, 6], default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--no-movies', action='store_true')
    p.add_argument('--movie-stride', type=int, default=1, help='Actual saved frames only; no interpolation')
    p.add_argument('--png-every-fs', type=float, help='PNG interval in physical fs; nearest saved frames plus endpoints. Default: first/middle/last')
    p.add_argument('--nq-display', type=int, default=160)
    p.add_argument('--dpi', type=int, default=85)
    p.add_argument('--t-max-fs', type=float)
    p.add_argument('--packet-scale', choices=['sqrt', 'linear'], default='sqrt')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (not np.all(np.isfinite([args.end_fs, args.every_fs, args.weak_g, args.strong_g]))
            or args.end_fs < 0 or args.every_fs <= 0 or min(args.weak_g, args.strong_g) < 0):
        p.error('Need finite end>=0, cadence>0, nonnegative g')
    if args.movie_stride < 1 or args.nq_display < 32 or args.dpi < 40:
        p.error('movie-stride>=1, nq-display>=32, dpi>=40 required')
    if args.png_every_fs is not None and (not np.isfinite(args.png_every_fs) or args.png_every_fs <= 0):
        p.error('--png-every-fs must be finite and positive')
    if args.t_max_fs is not None and (args.command != 'render' or not np.isfinite(args.t_max_fs) or args.t_max_fs < 0):
        p.error('--t-max-fs must be finite, nonnegative and render-only')
    if args.command != 'plan' and not args.no_render and not args.no_movies and shutil.which('ffmpeg') is None:
        p.error('ffmpeg required; use --no-movies for PNG only')
    tasks = build_tasks(args)
    if args.command == 'render':
        preflight(tasks, 'render')
        # Logs and plans, as well as the renderer, describe the actual archive.
        actual = []
        for _, out, options in tasks:
            c = Config(**json.loads((out/'config.json').read_text()))
            actual.append((c, out, dict(options, scenario=scenario_report(c))))
        tasks = actual
    plan = dict(command=args.command, jobs=args.jobs, cases=[dict(
        path=str(out), config=asdict(c), scenario=o['scenario'], end_fs=o['end_fs'],
        save_every_fs=o['every_fs']) for c, out, o in tasks])
    for row in plan['cases']:
        print(f'{args.command.upper()} {row["path"]}: omega={row["config"]["omega"]:g}, '
              f'g={row["config"]["g"]:g}, DSE={row["config"]["dse"]}', flush=True)
    if args.command == 'plan':
        print(json.dumps(plan, indent=2))
        return plan
    preflight(tasks, args.command)
    args.out.mkdir(parents=True, exist_ok=True)
    # Render plans must not replace original run provenance with default configs.
    if args.command == 'run':
        atomic_json(args.out/'paper_campaign.json', plan)
    start = time.perf_counter()
    worker = run_case if args.command == 'run' else render_task
    work = tasks if args.command == 'run' else [(out, not args.no_movies, args.movie_stride,
        args.nq_display, args.dpi, args.t_max_fs, args.packet_scale, args.png_every_fs) for _, out, _ in tasks]
    if args.jobs == 1:
        for task in work:
            worker(task)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(worker, work))
    roots = sorted({out.parent.parent for _, out, _ in tasks})
    if args.command == 'render' or not args.no_render:
        from .render import comparison
        from .paper_plots import resonance_comparison
        for root in roots:
            comparison(root, args.t_max_fs)
        for model_root in sorted({root.parent for root in roots}):
            resonance_comparison(model_root, args.t_max_fs)
    costs = [cost_report(root, time.perf_counter()-start) for root in roots]
    cases = [dict(row, case=str(root.relative_to(args.out)/row['case']))
             for root, cost in zip(roots, costs) for row in cost['cases']]
    report = dict(wall_seconds=time.perf_counter()-start, cases=cases,
                  selected_scenario_bytes=sum(r['total_bytes'] for r in costs),
                  note='Scenario costs include cases already present in those folders; measured on this host.')
    atomic_json(args.out/'paper_cost_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))
    return report


if __name__ == '__main__':
    main()
