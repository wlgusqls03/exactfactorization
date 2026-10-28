"""Six-case CPU campaign; compact coefficients, restart, automatic rendering.

No existing results are overwritten by run without explicit --resume. Render
may overwrite its own derived images. No full R*q*field cache is necessary.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
from .model import AU_FS, Config, Propagator, initial, observables


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def atomic_npz(path, **values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def source_record():
    root = Path(__file__).parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    return dict(files=files, git_commit=commit, python=platform.python_version(),
                numpy=np.__version__, hostname=platform.node())


def identity(c, every_fs):
    # Plot/document changes must not prevent exact continuation.
    return dict(config=asdict(c), save_every_fs=every_fs,
                model_sha256=hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest())


def simulate(c, out, end_fs=40., every_fs=.5, resume=False):
    out = Path(out)
    c.grid()
    if not np.isfinite(end_fs) or end_fs < 0 or not np.isfinite(every_fs) or every_fs <= 0:
        raise ValueError('Need finite end_fs>=0 and every_fs>0')
    ident = identity(c, every_fs)
    started = time.perf_counter()
    p = Propagator(c)
    total = int(math.ceil(end_fs/(c.dt*AU_FS)))
    cadence = max(1, int(round(every_fs/(c.dt*AU_FS))))
    records = []
    old_wall = 0.
    if resume:
        saved = load_npz(out/'restart.npz')
        if json.loads(str(saved['identity'].item())) != ident:
            raise ValueError('Resume rejected: Hamiltonian/grid/source/save cadence mismatch')
        u, step = saved['u'], int(saved['step'])
        if total < step:
            raise ValueError('Requested end is earlier than restart')
        records = [r for r in json.loads((out/'observables.json').read_text()) if r['step'] <= step]
        old_wall = float(saved['wall_seconds'])
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'{out}: use new directory or --resume')
        out.mkdir(parents=True, exist_ok=True)
        u, step = initial(c), 0
        atomic_json(out/'config.json', asdict(c))
        atomic_json(out/'provenance.json', dict(source=source_record(),
                    paper='https://doi.org/10.1039/D6CP02020H',
                    preparation='Eq27: bare adiabatic S1 * nuclear Gaussian * |0>',
                    photon_representation='finite Fock; NOT paper 500-point q-grid quantum reference',
                    nuclear_boundary='periodic FFT, endpoint excluded (spacing 12/500)',
                    width_amplitude=c.sigma_r, width_density_sd=c.sigma_r/np.sqrt(2),
                    coupling_definition='omega*g*q*D; DSE=0.5*g^2*(D@D)',
                    photon_number='bare Pauli-Fierz oscillator number; NOT a gauge-transformed detector count'))
        atomic_npz(out/'references.npz', R=p.R, bo_energy=p.bo_energy,
                   bo_vectors=p.bo_vectors, pol_energy=p.pol_energy, pol_vectors=p.pol_vectors)
    (out/'coefficients').mkdir(exist_ok=True)
    first_step = step
    if shutil.disk_usage(out).free < 100*1024**2:
        raise OSError('Less than 100 MiB free; refusing calculation')

    def save(current):
        o = observables(u, p)
        if not all(np.all(np.isfinite(v)) for v in o.values()):
            raise FloatingPointError('Non-finite TDSE observable')
        row = dict(step=current, t_fs=current*c.dt*AU_FS, **o)
        if records and records[-1]['step'] == current:
            records[-1] = row
        else:
            records.append(row)
        elapsed = old_wall+time.perf_counter()-started
        atomic_npz(out/'coefficients'/f'wave_{current:08d}.npz', u=u, step=current, t_fs=row['t_fs'])
        atomic_json(out/'observables.json', records)
        atomic_npz(out/'restart.npz', u=u, step=current, identity=json.dumps(ident, sort_keys=True), wall_seconds=elapsed)
        atomic_json(out/'status.json', dict(status='RUNNING', step=current, total_steps=total, t_fs=row['t_fs']))

    if not resume:
        save(step)
    try:
        while step < total:
            u = p.step(u)
            step += 1
            if step % cadence == 0 or step == total:
                save(step)
    except BaseException as error:
        atomic_json(out/'failure.json', dict(error=repr(error), attempted_step=step,
                    instruction='Resume from last atomic restart, not attempted_step.'))
        raise
    elapsed = time.perf_counter()-started
    energy = np.array([r['energy'] for r in records])
    norm = np.array([r['norm'] for r in records])
    status = dict(status='COMPLETE_NOT_CERTIFIED', step=step, total_steps=total,
                  t_fs=step*c.dt*AU_FS, wall_seconds=old_wall+elapsed,
                  latest_wall_seconds=elapsed, latest_steps=step-first_step,
                  max_norm_error=float(np.max(abs(norm-1))),
                  max_energy_drift_Ha=float(np.max(abs(energy-energy[0]))),
                  max_edge_probability=max(r['edge_probability'] for r in records),
                  max_top_fock_population=max(r['top_fock_population'] for r in records),
                  max_omitted_coupling_norm_Ha=max(r['omitted_coupling_norm_Ha'] for r in records),
                  warning='Finite-Fock model solved; conservation does not certify Fock/grid/time convergence.')
    atomic_json(out/'status.json', status)
    return status


def run_case(task):
    c, out, options = task
    print(f'RUN {out}: g={c.g:g}, DSE={c.dse}, nf={c.nf}', flush=True)
    status = simulate(c, out, options['end_fs'], options['every_fs'], options['resume'])
    if options['render']:
        from .render import render_case
        render_case(out, movies=options['movies'], movie_stride=options['movie_stride'],
                    nq=options['nq'], dpi=options['dpi'])
    print(f'DONE {out}: TDSE {status["latest_wall_seconds"]:.2f} s', flush=True)
    return status


def cost_report(out, elapsed):
    rows = []
    for status_file in sorted(Path(out).glob('dse_*/*/status.json')):
        folder = status_file.parent
        s = json.loads(status_file.read_text())
        render = json.loads((folder/'plots'/'render_report.json').read_text()) if (folder/'plots'/'render_report.json').exists() else {}
        files = list(folder.rglob('*'))
        rows.append(dict(case=str(folder.relative_to(out)), **s,
                         render_seconds=render.get('wall_seconds'),
                         bytes=sum(p.stat().st_size for p in files if p.is_file())))
    report = dict(campaign_wall_seconds=elapsed, cases=rows,
                  total_bytes=sum(r['bytes'] for r in rows),
                  note='Measured on this host; no GPU/server performance claim. Render cost depends on cadence/DPI/families.')
    atomic_json(Path(out)/'cost_report.json', report)
    return report


def render_task(task):
    from .render import render_case
    return render_case(*task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'render'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='Optional Config JSON; not a real-grid PEN config')
    parser.add_argument('--case', choices=['all', 'zero', 'weak', 'strong'], default='all')
    parser.add_argument('--dse', choices=['both', 'on', 'off'], default='both')
    parser.add_argument('--preset', choices=['nonadiabatic', 'rabi'], default='nonadiabatic')
    parser.add_argument('--resonance', choices=['on', 'off'], default='on')
    parser.add_argument('--weak-g', type=float, default=.001)
    parser.add_argument('--strong-g', type=float, default=.01)
    parser.add_argument('--end-fs', type=float, default=40.)
    parser.add_argument('--every-fs', type=float, default=.5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--jobs', type=int, choices=[1, 2, 3, 6], default=1)
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--no-movies', action='store_true')
    parser.add_argument('--movie-stride', type=int, default=1)
    parser.add_argument('--nq-display', type=int, default=160, help='Reconstruction only; use 500 for paper q sampling')
    parser.add_argument('--dpi', type=int, default=85, help='Movie DPI; snapshots use at least 140')
    parser.add_argument('--t-max-fs', type=float, help='Render only; cut existing trajectory without TDSE')
    args = parser.parse_args()
    if args.command == 'run' and args.t_max_fs is not None:
        parser.error('--t-max-fs is render-only; use --end-fs for propagation')
    if args.movie_stride < 1 or args.nq_display < 32 or args.dpi < 40:
        parser.error('movie-stride>=1, nq-display>=32, dpi>=40 required')
    if not args.no_render and not args.no_movies and shutil.which('ffmpeg') is None:
        parser.error('ffmpeg required for automatic movies; use --no-movies for PNG only')
    cases = ['zero', 'weak', 'strong'] if args.case == 'all' else [args.case]
    dses = ['off', 'on'] if args.dse == 'both' else [args.dse]
    start = time.perf_counter()
    if args.command == 'render':
        from .render import render_case, comparison
        tasks = [(args.out/f'dse_{dse}'/case, not args.no_movies, args.movie_stride,
                  args.nq_display, args.dpi, args.t_max_fs) for dse in dses for case in cases]
        if args.jobs == 1:
            for task in tasks:
                render_task(task)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(render_task, tasks))
        comparison(args.out, args.t_max_fs)
    else:
        if args.config:
            c = Config(**json.loads(args.config.read_text()))
        else:
            c = Config(r1=2., delta=.17) if args.preset == 'rabi' else Config()
            if args.resonance == 'off':
                c = replace(c, omega=.12 if args.preset == 'rabi' else .097)
        c.grid()
        args.out.mkdir(parents=True, exist_ok=True)
        options = dict(end_fs=args.end_fs, every_fs=args.every_fs, resume=args.resume,
                       render=not args.no_render, movies=not args.no_movies,
                       movie_stride=args.movie_stride, nq=args.nq_display, dpi=args.dpi)
        gs = dict(zero=0., weak=args.weak_g, strong=args.strong_g)
        tasks = [(replace(c, g=gs[case], dse=dse == 'on'), args.out/f'dse_{dse}'/case, options)
                 for dse in dses for case in cases]
        if args.jobs == 1:
            for task in tasks:
                run_case(task)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(run_case, tasks))
        if not args.no_render:
            from .render import comparison
            comparison(args.out)
    report = cost_report(args.out, time.perf_counter()-start)
    print(json.dumps(dict(wall_seconds=report['campaign_wall_seconds'], total_MiB=report['total_bytes']/1024**2), indent=2))


if __name__ == '__main__':
    main()
