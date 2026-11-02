"""Read-only CTMQC archive/restart audit; never advances a trajectory.

This checks saved numbers and files, NOT long-time, sampling or paper-method
convergence. Energy drift and the pair-intercept guard remain diagnostics,
not an automatic certificate of physical accuracy.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from ..pen_perspectives_2026.model import AU_FS
from ..pen_perspectives_2026.workflow import atomic_json, load_npz
from .compare_data import read_archive
from .core import Engine, observables
from .workflow import identity


def audit_case(path, target_fs=40.):
    path = Path(path)
    view = json.loads((path/'trajectory_config.json').read_text())['perspective']
    a = read_archive(path, view)
    c, s, rows = a.config, a.settings, a.rows
    errors, warnings = [], []
    z = load_npz(path/'restart.npz')
    saved_identity = json.loads(str(z['identity'].item()))
    cadence = float(saved_identity['every_fs'])
    compatible = saved_identity == identity(c, s, cadence)
    if not compatible:
        errors.append('Checkpoint identity differs from current physics/method/source; resume rejected.')
    state = {k: z[k] for k in ('X', 'P', 'coeff', 'integrated_forces', 'anchor')}
    if not all(np.isfinite(v).all() for v in state.values()):
        errors.append('Nonfinite restart state')
    dof, states = (2, 2) if view == 'el' else (1, 2*c.nf)
    expected = dict(X=(s.ntraj, dof), P=(s.ntraj, dof), coeff=(s.ntraj, states),
                    integrated_forces=(s.ntraj, states, dof), anchor=(s.ntraj, states, states))
    for key, shape in expected.items():
        if state[key].shape != shape:
            errors.append(f'Restart shape mismatch: {key}: {state[key].shape} != {shape}')
    if not all(np.isfinite(np.asarray(v)).all() for r in rows for v in r.values()
               if np.asarray(v).dtype.kind in 'fciu'):
        errors.append('Nonfinite observable')
    step = int(z['step'])
    if step != int(rows[-1]['step']) or not np.isclose(step*c.dt*AU_FS, rows[-1]['t_fs'], atol=1e-12):
        errors.append('Restart / last observable step or time mismatch')
    if a.status.get('step') != step or not a.status.get('status', '').startswith('COMPLETE'):
        warnings.append('Status not COMPLETE at the last restart; inspect failure/log before continuing.')
    frame_steps, norm_error, first_error, last_error = [], 0., None, None
    initial = Engine(c, s).initial() if compatible else None
    for p in a.paths:
        f = load_npz(p)
        fs = int(f['step']); frame_steps.append(fs)
        if not all(np.isfinite(v).all() for v in f.values() if v.dtype.kind in 'fciu'):
            errors.append(f'Nonfinite saved frame: {p.name}')
        for key in ('X', 'P', 'coeff'):
            if f[key].shape != expected[key]:
                errors.append(f'Frame shape mismatch: {p.name}/{key}')
        norm_error = max(norm_error, float(np.max(abs(np.sum(abs(f['coeff'])**2, axis=1)-1))))
        if not np.isclose(float(f['t_fs']), fs*c.dt*AU_FS, rtol=0, atol=1e-12):
            errors.append(f'Step/time mismatch: {p.name}')
        if fs == 0 and initial is not None:
            first_error = max(float(np.max(abs(f[k]-initial[k]))) for k in ('X', 'P', 'coeff'))
        if fs == step:
            last_error = max(float(np.max(abs(f[k]-state[k]))) for k in ('X', 'P', 'coeff'))
    if frame_steps != [int(r['step']) for r in rows]:
        errors.append('Frame inventory and observable steps differ')
    if first_error is None:
        warnings.append('Cannot independently verify initial frame/seed with current source')
    elif first_error > 1e-12:
        errors.append('Initial frame differs from recorded preparation/seed')
    if last_error is None or last_error > 1e-12:
        errors.append('Last frame and restart do not match')
    if norm_error > 1e-10:
        errors.append(f'Saved coefficient norm error {norm_error:g}')
    recomputed = {}
    if compatible and not errors:
        # evaluate/observables only: no Engine.step, no propagation.
        actual = observables(Engine(c, s), state)
        for key in ('energy', 'photon_number', 'q2', 'p2', 'bo_population',
                    'pol_population', 'cbo_population', 'kinetic_R', 'ct_population_rate'):
            if key in actual:
                recomputed[key] = float(np.max(abs(np.asarray(actual[key])-rows[-1][key])))
        if max(recomputed.values(), default=0.) > 1e-10:
            errors.append('Restart-derived observables disagree with last saved record')
    guards = np.array(z['guard_counts'], dtype=int)
    guard_fraction = float(guards[1]/guards[0]) if guards[0] else 0.
    if guards[1]:
        warnings.append('Pair-intercept guard was used; check guard/bandwidth/dt/seed sensitivity at longer times.')
    number_identity = max(abs(c.omega*r['q2']/2+r['p2']/(2*c.omega)-.5-r['photon_number']) for r in rows)
    population_sum_error = max(abs(sum(r[k])-1) for r in rows
                               for k in ('bo_population', 'pol_population', 'cbo_population') if k in r)
    energy = np.array([r['energy'] for r in rows])
    wall = float(a.status.get('wall_seconds', z['wall_seconds']))
    remaining_steps = max(0, int(math.ceil(target_fs/(AU_FS*c.dt)))-step)
    step_cadence = max(1, int(round(cadence/(AU_FS*c.dt))))
    remaining_frames = sum(i % step_cadence == 0 or i == step+remaining_steps
                           for i in range(step+1, step+remaining_steps+1))
    frame_bytes = float(np.mean([p.stat().st_size for p in a.paths])) if a.paths else 0.
    frame_raw_bytes = 0
    if a.paths:
        frame_raw_bytes = sum(v.nbytes for v in load_npz(a.paths[-1]).values())
    return dict(path=str(path.resolve()), file_checks_passed=not errors,
        resume_compatible=compatible, errors=errors, warnings=warnings,
        perspective=view, ntraj=s.ntraj, dt_au=c.dt, every_fs=cadence,
        omega=c.omega, g=c.g, dse=c.dse, initial=s.initial, step=step,
        t_fs=float(rows[-1]['t_fs']), frames=len(a.paths), observations=len(rows),
        max_saved_norm_error=norm_error, max_raw_norm_step_error=float(z['max_norm_step_error']),
        max_energy_drift_Ha=float(np.max(abs(energy-energy[0]))),
        max_population_sum_error=float(population_sum_error),
        photon_number_identity_max_abs=float(number_identity),
        first_frame_reproduction_max_abs=first_error, restart_frame_max_abs=last_error,
        last_observable_recomputation_max_abs=recomputed,
        initial_photon_number=rows[0]['photon_number'], initial_photon_number_sampling_se=rows[0]['photon_number_se'],
        guard_attempted=int(guards[0]), guard_disabled=int(guards[1]), guard_fraction=guard_fraction,
        max_CT_only_population_rate=max(float(np.max(abs(np.asarray(r['ct_population_rate'])))) for r in rows),
        max_outside_box_probability=max(r['edge_probability'] for r in rows),
        current_total_bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file()),
        measured_wall_seconds=wall,
        extrapolation=dict(target_fs=target_fs, additional_steps=remaining_steps,
            additional_frames=remaining_frames, seconds_at_pilot_rate=remaining_steps*wall/step if step else None,
            frames_bytes_at_pilot_compression=remaining_frames*frame_bytes,
            frames_uncompressed_bytes=remaining_frames*frame_raw_bytes,
            note='Timing extrapolation, not a benchmark of later dynamics. Excludes rendering/comparisons; '
                 'compression and guard cost can change. Keep space for restart, JSON, PNG and MP4.'),
        verdict='Saved-file integrity / short-run checks only; NOT long-time or sampling convergence certification.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='Audit JSON outside the input root')
    p.add_argument('--target-fs', type=float, default=40.)
    args = p.parse_args(argv)
    if not np.isfinite(args.target_fs) or args.target_fs < 0:
        p.error('Finite target-fs>=0 required')
    if args.out.resolve() == args.root.resolve() or args.root.resolve() in args.out.resolve().parents:
        p.error('Keep the audit JSON outside the input archive root')
    paths = sorted(q.parent for q in args.root.glob('*/*/dse_*/*/*/trajectory_config.json'))
    if not paths:
        p.error('No CTMQC cases found; root must contain model/resonance/dse/case/perspective/')
    cases = []
    for path in paths:
        try:
            case = audit_case(path, args.target_fs)
        except (OSError, ValueError, KeyError, TypeError) as error:
            case = dict(path=str(path), file_checks_passed=False, errors=[repr(error)])
        cases.append(case)
        print(path.relative_to(args.root), 'PASS' if case['file_checks_passed'] else 'FAIL',
              f't={case.get("t_fs")} fs', case['errors'], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, dict(cases=cases, no_propagation=True, no_input_writes=True,
                              all_file_checks_passed=all(c['file_checks_passed'] for c in cases)))
    if not all(c['file_checks_passed'] for c in cases):
        raise SystemExit('Archive checks failed; inspect audit JSON before resume.')


if __name__ == '__main__':
    main()
