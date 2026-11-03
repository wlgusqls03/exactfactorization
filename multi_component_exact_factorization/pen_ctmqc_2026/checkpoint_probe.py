"""Short, read-only timestep experiments from an existing CTMQC restart.

Each trial starts from the SAME complete saved ensemble, including force
integrals and eigenvector anchors. This is NOT resume or a restart converter:
no production files, frames, or new restart are written. The original norm
guard and dynamics are unchanged. A successful short probe is not proof of
long-time or timestep convergence; compare the common-endpoint diagnostics.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from ..pen_perspectives_2026.model import AU_FS, Config
from ..pen_perspectives_2026.workflow import atomic_json, load_npz
from .core import Engine, Settings, observables
from .workflow import identity


STATE_KEYS = ('X', 'P', 'coeff', 'integrated_forces', 'anchor')
OBSERVABLE_KEYS = ('energy', 'photon_number', 'kinetic_R', 'R_mean', 'R_sd',
                   'bo_population', 'pol_population', 'cbo_population',
                   'product_R_ge_4', 'signed_flux_R4', 'edge_probability')


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_checkpoint(path):
    """Refuse incompatible source, malformed or concurrently changing inputs."""
    path = Path(path)
    files = ['config.json', 'trajectory_config.json', 'restart.npz']
    files += [n for n in ('status.json', 'failure.json') if (path/n).exists()]
    hashes = {n: _hash(path/n) for n in files}
    c = Config(**json.loads((path/'config.json').read_text()))
    s = Settings(**json.loads((path/'trajectory_config.json').read_text()))
    c.grid()
    s.validate()
    saved = load_npz(path/'restart.npz')
    ident = json.loads(str(saved['identity'].item()))
    cadence = float(ident['every_fs'])
    if not np.isfinite(cadence) or cadence <= 0 or ident != identity(c, s, cadence):
        raise ValueError('Checkpoint physics/method/dt/source identity mismatch; no trial is allowed')
    step = int(saved['step'])
    if step < 0 or float(saved['step']) != step:
        raise ValueError('Invalid checkpoint step')
    state = {k: saved[k] for k in STATE_KEYS}
    dof, states = (2, 2) if s.perspective == 'el' else (1, 2*c.nf)
    shapes = dict(X=(s.ntraj, dof), P=(s.ntraj, dof), coeff=(s.ntraj, states),
                  integrated_forces=(s.ntraj, states, dof), anchor=(s.ntraj, states, states))
    for key, value in state.items():
        if value.shape != shapes[key] or not np.isfinite(value).all():
            raise ValueError(f'Invalid saved {key}: expected finite {shapes[key]}')
    if np.max(abs(np.sum(abs(state['coeff'])**2, axis=1)-1)) > 1e-8:
        raise ValueError('Checkpoint coefficients are not normalized')
    overlap = state['anchor'].conj().swapaxes(1, 2) @ state['anchor']
    if np.max(abs(overlap-np.eye(states))) > 1e-8:
        raise ValueError('Checkpoint anchors are not orthonormal')
    metadata = dict(path=str(path.resolve()), checkpoint_step=step,
        checkpoint_t_fs=step*c.dt*AU_FS, identity=ident, input_hashes=hashes,
        settings=asdict(s), original_dt_au=c.dt,
        status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {},
        failure=json.loads((path/'failure.json').read_text()) if (path/'failure.json').exists() else None)
    verify_inputs_unchanged(path, hashes)
    return c, s, state, metadata


def verify_inputs_unchanged(path, hashes):
    for name, digest in hashes.items():
        if _hash(Path(path)/name) != digest:
            raise RuntimeError(f'Input changed during probe: {name}. Stop the active writer and retry.')


def _measure(engine, state):
    row = observables(engine, state)
    _, detail = engine.evaluate(state, state['anchor'])
    row['max_abs_CT_coefficient_rate_au'] = float(np.max(abs(detail['rates'])))
    row['max_abs_CT_force_au'] = float(np.max(abs(detail['ct_force'])))
    if not all(np.isfinite(np.asarray(v)).all() for v in row.values()):
        raise FloatingPointError('Nonfinite probe observable')
    return row


def _trial(c, s, initial, factor, base_steps, t0_fs):
    engine = Engine(replace(c, dt=c.dt/factor), s)
    state = {k: v.copy() for k, v in initial.items()}
    count, max_norm, guards, error = 0, 0., np.zeros(3, dtype=np.int64), None
    failed_step_guards = None
    start = time.perf_counter()
    target_steps = base_steps*factor
    try:
        for _ in range(target_steps):
            # Engine.step checks BEFORE its existing tiny-error normalization.
            # An excessive error remains a failure; it is never hidden here.
            state, norm_error = engine.step(state)
            count += 1
            max_norm = max(max_norm, norm_error)
            guards += engine.last_step_guard_counts
    except FloatingPointError as exc:
        error = repr(exc)
        failed_step_guards = getattr(engine, 'last_step_guard_counts', None)
        if failed_step_guards is not None:
            failed_step_guards = failed_step_guards.tolist()
    # The last accepted state is distinct from the rejected trial state. Raw
    # rejected norm error is quoted in error, NOT included in accepted maximum.
    final = _measure(engine, state)
    result = dict(refinement_factor=factor, dt_au=c.dt/factor,
        status='FAILED_GUARD' if error else 'COMPLETE_SHORT_PROBE', error=error,
        accepted_steps=count, target_steps=target_steps,
        last_accepted_t_fs=t0_fs+count*(c.dt/factor)*AU_FS,
        rejected_step_end_t_fs=t0_fs+(count+1)*(c.dt/factor)*AU_FS if error else None,
        max_accepted_raw_norm_step_error=max_norm,
        guard_counts_accepted=dict(zip(('attempted', 'disabled', 'distant'), guards.tolist())),
        failed_step_guard_counts=failed_step_guards,
        last_accepted_observables=final, wall_seconds=time.perf_counter()-start)
    return result, state if error is None else None


def endpoint_difference(coarse, fine, coarse_obs, fine_obs):
    """Same trajectory sample and same time; phase-insensitive state comparison."""
    norms = np.sum(abs(coarse['coeff'])**2, axis=1)*np.sum(abs(fine['coeff'])**2, axis=1)
    overlap = np.sum(coarse['coeff'].conj()*fine['coeff'], axis=1)
    # Remove negative floating-roundoff (~1e-15) in a diagnostic only. Dynamics
    # and coefficient norms are never clipped or modified by this comparison.
    infidelity = np.maximum(0., 1-abs(overlap)**2/norms)
    return dict(
        R_rms_difference_a0=float(np.sqrt(np.mean((coarse['X'][:, 0]-fine['X'][:, 0])**2))),
        X_max_abs_difference_by_coordinate=np.max(abs(coarse['X']-fine['X']), axis=0).tolist(),
        P_max_abs_difference_by_coordinate=np.max(abs(coarse['P']-fine['P']), axis=0).tolist(),
        coefficient_projector_infidelity_mean=float(np.mean(infidelity)),
        coefficient_projector_infidelity_max=float(np.max(infidelity)),
        observables_max_abs_difference={k: float(np.max(abs(np.asarray(coarse_obs[k])-fine_obs[k])))
            for k in OBSERVABLE_KEYS if k in coarse_obs and k in fine_obs})


def probe_case(path, forward_fs=.2, factors=(1, 2, 4)):
    if not np.isfinite(forward_fs) or not 0 < forward_fs <= 1.:
        raise ValueError('Short probe requires 0 < forward_fs <= 1; not a production continuation')
    if (not factors or len(set(factors)) != len(factors)
            or any(not np.isfinite(f) or int(f) != f or not 1 <= f <= 64 for f in factors)):
        raise ValueError('Need distinct integer refinement factors in [1,64]')
    factors = sorted(int(f) for f in factors)
    c, s, state, meta = read_checkpoint(path)
    base_steps = max(1, int(math.ceil(forward_fs/(c.dt*AU_FS))))
    if base_steps*sum(factors) > 100000:
        raise ValueError('Probe would exceed 100000 steps; reduce horizon/refinement')
    before = _measure(Engine(c, s), state)
    trials, completed = [], []
    for factor in factors:
        print(f'PROBE {path}: dt={c.dt/factor:g} au, '
              f'{meta["checkpoint_t_fs"]:.6f} -> '
              f'{meta["checkpoint_t_fs"]+base_steps*c.dt*AU_FS:.6f} fs, '
              f'norm guard={s.max_norm_step_error:g}', flush=True)
        trial, endpoint = _trial(c, s, state, factor, base_steps, meta['checkpoint_t_fs'])
        trial['energy_change_from_checkpoint_Ha'] = trial['last_accepted_observables']['energy']-before['energy']
        trials.append(trial)
        if endpoint is not None:
            completed.append((trial, endpoint))
        print(f'  {trial["status"]}: last accepted t={trial["last_accepted_t_fs"]:.6f} fs'
              + (f'; {trial["error"]}' if trial['error'] else ''), flush=True)
    differences = []
    # Only completed trials at the common endpoint can be compared. Never
    # compare a failed coarse trial's earlier state to the fine endpoint.
    for (a, sa), (b, sb) in zip(completed[:-1], completed[1:]):
        differences.append(dict(coarse_dt_au=a['dt_au'], fine_dt_au=b['dt_au'],
            **endpoint_difference(sa, sb, a['last_accepted_observables'], b['last_accepted_observables'])))
    verify_inputs_unchanged(path, meta['input_hashes'])
    return dict(**meta, requested_forward_fs=forward_fs,
        actual_forward_fs=base_steps*c.dt*AU_FS, initial_observables=before,
        trials=trials, common_endpoint_differences=differences,
        no_input_writes=True, no_restart_written=True,
        probe_source_sha256=_hash(__file__),
        verdict='Diagnostic only. Guard pass is not timestep/long-time convergence. '
                'Refined trials inherit the original coarse trajectory history; '
                'they do not validate earlier dynamics or authorize production resume with another dt.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--root', type=Path, help='Campaign root; selects FAILED cases only')
    inputs.add_argument('--case-dir', type=Path, help='One explicit case, including a completed test case')
    p.add_argument('--out', type=Path, required=True, help='NEW JSON outside the input root')
    p.add_argument('--forward-fs', type=float, default=.2, help='Short horizon <=1 fs, rounded up to original steps')
    p.add_argument('--refine', type=int, nargs='+', default=[1, 2, 4], help='Divide ORIGINAL dt by each integer')
    args = p.parse_args(argv)
    root = (args.root or args.case_dir).resolve()
    out = args.out.resolve()
    if out == root or root in out.parents:
        p.error('Output JSON must be outside the input archive root')
    if out.exists():
        p.error('Choose a new output JSON; previous probes are not overwritten')
    if args.case_dir:
        paths = [args.case_dir]
    else:
        paths = [q.parent for q in sorted(root.glob('*/*/dse_*/*/*/status.json'))
                 if json.loads(q.read_text()).get('status') == 'FAILED']
    if not paths:
        p.error('No FAILED cases found; --case-dir allows an explicitly selected case')
    report = dict(cases=[], no_input_writes=True, no_restart_written=True,
                  purpose='Short checkpoint timestep probe, not production or physical certification')
    out.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for path in paths:
        try:
            item = probe_case(path, args.forward_fs, args.refine)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, FloatingPointError) as exc:
            item = dict(path=str(path.resolve()), setup_or_probe_error=repr(exc))
            errors.append(item)
            print(f'PROBE ERROR {path}: {exc!r}', flush=True)
        report['cases'].append(item)
        # A failed dt trial is diagnostic data, not a reason to lose the other
        # trials/cases. Incremental JSON also survives a later interruption.
        atomic_json(out, report)
    print(f'Probe JSON: {out}. No production restart or frames were written.', flush=True)
    if errors:
        raise SystemExit('Some checkpoints could not be probed; inspect setup_or_probe_error in JSON.')


if __name__ == '__main__':
    main()
