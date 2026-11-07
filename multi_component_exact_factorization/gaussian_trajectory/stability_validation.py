"""Staged, fail-closed pilot validation using the actual MTE/CTMQC/FSSH drivers.

Never substitutes native dynamics for Gaussian compression. Completion is not
convergence: dt, grid, support, K and ensemble comparisons remain necessary.
An existing directory is refused; checkpoints remain owned by each native CLI.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from .run import AU_FS, RunConfig, atomic_json, refresh_timeseries


def validate_stages(config, out, targets=(1., 10., 40.), *, movies=False, render=True):
    """Run progressively longer targets; stop on the first rejected state.

    The configuration is explicit. Only end_fs changes between stages. Failure
    rendering uses saved accepted frames, never the rejected candidate. Both
    numerical failure and rendering failure remain visible in the report.
    """
    from .channel_run import ChannelConfig, run as channel_run
    from .run import run as mte_run
    config.validate()
    targets = np.asarray(targets, dtype=float)
    if (targets.ndim != 1 or not len(targets) or not np.all(np.isfinite(targets))
            or np.any(targets <= 0) or np.any(np.diff(targets) <= 0)):
        raise ValueError('Targets must be finite, positive and strictly increasing')
    root = Path(out)
    if root.exists():
        raise FileExistsError('Use a new validation directory; existing evidence is preserved')
    root.mkdir(parents=True)
    case = root/'run'
    driver = channel_run if isinstance(config, ChannelConfig) else mte_run
    report = dict(method=config.method, compression=config.compression,
        integrator=config.integrator, q_operator=config.model.get('q_operator', 'fft'),
        target_times_fs=targets.tolist(), stages=[], numerical_convergence_certified=False,
        interpretation='Guarded completion is not accuracy; no masks/thresholds changed by this driver')
    started = time.monotonic()
    for target in targets:
        stage_start = time.monotonic()
        record = dict(target_fs=float(target))
        print('VALIDATION target=%.6f fs method=%s compression=%s' %
              (target, config.method, config.compression), flush=True)
        try:
            driver(replace(config, end_fs=float(target)), case, resume=case.exists(), render=False)
            record['status'] = 'COMPLETE_NOT_CERTIFIED'
        except Exception as exc:
            record.update(status='FAILED', error=repr(exc))
        record['wall_seconds'] = time.monotonic()-stage_start
        if (case/'status.json').exists():
            record['driver_status'] = json.loads((case/'status.json').read_text())
        if (case/'checkpoint.npz').exists():
            with np.load(case/'checkpoint.npz', allow_pickle=False) as checkpoint:
                record['last_saved_fs'] = int(checkpoint['step'])*config.dt_au*AU_FS
        if record['status'] == 'FAILED' and (case/'failure.json').exists():
            record['failure'] = json.loads((case/'failure.json').read_text())
        report['stages'].append(record)
        atomic_json(root/'validation_report.json', report)
        if record['status'] == 'FAILED':
            break
    report['all_targets_completed'] = (len(report['stages']) == len(targets)
                                       and report['stages'][-1]['status'] != 'FAILED')
    # Saved-frame summaries remain useful for failed pilots. No extrapolation
    # to the requested end time and no relabeling a failed run as complete.
    if any((case/'frames').glob('frame_*.npz')):
        refresh_timeseries(case)
        with np.load(case/'timeseries.npz', allow_pickle=False) as series:
            report['saved_diagnostics'] = dict(
                frames=len(series['times_fs']), last_saved_fs=float(series['times_fs'][-1]),
                max_energy_change_Ha=float(np.max(abs(series['energy']-series['energy'][0]))),
                max_projection_rate_residual=float(np.max(series['projection_residual'])),
                max_pnc_error=float(np.max(series['pnc_error'])))
        if render:
            try:
                from .renderer import render as base_render
                base_render(case, movies=movies)
                if isinstance(config, ChannelConfig):
                    from .channel_render import render as channel_render
                    channel_render(case, movies=movies)
                report['render_status'] = 'COMPLETE_SAVED_FRAMES_ONLY'
            except Exception as exc:
                report.update(render_status='FAILED', render_error=repr(exc))
    report['wall_seconds'] = time.monotonic()-started
    atomic_json(root/'validation_report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--targets-fs', nargs='+', type=float, default=[1., 10., 40.])
    p.add_argument('--movies', action='store_true')
    p.add_argument('--no-render', action='store_true')
    args = p.parse_args()
    raw = json.loads(args.config.read_text())
    if raw.get('method', 'mte') in ('ctmqc', 'tsh'):
        from .channel_run import ChannelConfig
        config = ChannelConfig(**raw)
    else:
        config = RunConfig(**raw)
    result = validate_stages(config, args.out, args.targets_fs,
                             movies=args.movies, render=not args.no_render)
    print(json.dumps(result, indent=2), flush=True)
    if not result['all_targets_completed'] or result.get('render_status') == 'FAILED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
