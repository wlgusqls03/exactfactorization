"""Full-duration lightweight convergence audit; stores only a small JSON report."""
import argparse
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
from .model import AU_FS, Config, Propagator, initial, observables
from .fields import analyze, diagnostics
from .workflow import atomic_json, source_record


def trajectory(c, end_fs, sample_times=None):
    p, u = Propagator(c), initial(c)
    steps = int(round(end_fs/(c.dt*AU_FS)))
    sample_steps = (set(np.linspace(0, steps, 41).astype(int)) if sample_times is None
                    else set(np.rint(np.asarray(sample_times)/(c.dt*AU_FS)).astype(int)))
    steps = max(sample_steps)
    rows = []
    for step in range(steps+1):
        if step in sample_steps:
            rows.append(dict(t_fs=step*c.dt*AU_FS, **observables(u, p)))
        if step < steps:
            u = p.step(u)
    return p, u, rows


def differences(base, other):
    t = [r['t_fs'] for r in base]
    tt = [r['t_fs'] for r in other]
    out = {}
    for key in ('photon_number', 'product_R_ge_4', 'signed_flux_R4', 'energy', 'bo_population'):
        a = np.array([r[key] for r in base])
        b = np.array([r[key] for r in other])
        if key == 'bo_population':
            a, b = a[:, 1], b[:, 1]
        out[key+'_max_difference'] = float(np.max(abs(a-np.interp(t, tt, b))))
    return out


def audit(out, end_fs=40.):
    started = time.perf_counter()
    report = dict(source=source_record(), end_fs=end_fs, cases={},
                  interpretation='dt/R checks concern the specified finite-Fock model; nf variation tests truncation, not solver errors.')
    for dse in (False, True):
        for name, g in (('zero', 0.), ('weak', .001), ('strong', .01)):
            c = Config(dse=dse, g=g)
            p, u, baseline = trajectory(c, end_fs)
            same_times = [r['t_fs'] for r in baseline]
            _, _, tfine = trajectory(replace(c, dt=c.dt/2), end_fs, same_times)
            _, _, rfine = trajectory(replace(c, nr=2*c.nr), end_fs, same_times)
            entry = dict(time_step=differences(baseline, tfine), nuclear_grid=differences(baseline, rfine),
                         fields_q500_box10=diagnostics(analyze(u, p, end_fs, nq=500, qmax=10)),
                         fields_q500_box16=diagnostics(analyze(u, p, end_fs, nq=500, qmax=16)),
                         endpoint=baseline[-1])
            if name == 'strong':
                _, _, f4 = trajectory(replace(c, nf=4), end_fs, same_times)
                _, _, f8 = trajectory(replace(c, nf=8), end_fs, same_times)
                entry['fock2_vs4'] = differences(baseline, f4)
                entry['fock4_vs8'] = differences(f4, f8)
            report['cases'][f'dse_{"on" if dse else "off"}/{name}'] = entry
            print(f'AUDIT {name} DSE={dse}: done', flush=True)
    report['wall_seconds'] = time.perf_counter()-started
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--end-fs', type=float, default=40.)
    args = parser.parse_args()
    if not np.isfinite(args.end_fs) or args.end_fs <= 0:
        parser.error('end-fs must be positive and finite')
    audit(args.out, args.end_fs)
