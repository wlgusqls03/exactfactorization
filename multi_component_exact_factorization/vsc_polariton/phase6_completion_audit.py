"""Read-only campaign comparisons. Never infer scientific PASS from completion."""
import argparse
import json
from pathlib import Path
import numpy as np
from .run_phase6_gpu import save_json, absolute_failures

LIMITS = dict(product=2e-4, flux=2e-6, nph=2e-3, mean_R=2e-4)


def collect(path):
    if path.is_file():
        with np.load(path) as a:
            return {k:a[k].copy() for k in a.files}
    files = sorted(path.glob('observable_*.npz'))
    if not files:
        raise FileNotFoundError(path)
    rows = []
    for p in files:
        with np.load(p) as a:
            rows.append({k:a[k].copy() for k in a.files})
    return {k:np.array([r[k] for r in rows]) for k in rows[0]}


def diagnostic(a):
    expected = np.arange(0., 1652.01, 4.)
    full = a['time_au'].shape == expected.shape and np.allclose(a['time_au'], expected, atol=1e-12, rtol=0)
    failures = {}
    for i in range(len(a['time_au'])):
        row = {k:v[i] for k,v in a.items()}
        for key, value in absolute_failures(row, a['energy'][0]).items():
            failures.setdefault(key, value)
    integrated = np.concatenate(([0.], np.cumsum(np.diff(a['time_au'])*(a['flux'][1:]+a['flux'][:-1])/2)))
    continuity = float(np.max(abs(a['product']-a['product'][0]-integrated)))
    return dict(pass_checks=bool(full and not failures and continuity < 2e-4),
                full_interval=bool(full), failures=failures, integrated_continuity=continuity,
                norm=float(np.max(abs(a['norm']-1))),
                energy_drift=float(np.max(abs(a['energy']-a['energy'][0]))),
                max_product=float(a['product'].max()), max_P_exc=float(a['P_exc'].max()))


def compare(a, b):
    np.testing.assert_allclose(a['time_au'], b['time_au'], atol=1e-12, rtol=0)
    errors = {k:float(np.max(abs(a[k]-b[k]))) for k in LIMITS}
    return dict(pass_checks=bool(all(np.isfinite(errors[k]) and errors[k] < lim for k,lim in LIMITS.items())),
                errors=errors, limits=LIMITS,
                bare_BO_character_difference=float(np.max(abs(a['P_exc']-b['P_exc']))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    a = p.parse_args(); root = a.out
    runs, comparisons, controls, missing = {}, {}, {}, []
    for case in ('free', 'barrier'):
        pair = []
        keys = ('free_F120','free_F120_dt00625') if case=='free' else ('barrier_F120','barrier_F160')
        for key in keys:
            data = collect(root/key/'full'); pair.append(data)
            runs[key] = diagnostic(data)
        comparisons[case+('_dt' if case=='free' else '_Fock')] = compare(*pair)
    for case in ('free', 'resonant', 'barrier'):
        for model in ('A', 'B', 'C'):
            key = f'{case}_{model}'
            pair = [collect(root/'controls'/f'{key}_dt{dt}.npz') for dt in (.125, .0625)]
            controls[key] = dict(coarse=diagnostic(pair[0]), fine=diagnostic(pair[1]),
                                 time_convergence=compare(*pair))
            fock = collect(root/'controls'/f'{key}_F160_dt0.125.npz')
            controls[key].update(fock_fine=diagnostic(fock), fock_convergence=compare(pair[0], fock))
            # Save dynamical model differences, not causal claims.
            full = collect(a.inputs/'resonant_observables.npz') if case == 'resonant' else collect(root/f'{case}_F120/full')
            controls[key]['full3D_difference'] = compare(full, pair[1])
    stationary = json.loads((root/'controls/stationary.json').read_text())
    historical = json.loads((a.inputs/'historical_validation.json').read_text())
    passed = all(v['pass_checks'] for v in runs.values()) and all(v['pass_checks'] for v in comparisons.values())
    passed &= all(all(v[k]['pass_checks'] for k in ('coarse','fine','time_convergence','fock_fine','fock_convergence')) for v in controls.values())
    passed &= stationary['status'] == 'PASS'
    passed &= historical['computed_gates_pass']
    # Existing resonant/spatial archives are not replaced or silently synthesized.
    missing.extend(['Final protected-source/artifact and regression review',
                    'Scientific review of full3D vs 2D-A/B/C and stationary-sector scope'])
    report = dict(status='COMPUTED_GATES_PASS_REVIEW_PENDING' if passed else 'FAIL',
                  phase6_pass=False, phase7_allowed=False, runs=runs, comparisons=comparisons,
                  controls=controls, stationary=stationary, historical=historical, outstanding=missing)
    save_json(root/'phase6_completion_validation.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
