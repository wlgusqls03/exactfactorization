"""Flag suspect completed runs without changing the author's propagation.

These are conservative review thresholds, NOT a theorem/certificate of CTMQC
accuracy. In particular, CTMQC energy change alone is not an integrator error.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import norm_intervals


def assess(rows):
    raw = np.asarray([r['raw_norm_error'] for r in rows])
    energy = np.asarray([r['energy'] for r in rows])
    if not len(rows) or not np.isfinite(raw).all() or not np.isfinite(energy).all():
        raise ValueError('Empty/nonfinite monitoring data')
    largest = float(raw.max())
    drift = float(np.max(abs(energy-energy[0])))
    if largest > 1e-3:
        verdict = 'LARGE_STEP_ERROR_DO_NOT_USE_AS_CONVERGED'
    elif largest > 1e-5 or drift > 1e-3:
        verdict = 'REQUIRES_NUMERICAL_REVIEW'
    else:
        verdict = 'NO_LARGE_ERRORS_DETECTED_NOT_CERTIFIED'
    return dict(verdict=verdict, certified=False,
        max_raw_norm_step_error=largest, max_energy_change_Ha=drift,
        raw_norm_review=norm_intervals(rows),
        thresholds=dict(raw_norm_review=1e-5, raw_norm_large=1e-3, energy_review_Ha=1e-3),
        note='Saved norm is renormalized. Energy change can include method error. '
             'Even a clean monitor still needs dt/Ntraj/M/seed convergence.')


def check_campaign(root, out, require_through_fs=None):
    root, out = Path(root), Path(out)
    cases = []
    for path in sorted(root.rglob('author_settings.json')):
        case = path.parent
        record = dict(case=str(case.relative_to(root)))
        try:
            rows = json.loads((case/'observables.json').read_text())
            record.update(assess(rows), last_saved_fs=rows[-1]['t_fs'])
            output = json.loads((case/'output_validation.json').read_text())
            record['native_output_passed'] = output['passed']
            meta = json.loads(path.read_text())
            record['run_parameters'] = {k:meta.get(k) for k in
                ('perspective','ntraj','seed','dt_au','m_parameter','algorithm','initial')}
            record['explicit_boundary_initialization'] = meta.get('adapter_safety_version',0) >= 2
            record['explicit_zero_soc_initialization'] = meta.get('adapter_safety_version',0) >= 3
            status = json.loads((case/'status.json').read_text())
            record['execution_status'] = status['status']
            if not output['passed']:
                record['verdict'] = 'OUTPUT_FAILED'
            elif status['status'] not in ('COMPLETE', 'COMPLETE_NOT_CERTIFIED'):
                record['verdict'] = 'EXECUTION_NOT_COMPLETE'
            elif require_through_fs is not None and rows[-1]['t_fs']+1e-8 < require_through_fs:
                record['verdict'] = 'INCOMPLETE_TIME_COVERAGE'
            elif not record['explicit_zero_soc_initialization'] and record['verdict']=='NO_LARGE_ERRORS_DETECTED_NOT_CERTIFIED':
                record['verdict'] = 'LEGACY_INITIALIZATION_UNVERIFIED_REBUILD_REQUIRED'
        except (OSError, ValueError, KeyError) as error:
            record.update(verdict='MISSING_OR_INVALID_VALIDATION', error=repr(error), certified=False)
        cases.append(record)
    if not cases:
        raise ValueError('No author cases found')
    report = dict(cases=cases, certified=False, required_through_fs=require_through_fs,
                  numerical_review_required=any(r['verdict'] != 'NO_LARGE_ERRORS_DETECTED_NOT_CERTIFIED' for r in cases))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-through-fs',type=float)
    a = p.parse_args()
    report = check_campaign(a.root,a.out,a.require_through_fs)
    for row in report['cases']:
        print(row['case'],row['verdict'], 'max raw norm =',row.get('max_raw_norm_step_error'),
              'max |dE| =',row.get('max_energy_change_Ha'))
    print('HEALTH',a.out,'(exit 2 means review required, not an output/plotting exception)')
    if report['numerical_review_required']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
