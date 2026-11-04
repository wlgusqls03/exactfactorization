"""Native output contract, separate from dynamical/convergence validation.

The pinned Fortran writer does not create directories and can continue after
OPEN fails (writing to fort.128). A zero return code is therefore insufficient.
No propagation equations are changed here. The v2 build omits the upstream
uninitialized trailing Vdiadiag columns and initializes open boundary flags.
"""
import json
from pathlib import Path
import re
import numpy as np
from .data import load_case, read_frame, frame_observables
from ..pen_perspectives_2026.model import AU_FS


def prepare_output_dirs(root):
    root = Path(root)
    for relative in ('adapter_frames', 'native/coeff', 'native/trajectories'):
        (root/relative).mkdir(parents=True, exist_ok=True)


def validate_boundary_log(log, dimensions, m_parameter):
    def value(name):
        match = re.search(r'^\s*'+name+r'\s+(.+)$', log, re.MULTILINE)
        if match is None:
            raise ValueError('Missing runtime boundary audit: '+name)
        return match.group(1).split()
    flags = value('ADAPTER_PERIODIC_FLAGS')
    if flags != ['F']*dimensions:
        raise ValueError('Expected explicitly nonperiodic coordinates: '+str(flags))
    if value('ADAPTER_DOUBLE_INTERCEPT') != ['F']:
        raise ValueError('Unsupported double-intercept runtime setting')
    actual_m = [float(x.replace('D','E')) for x in value('ADAPTER_M_PARAMETER')]
    if len(actual_m) != dimensions or not np.allclose(actual_m,m_parameter,rtol=1e-13,atol=0):
        raise ValueError('Runtime M_parameter disagrees with saved settings')
    return dict(periodic_flags=flags,m_parameter=actual_m,double_intercept=False)


def validate_native_outputs(root):
    """Check every scheduled native frame and its binary counterpart.

    Native text only includes multiples of dump; the adapter additionally records
    the final step. Tolerances reflect f14.6/f14.8 text rounding, not TDSE accuracy.
    Old RPE trailing uninitialized Vdiadiag columns are never interpreted; new
    builds omit them. Binary snapshots remain authoritative.
    """
    root = Path(root)
    report = dict(passed=False, warnings=[
        'Uninitialized upstream RPE Vdiadiag columns are omitted in v2; never interpreted in old output.',
        'Text/binary agreement is an I/O check, not dynamical convergence.'])
    try:
        c, meta, _ = load_case(root)
        steps, dump = meta['steps'], meta['dump_steps']
        n = meta['ntraj']
        d, k = (2, 2) if meta['perspective'] == 'el' else (1, 4)
        scheduled = np.arange(0, steps+1, dump, dtype=int)
        log = (root/'run.log').read_text(errors='replace')
        if re.search(r'error[e]?\s+opening', log, re.IGNORECASE):
            raise ValueError('Native OPEN failure in run.log')
        if list(root.glob('fort.*')):
            raise ValueError('Unexpected Fortran implicit-unit output: fort.*')
        if meta.get('adapter_safety_version',0) >= 2:
            report['runtime_boundary_audit'] = validate_boundary_log(log,d,meta['m_parameter'])

        def table(path, columns, rows):
            values = np.loadtxt(path, usecols=range(columns), ndmin=2)
            if values.shape != (rows, columns) or not np.isfinite(values).all():
                raise ValueError('Incomplete/nonfinite native table: '+str(path))
            return values

        energy = table(root/'native/CT_energy.dat', 2, len(scheduled))
        population = table(root/'native/BO_population.dat', 1+k, len(scheduled))
        photons = table(root/('native/electronic_photon_pop.dat' if d==2 else
                             'native/pops_photonic.dat'), 6 if d==2 else 9, len(scheduled))
        expected_t = scheduled*c.dt
        for values in (energy, population, photons):
            if np.max(abs(values[:, 0]-expected_t)) > 5.1e-5:
                raise ValueError('Native text time grid mismatch')
        for folder, pattern in (('coeff', 'coeff.*.dat'), ('trajectories', 'RPE.*.dat')):
            if len(list((root/'native'/folder).glob(pattern))) != len(scheduled):
                raise ValueError('Missing or extra native frames: '+folder)

        errors = dict(position=0., momentum=0., surface=0., conditional_energy=0.,
                      ensemble_energy=0., photon_number=0., density_matrix_trace=0.)
        for index, step in enumerate(scheduled):
            frame = read_frame(root/f'adapter_frames/frame_{step:010d}.bin', c, meta['perspective'])
            row = frame_observables(frame, c, meta['perspective'])
            coeff = table(root/f'native/coeff/coeff.{index:04d}.dat', d+2*k*k, n)
            rpe = table(root/f'native/trajectories/RPE.{index:04d}.dat', 2*d+1+k, n)
            errors['position'] = max(errors['position'], float(np.max(abs(rpe[:, :d]-frame['X']))))
            errors['momentum'] = max(errors['momentum'], float(np.max(abs(rpe[:, d:2*d]-frame['P']))))
            errors['surface'] = max(errors['surface'], float(np.max(abs(rpe[:, 2*d+1:]-frame['energies']))))
            diagonal = coeff[:, d+np.arange(k)*(k+1)]
            errors['density_matrix_trace'] = max(errors['density_matrix_trace'],
                                                 float(np.max(abs(diagonal.sum(axis=1)-1))))
            errors['conditional_energy'] = max(errors['conditional_energy'],
                float(np.max(abs(rpe[:, 2*d]-np.sum(diagonal*frame['energies'], axis=1)))))
            errors['ensemble_energy'] = max(errors['ensemble_energy'], abs(energy[index, 1]-row['energy']))
            # POL photon-major populations: columns 7,8 are P(n=0),P(n=1).
            photon = photons[index, 5 if d==2 else 8]
            errors['photon_number'] = max(errors['photon_number'], abs(photon-row['photon_number']))
        report.update(native_frames=len(scheduled), native_last_time_fs=float(expected_t[-1]*AU_FS), max_abs_errors=errors,
                      rpe_checked_columns=2*d+1+k)
        for key, error in errors.items():
            tolerance = 5.1e-7 if key in ('position', 'momentum', 'surface') else (
                2e-6 if key == 'conditional_energy' else (k*5.1e-9 if key == 'density_matrix_trace' else 5.1e-9))
            if error > tolerance:
                raise ValueError(f'Native/binary {key} mismatch: {error:g} > {tolerance:g}')
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        (root/'output_validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
