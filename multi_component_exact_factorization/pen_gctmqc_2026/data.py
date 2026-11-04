"""Read author snapshots and compute only directly available observables.

The stream contains fixed-basis conditional vectors, NOT a nuclear wavefunction.
No invented nuclear phase, exact EF potentials, or Python CT force is inserted.
"""
import json
from pathlib import Path
import numpy as np
from ..pen_perspectives_2026.model import AU_FS, Config, electronic, photon_operators
from ..pen_ctmqc_2026.core import hamiltonian

POL_PERM = np.array([0, 2, 1, 3])  # author photon-major -> Python electronic-major


def read_frame(path, config, perspective):
    with Path(path).open('rb') as stream:
        header = np.fromfile(stream, dtype=np.int64, count=4)
        if header.size != 4:
            raise ValueError('Truncated frame header')
        n, d, k, step = map(int, header)
        if n < 2 or (d, k) != ((2, 2) if perspective == 'el' else (1, 4)):
            raise ValueError('Invalid dimensions/endianness')
        expected_bytes=48+8*n*(4*d+3*k+d*k+k*k)
        if Path(path).stat().st_size!=expected_bytes:
            raise ValueError('Snapshot length disagrees with header')
        time_norm = np.fromfile(stream, dtype=np.float64, count=2)
        if time_norm.size != 2:
            raise ValueError('Truncated frame time')
        time_au, norm_error = time_norm
        def array(shape, dtype=np.float64):
            a = np.fromfile(stream, dtype=dtype, count=int(np.prod(shape)))
            if a.size != np.prod(shape) or not np.isfinite(a).all():
                raise ValueError('Truncated or nonfinite snapshot')
            return a.reshape(shape, order='F')
        X, velocity = array((n, d)), array((n, d))
        coeff, energies = array((n, k), np.complex128), array((n, k))
        mf, ct = array((n, d)), array((n, d))
        history, kli = array((n, d, k)), array((n, k, k))
        if stream.read(1):
            raise ValueError('Unexpected trailing snapshot bytes')
    if not np.isfinite(time_norm).all() or time_au < 0 or norm_error < 0:
        raise ValueError('Invalid time/norm monitor')
    if step<0 or not np.isclose(time_au,step*config.dt,rtol=1e-13,atol=1e-12):
        raise ValueError('Snapshot/config timestep mismatch')
    if perspective == 'pol':
        coeff = coeff[:, POL_PERM]
    masses = np.array([config.mass, 1.] if perspective == 'el' else [config.mass])
    return dict(t_fs=float(time_au*AU_FS), step=step, X=X, P=velocity*masses,
                coeff=coeff, energies=energies, mf_force=mf, ct_force=ct,
                integrated_forces=history, k_li=kli, raw_norm_error=float(norm_error))


def frame_observables(frame, c, view):
    X, P, u = (frame[k] for k in ('X', 'P', 'coeff'))
    n = len(X)
    v, grad = hamiltonian(c, view, X)
    e, vec = np.linalg.eigh(v)
    pop = abs(np.einsum('nik,ni->nk', vec, u))**2
    energy_local = np.einsum('ni,nij,nj->n', u.conj(), v, u).real
    mf = -np.einsum('ni,ndij,nj->nd', u.conj(), grad, u).real
    bo_vec = np.linalg.eigh(electronic(c, X[:, 0])[0])[1]
    norm = np.sum(abs(u)**2, axis=1)
    if view == 'pol':
        ep = u.reshape(n, 2, 2)
        bo = np.sum(abs(np.einsum('iek,ien->ikn', bo_vec, ep))**2, axis=2)
        number = np.sum(abs(ep[:, :, 1])**2, axis=1)
        ops = photon_operators(c)
        expect = lambda op: np.einsum('ien,nm,iem->i', ep.conj(), op, ep).real
        q, p = expect(ops['q']), expect(ops['p'])
        kinetic = P[:, 0]**2/(2*c.mass)
    else:
        bo = abs(np.einsum('nik,ni->nk', bo_vec, u))**2
        q, p = X[:, 1], P[:, 1]
        number = .5*(c.omega*q*q+p*p/c.omega)-.5
        kinetic = P[:, 0]**2/(2*c.mass)+.5*p*p
    width = c.sigma_r/np.sqrt(2)
    kernel = np.exp(-.5*((X[:, 0]-4)/width)**2)/(np.sqrt(2*np.pi)*width)
    row = dict(t_fs=frame['t_fs'], step=frame['step'], norm=float(norm.mean()),
        coefficient_norm_max_error=float(np.max(abs(norm-1))),
        raw_norm_error=frame['raw_norm_error'],
        energy=float(np.mean(kinetic+energy_local)),
        energy_se=float(np.std(kinetic+energy_local, ddof=1)/np.sqrt(n)),
        photon_number=float(number.mean()), photon_number_se=float(number.std(ddof=1)/np.sqrt(n)),
        kinetic_R=float(np.mean(P[:, 0]**2/(2*c.mass))),
        bo_population=bo.mean(axis=0).tolist(), basis_population=pop.mean(axis=0).tolist(),
        R_mean=float(X[:, 0].mean()), R_sd=float(X[:, 0].std()),
        q_mean=float(q.mean()), p_mean=float(p.mean()),
        product_R_ge_4=float(np.mean(X[:, 0]>=4)),
        signed_flux_R4=float(np.mean(kernel*P[:, 0]/c.mass)),
        mf_force_rms=np.sqrt(np.mean(frame['mf_force']**2, axis=0)).tolist(),
        ct_force_rms=np.sqrt(np.mean(frame['ct_force']**2, axis=0)).tolist(),
        author_mf_force_max_error=float(np.max(abs(mf-frame['mf_force']))),
        author_surface_max_error=float(np.max(abs(e-np.sort(frame['energies'], axis=1)))))
    row['pol_population' if view == 'pol' else 'cbo_population'] = row['basis_population']
    return row


def load_case(root):
    root = Path(root)
    c = Config(**json.loads((root/'config.json').read_text()))
    meta = json.loads((root/'author_settings.json').read_text())
    files = sorted((root/'adapter_frames').glob('frame_*.bin'))
    return c, meta, files


def quantum_momentum_bound_excess(frame, c, meta):
    """Independent bound implied by the pinned nonperiodic M/sigma cutoff.

    In coherence_corrections.f90, |Q_d| < M_d/sigma_d on every accepted
    corrected/density branch; k_ij=sum_d Q_d f_di/m_d. Violations therefore
    expose a boundary/settings/layout error, not a CTMQC energy-conservation test.
    """
    if meta['perspective'] == 'el':
        sigma, mass = np.array([c.sigma_r,1/np.sqrt(c.omega)]), np.array([c.mass,1.])
    else:
        sigma, mass = np.array([c.sigma_r]), np.array([c.mass])
    bound = np.sum(abs(frame['integrated_forces'])*(meta['m_parameter']/sigma/mass)[None,:,None],axis=1)
    return float(np.maximum(abs(frame['k_li'])-bound[:,:,None],0).max())


def analyze_case(root, expected_final_fs=None):
    root = Path(root)
    c, meta, files = load_case(root)
    if not files:
        raise ValueError('No author snapshots')
    rows = []
    bound_excess = 0.
    for f in files:
        frame = read_frame(f,c,meta['perspective'])
        rows.append(frame_observables(frame,c,meta['perspective']))
        if meta.get('adapter_safety_version',0) >= 2:
            bound_excess = max(bound_excess,quantum_momentum_bound_excess(frame,c,meta))
    t = np.array([r['t_fs'] for r in rows])
    if t[0] != 0 or np.any(np.diff(t)<=0):
        raise ValueError('Non-monotone or incomplete initial time series')
    if expected_final_fs is not None and t[-1]+1e-8 < expected_final_fs:
        raise ValueError('Author run stopped early: '+str(t[-1]))
    if expected_final_fs is not None:
        expected_steps = sorted(set(range(0, meta['steps']+1, meta['dump_steps'])) | {meta['steps']})
        if [r['step'] for r in rows] != expected_steps:
            raise ValueError('Missing, extra or misordered binary snapshots')
    energy = np.array([r['energy'] for r in rows])
    report = dict(frames=len(rows), last_saved_fs=float(t[-1]),
        max_normalized_norm_error=max(r['coefficient_norm_max_error'] for r in rows),
        max_raw_norm_step_error=max(r['raw_norm_error'] for r in rows),
        max_energy_drift_Ha=float(np.max(abs(energy-energy[0]))),
        final_energy_drift_Ha=float(energy[-1]-energy[0]),
        max_mf_force_error=max(r['author_mf_force_max_error'] for r in rows),
        max_surface_error=max(r['author_surface_max_error'] for r in rows),
        max_open_boundary_k_bound_excess=bound_excess if meta.get('adapter_safety_version',0)>=2 else None,
        numerical_convergence_certified=False,
        warnings=['Upstream renormalizes every electronic step; norm=1 is not an accuracy proof.',
                  'raw_norm_error is the maximum over all electronic steps since the previous saved frame.',
                  'CTMQC total energy need not be conserved; examine dt/Ntraj/M convergence.',
                  'Exact MCEF scalars/connections unavailable without nuclear wavefunction.'])
    if meta['perspective'] == 'el':
        report['warnings'].append('EL photon number is a Wigner/classical estimator, not Fock population.')
    if report['max_raw_norm_step_error'] > 1e-5:
        report['warnings'].append('Raw coefficient norm error exceeded 1e-5 before upstream renormalization; refine dt.')
    if report['max_energy_drift_Ha'] > 1e-3:
        report['warnings'].append('Energy drift exceeds 1e-3 Ha; examine method and timestep sensitivity before physical interpretation.')
    if report['max_mf_force_error'] > 1e-7 or report['max_surface_error'] > 1e-8:
        report['warnings'].append('Hamiltonian/phase-following diagnostic failed; DO NOT trust this run.')
    (root/'observables.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    (root/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
