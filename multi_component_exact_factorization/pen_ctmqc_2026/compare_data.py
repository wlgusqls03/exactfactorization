"""Read-only QD/CTMQC comparison adapters. No trajectory or TDSE propagation.

QD means the repository's finite-Fock TDSE, NOT the paper's full photon-grid
benchmark. EL has classical photon phase space, not a quantum Fock vector.
Density comparisons use matched smoothing; no lost mass is renormalized away.
Exact EF fields are extracted only from QD coefficients with the native H.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from ..pen_perspectives_2026.model import (
    Config, Propagator, derivative, electronic, oscillator, cbo)
from ..pen_perspectives_2026.fields import analyze
from ..pen_perspectives_2026.workflow import load_npz
from .core import Settings, hamiltonian
from .diagnostics import deposit, ratio

PHYSICAL = ('mass', 'omega', 'g', 'dse', 'k', 'a', 'b', 'r1', 'r2',
            'r3', 'delta', 'z', 'mu12')
METHODS = ('qd', 'el', 'pol')
LABELS = {'qd': 'QD (finite Fock)', 'el': 'CTMQC EL', 'pol': 'CTMQC POL'}


@dataclass
class Archive:
    path: Path
    method: str
    config: Config
    settings: object
    rows: list
    paths: list
    times: np.ndarray
    status: dict

    def values(self, key):
        if not self.rows or not all(key in r for r in self.rows):
            return None
        return np.asarray([r[key] for r in self.rows], dtype=float)


def read_archive(path, method):
    path = Path(path)
    c = Config(**json.loads((path/'config.json').read_text()))
    c.grid()
    s = None if method == 'qd' else Settings(**json.loads((path/'trajectory_config.json').read_text()))
    if s is not None:
        s.validate()
        if s.perspective != method:
            raise ValueError('Directory/perspective mismatch')
    rows = json.loads((path/'observables.json').read_text())
    t = np.array([r['t_fs'] for r in rows])
    if not len(t) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Need finite, strictly increasing observable times')
    folder, pattern = ('coefficients', 'wave_*.npz') if method == 'qd' else ('frames', 'trajectories_*.npz')
    paths = sorted((path/folder).glob(pattern))
    # Inventory only scalar time entries; do not retain full frames in memory.
    times = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            times.append(float(z['t_fs']))
    times = np.array(times)
    if len(times) and (not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError('Non-monotone frame times')
    status = json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
    return Archive(path, method, c, s, rows, paths, times, status)


def matching(a, b):
    """Numerical grids/dt may differ; physics may not. NF matters for QD/POL."""
    keys = list(PHYSICAL)
    if a.method != 'el' and b.method != 'el':
        keys.append('nf')
    return [k for k in keys if not np.isclose(getattr(a.config, k), getattr(b.config, k), rtol=1e-12, atol=1e-14)]


def load_group(ctroot, qdroot, relative):
    archives, warnings = {}, []
    for method in METHODS:
        path = Path(qdroot)/relative if method == 'qd' else Path(ctroot)/relative/method
        if not (path/'observables.json').exists():
            warnings.append(f'{method}: archive unavailable: {path}')
            continue
        try:
            a = read_archive(path, method)
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(f'{method}: unreadable archive: {error}')
            continue
        mismatch = sorted({key for b in archives.values() for key in matching(a, b)})
        if mismatch:
            warnings.append(f'{method}: SKIPPED Hamiltonian/Fock mismatch: {mismatch}')
            continue
        archives[method] = a
        if not str(a.status.get('status', '')).startswith('COMPLETE'):
            warnings.append(f'{method}: status={a.status.get("status", "unknown")}; only saved data are used')
        if method != 'qd' and a.settings.algorithm != 'ctmqc':
            warnings.append(f'{method}: algorithm={a.settings.algorithm}; not a CTMQC run')
    if 'el' in archives and archives['el'].settings.initial == 'paper':
        warnings.append('EL(paper) starts in upper local CBO, QD/POL in bare S1 x vacuum: different preparations.')
    warnings.append('QD is a finite-Fock reference, not a certified reproduction of the paper photon-grid QD.')
    warnings.append('EL photon variables are classical; quantum Fock/Pol populations are unavailable in EL.')
    if 'qd' in archives:
        warnings.append('QD uses periodic R FFT; CTMQC has open analytic boundaries. Inspect edge diagnostics.')
    return archives, warnings


def overlap(archives, tmax=None, frames=False):
    ts = [a.times if frames else a.values('t_fs') for a in archives.values()]
    if not ts or any(not len(t) for t in ts):
        return None
    lo, hi = max(t[0] for t in ts), min(t[-1] for t in ts)
    if tmax is not None:
        hi = min(hi, tmax)
    return (float(lo), float(hi)) if hi >= lo else None


def aligned_errors(archives, interval):
    """Scalar observable errors only: linear interpolation, no extrapolation."""
    if 'qd' not in archives or interval is None:
        return {}
    qd = archives['qd']
    t = qd.values('t_fs')
    mask = (t >= interval[0]) & (t <= interval[1])
    t = t[mask]
    result = {}
    for method, a in archives.items():
        if method == 'qd':
            continue
        result[method] = {}
        for key in ('photon_number', 'bo_population', 'kinetic_R', 'product_R_ge_4', 'signed_flux_R4'):
            q, v = qd.values(key), a.values(key)
            if q is None or v is None:
                continue
            q, v = (q[:, 1], v[:, 1]) if key == 'bo_population' else (q, v)
            err = np.interp(t, a.values('t_fs'), v)-q[mask]
            result[method][key] = dict(t_fs=t.tolist(), error=err.tolist(),
                sample_rms=float(np.sqrt(np.mean(err**2))) if len(err) else None,
                max_abs=float(np.max(abs(err))) if len(err) else None)
    return result


def nearest_frame(a, t, tolerance=None):
    if not len(a.times) or t < a.times[0]-1e-9 or t > a.times[-1]+1e-9:
        return None
    i = int(np.argmin(abs(a.times-t)))
    if tolerance is None:
        tolerance = .51*float(np.max(np.diff(a.times))) if len(a.times) > 1 else 1e-9
    return i if abs(a.times[i]-t) <= tolerance+1e-9 else None


def choose_snapshots(archives, interval, every_fs=5., explicit=None):
    anchor = archives.get('qd', next(iter(archives.values())))
    if interval is None or not len(anchor.times):
        return []
    lo, hi = interval
    wanted = np.arange(lo, hi+1e-9, every_fs) if explicit is None else np.asarray(explicit)
    wanted = np.r_[lo, wanted[(wanted >= lo) & (wanted <= hi)], hi]
    valid = np.flatnonzero((anchor.times >= lo-1e-9) & (anchor.times <= hi+1e-9))
    if not len(valid):
        return []
    indices = {int(valid[np.argmin(abs(anchor.times[valid]-t))]) for t in wanted}
    return [float(anchor.times[i]) for i in sorted(indices)]


class FrameReader:
    """One reference geometry and matched display kernels across all methods.

    QD native fields remain unsmoothed. Common-density/conditional-energy
    comparisons convolve *numerators and denominators*, not their quotient.
    h_R/h_q are display widths, independent of dynamics quantum-momentum Q.
    """
    def __init__(self, archives, density_width_scale=1., nr=241, nq=161):
        self.archives = archives
        self.c = archives.get('qd', next(iter(archives.values()))).config
        self.p = Propagator(archives['qd'].config) if 'qd' in archives else None
        # Deposit CT ensembles on the native QD grid when available. Linear
        # regridding of a narrow quantum packet can change its integrated mass
        # even before smoothing, contaminating a three-way density comparison.
        self.R = self.p.R.copy() if self.p is not None else np.linspace(self.c.rmin, self.c.rmax, nr)
        qmax = max(10., 8/np.sqrt(2*self.c.omega))
        self.q = np.linspace(-qmax, qmax, nq)
        self.widths = np.array([self.c.sigma_r/np.sqrt(2), 1/np.sqrt(2*self.c.omega)])*density_width_scale
        self.spacing = np.array([self.R[1]-self.R[0], self.q[1]-self.q[0]])
        self.ce, self.cv = cbo(self.c, self.R, self.q)

    def interp(self, values):
        """Display only; native derivatives are computed before interpolation."""
        values = np.asarray(values)
        shape = values.shape[1:]
        flat = values.reshape(len(values), -1)
        out = np.column_stack([np.interp(self.R, self.p.R, v, left=0., right=0.) for v in flat.T])
        return out.reshape((len(self.R),)+shape)

    def smooth(self, value, axes=(0, 1)):
        sigma = [0.]*value.ndim
        for axis in axes:
            sigma[axis] = self.widths[axis]/self.spacing[axis]
        fn = lambda v: gaussian_filter(v, sigma, mode='constant', truncate=5.)
        return fn(value.real)+1j*fn(value.imag) if np.iscomplexobj(value) else fn(value)

    def read(self, method, index):
        a = self.archives[method]
        z = load_npz(a.paths[index])
        c, R, q = a.config, self.R, self.q
        native = None
        if method == 'qd':
            u, p = z['u'], self.p
            rho = np.sum(abs(u)**2, axis=(1, 2))
            norm = rho.sum()*p.dr
            h = oscillator(c, q)[0]
            wave = np.einsum('ren,nq->rqe', u, h)
            edm = self.interp(wave[:, :, :, None]*wave[:, :, None, :].conj()/norm)
            raw_R = self.interp(rho/norm)
            raw_q = np.sum(abs(wave)**2, axis=(0, 2))*p.dr/norm
            smooth_edm = self.smooth(edm)
            bo = np.einsum('rek,ren->rkn', p.bo_vectors, u)
            bo_density = self.smooth(self.interp(np.sum(abs(bo)**2, axis=2)/norm), (0,))
            pol = np.einsum('rik,ri->rk', p.pol_vectors, u.reshape(c.nr, -1))
            pol_density = self.smooth(self.interp(abs(pol)**2/norm), (0,))
            energy_num = np.sum(abs(pol)**2*p.pol_energy, axis=1)/norm
            grad = hamiltonian(c, 'pol', p.R[:, None])[1][:, 0]
            flat = u.reshape(c.nr, -1)
            force_num = -np.einsum('ni,nij,nj->n', flat.conj(), grad, flat).real/norm
            current = np.sum((u.conj()*derivative(u, p.dr)).imag, axis=(1, 2))/(c.mass*norm)
            den = self.smooth(raw_R, (0,))
            outer_energy = ratio(self.smooth(self.interp(energy_num), (0,)), den)
            mf_force = ratio(self.smooth(self.interp(force_num), (0,)), den)
            ct_force = np.zeros_like(den)
            current = self.smooth(self.interp(current), (0,))
            native = analyze(u, p, float(z['t_fs']), nq=len(q), qmax=max(abs(q)))
            # Q=-chi''/(2M chi). Native third derivatives avoid differencing a
            # noisy density quotient, as in fields.analyze's epsilon2 force.
            ur, urr, urrr = [derivative(u, p.dr, n) for n in (1, 2, 3)]
            r1 = 2*np.sum((u.conj()*ur).real, axis=(1, 2))
            r2 = 2*np.sum((u.conj()*urr).real+abs(ur)**2, axis=(1, 2))
            r3 = 2*np.sum((u.conj()*urrr).real, axis=(1, 2))+6*np.sum((ur.conj()*urr).real, axis=(1, 2))
            curvature_r = ratio(r3, 2*rho)-ratio(r1*r2, rho**2)+.5*ratio(r1**3, rho**3)
            native['quantum_amplitude_force'] = curvature_r/(2*c.mass)
            native['hydrodynamic_force'] = native['force']+native['quantum_amplitude_force']
            fock = np.sum(abs(u)**2, axis=(0, 1))*p.dr/norm
            mean_force = float(np.sum(force_num)*p.dr)
        else:
            X, P, coeff = z['X'], z['P'], z['coeff']
            n = len(X)
            dm = coeff[:, :, None]*coeff[:, None, :].conj()
            raw_R, raw_q = None, None
            den = deposit(X[:, :1], [R], np.ones(n), self.widths[:1])
            if method == 'pol':
                dm = deposit(X[:, :1], [R], dm, self.widths[:1]).reshape(len(R), 2, c.nf, 2, c.nf)
                h = oscillator(c, q)[0]
                edm = np.einsum('renfm,nq,mq->rqef', dm, h, h)
                smooth_edm = self.smooth(edm, (1,))
                elph = coeff.reshape(n, 2, c.nf)
                bv = np.linalg.eigh(electronic(c, X[:, 0])[0])[1]
                bp = np.sum(abs(np.einsum('iek,ien->ikn', bv, elph))**2, axis=2)
                pv = np.linalg.eigh(hamiltonian(c, 'pol', X)[0])[1]
                pp = abs(np.einsum('nik,ni->nk', pv, coeff))**2
                pol_density = deposit(X[:, :1], [R], pp, self.widths[:1])
                fock = np.mean(np.sum(abs(elph)**2, axis=1), axis=0)
                energy = z['mean_energy']
            else:
                edm = deposit(X, [R, q], dm, self.widths)
                smooth_edm = edm
                bv = np.linalg.eigh(electronic(c, X[:, 0])[0])[1]
                bp = abs(np.einsum('nik,ni->nk', bv, coeff))**2
                pol_density, fock = None, None
                # EL saved mean_energy omits classical photon kinetic energy.
                # Add it ONLY for the outer H-T_R comparison, never for CBO.
                energy = z['mean_energy']+.5*P[:, 1]**2
            bo_density = deposit(X[:, :1], [R], bp, self.widths[:1])
            avg = lambda v: ratio(deposit(X[:, :1], [R], v, self.widths[:1]), den)
            outer_energy = avg(energy)
            mf_force, ct_force = avg(z['mf_force'][:, 0]), avg(z['ct_force'][:, 0])
            current = deposit(X[:, :1], [R], P[:, 0]/c.mass, self.widths[:1])
            mean_force = float(np.mean(z['mf_force'][:, 0]+z['ct_force'][:, 0]))
        joint = np.trace(smooth_edm, axis1=-2, axis2=-1).real
        local = np.einsum('rqek,rqef,rqfk->rqk', self.cv, smooth_edm, self.cv).real
        energy = ratio(np.sum(local*self.ce, axis=2), joint)
        rhoq = np.trapz(joint, R, axis=0)
        return dict(method=method, t_fs=float(z['t_fs']), R=R, q=q, joint=joint,
            rho_R=den, rho_q=rhoq, raw_R=raw_R, raw_q=raw_q, native=native,
            bo_density=bo_density, pol_density=pol_density, fock_population=fock,
            mean_CBO_energy=energy, outer_mean_energy=outer_energy, current_R=current,
            mf_force=mf_force, ct_force=ct_force, mean_force=mean_force,
            density_mass_R=float(np.trapz(den, R)), density_mass_q=float(np.trapz(rhoq, q)),
            photon_q_space='classical EL' if method == 'el' else 'quantum finite-Fock',
            kernels=self.widths.tolist())


def snapshot_metrics(frames):
    result = dict(times_fs={k: f['t_fs'] for k, f in frames.items()},
        displayed_mass={k: [f['density_mass_R'], f['density_mass_q']] for k, f in frames.items()},
        mean_force={k: f['mean_force'] for k, f in frames.items()},
        fock_populations={k: f['fock_population'].tolist() for k, f in frames.items()
                          if f['fock_population'] is not None})
    if 'qd' in frames:
        ref = frames['qd']
        result['matched_resolution_density_L1'] = {
            k: dict(R=float(np.trapz(abs(f['rho_R']-ref['rho_R']), ref['R'])),
                    q=float(np.trapz(abs(f['rho_q']-ref['rho_q']), ref['q'])))
            for k, f in frames.items() if k != 'qd'}
    return result
