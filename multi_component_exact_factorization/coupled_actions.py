"""Six continuum EF coupling actions evaluated from coherent Psi, in PG.

Central-five nuclear derivatives are diagnostics, NOT the native link/spectral
propagator. Native H Psi supplies the PG temporal scalar term. No residual is
assigned to one of the six actions to enforce a closure identity.
"""
from pathlib import Path
import shutil
import numpy as np
from scipy.fft import dst

from .core import derivative, covariant_square, apply_electronic_hamiltonian
from .external_potential import harmonic_potential

NAMES = ('ep_square', 'ep_ratio', 'en_square', 'en_ratio',
         'pn_square', 'pn_ratio', 'Ue_total', 'Up_total',
         'hBO', 'Hel', 'Tp', 'epsilon1', 'Hpr')


def _ratio(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape),
                     np.nan, dtype=np.result_type(a, b)), where=b > np.finfo(float).tiny)


def action_frame(read_psi, read_action, model, block_R=4, density_floor=1e-3,
                 electronic_method='finite', map_stride=2):
    """Callbacks return coherent (x,q,R_ids); read_action is native H Psi.

Maps: electronic ||A Phi||_x, proton |A Lambda|/Lambda, all in Ha.
Summaries: common occupied-support rho weighted RMS and complex mean. Maps
are decimated only AFTER derivatives and statistics; the wavefunction is not.
"""
    m = model
    nx, nq, nr = len(m.x), len(m.q), len(m.R)
    if min(nq, nr) < 5 or block_R < 1 or map_stride < 1 or density_floor <= 0:
        raise ValueError('Need >=5 nuclear points, positive blocks/stride/floor')
    if electronic_method not in ('finite', 'spectral'):
        raise ValueError('Unknown electronic kinetic representation')
    rho, a, b = (np.empty((nq, nr)) for _ in range(3))
    def read(ids):
        psi = np.asarray(read_psi(ids), complex)
        if psi.shape != (nx, nq, len(ids)) or not np.isfinite(psi).all():
            raise ValueError('Invalid coherent Psi block')
        return psi
    # First pass: site connections, with true periodic R halos.
    for start in range(0, nr, block_R):
        stop = min(nr, start+block_R)
        ids = np.arange(start-2, stop+2) % nr
        psi = read(ids)
        F = np.sqrt(np.sum(np.abs(psi)**2, axis=0)*m.dx)
        phi = _ratio(psi, F[None])
        u = phi[:, :, 2:-2]
        rho[:, start:stop] = F[:, 2:-2]**2
        a[:, start:stop] = np.sum(u.conj()*(-1j*derivative(u, m.dq, 1)), axis=0).real*m.dx
        b[:, start:stop] = np.sum(u.conj()*(-1j*derivative(phi, m.dR, 2)[:, :, 2:-2]), axis=0).real*m.dx
    F = np.sqrt(rho)
    chi = np.sqrt(np.sum(rho, axis=0)*m.dq)
    lam = _ratio(F, chi[None])
    # Undefined zero-density nodes carry no weight in the conditional average.
    alpha = _ratio(np.sum(np.where(rho > 0, rho*np.nan_to_num(b), 0), axis=0)*m.dq, chi**2)
    invalid_b_mass = np.sum(np.where(~np.isfinite(b), rho, 0), axis=0)*m.dq
    alpha[invalid_b_mass > 1e-12*np.maximum(chi**2, np.finfo(float).tiny)] = np.nan
    logq = _ratio(derivative(F, m.dq, 0), F)
    logR = _ratio(derivative(F, m.dR, 1), F)
    logchi = _ratio(derivative(chi, m.dR, 0), chi)
    delta = b-alpha[None]
    pD = -1j*derivative(lam, m.dR, 1)+delta*lam
    pn1 = covariant_square(lam, delta, m.dR, 1, +1)/(2*m.heavy_mass)
    pn2 = (alpha-1j*logchi)[None]*pD/m.heavy_mass
    tp = covariant_square(lam, a, m.dq, 0, +1)/(2*m.proton_mass)
    maps = np.full((len(NAMES), nq, nr), np.nan)
    means = np.full((len(NAMES), nq, nr), np.nan+0j)
    closure_error = np.full((nq, nr), np.nan)
    trap = harmonic_potential(m.R, vars(m))
    for start in range(0, nr, block_R):
        stop = min(nr, start+block_R)
        ids = np.arange(start-2, stop+2) % nr
        core = slice(start, stop)
        psi = read(ids)
        phi = _ratio(psi, F[:, ids][None])
        u = phi[:, :, 2:-2]
        action = np.asarray(read_action(np.arange(start, stop)), complex)
        if action.shape != u.shape or not np.isfinite(action).all():
            raise ValueError('Invalid native H Psi block')
        def store(name, value):
            j = NAMES.index(name)
            maps[j, :, core] = np.sqrt(np.sum(np.abs(value)**2, axis=0)*m.dx)
            means[j, :, core] = np.sum(u.conj()*value, axis=0)*m.dx
        ue = np.zeros_like(u)
        for prefix, axis, spacing, mass, vector, log in (
            ('ep', 1, m.dq, m.proton_mass, a, logq),
            ('en', 2, m.dR, m.heavy_mass, b, logR),
        ):
            if axis == 1:
                d = -1j*derivative(u, spacing, 1)-vector[:, core][None]*u
                square = covariant_square(u, vector[:, core][None], spacing, 1, -1)/(2*mass)
            else:
                d = -1j*derivative(phi, spacing, 2)[:, :, 2:-2]-vector[:, core][None]*u
                square = covariant_square(phi, vector[:, ids][None], spacing, 2, -1)[:, :, 2:-2]/(2*mass)
            ratio = (vector[:, core]-1j*log[:, core])[None]*d/mass
            store(prefix+'_square', square)
            store(prefix+'_ratio', ratio)
            ue += square+ratio
            del d, square, ratio
        store('Ue_total', ue)
        if electronic_method == 'spectral':
            modes = np.arange(1, nx+1)*np.pi/((nx+1)*m.dx)
            hbo = dst(dst(u, type=1, axis=0, norm='ortho')*
                      (modes**2/2)[:, None, None], type=1, axis=0, norm='ortho')
            hbo += (m.potential[:, :, core]-trap[core])*u
        else:
            from types import SimpleNamespace
            hbo = apply_electronic_hamiltonian(u, SimpleNamespace(
                dx=m.dx, potential=m.potential[:, :, core]-trap[core]))
        store('hBO', hbo)
        hel = hbo+ue
        store('Hel', hel)
        # Exact PG temporal mean for the supplied state and native action.
        gd = -np.sum(u.conj()*_ratio(action, F[:, core][None]), axis=0).real*m.dx
        eps1 = means[NAMES.index('Hel'), :, core].real+gd
        local_lam = lam[:, core]
        for name, val in (('pn_square', pn1[:, core]), ('pn_ratio', pn2[:, core]),
                          ('Up_total', pn1[:, core]+pn2[:, core]), ('Tp', tp[:, core]),
                          ('epsilon1', eps1*local_lam),
                          ('Hpr', tp[:, core]+eps1*local_lam+pn1[:, core]+pn2[:, core])):
            ratio = _ratio(val, local_lam)
            maps[NAMES.index(name), :, core] = np.abs(ratio)
            means[NAMES.index(name), :, core] = ratio
        # Compare differential full-H action to the supplied native action.
        diagnostic = (hbo+trap[core]*u)*F[:, core][None]
        diagnostic -= derivative(psi[:, :, 2:-2], m.dq, 1, order=2)/(2*m.proton_mass)
        diagnostic -= derivative(psi, m.dR, 2, order=2)[:, :, 2:-2]/(2*m.heavy_mass)
        closure_error[:, core] = np.sum(np.abs(diagnostic-action)**2, axis=0)*m.dx
    valid = (rho >= density_floor) & np.all(np.isfinite(maps), axis=0) & np.all(np.isfinite(means), axis=0)
    weight = np.where(valid, rho, 0)
    support_mass = weight.sum()*m.dq*m.dR
    if support_mass <= 0:
        raise ValueError('No finite occupied support for coupling diagnostics')
    w = weight/weight.sum()
    summary = dict(
        support_mass=float(support_mass), norm=float(rho.sum()*m.dq*m.dR),
        rms=np.sqrt(np.sum(w[None]*np.where(valid[None], maps, 0)**2, axis=(1,2))),
        mean_real=np.sum(w[None]*np.where(valid[None], means.real, 0), axis=(1,2)),
        mean_imag=np.sum(w[None]*np.where(valid[None], means.imag, 0), axis=(1,2)),
        native_action_difference=float(np.sqrt(np.sum(np.where(valid, closure_error, 0))*m.dq*m.dR/support_mass)),
    )
    return dict(magnitude=maps[:, ::map_stride, ::map_stride],
                expectation_real=means.real[:, ::map_stride, ::map_stride],
                expectation_imag=means.imag[:, ::map_stride, ::map_stride],
                rho=rho[::map_stride, ::map_stride],
                delta=delta[::map_stride, ::map_stride],
                valid=valid[::map_stride, ::map_stride], **summary)


class CoupledActionRecorder:
    """Optional disk-staged full-Psi diagnostic. Does not alter propagation."""
    def __init__(self, outdir, frames, model, *, block_R=4, map_stride=2,
                 electronic_method='finite', source='full_Psi'):
        self.model, self.block, self.stride = model, block_R, map_stride
        self.method, self.source = electronic_method, source
        self.outdir = Path(outdir)
        self.final = self.outdir/'coupled_action_diagnostics.npz'
        if self.final.exists():
            raise FileExistsError(f'Use a fresh output directory: {self.final}')
        shape = (frames, len(model.q[::map_stride]), len(model.R[::map_stride]))
        required = np.prod(shape)*(3*len(NAMES)+2)*8+np.prod(shape)
        if shutil.disk_usage(self.outdir).free < 2*required:
            raise OSError(f'Coupling diagnostic needs about {2*required/1024**3:.2f} GiB staging+output headroom')
        self.arrays, self.paths, self.rows, self.times = {}, [], [], []
        for key, sh, dtype in [('magnitude',(frames,len(NAMES),*shape[1:]),float),
                               ('expectation_real',(frames,len(NAMES),*shape[1:]),float),
                               ('expectation_imag',(frames,len(NAMES),*shape[1:]),float),
                               ('rho',shape,float),('delta',shape,float),('valid',shape,bool)]:
            path = self.outdir/f'.coupled_{key}.partial.npy'
            if path.exists():
                raise FileExistsError(path)
            self.paths.append(path)
            self.arrays[key] = np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=sh)

    def save(self, time_fs, read_psi, read_action):
        result = action_frame(read_psi, read_action, self.model, self.block,
                              electronic_method=self.method, map_stride=self.stride)
        f = len(self.times)
        for key, array in self.arrays.items():
            array[f] = result.pop(key)
            array.flush()
        self.rows.append(result)
        self.times.append(time_fs)

    def finish(self):
        n = len(self.times)
        if not n:
            return
        payload = {k: a[:n] for k,a in self.arrays.items()}
        payload.update({k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]})
        payload.update(times_fs=np.asarray(self.times), q=self.model.q[::self.stride],
                       R=self.model.R[::self.stride], names=np.array(NAMES),
                       source=np.array(self.source), gauge=np.array('positive_density'),
                       method=np.array('central5_continuum_actions_native_temporal_term'),
                       electronic_method=np.array(self.method), density_floor=np.array(1e-3),
                       map_stride=np.array(self.stride))
        np.savez_compressed(self.final, **payload)
        for a in self.arrays.values():
            a._mmap.close()
        for path in self.paths:
            path.unlink()
        print(f'Coupled-action diagnostics saved: {self.final}', flush=True)
