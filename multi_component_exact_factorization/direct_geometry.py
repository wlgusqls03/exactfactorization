"""Saved-frame EF diagnostics from derivatives of the coherent wavefunction.

No BO overlaps/links are used. q/R use periodic fourth-order central first
derivatives, matching the nuclear grid topology. Undefined nodal values are
NaN, never silently filled with a physical zero. These diagnostics do not
replace the native finite-link Hamiltonian or its scalar decomposition.
"""
from pathlib import Path
import shutil
import time
import numpy as np


def derivative5(value, spacing, axis):
    return (np.roll(value, 2, axis=axis)-8*np.roll(value, 1, axis=axis)
            +8*np.roll(value, -1, axis=axis)-np.roll(value, -2, axis=axis))/(12*spacing)


def direct_geometry_frame(read_R, shape, dx, dq, dR, proton_mass, heavy_mass, block_R=8):
    """read_R(indices) supplies coherent Psi(x,q,R_indices), including halos.

    Normalizing Psi directly includes all spatial BO-basis derivatives, not
    just coefficient derivatives. Spectral runs supply the FULL Psi, not a
    BO projection. Block boundaries use two exact periodic halo cells.
    """
    nx, nq, nr = shape
    if min(nq, nr) < 5 or block_R < 1:
        raise ValueError('direct geometry requires >=5 q/R cells and positive block size')
    if min(dx, dq, dR, proton_mass, heavy_mass) <= 0:
        raise ValueError('spacings and masses must be positive')
    result = {key: np.empty((nq, nr)) for key in ('direct_geo1_q', 'direct_geo1_R')}
    result.update({key: np.empty(nr) for key in ('direct_geo2_R', 'direct_internal_q_kinetic')})
    tiny = np.finfo(float).tiny
    for start in range(0, nr, block_R):
        stop = min(start+block_R, nr)
        ids = np.arange(start-2, stop+2) % nr
        psi = np.asarray(read_R(ids), dtype=np.complex128)
        if psi.shape != (nx, nq, len(ids)) or not np.isfinite(psi).all():
            raise ValueError('invalid coherent Psi block')
        rho = np.sum(np.abs(psi)**2, axis=0)*dx
        heavy = rho.sum(axis=0)*dq
        phi = np.divide(psi, np.sqrt(rho)[None], out=np.full_like(psi, np.nan),
                        where=rho[None] > tiny)
        gamma = np.divide(psi, np.sqrt(heavy)[None, None], out=np.full_like(psi, np.nan),
                          where=heavy[None, None] > tiny)
        core = slice(2, -2)
        def dR_core(a):
            return (a[:, :, :-4]-8*a[:, :, 1:-3]+8*a[:, :, 3:-1]-a[:, :, 4:])/(12*dR)
        # Project derivatives orthogonally to normalized conditional states:
        # ||(1-|u><u|)du||^2 is nonnegative without cancellation/clipping.
        for name, derivative, mass in (
                ('direct_geo1_q', derivative5(phi, dq, 1)[:, :, core], proton_mass),
                ('direct_geo1_R', dR_core(phi), heavy_mass)):
            u = phi[:, :, core]
            inner = np.sum(u.conj()*derivative, axis=0)*dx
            perpendicular = derivative-u*inner[None]
            result[name][:, start:stop] = np.sum(np.abs(perpendicular)**2, axis=0)*dx/(2*mass)
        derivative = dR_core(gamma)
        u = gamma[:, :, core]
        inner = np.sum(u.conj()*derivative, axis=(0, 1))*dx*dq
        result['direct_geo2_R'][start:stop] = np.sum(
            np.abs(derivative-u*inner[None, None])**2, axis=(0, 1))*dx*dq/(2*heavy_mass)
        # The second-level internal q term is kinetic, NOT a pure metric.
        result['direct_internal_q_kinetic'][start:stop] = np.sum(
            np.abs(derivative5(gamma, dq, 1)[:, :, core])**2,
            axis=(0, 1))*dx*dq/(2*proton_mass)
    return result


class GeometryRecorder:
    """Disk-staged arrays; no growing geometry history in RAM."""
    def __init__(self, outdir, frames, model, block_R=8):
        self.model, self.block_R = model, block_R
        self.count, self.seconds = 0, 0.
        self.arrays, self.paths = {}, []
        required = frames*len(model.R)*(2*len(model.q)+2)*8
        if shutil.disk_usage(outdir).free < required:
            raise OSError(f'Direct geometry staging needs {required/1024**3:.2f} GiB; use a larger output filesystem')
        print(f'Direct geometry staging: {required/1024**3:.2f} GiB raw; final NPZ needs additional space')
        for key in ('direct_geo1_q', 'direct_geo1_R', 'direct_geo2_R', 'direct_internal_q_kinetic'):
            shape = ((frames, len(model.q), len(model.R)) if key.startswith('direct_geo1')
                     else (frames, len(model.R)))
            path = Path(outdir)/('.'+key+'.partial.npy')
            # Never overwrite another run's recoverable staging files.
            if path.exists():
                raise FileExistsError(f'Use a fresh outdir: {path}')
            self.paths.append(path)
            self.arrays[key] = np.lib.format.open_memmap(path, mode='w+', dtype=float, shape=shape)

    def save(self, read_R):
        started = time.perf_counter()
        m = self.model
        fields = direct_geometry_frame(read_R, (len(m.x), len(m.q), len(m.R)),
            m.dx, m.dq, m.dR, m.proton_mass, m.heavy_mass, self.block_R)
        for key, value in fields.items():
            self.arrays[key][self.count] = value
            self.arrays[key].flush()
        self.count += 1
        self.seconds += time.perf_counter()-started

    def payload(self):
        return {**{k: a[:self.count] for k, a in self.arrays.items()},
                'direct_geometry_method': np.array('coherent_Psi_PG_periodic_central5_no_overlap'),
                'direct_geometry_node_policy': np.array('NaN_where_conditional_state_or_stencil_undefined'),
                'direct_geometry_scalar_policy': np.array('independent_diagnostics_not_native_total_replacement'),
                'direct_internal_q_definition': np.array('integral_dx_dq_abs_dq_Gamma_squared_over_2mp_not_pure_metric'),
                'direct_geometry_seconds': np.array(self.seconds)}

    def cleanup(self):
        # Called only after the complete NPZ was successfully written.
        for a in self.arrays.values():
            a._mmap.close()
        for path in self.paths:
            path.unlink()
