"""Memory-bounded CUDA kernels for the full-grid spectral TDSE reference."""

from __future__ import annotations

import numpy as np

from multi_component_exact_factorization.spectral_tdse import (
    spectral_kinetic_energies,
)
from multi_component_exact_factorization_gpu.gpu_core import cp


_POTENTIAL_PHASE_KERNEL = None


def _potential_phase_kernel():
    global _POTENTIAL_PHASE_KERNEL
    if _POTENTIAL_PHASE_KERNEL is None:
        _POTENTIAL_PHASE_KERNEL = cp.RawKernel(r'''\
extern "C" __global__ void mcef_apply_real_diagonal_phase(
    double2* values, const double* diagonal, const double tau,
    const long long size
) {
    const long long index = (long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (index >= size) return;
    const double angle = tau*diagonal[index];
    double sine, cosine;
    sincos(angle, &sine, &cosine);
    const double2 value = values[index];
    values[index] = make_double2(
        value.x*cosine+value.y*sine,
        value.y*cosine-value.x*sine
    );
}
''', "mcef_apply_real_diagonal_phase")
    return _POTENTIAL_PHASE_KERNEL


def apply_real_diagonal_phase_inplace(values, diagonal, tau):
    """Apply ``exp(-i*tau*diagonal)`` without a full complex temporary."""
    if values.dtype != cp.complex128 or diagonal.dtype != cp.float64:
        raise TypeError("spectral TDSE phase kernel requires complex128/float64")
    if values.shape != diagonal.shape or not values.flags.c_contiguous:
        raise ValueError("phase arrays must be equal-shaped contiguous arrays")
    size = int(values.size)
    threads = 256
    _potential_phase_kernel()(
        ((size+threads-1)//threads,), (threads,),
        (values, diagonal, np.float64(tau), np.int64(size)),
    )


def _dst1_ortho(values):
    """Complex orthonormal DST-I through an odd-extension complex FFT."""
    length = int(values.shape[0])
    shape = (2*(length+1),)+values.shape[1:]
    extension = cp.zeros(shape, dtype=cp.complex128)
    extension[1:length+1] = values
    extension[length+2:] = -values[::-1]
    transformed = cp.fft.fft(extension, axis=0)
    # FFT[k] = -2 i sum_j x_j sin(pi*j*k/(N+1)).
    return (1j/np.sqrt(2.0*(length+1)))*transformed[1:length+1]


class SpectralTDSEGPU:
    """Second-order split operator with bounded transform workspaces.

    q transforms are blocked along R, R transforms along x, and the odd-
    extension electronic DST is blocked along R.  This keeps production
    memory near one full complex wavefunction plus the real potential instead
    of allocating several full 3D FFT temporaries.
    """

    def __init__(self, cpu_model, *, q_block_R=8, R_block_x=8, x_block_R=4):
        self.model = cpu_model
        self.potential = cp.ascontiguousarray(
            cp.asarray(cpu_model.potential, dtype=cp.float64)
        )
        tx, tq, tR = spectral_kinetic_energies(cpu_model)
        self.kinetic_x = cp.asarray(tx, dtype=cp.float64)
        self.kinetic_q = cp.asarray(tq, dtype=cp.float64)
        self.kinetic_R = cp.asarray(tR, dtype=cp.float64)
        self.q_block_R = max(1, int(q_block_R))
        self.R_block_x = max(1, int(R_block_x))
        self.x_block_R = max(1, int(x_block_R))
        self._phase_cache = {}

    def _phase(self, axis, tau):
        key = (axis, float(tau))
        phase = self._phase_cache.get(key)
        if phase is None:
            energies = {
                "x": self.kinetic_x,
                "q": self.kinetic_q,
                "R": self.kinetic_R,
            }[axis]
            phase = cp.exp(-1j*float(tau)*energies).astype(
                cp.complex128, copy=False
            )
            self._phase_cache[key] = phase
        return phase

    def apply_kinetic_inplace(self, wavefunction, tau):
        nx, _, nR = wavefunction.shape

        # Dirichlet electron kinetic: continuum sine modes, not a 5-point
        # finite-difference eigenvalue approximation.
        x_phase = self._phase("x", tau)[:, None, None]
        for start in range(0, nR, self.x_block_R):
            stop = min(start+self.x_block_R, nR)
            transformed = _dst1_ortho(wavefunction[:, :, start:stop])
            cp.multiply(transformed, x_phase, out=transformed)
            recovered = _dst1_ortho(transformed)
            wavefunction[:, :, start:stop] = recovered
            del transformed, recovered

        # Existing q/R arrays are periodic endpoint=False grids, so FFT
        # modes are the exact spectral representation for those grids.
        q_phase = self._phase("q", tau)[None, :, None]
        for start in range(0, nR, self.q_block_R):
            stop = min(start+self.q_block_R, nR)
            transformed = cp.fft.fft(
                wavefunction[:, :, start:stop], axis=1, norm="ortho"
            )
            cp.multiply(transformed, q_phase, out=transformed)
            recovered = cp.fft.ifft(transformed, axis=1, norm="ortho")
            wavefunction[:, :, start:stop] = recovered
            del transformed, recovered

        R_phase = self._phase("R", tau)[None, None, :]
        for start in range(0, nx, self.R_block_x):
            stop = min(start+self.R_block_x, nx)
            transformed = cp.fft.fft(
                wavefunction[start:stop], axis=2, norm="ortho"
            )
            cp.multiply(transformed, R_phase, out=transformed)
            recovered = cp.fft.ifft(transformed, axis=2, norm="ortho")
            wavefunction[start:stop] = recovered
            del transformed, recovered

    def step(self, wavefunction, dt):
        """In-place ``V/2 -> T -> V/2`` Strang split step."""
        apply_real_diagonal_phase_inplace(
            wavefunction, self.potential, 0.5*dt
        )
        self.apply_kinetic_inplace(wavefunction, dt)
        apply_real_diagonal_phase_inplace(
            wavefunction, self.potential, 0.5*dt
        )
        return wavefunction

    def energy(self, wavefunction):
        """Real Parseval energy without constructing ``H Psi``."""
        model = self.model
        cell = model.dx*model.dq*model.dR
        density = cp.real(wavefunction*cp.conj(wavefunction))
        norm = cp.sum(density, dtype=cp.float64)*cell
        potential = cp.sum(
            density*self.potential, dtype=cp.float64
        )*cell
        del density
        tx = cp.asarray(0.0, dtype=cp.float64)
        tq = cp.asarray(0.0, dtype=cp.float64)
        tR = cp.asarray(0.0, dtype=cp.float64)
        nx, _, nR = wavefunction.shape
        for start in range(0, nR, self.x_block_R):
            stop = min(start+self.x_block_R, nR)
            modes = _dst1_ortho(wavefunction[:, :, start:stop])
            tx += cp.sum(
                cp.real(modes*cp.conj(modes))*self.kinetic_x[:, None, None],
                dtype=cp.float64,
            )*cell
        for start in range(0, nR, self.q_block_R):
            stop = min(start+self.q_block_R, nR)
            modes = cp.fft.fft(
                wavefunction[:, :, start:stop], axis=1, norm="ortho"
            )
            tq += cp.sum(
                cp.real(modes*cp.conj(modes))*self.kinetic_q[None, :, None],
                dtype=cp.float64,
            )*cell
        for start in range(0, nx, self.R_block_x):
            stop = min(start+self.R_block_x, nx)
            modes = cp.fft.fft(
                wavefunction[start:stop], axis=2, norm="ortho"
            )
            tR += cp.sum(
                cp.real(modes*cp.conj(modes))*self.kinetic_R[None, None, :],
                dtype=cp.float64,
            )*cell
        return {
            "norm": norm, "kinetic_x": tx, "kinetic_q": tq,
            "kinetic_R": tR, "potential": potential,
            "energy": tx+tq+tR+potential,
        }

    def action(self, wavefunction):
        """Return the instantaneous full spectral ``H Psi``."""
        action = self.potential*wavefunction
        nx, _, nR = wavefunction.shape
        for start in range(0, nR, self.x_block_R):
            stop = min(start+self.x_block_R, nR)
            modes = _dst1_ortho(wavefunction[:, :, start:stop])
            cp.multiply(
                modes, self.kinetic_x[:, None, None], out=modes
            )
            action[:, :, start:stop] += _dst1_ortho(modes)
        for start in range(0, nR, self.q_block_R):
            stop = min(start+self.q_block_R, nR)
            modes = cp.fft.fft(
                wavefunction[:, :, start:stop], axis=1, norm="ortho"
            )
            cp.multiply(
                modes, self.kinetic_q[None, :, None], out=modes
            )
            action[:, :, start:stop] += cp.fft.ifft(
                modes, axis=1, norm="ortho"
            )
        for start in range(0, nx, self.R_block_x):
            stop = min(start+self.R_block_x, nx)
            modes = cp.fft.fft(
                wavefunction[start:stop], axis=2, norm="ortho"
            )
            cp.multiply(
                modes, self.kinetic_R[None, None, :], out=modes
            )
            action[start:stop] += cp.fft.ifft(
                modes, axis=2, norm="ortho"
            )
        return cp.ascontiguousarray(action)


def initialize_full_wavefunction_gpu(basis, excitation, marginal, block_R=8):
    """Reconstruct only the selected initial BO state in bounded blocks."""
    _, nx, nq, nR = basis.states.shape
    wavefunction = cp.empty((nx, nq, nR), dtype=cp.complex128)
    block_R = max(1, int(block_R))
    for start in range(0, nR, block_R):
        stop = min(start+block_R, nR)
        state = cp.asarray(
            np.asarray(basis.states[excitation, :, :, start:stop]),
            dtype=cp.float64,
        )
        amplitude = cp.asarray(
            marginal[:, start:stop], dtype=cp.complex128
        )
        wavefunction[:, :, start:stop] = state*amplitude[None, :, :]
    return cp.ascontiguousarray(wavefunction)


def project_full_wavefunction_to_bo(
    wavefunction, states, dx, *, block_R=4,
):
    """Project a full GPU wavefunction onto a disk-backed BO basis."""
    n_states, _, nq, nR = states.shape
    coefficients = np.empty((n_states, nq, nR), dtype=np.complex128)
    block_R = max(1, int(block_R))
    for start in range(0, nR, block_R):
        stop = min(start+block_R, nR)
        state_block = cp.asarray(
            np.asarray(states[:, :, :, start:stop]), dtype=cp.float64
        )
        projected = cp.einsum(
            "jxqr,xqr->jqr", cp.conj(state_block),
            wavefunction[:, :, start:stop], optimize=True,
        )*dx
        coefficients[:, :, start:stop] = cp.asnumpy(projected)
        del state_block, projected
    return coefficients


class SpectralConditionalAnalysisGPU:
    """Evaluate conditional electronic/q energies in the spectral algebra."""

    def __init__(self, cpu_model, states, *, block_R=2):
        self.model = cpu_model
        self.states = states
        self.block_R = max(1, int(block_R))
        self.potential = cp.ascontiguousarray(
            cp.asarray(cpu_model.potential, dtype=cp.float64)
        )
        tx, tq, _ = spectral_kinetic_energies(cpu_model)
        self.kinetic_x = cp.asarray(tx, dtype=cp.float64)
        self.kinetic_q = cp.asarray(tq, dtype=cp.float64)

    def energies(self, coefficients, lam, c_norm, lam_norm):
        """Return ``<Phi|H_e|Phi>(q,R)`` and ``<Gamma|T_q|Gamma>(R)``.

        Reconstruction is R-blocked.  It includes BO-state variation before
        applying the FFT q kinetic, so it does not invoke a product rule or a
        truncated neighbor stencil.
        """
        _, _, nq, nR = self.states.shape
        electronic = cp.empty((nq, nR), dtype=cp.complex128)
        proton_kinetic = cp.empty(nR, dtype=cp.float64)
        tiny = cp.asarray(1.0e-300, dtype=cp.float64)
        for start in range(0, nR, self.block_R):
            stop = min(start+self.block_R, nR)
            state_block = cp.asarray(
                np.asarray(self.states[:, :, :, start:stop]),
                dtype=cp.float64,
            )
            phi = cp.einsum(
                "jqR,jxqR->xqR", coefficients[:, :, start:stop],
                state_block, optimize=True,
            )
            x_modes = _dst1_ortho(phi)
            cp.multiply(
                x_modes, self.kinetic_x[:, None, None], out=x_modes
            )
            h_e_phi = _dst1_ortho(x_modes)
            h_e_phi += self.potential[:, :, start:stop]*phi
            electronic[:, start:stop] = cp.sum(
                cp.conj(phi)*h_e_phi, axis=0, dtype=cp.complex128
            )*self.model.dx/cp.maximum(c_norm[:, start:stop], tiny)

            gamma = phi*lam[None, :, start:stop]
            q_modes = cp.fft.fft(gamma, axis=1, norm="ortho")
            proton_kinetic[start:stop] = cp.sum(
                cp.real(q_modes*cp.conj(q_modes))
                *self.kinetic_q[None, :, None],
                axis=(0, 1), dtype=cp.float64,
            )*self.model.dx*self.model.dq/cp.maximum(
                lam_norm[start:stop], tiny
            )
            del state_block, phi, x_modes, h_e_phi, gamma, q_modes
        return electronic, proton_kinetic
