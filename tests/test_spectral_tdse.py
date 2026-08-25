import unittest
from types import SimpleNamespace

import numpy as np
from scipy.fft import idst

from multi_component_exact_factorization.spectral_tdse import (
    spectral_energy_numpy,
    spectral_action_numpy,
    spectral_kinetic_energies,
    split_step_numpy,
)


def _model():
    nx, nq, nR = 8, 7, 6
    x_left, x_right = -2.0, 2.0
    dx = (x_right-x_left)/(nx+1)
    x = x_left+dx*np.arange(1, nx+1)
    q = np.arange(nq)*0.4-1.2
    R = np.arange(nR)*0.3+0.2
    potential = (
        0.07*x[:, None, None]**2
        +0.03*np.cos(q)[None, :, None]
        +0.05*np.sin(R)[None, None, :]
        +0.02*x[:, None, None]*q[None, :, None]
    )
    return SimpleNamespace(
        x=x, q=q, R=R, dx=dx, dq=0.4, dR=0.3,
        x_left=x_left, x_right=x_right,
        proton_mass=5.0, heavy_mass=13.0,
        potential=potential,
    )


class SpectralTDSETests(unittest.TestCase):
    def test_free_spectral_mode_has_exact_phase(self):
        model = _model()
        model.potential = np.zeros_like(model.potential)
        tx, tq, tR = spectral_kinetic_energies(model)
        electron_mode = np.zeros(len(model.x))
        electron_mode[2] = 1.0
        electron = idst(electron_mode, type=1, norm="ortho")
        iq, iR = 2, 1
        q_mode = np.exp(2j*np.pi*iq*np.arange(len(model.q))/len(model.q))
        R_mode = np.exp(2j*np.pi*iR*np.arange(len(model.R))/len(model.R))
        psi = electron[:, None, None]*q_mode[None, :, None]*R_mode[None, None, :]
        dt = 0.037
        stepped = split_step_numpy(psi, dt, model)
        expected = psi*np.exp(-1j*dt*(tx[2]+tq[iq]+tR[iR]))
        self.assertLess(np.max(np.abs(stepped-expected)), 2.0e-14)

    def test_split_step_is_unitary_and_time_reversible(self):
        model = _model()
        rng = np.random.default_rng(714)
        psi = rng.normal(size=model.potential.shape)+1j*rng.normal(
            size=model.potential.shape
        )
        cell = model.dx*model.dq*model.dR
        psi /= np.sqrt(np.sum(np.abs(psi)**2)*cell)
        stepped = split_step_numpy(psi, 0.023, model)
        norm = np.sum(np.abs(stepped)**2)*cell
        recovered = split_step_numpy(stepped, -0.023, model)
        self.assertLess(abs(norm-1.0), 3.0e-15)
        self.assertLess(np.max(np.abs(recovered-psi)), 3.0e-14)

    def test_global_time_error_is_second_order(self):
        model = _model()
        rng = np.random.default_rng(827)
        psi0 = rng.normal(size=model.potential.shape)+1j*rng.normal(
            size=model.potential.shape
        )

        def evolve(dt, steps):
            values = psi0.copy()
            for _ in range(steps):
                values = split_step_numpy(values, dt, model)
            return values

        coarse = evolve(0.02, 4)
        medium = evolve(0.01, 8)
        reference = evolve(0.00125, 64)
        error_coarse = np.linalg.norm(coarse-reference)
        error_medium = np.linalg.norm(medium-reference)
        self.assertGreater(error_coarse/error_medium, 3.7)

    def test_parseval_energy_is_real_and_finite(self):
        model = _model()
        psi = np.ones(model.potential.shape, dtype=complex)
        result = spectral_energy_numpy(psi, model)
        for value in result.values():
            self.assertTrue(np.isfinite(value))
            self.assertIsInstance(value, float)
        action = spectral_action_numpy(psi, model)
        expectation = (
            np.vdot(psi, action)*model.dx*model.dq*model.dR
        )
        self.assertLess(abs(expectation.imag), 2.0e-13)
        self.assertAlmostEqual(expectation.real, result["energy"], places=12)


if __name__ == "__main__":
    unittest.main()
