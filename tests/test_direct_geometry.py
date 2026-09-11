import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from pathlib import Path
import numpy as np

from multi_component_exact_factorization.direct_geometry import direct_geometry_frame, GeometryRecorder


class DirectGeometryTests(unittest.TestCase):
    def calculate(self, psi, block=3):
        return direct_geometry_frame(lambda ids: psi[:, :, ids], psi.shape,
                                     1., 2*np.pi/psi.shape[1], 2*np.pi/psi.shape[2], 2., 3., block)

    def test_rotating_conditional_state_matches_analytic_stencil(self):
        n = 32
        q = np.arange(n)*2*np.pi/n
        angle = q[:, None]+q[None, :]
        psi = np.array([np.cos(angle), np.sin(angle)], dtype=complex)
        original = psi.copy()
        fields = self.calculate(psi)
        h = 2*np.pi/n
        symbol = (8*np.sin(h)-np.sin(2*h))/(6*h)
        np.testing.assert_allclose(fields['direct_geo1_q'], symbol**2/4, atol=1e-14)
        np.testing.assert_allclose(fields['direct_geo1_R'], symbol**2/6, atol=1e-14)
        np.testing.assert_allclose(fields['direct_geo2_R'], symbol**2/6, atol=1e-14)
        np.testing.assert_allclose(fields['direct_internal_q_kinetic'], symbol**2/4, atol=1e-14)
        for key, value in self.calculate(psi, block=32).items():
            np.testing.assert_allclose(fields[key], value, atol=1e-14)
        np.testing.assert_array_equal(psi, original)

    def test_proton_motion_is_kinetic_not_electronic_geometry(self):
        q = np.arange(32)*2*np.pi/32
        psi = np.broadcast_to(np.exp(1j*q)[None, :, None], (1, 32, 16)).copy()
        fields = self.calculate(psi)
        np.testing.assert_allclose(fields['direct_geo1_q'], 0., atol=1e-28)
        self.assertTrue(np.all(fields['direct_internal_q_kinetic'] > .2))
        np.testing.assert_allclose(fields['direct_geo2_R'], 0., atol=1e-28)

    def test_direct_derivative_converges_to_continuum_metric(self):
        errors = []
        for n in (16, 32):
            grid = np.arange(n)*2*np.pi/n
            angle = grid[:, None]+grid[None, :]
            fields = self.calculate(np.array([np.cos(angle), np.sin(angle)]))
            errors.append(np.max(np.abs(fields['direct_geo2_R']-1/6)))
        self.assertGreater(errors[0]/errors[1], 14.)

    def test_coherent_bo_reader_includes_varying_basis(self):
        n = 16
        angle = np.arange(n)[:, None]*2*np.pi/n+np.zeros((n, n))
        states = np.array([np.cos(angle), np.sin(angle)])[None]
        coefficients = np.ones((1, n, n), complex)
        psi = np.einsum('nqR,nxqR->xqR', coefficients, states)
        fields = direct_geometry_frame(lambda ids: np.einsum('nqR,nxqR->xqR',
            coefficients[:, :, ids], states[:, :, :, ids]), psi.shape, 1., 2*np.pi/n, 2*np.pi/n, 2., 3.)
        np.testing.assert_allclose(fields['direct_geo1_q'], self.calculate(psi)['direct_geo1_q'])
        self.assertGreater(fields['direct_geo1_q'].min(), .2)

    def test_nodes_are_explicitly_undefined(self):
        psi = np.ones((2, 8, 8), complex)
        psi[:, 3, :] = 0
        fields = self.calculate(psi)
        self.assertTrue(np.isnan(fields['direct_geo1_q'][3]).all())
        self.assertTrue(np.isfinite(fields['direct_geo2_R']).all())

    def test_staged_saved_frames_round_trip_and_cleanup(self):
        with TemporaryDirectory() as directory:
            m = SimpleNamespace(x=np.arange(2), q=np.arange(8), R=np.arange(8),
                                dx=1., dq=1., dR=1., proton_mass=2., heavy_mass=3.)
            recorder = GeometryRecorder(directory, 3, m, 2)
            psi = np.ones((2, 8, 8), complex)
            recorder.save(lambda ids: psi[:, :, ids])
            recorder.save(lambda ids: psi[:, :, ids]*2)
            path = Path(directory)/'result.npz'
            np.savez_compressed(path, **recorder.payload())
            recorder.cleanup()
            with np.load(path) as stored:
                self.assertEqual(stored['direct_geo1_q'].shape, (2, 8, 8))
                np.testing.assert_allclose(stored['direct_geo2_R'], 0., atol=1e-28)
            self.assertFalse(list(Path(directory).glob('*.partial.npy')))


if __name__ == '__main__':
    unittest.main()
