"""Formula-level B1 tests; short arrays, no production propagation."""
import argparse
import unittest

import numpy as np

from multi_component_exact_factorization import core as original
from multi_component_exact_factorization.gaussian_trajectory.models import (
    InternalModel, ModelConfig, erf_inverse_gradient)


class ModelTests(unittest.TestCase):
    def normalized_random(self, model, count=3):
        rng = np.random.default_rng(51)
        G = rng.normal(size=(count, len(model.q), model.nk))
        G = G+1j*rng.normal(size=G.shape)
        norms = np.einsum('iqk,q->i', abs(G)**2, model.quadrature)
        return G/np.sqrt(norms[:, None, None])

    def test_action_hermitian_and_matched_operator_force(self):
        configs = [ModelConfig(), ModelConfig(name='pen-rabi', dse=True),
                   ModelConfig(name='mcef', nx=9, nq=32)]
        for config in configs:
            with self.subTest(model=config.name, dse=config.dse):
                m = InternalModel(config)
                R = m.initial_R_center+np.array([-.2, 0., .3])
                G = self.normalized_random(m)
                F = self.normalized_random(m)*np.exp(.3j*m.q[None, :, None])
                inner = lambda a, b: np.einsum('iqk,iqk,q->i', a.conj(), b, m.quadrature)
                np.testing.assert_allclose(inner(F, m.action(R, G)),
                                           inner(m.action(R, F), G), atol=5e-12)
                h = 2e-5
                energy = lambda x: inner(G, m.action(x, G)).real+m.external_potential(x)
                numeric = -(energy(R+h)-energy(R-h))/(2*h)
                np.testing.assert_allclose(m.force(R, G), numeric, atol=2e-9, rtol=2e-7)
                self.assertLess(np.max(abs(2*inner(G, -1j*m.action(R, G)).real)), 2e-11)

    def test_vacuum_norm_and_photon_phase_information(self):
        for dse in (False, True):
            m = InternalModel(ModelConfig(dse=dse))
            R = np.array([1.8, 2., 2.2])
            G = m.initial(R)
            obs = m.observables(R, G)
            np.testing.assert_allclose(obs['norm'], 1., atol=3e-15)
            np.testing.assert_allclose(obs['bo_population'][:, 1], 1., atol=4e-15)
            np.testing.assert_allclose(obs['photon_number'], 0., atol=5e-10)
            self.assertLess(np.max(obs['q_edge_probability']), 1e-10)
            boost = 2*np.pi/(m.config.qmax-m.config.qmin)
            boosted = G*np.exp(1j*boost*m.q[None, :, None])
            moved = m.observables(R, boosted)
            np.testing.assert_allclose(abs(boosted)**2, abs(G)**2, atol=1e-16)
            np.testing.assert_allclose(moved['photon_number']-obs['photon_number'],
                                       boost**2/(2*m.omega), atol=1e-10)
            np.testing.assert_allclose(moved['p_mean'], boost, atol=1e-10)

    def test_mcef_potential_matches_existing_model_not_frozen_nucleus(self):
        cfg = ModelConfig(name='mcef', nx=9, nq=32)
        m = InternalModel(cfg)
        parser = argparse.ArgumentParser()
        original.add_model_arguments(parser)
        args = parser.parse_args(['--nx', '9', '--nq', '32', '--nR', '2',
                                  '--x-min', str(cfg.xmin), '--x-max', str(cfg.xmax),
                                  '--q-min', str(cfg.qmin), '--q-max', str(cfg.qmax),
                                  '--R-min', '9', '--R-max', '10',
                                  '--erf-r-qr', '1', '--heavy-trap-alpha', '.03'])
        old = original.build_model(args)
        expected = old.potential.transpose(2, 1, 0)
        actual = m.potential(old.R)+m.external_potential(old.R)[:, None, None]
        np.testing.assert_allclose(actual, expected, atol=3e-16)
        self.assertEqual(m.mass_R, old.heavy_mass)
        self.assertEqual(m.mass_q, old.proton_mass)
        G = m.initial(old.R)
        np.testing.assert_allclose(np.einsum('iqk,q->i', abs(G)**2, m.quadrature), 1., atol=1e-14)
        C = m.initial_coefficients(old.R)
        np.testing.assert_allclose(np.sum(abs(C)**2, axis=2), 1., atol=1e-14)
        self.assertFalse('photon_number' in m.observables(old.R, G))

    def test_erf_derivative_including_coincident_electron(self):
        x = np.array([-5., -.1, -1e-5, 0., 1e-5, .1, 5.])
        radius = 4.
        h = 1e-4
        numeric = (original.erf_inverse(x+h, radius)-original.erf_inverse(x-h, radius))/(2*h)
        np.testing.assert_allclose(erf_inverse_gradient(x, radius), numeric, atol=5e-11)
        self.assertEqual(erf_inverse_gradient(np.array([0.]), radius)[0], 0.)

    def test_invalid_model_and_photon_parameters_fail(self):
        for cfg in [ModelConfig(name='tsh'), ModelConfig(nq=7), ModelConfig(omega=0.),
                    ModelConfig(dse='off'), ModelConfig(nq=32.5), ModelConfig(qmax=-15.)]:
            with self.subTest(config=cfg), self.assertRaises((ValueError, KeyError)):
                InternalModel(cfg)


if __name__ == '__main__':
    unittest.main()
