"""Explicit spatial-operator choices: formula consistency, not long-time proof."""
import unittest
from unittest.mock import patch

import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.internal_channels import InternalChannels
from multi_component_exact_factorization.gaussian_trajectory.models import InternalModel, ModelConfig
from multi_component_exact_factorization.gaussian_trajectory.q_operators import (
    kinetic_action, kinetic_matrix, kinetic_symbol, periodic_derivative)


class PeriodicQOperatorTests(unittest.TestCase):
    def test_finite_difference_matrices_match_explicit_periodic_stencils(self):
        n, h, mass = 16, .3, 1.7
        coefficients = {'fd2': {0: -2., -1: 1., 1: 1.},
                        'fd4': {0: -30./12., -1: 16./12., 1: 16./12., -2: -1./12., 2: -1./12.}}
        for operator, stencil in coefficients.items():
            with self.subTest(operator=operator):
                expected = np.zeros((n, n))
                for i in range(n):
                    for offset, coefficient in stencil.items():
                        expected[i, (i+offset)%n] = -coefficient/(2*mass*h*h)
                actual = kinetic_matrix(n, h, mass, operator=operator)
                np.testing.assert_allclose(actual, expected, atol=4e-15)
                np.testing.assert_array_equal(actual, actual.T)
                np.testing.assert_allclose(np.linalg.eigvalsh(actual),
                    np.sort(kinetic_symbol(n, h, mass, operator=operator)), atol=4e-14)
                self.assertGreater(np.linalg.eigvalsh(actual)[0], -4e-14)

    def test_fd_action_is_local_and_does_not_call_fft(self):
        signal = np.zeros((2, 32, 3), dtype=complex)
        signal[:, 15, :] = 1.+.4j
        for operator, radius in [('fd2', 1), ('fd4', 2)]:
            with self.subTest(operator=operator), patch('numpy.fft.fft', side_effect=AssertionError('FD must be local')):
                action = kinetic_action(signal, .4, 1., operator=operator)
                first = periodic_derivative(signal, .4, operator=operator)
                allowed = np.arange(15-radius, 16+radius)
                off = np.setdiff1d(np.arange(32), allowed)
                np.testing.assert_array_equal(action[:, off], 0.)
                np.testing.assert_array_equal(first[:, off], 0.)

    def test_first_derivative_antihermitian_and_not_kinetic_square(self):
        for operator in ('fd2', 'fd4'):
            with self.subTest(operator=operator):
                D = periodic_derivative(np.eye(16), .2, operator=operator, axis=0)
                np.testing.assert_array_equal(D, -D.T)
                T = kinetic_matrix(16, .2, 1., operator=operator)
                self.assertGreater(np.max(abs(D.T@D-2*T)), 1.)

    def test_model_action_internal_channels_and_operator_force_agree(self):
        rng = np.random.default_rng(419)
        for name in ('pen-nonadiabatic', 'mcef'):
            for operator in ('fft', 'fd2', 'fd4'):
                with self.subTest(model=name, operator=operator):
                    model = InternalModel(ModelConfig(name=name, nq=16, nx=4,
                        qmin=-14. if name.startswith('pen') else -6.,
                        qmax=14. if name.startswith('pen') else -2., q_operator=operator))
                    R = model.initial_R_center+np.array([-.07, .12])
                    G = rng.normal(size=(2, 16, model.nk))+1j*rng.normal(size=(2, 16, model.nk))
                    norm = np.einsum('q,iqk->i', model.quadrature, abs(G)**2)
                    G /= np.sqrt(norm)[:, None, None]
                    adapter = InternalChannels(model)
                    H, dH = adapter.matrices(R)
                    weight = np.sqrt(model.quadrature)[None, :, None]
                    expected = np.einsum('ijk,ik->ij', H, (G*weight).reshape(2, -1))
                    np.testing.assert_allclose(expected, (model.action(R, G)*weight).reshape(2, -1), atol=2e-14)
                    np.testing.assert_allclose(H, H.conj().transpose(0, 2, 1), atol=2e-14)
                    eigen = adapter.evaluate(R)
                    np.testing.assert_allclose(np.einsum('ijk,ikn->ijn', H, eigen.vectors),
                        eigen.vectors*eigen.energy[:, None, :], atol=4e-14)
                    step = 1e-5
                    energy = lambda r: np.einsum('q,iqk,iqk->i', model.quadrature,
                        G.conj(), model.action(r, G)).real+model.external_potential(r)
                    numeric_force = -(energy(R+step)-energy(R-step))/(2*step)
                    np.testing.assert_allclose(model.force(R, G), numeric_force, atol=2e-9)

    def test_default_fft_action_and_existing_observable_formulas_unchanged(self):
        model = InternalModel(ModelConfig(nq=32))
        R = np.array([1.8, 2.2])
        G = model.initial(R)*np.exp(.3j*model.q[None, :, None])
        # Existing implementation forms the scalar symbol before its product.
        old_kinetic = np.fft.ifft(np.fft.fft(G, axis=1)*(model.kq**2/(2*model.mass_q))[None, :, None], axis=1)
        old_action = old_kinetic+np.einsum('iqkl,iql->iqk', model.potential(R), G)
        np.testing.assert_array_equal(model.action(R, G), old_action)
        old_D = np.fft.ifft(np.fft.fft(G, axis=1)*(1j*model.kq[None, :, None]), axis=1)
        np.testing.assert_array_equal(model.derivative_q(G), old_D)
        old_p2 = np.einsum('q,iqk,iqk->i', model.quadrature, old_D.conj(), old_D).real
        np.testing.assert_allclose(model.observables(R, G)['p2'], old_p2, atol=3e-16)

    def test_harmonic_grid_convergence_and_unsubtracted_vacuum_bias(self):
        for operator, minimum_ratio in [('fd2', 3.5), ('fd4', 12.)]:
            errors, vacuum_errors = [], []
            for nq in (32, 64, 128):
                model = InternalModel(ModelConfig(nq=nq, q_operator=operator))
                H = model.kinetic_q_matrix()+np.diag(.5*model.omega**2*model.q**2)
                eigen = np.linalg.eigvalsh(H)[:3]
                errors.append(float(np.max(abs(eigen-model.omega*(np.arange(3)+.5)))))
                audit = model.photon_discretization_audit()
                vacuum_errors.append(abs(audit['sampled_vacuum_photon_number']))
                R = np.array([2.])
                G = model.initial(R)
                observable = model.observables(R, G)
                self.assertLess(observable['photon_number'][0], 0.)
                self.assertAlmostEqual(observable['photon_number'][0], audit['sampled_vacuum_photon_number'], places=13)
                p2 = 2*model.mass_q*np.einsum('q,iqk,iqk->i', model.quadrature,
                    G.conj(), model.kinetic_q_action(G)).real
                np.testing.assert_allclose(observable['p2'], p2, atol=5e-14)
                self.assertIn('no offset subtraction', audit['reference'])
                self.assertEqual(model.metadata()['photon_discretization']['q_operator'], operator)
            self.assertGreater(errors[0]/errors[1], minimum_ratio)
            self.assertGreater(errors[1]/errors[2], minimum_ratio)
            self.assertGreater(vacuum_errors[0]/vacuum_errors[1], minimum_ratio)
            self.assertGreater(vacuum_errors[1]/vacuum_errors[2], minimum_ratio)

    def test_invalid_operator_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, 'q_operator'):
            InternalModel(ModelConfig(q_operator='masked'))
        for kwargs in [dict(operator='bad'), dict(order=3)]:
            with self.assertRaises(ValueError):
                periodic_derivative(np.ones((2, 8)), .2, **kwargs)
        with self.assertRaises(ValueError):
            kinetic_action(np.ones((2, 8)), .2, 0., operator='fd4')


if __name__ == '__main__':
    unittest.main()
