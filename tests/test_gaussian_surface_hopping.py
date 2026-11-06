"""Formula/event regressions for the isolated internal-surface FSSH backend."""
import copy
import unittest

import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.surface_hopping import (
    HoppingStepError, apply_hops, coefficient_rhs, draw_hop_targets,
    hopping_probabilities, pair_population_flux, sample_active_states,
)


class SurfaceHoppingTests(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[np.sqrt(.7), np.sqrt(.3)]], complex)
        self.d = np.array([[[0., .2], [-.2, 0.]]], complex)
        self.active = np.array([0])

    def test_native_real_coupling_formula_and_sign(self):
        # G-CTMQC shopping.f90 TSHFS expression, without optional forced hops.
        probability = hopping_probabilities(self.A, self.d, 3., self.active, .1)
        expected = 2*.1*3*.2*np.real(self.A[0, 0].conjugate()*self.A[0, 1])/.7
        np.testing.assert_allclose(probability, [[0., expected]], atol=1e-15)
        np.testing.assert_array_equal(
            hopping_probabilities(self.A, self.d, -3., self.active, .1), [[0., 0.]])

    def test_pair_flux_reconstructs_exact_population_rate(self):
        rng = np.random.default_rng(19)
        A = rng.normal(size=(6, 4))+1j*rng.normal(size=(6, 4))
        A /= np.linalg.norm(A, axis=1)[:, None]
        B = rng.normal(size=(6, 4, 4))+1j*rng.normal(size=(6, 4, 4))
        d = B-B.conj().swapaxes(-1, -2)
        E = rng.normal(size=(6, 4))
        velocity = rng.normal(size=6)
        rate = coefficient_rhs(A, E, d, velocity)
        flux = pair_population_flux(A, d, velocity)
        np.testing.assert_allclose(-flux.sum(axis=2), 2*np.real(A.conj()*rate), atol=2e-14)
        np.testing.assert_allclose(flux, -flux.swapaxes(-1, -2), atol=2e-14)
        np.testing.assert_allclose(np.sum(2*np.real(A.conj()*rate), axis=1), 0, atol=2e-14)

    def test_complex_channel_gauge_invariance(self):
        theta = np.array([[.731, -1.956]])
        Anew = self.A*np.exp(-1j*theta)
        dnew = self.d*np.exp(1j*(theta[:, None, :]-theta[:, :, None]))
        # Diagonal derivative of the channel phase is a pure imaginary connection.
        dnew[:, 0, 0] += .7j
        dnew[:, 1, 1] -= 1.1j
        np.testing.assert_allclose(pair_population_flux(Anew, dnew, 1.),
                                   pair_population_flux(self.A, self.d, 1.), atol=1e-15)
        np.testing.assert_allclose(hopping_probabilities(Anew, dnew, 1., self.active, .1),
                                   hopping_probabilities(self.A, self.d, 1., self.active, .1))

    def test_overall_coefficient_scale_does_not_change_hops(self):
        np.testing.assert_allclose(
            hopping_probabilities(self.A*3, self.d, 2., self.active, .1),
            hopping_probabilities(self.A, self.d, 2., self.active, .1))

    def test_probability_above_one_requires_substep(self):
        with self.assertRaisesRegex(HoppingStepError, 'reduce dt'):
            hopping_probabilities(self.A, self.d, 30., self.active, 1.)
        with self.assertRaises(HoppingStepError):
            draw_hop_targets([[0, 1.01]], self.active, np.random.default_rng(1))

    def test_vanishing_active_population_is_not_floored(self):
        with self.assertRaisesRegex(HoppingStepError, 'Vanishing'):
            hopping_probabilities([[0, 1]], self.d, 1., self.active, .1)

    def test_uncoupled_and_zero_velocity_never_hop(self):
        for d, velocity in ((self.d*0, 1.), (self.d, 0.)):
            probability = hopping_probabilities(self.A, d, velocity, self.active, .1)
            np.testing.assert_array_equal(probability, [[0., 0.]])
            np.testing.assert_array_equal(draw_hop_targets(probability, self.active,
                                                          np.random.default_rng(2)), [0])

    def test_uphill_downhill_and_sign_preservation(self):
        P = np.array([2., -2., 2., -2.])
        E = np.tile([0., .25], (4, 1))
        active = np.array([0, 0, 1, 1])
        result = apply_hops(P, active, 1-active, E, 2.)
        np.testing.assert_array_equal(result.accepted, [True]*4)
        np.testing.assert_allclose(result.momentum, [np.sqrt(3), -np.sqrt(3), np.sqrt(5), -np.sqrt(5)])
        np.testing.assert_allclose(result.energy_residual, 0, atol=3e-16)
        np.testing.assert_array_equal(P, [2, -2, 2, -2])  # never mutate input

    def test_frustrated_keep_and_explicit_reverse(self):
        for policy, expected in (('keep', .2), ('reverse', -.2)):
            result = apply_hops([.2], self.active, np.array([1]), [[0, 1]], 1.,
                                frustrated_policy=policy)
            np.testing.assert_array_equal(result.frustrated, [True])
            np.testing.assert_array_equal(result.accepted, [False])
            np.testing.assert_array_equal(result.active, [0])
            np.testing.assert_allclose(result.momentum, [expected])
            np.testing.assert_array_equal(result.energy_residual, [0.])

    def test_threshold_hop_and_zero_momentum_policy(self):
        result = apply_hops([2.], self.active, np.array([1]), [[0, 1]], 2.)
        np.testing.assert_array_equal(result.accepted, [True])
        np.testing.assert_array_equal(result.momentum, [0.])
        with self.assertRaisesRegex(HoppingStepError, 'no declared direction'):
            apply_hops([0.], np.array([1]), self.active, [[0, 1]], 2.)
        # A zero-gap hop does not need an invented momentum direction.
        result = apply_hops([0.], self.active, np.array([1]), [[0, 0]], 2.)
        np.testing.assert_array_equal(result.accepted, [True])
        np.testing.assert_array_equal(result.momentum, [0.])

    def test_stochastic_initial_population(self):
        A = np.repeat(self.A, 20000, axis=0)
        original = A.copy()
        indices = sample_active_states(A, np.random.default_rng(156))
        self.assertLess(abs(np.mean(indices == 1)-.3), .012)
        np.testing.assert_array_equal(original, A)

    def test_rng_checkpoint_round_trip(self):
        rng = np.random.default_rng(2026)
        sample_active_states(self.A, rng)
        state = copy.deepcopy(rng.bit_generator.state)
        probability = np.tile([0, .4], (100, 1))
        active = np.zeros(100, dtype=int)
        first = draw_hop_targets(probability, active, rng)
        restored = np.random.default_rng()
        restored.bit_generator.state = state
        np.testing.assert_array_equal(first, draw_hop_targets(probability, active, restored))

    def test_short_manufactured_population_transfer(self):
        # Rotating two-channel basis with equal energies, d01=.7, v=1.
        # Exact A=(cos(.7t), sin(.7t)); no energetic forces/rescaling needed.
        # This checks a sequence of *real stochastic events*, not only a formula.
        count, dt = 2000, .01
        rng = np.random.default_rng(20261106)
        active, P = np.zeros(count, dtype=int), np.ones(count)
        d = np.tile([[[0, .7], [-.7, 0]]], (count, 1, 1))
        energy, accepted = np.zeros((count, 2)), 0
        for step in range(100):
            angle = .7*(step+.5)*dt
            A = np.tile([np.cos(angle), np.sin(angle)], (count, 1))
            probability = hopping_probabilities(A, d, P, active, dt)
            targets = draw_hop_targets(probability, active, rng)
            result = apply_hops(P, active, targets, energy, 1.)
            P, active = result.momentum, result.active
            accepted += np.count_nonzero(result.accepted)
            np.testing.assert_allclose(result.energy_residual, 0, atol=1e-15)
        self.assertGreater(accepted, 100)
        self.assertLess(abs(np.mean(active)-np.sin(.7)**2), .035)

    def test_bad_geometry_is_not_silently_fixed(self):
        with self.assertRaisesRegex(ValueError, 'anti-Hermitian'):
            pair_population_flux(self.A, [[[0, 1], [1, 0]]], 1.)
        with self.assertRaises(ValueError):
            hopping_probabilities(self.A, self.d, 1., [0.5], .1)
        with self.assertRaises(ValueError):
            apply_hops([1], self.active, np.array([1]), [[0, 1]], -1.)
        with self.assertRaises(ValueError):
            sample_active_states([[0, 0]], np.random.default_rng(5))


if __name__ == '__main__':
    unittest.main()
