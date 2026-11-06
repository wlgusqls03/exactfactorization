"""Independent analytic tests for reference D1--D2 and Q1--Q10."""
import unittest

import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.mls import (
    MLSConfig, MLSFailure, build_mls_operator, reconstruct_R_derivatives)
from multi_component_exact_factorization.gaussian_trajectory.heavy_density import (
    DensityFailure, InterceptCancellationError, reconstruct_heavy_density,
    reconstruct_kde_density, two_channel_intercept_correction)


CONVENTIONS = {"basis_convention": "fixed primitive electronic basis",
               "gauge_convention": "one smooth common outer EF gauge"}


class MLSTests(unittest.TestCase):
    def setUp(self):
        self.r = np.array([-2., -1.73, -1.1, -.91, -.25, .1, .38, .77, 1.4, 1.9, 2.3])

    def test_complex_polynomial_factorial_and_scaling(self):
        for degree in (2, 3, 4):
            c = np.arange(1, degree + 2) + 1j * np.arange(degree + 1)
            values = np.polynomial.polynomial.polyval(self.r, c)
            d = np.polynomial.polynomial.polyval(self.r, np.polynomial.polynomial.polyder(c))
            d2 = np.polynomial.polynomial.polyval(self.r, np.polynomial.polynomial.polyder(c, 2))
            one, two, audit = reconstruct_R_derivatives(
                self.r, values, config=MLSConfig(degree=degree, neighbors=9), **CONVENTIONS)
            np.testing.assert_allclose(one, d, rtol=2e-12, atol=2e-11)
            np.testing.assert_allclose(two, d2, rtol=2e-12, atol=3e-11)
            self.assertLess(audit["local_fit_rms"].max(), 2e-12)

    def test_minimum_three_points_and_boundary(self):
        r = np.array([-.7, 0., 1.2])
        d, d2, _ = reconstruct_R_derivatives(r, r ** 2, config=MLSConfig(neighbors=3), **CONVENTIONS)
        np.testing.assert_allclose(d, 2*r, atol=1e-14)
        np.testing.assert_allclose(d2, 2., atol=1e-14)
        with self.assertRaises(MLSFailure):
            build_mls_operator(r, MLSConfig(neighbors=3, boundary="require_two_sided"))

    def test_constant_unitary_basis_covariance(self):
        y = np.stack([self.r + 1j*self.r**2, 2*self.r**2 - 1j, self.r**3 + 1], axis=-1)
        u, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3,3)) +
                            1j*np.random.default_rng(6).normal(size=(3,3)))
        op = build_mls_operator(self.r, MLSConfig(degree=3, neighbors=8))
        d, d2, _ = op.apply(y, **CONVENTIONS)
        du, d2u, _ = op.apply(y @ u, **CONVENTIONS)
        np.testing.assert_allclose(du, d @ u, atol=1e-13)
        np.testing.assert_allclose(d2u, d2 @ u, atol=1e-13)

    def test_physical_phase_is_not_aligned_away(self):
        r = np.linspace(-.1, .1, 31)
        y = np.exp(1.7j*r)
        d, d2, audit = reconstruct_R_derivatives(r, y, config=MLSConfig(degree=4), **CONVENTIONS)
        np.testing.assert_allclose(d, 1.7j*y, atol=1e-7)
        np.testing.assert_allclose(d2, -1.7**2*y, atol=4e-5)
        self.assertFalse(audit["phase_alignment_performed"])

    def test_geometry_current_positions_not_initial_order(self):
        permutation = np.random.default_rng(8).permutation(len(self.r))
        r = self.r[permutation]
        d, d2, _ = reconstruct_R_derivatives(r, r**2, **CONVENTIONS)
        np.testing.assert_allclose(d, 2*r, atol=2e-14)
        np.testing.assert_allclose(d2, 2., atol=2e-14)

    def test_complex_multiaxis_values(self):
        q = np.array([-.5, .2, .6])
        values = (self.r[:, None, None] ** 2 + q[None, :, None]) * np.array([1., 1j])[None,None,:]
        d, _, _ = reconstruct_R_derivatives(self.r, values, **CONVENTIONS)
        np.testing.assert_allclose(d, np.broadcast_to(2*self.r[:,None,None]*np.array([1.,1j]), values.shape), atol=1e-14)

    def test_geometry_failures_are_explicit(self):
        with self.assertRaises(MLSFailure):
            build_mls_operator(np.array([0., 0., 0., 1., 1.]))
        with self.assertRaises(MLSFailure):
            build_mls_operator(self.r, MLSConfig(radius=.001))
        with self.assertRaises(MLSFailure):
            build_mls_operator(self.r, MLSConfig(condition_limit=1.))
        _, _, audit = reconstruct_R_derivatives(self.r, self.r, config=MLSConfig(radius=.001, failure_policy="nan"), **CONVENTIONS)
        self.assertTrue(all(audit["errors"]))

    def test_duplicate_multistream_guard_and_report(self):
        r = np.array([-2., -1., 0., 0., 1., 2.])
        y = r**2 + 0j
        y[3] += 1j
        with self.assertRaises(MLSFailure):
            reconstruct_R_derivatives(r, y, **CONVENTIONS)
        _, _, audit = reconstruct_R_derivatives(r, y, config=MLSConfig(multistream_policy="report"), **CONVENTIONS)
        self.assertAlmostEqual(audit["duplicate_scatter"][0]["rms"], .5)
        self.assertGreater(audit["local_fit_rms"].max(), .1)

    def test_conventions_and_config_are_required(self):
        with self.assertRaises(ValueError):
            reconstruct_R_derivatives(self.r, self.r, basis_convention="", gauge_convention="unknown")
        with self.assertRaises(ValueError):
            build_mls_operator(self.r, MLSConfig(degree=1))

    def test_configurable_kernels_and_fixed_scale(self):
        for kernel in ("uniform", "gaussian", "tricube"):
            d, d2, _ = reconstruct_R_derivatives(self.r, self.r**2,
                config=MLSConfig(kernel=kernel, scale_policy="fixed", fixed_scale=5., balance="nearest"), **CONVENTIONS)
            np.testing.assert_allclose(d, 2*self.r, atol=3e-14)
            np.testing.assert_allclose(d2, 2., atol=3e-14)


class HeavyDensityTests(unittest.TestCase):
    def test_single_gaussian_moments_and_log_derivative(self):
        r = np.array([-1., 1.]) + 2.
        x = np.linspace(-2,6,41)
        rec = reconstruct_heavy_density(r, [.5,.5], [[1.],[1.]], x, channel_basis="one BO channel")
        np.testing.assert_allclose(rec.density, np.exp(-.5*(x-2)**2)/np.sqrt(2*np.pi))
        np.testing.assert_allclose(rec.quantum_momentum, (x-2)/2)
        self.assertEqual(rec.centers[0], 2.)
        self.assertEqual(rec.widths[0], 1.)

    def test_channel_weighting_and_posterior_not_population(self):
        r = np.array([-2., -1., 1., 3.])
        eta = np.array([.1, .2, .3, .4])
        p = np.array([[.9,.1],[.7,.3],[.2,.8],[.1,.9]])
        rec = reconstruct_heavy_density(r, eta, p, channel_basis="two internal channels")
        expected_weights = eta @ p
        np.testing.assert_allclose(rec.channel_weights, expected_weights)
        centers = np.sum(eta[:,None]*p*r[:,None], axis=0)/expected_weights
        np.testing.assert_allclose(rec.centers, centers)
        self.assertGreater(np.max(np.abs(rec.posterior-p)), .05)
        eps = 1e-5
        plus = reconstruct_heavy_density(r, eta, p, r+eps, channel_basis="two internal channels")
        minus = reconstruct_heavy_density(r, eta, p, r-eps, channel_basis="two internal channels")
        np.testing.assert_allclose(rec.quantum_momentum, -.5*(plus.log_density-minus.log_density)/(2*eps), atol=1e-10)

    def test_zero_channel_and_explicit_width_regularization(self):
        rec = reconstruct_heavy_density([-1.,1.], [.5,.5], [[1.,0.],[1.,0.]], channel_basis="declared")
        self.assertEqual(rec.audit["zero_population_channels"], [1])
        self.assertTrue(np.all(rec.posterior[:,1] == 0))
        with self.assertRaises(DensityFailure):
            reconstruct_heavy_density([-1.,1.], [.5,.5], np.eye(2), channel_basis="declared")
        regularized = reconstruct_heavy_density([-1.,1.], [.5,.5], np.eye(2), channel_basis="declared", width_floor=.2)
        self.assertEqual(regularized.audit["regularized_channels"], [0,1])
        np.testing.assert_equal(regularized.widths, .2)

    def test_log_domain_handles_far_tail_without_density_floor(self):
        rec = reconstruct_heavy_density([-1.,1.], [.5,.5], [[1.],[1.]], [100.], channel_basis="one")
        self.assertEqual(rec.density[0], 0.)
        self.assertAlmostEqual(rec.quantum_momentum[0], 50.)
        self.assertTrue(np.isfinite(rec.log_density[0]))
        self.assertEqual(rec.audit["underflow_points"], 1)

    def test_kde_is_labelled_differently(self):
        rec = reconstruct_kde_density([-1.,1.], [.5,.5], [0.], bandwidth=.3)
        self.assertEqual(rec.audit["mode"], "trajectory_kde")
        self.assertAlmostEqual(rec.quantum_momentum[0], 0.)

    def test_no_silent_renormalization(self):
        with self.assertRaises(ValueError):
            reconstruct_heavy_density([-1.,1.], [.3,.5], [[1.],[1.]], channel_basis="one")
        with self.assertRaises(ValueError):
            reconstruct_heavy_density([-1.,1.], [.5,.5], [[.3],[1.]], channel_basis="one")

    def test_intercept_corrects_ensemble_not_every_trajectory(self):
        r = np.array([-1.,.2,2.])
        eta = np.array([.2,.3,.5])
        p = np.array([[.3,.7],[.5,.5],[.8,.2]])
        f = np.array([[2.,0.],[1.,0.],[3.,0.]])
        result = two_channel_intercept_correction(r, eta, p, f, raw_quantum_momentum=r/2,
            slope=.5, uncoupled_channels=True, channel_basis="uncoupled states", mass=2.)
        self.assertLess(abs(result["corrected_ensemble_population_rate"]), 1e-16)
        self.assertGreater(abs(result["raw_ensemble_population_rate"]), .1)
        self.assertTrue(np.any(result["corrected"] != 0))
        np.testing.assert_array_equal(result["raw"], r/2)

    def test_intercept_cancellation_and_invalid_coupled_use(self):
        kwargs = dict(R=[-1.,1.], eta=[.5,.5], populations=np.full((2,2),.5),
                      force_history=[[1.,0.],[-1.,0.]], raw_quantum_momentum=[-.5,.5],
                      slope=.5, channel_basis="uncoupled states")
        with self.assertRaises(ValueError):
            two_channel_intercept_correction(**kwargs)
        with self.assertRaises(InterceptCancellationError):
            two_channel_intercept_correction(**kwargs, uncoupled_channels=True)
        kwargs["force_history"] = np.zeros((2,2))
        result = two_channel_intercept_correction(**kwargs, uncoupled_channels=True)
        self.assertFalse(result["active"])
        self.assertIsNone(result["intercept"])
        np.testing.assert_array_equal(result["raw"], result["corrected"])


if __name__ == "__main__":
    unittest.main()
