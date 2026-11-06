"""Manufactured current-state fields and explicitly reference-only identities."""
import unittest

import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.fields import (
    analyze_fields, periodic_q_derivative, bridge_scalar_reference,
    bridge_continuity_reference)
from multi_component_exact_factorization.gaussian_trajectory.mls import (
    MLSConfig, build_mls_operator)


class FieldTests(unittest.TestCase):
    def setUp(self):
        self.r = np.linspace(-.5, .5, 61)
        self.q = np.linspace(-np.pi, np.pi, 128, endpoint=False)
        self.w = np.full(len(self.q), 2*np.pi/len(self.q))
        self.config = MLSConfig(degree=4, neighbors=9)
        self.kw = dict(basis_convention="fixed electronic primitive basis",
                       gauge_convention="smooth common outer gauge",
                       mls_config=self.config)

    def state(self, rphase=.7, qphase=2.):
        amplitude = np.sqrt((1.+.2*np.cos(self.q))/(2*np.pi))
        phase = np.exp(1j*(rphase*self.r[:,None] + qphase*self.q[None,:]))
        return amplitude[None,:,None]*phase[:,:,None]*np.array([np.sqrt(.3), 1j*np.sqrt(.7)])[None,None,:]

    def test_fields_keep_physical_phase_and_zero_imaginary_quotient(self):
        fields = analyze_fields(self.r, self.state(), self.w, self.q, **self.kw)
        np.testing.assert_allclose(fields["a"], 2., atol=1e-13)
        np.testing.assert_allclose(fields["b"], .7, atol=3e-7)
        np.testing.assert_allclose(fields["alpha"], .7, atol=3e-7)
        np.testing.assert_allclose(fields["delta_b"], 0., atol=1e-13)
        for key in ("a_imaginary_defect", "b_imaginary_defect", "a_quotient_residual", "b_quotient_residual"):
            np.testing.assert_allclose(fields[key], 0., atol=2e-14)
        self.assertFalse(fields["outer_phase_assumed_zero"])
        self.assertFalse(fields["scalar_potentials_available"])

    def test_alpha_omitted_tail_accounting(self):
        density = (1+.8*np.sin(self.q))/(2*np.pi)
        state = np.sqrt(density)[None,:,None]*np.exp(1j*self.r[:,None,None]*np.sin(self.q)[None,:,None])
        fields = analyze_fields(self.r, state, self.w, self.q, support_relative=.8, **self.kw)
        np.testing.assert_allclose(fields["alpha"], .4, atol=2e-6)
        self.assertTrue(np.all(fields["excluded_mass"] > .1))
        self.assertTrue(np.all(np.abs(fields["alpha_excluded_contribution"]) > .01))
        np.testing.assert_allclose(fields["alpha_bridge_residual"], 0., atol=1e-15)
        self.assertTrue(np.isnan(fields["b"][~fields["support"]]).all())

    def test_current_density_bounded_at_nodes(self):
        amplitude = np.sin(self.q)/np.sqrt(np.pi)
        state = np.broadcast_to(amplitude[None,:,None], (len(self.r),len(self.q),1)).astype(complex)
        fields = analyze_fields(self.r, state, self.w, self.q, support_relative=1e-8, **self.kw)
        self.assertTrue(np.all(np.isfinite(fields["alpha"])))
        self.assertTrue(np.all(np.isfinite(fields["a_numerator"])))
        self.assertTrue(np.any(~fields["support"]))
        np.testing.assert_allclose(fields["a"][fields["support"]], 0., atol=1e-12)

    def test_common_unitary_basis_covariance(self):
        u = np.array([[1., 1j],[1j,1.]])/np.sqrt(2)
        state = self.state()
        original = analyze_fields(self.r, state, self.w, self.q, **self.kw)
        rotated = analyze_fields(self.r, state @ u, self.w, self.q, **self.kw)
        for key in ("rho", "a", "b", "alpha", "delta_b"):
            np.testing.assert_allclose(original[key], rotated[key], atol=4e-14)

    def test_periodic_derivatives_and_boundary_audit(self):
        state = np.exp(3j*self.q)[None,:,None]
        np.testing.assert_allclose(periodic_q_derivative(state, self.q), 3j*state, atol=1e-13)
        np.testing.assert_allclose(periodic_q_derivative(state, self.q, 2), -9*state, atol=8e-12)
        fields = analyze_fields(self.r, self.state(), self.w, self.q, **self.kw)
        self.assertTrue(np.all(fields["q_edge_mass"] > 0.))
        self.assertEqual(fields["q_boundary"], "periodic_fft")

    def test_invalid_grid_and_norm_fail_without_silent_repair(self):
        with self.assertRaises(ValueError):
            analyze_fields(self.r, 2*self.state(), self.w, self.q, **self.kw)
        with self.assertRaises(ValueError):
            analyze_fields(self.r, self.state(), self.w/2, self.q, **self.kw)
        bad_q = self.q.copy()
        bad_q[4] += .001
        with self.assertRaises(ValueError):
            analyze_fields(self.r, self.state(), self.w, bad_q, **self.kw)

    def test_imaginary_alpha_defect_is_measured_before_discard(self):
        # Finite MLS does not satisfy exact differential product rules.
        # A deliberately low-order reconstruction reports the defect rather
        # than presenting normalized coefficients as an accuracy proof.
        fields = analyze_fields(self.r, self.state(rphase=4), self.w, self.q,
            basis_convention="fixed basis", gauge_convention="common gauge",
            mls_config=MLSConfig(degree=2, neighbors=9))
        self.assertGreater(np.max(np.abs(fields["alpha_imaginary_defect"])), 1e-3)

    def test_F6_reference_full_positive_gauge_only(self):
        density = np.ones((len(self.r),len(self.q)))/(2*np.pi)
        args = dict(epsilon1=.2, rho=density, a=2., b=0., alpha=np.zeros(len(self.r)),
                    Lambda_q=0., Lambda_R=0., quadrature=self.w, mq=1., mass_R=3.)
        rec = bridge_scalar_reference(**args, gauge_convention="full_positive_marginals")
        np.testing.assert_allclose(rec["epsilon2_reconstructed"], 2.2)
        self.assertFalse(rec["propagation_closure"])
        with self.assertRaises(ValueError):
            bridge_scalar_reference(**args, gauge_convention="trajectory_gauge")

    def test_F8_retains_outer_phase_not_full_positive_assumption(self):
        # Positive conditional density, constant-in-q phase/alpha velocities,
        # and a known R-dependent bridge amplitude; test S_adv exactly.
        density = (1 + .2*self.r[:,None]*np.cos(self.q)[None,:])/(2*np.pi)
        op = build_mls_operator(self.r, self.config)
        def derivative_R(values):
            return op.apply(values, basis_convention="fixed basis", gauge_convention="common gauge")[0]
        phase = np.full(len(self.r), 3.)
        alpha = np.full(len(self.r), 2.)
        result = bridge_continuity_reference(density, np.ones(len(self.r)), phase,
            np.zeros_like(density), np.broadcast_to(alpha[:,None],density.shape), alpha,
            derivative_q=lambda x: periodic_q_derivative(x,self.q), derivative_R=derivative_R,
            mq=1., mass_R=5., bridge_gauge="positive")
        np.testing.assert_allclose(result["S_adv"], np.broadcast_to(-.2*np.cos(self.q)/(2*np.pi),density.shape), atol=1e-13)
        np.testing.assert_allclose(result["S_q"], 0.)
        np.testing.assert_allclose(result["S_rel"], 0.)
        self.assertEqual(result["target"], "partial_t_abs_Lambda_squared")
        self.assertFalse(result["propagation_closure"])


if __name__ == "__main__":
    unittest.main()
