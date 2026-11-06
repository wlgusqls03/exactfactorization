"""Equation-level CTMQC tests, not certification of a Gaussian trajectory run."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm

from multi_component_exact_factorization.gaussian_trajectory.ctmqc_channels import (
    ct_channel_rates, ctmqc_channel_rhs, pair_quantum_momentum)
from multi_component_exact_factorization.gaussian_trajectory.heavy_density import (
    DensityFailure, InterceptCancellationError)


class CTMQCChannelTests(unittest.TestCase):
    def setup_channels(self, ntraj=5, nstate=3):
        R = np.linspace(-2., 2., ntraj)
        eta = np.arange(1., ntraj+1.)
        eta /= eta.sum()
        p = np.tile(np.arange(1., nstate+1.), (ntraj, 1))
        p /= p.sum(axis=1)[:, None]
        A = np.sqrt(p).astype(complex)
        E = np.broadcast_to(np.arange(nstate)*.2, A.shape).copy()
        d = np.zeros((ntraj, nstate, nstate), complex)
        if nstate > 1:
            d[:, 0, 1] = .15
            d[:, 1, 0] = -.15
        force = (E[:, :, None]-E[:, None, :])*d
        force[:, np.arange(nstate), np.arange(nstate)] = -.1*np.arange(nstate)
        f = (1.+.03*R[:, None])*np.arange(nstate)[None, :]
        return dict(amplitudes=A, energies=E, derivative_couplings=d, force_matrix=force,
                    force_history=f, R=R, P=R*.3+1., mass=7., eta=eta,
                    channel_basis="complete_internal_eigenbasis")

    def test_raw_formula_matches_mean_history_and_variance(self):
        args = self.setup_channels()
        result = ctmqc_channel_rhs(**args)
        p, f = result["populations"], args["force_history"]
        Q, M = result["raw_quantum_momentum"], args["mass"]
        mean = np.sum(p*f, axis=1)
        expected = Q[:, None]/M*(f-mean[:, None])*args["amplitudes"]
        variance = np.sum(p*(f-mean[:, None])**2, axis=1)
        assert_allclose(result["ct_amplitude_dot"], expected, atol=1.e-16)
        assert_allclose(result["ct_force"], 2.*Q/M*variance, atol=1.e-16)
        self.assertLess(result["audit"]["norm_rate_max_abs"], 1.e-15)
        self.assertEqual(result["audit"]["coefficient_ct_prefactor"], 1.)
        self.assertEqual(result["audit"]["unordered_pair_force_prefactor"], 2.)
        self.assertEqual(result["pair_quantum_momentum"].shape, (5, 0))
        self.assertEqual(result["audit"]["pair_intercepts"], [])
        explicit_pairs = np.repeat(Q[:, None], 3, axis=1)
        pair_result = ct_channel_rates(
            args["amplitudes"], args["energies"], args["derivative_couplings"],
            args["force_matrix"], f, args["P"], M, explicit_pairs)
        assert_allclose(result["ct_amplitude_dot"], pair_result["ct_amplitude_dot"], atol=1.e-16)
        assert_allclose(result["ct_force"], pair_result["ct_force"], atol=1.e-16)

    def test_force_history_is_literal_surface_force_without_reset(self):
        args = self.setup_channels(nstate=2)
        args["amplitudes"][:] = [np.sqrt(1.e-8), np.sqrt(1.-1.e-8)]
        result = ctmqc_channel_rhs(**args)
        expected = np.diagonal(args["force_matrix"], axis1=1, axis2=2).real
        assert_allclose(result["force_history_dot"], expected, atol=0)
        self.assertEqual(result["audit"]["force_history_policy"],
                         "literal_integrated_adiabatic_force_no_resets")

    def test_force_contains_ehrenfest_coherences(self):
        args = self.setup_channels(nstate=2)
        args["force_history"][:] = 0
        result = ctmqc_channel_rhs(**args)
        A, f = args["amplitudes"], args["force_matrix"]
        expected = np.einsum("in,inm,im->i", A.conj(), f, A).real
        assert_allclose(result["heavy_force"], expected, atol=1.e-16)
        diagonal_only = np.sum(abs(A)**2*np.diagonal(f, axis1=1, axis2=2).real, axis=1)
        self.assertGreater(np.max(abs(expected-diagonal_only)), .01)

    def test_common_history_shift_changes_neither_equation(self):
        args = self.setup_channels()
        before = ctmqc_channel_rhs(**args)
        args["force_history"] += (np.arange(5)*8.)[:, None]
        after = ctmqc_channel_rhs(**args)
        assert_allclose(after["amplitude_dot"], before["amplitude_dot"], atol=3.e-16)
        assert_allclose(after["heavy_force"], before["heavy_force"], atol=3.e-16)

    def test_multichannel_intercept_has_zero_weighted_ct_transfer(self):
        args = self.setup_channels(ntraj=7, nstate=4)
        args["derivative_couplings"][:] = 0
        diagonal = np.diagonal(args["force_matrix"], axis1=1, axis2=2).copy()
        args["force_matrix"][:] = 0
        args["force_matrix"][:, np.arange(4), np.arange(4)] = diagonal
        result = ctmqc_channel_rhs(**args, quantum_momentum_mode="pair_intercept")
        self.assertEqual(result["pair_quantum_momentum"].shape, (7, 6))
        assert_allclose(result["audit"]["ct_ensemble_population_rate"], 0, atol=1.e-16)
        self.assertGreater(np.max(abs(result["ct_amplitude_dot"])), .001)
        for pair in result["audit"]["pair_intercepts"]:
            self.assertEqual(pair["status"], "pair_intercept")
            self.assertLess(abs(pair["used_transfer_numerator"]), 1.e-16)

    def test_variable_slope_intercept_uses_slope_in_weights(self):
        R = np.array([-2., .5, 3.])
        eta = np.array([.2, .3, .5])
        p = np.array([[.3, .7], [.6, .4], [.8, .2]])
        f = np.array([[.1, .9], [.4, .8], [.6, .9]])
        slope = np.array([.3, 1., 2.])
        raw = np.array([-.8, .2, 2.1])
        Q, pairs, records = pair_quantum_momentum(R, eta, p, f, raw, slope, mode="pair_intercept")
        source = eta*p[:, 0]*p[:, 1]*(f[:, 0]-f[:, 1])
        center = np.dot(R, slope*source)/np.sum(slope*source)
        assert_allclose(Q[:, 0], slope*(R-center), atol=3.e-16)
        self.assertAlmostEqual(records[0]["intercept"], center)
        self.assertLess(abs(np.dot(Q[:, 0], source)), 1.e-16)
        self.assertEqual(pairs, [(0, 1)])

    def test_intercept_cancellation_is_not_silently_disabled(self):
        R, eta = np.array([-1., 1.]), np.full(2, .5)
        p = np.full((2, 2), .5)
        f = np.array([[1., 0], [-1., 0.]])
        with self.assertRaises(InterceptCancellationError):
            pair_quantum_momentum(R, eta, p, f, R, np.ones(2), mode="pair_intercept")
        Q, _, records = pair_quantum_momentum(
            R, eta, p, f, R, np.ones(2), mode="pair_intercept", intercept_policy="density_fallback")
        assert_allclose(Q[:, 0], R)
        self.assertEqual(records[0]["status"], "density_fallback_cancellation")
        self.assertGreater(abs(records[0]["used_transfer_numerator"]), .1)

    def test_exact_zero_source_needs_no_intercept(self):
        R, eta = np.array([-1., 1.]), np.full(2, .5)
        Q, _, records = pair_quantum_momentum(
            R, eta, np.full((2, 2), .5), np.ones((2, 2)), R, np.ones(2), mode="pair_intercept")
        assert_allclose(Q, 0)
        self.assertEqual(records[0]["status"], "zero_source")
        self.assertIsNone(records[0]["intercept"])

    def test_state_gaussian_raw_momentum_width_convention(self):
        args = self.setup_channels()
        args["eta"][:] = .2
        result = ctmqc_channel_rhs(**args)
        # Sample variance=2, standard deviation=sqrt(2), hence Q=R/4.
        assert_allclose(result["raw_quantum_momentum"], args["R"]/4., atol=1.e-16)
        assert_allclose(result["quantum_momentum_slope"], .25, atol=1.e-16)
        self.assertTrue(result["density"].audit["log_derivative_uses_posterior"])

    def test_kde_slope_and_raw_are_independent_from_modified_q(self):
        args = self.setup_channels()
        result = ctmqc_channel_rhs(**args, density_mode="kde", bandwidth=.8,
                                  quantum_momentum_mode="pair_intercept")
        assert_allclose(result["quantum_momentum_slope"], 1./(2.*.8**2))
        self.assertGreater(np.max(abs(result["pair_quantum_momentum"][:, 0]-
                                     result["raw_quantum_momentum"])), .01)

    def test_complex_channel_basis_phase_covariance(self):
        args = self.setup_channels()
        first = ctmqc_channel_rhs(**args)
        phases = np.exp(1j*np.array([.7, -.2, 1.3]))
        args["amplitudes"] *= phases.conj()[None, :]
        unitary = phases.conj()[None, :, None]*phases[None, None, :]
        args["derivative_couplings"] *= unitary
        args["force_matrix"] *= unitary
        second = ctmqc_channel_rhs(**args)
        assert_allclose(second["amplitude_dot"], first["amplitude_dot"]*phases.conj(), atol=1.e-16)
        assert_allclose(second["heavy_force"], first["heavy_force"], atol=1.e-16)

    def test_bad_input_norm_and_force_are_not_repaired(self):
        args = self.setup_channels()
        args["amplitudes"] *= 1.001
        with self.assertRaisesRegex(ValueError, "Channel norm defect"):
            ctmqc_channel_rhs(**args)
        args = self.setup_channels()
        args["force_matrix"][:, 0, 1] += .1
        with self.assertRaisesRegex(ValueError, "Inconsistent adiabatic"):
            ctmqc_channel_rhs(**args)

    def test_width_policy_is_explicit_and_audited(self):
        args = self.setup_channels()
        args["R"][:] = 0
        with self.assertRaises(DensityFailure):
            ctmqc_channel_rhs(**args)
        result = ctmqc_channel_rhs(**args, width_floor=.1)
        self.assertEqual(result["density"].audit["regularized_channels"], [0, 1, 2])

    def test_pure_state_and_single_channel_remain_well_defined(self):
        args = self.setup_channels(nstate=1)
        result = ctmqc_channel_rhs(**args, quantum_momentum_mode="pair_intercept")
        self.assertEqual(result["pair_quantum_momentum"].shape, (5, 0))
        assert_allclose(result["ct_amplitude_dot"], 0)
        assert_allclose(result["ct_force"], 0)

    def test_complete_vector_rk4_not_componentwise_update(self):
        # Regression for the independent native componentwise-RK audit.  This
        # module supplies a simultaneous vector RHS; no native step is reused.
        E = np.array([[.2, .7]])
        d = np.array([[[0., 1.], [-1., 0.]]])
        force = (E[:, :, None]-E[:, None, :])*d
        initial = np.array([[np.sqrt(.6), 1j*np.sqrt(.4)]])
        generator = -1j*np.diag(E[0])-d[0]
        exact = expm(generator)@initial[0]

        def rhs(A):
            return ct_channel_rates(A, E, d, force, np.zeros((1, 2)), np.ones(1), 1.,
                                    np.zeros((1, 1)), normalization_tolerance=.02)["amplitude_dot"]

        errors = []
        for steps in (10, 20, 40):
            state, dt = initial.copy(), 1./steps
            for _ in range(steps):
                k1 = rhs(state)
                k2 = rhs(state+.5*dt*k1)
                k3 = rhs(state+.5*dt*k2)
                k4 = rhs(state+dt*k3)
                state += dt*(k1+2.*k2+2.*k3+k4)/6.
            errors.append(np.linalg.norm(state[0]-exact))
        self.assertGreater(errors[0]/errors[1], 15.)
        self.assertGreater(errors[1]/errors[2], 15.)
        self.assertLess(errors[-1], 1.e-7)

    def test_ct_energy_rate_is_reported_not_assumed_zero(self):
        args = self.setup_channels()
        result = ctmqc_channel_rhs(**args)
        pdot = 2.*np.real(args["amplitudes"].conj()*result["ct_amplitude_dot"])
        energy_rate = args["P"]/args["mass"]*result["ct_force"] + np.sum(args["energies"]*pdot, axis=1)
        self.assertGreater(np.max(abs(energy_rate)), 1.e-3)
        self.assertFalse(result["audit"]["energy_conservation_guaranteed"])


if __name__ == "__main__":
    unittest.main()
