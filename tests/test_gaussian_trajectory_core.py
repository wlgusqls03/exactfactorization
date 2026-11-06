"""Formula-level tests for S3, P3--P18 and finite-domain N1 (no dynamics claim)."""
import unittest

import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.gaussian import (
    GaussianRankError, build_bridge, pack_parameters, unpack_parameters,
    project_amplitude_rate, select_support, split_and_project, split_complex_rate,
)


def grid(lo=-12, hi=12, n=1201):
    q = np.linspace(lo, hi, n)
    omega = np.full(n, q[1] - q[0])
    omega[[0, -1]] *= 0.5
    return q, omega


def make_bridge(q=None, omega=None, weights=(1.0,), centers=(0.3,), widths=(0.8,),
                coordinates="physical", normalize_discrete=True):
    if q is None:
        q, omega = grid()
    parameters = pack_parameters(weights, centers, widths, coordinates=coordinates)
    return build_bridge(q, omega, parameters, components=len(weights),
                        coordinates=coordinates, normalize_discrete=normalize_discrete)


class GaussianBridgeTests(unittest.TestCase):
    def test_normalization_positive_and_complex_state_norm(self):
        b = make_bridge(weights=(0.2, 0.8), centers=(-1.4, 1.1), widths=(0.45, 1.2))
        C = np.stack((np.cos(b.q), 1j * np.sin(b.q)), axis=1)
        G = b.amplitude[:, None] * C
        self.assertTrue(np.all(b.rho > 0))
        self.assertAlmostEqual(b.quadrature @ b.rho, 1.0, places=14)
        np.testing.assert_allclose(np.sum(abs(G)**2, axis=1), b.rho, atol=1e-15)
        self.assertAlmostEqual(b.quadrature @ np.sum(abs(G)**2, axis=1), 1.0, places=14)

    def check_jacobian(self, coordinates, lo, hi):
        q, omega = grid(lo, hi, 801)
        b = make_bridge(q, omega, (0.31, 0.69), (-0.6, 0.8), (0.55, 1.3), coordinates)
        numeric = np.empty_like(b.jacobian)
        for j in range(b.parameters.size):
            dz = np.zeros_like(b.parameters)
            dz[j] = 2e-6
            def value(z):
                return build_bridge(q, omega, z, components=2, coordinates=coordinates).rho
            numeric[:, j] = (value(b.parameters + dz) - value(b.parameters - dz)) / (2 * dz[j])
        # Central differences at h=2e-6 accumulate <7e-10 roundoff near zeros.
        np.testing.assert_allclose(b.jacobian, numeric, rtol=2e-7, atol=1e-9)
        np.testing.assert_allclose(omega @ b.jacobian, 0, atol=3e-14)

    def test_physical_jacobian_real_line_and_finite_domain_N1(self):
        self.check_jacobian("physical", -12, 12)
        self.check_jacobian("physical", -0.4, 1.1)

    def test_transformed_jacobian_real_line_and_finite_domain_N1(self):
        self.check_jacobian("transformed", -12, 12)
        self.check_jacobian("transformed", -0.4, 1.1)

    def test_real_line_mode_reports_missing_tail(self):
        q, omega = grid(-0.5, 0.5, 401)
        b = make_bridge(q, omega, centers=(0,), widths=(1,), normalize_discrete=False)
        self.assertLess(b.audit["grid_norm"], 0.4)
        self.assertGreater(b.audit["grid_norm_error"], 0.6)
        JQ = b.rho * q
        Js = b.rho * (q*q - 1)
        np.testing.assert_allclose(b.jacobian, np.stack((JQ, Js), axis=1), atol=1e-16)

    def test_stable_discrete_normalization_far_off_domain(self):
        q, omega = grid(-1, 1, 101)
        b = make_bridge(q, omega, centers=(50,), widths=(0.5,))
        self.assertAlmostEqual(omega @ b.rho, 1.0, places=11)
        self.assertLess(b.audit["kernel_log_grid_norm_before_normalization"][0], -4000)
        self.assertTrue(np.all(np.isfinite(b.jacobian)))

    def test_parameter_roundtrip_and_invalid_input(self):
        w, Q, s = [0.2, 0.35, 0.45], [-1, 0, 1], [0.7, 0.8, 0.9]
        for coordinates in ("physical", "transformed"):
            z = pack_parameters(w, Q, s, coordinates=coordinates)
            for actual, expected in zip(unpack_parameters(z, 3, coordinates=coordinates), (w, Q, s)):
                np.testing.assert_allclose(actual, expected)
        for args in (([0.5, 0.6], [0, 1], [1, 1]), ([1], [0], [0])):
            with self.assertRaises(ValueError):
                pack_parameters(*args)
        with self.assertRaises(ValueError):
            pack_parameters([0, 1], [0, 1], [1, 1], coordinates="transformed")
        with self.assertRaises(ValueError):
            build_bridge([0, 0], [1, 1], [0, 1], components=1)
        with self.assertRaises(ValueError):
            build_bridge([0, 1], [0, 1], [0, 1], components=1)


class GaussianProjectionTests(unittest.TestCase):
    def test_pure_phase_rate_preserved(self):
        b = make_bridge()
        C = np.stack((np.cos(0.3*b.q), np.sin(0.3*b.q)*np.exp(0.7j*b.q)), axis=1)
        G_rhs = -1j * 1.7 * b.amplitude[:, None] * C
        result = split_and_project(G_rhs, C, b)
        np.testing.assert_allclose(result.C_dot, -1.7j*C, atol=8e-16)
        np.testing.assert_allclose(result.amplitude_rate_requested, 0, atol=2e-16)
        np.testing.assert_allclose(result.Lambda_dot, 0, atol=2e-16)
        self.assertLess(result.audit["max_pnc_rate_error"], 2e-15)

    def test_pure_real_radial_rate_leaves_coefficients_unchanged(self):
        b = make_bridge()
        C = np.stack((np.full_like(b.q, 0.6), np.full(b.q.shape, 0.8j)), axis=1)
        r = b.amplitude * np.cos(b.q)
        C_dot, r_found, audit = split_complex_rate(C, r[:, None]*C, b.amplitude)
        np.testing.assert_allclose(r_found, r, atol=2e-16)
        np.testing.assert_allclose(C_dot, 0, atol=3e-16)
        self.assertLess(audit["max_pnc_rate_error"], 1e-15)

    def test_single_gaussian_metric_moments_and_tangent_translation_breathing(self):
        b = make_bridge()
        z_dot = np.array([0.45, -0.08])
        r = (b.jacobian @ z_dot) / (2*b.amplitude)
        p = project_amplitude_rate(b, r)
        np.testing.assert_allclose(p.parameter_dot, z_dot, atol=5e-16)
        np.testing.assert_allclose(p.matrix, np.diag([1/0.8**2, 2/0.8**2]), atol=2e-14)
        density_dot = 2*b.amplitude*r
        moments = [b.quadrature @ ((b.q - 0.3)*density_dot),
                   b.quadrature @ (((b.q - 0.3)**2 - 0.8**2)*density_dot)/(2*0.8)]
        np.testing.assert_allclose(p.parameter_dot, moments, atol=1e-15)
        self.assertLess(p.audit["projection_residual_l2"], 1e-15)

    def test_K1_moments_for_non_tangent_rate(self):
        b = make_bridge()
        r = b.amplitude * (0.4*np.cos(1.9*b.q) + 0.3*np.sin(b.q))
        p = project_amplitude_rate(b, r)
        density_dot = 2*b.amplitude*r
        moments = [b.quadrature @ ((b.q-b.centers[0])*density_dot),
                   b.quadrature @ (((b.q-b.centers[0])**2-b.widths[0]**2)*density_dot)/(2*b.widths[0])]
        np.testing.assert_allclose(p.parameter_dot, moments, atol=2e-15)
        self.assertGreater(p.audit["projection_residual_l2"], 0.01)

    def test_projection_objectives_normal_equations_and_transformed_velocity(self):
        b = make_bridge(weights=(0.35, 0.65), centers=(-1.2, 1.2), widths=(0.5, 0.9))
        r = b.amplitude * np.sin(1.7*b.q)
        p = project_amplitude_rate(b, r)
        weighted_density_residual = b.jacobian @ p.parameter_dot - 2*b.amplitude*r
        density_objective = 0.25 * b.quadrature @ (weighted_density_residual**2 / b.rho)
        amplitude_objective = b.quadrature @ (p.Lambda_dot-r)**2
        self.assertAlmostEqual(density_objective, amplitude_objective, places=14)
        np.testing.assert_allclose(p.matrix @ p.parameter_dot, p.rhs, atol=2e-14)
        bt = make_bridge(weights=b.weights, centers=b.centers, widths=b.widths, coordinates="transformed")
        pt = project_amplitude_rate(bt, r)
        np.testing.assert_allclose(pt.Lambda_dot, p.Lambda_dot, atol=2e-15)

    def test_actual_P15_PNC_and_wrong_split_counterexample(self):
        b = make_bridge()
        C = np.stack((np.cos(0.6*b.q), np.sin(0.6*b.q)*np.exp(0.2j*b.q)), axis=1)
        G_rhs = b.amplitude[:, None] * np.stack((np.exp(1j*b.q), np.cos(0.7*b.q)), axis=1)
        result = split_and_project(G_rhs, C, b)
        r = result.amplitude_rate_requested
        np.testing.assert_allclose(result.G_dot_actual-G_rhs, (result.Lambda_dot-r)[:, None]*C, atol=3e-16)
        self.assertLess(result.audit["max_pnc_rate_error"], 4e-15)
        self.assertAlmostEqual(result.audit["actual_vs_eom_rate_l2"],
                               result.audit["projection_residual_l2"], places=14)
        wrong_C_dot = (G_rhs - result.Lambda_dot[:, None]*C) / b.amplitude[:, None]
        wrong_pnc_rate = 2*np.real(np.sum(C.conj()*wrong_C_dot, axis=1))
        self.assertGreater(np.max(abs(wrong_pnc_rate)), 0.1)

    def test_duplicate_and_zero_or_tiny_weight_rank_failures(self):
        cases = [((0.5, 0.5), (0, 0), (1, 1)),
                 ((0.0, 1.0), (-1, 1), (1, 1)),
                 ((1e-18, 1.0-1e-18), (-1, 1), (1, 1))]
        for weights, centers, widths in cases:
            b = make_bridge(weights=weights, centers=centers, widths=widths)
            with self.assertRaises(GaussianRankError) as ctx:
                project_amplitude_rate(b, np.zeros_like(b.q))
            self.assertLess(ctx.exception.audit["rank"], 5)
        b = make_bridge(weights=(0.5, 0.5), centers=(0, 0), widths=(1, 1))
        p = project_amplitude_rate(b, b.amplitude*b.q, rank_policy="truncate")
        self.assertEqual(p.audit["regularization"], "explicit_truncated_SVD")
        self.assertGreater(p.audit["discarded_directions"], 0)

    def test_nodes_require_explicit_mask_and_record_excluded_rhs(self):
        q, omega = grid(-80, 80, 1601)
        b = make_bridge(q, omega, centers=(0,), widths=(1,))
        self.assertGreater(b.audit["density_underflow_points"], 0)
        C = np.ones((len(q), 1), dtype=complex)
        with self.assertRaises(ValueError):
            split_complex_rate(C, np.zeros_like(C), b.amplitude)
        G_rhs = -1j*b.amplitude[:, None]*C
        result = split_and_project(G_rhs, C, b, support_relative=1e-8, max_excluded_mass=1e-7)
        self.assertTrue(np.all(np.isfinite(result.C_dot)))
        self.assertGreater(result.audit["excluded_mass"], 0)
        self.assertLess(result.audit["excluded_mass"], 1e-7)
        self.assertGreater(result.audit["discarded_rhs_l2"], 0)
        np.testing.assert_allclose(result.C_dot[~result.support], 0)
        with self.assertRaises(ValueError):
            select_support(b, support_relative=0.1, max_excluded_mass=1e-10)

    def test_underresolved_narrow_component_and_PNC_fail_explicitly(self):
        q, omega = grid(-1, 1, 11)
        b = make_bridge(q, omega, centers=(0,), widths=(1e-4,))
        mask, _ = select_support(b)
        with self.assertRaises(GaussianRankError):
            project_amplitude_rate(b, np.zeros_like(q), support=mask)
        b = make_bridge()
        C = np.full((len(b.q), 1), 1.1, dtype=complex)
        with self.assertRaises(ValueError):
            split_complex_rate(C, np.zeros_like(C), b.amplitude)


if __name__ == "__main__":
    unittest.main()
