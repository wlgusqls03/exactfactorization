import unittest

import numpy as np

from multi_component_exact_factorization.core import (
    weak_log_amplitude_gradient,
)


class WeakLogAmplitudeTests(unittest.TestCase):
    def test_periodic_smooth_amplitude(self):
        n = 96
        grid = 2.0*np.pi*np.arange(n)/n
        spacing = 2.0*np.pi/n
        factor = np.exp(0.2*np.cos(grid)+0.3j*np.sin(2.0*grid))
        actual, diagnostics = weak_log_amplitude_gradient(
            factor, spacing, 0, delta=1.0e-12,
            smoothing_length=0.0, tolerance=1.0e-11,
            max_iterations=10,
        )
        expected = -0.2*np.sin(grid)
        self.assertLess(np.max(np.abs(actual-expected)), 2.0e-5)
        self.assertLess(float(diagnostics["weak_log_residual"]), 1.0e-10)

    def test_node_is_finite(self):
        n = 64
        grid = 2.0*np.pi*np.arange(n)/n
        factor = np.sin(grid)*np.exp(0.2j*np.cos(grid))
        actual, diagnostics = weak_log_amplitude_gradient(
            factor, 2.0*np.pi/n, 0, delta=1.0e-8,
            smoothing_length=0.05, tolerance=1.0e-8,
            max_iterations=80,
        )
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertLess(float(diagnostics["weak_log_residual"]), 1.0e-8)
        self.assertEqual(float(diagnostics["weak_log_unconverged_lines"]), 0.0)

    def test_fourier_preconditioner_converges_narrow_joint_gaussian(self):
        nq, nR = 174, 120
        dq, dR = 0.04, 0.02
        q = (np.arange(nq)-nq//2)*dq
        R = (np.arange(nR)-nR//2)*dR
        proton = np.exp(-0.25*(q/0.169222)**2)
        heavy = np.exp(-0.25*(R/0.105435)**2)
        xi = proton[:, None]*heavy[None, :]
        for factor, spacing, axis in (
            (xi, dq, 0), (xi, dR, 1), (heavy, dR, 0),
        ):
            _, diagnostics = weak_log_amplitude_gradient(
                factor, spacing, axis, delta=1.0e-10,
                smoothing_length=0.04, tolerance=1.0e-8,
                max_iterations=40,
            )
            self.assertLess(
                float(diagnostics["weak_log_residual"]), 1.0e-8
            )
            self.assertEqual(
                float(diagnostics["weak_log_unconverged_lines"]), 0.0
            )
            self.assertLessEqual(
                float(diagnostics["weak_log_iterations"]), 40.0
            )


if __name__ == "__main__":
    unittest.main()
