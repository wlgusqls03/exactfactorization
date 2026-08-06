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
        self.assertTrue(np.isfinite(diagnostics["weak_log_residual"]))


if __name__ == "__main__":
    unittest.main()
