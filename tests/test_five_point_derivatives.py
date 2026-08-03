import unittest

import numpy as np

from multi_component_exact_factorization.core import (
    covariant_square,
    derivative,
    logarithmic_components,
    occupied_support_mask,
)


class FivePointDerivativeTests(unittest.TestCase):
    def test_polynomial_is_differentiated_at_nonperiodic_boundaries(self):
        grid = np.linspace(-1.3, 2.1, 17)
        spacing = grid[1]-grid[0]
        values = 0.7*grid**4-0.2*grid**3+1.3*grid**2-0.4*grid+2.0
        exact_first = 2.8*grid**3-0.6*grid**2+2.6*grid-0.4
        exact_second = 8.4*grid**2-1.2*grid+2.6
        self.assertTrue(np.allclose(
            derivative(values, spacing, axis=0, order=1),
            exact_first, atol=2.0e-12, rtol=2.0e-12,
        ))
        self.assertTrue(np.allclose(
            derivative(values, spacing, axis=0, order=2),
            exact_second, atol=2.0e-11, rtol=2.0e-11,
        ))

    def test_independent_second_derivative_penalizes_checkerboard(self):
        spacing = 0.08
        checkerboard = (-1.0)**np.arange(31)
        kinetic = -derivative(checkerboard, spacing, axis=0, order=2)
        expected = (16.0/(3.0*spacing**2))*checkerboard[2:-2]
        self.assertTrue(np.allclose(kinetic[2:-2], expected))
        twice_first = -derivative(
            derivative(checkerboard, spacing, axis=0, order=1),
            spacing, axis=0, order=1,
        )
        self.assertGreater(
            np.linalg.norm(kinetic[4:-4]),
            100.0*np.linalg.norm(twice_first[4:-4]),
        )

    def test_zero_vector_covariant_square_is_independent_laplacian(self):
        grid = np.linspace(-1.0, 1.0, 21)
        spacing = grid[1]-grid[0]
        factor = np.exp(-grid**2)+0.2j*np.sin(2.0*grid)
        expected = -derivative(factor, spacing, axis=0, order=2)
        actual = covariant_square(
            factor, np.zeros_like(grid), spacing, axis=0, sign=+1
        )
        self.assertTrue(np.allclose(actual, expected, atol=1.0e-13))

    def test_support_mask_changes_amplitude_not_phase_momentum(self):
        grid = np.linspace(-3.0, 3.0, 41)
        spacing = grid[1]-grid[0]
        factor = np.exp(-0.6*grid**2+0.4j*grid)
        phase, logamp = logarithmic_components(
            factor, spacing, axis=0, numerical_floor=1.0e-14
        )
        mask = occupied_support_mask(np.abs(factor)**2, 1.0e-4)
        raw = phase-1j*logamp
        effective = phase-1j*mask*logamp
        self.assertTrue(np.allclose(effective.real, raw.real))
        self.assertLess(abs(effective.imag[0]), abs(raw.imag[0]))
        self.assertAlmostEqual(mask[len(mask)//2], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
