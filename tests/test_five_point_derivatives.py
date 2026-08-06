import unittest

import numpy as np

from multi_component_exact_factorization.core import (
    covariant_square,
    derivative,
    logarithmic_components,
    occupied_support_mask,
    periodic_five_point_second_eigenvalues,
)


class FivePointDerivativeTests(unittest.TestCase):
    def test_periodic_fourier_mode_including_boundary_points(self):
        count = 64
        length = 2.0*np.pi
        spacing = length/count
        grid = np.arange(count)*spacing
        wave_number = 3.0
        values = np.exp(1j*wave_number*grid)
        exact_first = 1j*wave_number*values
        exact_second = -(wave_number**2)*values
        self.assertTrue(np.allclose(
            derivative(values, spacing, axis=0, order=1),
            exact_first, atol=3.0e-4, rtol=3.0e-4,
        ))
        self.assertTrue(np.allclose(
            derivative(values, spacing, axis=0, order=2),
            exact_second, atol=2.0e-4, rtol=2.0e-4,
        ))

    def test_periodic_operators_have_exact_discrete_adjoint_structure(self):
        count = 17
        spacing = 0.08
        identity = np.eye(count)
        d1 = derivative(identity, spacing, axis=0, order=1)
        d2 = derivative(identity, spacing, axis=0, order=2)
        self.assertTrue(np.allclose(d1+d1.T, 0.0, atol=1.0e-14))
        self.assertTrue(np.allclose(d2-d2.T, 0.0, atol=1.0e-14))

    def test_fft_symbol_matches_periodic_five_point_operator(self):
        count = 18
        spacing = 0.07
        rng = np.random.default_rng(13)
        values = rng.normal(size=count)+1j*rng.normal(size=count)
        symbol = periodic_five_point_second_eigenvalues(count, spacing)
        spectral = np.fft.ifft(np.fft.fft(values)*symbol)
        finite_difference = derivative(
            values, spacing, axis=0, order=2
        )
        self.assertTrue(np.allclose(
            spectral, finite_difference, atol=2.0e-12, rtol=2.0e-13
        ))

    def test_independent_second_derivative_penalizes_checkerboard(self):
        spacing = 0.08
        checkerboard = (-1.0)**np.arange(31)
        kinetic = -derivative(checkerboard, spacing, axis=0, order=2)
        # 홀수 점 periodic grid에서는 (-1)^j가 wrap 경계에서 완전한 Fourier
        # mode가 아니므로 경계 두 점을 제외한 interior 값을 검사한다.
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

    def test_varying_real_vector_covariant_square_is_hermitian(self):
        count = 31
        spacing = 2.0*np.pi/count
        grid = np.arange(count)*spacing
        vector = 0.3*np.sin(grid)+0.1*np.cos(2.0*grid)
        identity = np.eye(count, dtype=complex)
        for sign in (-1, +1):
            operator = covariant_square(
                identity, vector[:, None], spacing, axis=0, sign=sign
            )
            self.assertTrue(np.allclose(
                operator, operator.conj().T, atol=2.0e-13, rtol=2.0e-13
            ))

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
