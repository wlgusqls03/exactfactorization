import unittest

import numpy as np

from multi_component_exact_factorization.audit_pg_curvature import curvature_terms


class PGCurvatureTests(unittest.TestCase):
    def test_gaussian_ratio_and_density_scaling(self):
        q = np.linspace(-8, 8, 801)
        R = np.linspace(-6, 6, 601)
        rho = np.exp(-q[:, None]**2-R[None, :]**2)
        z = np.zeros_like(rho)
        terms = curvature_terms(rho, z, z, .02, .02, 2., 3.)
        # F=exp(-(q^2+R^2)/2): F''/F = coordinate^2-1.
        np.testing.assert_allclose(terms[0][300:500, 300], (q[300:500]**2-1)/4, atol=1e-7)
        np.testing.assert_allclose(terms[1][400, 200:400], (R[200:400]**2-1)/6, atol=1e-7)
        scaled = curvature_terms(7*rho, z, z, .02, .02, 2., 3.)
        np.testing.assert_allclose(terms[0][300:500,200:400], scaled[0][300:500,200:400], atol=1e-10)

    def test_constant_amplitude_and_momentum_sign(self):
        rho = np.ones((8, 9))
        Qq, QR, Kq, KR = curvature_terms(rho, 2*rho, -3*rho, .1, .2, 4, 9)
        np.testing.assert_array_equal(Qq, 0)
        np.testing.assert_array_equal(QR, 0)
        np.testing.assert_array_equal(Kq, -.5)
        np.testing.assert_array_equal(KR, -.5)


if __name__ == '__main__':
    unittest.main()
