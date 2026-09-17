import unittest
import numpy as np
from multi_component_exact_factorization.coupling_diagnostics import coupling_frame, summarize


class CouplingTests(unittest.TestCase):
    def test_uniform_factors_constant_connection(self):
        rho = np.ones((12, 32))/12
        heavy = np.ones(32)
        b = np.full_like(rho, 3.)
        alpha = np.ones(32)
        delta, ratios = coupling_frame(rho, heavy, b, alpha, 1., .1, 10.)
        np.testing.assert_allclose(delta, 2.)
        np.testing.assert_allclose(ratios[0], .2, atol=1e-12)
        np.testing.assert_allclose(ratios[1], .2, atol=1e-12)
        np.testing.assert_allclose(ratios[2], 0, atol=1e-12)
        row = summarize(rho, delta, ratios)
        self.assertAlmostEqual(row['rms'][3], .4)

    def test_separable_pg_zero_connection(self):
        q = np.arange(16)*2*np.pi/16
        R = np.arange(32)*2*np.pi/32
        rho = (1+.1*np.cos(q))[:, None]*(1+.2*np.cos(R))[None, :]
        heavy = rho.sum(axis=0)*(q[1]-q[0])
        delta, ratios = coupling_frame(rho, heavy, rho*0, R*0,
                                       q[1]-q[0], R[1]-R[0], 12000.)
        for a in ratios:
            np.testing.assert_allclose(a, 0, atol=1e-12)

    def test_empty_support(self):
        a = np.zeros((2,3))
        self.assertEqual(summarize(a, a, (a,a,a))['support_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
