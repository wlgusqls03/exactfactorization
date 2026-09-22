import unittest
import numpy as np
from scipy.linalg import expm
from multi_component_exact_factorization.vsc_polariton.phase6_orthogonal_packets import orthogonal_rotation


class OrthogonalPacketsTests(unittest.TestCase):
    def test_operator_and_exponential(self):
        for n in (3, 120, 160):
            w, v, d = orthogonal_rotation(n)
            q = np.diag(np.sqrt(np.arange(1, n)), 1)
            q += q.T.copy()
            np.testing.assert_allclose((v*w)@v.T, q, atol=2e-13, rtol=0)
            np.testing.assert_allclose((v*np.exp(-.17j*w))@v.T,
                                       expm(-.17j*q), atol=3e-14, rtol=0)

    def test_repeated_roundtrip_without_renormalization(self):
        for n in (120, 160):
            _, v, _ = orthogonal_rotation(n)
            rng = np.random.default_rng(7)
            p = rng.normal(size=n)+1j*rng.normal(size=n)
            p /= np.linalg.norm(p)
            for _ in range(6000):
                p = (p@v)@v.T
            self.assertLess(abs(np.vdot(p,p).real-1), 1e-10)
