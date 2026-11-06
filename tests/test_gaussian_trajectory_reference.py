"""Manufactured operator/basis audits before any trajectory propagation."""
import unittest
import numpy as np
from multi_component_exact_factorization.gaussian_trajectory.reference import (
    covariant_derivatives, outer_expanded, outer_compact, material_rate,
    project_internal, lift_internal_rate, require_backend)


class ReferenceTests(unittest.TestCase):
    def test_compact_expanded_and_material(self):
        R = np.arange(128)*2*np.pi/128
        k = np.fft.fftfreq(128, 1/128)
        derivative = lambda u: np.fft.ifft(1j*k[:, None]*np.fft.fft(u, axis=0), axis=0)
        G = np.stack([np.exp(2j*R), .7*np.exp(-3j*R)], -1)
        a, da = .3+.2*np.cos(R), -.2*np.sin(R)
        L = -.2*np.sin(R)+.4j*np.cos(2*R)
        e = .4+.1*np.sin(2*R)
        H = G*np.array([.9, 1.3])
        compact = outer_compact(G, H, derivative, a, L, e, 17.)
        expanded = outer_expanded(G, H, derivative(G), derivative(derivative(G)), a, da, L, e, 17.)
        np.testing.assert_allclose(compact, expanded, atol=2e-13)
        np.testing.assert_allclose(material_rate(expanded, derivative(G), .13),
                                   -1j*compact+.13*derivative(G), atol=2e-13)

    def test_basis_covariance(self):
        # U=diag(exp(i*k_j R)); physical derivative U*(G'+d G).
        R = np.linspace(-1, 1, 31)
        phase = np.array([.3, -.7])
        d = np.diag(1j*phase); D = np.diag(-phase**2)
        G = np.stack([1+R+1j*R**2, .7+2j*R], -1)
        G1 = np.stack([1+2j*R, np.full_like(R, 2j, dtype=complex)], -1)
        G2 = np.stack([np.full_like(R, 2j, dtype=complex), np.zeros_like(R)], -1)
        first, second = covariant_derivatives(G, G1, G2, d, D)
        np.testing.assert_allclose(first, G1+1j*phase*G, atol=1e-14)
        np.testing.assert_allclose(second, G2+2j*phase*G1-phase**2*G, atol=1e-14)

    def test_internal_subspace_guard_and_moving_basis(self):
        basis = np.eye(4).reshape(2, 2, 4).astype(complex)
        G = np.array([[1, 2j], [.3, -.5j]])/np.sqrt(5.34)
        A, info = project_internal(G, basis, np.ones(2))
        self.assertLess(info['outside_population'], 1e-28)
        np.testing.assert_allclose(lift_internal_rate(A, -1j*A, basis, 2j*basis, .2), -.6j*G)
        with self.assertRaisesRegex(ValueError, 'round-trip'):
            project_internal(G, basis[:, :, :2], np.ones(2))

    def test_unsupported_backend_never_falls_back(self):
        require_backend('mte')
        for backend in ('tsh', 'ctmqc', 'ef-g-mls'):
            with self.assertRaises(NotImplementedError):
                require_backend(backend)
