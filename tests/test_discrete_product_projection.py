import unittest
from types import SimpleNamespace

import numpy as np

from multi_component_exact_factorization.core import (
    derivative,
    project_discrete_product_residual,
    reconstruct_psi,
)


class DiscreteProductProjectionTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            dx=0.2, dq=2.0*np.pi/9, dR=2.0*np.pi/8,
            proton_mass=7.0, heavy_mass=19.0,
        )
        nx, nq, nR = 7, 9, 8
        x = np.arange(nx)[:, None, None]
        q = np.arange(nq)[None, :, None]*self.model.dq
        R = np.arange(nR)[None, None, :]*self.model.dR
        phi = (1.0+0.08*np.cos(q)+0.05*np.sin(R))*np.exp(
            1j*(0.17*x+0.11*np.sin(q)+0.07*np.cos(R))
        )
        phi /= np.sqrt(np.sum(np.abs(phi)**2, axis=0)*self.model.dx)[None]
        lam = (1.0+0.12*np.cos(q[0])+0.04*np.sin(R[0]))*np.exp(
            1j*(0.09*np.sin(q[0])+0.05*np.cos(R[0]))
        )
        lam /= np.sqrt(np.sum(np.abs(lam)**2, axis=0)*self.model.dq)[None]
        chi = (1.0+0.1*np.cos(R.ravel()))*np.exp(0.08j*np.sin(R.ravel()))
        chi /= np.sqrt(np.sum(np.abs(chi)**2)*self.model.dR)
        self.phi, self.lam, self.chi = phi, lam, chi

    def test_projection_matches_full_discrete_nuclear_tangent(self):
        rng = np.random.default_rng(42)
        random_complex = lambda shape: (
            rng.normal(size=shape)+1j*rng.normal(size=shape)
        )*1.0e-3
        dphi = random_complex(self.phi.shape)
        dlam = random_complex(self.lam.shape)
        dchi = random_complex(self.chi.shape)
        dphi, dlam, dchi, diagnostics = project_discrete_product_residual(
            self.phi, self.lam, self.chi, dphi, dlam, dchi,
            self.model, support_floor_phi=0.0, support_floor_lam=0.0,
        )
        psi = reconstruct_psi(self.phi, self.lam, self.chi)
        product_rhs = (
            dphi*self.lam[None]*self.chi[None, None]
            +self.phi*dlam[None]*self.chi[None, None]
            +self.phi*self.lam[None]*dchi[None, None]
        )
        target_rhs = -1j*(
            -0.5*derivative(
                psi, self.model.dq, axis=1, order=2
            )/self.model.proton_mass
            -0.5*derivative(
                psi, self.model.dR, axis=2, order=2
            )/self.model.heavy_mass
        )
        self.assertTrue(np.allclose(
            product_rhs, target_rhs, atol=2.0e-14, rtol=2.0e-13
        ))
        self.assertLess(diagnostics["effective_product_residual_l2"], 1.0e-13)
        self.assertLess(
            abs(diagnostics["full_norm_rate_after_product_projection"]),
            1.0e-13,
        )


if __name__ == "__main__":
    unittest.main()
