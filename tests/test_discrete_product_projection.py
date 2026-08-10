import unittest
from types import SimpleNamespace

import numpy as np

from multi_component_exact_factorization.core import (
    derivative,
    flat_top_on_for_probability_budget,
    flat_top_support_mask,
    mask_threshold_for_probability_budget,
    occupied_support_mask,
    project_discrete_product_residual,
    reconstruct_psi,
    suppressed_probability,
)


class DiscreteProductProjectionTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            dx=0.2, dq=2.0*np.pi/9, dR=2.0*np.pi/8,
            proton_mass=7.0, heavy_mass=19.0,
            product_projection_backend="nested_inverse",
            projection_tau_phi=1.0e-10,
            projection_tau_lam=1.0e-10,
            projection_tau_chi=1.0e-10,
            projection_support_epsilon=1.0e-12,
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
        self.assertTrue(np.isfinite(
            diagnostics["relative_product_projection_l2"]
        ))
        self.assertTrue(np.isfinite(
            diagnostics["relative_support_product_projection_l2"]
        ))
        self.assertGreaterEqual(diagnostics["outer_probability_q"], 0.0)
        self.assertGreaterEqual(diagnostics["outer_probability_R"], 0.0)

    def test_mask_residual_decomposition_is_exact(self):
        rng = np.random.default_rng(7)
        masked = tuple(
            (rng.normal(size=shape)+1j*rng.normal(size=shape))*1.0e-3
            for shape in (self.phi.shape, self.lam.shape, self.chi.shape)
        )
        unmasked = tuple(
            value+(rng.normal(size=value.shape)+1j*rng.normal(size=value.shape))*1.0e-4
            for value in masked
        )
        _, _, _, diagnostics = project_discrete_product_residual(
            self.phi, self.lam, self.chi, *masked, self.model,
            support_floor_phi=0.0, support_floor_lam=0.0,
            unmasked_rhs=unmasked,
        )
        total = diagnostics["product_residual_l2"]
        pieces = (
            diagnostics["product_residual_without_mask_l2"]
            +diagnostics["product_residual_due_to_mask_l2"]
        )
        self.assertLessEqual(total, pieces+1.0e-14)
        a = diagnostics["product_residual_without_mask_l2"]
        b = diagnostics["product_residual_due_to_mask_l2"]
        cosine = diagnostics["product_mask_nonmask_alignment"]
        self.assertAlmostEqual(
            total**2, a**2+b**2+2.0*a*b*cosine, delta=1.0e-13
        )
        self.assertLessEqual(
            abs(cosine), 1.0+1.0e-12
        )

    def test_probability_budget_inverts_suppressed_mass(self):
        density = np.exp(-np.linspace(-4.0, 4.0, 101)**2)
        for budget in (1.0e-9, 1.0e-7, 1.0e-5):
            eta = mask_threshold_for_probability_budget(density, budget)
            mask = occupied_support_mask(density, eta)
            measured = suppressed_probability(density, mask, 1.0)
            self.assertAlmostEqual(measured, budget, delta=budget*1.0e-6)

    def test_flat_top_budget_has_exact_plateau_and_requested_mass(self):
        density = np.exp(-np.linspace(-7.0, 7.0, 401)**2)
        for budget in (1.0e-10, 1.0e-8, 1.0e-6):
            onset = flat_top_on_for_probability_budget(
                density, budget, transition_decades=3.0
            )
            mask = flat_top_support_mask(density, onset, 3.0)
            measured = suppressed_probability(density, mask)
            self.assertAlmostEqual(measured, budget, delta=budget*2.0e-6)
            relative = density/np.max(density)
            self.assertTrue(np.all(mask[relative >= onset] == 1.0))
            self.assertTrue(np.all(mask[relative <= onset*1.0e-3] == 0.0))

    def test_weighted_projection_limits_tail_factor_correction(self):
        rng = np.random.default_rng(19)
        dphi = (
            rng.normal(size=self.phi.shape)+1j*rng.normal(size=self.phi.shape)
        )*1.0e-3
        dlam = np.zeros_like(self.lam)
        dchi = np.zeros_like(self.chi)
        chi = self.chi.copy()
        chi[0] *= 1.0e-10
        legacy_model = SimpleNamespace(**vars(self.model))
        weighted_model = SimpleNamespace(**vars(self.model))
        weighted_model.product_projection_backend = "weighted_tikhonov"
        _, _, _, legacy = project_discrete_product_residual(
            self.phi, self.lam, chi, dphi, dlam, dchi, legacy_model,
            support_floor_phi=1.0e-10, support_floor_lam=1.0e-10,
        )
        _, _, _, weighted = project_discrete_product_residual(
            self.phi, self.lam, chi, dphi, dlam, dchi, weighted_model,
            support_floor_phi=1.0e-10, support_floor_lam=1.0e-10,
        )
        self.assertLessEqual(
            weighted["product_correction_phi"],
            legacy["product_correction_phi"]+1.0e-15,
        )
        self.assertLessEqual(
            weighted["product_correction_chi"],
            legacy["product_correction_chi"]+1.0e-15,
        )
        self.assertTrue(np.isfinite(
            weighted["inverse_support_product_correction_phi"]
        ))

    def test_weighted_projection_keeps_strong_factor_tangents(self):
        model = SimpleNamespace(**vars(self.model))
        model.product_projection_backend = "weighted_tikhonov"
        zeros = (
            np.zeros_like(self.phi), np.zeros_like(self.lam),
            np.zeros_like(self.chi),
        )
        dphi, dlam, _, diagnostics = project_discrete_product_residual(
            self.phi, self.lam, self.chi, *zeros, model,
            support_floor_phi=1.0e-10, support_floor_lam=1.0e-10,
        )
        phi_tangent = np.sum(np.conj(self.phi)*dphi, axis=0)*self.model.dx
        lam_tangent = np.sum(np.conj(self.lam)*dlam, axis=0)*self.model.dq
        self.assertLess(np.max(np.abs(phi_tangent)), 2.0e-14)
        self.assertLess(np.max(np.abs(lam_tangent)), 2.0e-14)
        self.assertTrue(np.isfinite(
            diagnostics["inverse_support_product_correction_chi"]
        ))


if __name__ == "__main__":
    unittest.main()
