from unittest.mock import patch
import sys
import unittest

import numpy as np

from multi_component_exact_factorization.core import (
    build_model,
    initial_factors,
    instantaneous_functionals,
    reconstruct_psi,
    remove_local_norm_generator,
)
from multi_component_exact_factorization.propagate import coupled_rhs, parse_args


class LocalNormCorrectionHelperTests(unittest.TestCase):
    def test_recovers_known_antihermitian_component_with_nonunit_norm(self):
        rng = np.random.default_rng(20260731)
        factor = rng.normal(size=(7, 3, 2))+1j*rng.normal(size=(7, 3, 2))
        factor *= np.array([0.7, 1.0, 1.3])[None, :, None]
        eta = np.array([[0.037, -0.021], [0.012, 0.044], [-0.015, 0.029]])
        real_weight = np.array([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]])
        hermitian_action = real_weight[None, :, :]*factor
        raw_action = hermitian_action+1j*eta[None, :, :]*factor

        corrected, gamma, raw_rate, corrected_rate = remove_local_norm_generator(
            factor, raw_action, spacing=0.08, axis=0
        )

        self.assertTrue(np.allclose(gamma, eta, atol=2.0e-15, rtol=0.0))
        self.assertGreater(float(np.max(np.abs(raw_rate))), 1.0e-3)
        self.assertLess(float(np.max(np.abs(corrected_rate))), 2.0e-15)
        self.assertTrue(np.allclose(corrected, hermitian_action, atol=2.0e-15))


class SmoothGaugeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", [
            "test", "--electron-excitation", "1",
            "--heavy-trap-alpha", "0",
            "--erf-r-qr", "4.5",
            "--proton-force-constant", "0.1",
            "--heavy-force-constant", "0.1",
        ]):
            cls.args = parse_args()
        cls.model = build_model(cls.args)
        cls.phi, cls.lam, cls.chi = initial_factors(cls.model, cls.args)

    def smooth_gauge_factors(self):
        model = self.model
        Lq = len(model.q)*model.dq
        LR = len(model.R)*model.dR
        theta1 = (
            0.7*np.sin(2.0*np.pi*(model.q-model.q[0])/Lq)[:, None]
            +0.4*np.sin(2.0*np.pi*(model.R-model.R[0])/LR)[None, :]
        )
        theta2 = 0.5*np.sin(2.0*np.pi*(model.R-model.R[0])/LR)
        phi = self.phi*np.exp(1j*theta1)[None, :, :]
        lam = self.lam*np.exp(-1j*theta1+1j*theta2[None, :])
        chi = self.chi*np.exp(-1j*theta2)
        return phi, lam, chi

    def test_nested_smooth_gauge_preserves_psi_and_removes_raw_rates(self):
        phi, lam, chi = self.smooth_gauge_factors()
        before = reconstruct_psi(self.phi, self.lam, self.chi)
        after = reconstruct_psi(phi, lam, chi)
        self.assertLess(float(np.max(np.abs(after-before))), 5.0e-16)

        fields = instantaneous_functionals(
            phi, lam, chi, self.model, floor=self.args.ratio_floor,
            mask_threshold_phi=self.args.mask_threshold_phi,
            mask_threshold_lam=self.args.mask_threshold_lam,
        )
        self.assertGreater(
            max(
                float(np.max(np.abs(fields["raw_rate_phi"]))),
                float(np.max(np.abs(fields["raw_rate_lam"]))),
            ),
            1.0e-8,
        )
        self.assertLess(
            float(np.max(np.abs(fields["corrected_rate_phi"]))), 1.0e-12
        )
        self.assertLess(
            float(np.max(np.abs(fields["corrected_rate_lam"]))), 1.0e-12
        )
        self.assertTrue(np.allclose(
            fields["hpr_lam"],
            fields["base_lam"]+fields["epsilon_1"]*lam,
            atol=2.0e-13,
            rtol=2.0e-13,
        ))

    def test_nested_gamma_transfer_preserves_full_product_pointwise(self):
        phi, lam, chi = self.smooth_gauge_factors()
        fields = instantaneous_functionals(
            phi, lam, chi, self.model, floor=self.args.ratio_floor,
            mask_threshold_phi=self.args.mask_threshold_phi,
            mask_threshold_lam=self.args.mask_threshold_lam,
        )
        gamma_phi = fields["gamma_phi"]
        gamma_lam = fields["gamma_lam"]
        self.assertTrue(np.allclose(
            fields["support_gamma_phi"],
            fields["tail_gate_phi"]*fields["mask_phi"]*gamma_phi,
        ))
        self.assertTrue(np.allclose(
            fields["support_gamma_lam"],
            fields["tail_gate_lam"]*fields["mask_lam"]*gamma_lam,
        ))
        delta_phi = -gamma_phi[None, :, :]*phi
        delta_lam = (gamma_phi-gamma_lam[None, :])*lam
        delta_chi = gamma_lam*chi
        delta_psi = (
            delta_phi*lam[None, :, :]*chi[None, None, :]
            +phi*delta_lam[None, :, :]*chi[None, None, :]
            +phi*lam[None, :, :]*delta_chi[None, None, :]
        )
        self.assertLess(float(np.max(np.abs(delta_psi))), 2.0e-14)


if __name__ == "__main__":
    unittest.main()
