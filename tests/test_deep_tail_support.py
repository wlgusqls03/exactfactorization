import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multi_component_exact_factorization.core import (
    build_model,
    deep_tail_gate,
    pnc_project,
    project_discrete_product_residual,
    reconstruct_psi,
    soft_inverse,
)
from multi_component_exact_factorization.propagate import parse_args


class DeepTailSupportTests(unittest.TestCase):
    def test_gate_has_exact_zero_one_and_smooth_transition(self):
        density = np.array([1.0e-14, 1.0e-13, 1.0e-12, 1.0e-11, 1.0])
        gate = deep_tail_gate(density, 1.0e-12)
        self.assertEqual(gate[0], 0.0)
        self.assertEqual(gate[-2], 1.0)
        self.assertEqual(gate[-1], 1.0)
        self.assertAlmostEqual(gate[2], 0.5, places=14)
        self.assertTrue(np.all(np.diff(gate) >= 0.0))
        self.assertTrue(np.array_equal(
            deep_tail_gate(density, 0.0), np.ones_like(density)
        ))

    def test_support_aware_pnc_preserves_product_and_skips_empty_slice(self):
        model = SimpleNamespace(
            dx=1.0, dq=1.0, deep_tail_zero_threshold=1.0e-8
        )
        phi = np.ones((2, 2, 2), complex)
        phi[:, 0, 0] *= 3.0
        phi[:, 1, 1] *= 2.0
        lam = np.ones((2, 2), complex)
        chi = np.array([1.0e-12, 1.0], complex)
        before = reconstruct_psi(phi, lam, chi)
        projected = pnc_project(phi, lam, chi, model)
        phi_out, lam_out, chi_out, _ = projected
        self.assertTrue(np.allclose(
            reconstruct_psi(phi_out, lam_out, chi_out), before,
            atol=1.0e-14, rtol=1.0e-14,
        ))
        # The R=0 full-Psi marginal is below the exact-zero cutoff.
        self.assertTrue(np.array_equal(phi_out[:, :, 0], phi[:, :, 0]))
        self.assertAlmostEqual(
            np.sum(np.abs(phi_out[:, 1, 1])**2)*model.dx, 1.0
        )

    def test_product_inverse_correction_is_exactly_zero_in_empty_tail(self):
        model = SimpleNamespace(
            dx=1.0, dq=2.0*np.pi/5, dR=2.0*np.pi/5,
            proton_mass=7.0, heavy_mass=19.0,
            product_projection_backend="weighted_tikhonov",
            projection_tau_phi=1.0e-10,
            projection_tau_lam=1.0e-10,
            projection_tau_chi=1.0e-10,
            projection_support_epsilon=1.0e-12,
            deep_tail_zero_threshold=1.0e-8,
        )
        phi = np.ones((5, 5, 5), complex)/np.sqrt(5.0)
        lam = np.ones((5, 5), complex)/np.sqrt(5.0*model.dq)
        chi = np.ones(5, complex)
        chi[0] = 1.0e-12
        zeros = (np.zeros_like(phi), np.zeros_like(lam), np.zeros_like(chi))
        dphi, _, _, _ = project_discrete_product_residual(
            phi, lam, chi, *zeros, model,
            support_floor_phi=1.0e-10, support_floor_lam=1.0e-10,
        )
        self.assertTrue(np.array_equal(dphi[:, :, 0], zeros[0][:, :, 0]))

    def test_full_nuclear_range_matches_electronic_box(self):
        with patch("sys.argv", ["propagate", "--full-nuclear-range"]):
            args = parse_args()
        model = build_model(args)
        self.assertEqual(args.q_min, args.left_position)
        self.assertEqual(args.R_min, args.left_position)
        self.assertEqual(args.q_max, args.x_max)
        self.assertEqual(args.R_max, args.x_max)
        self.assertAlmostEqual(model.q[0], args.left_position)
        self.assertAlmostEqual(model.R[0], args.left_position)
        self.assertAlmostEqual(model.dq, 14.0/args.nq)
        self.assertAlmostEqual(model.dR, 14.0/args.nR)

    def test_symmetric_box_preset_updates_effective_ranges(self):
        argv = [
            "propagate", "--symmetric-box-half-width", "10",
            "--full-nuclear-range", "--nx", "249", "--nq", "500",
            "--nR", "1000",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        model = build_model(args)
        self.assertEqual((model.x_left, model.x_right), (-10.0, 10.0))
        self.assertEqual((args.left_position, args.x_max), (-10.0, 10.0))
        self.assertEqual((args.q_min, args.q_max), (-10.0, 10.0))
        self.assertEqual((args.R_min, args.R_max), (-10.0, 10.0))
        self.assertAlmostEqual(model.dx, 0.08)
        self.assertAlmostEqual(model.dq, 0.04)
        self.assertAlmostEqual(model.dR, 0.02)

    def test_right_fixed_charge_adds_complete_soft_coulomb_terms(self):
        argv = [
            "propagate", "--nx", "5", "--nq", "5", "--nR", "5",
            "--left-charge", "0.7",
        ]
        with patch("sys.argv", argv):
            args_zero = parse_args()
        model_zero = build_model(args_zero)
        with patch("sys.argv", argv + ["--right-charge", "1.3"]):
            args_right = parse_args()
        model_right = build_model(args_right)

        xx = model_right.x[:, None, None]
        qq = model_right.q[None, :, None]
        RR = model_right.R[None, None, :]
        expected = 1.3 * (
            -soft_inverse(xx-model_right.x_right, args_right.soft_e_right)
            +soft_inverse(qq-model_right.x_right, args_right.soft_p_right)
            +args_right.heavy_charge
            * soft_inverse(RR-model_right.x_right, args_right.soft_right_heavy)
            +args_right.left_charge
            * soft_inverse(
                model_right.x_right-model_right.x_left,
                args_right.soft_left_right,
            )
        )
        self.assertTrue(np.allclose(
            model_right.potential-model_zero.potential, expected,
            rtol=1.0e-14, atol=1.0e-14,
        ))


if __name__ == "__main__":
    unittest.main()
