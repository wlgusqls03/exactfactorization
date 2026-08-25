import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multi_component_exact_factorization.core import (
    build_model,
    bare_inverse,
    crossing_reference_positions,
    deep_tail_gate,
    pnc_project,
    project_discrete_product_residual,
    reconstruct_psi,
    erf_inverse,
    soft_inverse,
)
from multi_component_exact_factorization.propagate import parse_args


class DeepTailSupportTests(unittest.TestCase):
    def test_default_geometry_separates_fixed_center_and_electron_wall(self):
        with patch("sys.argv", [
            "propagate", "--heavy-trap-alpha", "0.05",
            "--erf-r-qr", "4.5",
        ]):
            args = parse_args()
        model = build_model(args)

        self.assertEqual((model.x_left, model.x_right), (-22.0, 22.0))
        self.assertEqual((args.left_position, args.left_charge), (-9.5, 1.0))
        self.assertEqual((args.right_position, args.right_charge), (9.5, 0.0))
        self.assertEqual((args.q_min, args.q_max, args.q0), (-12.0, 12.0, 0.0))
        self.assertEqual((args.R_min, args.R_max, args.R0), (2.0, 18.0, 10.0))
        self.assertEqual((model.heavy_trap_center, model.heavy_trap_alpha), (9.5, 0.05))
        self.assertEqual((args.nx, args.nq, args.nR), (151, 151, 151))
        self.assertAlmostEqual(model.dx, 44.0/152.0)
        self.assertAlmostEqual(model.dq, 24.0/151.0)
        self.assertAlmostEqual(model.dR, 16.0/151.0)
        self.assertEqual(
            crossing_reference_positions(model, args, "q"), (-9.5, 9.5)
        )
        R_references = crossing_reference_positions(model, args, "R")
        self.assertEqual(R_references[0], -9.5)
        self.assertAlmostEqual(R_references[1], 18.0)

    def test_harmonic_heavy_trap_is_real_diagonal_and_centered(self):
        argv = [
            "propagate", "--nx", "5", "--nq", "5", "--nR", "32",
            "--heavy-trap-alpha", "0", "--erf-r-qr", "4.5",
        ]
        with patch("sys.argv", argv):
            args_free = parse_args()
        free = build_model(args_free)
        trapped_argv = list(argv)
        trapped_argv[trapped_argv.index("--heavy-trap-alpha")+1] = "0.125"
        with patch("sys.argv", trapped_argv):
            args_trapped = parse_args()
        trapped = build_model(args_trapped)
        expected = 0.125*(trapped.R-9.5)**2
        difference = trapped.potential-free.potential
        self.assertTrue(np.allclose(
            difference, expected[None, None, :], atol=2.0e-14, rtol=0.0,
        ))
        center = int(np.argmin(np.abs(trapped.R-9.5)))
        self.assertAlmostEqual(expected[center], 0.0, places=14)

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
        with patch("sys.argv", [
            "propagate", "--full-nuclear-range",
            "--heavy-trap-alpha", "0", "--erf-r-qr", "4.5",
        ]):
            args = parse_args()
        model = build_model(args)
        self.assertEqual(args.q_min, args.x_min)
        self.assertEqual(args.R_min, args.x_min)
        self.assertEqual(args.q_max, args.x_max)
        self.assertEqual(args.R_max, args.x_max)
        self.assertAlmostEqual(model.q[0], args.x_min)
        self.assertAlmostEqual(model.R[0], args.x_min)
        self.assertAlmostEqual(model.dq, 44.0/args.nq)
        self.assertAlmostEqual(model.dR, 44.0/args.nR)

    def test_symmetric_box_preset_updates_effective_ranges(self):
        argv = [
            "propagate", "--symmetric-box-half-width", "10",
            "--full-nuclear-range", "--nx", "249", "--nq", "500",
            "--nR", "1000", "--heavy-trap-alpha", "0",
            "--erf-r-qr", "4.5",
            "--interaction-model", "legacy-soft-coulomb",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        model = build_model(args)
        self.assertEqual((model.x_left, model.x_right), (-10.0, 10.0))
        self.assertEqual((args.x_min, args.x_max), (-10.0, 10.0))
        self.assertEqual(args.left_position, -10.0)
        self.assertEqual((args.q_min, args.q_max), (-10.0, 10.0))
        self.assertEqual((args.R_min, args.R_max), (-10.0, 10.0))
        self.assertAlmostEqual(model.dx, 0.08)
        self.assertAlmostEqual(model.dq, 0.04)
        self.assertAlmostEqual(model.dR, 0.02)

    def test_right_fixed_charge_adds_complete_soft_coulomb_terms(self):
        argv = [
            "propagate", "--nx", "5", "--nq", "5", "--nR", "5",
            "--left-charge", "0.7", "--right-charge", "0.0",
            "--heavy-trap-alpha", "0",
            "--interaction-model", "legacy-soft-coulomb",
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
            -soft_inverse(xx-args_right.right_position, args_right.soft_e_right)
            +soft_inverse(qq-args_right.right_position, args_right.soft_p_right)
            +args_right.heavy_charge
            * soft_inverse(
                RR-args_right.right_position, args_right.soft_right_heavy
            )
            +args_right.left_charge
            * soft_inverse(
                args_right.right_position-args_right.left_position,
                args_right.soft_left_right,
            )
        )
        self.assertTrue(np.allclose(
            model_right.potential-model_zero.potential, expected,
            rtol=1.0e-14, atol=1.0e-14,
        ))

    def test_erf_kernel_has_analytic_origin_and_new_hamiltonian_terms(self):
        origin = erf_inverse(np.array([0.0]), 3.1)[0]
        self.assertAlmostEqual(origin, 2.0/(np.sqrt(np.pi)*3.1), places=15)
        argv = [
            "propagate", "--nx", "7", "--nq", "6", "--nR", "5",
            "--heavy-trap-alpha", "0.02", "--erf-r-qr", "4.7",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        model = build_model(args)
        self.assertEqual(model.interaction_model, "erf_shin_metiu")
        self.assertEqual(model.fixed_ion_separation, 19.0)
        self.assertTrue(np.all(np.isfinite(model.potential)))
        self.assertEqual((args.left_position, args.heavy_trap_center), (-9.5, 9.5))
        xx = model.x[:, None, None]
        qq = model.q[None, :, None]
        RR = model.R[None, None, :]
        expected = (
            bare_inverse(9.5+qq, label="test q-left")
            -erf_inverse(9.5+xx, 3.1)
            +args.heavy_charge*bare_inverse(9.5+RR, label="test R-left")
            -erf_inverse(qq-xx, 5.0)
            -erf_inverse(RR-xx, 4.0)
            +erf_inverse(qq-RR, 4.7)
            +0.02*(RR-9.5)**2
        )
        self.assertTrue(np.allclose(
            model.potential, expected, rtol=2.0e-15, atol=2.0e-15,
        ))

    def test_literature_coupling_presets_and_explicit_overrides(self):
        base = [
            "propagate", "--nx", "5", "--nq", "6", "--nR", "7",
            "--heavy-trap-alpha", "0", "--erf-r-qr", "4.5",
        ]
        with patch("sys.argv", base):
            strong_args = parse_args()
        strong = build_model(strong_args)
        self.assertEqual(strong.coupling_regime, "strong")
        self.assertEqual(
            (strong.erf_r_lx, strong.erf_r_qx, strong.erf_r_Rx),
            (3.1, 5.0, 4.0),
        )

        with patch("sys.argv", base + ["--coupling-regime", "weak"]):
            weak_args = parse_args()
        weak = build_model(weak_args)
        self.assertEqual(weak.coupling_regime, "weak")
        self.assertEqual(
            (weak.erf_r_lx, weak.erf_r_qx, weak.erf_r_Rx),
            (2.9, 3.8, 5.5),
        )

        with patch("sys.argv", base + [
            "--coupling-regime", "weak", "--erf-r-qx", "4.2",
        ]):
            override_args = parse_args()
        override = build_model(override_args)
        self.assertEqual(override.coupling_regime, "weak+override")
        self.assertEqual(
            (override.erf_r_lx, override.erf_r_qx, override.erf_r_Rx),
            (2.9, 4.2, 5.5),
        )

    def test_custom_coupling_regime_requires_all_three_ranges(self):
        argv = [
            "propagate", "--nx", "5", "--nq", "6", "--nR", "7",
            "--heavy-trap-alpha", "0", "--erf-r-qr", "4.5",
            "--coupling-regime", "custom", "--erf-r-lx", "3.0",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        with self.assertRaisesRegex(ValueError, "모두 명시"):
            build_model(args)


if __name__ == "__main__":
    unittest.main()
