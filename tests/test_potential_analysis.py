from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from multi_component_exact_factorization.compare_observables import compare
from multi_component_exact_factorization.potential_analysis import (
    gauge_invariant_diagnostics,
    phase_gradient,
)
from multi_component_exact_factorization.visualize import (
    LoadedArchive,
    supported_potential_fields,
)


class SupportedPotentialTests(unittest.TestCase):
    def test_masks_unoccupied_tail_without_changing_raw_support(self):
        lam = np.array([[[0.01, 1.0], [0.02, 2.0]]], complex)
        chi = np.array([[0.01, 1.0]], complex)
        raw2 = np.arange(4.0).reshape(1, 2, 2)
        raw1 = np.arange(2.0).reshape(1, 2)
        data = LoadedArchive({
            "lambda_wavefunction": lam, "chi": chi,
            "epsilon_1": raw2, "a": raw2+1, "b": raw2+2,
            "epsilon_2": raw1, "alpha": raw1+1,
        })
        fields = supported_potential_fields(data, 0, support_floor=1.0e-3)
        self.assertTrue(np.isnan(fields["epsilon_1"][0, 0]))
        self.assertEqual(fields["epsilon_1"][1, 1], raw2[0, 1, 1])
        self.assertTrue(np.isnan(fields["epsilon_2"][0]))
        self.assertEqual(fields["epsilon_2"][1], raw1[0, 1])


class GaugeInvariantDiagnosticsTests(unittest.TestCase):
    def test_constant_factors_have_zero_currents_and_forces(self):
        nt, nq, nR = 3, 6, 5
        q = np.arange(nq, dtype=float)
        R = np.arange(nR, dtype=float)
        lam = np.ones((nt, nq, nR), complex)/np.sqrt(nq)
        chi = np.ones((nt, nR), complex)/np.sqrt(nR)
        data = LoadedArchive({
            "times_fs": np.linspace(0.0, 0.1, nt), "q": q, "R": R,
            "lambda_wavefunction": lam, "chi": chi,
            "a": np.zeros((nt, nq, nR)), "b": np.zeros((nt, nq, nR)),
            "alpha": np.zeros((nt, nR)),
            "epsilon_1": np.zeros((nt, nq, nR)),
            "epsilon_2": np.zeros((nt, nR)),
            "args": np.array([{"proton_mass": 2.0, "heavy_mass": 5.0}], dtype=object),
        })
        diagnostics = gauge_invariant_diagnostics(data)
        for key in (
            "momentum_q", "momentum_R_outer", "proton_current",
            "heavy_current", "force_q", "force_R", "curvature_qR",
        ):
            self.assertLess(float(np.max(np.abs(diagnostics[key]))), 1.0e-14)

    def test_nested_periodic_gauge_preserves_mechanical_momenta(self):
        nt, nq, nR = 3, 12, 10
        q = np.linspace(-1.0, 1.0, nq, endpoint=False)
        R = np.linspace(3.0, 5.0, nR, endpoint=False)
        dq, dR = q[1]-q[0], R[1]-R[0]
        theta1 = (
            0.23*np.sin(2*np.pi*(q[:, None]-q[0])/(nq*dq))
            +0.17*np.sin(2*np.pi*(R[None, :]-R[0])/(nR*dR))
        )
        theta2 = 0.19*np.sin(2*np.pi*(R-R[0])/(nR*dR))
        lam = np.ones((nt, nq, nR), complex)/np.sqrt(nq*dq)
        chi = np.ones((nt, nR), complex)/np.sqrt(nR*dR)
        phase1 = np.exp(1j*theta1)[None, :, :]
        phase2 = np.exp(1j*theta2)[None, :]
        transformed_lam = lam*phase2[:, None, :]/phase1
        transformed_chi = chi/phase2
        a = np.broadcast_to(
            phase_gradient(phase1, dq, axis=1), (nt, nq, nR)
        ).copy()
        b = np.broadcast_to(
            phase_gradient(phase1, dR, axis=2), (nt, nq, nR)
        ).copy()
        alpha = np.broadcast_to(
            phase_gradient(phase2, dR, axis=1), (nt, nR)
        ).copy()
        data = LoadedArchive({
            "times_fs": np.linspace(0.0, 0.1, nt), "q": q, "R": R,
            "lambda_wavefunction": transformed_lam, "chi": transformed_chi,
            "a": a, "b": b, "alpha": alpha,
            "epsilon_1": np.zeros((nt, nq, nR)),
            "epsilon_2": np.zeros((nt, nR)),
            "args": np.array([{"proton_mass": 2.0, "heavy_mass": 5.0}], dtype=object),
        })
        diagnostics = gauge_invariant_diagnostics(data)
        for key in (
            "momentum_q", "momentum_R_outer", "proton_current",
            "heavy_current", "curvature_qR",
        ):
            self.assertLess(float(np.max(np.abs(diagnostics[key]))), 1.0e-13)


class ObservableComparisonTests(unittest.TestCase):
    def test_matches_common_times_and_reports_population_error(self):
        with TemporaryDirectory() as temporary:
            paths = [Path(temporary)/name for name in ("reference.npz", "test.npz")]
            times = np.array([0.0, 0.5, 1.0])
            common = dict(
                times_fs=times, x=np.arange(2.0), q=np.arange(2.0), R=np.arange(2.0),
                electron_density=np.full((3, 2), 0.5),
                proton_density=np.full((3, 2), 0.5),
                heavy_density=np.full((3, 2), 0.5),
                electron_mean=np.zeros(3), proton_mean=np.zeros(3), heavy_mean=np.zeros(3),
                electron_width=np.ones(3), proton_width=np.ones(3), heavy_width=np.ones(3),
                electron_left_population=np.full(3, 0.5),
                electron_right_population=np.full(3, 0.5),
                electron_rearranged_density=np.zeros(3),
                state_basis_residual=np.zeros(3),
            )
            np.savez(paths[0], **common, state_populations=np.tile([0.2, 0.8], (3, 1)))
            changed = dict(common)
            changed["proton_mean"] = np.array([0.0, 0.01, 0.02])
            np.savez(paths[1], **changed, state_populations=np.tile([0.25, 0.75], (3, 1)))
            result = compare(paths[0], paths[1])
            self.assertEqual(result["common_frames"], 3)
            self.assertAlmostEqual(result["max_abs_proton_mean"], 0.02)
            self.assertAlmostEqual(result["max_abs_population_0"], 0.05)


if __name__ == "__main__":
    unittest.main()
