from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from multi_component_exact_factorization import born_huang_report
from multi_component_exact_factorization.report_plot_style import (
    density_display_alpha,
    density_weighted_shift,
)


class BornHuangReportTests(unittest.TestCase):
    def synthetic_data(self):
        nt, ns, nq, nR = 3, 3, 7, 5
        x = np.linspace(-5.0, 5.0, 11)
        q = np.linspace(-2.0, 2.0, nq, endpoint=False)
        R = np.linspace(3.0, 5.0, nR, endpoint=False)
        dq, dR = q[1]-q[0], R[1]-R[0]
        lam = np.ones((nt, nq, nR), complex)/np.sqrt(nq*dq)
        chi = np.ones((nt, nR), complex)/np.sqrt(nR*dR)
        norm = np.array([1.0, 1.0001, 0.9999])
        populations = norm[:, None]*np.array([
            [0.0, 1.0, 0.0], [0.01, 0.98, 0.01], [0.02, 0.95, 0.03],
        ])
        energies = np.empty((ns, nq, nR))
        for state in range(ns):
            energies[state] = state+0.01*q[:, None]+0.02*R[None, :]
        qR_field = np.empty((nt, nq, nR))
        for frame in range(nt):
            qR_field[frame] = (
                0.1*frame+0.03*q[:, None]-0.02*R[None, :]
            )
        return dict(
            times_fs=np.array([0.0, 0.5, 1.0]), x=x, q=q, R=R,
            lambda_wavefunction=lam, chi=chi, norm=norm,
            bo_populations=populations, bo_energies=energies,
            electron_density=np.asarray([
                np.exp(-0.5*((x-(0.1*frame))/1.2)**2)
                for frame in range(nt)
            ]),
            epsilon_1=qR_field, a=0.2*qR_field, b=-0.1*qR_field,
            epsilon_2=np.array([
                0.1*frame+0.02*R for frame in range(nt)
            ]),
            alpha=np.array([
                0.01*frame-0.005*R for frame in range(nt)
            ]),
            args=np.array([{
                "electron_excitation": 1, "proton_mass": 1836.0,
                "heavy_mass": 20000.0,
            }], dtype=object),
        )

    def test_observables_are_normalized_and_population_corrected(self):
        data = self.synthetic_data()
        obs = born_huang_report.calculate_observables(data)
        dq, dR = obs["q"][1]-obs["q"][0], obs["R"][1]-obs["R"][0]
        self.assertTrue(np.allclose(
            np.sum(obs["nuclear_joint_density"], axis=(1, 2))*dq*dR, 1.0
        ))
        self.assertTrue(np.allclose(
            np.sum(obs["normalized_state_populations"], axis=1), 1.0
        ))
        self.assertEqual(obs["mean_bo_energies"].shape, (3, 3))

    def test_plotting_support_does_not_change_saved_field_values(self):
        values = np.array([9.0, 2.0, -1.0, 7.0])
        original = values.copy()
        density = np.array([1.0e-8, 0.4, 0.6, 1.0e-8])
        shifted = density_weighted_shift(values, density, floor=1.0e-3)
        support = density >= 1.0e-3*np.max(density)
        self.assertTrue(np.array_equal(values, original))
        self.assertAlmostEqual(
            float(np.sum(density[support]*shifted[support])), 0.0, places=14
        )
        opacity = density_display_alpha(density, floor=1.0e-3)
        self.assertTrue(np.all((opacity >= 0.0) & (opacity <= 1.0)))
        self.assertEqual(float(opacity[0]), 0.0)
        self.assertEqual(float(opacity[1]), 1.0)

    def test_display_opacity_is_strictly_closed_under_roundoff(self):
        density = np.logspace(-300, 100, 10001)
        density = np.r_[density, np.nan, -1.0, 0.0]
        opacity = density_display_alpha(density, floor=1.0e-3)
        self.assertTrue(np.all(np.isfinite(opacity)))
        self.assertGreaterEqual(float(np.min(opacity)), 0.0)
        self.assertLessEqual(float(np.max(opacity)), 1.0)
        self.assertEqual(float(opacity[-3]), 0.0)
        self.assertEqual(float(opacity[-2]), 0.0)
        self.assertEqual(float(opacity[-1]), 0.0)

    def test_connections_share_one_plot_scale(self):
        data = self.synthetic_data()
        obs = born_huang_report.calculate_observables(data)
        diagnostics = born_huang_report._diagnostics(data)
        fields = [
            born_huang_report._potential_frame_fields(
                data, obs, diagnostics, frame
            )
            for frame in range(len(obs["times_fs"]))
        ]
        limits = born_huang_report._potential_limits(fields)
        self.assertEqual(limits["a"], limits["b"])
        self.assertTrue(np.allclose(
            fields[-1]["eps2"][fields[-1]["heavy_support"]],
            fields[-1]["eps2_full"][fields[-1]["heavy_support"]],
        ))

    def test_static_report_does_not_require_coefficients_or_basis_states(self):
        data = self.synthetic_data()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root/"multi_component_born_huang_ef_gpu.npz"
            np.savez(archive, **data)
            report = root/"report"
            born_huang_report.run(
                archive, report, no_animation=True, dpi=50
            )
            for name in (
                "01_particle_motion.png",
                "02_electronic_transitions.png",
                "03_exact_potentials.png",
                "04_numerical_reliability.png",
                "05_born_huang_surface_dynamics.png",
                "report_observables.npz",
            ):
                self.assertTrue((report/name).is_file(), name)

    def test_three_animations_write_without_electronic_grid(self):
        data = self.synthetic_data()
        obs = born_huang_report.calculate_observables(data)
        diagnostics = born_huang_report._diagnostics(data)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            born_huang_report.make_overview_animation(
                data, obs, diagnostics, root, fps=2, max_frames=3, dpi=35, fmt="gif"
            )
            born_huang_report.make_potential_animation(
                data, obs, root, fps=2, max_frames=3, dpi=35, fmt="gif"
            )
            born_huang_report.make_state_ladder_animation(
                data, obs, root, fps=2, max_frames=3, dpi=35, fmt="gif"
            )
            for name in (
                "mcef_dynamics_overview.gif",
                "mcef_exact_potentials.gif",
                "mcef_physical_interpretation.gif",
            ):
                self.assertGreater((root/name).stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()
