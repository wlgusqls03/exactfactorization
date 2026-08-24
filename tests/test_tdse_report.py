from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from multi_component_exact_factorization import tdse_report


class TDSEReportTests(unittest.TestCase):
    def _write_archive(self, root):
        root = Path(root)
        q = np.linspace(-2.0, 2.0, 5, endpoint=False)
        R = np.linspace(-3.0, 3.0, 7, endpoint=False)
        dq, dR = q[1]-q[0], R[1]-R[0]
        times = np.array([0.0, 0.5, 1.0])
        joint = []
        for time in times:
            values = np.exp(-((q[:, None]+0.2*time)**2+(R[None, :]-0.3*time)**2))
            values /= np.sum(values)*dq*dR
            joint.append(values)
        joint = np.asarray(joint)
        proton = np.sum(joint, axis=2)*dR
        heavy = np.sum(joint, axis=1)*dq
        populations = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.7, 0.3],
        ])
        x = np.linspace(-4.0, 4.0, 8)
        electron = np.asarray([
            np.exp(-0.5*((x-0.1*time)/0.8)**2) for time in times
        ])
        electron /= np.sum(electron, axis=1)[:, None]*(x[1]-x[0])
        archive = root/"multi_component_discrete_tdse_gpu.npz"
        np.savez_compressed(
            archive,
            kind=np.array("direct_discrete_born_huang_tdse_gpu"),
            times_fs=times, q=q, R=R, x=x,
            norm=np.array([1.0, 1.0+1e-12, 1.0-2e-12]),
            energy=np.array([-0.5, -0.5+1e-13, -0.5-2e-13]),
            energy_imaginary_defect=np.array([1e-16, 2e-16, 3e-16]),
            norm_rate=np.array([1e-16, 2e-16, 1e-16]),
            bo_populations=populations,
            bo_energies=np.stack((
                -0.5+0.01*q[:, None]**2+0.01*R[None, :]**2,
                -0.3+0.02*q[:, None]**2+0.01*R[None, :]**2,
            )),
            joint_density=joint,
            proton_density=proton, heavy_density=heavy,
            electron_density=electron,
            outer_probability_q=np.array([1e-12, 2e-12, 3e-12]),
            outer_probability_R=np.array([1e-14, 2e-14, 3e-14]),
            fixed_center_crossing_q=np.array([1e-13, 2e-13, 3e-13]),
            fixed_center_crossing_R=np.array([1e-15, 2e-15, 3e-15]),
            args=np.array([{
                "electron_excitation": 1,
                "left_position": -1.5,
                "right_position": 1.5,
                "proton_mass": 1836.0,
                "heavy_mass": 3672.0,
            }], dtype=object),
        )
        shape = (len(times), len(q), len(R))
        np.savez_compressed(
            root/"tdse_exact_factorization_fields.npz",
            times_fs=times, q=q, R=R,
            epsilon_1=np.broadcast_to(q[None, :, None]+0.2*R[None, None, :], shape),
            epsilon_2=np.broadcast_to(R[None, :], (len(times), len(R))),
            a=np.full(shape, 0.03), b=np.full(shape, -0.02),
            alpha=np.full((len(times), len(R)), 0.01),
            sphi_q1=np.exp(1j*np.full(shape, 0.03*dq)),
            sphi_R1=np.exp(-1j*np.full(shape, 0.02*dR)),
            sgamma_R1=np.exp(1j*np.full((len(times), len(R)), 0.01*dR)),
            factorization_residual=np.zeros(len(times)),
        )
        return archive, joint

    def test_static_tdse_report_uses_reduced_archive_and_derived_fields(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, joint = self._write_archive(root)
            report = root/"report"
            obs = tdse_report.run(
                archive, report, no_animation=True, dpi=45,
                snapshot_count=3,
            )
            self.assertTrue(np.array_equal(obs["joint_density"], joint))
            for name in (
                "01_tdse_particle_motion.png",
                "02_tdse_joint_density.png",
                "03_tdse_electronic_dynamics.png",
                "04_tdse_numerical_reliability.png",
                "05_tdse_exact_factorization_fields.png",
                "06_tdse_transport_and_drive.png",
                "07_tdse_discrete_link_geometry.png",
                "08_tdse_joint_density_relative_log.png",
                "tdse_report_observables.npz",
            ):
                self.assertTrue((report/name).is_file(), name)

    def test_loader_never_requires_large_tdse_coefficients(self):
        with TemporaryDirectory() as temporary:
            archive, _ = self._write_archive(temporary)
            data = tdse_report.load_observables(archive)
            self.assertNotIn("tdse_coefficients", data)
            self.assertEqual(data["joint_density"].shape, (3, 5, 7))

    def test_all_tdse_movies_render_from_small_archive(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, _ = self._write_archive(root)
            report = root/"report"
            tdse_report.run(
                archive, report, no_animation=False, dpi=35,
                snapshot_count=2, max_frames=2, animation_dpi=25,
                fps=2, fmt="gif",
            )
            for name in (
                "tdse_dynamics_overview.gif",
                "tdse_joint_density_relative_log.gif",
                "particle_marginals_fixed_scale.gif",
                "particle_marginals_relative_log.gif",
                "tdse_exact_factorization_fields.gif",
                "tdse_transport_and_drive.gif",
                "heavy_coordinate_dynamics.gif",
                "proton_coordinate_dynamics.gif",
            ):
                self.assertTrue((report/name).is_file(), name)

    def test_support_aware_phase_lift_ignores_empty_tail_winding(self):
        spacing = 0.2
        density = np.zeros((2, 7))
        density[:, 3:6] = np.array([0.5, 1.0, 0.5])
        phase = np.array([
            [0.0, 0.0, 0.0, 0.1, 0.12, 0.14, 0.0],
            [0.0, 2.9, -2.9, 0.11, 0.13, 0.15, 0.0],
        ])
        lifted, support, turns = tdse_report.support_aware_temporal_lift_1d(
            phase/spacing, density, spacing, floor=1.0e-3,
        )
        self.assertTrue(np.all(support[:, 3:6]))
        self.assertTrue(np.allclose(lifted[0, 3:6]*spacing, phase[0, 3:6]))
        self.assertTrue(np.allclose(lifted[1, 3:6]*spacing, phase[1, 3:6]))
        self.assertTrue(np.array_equal(turns, np.zeros(2, dtype=int)))
        naive = np.unwrap(phase, axis=1)/spacing
        self.assertGreater(abs(naive[1, 4]-lifted[1, 4]), 20.0)

    def test_continuity_current_is_branch_free_and_finite(self):
        coordinate = np.linspace(-2.0, 2.0, 41)
        times = np.linspace(0.0, 1.0, 7)
        density = np.asarray([
            np.exp(-((coordinate-0.2*time)/0.45)**2) for time in times
        ])
        density /= np.sum(density, axis=1)[:, None]*(coordinate[1]-coordinate[0])
        current = tdse_report.continuity_current_1d(
            density, times, coordinate[1]-coordinate[0],
        )
        self.assertEqual(current.shape, density.shape)
        self.assertTrue(np.all(np.isfinite(current)))
        self.assertGreater(np.max(np.abs(current)), 0.0)


if __name__ == "__main__":
    unittest.main()
