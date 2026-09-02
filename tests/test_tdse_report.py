import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from multi_component_exact_factorization import (
    report_plot_style,
    render_final_visualizations,
    render_tdse_tdpes_gauges,
    tdse_collision_report,
    tdse_report,
)


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
        state_density_q = populations[:, :, None]*proton[:, None, :]
        state_density_R = populations[:, :, None]*heavy[:, None, :]
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
        epsilon_1 = np.broadcast_to(
            q[None, :, None]+0.2*R[None, None, :], shape,
        )
        epsilon_2 = np.broadcast_to(R[None, :], (len(times), len(R)))
        a = np.broadcast_to(
            (0.03+0.004*times)[:, None, None], shape,
        ).copy()
        b = np.full(shape, -0.02)
        alpha = np.broadcast_to(
            (0.01-0.002*times)[:, None], (len(times), len(R)),
        ).copy()
        np.savez_compressed(
            root/"tdse_exact_factorization_fields.npz",
            times_fs=times, q=q, R=R,
            epsilon_1=epsilon_1,
            epsilon_1_gi=0.65*epsilon_1,
            epsilon_2=epsilon_2,
            epsilon_2_gi=0.55*epsilon_2,
            a=a, b=b, alpha=alpha,
            sphi_q1=np.exp(1j*a*dq),
            sphi_R1=np.exp(-1j*np.full(shape, 0.02*dR)),
            sgamma_R1=np.exp(1j*alpha*dR),
            bo_state_density_q=state_density_q,
            bo_state_density_R=state_density_R,
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
            for gauge in ("positive_gauge", "zero_potential_gauge"):
                for name in (
                    "01_tdse_particle_motion.png",
                    "02_tdse_joint_density.png",
                    "03_tdse_electronic_dynamics.png",
                    "04_tdse_numerical_reliability.png",
                    "05_tdse_exact_factorization_fields.png",
                    "06_tdse_transport_and_drive.png",
                    "07_tdse_discrete_link_geometry.png",
                    "08_tdse_joint_density_relative_log.png",
                    "09_tdse_collision_snapshots.png",
                    "10_tdse_relative_collision_diagnostics.png",
                    "11_tdse_tdpes_gi_gd_decomposition.png",
                    "tdse_collision_observables.npz",
                    "tdse_report_observables.npz",
                ):
                    self.assertTrue((report/gauge/name).is_file(), (gauge, name))

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
            for gauge in ("positive_gauge", "zero_potential_gauge"):
                for name in (
                    "tdse_dynamics_overview.gif",
                    "tdse_joint_density_relative_log.gif",
                    "particle_marginals_fixed_scale.gif",
                    "particle_marginals_relative_log.gif",
                    "tdse_bo_surface_dynamics.gif",
                    "tdse_exact_factorization_fields.gif",
                    "tdse_all_exact_potentials.gif",
                    "tdse_transport_and_drive.gif",
                    "tdse_tdpes_gi_gd_decomposition.gif",
                    "heavy_coordinate_dynamics.gif",
                    "proton_coordinate_dynamics.gif",
                    "tdse_collision_dynamics.gif",
                ):
                    self.assertTrue((report/gauge/name).is_file(), (gauge, name))

    def test_axial_gauge_preserves_transport_force_and_tdpes_identity(self):
        with TemporaryDirectory() as temporary:
            archive, _ = self._write_archive(temporary)
            obs = tdse_report.calculate_observables(
                tdse_report.load_observables(archive)
            )
            ef = tdse_report._load_ef_fields(obs)
            positive = [
                tdse_report._ef_frame(obs, ef, frame)
                for frame in range(len(obs["times_fs"]))
            ]
            # Clear derived caches before changing the native links/scalars.
            ef.pop("_prepared_geometry", None)
            ef.pop("plot_limits", None)
            tdse_report.transform_to_zero_potential_gauge(obs, ef)
            zero = [
                tdse_report._ef_frame(obs, ef, frame)
                for frame in range(len(obs["times_fs"]))
            ]
            # The last index is the periodic closing seam carrying the Wilson
            # loop; every ordinary forward bond is axial-gauge zero.
            self.assertLess(np.max(np.abs(ef["a"][:, :-1, :])), 2.0e-15)
            self.assertLess(np.max(np.abs(ef["alpha"][:, :-1])), 2.0e-15)
            for before, after in zip(positive, zero):
                for key in (
                    "momentum_q_full", "momentum_R_first_full",
                    "momentum_R_full", "proton_current_full",
                    "first_heavy_current_full", "heavy_current_full",
                ):
                    self.assertTrue(np.allclose(
                        before[key], after[key], rtol=0.0, atol=2.0e-14,
                    ), key)
                self.assertTrue(np.allclose(
                    before["force_q_full"][:-1],
                    after["force_q_full"][:-1],
                    rtol=0.0, atol=2.0e-13,
                ))
                self.assertTrue(np.allclose(
                    before["force_R_full"][:-1],
                    after["force_R_full"][:-1],
                    rtol=0.0, atol=2.0e-13,
                ))
            for frame in range(len(obs["times_fs"])):
                pieces = tdse_report._tdpes_components_frame(obs, ef, frame)
                self.assertTrue(np.allclose(
                    pieces["epsilon_1_total"],
                    pieces["epsilon_1_gi"]+pieces["epsilon_1_gd"],
                    rtol=0.0, atol=2.0e-15,
                ))
                self.assertTrue(np.allclose(
                    pieces["epsilon_2_total"],
                    pieces["epsilon_2_gi"]+pieces["epsilon_2_gd"],
                    rtol=0.0, atol=2.0e-15,
                ))

    def test_tdpes_decomposition_uses_shared_level_wise_scales(self):
        with TemporaryDirectory() as temporary:
            archive, _ = self._write_archive(temporary)
            obs = tdse_report.calculate_observables(
                tdse_report.load_observables(archive)
            )
            ef = tdse_report._load_ef_fields(obs)
            limits = tdse_report._tdpes_decomposition_limits(obs, ef)
            first_level = [limits[key] for key in (
                "epsilon_1_total", "epsilon_1_gi", "epsilon_1_gd",
            )]
            second_level = [limits[key] for key in (
                "epsilon_2_total", "epsilon_2_gi", "epsilon_2_gd",
            )]
            self.assertTrue(all(item == first_level[0] for item in first_level))
            self.assertTrue(all(item == second_level[0] for item in second_level))
            self.assertNotEqual(first_level[0], second_level[0])

    def test_gauge_only_renderer_writes_both_static_products(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archive(root)
            output = root/"gauge_only"
            args = render_tdse_tdpes_gauges.parse_args([
                str(root), "--outdir", str(output), "--no-animation",
                "--dpi", "35", "--surface-count", "2",
            ])
            products = render_tdse_tdpes_gauges.run(args)
            self.assertEqual(len(products), 2)
            for gauge in ("positive_gauge", "zero_potential_gauge"):
                self.assertTrue((
                    output/gauge/"11_tdse_tdpes_gi_gd_decomposition.png"
                ).is_file())

    def test_final_visualization_command_reuses_reduced_tdse_products(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archive(root)
            output = root/"report"/"final_visualizations"
            args = render_final_visualizations.parse_args([
                str(root), "--no-animation",
                "--snapshot-count", "2", "--max-frames", "2",
                "--dpi", "30", "--heavy-min", "-3", "--heavy-max", "3",
            ])
            products = render_final_visualizations.run(args)
            for name in (
                "marginal_time_position_snapshots.png",
                "joint_density_qR_snapshots.png",
                "joint_velocity_snapshots.png",
                "vector_potential_composite_snapshots.png",
                "current_density_composite_snapshots.png",
                "heavy_analysis_snapshots.png",
                "bo_combined_snapshots.png",
                "final_visualizations_manifest.txt",
            ):
                self.assertTrue((output/name).is_file(), name)
            for directory in (
                "marginal_time_position_frames",
                "joint_density_qR_frames",
                "joint_velocity_frames",
                "vector_potential_composite_frames",
                "current_density_composite_frames",
                "heavy_analysis_frames",
                "bo_combined_frames",
            ):
                self.assertEqual(len(list((output/directory).glob("*.png"))), 2)
            self.assertGreaterEqual(len(products), 22)

    def test_joint_velocity_uses_mass_scaled_positive_gauge_connections(self):
        q = np.array([-1.0, 0.0, 1.0])
        R = np.array([2.0, 3.0])
        density = np.ones((1, len(q), len(R)))
        obs = {
            "q": q,
            "R": R,
            "times_fs": np.array([0.0]),
            "joint_density": density,
            "options": {"proton_mass": 2.0, "heavy_mass": 8.0},
        }
        ef = {
            "a": np.full_like(density, 4.0),
            "b": np.full_like(density, -8.0),
        }
        args = argparse.Namespace(
            max_frames=1, velocity_q_points=3, velocity_R_points=2,
            support_floor=1.0e-4,
        )
        prep = render_final_visualizations._joint_velocity_preparation(
            obs, ef, args,
        )
        velocity_q, velocity_R = (
            render_final_visualizations._joint_velocity_frame(
                obs, ef, prep, 0, args.support_floor,
            )
        )
        self.assertTrue(np.allclose(velocity_q.compressed(), 2.0))
        self.assertTrue(np.allclose(velocity_R.compressed(), -1.0))

    def test_joint_velocity_only_command_writes_movie_and_snapshots(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archive(root)
            output = root/"velocity_only"
            args = render_final_visualizations.parse_args([
                str(root), "--outdir", str(output), "--only", "velocity",
                "--format", "gif", "--snapshot-count", "2",
                "--max-frames", "2", "--dpi", "30",
                "--animation-dpi", "25", "--fps", "2",
                "--velocity-q-points", "3", "--velocity-R-points", "3",
            ])
            render_final_visualizations.run(args)
            self.assertTrue((output/"joint_velocity_movie.gif").is_file())
            self.assertTrue((output/"joint_velocity_snapshots.png").is_file())
            self.assertEqual(
                len(list((output/"joint_velocity_frames").glob("*.png"))),
                2,
            )

    def test_current_fields_follow_mcef_probability_current_definitions(self):
        q = np.array([-1.0, 0.0, 1.0])
        R = np.array([2.0, 3.0])
        dq, dR = 1.0, 1.0
        joint = np.full((1, len(q), len(R)), 1.0/6.0)
        heavy = np.sum(joint, axis=1)*dq
        obs = {
            "q": q,
            "R": R,
            "dq": dq,
            "dR": dR,
            "times_fs": np.array([0.0]),
            "joint_density": joint,
            "proton_density": np.sum(joint, axis=2)*dR,
            "heavy_density": heavy,
            "options": {"proton_mass": 2.0, "heavy_mass": 8.0},
        }
        ef = {
            "a": np.full_like(joint, 4.0),
            "b": np.full_like(joint, -8.0),
            "alpha": np.full_like(heavy, -8.0),
        }
        args = argparse.Namespace(max_frames=1, support_floor=1.0e-4)
        prep = render_final_visualizations._current_preparation(obs, ef, args)
        current = render_final_visualizations._current_frame(obs, ef, prep, 0)
        self.assertTrue(np.allclose(current["proton"], 2.0*joint[0]))
        self.assertTrue(np.allclose(current["heavy_joint"], -joint[0]))
        self.assertTrue(np.allclose(current["heavy_marginal"], -heavy[0]))
        self.assertTrue(np.allclose(
            np.sum(current["heavy_joint"], axis=0)*dq,
            current["heavy_marginal"],
        ))

    def test_current_only_command_writes_movie_and_snapshots(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archive(root)
            output = root/"current_only"
            args = render_final_visualizations.parse_args([
                str(root), "--outdir", str(output), "--only", "current",
                "--format", "gif", "--snapshot-count", "2",
                "--max-frames", "2", "--dpi", "30",
                "--animation-dpi", "25", "--fps", "2",
            ])
            render_final_visualizations.run(args)
            self.assertTrue((
                output/"current_density_composite_movie.gif"
            ).is_file())
            self.assertTrue((
                output/"current_density_composite_snapshots.png"
            ).is_file())
            self.assertEqual(len(list((
                output/"current_density_composite_frames"
            ).glob("*.png"))), 2)

    def test_signed_maps_use_blue_white_red_with_gray_mask(self):
        from matplotlib.colors import to_rgba

        self.assertEqual(report_plot_style.SIGNED_CMAP, "RdBu_r")
        cmap = report_plot_style.masked_cmap(report_plot_style.SIGNED_CMAP)
        self.assertTrue(np.allclose(
            cmap.get_bad(), to_rgba(report_plot_style.MASK_COLOR),
        ))
        self.assertGreater(cmap(1.0)[0], cmap(1.0)[2])
        self.assertGreater(cmap(0.0)[2], cmap(0.0)[0])

    def test_heavy_force_split_is_an_exact_forward_bond_identity(self):
        R = np.linspace(0.0, 4.0, 5, endpoint=False)
        dR = R[1]-R[0]
        times = np.array([0.0, 0.5])
        density = np.ones((len(times), len(R)))
        density /= np.sum(density, axis=1)[:, None]*dR
        trap_alpha, trap_center = 0.23, 1.7
        intrinsic = np.asarray([
            0.3*R+0.04*time*R**2 for time in times
        ])
        trap = trap_alpha*(R-trap_center)**2
        obs = {
            "R": R, "dR": dR, "times_fs": times,
            "heavy_density": density,
            "options": {
                "heavy_trap_alpha": trap_alpha,
                "heavy_trap_center": trap_center,
            },
        }
        ef_zero = {"epsilon_2": intrinsic+trap[None, :]}
        args = argparse.Namespace(
            support_floor=1.0e-4, heavy_min=0.0, heavy_max=3.2,
            max_frames=2,
        )
        prep = render_final_visualizations._heavy_preparation(
            obs, ef_zero, np.zeros_like(density), args,
        )
        expected_harmonic = -tdse_report._forward_bond_derivative(
            trap, dR, axis=0,
        )
        expected_driven = -tdse_report._forward_bond_derivative(
            intrinsic, dR, axis=1,
        )
        self.assertTrue(np.allclose(
            prep["harmonic_force"], expected_harmonic,
            rtol=0.0, atol=2.0e-15,
        ))
        self.assertTrue(np.allclose(
            prep["driven_force"], expected_driven,
            rtol=0.0, atol=2.0e-15,
        ))
        self.assertTrue(np.allclose(
            prep["total_force"],
            prep["driven_force"]+prep["harmonic_force"][None, :],
            rtol=0.0, atol=2.0e-15,
        ))
        self.assertLessEqual(prep["force_decomposition_max_abs"], 2.0e-15)

    def test_relative_collision_reduction_preserves_mass_and_crossing(self):
        q = np.array([-1.0, 0.0, 1.0])
        R = np.array([-0.5, 0.5])
        times = np.array([0.0, 0.5, 1.0])
        dq, dR = 1.0, 1.0
        joint = np.zeros((3, len(q), len(R)))
        # Move all probability from q<R to q>R across the saved frames.
        joint[0, 0, 1] = 1.0/(dq*dR)
        joint[1, 1, 0] = 1.0/(dq*dR)
        joint[2, 2, 0] = 1.0/(dq*dR)
        collision = tdse_collision_report.relative_observables({
            "q": q, "R": R, "times_fs": times,
            "joint_density": joint, "dq": dq, "dR": dR,
            "options": {"proton_mass": 1.0, "heavy_mass": 3.0},
        })
        self.assertTrue(np.allclose(collision["relative_norm"], 1.0))
        self.assertTrue(np.allclose(
            collision["p_q_greater_R"], (0.0, 1.0, 1.0)
        ))
        self.assertTrue(np.all(np.isfinite(collision["crossing_rate_au"])))
        self.assertTrue(np.allclose(collision["s_mean"], (-1.5, 0.5, 1.5)))

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
