import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from multi_component_exact_factorization import (
    report_plot_style,
    render_final_visualizations,
    render_tdse_tdpes_gauges,
    tdse_collision_report,
    tdse_report,
)


class TDSEReportTests(unittest.TestCase):
    def test_nested_contour_crop_preserves_full_grid_vertices(self):
        plot = render_final_visualizations.plt
        q, R = np.linspace(-12, 28, 300), np.linspace(5, 14, 180)
        density = np.exp(-((q[:, None]+4)**2+(R[None, :]-9.5)**2)/0.4)
        fig, axes = plot.subplots(1, 2)
        try:
            cropped = render_final_visualizations._joint_linear_contours(
                axes[0], {"q": q, "R": R}, density,
            )
            full = axes[1].contour(q, R, density.T,
                                   levels=cropped.levels)
            for a, b in zip(cropped.allsegs, full.allsegs):
                a, b = np.concatenate(a), np.concatenate(b)
                a = a[np.lexsort((a[:, 1], a[:, 0]))]
                b = b[np.lexsort((b[:, 1], b[:, 0]))]
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-13)
        finally:
            plot.close(fig)

    def test_final_visualization_uses_shared_point_one_percent_focus(self):
        args = render_final_visualizations.parse_args(["dummy-run"])
        self.assertEqual(args.analysis_focus_floor, 1.0e-3)
        self.assertEqual(args.tdpes_energy_reference, "raw")

        density = np.array([[
            [1.0, 9.9e-4],
            [1.0e-3, 2.0e-3],
        ]])
        obs = {
            "q": np.array([0.0, 1.0]),
            "R": np.array([2.0, 3.0]),
            "joint_density": density,
        }
        active, _, _ = render_final_visualizations._frame_focus(
            obs, 0, args.analysis_focus_floor,
        )
        np.testing.assert_array_equal(
            active,
            np.array([[True, False], [True, True]]),
        )

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
        electron_proton = electron[:, :, None]*proton[:, None, :]
        state_density_q = populations[:, :, None]*proton[:, None, :]
        state_density_R = populations[:, :, None]*heavy[:, None, :]
        channel_density_qR = populations[:, :, None, None]*joint[:, None, :, :]
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
            epsilon_1_wbo=0.60*epsilon_1,
            epsilon_2=epsilon_2,
            epsilon_2_gi=0.55*epsilon_2,
            a=a, b=b, alpha=alpha,
            sphi_q1=np.exp(1j*a*dq),
            sphi_R1=np.exp(-1j*np.full(shape, 0.02*dR)),
            sgamma_R1=np.exp(1j*alpha*dR),
            bo_state_density_q=state_density_q,
            bo_state_density_R=state_density_R,
            bo_channel_density_qR=channel_density_qR,
            factorization_residual=np.zeros(len(times)),
            electron_proton_density=electron_proton,
        )
        return archive, joint

    def test_static_tdse_report_uses_reduced_archive_and_derived_fields(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, joint = self._write_archive(root)
            report = root/"report"
            obs = tdse_report.run(
                archive, report, no_animation=True, dpi=45,
                snapshot_count=3, gauge_mode="both",
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
                fps=2, fmt="gif", gauge_mode="both",
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
            ef = tdse_report._load_ef_fields(
                obs,
                field_keys=(
                    "epsilon_1", "epsilon_1_gi", "epsilon_1_wbo",
                    "epsilon_2", "epsilon_2_gi", "a", "b", "alpha",
                    "bo_channel_density_qR", "electron_proton_density",
                ),
                link_keys=("sphi_q1", "sphi_R1", "sgamma_R1"),
            )
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
                "--dpi", "35", "--surface-count", "2", "--gauge", "both",
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
                "nested_factorization_analysis_absolute_snapshots.png",
                "heavy_analysis_snapshots.png",
                "bo_combined_snapshots.png",
                "bo_3d_channel_dynamics_snapshots.png",
                "tdpes1_origin_positive_gauge_snapshots.png",
                "tdpes2_origin_positive_gauge_snapshots.png",
                "final_visualizations_manifest.txt",
            ):
                self.assertTrue((output/name).is_file(), name)
            for directory in (
                "marginal_time_position_frames",
                "joint_density_qR_frames",
                "joint_velocity_frames",
                "vector_potential_composite_frames",
                "current_density_composite_frames",
                "nested_factorization_analysis_absolute_frames",
                "heavy_analysis_frames",
                "bo_combined_frames",
                "bo_3d_channel_frames",
                "tdpes1_origin_positive_gauge_frames",
                "tdpes2_origin_positive_gauge_frames",
            ):
                self.assertEqual(len(list((output/directory).glob("*.png"))), 2)
            self.assertGreaterEqual(len(products), 25)
            self.assertEqual(args.tdpes_color_scale, "linear")
            self.assertFalse((output/"tdpes1_origin_snapshots.png").exists())
            manifest = (output/"final_visualizations_manifest.txt").read_text()
            self.assertIn("nested_potential_gauge=positive_density", manifest)
            self.assertIn(
                "default_scalar_vector_gauge=positive_density", manifest,
            )
            self.assertIn(
                "zero_potential_gauge_usage="
                "heavy_force_from_minus_partial_R_epsilon_2_only",
                manifest,
            )
            self.assertIn("tdpes2_gauge=positive_density", manifest)
            self.assertIn("tdpes2_max_identity_residual=", manifest)

    def test_bo3d_and_tdpes1_only_commands_write_movies_and_snapshots(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archive(root)
            output = root/"new_analysis"
            args = render_final_visualizations.parse_args([
                str(root), "--outdir", str(output), "--only", "bo3d", "tdpes1",
                "--format", "gif", "--snapshot-count", "2", "--max-frames", "2",
                "--dpi", "30", "--animation-dpi", "25", "--fps", "2",
                "--bo3d-q-points", "5", "--bo3d-R-points", "7",
            ])
            render_final_visualizations.run(args)
            for name in (
                "bo_3d_channel_dynamics_movie.gif",
                "bo_3d_channel_dynamics_snapshots.png",
                "tdpes1_origin_positive_gauge_movie.gif",
                "tdpes1_origin_positive_gauge_snapshots.png",
            ):
                self.assertTrue((output/name).is_file(), name)
            manifest = (output/"final_visualizations_manifest.txt").read_text()
            self.assertIn("bo3d_density=rho_j", manifest)
            self.assertIn("tdpes1_discrete_identity=", manifest)

    def test_link_metric_is_zero_for_unit_links_and_positive_below_unit(self):
        unit = np.exp(1j*np.linspace(0.0, 0.4, 12)).reshape(3, 4)
        metric = render_final_visualizations._site_link_metric(unit, 0.2, axis=0)
        self.assertLess(np.max(np.abs(metric)), 2.0e-14)
        reduced = 0.8*unit
        metric = render_final_visualizations._site_link_metric(reduced, 0.2, axis=1)
        self.assertTrue(np.all(metric > 0.0))

    def test_tdpes1_origin_uses_one_origin_and_closes_displayed_identity(self):
        with TemporaryDirectory() as temporary:
            archive, _ = self._write_archive(temporary)
            obs = tdse_report.calculate_observables(tdse_report.load_observables(archive))
            ef = tdse_report._load_ef_fields(
                obs,
                field_keys=(
                    "epsilon_1", "epsilon_1_gi", "epsilon_1_wbo",
                    "bo_channel_density_qR",
                    "epsilon_2", "a", "b", "alpha",
                ),
                link_keys=("sphi_q1", "sphi_R1", "sgamma_R1"),
            )
            self.assertEqual(ef["gauge"], "positive_density")
            args = argparse.Namespace(support_floor=1.0e-4, decades=6.0, max_frames=3,
                                      tdpes_energy_reference="initial")
            prep = render_final_visualizations._tdpes1_origin_preparation(obs, ef, args)
            original_wbo = ef['epsilon_1_wbo'].copy()
            initial = render_final_visualizations._tdpes1_origin_frame(
                obs, ef, prep, 0,
            )
            frame = render_final_visualizations._tdpes1_origin_frame(obs, ef, prep, 1)
            again = render_final_visualizations._tdpes1_origin_frame(obs, ef, prep, 1)
            self.assertTrue(np.array_equal(original_wbo, ef['epsilon_1_wbo']))
            self.assertTrue(np.array_equal(frame['wbo'], again['wbo']))
            self.assertEqual(prep["energy_reference_mode"], "initial")
            self.assertEqual(
                initial["energy_reference"], frame["energy_reference"],
            )
            support = (
                obs["joint_density"][0]
                >= args.support_floor*np.max(obs["joint_density"][0])
            )
            self.assertAlmostEqual(
                prep["fixed_energy_reference"],
                np.average(
                    initial["total_raw"][support],
                    weights=obs["joint_density"][0][support],
                ),
            )
            p0 = ef["bo_channel_density_qR"][1, 0]/obs["joint_density"][1]
            p1 = ef["bo_channel_density_qR"][1, 1]/obs["joint_density"][1]
            reference = frame["energy_reference"]
            np.testing.assert_allclose(
                frame["wbo_1"], p0*(obs["bo_energies"][0]-reference),
            )
            np.testing.assert_allclose(
                frame["wbo_2_pure"], p1*(obs["bo_energies"][1]-reference),
            )
            np.testing.assert_allclose(
                frame["wbo_2"], frame["wbo"]-frame["wbo_1"],
            )
            np.testing.assert_allclose(
                frame["wbo_higher"], frame["wbo_2"]-frame["wbo_2_pure"],
            )
            ef["bo_channel_density_qR"][1, 0] = 0
            empty = render_final_visualizations._tdpes1_origin_frame(obs, ef, prep, 1)
            np.testing.assert_array_equal(empty["wbo_1"], 0)
            self.assertTrue(np.allclose(
                10**frame['joint_log'], obs['joint_density'][1]/obs['joint_density'][1].max()))
            self.assertTrue(np.allclose(
                frame["native_total"], frame["native_gi"]+frame["gd_raw"],
                rtol=0.0, atol=2.0e-15,
            ))
            self.assertTrue(np.allclose(
                frame["total"],
                frame["wbo_1"]+frame["wbo_2"]+frame["gd"]
                +frame["geo_q"]+frame["geo_R"],
                rtol=0.0, atol=2.0e-15,
            ))
            self.assertLess(np.max(np.abs(frame["identity_residual"])), 2.0e-15)
            self.assertTrue(np.allclose(
                frame["gi_limit"],
                frame["wbo"]+frame["geo_q"]+frame["geo_R"],
                rtol=0.0, atol=2.0e-15,
            ))

    def test_origin_and_nested_prefer_stored_tdpes_decomposition(self):
        with TemporaryDirectory() as temporary:
            archive, _ = self._write_archive(temporary)
            obs = tdse_report.calculate_observables(
                tdse_report.load_observables(archive)
            )
            ef = tdse_report._load_ef_fields(
                obs,
                field_keys=(
                    "epsilon_1", "epsilon_1_gi", "epsilon_1_wbo",
                    "epsilon_2", "epsilon_2_gi", "a", "b", "alpha",
                    "bo_channel_density_qR", "electron_proton_density",
                ),
                link_keys=("sphi_q1", "sphi_R1", "sgamma_R1"),
            )
            shape = obs["joint_density"].shape
            line_shape = obs["heavy_density"].shape
            components_1 = {
                "tdpes1_wbo_0": np.full(shape, 0.11),
                "tdpes1_wbo_excited": np.full(shape, 0.07),
                "tdpes1_gd": np.full(shape, -0.03),
                "tdpes1_geo_q": np.full(shape, 0.02),
                "tdpes1_geo_R": np.full(shape, 0.01),
            }
            components_2 = {
                "tdpes2_wbo_0": np.full(line_shape, 0.13),
                "tdpes2_wbo_excited": np.full(line_shape, 0.05),
                "tdpes2_gd": np.full(line_shape, -0.02),
                "tdpes2_geo_q": np.full(line_shape, 0.03),
                "tdpes2_geo_R": np.full(line_shape, 0.01),
            }
            ef.update(components_1)
            ef.update(components_2)
            ef["tdpes1_total"] = sum(components_1.values())
            ef["tdpes2_total"] = sum(components_2.values())
            # A complete modern cache must not need legacy scalar/link arrays.
            for key in ("epsilon_1", "epsilon_1_gi", "epsilon_1_wbo",
                        "epsilon_2", "epsilon_2_gi", "sphi_q1", "sphi_R1", "sgamma_R1"):
                ef.pop(key, None)
            args = argparse.Namespace(
                support_floor=1.0e-4, analysis_focus_floor=1.0e-2,
                decades=6.0, max_frames=3,
                tdpes_energy_reference="raw",
            )
            prep_1 = render_final_visualizations._tdpes1_origin_preparation(
                obs, ef, args,
            )
            frame_1 = render_final_visualizations._tdpes1_origin_frame(
                obs, ef, prep_1, 1,
            )
            self.assertTrue(frame_1["stored_decomposition"])
            np.testing.assert_allclose(
                frame_1["total_raw"], ef["tdpes1_total"][1],
            )
            prep_2 = render_final_visualizations._tdpes2_origin_preparation(
                obs, ef, args,
            )
            frame_2 = render_final_visualizations._tdpes2_origin_frame(
                obs, ef, prep_2, 1,
            )
            self.assertTrue(frame_2["stored_decomposition"])
            np.testing.assert_allclose(
                frame_2["total_raw"], ef["tdpes2_total"][1],
            )
            nested = render_final_visualizations._nested_frame(obs, ef, 1, args)
            expected_1 = ef["tdpes1_total"][1]
            expected_2 = ef["tdpes2_total"][1]
            np.testing.assert_array_equal(frame_1["total"], expected_1)
            np.testing.assert_array_equal(frame_2["total"], expected_2)
            for current in (frame_1, frame_2):
                self.assertEqual(current["energy_reference"], 0.0)
                np.testing.assert_allclose(current["identity_residual"], 0.0, atol=1e-15)
            np.testing.assert_allclose(
                nested["epsilon_1"], expected_1, rtol=0.0, atol=1.0e-15,
            )
            np.testing.assert_allclose(
                nested["epsilon_2"], expected_2, rtol=0.0, atol=1.0e-15,
            )

    def test_tdpes2_origin_uses_positive_gauge_and_closes_identity(self):
        with TemporaryDirectory() as temporary:
            archive, _ = self._write_archive(temporary)
            obs = tdse_report.calculate_observables(
                tdse_report.load_observables(archive)
            )
            ef = tdse_report._load_ef_fields(
                obs,
                field_keys=(
                    "epsilon_2", "epsilon_2_gi", "epsilon_1_wbo",
                    "bo_channel_density_qR",
                ),
                link_keys=("sgamma_R1",),
            )
            self.assertEqual(ef["gauge"], "positive_density")
            args = argparse.Namespace(
                support_floor=1.0e-4, analysis_focus_floor=1.0e-2,
                max_frames=3, scale_sample_frames=3, tdpes_energy_reference="initial",
            )
            prep = render_final_visualizations._tdpes2_origin_preparation(
                obs, ef, args,
            )
            initial = render_final_visualizations._tdpes2_origin_frame(
                obs, ef, prep, 0,
            )
            frame = render_final_visualizations._tdpes2_origin_frame(
                obs, ef, prep, 1,
            )
            self.assertEqual(prep["energy_reference_mode"], "initial")
            self.assertEqual(
                initial["energy_reference"], frame["energy_reference"],
            )
            support = (
                obs["heavy_density"][0]
                >= args.support_floor*np.max(obs["heavy_density"][0])
            )
            self.assertAlmostEqual(
                prep["fixed_energy_reference"],
                np.average(
                    initial["total_raw"][support],
                    weights=obs["heavy_density"][0][support],
                ),
            )
            np.testing.assert_allclose(
                frame["native_total"], frame["native_gi"]+frame["gd"],
                rtol=0.0, atol=2.0e-15,
            )
            np.testing.assert_allclose(
                frame["total"],
                frame["wbo_1"]+frame["wbo_2"]+frame["gd"]
                +frame["geo_q"]+frame["geo_R"],
                rtol=0.0, atol=2.0e-15,
            )
            np.testing.assert_allclose(
                frame["gi"],
                frame["wbo_1"]+frame["wbo_2"]
                +frame["geo_q"]+frame["geo_R"],
                rtol=0.0, atol=2.0e-15,
            )
            np.testing.assert_allclose(
                frame["total"], frame["gi"]+frame["gd"],
                rtol=0.0, atol=2.0e-15,
            )
            self.assertLess(
                np.max(np.abs(frame["identity_residual"])), 2.0e-15,
            )
            self.assertEqual(frame["bo_reference"].shape, (2, len(obs["R"])))

    def test_tdpes_both_reference_modes_write_distinct_products(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archive(root)
            output = root/"both_reference_modes"
            args = render_final_visualizations.parse_args([
                str(root), "--outdir", str(output),
                "--only", "tdpes1", "tdpes2",
                "--tdpes-energy-reference", "both",
                "--no-animation", "--snapshot-count", "2",
                "--max-frames", "2", "--dpi", "30",
            ])
            render_final_visualizations.run(args)
            for name in (
                "tdpes1_origin_positive_gauge_snapshots.png",
                "tdpes1_origin_positive_gauge_initial_reference_snapshots.png",
                "tdpes2_origin_positive_gauge_snapshots.png",
                "tdpes2_origin_positive_gauge_initial_reference_snapshots.png",
            ):
                self.assertTrue((output/name).is_file(), name)
            manifest = (output/"final_visualizations_manifest.txt").read_text()
            self.assertIn(
                "tdpes1_rendered_energy_reference_modes=raw,initial",
                manifest,
            )
            self.assertIn(
                "tdpes2_rendered_energy_reference_modes=raw,initial",
                manifest,
            )

    def test_geometry_log_movie_writes_reference_independent_aliases(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archive(root)
            output = root/"geometry_log"
            args = render_final_visualizations.parse_args([
                str(root), "--outdir", str(output), "--only", "geometry",
                "--format", "gif", "--snapshot-count", "2",
                "--max-frames", "2", "--scale-sample-frames", "2",
                "--dpi", "30", "--animation-dpi", "25", "--fps", "2",
            ])
            render_final_visualizations.run(args)
            fixed = output/"tdpes_geometry_log_movie.gif"
            self.assertTrue(fixed.is_file())
            manifest = (output/"final_visualizations_manifest.txt").read_text()
            self.assertIn("geometry_energy_reference_dependence=none", manifest)
            self.assertIn(
                "geometry_movie=single_reference_independent_product",
                manifest,
            )

    def test_movie_density_and_geometry_scales_do_not_change(self):
        module = render_final_visualizations
        nested_update, geometry_update = module._update_nested_composite, module._update_tdpes_geometry

        def check_nested(state, *args, **kwargs):
            names = ("electron_proton_image", "conditional_image")
            before = [state[name].get_clim() for name in names]
            result = nested_update(state, *args, **kwargs)
            self.assertEqual(before, [state[name].get_clim() for name in names])
            return result

        def check_geometry(state, axes, *args, **kwargs):
            result = geometry_update(state, axes, *args, **kwargs)
            for axis in axes[2:]:
                self.assertEqual(axis.get_ylim(), (1e-7, 1e-1))
            return result

        with TemporaryDirectory() as temporary:
            self._write_archive(temporary)
            args = module.parse_args([
                temporary, "--only", "nested", "geometry", "--format", "gif",
                "--snapshot-count", "2", "--max-frames", "2", "--dpi", "30",
                "--animation-dpi", "25",
            ])
            with patch.object(module, "_update_nested_composite", side_effect=check_nested) as n:
                with patch.object(module, "_update_tdpes_geometry", side_effect=check_geometry) as g:
                    module.run(args)
            self.assertGreater(n.call_count, 0)
            self.assertGreater(g.call_count, 0)

    def test_geometry_terms_match_existing_tdpes_decompositions(self):
        with TemporaryDirectory() as temporary:
            archive, _ = self._write_archive(temporary)
            obs = tdse_report.calculate_observables(
                tdse_report.load_observables(archive)
            )
            ef = tdse_report._load_ef_fields(
                obs,
                field_keys=(
                    "epsilon_1", "epsilon_1_gi", "epsilon_1_wbo",
                    "epsilon_2", "epsilon_2_gi", "bo_channel_density_qR",
                ),
                link_keys=("sphi_q1", "sphi_R1", "sgamma_R1"),
            )
            args = argparse.Namespace(
                support_floor=1.0e-4, analysis_focus_floor=1.0e-2,
                max_frames=3, scale_sample_frames=3, decades=6.0,
                tdpes_contour_q_points=20, tdpes_contour_R_points=20,
            )
            prep = render_final_visualizations._tdpes_geometry_preparation(
                obs, ef, args,
            )
            geometry = render_final_visualizations._tdpes_geometry_frame(
                obs, ef, prep, 1,
            )
            prep1 = dict(
                prep, energy_reference_mode="raw",
            )
            first = render_final_visualizations._tdpes1_origin_frame(
                obs, ef, prep1, 1,
            )
            second = render_final_visualizations._tdpes2_origin_frame(
                obs, ef, prep1, 1,
            )
            for geometry_key, decomposition, decomposition_key in (
                ("geo1_q", first, "geo_q"),
                ("geo1_R", first, "geo_R"),
                ("geo2_q", second, "geo_q"),
                ("geo2_R", second, "geo_R"),
            ):
                np.testing.assert_allclose(
                    geometry[geometry_key], decomposition[decomposition_key],
                    rtol=0.0, atol=0.0,
                )
            self.assertGreater(prep["bound"], prep["lower"])

    def test_first_level_only_gauge_matches_complete_zero_gauge(self):
        with TemporaryDirectory() as temporary:
            archive, _ = self._write_archive(temporary)
            obs = tdse_report.calculate_observables(
                tdse_report.load_observables(archive)
            )
            complete = tdse_report._load_ef_fields(obs)
            first = tdse_report._load_ef_fields(
                obs, field_keys=("epsilon_1",),
                link_keys=("sphi_q1", "sphi_R1"),
            )
            tdse_report.transform_to_zero_potential_gauge(obs, complete)
            tdse_report.transform_first_level_to_q_axial_gauge(obs, first)
            for key in ("epsilon_1", "sphi_q1", "sphi_R1"):
                self.assertTrue(np.allclose(
                    first[key], complete[key], rtol=0.0, atol=2.0e-15,
                ), key)
            self.assertEqual(first["gauge"], "first_level_q_axial")

    def test_analysis_focus_tracks_moving_support_and_keeps_branches(self):
        q = np.arange(100, dtype=float)
        R = np.arange(40, dtype=float)
        rho = np.full((2, 100, 40), 1e-8)
        rho[0, 20:24, 10:13] = 1
        rho[1, 60:64, 25:28] = 1
        rho[1, 75:78, 25:28] = .1
        obs = {'q': q, 'R': R, 'joint_density': rho}
        active, _, first = render_final_visualizations._frame_focus(obs, 0, .01)
        _, _, last = render_final_visualizations._frame_focus(obs, 1, .01)
        self.assertFalse(active[0, 0])
        self.assertLess(first[0][1], last[0][0])
        self.assertGreaterEqual(last[0][1], 77)

    def test_nested_factorization_only_command_writes_movie_and_snapshots(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_archive(root)
            output = root/"nested_only"
            args = render_final_visualizations.parse_args([
                str(root), "--outdir", str(output), "--only", "nested",
                "--format", "gif", "--snapshot-count", "2",
                "--max-frames", "2", "--dpi", "30",
                "--animation-dpi", "25", "--fps", "2",
            ])
            render_final_visualizations.run(args)
            self.assertTrue((
                output/"nested_factorization_analysis_absolute_movie.gif"
            ).is_file())
            self.assertTrue((
                output/"nested_factorization_analysis_absolute_snapshots.png"
            ).is_file())
            self.assertEqual(len(list((
                output/"nested_factorization_analysis_absolute_frames"
            ).glob("*.png"))), 2)
            manifest = (output/"final_visualizations_manifest.txt").read_text()
            self.assertIn(
                "nested_density_display=absolute_linear_trajectory_fixed",
                manifest,
            )
            self.assertIn("nested_electron_proton_vmax=", manifest)
            self.assertIn("nested_conditional_proton_vmax=", manifest)
            self.assertTrue((output/'nested_factorization_analysis_absolute_movie.gif').is_file())
            self.assertTrue((output/'nested_factorization_analysis_absolute_snapshots.png').is_file())
            self.assertIn('nested_density_contour_modes=absolute', manifest)
            self.assertIn('nested_absolute_density_cutoff=', manifest)

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
