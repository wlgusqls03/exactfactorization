from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from multi_component_exact_factorization import propagate, render_all


class RenderAllDiscoveryTests(unittest.TestCase):
    @patch.object(render_all, "render_completed_run")
    @patch.object(propagate, "run", return_value=Path("saved/result.npz"))
    def test_propagation_main_renders_only_after_run_returns(
        self, propagation_run, render_completed_run
    ):
        args = Namespace(render_after=True, render_fast=True)

        result = propagate.main(args)

        self.assertEqual(result, Path("saved/result.npz"))
        propagation_run.assert_called_once_with(args)
        render_completed_run.assert_called_once_with(
            Path("saved/result.npz"), fast=True
        )

    @patch.object(render_all, "run")
    def test_completed_run_uses_standard_fast_render_arguments(self, run):
        archive = Path("results/20260731/test/result.npz")

        render_all.render_completed_run(archive, fast=True)

        run.assert_called_once()
        args = run.call_args.args[0]
        self.assertEqual(args.run, str(archive))
        self.assertTrue(args.fast)
        self.assertFalse(args.no_animation)
        self.assertFalse(args.no_3d)

    def test_bare_name_selects_newest_dated_run(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/"results"
            older = root/"20260729"/"same_name"
            newer = root/"20260730"/"same_name"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            self.assertEqual(
                render_all.resolve_run_input("same_name", root),
                newer.resolve(),
            )

    def test_finds_gpu_archive(self):
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)/"run"
            run_dir.mkdir()
            archive = run_dir/"multi_component_direct_ef_gpu.npz"
            archive.touch()
            self.assertEqual(
                render_all.find_archive(run_dir.resolve()),
                (archive.resolve(), run_dir.resolve()),
            )

    def test_reads_excited_state_metadata(self):
        with TemporaryDirectory() as temporary:
            archive = Path(temporary)/"result.npz"
            np.savez(
                archive,
                args=np.array([{"electron_excitation": 2}], dtype=object),
            )
            self.assertEqual(
                render_all.archive_state(archive),
                (2, "excited state n=2"),
            )

    @patch.object(render_all.compact_report, "run")
    @patch.object(render_all.visualize, "run")
    def test_default_render_creates_only_question_oriented_report(
        self, legacy_visualize, compact_run
    ):
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)/"results"/"20260731"/"compact"
            run_dir.mkdir(parents=True)
            archive = run_dir/"multi_component_direct_ef_gpu.npz"
            np.savez(
                archive,
                args=np.array([{"electron_excitation": 1}], dtype=object),
            )
            args = render_all.parse_args([
                str(run_dir), "--no-animation", "--no-3d",
            ])
            loaded = render_all.visualize.LoadedArchive({
                "args": np.array([{"electron_excitation": 1}], dtype=object),
            })
            decomposition = (
                np.empty((6, 1, 1)), np.empty((1, 6, 1, 1)),
                np.empty((1, 6)), np.empty(1),
            )
            with patch.object(
                render_all.visualize, "load_archive", return_value=loaded
            ), patch.object(
                render_all.excited_state_analysis,
                "calculate_state_decomposition",
                return_value=decomposition,
            ):
                render_all.run(args)

            compact_run.assert_called_once()
            self.assertIs(compact_run.call_args.kwargs["decomposition"], decomposition)
            self.assertTrue(compact_run.call_args.kwargs["no_animation"])
            legacy_visualize.assert_not_called()

    @patch.object(render_all.visualize_3d, "run")
    @patch.object(render_all.excited_state_analysis, "run")
    @patch.object(render_all.dynamics_analysis, "run")
    @patch.object(render_all.visualize, "run")
    def test_orchestrates_all_analyses_with_automatic_state_count(
        self, visualize_run, dynamics_run, excited_run, visualize_3d_run
    ):
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)/"results"/"20260730"/"excited_run"
            run_dir.mkdir(parents=True)
            archive = run_dir/"multi_component_direct_ef_gpu.npz"
            np.savez(
                archive,
                args=np.array([{"electron_excitation": 2}], dtype=object),
            )
            args = Namespace(
                run=str(run_dir), n_states=0, snapshots=5, dpi=180,
                animation_dpi=120, fps=12, max_frames=180, format="mp4",
                no_animation=False, no_3d=False, max_axis_points=24,
                max_3d_frames=80, surface_count=7, fast=False,
                low_memory=False, all_products=True,
            )
            loaded = render_all.visualize.LoadedArchive({
                "args": np.array([{"electron_excitation": 2}], dtype=object),
            })
            decomposition = (
                np.empty((4, 1, 1)), np.empty((1, 4, 1, 1)),
                np.empty((1, 4)), np.empty(1),
            )
            with patch.object(
                render_all.visualize, "load_archive", return_value=loaded
            ), patch.object(
                render_all.excited_state_analysis,
                "calculate_state_decomposition",
                return_value=decomposition,
            ):
                render_all.run(args)

            visualize_run.assert_called_once()
            dynamics_run.assert_called_once()
            excited_run.assert_called_once()
            visualize_3d_run.assert_called_once()
            self.assertEqual(dynamics_run.call_args.args[0].n_states, 6)
            self.assertEqual(excited_run.call_args.args[0].n_states, 6)
            self.assertIs(dynamics_run.call_args.kwargs["decomposition"], decomposition)
            self.assertIs(excited_run.call_args.kwargs["decomposition"], decomposition)


if __name__ == "__main__":
    unittest.main()
