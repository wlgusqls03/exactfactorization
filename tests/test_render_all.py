from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from multi_component_exact_factorization import render_all


class RenderAllDiscoveryTests(unittest.TestCase):
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
                max_3d_frames=80, surface_count=7,
            )
            render_all.run(args)

            visualize_run.assert_called_once()
            dynamics_run.assert_called_once()
            excited_run.assert_called_once()
            visualize_3d_run.assert_called_once()
            self.assertEqual(dynamics_run.call_args.args[0].n_states, 4)
            self.assertEqual(excited_run.call_args.args[0].n_states, 4)


if __name__ == "__main__":
    unittest.main()
