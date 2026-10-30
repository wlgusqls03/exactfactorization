"""Snapshot selection uses cached physical times, never MP4 playback seconds."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from multi_component_exact_factorization.pen_perspectives_2026.render import png_indices
from multi_component_exact_factorization.pen_perspectives_2026.resonance import paper_config
from multi_component_exact_factorization.pen_perspectives_2026.workflow import simulate
from multi_component_exact_factorization.pen_perspectives_2026.paper_campaign import (
    main, parser, build_tasks)


class PngIntervalTests(unittest.TestCase):
    def test_legacy_selection(self):
        self.assertEqual(png_indices([0]), [0])
        self.assertEqual(png_indices([0, 1]), [0, 1])
        self.assertEqual(png_indices(np.arange(81)*.5), [0, 40, 80])

    def test_regular_intervals_and_endpoints(self):
        times = np.arange(401)*.1
        self.assertEqual(png_indices(times, 5), list(range(0, 401, 50)))
        self.assertEqual(png_indices(times, 3), list(range(0, 391, 30))+[400])
        self.assertEqual(png_indices(times[:394], 5), list(range(0, 351, 50))+[393])
        self.assertEqual(png_indices([0, 4, 6, 10], 5), [0, 1, 3])  # tie -> earlier
        self.assertEqual(png_indices([0, 1, 2], 1e-100), [0, 1, 2])
        self.assertEqual(png_indices([0, 1, 2], 100), [0, 2])

    def test_irregular_saved_frames_nearest_not_interpolated(self):
        rng = np.random.default_rng(831)
        for every in [.03, .37, 3., 5., 100.]:
            times = np.cumsum(rng.uniform(.03, .7, 85))
            targets = np.arange(np.ceil(times[0]/every), np.floor(times[-1]/every)+1)*every
            expected = sorted({0, len(times)-1, *(int(np.argmin(abs(times-t))) for t in targets)})
            self.assertEqual(png_indices(times, every), expected)

    def test_invalid_input(self):
        for every in [0, -3, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                png_indices([0, 1, 2], every)
        for times in [[], [np.nan], [1, 0], [0, 0]]:
            with self.assertRaises(ValueError):
                png_indices(times, 3)

    def test_cli_forwards_interval_and_rejects_invalid(self):
        args = parser().parse_args(['plan', '--out', '/tmp/unused-pen-png-plan', '--png-every-fs', '5'])
        self.assertEqual(build_tasks(args)[0][2]['png_every_fs'], 5)
        for interval in ['0', '-1', 'nan', 'inf']:
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                main(['plan', '--out', '/tmp/unused-pen-png-plan', '--png-every-fs', interval])

    def test_png_only_render_preserves_coefficients_restart_and_movies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root/'rabi/on/dse_off/strong'
            simulate(paper_config('rabi', 'on', nr=64), case, .02, .005)
            original = {p: p.read_bytes() for p in (case/'coefficients').glob('*.npz')}
            original[case/'restart.npz'] = (case/'restart.npz').read_bytes()
            movie = case/'plots/A_diagnostics_clean/A_diagnostics_clean.mp4'
            movie.parent.mkdir(parents=True)
            movie.write_bytes(b'existing MP4 must not be touched')
            original[movie] = movie.read_bytes()
            with patch('multi_component_exact_factorization.pen_perspectives_2026.workflow.simulate',
                       side_effect=AssertionError('Rendering must not propagate')):
                main(['render', '--out', temp, '--model', 'rabi', '--resonance', 'on',
                      '--case', 'strong', '--dse', 'off', '--no-movies',
                      '--png-every-fs', '.006', '--nq-display', '64', '--dpi', '40'])
            report = json.loads((case/'plots/render_report.json').read_text())
            selection = json.loads((case/'plots/png_selection.json').read_text())
            expected = png_indices(report['frame_times_fs'], .006)
            self.assertGreater(len(expected), 3)
            self.assertEqual(report['png_frame_indices'], expected)
            self.assertEqual(report['movie_frames'], 0)
            self.assertEqual(len(selection['frames']), len(expected))
            self.assertEqual(len(list((case/'plots').rglob('frame_*.png'))), 7*len(expected))
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content, str(path))


if __name__ == '__main__':
    unittest.main()
