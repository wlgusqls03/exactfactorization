"""Short tests only: Table 2, routing, initial preparation, restart and summaries."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from multi_component_exact_factorization.pen_perspectives_2026.model import (
    Config, Propagator, initial, observables)
from multi_component_exact_factorization.pen_perspectives_2026.resonance import (
    paper_config, scenario_report, select_frequency)
from multi_component_exact_factorization.pen_perspectives_2026.paper_campaign import (
    parser, build_tasks, preflight, main)
from multi_component_exact_factorization.pen_perspectives_2026.workflow import simulate


class PaperCampaignTests(unittest.TestCase):
    def test_table2_four_scenarios(self):
        for model, frequencies in [('nonadiabatic', [.17, .097]), ('rabi', [.17, .12])]:
            for resonance, omega in zip(['on', 'off'], frequencies):
                c = paper_config(model, resonance)
                self.assertEqual((c.omega, c.r1, c.r2, c.r3, c.delta),
                                 (omega, 2. if model == 'rabi' else 6., 2., 3.875, .17 if model == 'rabi' else 0.))
                self.assertEqual((c.mass, c.k, c.a, c.b, c.g, c.mu12, c.z),
                                 (20000., .02, 3., .01, .01, -1., 1.))
                self.assertEqual((c.nr, c.rmin, c.rmax, c.dt, c.nf, c.dse), (500, -2., 10., .1, 2, False))
                self.assertEqual(select_frequency(c, 'paper-off', model)[0].omega, frequencies[1])
        with self.assertRaises(ValueError):
            paper_config('rabi', 'on', delta=0.)

    def test_rabi_preparation_and_spectral_estimate(self):
        for resonance in ['on', 'off']:
            c = paper_config('rabi', resonance)
            p = Propagator(c)
            u = initial(c)
            o = observables(u, p)
            self.assertAlmostEqual(o['norm'], 1, places=13)
            self.assertAlmostEqual(o['photon_number'], 0, places=13)
            self.assertAlmostEqual(o['bo_population'][1], 1, places=13)
            report = scenario_report(c)
            self.assertEqual(report['model'], 'rabi')
            self.assertEqual(report['resonance'], resonance)
            period = report['rwa_reference']['period_fs']
            self.assertTrue((25 < period < 28) if resonance == 'on' else (2.9 < period < 3.2))
            # No equal-pol-population assumption is inserted in initialization.
            self.assertAlmostEqual(sum(o['pol_population']), 1, places=12)
        zero = paper_config('rabi', 'on', g=0.)
        p, u = Propagator(zero), initial(zero)
        density0 = np.sum(abs(u)**2, axis=(1, 2))
        for _ in range(100):  # 0.24 fs, not a long Rabi trajectory
            u = p.step(u)
        self.assertLess(np.max(abs(np.sum(abs(u)**2, axis=(1, 2))-density0)), 1e-8)
        self.assertLess(observables(u, p)['photon_number'], 1e-20)

    def test_plan_no_writes_and_unique_24_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'new'
            args = parser().parse_args(['plan', '--out', str(root), '--model', 'all', '--resonance', 'both',
                                       '--case', 'all', '--dse', 'both'])
            tasks = build_tasks(args)
            self.assertEqual(len(tasks), 24)
            self.assertEqual(len({str(t[1]) for t in tasks}), 24)
            self.assertFalse(root.exists())
            # Paper defaults do not inherit legacy bare-gap CLI default.
            default = build_tasks(parser().parse_args(['plan', '--out', str(root)]))[0]
            self.assertEqual(default[0].omega, .17)
            self.assertEqual(default[1].relative_to(root).as_posix(), 'nonadiabatic/on/dse_off/strong')

    def test_preflight_refuses_overwrite_and_bad_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            args = parser().parse_args(['run', '--out', temp, '--model', 'rabi', '--end-fs', '.01', '--no-render'])
            tasks = build_tasks(args)
            c, out, opts = tasks[0]
            preflight(tasks, 'run')
            simulate(c, out, .01, args.every_fs)
            with self.assertRaises(FileExistsError):
                preflight(tasks, 'run')
            opts['resume'] = True
            preflight(tasks, 'run')
            with self.assertRaises(ValueError):
                preflight([(replace(c, omega=.12), out, opts)], 'run')
            preflight([(replace(c, omega=.12), out, opts)], 'render')  # saved config used

    def test_cli_all_scenarios_parallel_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            cmd = [sys.executable, '-m', 'multi_component_exact_factorization.pen_perspectives_2026.paper_campaign',
                   'run', '--out', temp, '--model', 'all', '--resonance', 'both', '--case', 'all',
                   '--dse', 'both', '--end-fs', '.01', '--no-render', '--jobs', '2']
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            configs = list(Path(temp).glob('*/*/dse_*/*/config.json'))
            self.assertEqual(len(configs), 24)
            for path in configs:
                status = json.loads(path.with_name('status.json').read_text())
                self.assertLess(status['max_norm_error'], 1e-12)
                self.assertLess(status['max_energy_drift_Ha'], 1e-8)
                scenario = json.loads(path.with_name('scenario.json').read_text())
                self.assertIn(scenario['resonance'], ['on', 'off'])
            subprocess.run(cmd+['--resume', '--end-fs', '.02'], capture_output=True, text=True, check=True)
            for path in configs:
                self.assertGreaterEqual(json.loads(path.with_name('status.json').read_text())['t_fs'], .02)

    def test_exchange_summary_and_missing_off_fallback(self):
        from multi_component_exact_factorization.pen_perspectives_2026.render import render_case
        from multi_component_exact_factorization.pen_perspectives_2026.paper_plots import resonance_comparison
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'rabi'
            case = root/'on/dse_off/strong'
            simulate(paper_config('rabi', 'on', nr=64), case, 0, .1)
            report = render_case(case, movies=False, nq=64, dpi=40)
            self.assertEqual(report['scenario']['model'], 'rabi')
            summary = json.loads((case/'plots/exchange_summary.json').read_text())
            self.assertAlmostEqual(summary['photon_number'][0], 0.)
            self.assertAlmostEqual(summary['R_mean'][0], 2., places=3)
            self.assertEqual(len(list((case/'plots').rglob('frame_*.png'))), 7)
            self.assertEqual(len(resonance_comparison(root)), 1)
            self.assertTrue((root/'resonance_comparison/on_off_dse_off.png').exists())
            self.assertTrue((root/'resonance_comparison/on_off_paper_window_dse_off.png').exists())
            self.assertTrue((case/'plots/quadrature_checks.json').exists())

    def test_off_resonance_quadrature_is_not_propagation_error(self):
        from multi_component_exact_factorization.pen_perspectives_2026.fields import analyze, diagnostics
        c = paper_config('rabi', 'off')
        p, u = Propagator(c), initial(c)
        narrow = diagnostics(analyze(u, p, nq=160, qmax=10))
        wide = diagnostics(analyze(u, p, nq=320, qmax=16))
        self.assertGreater(narrow['epsilon2_bridge_rms'], 1e-7)
        self.assertLess(wide['epsilon2_bridge_rms'], 1e-11)


if __name__ == '__main__':
    unittest.main()
