"""Tiny checkpoint probes and orchestration tests; never run 40 fs locally."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from multi_component_exact_factorization.pen_perspectives_2026.resonance import paper_config
from multi_component_exact_factorization.pen_perspectives_2026.workflow import atomic_json, atomic_npz, load_npz
from multi_component_exact_factorization.pen_ctmqc_2026 import workflow
from multi_component_exact_factorization.pen_ctmqc_2026.core import Engine, Settings
from multi_component_exact_factorization.pen_ctmqc_2026.checkpoint_probe import (
    probe_case, read_checkpoint, main as probe_main)


def _fake_action(task):
    """Top-level/picklable action exercises the actual ProcessPoolExecutor."""
    out = task[2]
    if out.name == 'bad':
        raise FloatingPointError('manufactured guard failure before status write')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'status.json', dict(status='COMPLETE_NOT_CERTIFIED', t_fs=0.))


def _digests(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class RecoveryTests(unittest.TestCase):
    def _tasks(self, root):
        c, s = paper_config('rabi', 'off', nr=32), Settings('pol', ntraj=8)
        return [(c, s, root/name, dict(resume=False, end_fs=.01, every_fs=.01))
                for name in ('bad', 'after1', 'after2', 'after3')]

    def _case(self, root, view='pol'):
        c = paper_config('rabi', 'on', nr=32)
        s = Settings(view, ntraj=8)
        path = root/'input'/view
        workflow.simulate(c, s, path, .007, .005)
        return c, s, path

    def test_failed_action_does_not_cancel_later_cases_sequential_or_parallel(self):
        for jobs in (1, 2):
            with tempfile.TemporaryDirectory() as tmp:
                tasks = self._tasks(Path(tmp))
                result = workflow.execute_tasks(tasks, _fake_action, jobs)
                self.assertEqual([r['task_status'] for r in result],
                                 ['FAILED', 'SUCCEEDED', 'SUCCEEDED', 'SUCCEEDED'])
                self.assertIn('manufactured guard', result[0]['task_error'])
                for task in tasks[1:]:
                    self.assertTrue((task[2]/'status.json').exists())

    def test_interrupt_is_not_a_case_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = self._tasks(Path(tmp))
            with patch.object(workflow, 'run_task', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    workflow.execute_tasks(tasks, workflow.run_task, 1)

    def test_campaign_records_failure_and_returns_nonzero_after_other_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = self._tasks(root)
            with patch.object(workflow, 'build_tasks', return_value=tasks), \
                 patch.object(workflow, 'run_task', _fake_action), \
                 patch('multi_component_exact_factorization.pen_ctmqc_2026.render.comparison') as comparison:
                with self.assertRaises(SystemExit):
                    workflow.main(['run', '--out', str(root), '--no-movies', '--jobs', '2'])
                comparison.assert_not_called()
            report = json.loads((root/'ctmqc_campaign_report.json').read_text())
            self.assertEqual(len(report['failed_tasks']), 1)
            self.assertEqual(report['cases'][0]['status'], 'UNAVAILABLE')
            self.assertEqual(report['comparison_status'], 'skipped_task_failure')
            self.assertEqual(report['tasks'][-1]['task_status'], 'SUCCEEDED')

    def test_common_checkpoint_endpoints_and_no_input_writes_both_perspectives(self):
        for view in ('el', 'pol'):
            with tempfile.TemporaryDirectory() as tmp:
                c, s, path = self._case(Path(tmp), view)
                before = _digests(path)
                result = probe_case(path, forward_fs=.005)
                self.assertEqual(before, _digests(path))
                self.assertEqual([t['dt_au'] for t in result['trials']], [.1, .05, .025])
                self.assertTrue(all(t['status'] == 'COMPLETE_SHORT_PROBE' for t in result['trials']))
                self.assertEqual(len(result['common_endpoint_differences']), 2)
                for trial in result['trials']:
                    self.assertAlmostEqual(trial['last_accepted_t_fs'],
                                           result['checkpoint_t_fs']+result['actual_forward_fs'], places=14)
                # Independently continue the unchanged original dt; factor=1
                # must be exactly that operation, not a fresh initial ensemble.
                _, _, state, _ = read_checkpoint(path)
                engine = Engine(c, s)
                for _ in range(result['trials'][0]['target_steps']):
                    state, _ = engine.step(state)
                from multi_component_exact_factorization.pen_ctmqc_2026.core import observables
                self.assertEqual(observables(engine, state)['photon_number'],
                                 result['trials'][0]['last_accepted_observables']['photon_number'])

    def test_failed_coarse_trial_keeps_guard_and_runs_fine_trials_from_same_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _, path = self._case(Path(tmp))
            before = _digests(path)
            original_step = Engine.step

            def fail_coarse(engine, state):
                if engine.c.dt == c.dt:
                    raise FloatingPointError('manufactured coarse-step guard failure')
                return original_step(engine, state)

            with patch.object(Engine, 'step', fail_coarse):
                result = probe_case(path, forward_fs=.005)
            self.assertEqual(before, _digests(path))
            self.assertEqual(result['settings']['max_norm_step_error'], 1e-5)
            self.assertEqual(result['trials'][0]['accepted_steps'], 0)
            self.assertEqual(result['trials'][0]['last_accepted_t_fs'], result['checkpoint_t_fs'])
            self.assertEqual(result['trials'][0]['status'], 'FAILED_GUARD')
            self.assertEqual(len(result['common_endpoint_differences']), 1)
            self.assertEqual(result['common_endpoint_differences'][0]['coarse_dt_au'], .05)

    def test_incompatible_source_and_malformed_state_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = self._case(Path(tmp))
            saved = load_npz(path/'restart.npz')
            identity = json.loads(str(saved['identity'].item()))
            identity['dynamics_hashes']['core.py'] = 'wrong'
            corrupt = dict(saved, identity=json.dumps(identity))
            atomic_npz(path/'restart.npz', **corrupt)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                probe_case(path, .005)
            corrupt = dict(saved, P=saved['P'][:1])
            atomic_npz(path/'restart.npz', **corrupt)
            with self.assertRaisesRegex(ValueError, 'Invalid saved P'):
                probe_case(path, .005)

    def test_real_RK_norm_guard_fails_coarse_but_not_fine_in_manufactured_fast_model(self):
        # NOT the user's 25--33 fs failure: a separate fast Hamiltonian gives
        # a real (unmocked) RK norm violation and exercises the diagnostic path.
        with tempfile.TemporaryDirectory() as tmp:
            c = replace(paper_config('rabi', 'on', nr=32), omega=10., g=0.)
            path = Path(tmp)/'fast_test'
            workflow.simulate(c, Settings('pol', ntraj=8), path, 0., .005)
            before = _digests(path)
            result = probe_case(path, .005)
            self.assertEqual(result['trials'][0]['status'], 'FAILED_GUARD')
            self.assertIn('Unnormalized per-step coefficient norm error', result['trials'][0]['error'])
            self.assertEqual(result['trials'][1]['status'], 'COMPLETE_SHORT_PROBE')
            self.assertEqual(result['trials'][2]['status'], 'COMPLETE_SHORT_PROBE')
            self.assertEqual(before, _digests(path))

    def test_probe_refuses_long_runs_and_changed_dt_is_still_not_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, s, path = self._case(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'Short probe'):
                probe_case(path, 40.)
            with self.assertRaises(ValueError):
                probe_case(path, .005, (1, 0))
            with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                workflow.simulate(replace(c, dt=c.dt/2), s, path, .01, .005, True)

    def test_cli_selects_failed_only_and_rejects_input_output_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'campaign'
            case = root/'rabi'/'on'/'dse_off'/'strong'/'pol'
            good = root/'rabi'/'off'/'dse_off'/'strong'/'pol'
            c, s = paper_config('rabi', 'on', nr=32), Settings('pol', ntraj=8)
            workflow.simulate(c, s, case, .007, .005)
            workflow.simulate(c, s, good, .007, .005)
            atomic_json(case/'status.json', dict(status='FAILED'))
            original = _digests(root)
            out = Path(tmp)/'probe.json'
            probe_main(['--root', str(root), '--out', str(out), '--forward-fs', '.005'])
            report = json.loads(out.read_text())
            self.assertEqual(len(report['cases']), 1)
            self.assertEqual(report['cases'][0]['path'], str(case.resolve()))
            self.assertEqual(original, _digests(root))
            with self.assertRaises(SystemExit):
                probe_main(['--root', str(root), '--out', str(root/'bad.json')])
            with self.assertRaises(SystemExit):
                probe_main(['--root', str(root), '--out', str(out)])


if __name__ == '__main__':
    unittest.main()
