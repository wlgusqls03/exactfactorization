"""Staged runs must stop on failure and retain rejected-state evidence."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from multi_component_exact_factorization.gaussian_trajectory.run import AU_FS, RunConfig
from multi_component_exact_factorization.gaussian_trajectory.stability_validation import validate_stages
from multi_component_exact_factorization.gaussian_trajectory.channel_run import ChannelConfig


class StagedValidationTests(unittest.TestCase):
    def config(self, **kwargs):
        return ChannelConfig(model=dict(name='pen-rabi', nq=32, g=.01),
            ntraj=7, dt_au=.004, every_fs=.004*AU_FS, compression='gaussian',
            integrator='sphere-rk4', **kwargs)

    def test_real_stages_resume_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'pilot'
            c = self.config()
            result = validate_stages(c, out, [2*c.dt_au*AU_FS, 4*c.dt_au*AU_FS], render=False)
            self.assertTrue(result['all_targets_completed'])
            self.assertFalse(result['numerical_convergence_certified'])
            self.assertEqual(len(result['stages']), 2)
            self.assertAlmostEqual(result['saved_diagnostics']['last_saved_fs'], 4*c.dt_au*AU_FS)
            with self.assertRaises(FileExistsError):
                validate_stages(c, out, render=False)

    def test_failure_stops_before_long_targets_and_records_auxiliary_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'pilot'
            c = self.config(sphere_max_auxiliary_defect=1e-16)
            result = validate_stages(c, out, [.01, 10, 40], render=False)
            self.assertFalse(result['all_targets_completed'])
            self.assertEqual(len(result['stages']), 1)
            self.assertEqual(result['stages'][0]['last_saved_fs'], 0.)
            self.assertIn('integrator_rejection', result['stages'][0]['failure'])
            self.assertEqual(result['saved_diagnostics']['frames'], 1)

    def test_invalid_targets_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'pilot'
            for targets in ([], [0], [float('nan')], [1, 1], [10, 1]):
                with self.assertRaises(ValueError):
                    validate_stages(self.config(), out, targets, render=False)
                self.assertFalse(out.exists())

    def test_mte_rejected_step_not_misreported_as_last_saved_plus_one(self):
        from multi_component_exact_factorization.gaussian_trajectory.run import Engine, run
        c = RunConfig(model=dict(name='pen-rabi', nq=32, g=.01), ntraj=7,
            dt_au=.004, end_fs=.01, every_fs=.01, integrator='sphere-rk4')
        original = Engine.step
        count = [0]
        def fail_third(engine, state):
            count[0] += 1
            if count[0] == 3:
                raise FloatingPointError('test rejection')
            return original(engine, state)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'pilot'
            with patch.object(Engine, 'step', fail_third):
                with self.assertRaises(FloatingPointError):
                    run(c, out, render=False)
            failure = json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['last_saved_step'], 0)
            self.assertEqual(failure['attempted_step'], 3)
            self.assertEqual(failure['propagation_location']['last_accepted_step'], 2)

    def test_snapshot_failure_keeps_durable_time_and_is_not_a_rejected_step(self):
        from multi_component_exact_factorization.gaussian_trajectory import run as mte_module
        from multi_component_exact_factorization.gaussian_trajectory import channel_run as channel_module
        import numpy as np
        for module, snapshot_name, config in (
                (mte_module, 'snapshot', RunConfig(**{
                    k: v for k, v in self.config().base_config().__dict__.items()})),
                (channel_module, 'channel_snapshot', self.config())):
            original = getattr(module, snapshot_name)
            count = [0]
            def fail_second(*args, **kwargs):
                count[0] += 1
                if count[0] == 2:
                    raise OSError('test snapshot write failure')
                return original(*args, **kwargs)
            c = replace(config, end_fs=2*config.dt_au*AU_FS,
                        every_fs=2*config.dt_au*AU_FS)
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/'pilot'
                with patch.object(module, snapshot_name, fail_second):
                    with self.assertRaises(OSError):
                        module.run(c, out, render=False)
                failure = json.loads((out/'failure.json').read_text())
                self.assertEqual(failure['phase'], 'snapshot_and_save')
                self.assertIsNone(failure['attempted_step'])
                self.assertEqual(failure['last_saved_step'], 0)
                self.assertEqual(failure['last_saved_fs'], 0.)
                with np.load(out/'checkpoint.npz') as checkpoint:
                    self.assertEqual(int(checkpoint['step']), 0)


if __name__ == '__main__':
    unittest.main()
