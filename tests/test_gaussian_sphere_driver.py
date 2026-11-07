"""Opt-in sphere RK4 through real MTE/CTMQC/FSSH drivers, not a long-run claim."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.channel_run import (
    ChannelConfig, ChannelEngine, channel_snapshot, run,
)
from multi_component_exact_factorization.gaussian_trajectory.models import ModelConfig
from multi_component_exact_factorization.gaussian_trajectory.run import AU_FS, Engine, RunConfig
from multi_component_exact_factorization.gaussian_trajectory.sphere_integrator import SphereStepError


def config(**changes):
    values = dict(model=dict(name='pen-nonadiabatic', nq=32, g=.01), ntraj=7,
                  method='ctmqc', compression='gaussian', integrator='sphere-rk4',
                  dt_au=.002, end_fs=4*.002*AU_FS, every_fs=2*.002*AU_FS,
                  support_relative=0., max_internal_dimension=64)
    values.update(changes)
    return ChannelConfig(**values)


class SphereDriverTests(unittest.TestCase):
    def test_legacy_defaults_are_not_replaced(self):
        self.assertEqual(RunConfig().integrator, 'dop853')
        self.assertEqual(ChannelConfig().integrator, 'rk4')
        self.assertEqual(ModelConfig().q_operator, 'fft')
        self.assertEqual(ChannelConfig().support_relative, 0.)

    def test_mte_and_two_channel_methods_use_new_step_with_complete_state(self):
        for method in ('mte', 'ctmqc', 'tsh'):
            with self.subTest(method=method):
                c = config(method='ctmqc' if method == 'mte' else method)
                engine = Engine(c.base_config()) if method == 'mte' else ChannelEngine(c)
                state, _ = engine.initial()
                original = {key: value.copy() for key, value in state.items()}
                for _ in range(3):
                    state = engine.step(state)
                self.assertLess(engine.guard(state), 2e-14)
                np.testing.assert_allclose(np.sum(abs(state['C'])**2, axis=-1), 1., atol=2e-15)
                self.assertEqual(engine.last_integrator_audit['integrator'], 'sphere-rk4')
                self.assertEqual(engine.last_integrator_audit['rhs_evaluations'], 4)
                self.assertGreater(engine.audit['sphere_max_auxiliary_stage_norm_error'], 0)
                self.assertEqual(engine.c.support_relative, 0.)
                if method == 'ctmqc':
                    self.assertGreater(np.max(abs(state['f'])), 0.)
                    np.testing.assert_array_equal(original['f'], 0.)
                if method == 'tsh':
                    self.assertTrue(np.all(engine.active >= 0))
                    self.assertLess(engine.audit['max_hop_energy_error'], 1e-12)

    def test_small_step_matches_unmodified_RK4_on_same_physical_initial_data(self):
        for method in ('ctmqc', 'tsh'):
            with self.subTest(method=method):
                sphere = ChannelEngine(config(method=method, dt_au=.001))
                legacy = ChannelEngine(config(method=method, dt_au=.001, integrator='rk4'))
                a, _ = sphere.initial(); b, _ = legacy.initial()
                a = sphere.step(a); b = legacy.step(b)
                Ga = sphere.reconstruct(a)[0]; Gb = legacy.reconstruct(b)[0]
                error = np.sqrt(np.einsum('q,iqk->i', sphere.model.quadrature, abs(Ga-Gb)**2))
                self.assertLess(error.max(), 2e-11)
                np.testing.assert_allclose(a['R'], b['R'], atol=1e-13, rtol=0)
                np.testing.assert_allclose(a['P'], b['P'], atol=1e-13, rtol=0)

    def test_operator_selection_survives_snapshot_and_uses_matched_channels(self):
        for operator in ('fft', 'fd2', 'fd4'):
            with self.subTest(operator=operator):
                engine = ChannelEngine(config(model=dict(name='pen-rabi', nq=32, g=.01,
                                                         q_operator=operator)))
                state, _ = engine.initial(); state = engine.step(state)
                frame, audit = channel_snapshot(engine, state, engine.c.dt_au*AU_FS)
                self.assertEqual(audit['q_operator'], operator)
                self.assertEqual(audit['integrator']['integrator'], 'sphere-rk4')
                np.testing.assert_allclose(frame['channel_population'].sum(axis=1), 1., atol=2e-13)
                self.assertLess(engine.audit['max_roundtrip_error'], 1e-12)
                G = frame['G']; H = engine.channels.matrices(state['R'])[0]
                weighted = (G*np.sqrt(engine.model.quadrature)[None, :, None]).reshape(7, -1)
                expected = (engine.model.action(state['R'], G)*
                            np.sqrt(engine.model.quadrature)[None, :, None]).reshape(7, -1)
                np.testing.assert_allclose(np.einsum('ijk,ik->ij', H, weighted), expected, atol=2e-13)

    def test_native_sphere_configuration_refused_without_creating_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'invalid'
            with self.assertRaisesRegex(ValueError, 'requires Gaussian'):
                run(config(compression='native'), out, render=False)
            self.assertFalse(out.exists())
        with self.assertRaisesRegex(ValueError, 'requires Gaussian'):
            Engine(RunConfig(compression='native', integrator='sphere-rk4'))

    def test_failed_sphere_step_does_not_draw_hop_or_change_initial_state(self):
        engine = ChannelEngine(config(method='tsh', sphere_max_auxiliary_defect=1e-12))
        state, _ = engine.initial()
        original = {key: value.copy() for key, value in state.items()}
        rng = json.dumps(engine.rng.bit_generator.state, sort_keys=True)
        active = engine.active.copy(); anchor = engine.anchor.copy(); hops = engine.hops.copy()
        with self.assertRaises(SphereStepError):
            engine.step(state)
        self.assertEqual(json.dumps(engine.rng.bit_generator.state, sort_keys=True), rng)
        self.assertEqual(engine.hops, hops)
        np.testing.assert_array_equal(engine.active, active)
        np.testing.assert_array_equal(engine.anchor, anchor)
        for key in state:
            np.testing.assert_array_equal(state[key], original[key])

    def test_resume_is_bitwise_with_histories_basis_and_hop_random_state(self):
        for method in ('ctmqc', 'tsh'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); whole = config(method=method)
                half = replace(whole, end_fs=2*whole.dt_au*AU_FS)
                run(whole, root/'whole', render=False)
                run(half, root/'split', render=False)
                run(whole, root/'split', resume=True, render=False)
                with np.load(root/'whole'/'checkpoint.npz') as a, np.load(root/'split'/'checkpoint.npz') as b:
                    for key in a.files:
                        if key.startswith('state_') or key in ('anchor', 'active', 'rng_json', 'hops_json'):
                            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                    self.assertEqual(int(b['step']), 4)
                metadata = json.loads((root/'split'/'metadata.json').read_text())
                self.assertIn('homogeneous', metadata['integrator_policy'])
                self.assertEqual(json.loads((root/'split'/'status.json').read_text())['status'],
                                 'COMPLETE_NOT_CERTIFIED')
                status_before = (root/'split'/'status.json').read_bytes()
                checkpoint_before = (root/'split'/'checkpoint.npz').read_bytes()
                with self.assertRaisesRegex(ValueError, 'config/source mismatch'):
                    run(replace(whole, integrator='rk4'), root/'split', resume=True, render=False)
                self.assertEqual((root/'split'/'status.json').read_bytes(), status_before)
                self.assertEqual((root/'split'/'checkpoint.npz').read_bytes(), checkpoint_before)

    def test_rejected_step_records_auxiliary_failure_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'rejected'
            with self.assertRaises(SphereStepError):
                run(config(sphere_max_auxiliary_defect=1e-12), root, render=False)
            failure = json.loads((root/'failure.json').read_text())
            self.assertEqual(failure['attempted_step'], 1)
            self.assertGreater(failure['integrator_rejection']['max_auxiliary_stage_norm_error'], 1e-12)
            with np.load(root/'checkpoint.npz') as checkpoint:
                self.assertEqual(int(checkpoint['step']), 0)
                np.testing.assert_allclose(np.sum(abs(checkpoint['state_C'])**2, axis=-1), 1., atol=1e-14)


if __name__ == '__main__':
    unittest.main()
