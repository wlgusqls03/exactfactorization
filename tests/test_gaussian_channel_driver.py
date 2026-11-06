"""Complete-channel adapter and short CTMQC/FSSH driver checks.

These are formula/interface tests, not a convergence claim for dense channel
pilots or for the experimental Gaussian-projected dynamics. No long runs.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.channel_run import (
    ChannelConfig, ChannelEngine, channel_snapshot, run,
)
from multi_component_exact_factorization.gaussian_trajectory.internal_channels import InternalChannels
from multi_component_exact_factorization.gaussian_trajectory.models import InternalModel, ModelConfig
from multi_component_exact_factorization.gaussian_trajectory.run import AU_FS
from multi_component_exact_factorization.gaussian_trajectory.surface_hopping import coefficient_rhs


def configuration(**changes):
    values = dict(model=dict(name='pen-rabi', nq=32, qmin=-14., qmax=14., g=.01),
                  ntraj=7, method='ctmqc', compression='native', dt_au=.004,
                  end_fs=4*.004*AU_FS, every_fs=.004*AU_FS,
                  support_relative=1e-10, max_internal_dimension=64)
    values.update(changes)
    return ChannelConfig(**values)


class CompleteChannelTests(unittest.TestCase):
    def test_matrix_action_and_hermiticity_match_existing_models(self):
        rng = np.random.default_rng(904)
        models = [ModelConfig(name='pen-nonadiabatic', nq=16, g=.01, dse=True),
                  ModelConfig(name='pen-rabi', nq=16, g=.01, dse=False),
                  ModelConfig(name='mcef', nq=8, nx=8, qmin=-6., qmax=-2.)]
        for config in models:
            with self.subTest(model=config.name):
                model = InternalModel(config)
                R = model.initial_R_center+np.array([-.1, .07])
                channels = InternalChannels(model, max_dimension=64)
                H, dH = channels.matrices(R)
                G = rng.normal(size=(2, len(model.q), model.nk))+1j*rng.normal(size=(2, len(model.q), model.nk))
                weight = np.sqrt(model.quadrature)[None, :, None]
                vector = (G*weight).reshape(2, -1)
                action = np.einsum('ijk,ik->ij', H, vector)
                np.testing.assert_allclose(action, (model.action(R, G)*weight).reshape(2, -1), atol=3e-13)
                np.testing.assert_allclose(H, H.conj().transpose(0, 2, 1), atol=3e-14)
                h = 1e-5
                numerical = (channels.matrices(R+h)[0]-channels.matrices(R-h)[0])/(2*h)
                np.testing.assert_allclose(dH, numerical, atol=3e-10, rtol=2e-7)

    def test_complete_project_lift_and_no_implicit_truncation(self):
        model = InternalModel(ModelConfig(name='pen-rabi', nq=16, g=.01))
        adapter = InternalChannels(model, max_dimension=32)
        channels = adapter.evaluate(np.array([1.8, 2.1]))
        rng = np.random.default_rng(905)
        G = rng.normal(size=(2, 16, 2))+1j*rng.normal(size=(2, 16, 2))
        A = channels.project(G, model.quadrature)
        self.assertEqual(A.shape, (2, 32))
        np.testing.assert_allclose(channels.lift(A, model.quadrature, 2), G, atol=1e-13)
        np.testing.assert_allclose(np.sum(abs(A)**2, axis=1),
                                   np.einsum('q,iqk->i', model.quadrature, abs(G)**2), atol=3e-12)
        with self.assertRaisesRegex(ValueError, 'no implicit state truncation'):
            InternalChannels(model, max_dimension=16)

    def test_material_nac_and_moving_basis_cancel(self):
        model = InternalModel(ModelConfig(name='pen-nonadiabatic', nq=16, g=.01))
        adapter = InternalChannels(model, max_dimension=32)
        R = np.array([1.9, 2.2])
        channels = adapter.evaluate(R)
        h = 1e-5
        plus = adapter.evaluate(R+h, channels.vectors)
        minus = adapter.evaluate(R-h, channels.vectors)
        U_R = (plus.vectors-minus.vectors)/(2*h)
        np.testing.assert_allclose(U_R, np.einsum('ikn,inm->ikm', channels.vectors, channels.nac),
                                   atol=2e-8, rtol=2e-5)
        G = model.initial(R)
        A = channels.project(G, model.quadrature)
        velocity = np.array([.013, -.021])
        A_dot = coefficient_rhs(A, channels.energy, channels.nac, velocity)
        moving = np.einsum('ijn,in->ij', U_R, A)*velocity[:, None]
        material = np.einsum('ijn,in->ij', channels.vectors, A_dot)+moving
        expected = (-1j*model.action(R, G)*np.sqrt(model.quadrature)[None, :, None]).reshape(2, -1)
        np.testing.assert_allclose(material, expected, atol=2e-9, rtol=2e-7)

    def test_zero_ct_history_recovers_primitive_schrodinger_and_mte_force(self):
        for model in (dict(name='pen-nonadiabatic', nq=32, g=.01),
                      dict(name='mcef', nq=8, nx=8, qmin=-6., qmax=-2.)):
            with self.subTest(model=model['name']):
                engine = ChannelEngine(configuration(model=model))
                state, _ = engine.initial()
                G, _ = engine.reconstruct(state)
                rate, force, extra, channels, _, diagnostic = engine.native_rates(state, G)
                np.testing.assert_array_equal(state['f'], 0.)
                np.testing.assert_allclose(rate, -1j*engine.model.action(state['R'], G), atol=2e-14)
                np.testing.assert_allclose(force, engine.model.force(state['R'], G), atol=2e-14)
                np.testing.assert_array_equal(diagnostic['ct_force'], 0.)
                np.testing.assert_allclose(extra['f'], np.diagonal(channels.force_matrix, axis1=1, axis2=2).real)

    def test_tsh_force_is_active_surface_including_heavy_trap_once(self):
        for model in (dict(name='pen-rabi', nq=32, g=.01),
                      dict(name='mcef', nq=8, nx=8, qmin=-6., qmax=-2.)):
            with self.subTest(model=model['name']):
                engine = ChannelEngine(configuration(model=model, method='tsh'))
                state, _ = engine.initial()
                G, _ = engine.reconstruct(state)
                engine.active = np.arange(engine.c.ntraj) % engine.channels.dimension
                rate, force, extra, channels, _, _ = engine.native_rates(state, G)
                expected = channels.force_matrix[np.arange(engine.c.ntraj), engine.active, engine.active].real
                if not engine.model.is_photon:
                    expected = expected-.06*(state['R']-9.5)
                np.testing.assert_allclose(force, expected, atol=2e-14)
                np.testing.assert_allclose(rate, -1j*engine.model.action(state['R'], G), atol=2e-14)
                self.assertEqual(extra, {})


class ChannelDriverTests(unittest.TestCase):
    def test_tiny_native_and_gaussian_steps_and_snapshot(self):
        for method in ('ctmqc', 'tsh'):
            for compression in ('native', 'gaussian'):
                with self.subTest(method=method, compression=compression):
                    engine = ChannelEngine(configuration(method=method, compression=compression))
                    state, _ = engine.initial()
                    for _ in range(2):
                        state = engine.step(state)
                    self.assertLess(engine.guard(state), 1e-7)
                    frame, audit = channel_snapshot(engine, state, 2*engine.c.dt_au*AU_FS)
                    self.assertEqual(frame['channel_population'].shape, (7, 64))
                    np.testing.assert_allclose(frame['channel_population'].sum(axis=1), 1., atol=1e-7)
                    self.assertEqual(audit['retained_channels'], 64)
                    self.assertLess(engine.audit['max_roundtrip_error'], 1e-12)
                    if method == 'ctmqc':
                        self.assertGreater(np.max(abs(state['f'])), 0.)
                        self.assertIn('Q_pairs', frame)
                    else:
                        self.assertIn('active_surface_energy', frame)
                        self.assertTrue(np.all((frame['active_state'] >= 0) & (frame['active_state'] < 64)))

    def test_checkpoint_resume_preserves_history_active_anchor_and_rng(self):
        for method in ('ctmqc', 'tsh'):
            for compression in ('native', 'gaussian'):
                with self.subTest(method=method, compression=compression), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    whole = configuration(method=method, compression=compression)
                    half = replace(whole, end_fs=2*whole.dt_au*AU_FS)
                    run(whole, root/'whole', render=False)
                    run(half, root/'split', render=False)
                    with np.load(root/'split'/'checkpoint.npz') as middle:
                        halfway_rng = str(middle['rng_json'])
                    run(whole, root/'split', resume=True, render=False)
                    with np.load(root/'whole'/'checkpoint.npz') as x, np.load(root/'split'/'checkpoint.npz') as y:
                        self.assertEqual(int(y['step']), 4)
                        for key in x.files:
                            if key.startswith('state_') or key in ('active', 'anchor', 'rng_json', 'hops_json'):
                                np.testing.assert_array_equal(x[key], y[key], err_msg=key)
                        if method == 'ctmqc':
                            self.assertGreater(np.max(abs(y['state_f'])), 0.)
                        else:
                            self.assertNotEqual(str(y['rng_json']), halfway_rng)
                    status = json.loads((root/'split'/'status.json').read_text())
                    self.assertEqual(status['status'], 'COMPLETE_NOT_CERTIFIED')
                    self.assertLess(status['t_fs'], .001)
                    with self.assertRaises(FileExistsError):
                        run(whole, root/'whole', render=False)
                    preserved_status = (root/'split'/'status.json').read_bytes()
                    preserved_checkpoint = (root/'split'/'checkpoint.npz').read_bytes()
                    with self.assertRaisesRegex(ValueError, 'config/source mismatch'):
                        run(replace(whole, dt_au=.002), root/'split', resume=True, render=False)
                    self.assertEqual((root/'split'/'status.json').read_bytes(), preserved_status)
                    self.assertEqual((root/'split'/'checkpoint.npz').read_bytes(), preserved_checkpoint)

    def test_invalid_method_and_integrator_never_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            for invalid in (configuration(method='mte'), configuration(integrator='dop853')):
                path = Path(tmp)/'not_created'
                with self.assertRaises(ValueError):
                    run(invalid, path, render=False)
                self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
