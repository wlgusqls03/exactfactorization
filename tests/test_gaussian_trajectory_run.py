import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import json
import numpy as np
from multi_component_exact_factorization.gaussian_trajectory.run import (
    RunConfig, Engine, fit_initial_bridge, run)


def config(**kw):
    defaults = dict(model=dict(name='pen-rabi', nq=64, qmin=-14., qmax=14., g=0.),
                    ntraj=7, dt_au=.02, end_fs=.002, every_fs=.001, integrator='rk4', support_relative=1e-10,
                    mls=dict(neighbors=7, multistream_policy='report'))
    defaults.update(kw)
    return RunConfig(**defaults)


class RunTests(unittest.TestCase):
    def test_native_gaussian_uncoupled_short(self):
        engines = [Engine(config(compression=c)) for c in ('native', 'gaussian')]
        states = [engine.initial()[0] for engine in engines]
        for _ in range(10):
            states = [engine.step(s) for engine, s in zip(engines, states)]
        waves = [engine.reconstruct(s)[0] for engine, s in zip(engines, states)]
        np.testing.assert_allclose(waves[0], waves[1], atol=2e-7)
        for engine, s in zip(engines, states):
            self.assertLess(engine.guard(s), 1e-7)
            self.assertLess(abs(engine.model.observables(s['R'], engine.reconstruct(s)[0])['photon_number']).max(), 1e-8)

    def test_distinct_fit_rank_and_error(self):
        q = np.linspace(-8, 8, 128, endpoint=False); w = np.full(128, q[1]-q[0])
        rho = np.exp(-q*q/2)/np.sqrt(2*np.pi)
        z, audit = fit_initial_bridge(q, w, rho, 2)
        self.assertEqual(audit['tangent_rank'], 5)
        self.assertLess(audit['amplitude_l2_error'], .02)
        self.assertGreater(z[2]-z[1], .5)

    def test_initial_fit_underflow_is_not_divided_by_zero(self):
        q = np.linspace(-14, 14, 256, endpoint=False)
        w = np.full(len(q), q[1]-q[0])
        rho = np.exp(-.5*((q+4)/.32)**2)
        with np.errstate(divide='raise', invalid='raise'):
            _, audit = fit_initial_bridge(q, w, rho, 1)
        self.assertEqual(audit['tangent_rank'], 2)
        self.assertGreater(audit['rank_excluded_underflow_points'], 0)

    def test_initial_failure_leaves_diagnostic_not_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'failed'
            with patch.object(Engine, 'initial', side_effect=ValueError('bad initial fit')):
                with self.assertRaisesRegex(ValueError, 'bad initial fit'):
                    run(config(), root, render=False)
            self.assertEqual(json.loads((root/'status.json').read_text())['phase'], 'initialization')
            self.assertTrue((root/'config.json').is_file())
            self.assertTrue((root/'failure.json').is_file())
            self.assertFalse((root/'checkpoint.npz').exists())

    def test_restart_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = config(end_fs=.0039)
            run(a, root/'whole', render=False)
            b = config(end_fs=.0019)
            run(b, root/'split', render=False)
            run(a, root/'split', resume=True, render=False)
            with np.load(root/'whole'/'checkpoint.npz') as x, np.load(root/'split'/'checkpoint.npz') as y:
                for key in x.files:
                    if key.startswith('state_'): np.testing.assert_array_equal(x[key], y[key])
            with self.assertRaises(FileExistsError): run(a, root/'whole', render=False)
            with self.assertRaisesRegex(ValueError, 'config mismatch'):
                run(config(end_fs=.005, dt_au=.01), root/'split', resume=True, render=False)

    def test_unsupported_method_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'unsupported'
            with self.assertRaises(NotImplementedError):
                run(config(method='ctmqc'), out, render=False)
            self.assertFalse(out.exists())

    def test_adaptive_restart_tolerance(self):
        engine = Engine(config(integrator='dop853', support_relative=0.))
        state, _ = engine.initial()
        full, _ = engine.advance(state, 0, 11)
        split, _ = engine.advance(state, 0, 5)
        split, _ = engine.advance(split, 5, 11)
        # Global error in tiny-tail normalized C is more sensitive than in G.
        for key in full: np.testing.assert_allclose(full[key], split[key], atol=1e-8, rtol=1e-8)
        Gfull = engine.reconstruct(full)[0]; Gsplit = engine.reconstruct(split)[0]
        error = np.sqrt(np.einsum('q,iqk->i', engine.model.quadrature, abs(Gfull-Gsplit)**2))
        self.assertLess(error.max(), 1e-10)
