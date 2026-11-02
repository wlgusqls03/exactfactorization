"""Small deterministic tests, no long/production trajectories."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from multi_component_exact_factorization.pen_perspectives_2026.model import (
    Config, electronic, local_hamiltonian, AU_FS)
from multi_component_exact_factorization.pen_perspectives_2026.resonance import paper_config
from multi_component_exact_factorization.pen_ctmqc_2026.core import (
    Engine, Settings, hamiltonian, observables, align_bands)
from multi_component_exact_factorization.pen_ctmqc_2026.quantum_momentum import (
    coupled_terms, density_quantum_momentum)
from multi_component_exact_factorization.pen_ctmqc_2026.workflow import (
    parser, build_tasks, preflight, simulate, capabilities)
from multi_component_exact_factorization.pen_ctmqc_2026.diagnostics import prepare_frame, deposit


class CTMQCTests(unittest.TestCase):
    def test_analytic_gradients_both_views_and_dse(self):
        rng = np.random.default_rng(1)
        for dse in (False, True):
            for view in ('el', 'pol'):
                c = Config(dse=dse, nf=3)
                X = np.column_stack((rng.uniform(1., 6., 12), rng.normal(size=12))) if view == 'el' else rng.uniform(1., 6., (12, 1))
                v, grad = hamiltonian(c, view, X)
                for axis in range(X.shape[1]):
                    dx = np.zeros_like(X)
                    dx[:, axis] = 1e-5
                    numeric = (hamiltonian(c, view, X+dx)[0]-hamiltonian(c, view, X-dx)[0])/2e-5
                    np.testing.assert_allclose(grad[:, axis], numeric, atol=1e-10, rtol=1e-7)
                np.testing.assert_allclose(v, v.swapaxes(1, 2), atol=0)
                if view == 'pol':
                    np.testing.assert_array_equal(v, local_hamiltonian(c, X[:, 0]))

    def test_common_wigner_draws_and_initial_norm(self):
        c = Config()
        el, pol = (Engine(c, Settings(view, ntraj=10000, initial='bare')).initial() for view in ('el', 'pol'))
        np.testing.assert_array_equal(el['X'][:, 0], pol['X'][:, 0])
        np.testing.assert_array_equal(el['P'][:, 0], pol['P'][:, 0])
        self.assertAlmostEqual(el['X'][:, 0].std(), c.sigma_r/np.sqrt(2), delta=.004)
        self.assertAlmostEqual(el['P'][:, 0].std(), 1/np.sqrt(2)/c.sigma_r, delta=.09)
        np.testing.assert_allclose(np.sum(abs(pol['coeff'])**2, axis=1), 1, atol=1e-14)
        np.testing.assert_array_equal(pol['coeff'][:, 1::c.nf], 0)
        eng = Engine(c, Settings('el', ntraj=10000))
        o = observables(eng, eng.initial())
        self.assertLess(abs(o['photon_number']), 4*o['photon_number_se'])
        self.assertAlmostEqual(o['cbo_population'][1], 1., places=13)

    def test_pairwise_population_constraint_and_force_covariance(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(35, 2))
        p = rng.uniform(.15, 1., (35, 4)); p /= p.sum(axis=1)[:, None]
        f = rng.normal(size=(35, 4, 2))
        m, widths = np.array([10., 1.]), np.array([.5, 1.])
        rate, force, detail = coupled_terms(X, p, f, m, widths)
        np.testing.assert_allclose(np.sum(p*rate, axis=1), 0., atol=1e-13)
        np.testing.assert_allclose(detail['ct_population_rate'], 0., atol=1e-13)
        self.assertGreater(np.linalg.norm(force), 0.)
        rate, force, _ = coupled_terms(X, p, f, m, widths, method='density')
        q = density_quantum_momentum(X, widths)
        mean = np.einsum('nk,nkd->nd', p, f)
        covariance = np.einsum('nk,nka,nkb->nab', p, f, f)-mean[:, :, None]*mean[:, None, :]
        np.testing.assert_allclose(force, 2*np.einsum('na,nab->nb', q/m, covariance), atol=1e-13)
        np.testing.assert_allclose(rate, np.sum(q[:, None]/m*(f-mean[:, None]), axis=2), atol=1e-13)

    def test_singular_intercept_is_reported_not_infinite(self):
        X = np.array([[-1.], [1.]])
        p = np.full((2, 2), .5)
        f = np.array([[[1.], [0.]], [[-1.], [0.]]])
        rate, force, info = coupled_terms(X, p, f, np.ones(1), np.ones(1))
        np.testing.assert_array_equal(rate, 0)
        np.testing.assert_array_equal(force, 0)
        self.assertEqual(info['q_disabled_directions'], 1)

    def test_distant_intercept_guard_reproduces_short_EL_failure_case(self):
        # Without the center-distance guard this seed produced a center >2000
        # near an ensemble spanning q in [-5,5], failing at step 53.
        e = Engine(paper_config('nonadiabatic', 'on'), Settings('el', ntraj=128))
        z, guarded = e.initial(), 0
        for _ in range(420):  # only 1.016 fs
            z, err = e.step(z)
            guarded += int(e.last_step_guard_counts[2])
            self.assertLess(err, e.s.max_norm_step_error)
        self.assertGreater(guarded, 0)
        self.assertTrue(np.all(np.isfinite(z['coeff'])))
        self.assertLess(np.max(abs(np.asarray(observables(e, z)['ct_population_rate']))), 1e-12)

    def test_eigenprojector_tracking_under_sign_and_permutation(self):
        anchor = np.eye(4)[None]
        permutation = [2, 0, 3, 1]
        vectors = anchor[:, :, permutation]*np.array([-1., 1., -1., 1.])
        energies, result = align_bands(np.arange(4.)[None, permutation], vectors, anchor)
        np.testing.assert_array_equal(energies, np.arange(4.)[None])
        np.testing.assert_allclose(abs(result), anchor)

    def test_g0_decoupling_and_ehrenfest_energy(self):
        for view in ('el', 'pol'):
            c = paper_config('rabi', 'on', g=0., dt=.1)
            e = Engine(c, Settings(view, ntraj=24, algorithm='ehrenfest'))
            z = e.initial(); before = observables(e, z)
            for _ in range(120):
                z, _ = e.step(z)
            after = observables(e, z)
            self.assertLess(abs(after['photon_number']-before['photon_number']), 1e-9)
            self.assertLess(abs(after['energy']-before['energy']), 1e-9)
            self.assertLess(after['coefficient_norm_max_error'], 1e-13)
            np.testing.assert_array_equal(after['ct_force_rms'], 0.)

    def test_frozen_R_matches_exact_local_quantum_propagation(self):
        c = replace(paper_config('rabi', 'on', dt=.05), b=0., z=0.)
        e = Engine(c, Settings('pol', ntraj=4, algorithm='ctmqc'))
        z = e.initial(); z['X'][:] = c.r2; z['P'][:] = 0.
        v = local_hamiltonian(c, z['X'][:, 0])
        energies, vectors = np.linalg.eigh(v)
        z['anchor'] = vectors
        before = z['coeff'].copy()
        for _ in range(200): z, _ = e.step(z)
        ad = np.einsum('nik,ni->nk', vectors, before)
        exact = np.einsum('nik,nk->ni', vectors, ad*np.exp(-1j*energies*10.))
        np.testing.assert_allclose(z['coeff'], exact, atol=3e-10)
        np.testing.assert_allclose(z['X'], c.r2, atol=1e-14)

    def test_all_eight_paper_configs_and_routing(self):
        args = parser().parse_args(['plan', '--out', '/tmp/ctmqc-plan-not-created'])
        tasks = build_tasks(args)
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len({str(t[2]) for t in tasks}), 8)
        for c, s, out, opt in tasks:
            model, resonance = out.parts[-5:-3]
            self.assertEqual(c, paper_config(model, resonance))
            self.assertEqual(s.ntraj, 10000)
            self.assertEqual(c.nf, 2)
        self.assertIn('exact a,b,alpha', capabilities()['unavailable'])

    def test_resume_identical_and_preflight_protects_old_results(self):
        c = paper_config('rabi', 'off', nr=32)
        s = Settings('pol', ntraj=12)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'split', Path(tmp)/'whole'
            simulate(c, s, a, .012, .01)
            task = [(c, s, a, dict(resume=False, end_fs=.025, every_fs=.01))]
            with self.assertRaises(FileExistsError): preflight(task, 'run')
            with self.assertRaises(ValueError): simulate(c, replace(s, seed=9), a, .025, .01, True)
            simulate(c, s, a, .025, .01, True)
            simulate(c, s, b, .025, .01)
            with np.load(a/'restart.npz') as za, np.load(b/'restart.npz') as zb:
                for key in ('X', 'P', 'coeff', 'integrated_forces', 'anchor'):
                    np.testing.assert_array_equal(za[key], zb[key])

    def test_display_is_mixed_density_and_no_exact_fields(self):
        for view in ('el', 'pol'):
            c = Config(); s = Settings(view, ntraj=48)
            e = Engine(c, s); z = e.initial(); _, d = e.evaluate(z, z['anchor'])
            frame = dict(X=z['X'], P=z['P'], coeff=z['coeff'], mean_energy=d['mean_energy'],
                         mf_force=d['mf_force'], ct_force=d['ct_force'], q_pair_rms=d['q_pair_weighted_local_rms'], t_fs=0.)
            f = prepare_frame(c, s, frame)
            self.assertAlmostEqual(f['display_mass'], 1., delta=2e-4)
            mask = f['joint'] > 1e-5*f['joint'].max()
            np.testing.assert_allclose(f['local_CBO_population'][mask].sum(axis=1), 1., atol=1e-13)
            self.assertNotIn('epsilon2', f)
            self.assertNotIn('alpha', f)
            self.assertEqual(f['joint'].shape, (241, 161))
        deposited = deposit(np.array([[100.]]), [np.linspace(-1, 1, 10)], np.ones(1), [.1])
        np.testing.assert_array_equal(deposited, 0)

    def test_dt_convergence_both_views(self):
        for view in ('el', 'pol'):
            states = []
            for dt in (.2, .1, .05):
                e = Engine(paper_config('rabi', 'on', dt=dt), Settings(view, ntraj=16))
                z = e.initial()
                for _ in range(int(round(4/dt))): z, _ = e.step(z)
                states.append(z['coeff'])
            coarse = np.linalg.norm(states[0]-states[2])
            fine = np.linalg.norm(states[1]-states[2])
            self.assertLess(fine, coarse*.2)


if __name__ == '__main__':
    unittest.main()
