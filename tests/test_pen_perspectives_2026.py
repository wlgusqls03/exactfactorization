"""Small unit/regression tests for the separate 2026 two-Fock model."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import unittest
import numpy as np
from scipy.linalg import expm
from multi_component_exact_factorization.pen_perspectives_2026.model import (
    AU_FS, Config, Propagator, cbo, electronic, initial, local_hamiltonian,
    observables, oscillator, photon_operators, interval_integral)
from multi_component_exact_factorization.pen_perspectives_2026.fields import analyze, diagnostics
from multi_component_exact_factorization.pen_perspectives_2026.workflow import simulate, load_npz


class Perspectives2026Tests(unittest.TestCase):
    def test_exact_resonance_and_legacy_selection(self):
        from multi_component_exact_factorization.pen_perspectives_2026.resonance import select_frequency
        c, report = select_frequency(Config(), 'bare-gap')
        self.assertAlmostEqual(c.omega, .160000000000863, places=14)
        self.assertTrue(report['matches_bare_gap'])
        self.assertEqual(report['detuning_Ha'], 0)
        # Calibration must evaluate R0 exactly, not the closest grid point 2.008.
        self.assertNotEqual(c.grid()[np.argmin(abs(c.grid()-c.r2))], c.r2)
        self.assertEqual(select_frequency(c, 'paper-on')[0].omega, .17)
        custom = replace(c, omega=.123)
        self.assertEqual(select_frequency(custom, 'configured')[0], custom)
        self.assertEqual(select_frequency(c, 'off')[0].omega, .097)
        # Bare resonance stays common to DSE-on/off and g controls.
        for g in (0., .001, .01):
            for dse in (False, True):
                chosen, _ = select_frequency(replace(c, g=g, dse=dse), 'bare-gap')
                self.assertEqual(chosen.omega, c.omega)

    def test_cli_default_bare_gap_and_explicit_legacy(self):
        import sys
        with tempfile.TemporaryDirectory() as temp:
            for extra, value in (([], .160000000000863), (['--resonance', 'paper-on'], .17)):
                root = Path(temp)/str(value)
                subprocess.run([sys.executable, '-m', 'multi_component_exact_factorization.pen_perspectives_2026.workflow',
                    'run', '--out', str(root), '--end-fs', '0', '--case', 'strong', '--dse', 'off',
                    '--no-render']+extra, check=True, capture_output=True, text=True)
                saved = json.loads((root/'dse_off/strong/config.json').read_text())
                self.assertAlmostEqual(saved['omega'], value, places=14)
                provenance = json.loads((root/'dse_off/strong/resonance.json').read_text())
                self.assertEqual(provenance['R0'], 2.)

    def test_paper_parameters_and_width(self):
        c = Config()
        self.assertEqual((c.nr, c.rmin, c.rmax, c.dt, c.nf), (500, -2., 10., .1, 2))
        self.assertAlmostEqual(c.grid()[1]-c.grid()[0], .024)
        self.assertAlmostEqual(c.sigma_r, .22360679774997896)
        from multi_component_exact_factorization.vsc_polariton.model_a.model import Config as Old, potential
        R, q = np.array([2., 3.8, 6.]), np.array([-1., 0., 1.5])
        e, _ = cbo(c, R, q)
        np.testing.assert_allclose(e, np.linalg.eigvalsh(potential(Old(), R, q)), atol=1e-15)

    def test_project_before_squaring_photon(self):
        c = Config()
        op = photon_operators(c)
        np.testing.assert_allclose(c.omega*op['q2']/2+op['p2']/(2*c.omega)-np.eye(2)/2, op['number'], atol=1e-14)
        self.assertGreater(abs(op['q2'][1, 1]-(op['q']@op['q'])[1, 1]), 1.)
        for n in [0, 1]:
            p = Propagator(c)
            u = initial(c)
            u[:, :, n] = u[:, :, 0].copy()
            if n:
                u[:, :, 0] = 0
            self.assertAlmostEqual(observables(u, p)['photon_number'], n, places=12)

    def test_initial_bare_excited_vacuum(self):
        for dse in [True, False]:
            c = Config(dse=dse)
            o = observables(initial(c), Propagator(c))
            self.assertAlmostEqual(o['norm'], 1, places=13)
            self.assertAlmostEqual(o['bo_population'][1], 1, places=13)
            self.assertAlmostEqual(o['photon_number'], 0, places=13)
            self.assertAlmostEqual(o['q_mean'], 0, places=13)
            self.assertAlmostEqual(o['p_mean'], 0, places=13)

    def test_dividing_surface_partial_cell(self):
        for nr in (100, 500, 1000):
            R = Config(nr=nr).grid()
            density = 1+np.cos(2*np.pi*(R+2)/12)
            exact = 6.+12/(2*np.pi)*(np.sin(2*np.pi)-np.sin(np.pi))
            self.assertAlmostEqual(interval_integral(density, R, 4., 10.), exact, places=12)

    def test_dse_operator_and_zero_equivalence(self):
        c = Config(nr=32)
        R = c.grid()
        dip = electronic(c, R)[1]
        diff = local_hamiltonian(replace(c, dse=True), R)-local_hamiltonian(c, R)
        want = np.stack([np.kron(.5*c.g**2*(d@d), np.eye(c.nf)) for d in dip])
        np.testing.assert_allclose(diff, want, atol=1e-16)
        self.assertNotEqual(diff[10, 0, 2], 0)  # nuclear-transition cross term
        off = Propagator(replace(c, g=0, dse=False))
        on = Propagator(replace(c, g=0, dse=True))
        np.testing.assert_array_equal(on.step(initial(c)), off.step(initial(c)))

    def test_hermitian_and_strang_order(self):
        c = Config(nr=8, dt=.2, dse=True)
        p = Propagator(c)
        eye = np.eye(c.nr*2*c.nf)
        H = np.stack([p.action(eye[:, j].reshape(c.nr, 2, c.nf)).ravel() for j in range(len(eye))], axis=1)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-15)
        u = initial(c)
        exact = expm(-1j*c.dt*H)@u.ravel()
        coarse = np.linalg.norm(p.step(u).ravel()-exact)
        fine = Propagator(replace(c, dt=c.dt/2))
        error = np.linalg.norm(fine.step(fine.step(u)).ravel()-exact)
        self.assertGreater(coarse/error, 3.8)

    def test_local_q_projection_agrees(self):
        c = Config(nr=16)
        R = c.grid()
        q = np.linspace(-18, 18, 1600, endpoint=False)
        dq = q[1]-q[0]
        h, _, _ = oscillator(c, q)
        e, v = cbo(c, R, q)
        V = (v*e[..., None, :])@v.swapaxes(-1, -2)
        projected = np.einsum('nq,rqij,mq->rinjm', h, V, h)*dq
        projected = projected.reshape(c.nr, 2*c.nf, 2*c.nf)
        projected = projected+np.kron(np.eye(2), photon_operators(c)['p2']/2)[None]
        np.testing.assert_allclose(projected, local_hamiltonian(c, R), atol=2e-14)

    def test_ef_with_projection_source(self):
        c = Config(nr=256, dse=True)
        p = Propagator(c)
        u = initial(c)
        for _ in range(400):
            u = p.step(u)
        f = analyze(u, p, nq=200, qmax=15)
        d = diagnostics(f)
        for key in ['alpha_bridge_rms', 'epsilon2_bridge_rms', 'epsilon2_inversion_rms', 'continuity_rms']:
            self.assertLess(d[key], 1e-9, key)
        for key in ('epsilon1_GI_rms', 'epsilon2_GI_rms', 'weighted_pol_budget_rms'):
            self.assertLess(d[key], 1e-13, key)
        self.assertGreater(d['projection_source_rms'], 1e-7)
        mask = f['rho_qR'] > 1e-5*f['rho_qR'].max()
        old_residual = f['density_dt']-f['S_q']-f['S_adv']-f['S_rel']
        np.testing.assert_allclose(old_residual[mask], f['S_projection'][mask], atol=1e-12)

    def test_resolved_gi_budgets_dse_controls(self):
        rng = np.random.default_rng(14)
        for dse in (False, True):
            c = Config(nr=64, dse=dse)
            p = Propagator(c)
            # Nontrivial arbitrary complex state tests independent projections.
            u = rng.normal(size=(64, 2, 2))+1j*rng.normal(size=(64, 2, 2))
            u /= np.sqrt(np.sum(abs(u)**2)*p.dr)
            f = analyze(u, p, nq=64)
            np.testing.assert_allclose(f['epsilon1_wCBO'], f['epsilon1_cond'], atol=1e-13)
            np.testing.assert_allclose(f['epsilon2_wPol'], f['epsilon2_cond'], atol=1e-13)
            np.testing.assert_allclose(sum(f[k] for k in ('epsilon2_wBO', 'epsilon2_pkin',
                'epsilon2_ppot', 'epsilon2_LM', 'epsilon2_DSE')), f['epsilon2_wPol'], atol=1e-13)
            if not dse:
                np.testing.assert_array_equal(f['epsilon2_DSE'], 0.)

    def test_peak_slice_and_gi_layout(self):
        import matplotlib.pyplot as plt
        from multi_component_exact_factorization.pen_perspectives_2026.render import (
            prepare, sources, gi_components, packet_height)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'case'
            c = Config(nr=128)
            simulate(c, root, 0, .5)
            p = Propagator(c)
            paths = sorted((root/'coefficients').glob('*.npz'))
            display, refs, _, _ = prepare(paths, p, 64)
            f = analyze(initial(c), p, nq=64)
            # Force a different tracked coordinate; fixed cut must stay fixed.
            peak = display['i']+5
            f['rho_R'] = np.roll(f['rho_R'], 5)
            fig = sources(f, display, 0, refs)
            axes = [ax for ax in fig.axes if ax.get_title()]
            self.assertEqual(len(axes), 8)
            self.assertIn('partial_R', axes[3].get_title())
            self.assertIn(f'{f["R"][peak]:.3f}', axes[6].get_title())
            np.testing.assert_allclose(axes[4].get_ylim(), axes[6].get_ylim())
            np.testing.assert_allclose(axes[5].get_ylim(), axes[7].get_ylim())
            plt.close(fig)
            fig = gi_components(f, display, refs)
            panels = [ax for ax in fig.axes if ax.get_title().startswith('(')]
            self.assertEqual(len(panels), 6)
            self.assertTrue(any('R$-slice' in ax.get_title() for ax in panels))
            plt.close(fig)
        linear = packet_height(np.array([1., .01]), 1., 'linear')
        enhanced = packet_height(np.array([1., .01]), 1., 'sqrt')
        self.assertEqual(linear[0], enhanced[0])
        self.assertAlmostEqual(enhanced[1]/linear[1], 10.)

    def test_zero_a_and_projection(self):
        c = Config(nr=256, g=0)
        p = Propagator(c)
        u = initial(c)
        for _ in range(100):
            u = p.step(u)
        f = analyze(u, p)
        mask = f['rho_qR'] > 1e-5*f['rho_qR'].max()
        self.assertLess(np.max(abs(f['a'][mask])), 1e-12)
        self.assertLess(np.max(abs(f['S_projection'][mask])), 1e-12)

    def test_resume_and_all_six(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = Config(nr=64)
            for dse in [False, True]:
                for name, g in [('zero', 0.), ('weak', .001), ('strong', .01)]:
                    status = simulate(replace(c, g=g, dse=dse), root/f'{name}{dse}', .01, .005)
                    self.assertLess(status['max_norm_error'], 1e-12)
            simulate(c, root/'split', 10*c.dt*AU_FS, .005)
            simulate(c, root/'split', 30*c.dt*AU_FS, .005, resume=True)
            simulate(c, root/'whole', 30*c.dt*AU_FS, .005)
            np.testing.assert_allclose(load_npz(root/'split'/'restart.npz')['u'], load_npz(root/'whole'/'restart.npz')['u'], atol=1e-14)
            with self.assertRaises(ValueError):
                simulate(replace(c, dse=True), root/'split', .1, .005, resume=True)
            with self.assertRaises(FileExistsError):
                simulate(c, root/'whole', .1, .005)

    def test_render_and_cut(self):
        from multi_component_exact_factorization.pen_perspectives_2026.render import render_case, frame_paths
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'case'
            simulate(Config(nr=64), root, .005, .01)
            self.assertEqual(len(frame_paths(root, 0)), 1)
            report = render_case(root, movies=False, nq=64, dpi=40, t_max_fs=0)
            self.assertEqual(report['cache_frames'], 1)
            self.assertEqual(len(list((root/'plots_to_0fs').rglob('frame_*.png'))), 7)

    def test_title_stamping_is_idempotent(self):
        import matplotlib.pyplot as plt
        from multi_component_exact_factorization.pen_perspectives_2026.render import stamp
        fig = plt.figure()
        fig.suptitle('Source decomposition')
        fields = dict(t_fs=1., g_cavity=.01, dse=False)
        stamp(fig, fields)
        expected = fig._suptitle.get_text()
        stamp(fig, fields)
        self.assertEqual(fig._suptitle.get_text(), expected)
        self.assertEqual(len(expected.splitlines()), 2)
        plt.close(fig)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe unavailable')
    def test_movies_odd_dpi(self):
        from multi_component_exact_factorization.pen_perspectives_2026.render import render_case
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'case'
            simulate(Config(nr=64), root, .003, .01)
            render_case(root, movies=True, nq=64, dpi=41)
            movies = list((root/'plots').rglob('*.mp4'))
            self.assertEqual(len(movies), 6)
            for movie in movies:
                info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                    'stream=width,height,nb_frames', '-of', 'json', str(movie)], text=True))['streams'][0]
                self.assertEqual(int(info['nb_frames']), 2)
                self.assertEqual(info['width'] % 2, 0)
                self.assertEqual(info['height'] % 2, 0)


if __name__ == '__main__':
    unittest.main()
