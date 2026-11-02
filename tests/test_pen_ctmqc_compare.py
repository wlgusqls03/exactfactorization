"""Read-only diagnostics tests; manufactured archives, no production dynamics."""
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from multi_component_exact_factorization.pen_perspectives_2026.model import (
    Config, Propagator, initial, observables)
from multi_component_exact_factorization.pen_perspectives_2026.workflow import atomic_json, atomic_npz
from multi_component_exact_factorization.pen_ctmqc_2026.core import Engine, Settings, observables as ct_observables
from multi_component_exact_factorization.pen_ctmqc_2026.compare_data import (
    Archive, load_group, matching, overlap, aligned_errors, nearest_frame,
    choose_snapshots, FrameReader, snapshot_metrics)
from multi_component_exact_factorization.pen_ctmqc_2026.compare import main, scan
from multi_component_exact_factorization.pen_ctmqc_2026 import compare_plot


RELATIVE = Path('rabi/on/dse_off/strong')


def fixture(root):
    c = Config(nr=128, r1=2., delta=.17)
    qdroot, ctroot = root/'qd', root/'ct'
    qp = qdroot/RELATIVE
    (qp/'coefficients').mkdir(parents=True)
    p = Propagator(c); u = initial(c)
    atomic_json(qp/'config.json', asdict(c))
    atomic_json(qp/'status.json', {'status': 'COMPLETE_SYNTHETIC_TEST_ONLY'})
    rows = []
    for k, t in enumerate((0., .1)):
        # Duplicate snapshots with a global phase are a data fixture, not a run.
        atomic_npz(qp/'coefficients'/f'wave_{k:08d}.npz', u=u*np.exp(-1j*t), t_fs=t)
        rows.append(dict(t_fs=t, **observables(u, p)))
    atomic_json(qp/'observables.json', rows)
    for view in ('el', 'pol'):
        path = ctroot/RELATIVE/view
        (path/'frames').mkdir(parents=True)
        s = Settings(view, ntraj=64, initial='bare')
        e = Engine(c, s); z = e.initial(); _, d = e.evaluate(z, z['anchor'])
        atomic_json(path/'config.json', asdict(c))
        atomic_json(path/'trajectory_config.json', asdict(s))
        atomic_json(path/'status.json', {'status': 'COMPLETE_SYNTHETIC_TEST_ONLY'})
        rows = []
        for k, t in enumerate((0., .1)):
            atomic_npz(path/'frames'/f'trajectories_{k:08d}.npz', X=z['X'], P=z['P'],
                       coeff=z['coeff'], t_fs=t, mean_energy=d['mean_energy'],
                       mf_force=d['mf_force'], ct_force=d['ct_force'])
            rows.append(dict(t_fs=t, **ct_observables(e, z)))
        atomic_json(path/'observables.json', rows)
    return ctroot, qdroot


def digests(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class CTMQCCompareTests(unittest.TestCase):
    def test_physics_matching_not_display_or_time_grid(self):
        c = Config()
        def archive(view, cfg): return Archive(Path('.'), view, cfg, None, [], [], np.array([]), {})
        qd = archive('qd', c)
        self.assertEqual(matching(qd, archive('pol', replace(c, nr=256, dt=.05))), [])
        self.assertEqual(matching(qd, archive('pol', replace(c, omega=.16))), ['omega'])
        self.assertEqual(matching(qd, archive('pol', replace(c, nf=4))), ['nf'])
        self.assertEqual(matching(qd, archive('el', replace(c, nf=4))), [])

    def test_frame_alignment_no_extrapolation_or_interpolated_wave(self):
        a = Archive(Path('.'), 'qd', Config(), None,
                    [dict(t_fs=t) for t in (0., 1., 2.)], [Path('a')]*3, np.array([0., 1., 2.]), {})
        self.assertIsNone(nearest_frame(a, -1.))
        self.assertIsNone(nearest_frame(a, 3.))
        self.assertIsNone(nearest_frame(a, .49, .1))
        self.assertEqual(nearest_frame(a, .5), 0)
        self.assertEqual(choose_snapshots({'qd': a}, (0., 2.), 1.2), [0., 1., 2.])
        self.assertIsNone(overlap({'qd': a}, -1))

    def test_scalar_interpolation_is_overlap_only(self):
        def a(method, ts):
            rows = [dict(t_fs=t, photon_number=2*t) for t in ts]
            return Archive(Path('.'), method, Config(), None, rows, [], np.array([]), {})
        archives = {'qd': a('qd', [0., .5, 1., 1.5]), 'pol': a('pol', [.25, 1.25])}
        interval = overlap(archives)
        self.assertEqual(interval, (.25, 1.25))
        result = aligned_errors(archives, interval)['pol']['photon_number']
        self.assertEqual(result['t_fs'], [.5, 1.])
        self.assertEqual(result['max_abs'], 0)

    def test_coarse_trace_keeps_interpolated_overlap_endpoints(self):
        a = Archive(Path('.'), 'pol', Config(), Settings('pol'),
                    [dict(t_fs=t) for t in (0., 1.)], [], np.array([]), {})
        fig, ax = plt.subplots()
        compare_plot.time_line(ax, a, np.array([1., 3.]), (0., .1))
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [0., .1])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [1., 1.2])
        plt.close(fig)

    def test_fallback_and_frequency_mismatch_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ct, qd = fixture(Path(tmp))
            path = ct/RELATIVE/'pol'/'config.json'
            c = replace(Config(nr=128, r1=2., delta=.17), omega=.12)
            atomic_json(path, asdict(c))
            archives, warnings = load_group(ct, qd, RELATIVE)
            self.assertEqual(set(archives), {'qd', 'el'})
            self.assertTrue(any('SKIPPED' in w and 'omega' in w for w in warnings))
            partial, warnings = load_group(ct, qd/'missing', RELATIVE)
            self.assertEqual(set(partial), {'el'})
            self.assertTrue(any('unavailable' in w for w in warnings))

    def test_matched_density_and_no_fake_el_pol_or_ef_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            ct, qd = fixture(Path(tmp))
            archives, _ = load_group(ct, qd, RELATIVE)
            r = FrameReader(archives)
            frames = {k: r.read(k, 0) for k in archives}
            for k, f in frames.items():
                self.assertAlmostEqual(f['density_mass_R'], 1., delta=2e-3)
                self.assertAlmostEqual(f['density_mass_q'], 1., delta=2e-3)
                np.testing.assert_allclose(f['bo_density'].sum(axis=1), f['rho_R'], atol=1e-13)
                self.assertTrue(np.isfinite(f['current_R']).all())
                if k != 'qd': self.assertIsNone(f['native'])
            self.assertIsNone(frames['el']['pol_density'])
            self.assertIsNone(frames['el']['fock_population'])
            np.testing.assert_allclose(frames['qd']['fock_population'], [1., 0.], atol=1e-13)
            # Initial pure vacuum gets exactly the same q smoothing for QD/POL.
            np.testing.assert_allclose(frames['qd']['rho_q'], frames['pol']['rho_q'], atol=2e-5)
            report = snapshot_metrics(frames)
            self.assertIn('el', report['matched_resolution_density_L1'])
            z = np.load(archives['el'].paths[0])
            mean = np.trapz(np.nan_to_num(frames['el']['outer_mean_energy'])*frames['el']['rho_R'], r.R)
            self.assertAlmostEqual(mean, np.mean(z['mean_energy']+.5*z['P'][:, 1]**2), delta=1e-5)

    def test_photon_identity_and_raw_el_value_not_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            ct, qd = fixture(Path(tmp))
            archives, _ = load_group(ct, qd, RELATIVE)
            for a in archives.values():
                n = a.config.omega*a.values('q2')/2+a.values('p2')/(2*a.config.omega)-.5
                np.testing.assert_allclose(n, a.values('photon_number'), atol=1e-14)
            archives['el'].rows[0]['photon_number'] = -.02
            self.assertEqual(archives['el'].values('photon_number')[0], -.02)

    def test_native_QD_force_decomposition_and_shared_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            ct, qd = fixture(Path(tmp))
            archives, _ = load_group(ct, qd, RELATIVE)
            reader = FrameReader(archives)
            frames = {k: reader.read(k, 0) for k in archives}
            f = frames['qd']['native']
            np.testing.assert_allclose(f['force']+f['quantum_amplitude_force'], f['hydrodynamic_force'], atol=0)
            scales = scan(reader, [0.], None, None)
            fig = compare_plot.density(frames, reader, scales, 'test')
            self.assertEqual(fig.axes[0].get_ylim(), fig.axes[2].get_ylim())
            self.assertEqual(fig.axes[2].get_ylim(), fig.axes[3].get_ylim())
            plt.close(fig)

    def test_cli_render_does_not_propagate_or_modify_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ct, qd = fixture(root)
            before_ct, before_qd = digests(ct), digests(qd)
            with patch.object(Propagator, 'step', side_effect=AssertionError('No propagation allowed')), \
                 patch.object(Engine, 'step', side_effect=AssertionError('No propagation allowed')):
                main(['--ctmqc-root', str(ct), '--qd-root', str(qd), '--out', str(root/'plots'),
                      '--snapshot-times', '0', '--t-max-fs', '0', '--dpi', '60'])
            self.assertEqual(digests(ct), before_ct)
            self.assertEqual(digests(qd), before_qd)
            self.assertTrue((root/'plots'/RELATIVE/'matching.json').exists())
            self.assertGreaterEqual(len(list((root/'plots').rglob('*.png'))), 8)
            self.assertFalse(list((root/'plots').rglob('*.npz')))

    def test_missing_root_cannot_silently_render_qd_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, qd = fixture(root)
            with patch.object(compare_plot, 'observations', side_effect=AssertionError('No rendering')), \
                 self.assertRaises(SystemExit) as error:
                main(['--ctmqc-root', str(root/'typo'), '--qd-root', str(qd), '--out', str(root/'plots')])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root/'plots').exists())

    def test_incomplete_or_too_short_groups_fail_before_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ct, qd = fixture(root)
            for extra in (['--require-through-fs', '40'], ['--require-through-fs', 'nan']):
                with self.assertRaises(SystemExit):
                    main(['--ctmqc-root', str(ct), '--qd-root', str(qd), '--out', str(root/'plots')]+extra)
                self.assertFalse((root/'plots').exists())
            archives, _ = load_group(ct, qd, RELATIVE)
            with patch('multi_component_exact_factorization.pen_ctmqc_2026.compare.load_group',
                       return_value=({'qd': archives['qd']}, ['CT missing'])), self.assertRaises(SystemExit):
                main(['--ctmqc-root', str(ct), '--qd-root', str(qd), '--out', str(root/'plots')])
            self.assertFalse((root/'plots').exists())

    def test_partial_opt_in_is_labelled_not_advertised_as_three_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, qd = fixture(root)
            main(['--ctmqc-root', str(root/'intentionally_absent'), '--qd-root', str(qd),
                  '--out', str(root/'plots'), '--allow-partial', '--t-max-fs', '0', '--dpi', '60'])
            import json
            report = json.loads((root/'plots'/RELATIVE/'matching.json').read_text())
            self.assertFalse(report['complete_three_way'])
            self.assertEqual(report['missing_methods'], ['el', 'pol'])
            archives, _ = load_group(root/'absent', qd, RELATIVE)
            self.assertIn('PARTIAL', compare_plot.method_heading(archives))
            self.assertNotIn('CTMQC EL', compare_plot.method_heading(archives))

    def test_coincident_traces_have_distinct_line_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            ct, qd = fixture(Path(tmp)); archives, _ = load_group(ct, qd, RELATIVE)
            fig, ax = plt.subplots()
            for a in archives.values():
                compare_plot.time_line(ax, a, np.zeros(2), (0., .1))
            self.assertEqual(len({line.get_linestyle() for line in ax.lines}), 3)
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
