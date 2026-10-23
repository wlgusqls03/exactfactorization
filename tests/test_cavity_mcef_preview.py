"""The MCEF display adapter must not change physics, axes or trap accounting."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from multi_component_exact_factorization_cavity.mcef_preview import adapt_bo_surfaces, adapt_frame, selected_member_frames


class MCEFPreviewTests(unittest.TestCase):
    def test_selected_member_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.npz'
            values = np.arange(10*3*4).reshape(10, 3, 4)
            np.savez_compressed(path, values=values)
            got = dict(selected_member_frames(path, 'values', [0, 3, 8]))
            self.assertEqual(list(got), [0, 3, 8])
            for i, frame in got.items():
                np.testing.assert_array_equal(frame, values[i])
            with self.assertRaises(ValueError):
                list(selected_member_frames(path, 'values', [10]))

    def test_axes_sources_masses_and_trap(self):
        q, R = np.array([-1., 0., 1.]), np.array([9., 9.5])
        value = np.arange(6.).reshape(3, 2) + 1
        alpha = np.array([.4, -.2])
        source = dict(q=q, R=R, time_fs=40., joint_density=value,
                      lambda_density=value/2, a=value*2, b=value*3,
                      alpha=np.broadcast_to(alpha, value.shape),
                      S_q=value*.1, S_adv=value*.2, S_rel=value*.3,
                      density_dt=value*.6, map_stride=2)
        scalars = dict(epsilon_1=value*.5, epsilon_1_gi=value*.4,
                       epsilon_2=np.array([2., 3.]), epsilon_2_gi=np.array([1., 2.]))
        options = dict(proton_mass=1836., heavy_mass=12000., heavy_trap_alpha=.03,
                       heavy_trap_center=9.5, interaction_model='erf_shin_metiu',
                       fixed_ion_separation=19.)
        out = adapt_frame(source, scalars, np.array([.2, .3]), options)
        trap = .03*(R-9.5)**2
        self.assertEqual(str(out['system']), 'mcef')
        self.assertEqual(out['mass_q'], 1836.)
        self.assertEqual(out['mass_R'], 12000.)
        np.testing.assert_array_equal(out['rho_qR'], value.T)
        np.testing.assert_array_equal(out['S_q'], source['S_q'].T)
        np.testing.assert_array_equal(out['S_rel'], source['S_rel'].T)
        np.testing.assert_allclose(out['epsilon1'], (scalars['epsilon_1']+trap).T)
        np.testing.assert_allclose(out['epsilon1_GI'], (scalars['epsilon_1_gi']+trap).T)
        np.testing.assert_allclose(out['epsilon1_GD'], (scalars['epsilon_1']-scalars['epsilon_1_gi']).T)
        np.testing.assert_allclose(out['epsilon2_GD'], scalars['epsilon_2']-scalars['epsilon_2_gi'])
        np.testing.assert_array_equal(out['delta_b'], source['b'].T-alpha[:, None])
        self.assertNotIn('omega', out)
        self.assertNotIn('epsilon2_reconstructed', out)

    def test_gi_absent_is_not_invented(self):
        value = np.ones((3, 2))
        source = dict(q=np.arange(3.), R=np.array([9., 9.5]), time_fs=0.,
                      joint_density=value, lambda_density=value, a=value,
                      b=value, alpha=np.zeros(2))
        out = adapt_frame(source, {'epsilon_1': value, 'epsilon_2': np.zeros(2)},
                          np.ones(2), dict(proton_mass=1836., heavy_mass=12000.))
        self.assertNotIn('epsilon1_GI', out)
        self.assertNotIn('epsilon2_GD', out)

    def test_bo_trap_never_double_counted(self):
        R = np.array([9., 9.5, 10.])
        bare = np.arange(2*4*3.).reshape(2, 4, 3)
        options = dict(heavy_trap_alpha=.03, heavy_trap_center=9.5)
        full = bare + .03*(R-9.5)**2
        result = adapt_bo_surfaces(full, R, options, 'harmonic_included')
        expected = full.transpose(2, 1, 0)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(adapt_bo_surfaces(bare, R, options, 'external_harmonic_v1'), expected)
        with self.assertRaises(ValueError):
            adapt_bo_surfaces(full, R, options, 'unknown')


if __name__ == '__main__':
    unittest.main()
