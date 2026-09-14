import unittest
import numpy as np
from multi_component_exact_factorization.density_contours import decade_levels, decade_color, automatic_absolute_cutoff
from multi_component_exact_factorization import render_final_visualizations as render


class DensityContourTests(unittest.TestCase):
    def test_contour_bounding_box_and_styles_match_cell_index_reference(self):
        class Capture:
            def contour(self, *args, **kwargs):
                return args, kwargs
        rng = np.random.default_rng(44)
        q, R = np.arange(37.), np.arange(29.)
        for density in (rng.random((37, 29))*.2,
                        np.pad(rng.random((9, 11))*.03, ((7, 21), (8, 10)))):
            original = density.copy()
            obs = dict(q=q, R=R)
            for compact in (True, False):
                (cq, cr, z), style = render._joint_linear_contours(Capture(), obs, density, compact)
                major, minor = decade_levels(1e-3, max(density.max(), .01))
                levels = np.sort(np.r_[major, minor])
                levels = levels[(levels > density.min()) & (levels < density.max())]
                indices = np.nonzero(density >= levels[0])
                crop = tuple(slice(max(0, int(i.min())-1), min(n, int(i.max())+2))
                             for i, n in zip(indices, density.shape))
                np.testing.assert_array_equal(cq, q[crop[0]])
                np.testing.assert_array_equal(cr, R[crop[1]])
                np.testing.assert_array_equal(z, density[crop].T)
                np.testing.assert_array_equal(style['levels'], levels)
                flags = [np.any(np.isclose(v, major, rtol=1e-10, atol=0)) for v in levels]
                expected_colors = [render.to_rgba(decade_color(v), .95) if f
                                   else render.to_rgba('black', .55) for v, f in zip(levels, flags)]
                np.testing.assert_array_equal(style['colors'], expected_colors)
                self.assertEqual(style['linestyles'], 'solid')
                self.assertEqual(style['linewidths'], [(.85 if compact else 1.15) if f
                                 else (.18 if compact else .25) for f in flags])
            np.testing.assert_array_equal(density, original)

    def test_outer_boundary_black_other_decades_unchanged(self):
        self.assertEqual(decade_color(1e-3), 'black')
        self.assertEqual(decade_color(1e-2), '#d89000')
        self.assertEqual(decade_color(1e-1), '#c000c0')

    def test_all_2d_focus_is_absolute_and_overlay_replaces_artists(self):
        q = np.linspace(-2, 2, 20)
        density = np.exp(-q[:, None]**2-q[None, :]**2)*.1
        obs = {'q': q, 'R': q, 'joint_density': np.array([density, density*.1])}
        for frame in (0, 1):
            active, _, _ = render._frame_focus(obs, frame, .9)
            np.testing.assert_array_equal(active, obs['joint_density'][frame] >= 1e-3)
        fig, ax = render.plt.subplots()
        try:
            first = render._absolute_overlay(ax, obs, 0)
            old = list(first.collections)
            render._absolute_overlay(ax, obs, 1)
            self.assertTrue(all(c not in ax.collections for c in old))
        finally:
            render.plt.close(fig)
        args = render.parse_args(['dummy'])
        self.assertEqual(args.nested_density_contours, 'absolute')
        self.assertEqual(args.nested_absolute_density_floor, 1e-3)

    def test_decades_and_uniform_minors(self):
        major, minor = decade_levels(1e-3, 1.)
        np.testing.assert_allclose(major, [1e-3, .01, .1, 1.])
        np.testing.assert_allclose(minor[minor > .1], np.arange(3, 20)*.05)
        np.testing.assert_allclose(minor[(minor > .01) & (minor < .1)], np.arange(3, 20)*.005)
        np.testing.assert_allclose(minor[minor < .01], np.arange(10, 20)*.0005)
        self.assertAlmostEqual(minor.min(), .005)

    def test_absolute_lowest_decade_stops_halfway(self):
        major, minor = decade_levels(1e-4, .2)
        self.assertAlmostEqual(major.min(), 1e-4)
        self.assertAlmostEqual(minor.min(), 5e-4)
        self.assertAlmostEqual(automatic_absolute_cutoff(np.array([.14, .2]), .001), 1e-4)

    def test_absolute_mask_does_not_follow_frame_peak(self):
        args = render.parse_args(['dummy'])
        args._nested_contour_mode = 'absolute'
        args._nested_absolute_cutoff = .01
        rho = np.array([[[1., .005], [.02, .1]], [[.1, .005], [.02, .1]]])
        obs = {'joint_density': rho, 'heavy_density': rho.sum(axis=1)}
        ef = {'electron_proton_density': np.ones_like(rho),
              'epsilon_1': np.zeros_like(rho), 'epsilon_2': np.zeros((2, 2))}
        obs.update(R=np.array([0., 1.]), dq=1., options={})
        a = render._nested_frame(obs, ef, 0, args)
        b = render._nested_frame(obs, ef, 1, args)
        np.testing.assert_array_equal(a['joint_opacity'], b['joint_opacity'])
        np.testing.assert_array_equal(a['joint_opacity'], [[1, 0], [1, 1]])
