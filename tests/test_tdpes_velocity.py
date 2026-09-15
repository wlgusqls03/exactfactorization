import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multi_component_exact_factorization.tdpes_velocity import overlay, prepare, FlowQuiver


class VelocityOverlayTests(unittest.TestCase):
    def setUp(self):
        q, R = np.linspace(-2, 2, 45), np.linspace(8, 12, 25)
        rho = np.ones((2, len(q), len(R)))*.1
        rho[:, :3] = 1e-5
        self.obs = dict(q=q, R=R, joint_density=rho,
                        options=dict(proton_mass=2., heavy_mass=4.))
        self.ef = dict(a=np.ones_like(rho)*2., b=np.ones_like(rho)*8.)

    def tearDown(self):
        plt.close('all')

    def test_same_velocity_all_panels_and_gauges(self):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        one = overlay(axes[0], self.obs, self.ef, 0)
        zero = dict(a=self.ef['a']*0, b=self.ef['b']*0,
                    gauge='zero_potential',
                    mechanical_q=self.ef['a'], mechanical_R_first=self.ef['b'])
        two = overlay(axes[1], self.obs, zero, 0)
        np.testing.assert_array_equal(one.U, two.U)
        np.testing.assert_array_equal(one.Umask, two.Umask)
        self.assertIs(axes[0]._tdpes_flow[0], axes[1]._tdpes_flow[0])
        axes[0].set(xlim=(-2, 2), ylim=(8, 12))
        axes[1].set(xlim=(-2, 2), ylim=(9, 10))
        fig.canvas.draw()
        self.assertFalse(np.allclose(one.angles, two.angles))
        np.testing.assert_allclose(one.U[~one.Umask], np.sqrt(5.))
        self.assertTrue(np.any(one.Umask))
        self.assertEqual(one.N, 38*18)

    def test_updates_do_not_accumulate_and_zero_speed_is_finite(self):
        self.ef['a'][1] = 0
        self.ef['b'][1] = 0
        fig, axis = plt.subplots()
        arrow = overlay(axis, self.obs, self.ef, 0)
        axis.set(xlim=(-2, 2), ylim=(8, 12))
        for f in (1, 0, 1):
            self.assertIs(overlay(axis, self.obs, self.ef, f), arrow)
            fig.canvas.draw()
        self.assertEqual(sum(isinstance(c, FlowQuiver) for c in axis.collections), 1)
        self.assertTrue(np.isfinite(arrow.angles).all())
        np.testing.assert_array_equal(arrow.U[~arrow.Umask], 0)

    def test_missing_velocity_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, 'require a/b'):
            prepare(self.obs, {})
        with self.assertRaisesRegex(ValueError, 'retained mechanical'):
            prepare(self.obs, dict(self.ef, gauge='zero_potential'))


if __name__ == '__main__':
    unittest.main()
