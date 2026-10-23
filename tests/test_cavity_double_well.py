import unittest

import numpy as np

from multi_component_exact_factorization_cavity.double_well_search import (
    bo_curve, local_levels, stationary_points, convergence_checks,
)
from multi_component_exact_factorization_cavity.core import DEFAULTS


class DoubleWellSearchTests(unittest.TestCase):
    def test_known_quartic_and_single_well(self):
        r = np.linspace(-3, 3, 601)
        t = stationary_points(r, (r*r-1)**2)
        self.assertEqual(len(t['barriers']), 1)
        b = t['barriers'][0]
        self.assertAlmostEqual(b['left']['R'], -1, places=5)
        self.assertAlmostEqual(b['right']['R'], 1, places=5)
        self.assertAlmostEqual(b['left_barrier_Ha'], 1, places=7)
        self.assertFalse(stationary_points(r, r*r)['double_well'])
        self.assertFalse(stationary_points(r, r)['double_well'])

    def test_real_model_one_parameter_change_and_defaults_preserved(self):
        before = dict(DEFAULTS)
        r = np.linspace(-8.8, 8.8, 101)
        old = bo_curve({}, r, nx=96)
        new = bo_curve({'erf_r_lx':4.}, r, nx=96)
        self.assertFalse(old['topology']['double_well'])
        self.assertTrue(new['topology']['double_well'])
        self.assertEqual(DEFAULTS, before)
        self.assertEqual(new['config']['heavy_fixed'], 9.5)
        self.assertEqual(new['config']['erf_r_qx'], 5.)

    def test_local_state_norm_and_dirichlet_wall(self):
        r = np.linspace(-4, 4, 401)
        result = local_levels(r, .01*(r*r-1)**2, 0., spacing=.04)
        np.testing.assert_allclose(result['wave'].T@result['wave']*result['spacing'], np.eye(6), atol=1e-12)
        self.assertTrue(np.all(result['grid'] < 0))
        self.assertTrue(np.all(np.diff(result['energy']) > 0))

    def test_convergence_gate_rejects_lost_well(self):
        r=np.linspace(-3,3,601)
        good={'topology':stationary_points(r,.01*(r*r-1)**2)}
        cases={name:good for name in ('native','fine','wide')}
        self.assertTrue(convergence_checks(cases)['passed'])
        cases['native']={'topology':stationary_points(r,r*r)}
        self.assertFalse(convergence_checks(cases)['passed'])


if __name__ == '__main__':
    unittest.main()
