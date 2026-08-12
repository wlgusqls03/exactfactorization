import unittest

import numpy as np

from multi_component_exact_factorization.core import (
    fixed_center_crossing_probabilities,
)


class FixedCenterCrossingTests(unittest.TestCase):
    def test_separates_left_right_and_normalizes(self):
        grid = np.arange(-3.0, 4.0)
        density = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
        left, right, total = fixed_center_crossing_probabilities(
            density, grid, 0.5, -1.0, 1.0, normalization=2.0,
        )
        self.assertEqual(left, (1.0+2.0)*0.5/2.0)
        self.assertEqual(right, (32.0+64.0)*0.5/2.0)
        self.assertEqual(total, left+right)

    def test_fixed_center_sites_are_not_counted_as_crossed(self):
        grid = np.array([-1.0, 0.0, 1.0])
        left, right, total = fixed_center_crossing_probabilities(
            np.ones(3), grid, 1.0, -1.0, 1.0,
        )
        self.assertEqual((left, right, total), (0.0, 0.0, 0.0))

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            fixed_center_crossing_probabilities(
                np.ones(2), np.ones(3), 1.0, -1.0, 1.0,
            )


if __name__ == "__main__":
    unittest.main()
