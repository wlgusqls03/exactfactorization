import argparse
import unittest

from multi_component_exact_factorization_gpu.throttle import (
    gpu_util_percent,
    throttle_delay,
)


class GPUThrottleTest(unittest.TestCase):
    def test_delay_matches_requested_duty_cycle(self):
        self.assertAlmostEqual(throttle_delay(2.0, 50.0), 2.0)
        self.assertAlmostEqual(throttle_delay(2.0, 80.0), 0.5)
        self.assertEqual(throttle_delay(2.0, 100.0), 0.0)

    def test_percentage_validation(self):
        self.assertEqual(gpu_util_percent("60"), 60.0)
        for invalid in ("0", "-1", "101", "not-a-number"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    gpu_util_percent(invalid)


if __name__ == "__main__":
    unittest.main()
