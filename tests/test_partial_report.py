import unittest

import numpy as np

from multi_component_exact_factorization import compact_report
from multi_component_exact_factorization.visualize import LoadedArchive


class PartialReportLabelTests(unittest.TestCase):
    def test_completed_or_legacy_archive_has_no_watermark(self):
        legacy = LoadedArchive({"times_fs": np.array([0.0, 1.0])})
        completed = LoadedArchive({
            "times_fs": np.array([0.0, 1.0]),
            "propagation_completed": np.array(True),
        })

        self.assertEqual(compact_report._trajectory_prefix(legacy), "")
        self.assertEqual(compact_report._trajectory_prefix(completed), "")

    def test_failed_archive_reports_reached_and_requested_times(self):
        partial = LoadedArchive({
            "times_fs": np.array([0.0, 5.078]),
            "propagation_completed": np.array(False),
            "requested_final_time_fs": np.array(50.0),
        })

        self.assertEqual(
            compact_report._trajectory_prefix(partial),
            "PARTIAL trajectory (5.078/50 fs) | ",
        )


if __name__ == "__main__":
    unittest.main()
