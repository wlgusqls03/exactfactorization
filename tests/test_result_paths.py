from pathlib import Path
import unittest

from result_paths import dated_results_dir


class DatedResultsDirTests(unittest.TestCase):
    def test_inserts_date_after_results(self):
        self.assertEqual(
            dated_results_dir("results/run_name", "20260730"),
            Path("results/20260730/run_name"),
        )

    def test_places_plain_relative_name_below_results_date(self):
        self.assertEqual(
            dated_results_dir("run_name", "20260730"),
            Path("results/20260730/run_name"),
        )

    def test_does_not_duplicate_existing_date(self):
        path = Path("results/20260730/run_name/figures")
        self.assertEqual(dated_results_dir(path, "20260730"), path)

    def test_preserves_absolute_results_root(self):
        self.assertEqual(
            dated_results_dir("/scratch/project/results/run_name", "20260730"),
            Path("/scratch/project/results/20260730/run_name"),
        )

    def test_dates_absolute_custom_directory_before_leaf(self):
        self.assertEqual(
            dated_results_dir("/scratch/run_name", "20260730"),
            Path("/scratch/20260730/run_name"),
        )

    def test_does_not_redate_descendant_of_absolute_dated_directory(self):
        path = Path("/scratch/20260730/run_name/figures")
        self.assertEqual(dated_results_dir(path, "20260730"), path)

    def test_rejects_invalid_explicit_date(self):
        with self.assertRaises(ValueError):
            dated_results_dir("results/run_name", "2026-07-30")


if __name__ == "__main__":
    unittest.main()
