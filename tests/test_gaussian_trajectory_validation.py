"""Fast physical-limit audits, separated from trajectory-pilot tests."""
from pathlib import Path
import tempfile
import unittest

from multi_component_exact_factorization.gaussian_trajectory.validation import (
    rabi_reference, vacuum_box_audit, phase_mode_dt_audit,
    mls_convergence_audit, run_validation)


class ValidationTests(unittest.TestCase):
    def test_fixed_R_rabi_exchange_and_zero_control(self):
        report = rabi_reference()
        zero,coupled = report['cases']
        self.assertLess(zero['max_ground_population'],1e-12)
        self.assertGreater(coupled['max_ground_population'],.8)
        for row in (zero,coupled):
            self.assertLess(row['norm_error'],1e-12)
            self.assertLess(row['energy_drift_Ha'],1e-12)
            self.assertLess(row['hermiticity_error'],1e-12)
            self.assertLess(row['max_q_edge_mass'],1e-15)
        self.assertEqual(report['dimension'],128)
        self.assertIn('FIXED-R',report['label'])

    def test_box_defect_and_MLS_convergence(self):
        rows = vacuum_box_audit()['rows']
        self.assertGreater(rows[0]['vacuum_action_defect_Ha'],1e-5)
        self.assertLess(rows[-1]['vacuum_action_defect_Ha'],1e-10)
        rows = mls_convergence_audit()['rows']
        self.assertLess(rows[-1]['first_relative_error'],rows[0]['first_relative_error']/100)
        self.assertLess(rows[-1]['second_relative_error'],rows[0]['second_relative_error']/100)

    def test_literal_K1_phase_only_RK4_convergence(self):
        rows = phase_mode_dt_audit()['rows']
        for row in rows:
            self.assertLess(row['parameter_change'],1e-12)
            self.assertAlmostEqual(row['native_l2_error'],row['K1_l2_error'],delta=1e-13)
        self.assertGreater(rows[0]['K1_l2_error']/rows[1]['K1_l2_error'],14.)
        self.assertLess(rows[-1]['K1_l2_error'],1e-9)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root/'original.txt'
            marker.write_text('keep me')
            with self.assertRaises(FileExistsError):
                run_validation(root)
            self.assertEqual(marker.read_text(),'keep me')


if __name__ == '__main__':
    unittest.main()
