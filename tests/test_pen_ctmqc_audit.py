"""Archive audit regression; tiny trajectory tests, never production runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from multi_component_exact_factorization.pen_ctmqc_2026.audit import audit_case, main
from multi_component_exact_factorization.pen_ctmqc_2026.core import Engine, Settings
from multi_component_exact_factorization.pen_ctmqc_2026.workflow import simulate
from multi_component_exact_factorization.pen_perspectives_2026.model import Config
from multi_component_exact_factorization.pen_perspectives_2026.workflow import atomic_json
from tests.test_pen_ctmqc_compare import digests


class CTMQCAuditTests(unittest.TestCase):
    def test_complete_archive_audit_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for view in ('el', 'pol'):
                out = root/view
                simulate(Config(nr=64), Settings(view, ntraj=16), out, .01, .01)
                before = digests(out)
                with patch.object(Engine, 'step', side_effect=AssertionError('No propagation')):
                    report = audit_case(out, target_fs=40.)
                self.assertTrue(report['file_checks_passed'], report['errors'])
                self.assertTrue(report['resume_compatible'])
                self.assertEqual(report['restart_frame_max_abs'], 0.)
                self.assertEqual(report['first_frame_reproduction_max_abs'], 0.)
                self.assertLess(report['photon_number_identity_max_abs'], 1e-14)
                self.assertGreater(report['extrapolation']['additional_steps'], 10000)
                self.assertEqual(before, digests(out))

    def test_inconsistent_saved_observables_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'pol'
            simulate(Config(nr=64), Settings('pol', ntraj=16), out, .01, .01)
            rows = json.loads((out/'observables.json').read_text())
            rows[-1]['energy'] += .01
            atomic_json(out/'observables.json', rows)
            report = audit_case(out)
            self.assertFalse(report['file_checks_passed'])
            self.assertTrue(any('observables disagree' in e for e in report['errors']))

    def test_missing_root_is_not_a_passing_audit(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            main(['--root', str(Path(tmp)/'missing'), '--out', str(Path(tmp)/'audit.json')])


if __name__ == '__main__':
    unittest.main()
