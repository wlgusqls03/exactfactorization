import tempfile
import unittest
from pathlib import Path
from .test_phase6_completion import packet
from ..phase6_stationary_compat import ground_pair,stationary
from ..run_phase6_finish import verify_manifest


class FinishTests(unittest.TestCase):
    def test_old_api(self):
        def old(op,k,which,tol,maxiter):return [0],[[1]]
        self.assertFalse(ground_pair(old,None,None)[2])

    def test_new_api(self):
        def new(op,k,which,tol,maxiter,v0):
            self.assertEqual(v0,42)
            return [0],[[1]]
        self.assertTrue(ground_pair(new,None,42)[2])

    def test_solver_error_not_hidden(self):
        def bad(op,k,which,tol,maxiter):raise TypeError('internal error')
        with self.assertRaisesRegex(TypeError,'internal error'):ground_pair(bad,None,None)

    def test_stationary_same_thresholds(self):
        with tempfile.TemporaryDirectory() as d:stationary(packet(),Path(d)/'s.json',gpu=False)

    def test_integrity_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'protected';f.write_text('unchanged')
            with self.assertRaises(RuntimeError):verify_manifest({str(f):'wrong'})
