import tempfile
import unittest
import numpy as np
from pathlib import Path
from .test_phase6_completion import packet
from ..phase6_stationary_compat import ground_pair,stationary
from ..run_phase6_finish import verify_manifest


class FinishTests(unittest.TestCase):
    def test_old_api(self):
        def old(op,k,which,tol,maxiter):
            self.assertEqual(which,'LA')
            return np.array([0]),np.array([[1]])
        self.assertFalse(ground_pair(old,None,None)[2])

    def test_new_api(self):
        def new(op,k,which,tol,maxiter,v0):
            self.assertEqual(v0,42)
            self.assertEqual(which,'LA')
            return np.array([0]),np.array([[1]])
        self.assertTrue(ground_pair(new,None,42)[2])

    def test_solver_error_not_hidden(self):
        def bad(op,k,which,tol,maxiter):raise TypeError('internal error')
        with self.assertRaisesRegex(TypeError,'internal error'):ground_pair(bad,None,None)

    def test_negative_operator_selects_ground_not_largest_magnitude(self):
        from scipy.sparse.linalg import LinearOperator,eigsh
        energies=np.array([-3.,-1.,2.,100.])
        op=LinearOperator((4,4),matvec=lambda v:-energies*v,dtype=np.float64)
        # Emulate the actual legacy API rejecting SA and not accepting v0.
        def legacy(op,k,which,tol,maxiter):
            if which not in ('LM','LA'):raise ValueError('legacy which')
            return eigsh(op,k=k,which=which,tol=tol,maxiter=maxiter)
        values,vectors,used=ground_pair(legacy,op,np.ones(4))
        self.assertFalse(used)
        self.assertAlmostEqual(values[0],-3.,places=11)
        self.assertLess(np.linalg.norm(energies*vectors[:,0]-values[0]*vectors[:,0]),1e-10)

    def test_stationary_same_thresholds(self):
        with tempfile.TemporaryDirectory() as d:stationary(packet(),Path(d)/'s.json',gpu=False)

    def test_integrity_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'protected';f.write_text('unchanged')
            with self.assertRaises(RuntimeError):verify_manifest({str(f):'wrong'})
