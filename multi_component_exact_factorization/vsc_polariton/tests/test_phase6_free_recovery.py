"""Exact vacuum reduction and bounded recovery safety, not physics tuning."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from multi_component_exact_factorization.vsc_polariton.tests.test_phase6_completion import packet
from multi_component_exact_factorization.vsc_polariton.phase6_gpu_backend import PFBackend
from multi_component_exact_factorization.vsc_polariton.run_phase6_free_recovery import boundary_only,job


class FreeRecoveryTests(unittest.TestCase):
    def test_exact_vacuum_subspace(self):
        p=packet();full=PFBackend(p,.125);a=p['psi'].copy()
        reduced=dict(p,psi=a[:,:,:1],photon=p['photon'][:1],rotation=np.ones((1,1)),displacement=np.zeros(1))
        h=PFBackend(reduced,.125);b=reduced['psi'].copy()
        for _ in range(128):a,b=full.step(a),h.step(b)
        np.testing.assert_allclose(a[:,:,:1],b,atol=1e-12,rtol=0)
        self.assertLess(np.linalg.norm(a[:,:,1:]),1e-12)
        for k in ('product','flux','energy','mean_R','P_exc'):
            self.assertAlmostEqual(full.observe(a)[k],h.observe(b)[k],places=11)

    def test_only_boundary_can_advance_scan(self):
        self.assertTrue(boundary_only(dict(status='FAILED_DIAGNOSTIC',failures={'electron_edge':{'value':1.1e-8}})))
        self.assertFalse(boundary_only(dict(status='FAILED_DIAGNOSTIC',failures={'electron_edge':{'value':float('nan')}})))
        for failures in ({'continuity':{}},{'norm':{}},{'electron_edge':{},'energy':{}}):
            self.assertFalse(boundary_only(dict(status='FAILED_DIAGNOSTIC',failures=failures)))

    def test_terminal_packet_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'full').mkdir()
            (out/'full/status.json').write_text(json.dumps(dict(input_sha256='old',dt=.125,status='COMPLETE_NOT_CERTIFIED')))
            with patch('multi_component_exact_factorization.vsc_polariton.run_phase6_free_recovery.load_packet',return_value=({},'new')):
                with self.assertRaises(RuntimeError):job(Path('input'),out,.125,0)


if __name__=='__main__':unittest.main()
