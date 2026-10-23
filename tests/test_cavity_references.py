"""Static Fock reference refinement must not mutate prior dynamics/results."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from multi_component_exact_factorization_cavity.reference_refine import refine
from multi_component_exact_factorization_cavity.workflow import run_case,write_json


class ReferenceRefinementTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.case=self.root/'vsc'/'strong'
        config=dict(nx=24,nR=48,nQ=24,Q_max=7.,R_left=-8.,R_right=8.,
            n_bo=3,n_vib=4,n_fock_analysis=2,t_final_fs=0.,dt_au=.01,
            local_state_method='single_well',edge_tolerance=.02,
            reference_R_points=4,reference_Q_points=5)
        with patch('multi_component_exact_factorization_cavity.render.render_case',
                   return_value={'test':'renderer mocked'}):
            run_case(config,self.case,campaign='vsc',case='strong',
                     cache=self.root/'cache',movies=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_refine_all_R_and_preserve_source_case(self):
        before={p:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.case.rglob('*') if p.is_file()}
        report=refine(self.case,self.root/'refined',fock=6,compare_fock=4)
        self.assertEqual(report['n_R'],48)
        self.assertTrue(report['all_input_hashes_unchanged'])
        self.assertTrue(report['physics_sources_verified'])
        for p,value in before.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),value)
        with np.load(self.case/'references.npz',allow_pickle=False) as a, np.load(
                report['refined_reference'],allow_pickle=False) as b:
            np.testing.assert_array_equal(a['cbo'],b['cbo'])
            np.testing.assert_array_equal(a['cbo_sample_energy'],b['cbo_sample_energy'])
            np.testing.assert_array_equal(a['R'],b['R'])
            self.assertEqual(b['pol_energy'].shape,(48,4))
            self.assertEqual(int(b['reference_fock_size']),6)
            self.assertEqual(int(b['original_population_fock_size']),2)
            self.assertGreater(np.max(abs(a['pol_energy']-b['pol_energy'])),1e-5)
        self.assertIsNotNone(report['global_vibrational_pairs']['refined'])
        with self.assertRaises(FileExistsError):
            refine(self.case,self.root/'refined',fock=6,compare_fock=4)

    def test_source_mismatch_and_inside_case_output_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside the original'):
            refine(self.case,self.case/'new_refs',fock=6,compare_fock=4)
        metadata=json.loads((self.case/'metadata.json').read_text())
        first=next(iter(metadata['physics_sources']))
        metadata['physics_sources'][first]='deliberate-test-mismatch'
        write_json(self.case/'metadata.json',metadata)
        with self.assertRaisesRegex(ValueError,'physics source hashes'):
            refine(self.case,self.root/'invalid',fock=6,compare_fock=4)
        self.assertFalse((self.root/'invalid').exists())


if __name__=='__main__':unittest.main()
