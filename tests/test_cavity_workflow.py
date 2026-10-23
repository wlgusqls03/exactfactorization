"""Initial-state and short checkpoint/resume workflow integration tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from multi_component_exact_factorization_cavity.workflow import run_case,write_npz,AU_PER_FS


class WorkflowTests(unittest.TestCase):
    def config(self):
        return dict(nx=24,nR=48,nQ=24,Q_max=7.,R_left=-8.,R_right=8.,
                    n_bo=3,n_vib=4,n_fock_analysis=4,t_final_fs=0.,dt_au=.01,
                    local_state_method='single_well',edge_tolerance=.02,
                    reference_R_points=4,reference_Q_points=5)

    def test_initial_state_pipeline_all_six_cases_and_collision_guard(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                'multi_component_exact_factorization_cavity.render.render_case',return_value={'test':'renderer mocked'}):
            root=Path(tmp)
            for campaign in ('pen','vsc'):
                for case in ('zero','weak','strong'):
                    out=root/campaign/case
                    status=run_case(self.config(),out,campaign=campaign,case=case,
                                    cache=root/'cache',movies=False)
                    self.assertEqual(status['step'],0)
                    with np.load(out/'observables.npz') as z:
                        self.assertLess(abs(z['norm'][0]-1),1e-12)
                        self.assertLess(abs(z['photon_number'][0]),1e-8)
                        self.assertEqual(z['bo_population'].shape,(1,3))
                    with np.load(out/'fields/fields_0000000.npz') as z:
                        self.assertIn('S_rel',z.files)
                        self.assertIn('epsilon1',z.files)
                        self.assertEqual(z['a'].dtype,np.float32)
                    self.assertTrue((out/'restart.npz').exists())
                    self.assertTrue((out/'references.npz').exists())
            with self.assertRaises(FileExistsError):
                run_case(self.config(),root/'pen/zero',campaign='pen',case='zero')
            # Zero-duration restart exercises identities/IO without propagation.
            resumed=run_case(self.config(),root/'pen/zero',campaign='pen',case='zero',
                             resume=True,cache=root/'cache',movies=False)
            self.assertEqual(resumed['step'],0)
            changed=self.config();changed['dt_au']=.02
            with self.assertRaisesRegex(ValueError,'Resume configuration mismatch'):
                run_case(changed,root/'pen/zero',campaign='pen',case='zero',resume=True)
            out=root/'pen/zero'
            config_before=(out/'config.json').read_bytes()
            metadata_before=(out/'metadata.json').read_bytes()
            with np.load(out/'restart.npz') as z:
                checkpoint={k:z[k].copy() for k in z.files}
            checkpoint['source_hash']=np.array('deliberately incompatible test source')
            write_npz(out/'restart.npz',**checkpoint)
            with self.assertRaisesRegex(ValueError,'source changed'):
                run_case(self.config(),out,campaign='pen',case='zero',resume=True)
            self.assertEqual((out/'config.json').read_bytes(),config_before)
            self.assertEqual((out/'metadata.json').read_bytes(),metadata_before)

    def test_undecided_vsc_rejected(self):
        c=self.config();c['local_state_method']=None
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'VSC requires'):
                run_case(c,Path(tmp)/'vsc',campaign='vsc')

    def test_nonzero_resume_matches_uninterrupted_propagation(self):
        """A real 2+2-step restart must equal an independent 4-step run."""
        c=self.config();dt=c['dt_au']
        c.update(field_every_fs=dt/AU_PER_FS,
                 checkpoint_every_fs=2*dt/AU_PER_FS,
                 t_final_fs=float(np.nextafter(2*dt/AU_PER_FS,0.)))
        with tempfile.TemporaryDirectory() as tmp, patch(
                'multi_component_exact_factorization_cavity.render.render_case',
                return_value={'test':'renderer mocked'}):
            root=Path(tmp);continued=root/'continued';whole=root/'uninterrupted'
            short=run_case(c,continued,campaign='pen',case='strong',
                           cache=root/'cache',movies=False)
            self.assertEqual(short['step'],2)
            # Only t_final_fs changes; all Hamiltonian, sampling and identity
            # settings remain identical across the checkpoint boundary.
            extended=dict(c,t_final_fs=float(np.nextafter(4*dt/AU_PER_FS,0.)))
            resumed=run_case(extended,continued,campaign='pen',case='strong',
                             resume=True,cache=root/'cache',movies=False)
            independent=run_case(extended,whole,campaign='pen',case='strong',
                                 cache=root/'cache',movies=False)
            self.assertEqual(resumed['step'],4)
            self.assertEqual(independent['step'],4)
            with np.load(continued/'restart.npz',allow_pickle=False) as a, np.load(
                    whole/'restart.npz',allow_pickle=False) as b:
                self.assertEqual(int(a['step']),4)
                np.testing.assert_allclose(a['psi'],b['psi'],rtol=0.,atol=2e-14)
                self.assertAlmostEqual(float(a['energy0']),float(b['energy0']),places=14)
            with np.load(continued/'observables.npz',allow_pickle=False) as a, np.load(
                    whole/'observables.npz',allow_pickle=False) as b:
                steps=a['step'].copy()
                np.testing.assert_array_equal(steps,np.arange(5))
                self.assertTrue(np.all(np.diff(steps)>0))
                self.assertEqual(len(np.unique(steps)),len(steps))
                np.testing.assert_allclose(a['norm'],b['norm'],rtol=0.,atol=2e-14)
                np.testing.assert_allclose(a['energy'],b['energy'],rtol=0.,atol=2e-14)
                self.assertLess(abs(float(a['norm'][-1])-1.),1e-12)
            journal=sorted((continued/'observables').glob('obs_*.npz'))
            journal_steps=[]
            for path in journal:
                with np.load(path,allow_pickle=False) as z:
                    journal_steps.append(int(z['step']))
            self.assertEqual(journal_steps,list(range(5)))
            self.assertEqual(len(journal_steps),len(set(journal_steps)))


if __name__=='__main__':unittest.main()
