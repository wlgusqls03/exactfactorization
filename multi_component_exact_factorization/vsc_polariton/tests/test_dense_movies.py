"""Rendering/replay orchestration tests, not physical convergence certificates."""
import json
import tempfile
import signal
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from multi_component_exact_factorization.vsc_polariton.tests.test_real_grid_mcef_preview import fixture
from multi_component_exact_factorization.vsc_polariton.tests.test_photon_real_grid import fixture as packet_fixture
from multi_component_exact_factorization.vsc_polariton.real_grid_mcef_fields import analyze
from multi_component_exact_factorization.vsc_polariton.vsc_movie_only import compact_fields,inspect,render,epsilon1_display
from multi_component_exact_factorization.vsc_polariton import run_vsc_dense_movies as runner
from multi_component_exact_factorization.vsc_polariton.photon_real_grid import RealGridPF,initial_grid


class DenseMovieTests(unittest.TestCase):
    def test_schedule_and_storage(self):
        steps=runner.schedule(1652,.125,4)
        self.assertEqual(len(steps),414)
        self.assertEqual(steps[-1],13216)
        self.assertEqual(np.max(np.diff(steps)),32)
        self.assertLess(runner.estimate((352,160,384),414)['required_free_GiB'],6)
        with self.assertRaises(ValueError):runner.schedule(1,.125,.03)

    def test_observation_comparison_rejects_difference(self):
        row=dict(norm=1.,energy=0.,product=.1,flux=0.,nph=0.,mean_R=0.,P_exc=0.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'obs.npz';np.savez(path,**row)
            self.assertEqual(runner.compare_observation(row,path)['status'],'PASS')
            row['product']=.2
            with self.assertRaises(ValueError):runner.compare_observation(row,path)

    def test_actual_frame_movie_no_stills_and_sparse_guard(self):
        u,p,Q=fixture();f=analyze(u,p,Q)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[]
            for i in range(2):
                f['time_au']=i*4.;path=root/f'fields_{i:03d}.npz'
                np.savez(path,**compact_fields(f));paths.append(path)
            c=render(paths,root/'movies',p['omega'],families=('state','nuclear','photon'),dpi=45,
                     signed_scale='linear',floor=1e-4)
            self.assertEqual(c['signed_scale'],'linear')
            self.assertEqual(c['omega_c_au'],float(p['omega']))
            self.assertEqual(c['floor'],1e-4)
            self.assertEqual(c['frame_count'],2)
            self.assertFalse(c['sparse_preview'])
            self.assertEqual(len(list((root/'movies').glob('*.mp4'))),3)
            self.assertFalse(list((root/'movies').glob('*.png')))
            self.assertFalse(list((root/'movies').glob('*.pdf')))
            f['time_au']=100.;np.savez(paths[1],**compact_fields(f))
            with self.assertRaises(ValueError):inspect(paths,p['omega'],1e-5,1e-8)

    def test_epsilon1_manual_scale_preserves_raw_fields_and_offsets(self):
        u,p,Q=fixture();f=analyze(u,p,Q)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[]
            for i in range(2):
                f['time_au']=4.*i;path=root/f'fields_{i:03d}.npz'
                np.savez(path,**compact_fields(f));paths.append(path)
            before=[path.read_bytes() for path in paths]
            c=inspect(paths,p['omega'],1e-5,1e-8);offsets=list(c['offsets_Ha'])
            maximum=c['limits']['e1']
            epsilon1_display(paths,c,.1,'linear')
            self.assertEqual(c['limits']['e1'],.1)
            self.assertEqual(c['epsilon1_display']['original_max_abs_ev'],maximum)
            self.assertEqual(c['offsets_Ha'],offsets)
            self.assertGreater(c['epsilon1_display']['records'][0]['saturated_site_fraction'],0)
            self.assertGreater(c['epsilon1_display']['records'][0]['saturated_joint_probability_fraction'],0)
            self.assertEqual([path.read_bytes() for path in paths],before)
            for limit in (-1,0,np.nan,np.inf):
                with self.assertRaises(ValueError):epsilon1_display(paths,c,limit,'linear')
            shown=render(paths,root/'manual',p['omega'],families=('photon',),dpi=45,
                         epsilon1_vmax_ev=.1,epsilon1_scale='linear')
            self.assertEqual(shown['epsilon1_display']['scale'],'linear')
            self.assertTrue((root/'manual/vsc_photon_movie.mp4').is_file())

    def test_identical_replay_and_completed_reuse(self):
        # Tiny CPU test double covers orchestration; not a production GPU check.
        p=packet_fixture();cfg=SimpleNamespace(dt=.01,half_Q=10.,nq=64)
        info=dict(gpu='CPU test',cupy='none',driver=0,runtime=0)
        gate=dict(identity=runner.identity(cfg,'fixture'),gpu_propagation_allowed=True,environment=info)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();out=root/'replay';out.mkdir()
            status=dict(status='COMPLETE_NOT_CERTIFIED',identity=gate['identity'],time_au=.08)
            (source/'status.json').write_text(json.dumps(status))
            h=RealGridPF(p,10,64,.01);u,_=initial_grid(p,h.Q)
            for step in range(9):
                if step%2==0:np.savez(source/f'observable_{step:07d}.npz',**h.observe(u))
                if step<8:u=h.step(u)
            args=SimpleNamespace(out=out,source_run=source,dt=.01,end_au=.08,every_au=.02,block=2,device=0)
            factory=lambda packet,half,nq,dt,*unused:RealGridPF(packet,half,nq,dt)
            with patch.object(runner,'RealGridPF',factory),patch.object(runner,'environment',return_value=info),patch.object(runner,'failures',return_value={}):
                original_observe=RealGridPF.observe
                calls=[0]
                def interrupt_once(model,wave):
                    calls[0]+=1
                    if calls[0]==3:signal.raise_signal(signal.SIGTERM)
                    return original_observe(model,wave)
                with patch.object(RealGridPF,'observe',interrupt_once):
                    runner.replay(args,p,'fixture',gate,status)
                self.assertEqual(json.loads((out/'replay_status.json').read_text())['status'],'INTERRUPTED')
                runner.replay(args,p,'fixture',gate,status)
                runner.replay(args,p,'fixture',gate,status)
            self.assertEqual(len(list((out/'fields').glob('fields_*.npz'))),5)
            self.assertFalse(list(out.glob('wave_*.npz')))
            with np.load(out/'restart.npz') as z:np.testing.assert_allclose(z['psi'],u,atol=1e-13)
            self.assertEqual(json.loads((out/'replay_status.json').read_text())['status'],'COMPLETE_DIAGNOSTIC')


if __name__=='__main__':unittest.main()
