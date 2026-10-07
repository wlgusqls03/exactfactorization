"""Short numerical and presentation tests, not long-time certification."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import matplotlib.pyplot as plt
from .model import Config,Propagator,initial,observables
from .factorization import analyze,compact
from .state_movies import Channels
from .component_detail import draw
from .extend_2000 import evolve
from .event_audit import sha


class ExtensionTests(unittest.TestCase):
    def setUp(self):self.c=Config(nr=64,nq=64,qmin=-16,qmax=16)

    def test_bare_initial_population_and_closure(self):
        f=Channels(self.c).project(initial(self.c))
        self.assertLess(f['BO_closure'],1e-12)
        self.assertLess(f['CBO_closure'],1e-12)
        np.testing.assert_allclose(f['BO_global'],[0,1],atol=1e-12)
        self.assertAlmostEqual(sum(f['CBO_global']),1,places=12)
        self.assertEqual(f['BO_q0_density'].shape,(64,2))
        self.assertEqual(f['CBO_R1_density'].shape,(64,2))

    def test_global_projection_matches_existing_observables(self):
        p=Propagator(self.c);u=initial(self.c)
        for _ in range(8):u=p.step(u)
        f=Channels(self.c).project(u);o=observables(u,p)
        np.testing.assert_allclose(f['BO_global'],[o['P_S0'],o['P_S1']],atol=1e-12)

    def test_channel_eigenvector_phase_invariance(self):
        ch=Channels(self.c);u=initial(self.c);a=ch.project(u)
        ch.bo=ch.bo.astype(complex)*np.exp(.7j);ch.cbo=ch.cbo.astype(complex)*np.exp(-.4j)
        b=ch.project(u)
        np.testing.assert_allclose(a['CBO_q0_density'],b['CBO_q0_density'],atol=1e-12)
        np.testing.assert_allclose(a['BO_global'],b['BO_global'],atol=1e-12)

    def test_dense_run_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';c=self.c
            s=evolve(c,root,end=.2,every=.1)
            p=Propagator(c);u=initial(c)
            for _ in range(4):u=p.step(u)
            with np.load(root/'restart.npz') as z:np.testing.assert_allclose(z['psi'],u,atol=1e-14)
            before=sha(root/'restart.npz')
            evolve(c,root,end=.2,every=.1,resume=True)
            self.assertEqual(before,sha(root/'restart.npz'))
            self.assertEqual(len(list((root/'states').glob('*.npz'))),3)
            self.assertTrue(s['complete']);self.assertFalse(s['phase_pass'])
            with self.assertRaises(FileExistsError):evolve(c,root,end=.2,every=.1)
            with self.assertRaises(ValueError):evolve(c,root,end=.3,every=.1,resume=True)

    def test_continuation_preserves_old_inputs(self):
        # Synthetic checkpoint tests restart mechanics, not physical time 1250.
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);old=base/'old';(old/'waves').mkdir(parents=True)
            u=initial(self.c);p=Propagator(self.c);row=observables(u,p)
            row.update(step=0,time_au=0.,time_fs=0.)
            np.savez(old/'waves/wave_1250.npz',psi=u,time_au=1250.)
            (old/'observables.json').write_text(json.dumps([row]));(old/'ef_diagnostics.json').write_text('[]')
            before={str(x):sha(x) for x in old.rglob('*') if x.is_file()}
            result=evolve(self.c,base/'new',end=1250.1,every=.1,source=old,dense=False)
            with np.load(base/'new/restart.npz') as z:np.testing.assert_allclose(z['psi'],p.step(p.step(u)),atol=1e-14)
            self.assertTrue(result['complete'])
            self.assertEqual(before,{str(x):sha(x) for x in old.rglob('*') if x.is_file()})

    def test_component_axes_are_separate(self):
        f=compact(analyze(initial(self.c),Propagator(self.c)));f['time_au']=0.
        f['cut_q0_geo_q'][:]=20
        fig=plt.figure();record=draw(fig,f,20,1,(-.5,.6))
        self.assertEqual(len(fig.axes),6)
        self.assertEqual(fig.axes[0].get_ylim(),(-.5,.6))
        self.assertGreater(fig.axes[2].get_ylim()[1],20)
        self.assertIn('cut_q0_epsilon1',record);plt.close(fig)


if __name__=='__main__':unittest.main()
