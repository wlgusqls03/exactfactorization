"""New backend tests never mutate historical source/results."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from multi_component_exact_factorization_cavity.backend import Hamiltonian,memory_estimate
from multi_component_exact_factorization_cavity.core import build_molecular,vacuum
from multi_component_exact_factorization_cavity.workflow import save_tree,load_tree,compact_fields


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.p=build_molecular(dict(nx=12,nR=20,nQ=16,R_left=-8,R_right=8,R0=-4))
        self.p['barrier_R']=0.
        self.Q=np.linspace(-6,6,16,endpoint=False)
        self.h=Hamiltonian(self.p,self.Q,.1,.02,block=4)
        rng=np.random.default_rng(41)
        u=rng.normal(size=(20,12,16))+1j*rng.normal(size=(20,12,16))
        self.u=u/np.sqrt(np.sum(abs(u)**2)*self.p['dx']*self.p['dR']*.75)

    def test_unitarity_and_reversibility(self):
        v=self.h.step(self.u.copy(),.01)
        np.testing.assert_allclose(np.linalg.norm(v),np.linalg.norm(self.u),rtol=3e-15)
        np.testing.assert_allclose(self.h.step(v,-.01),self.u,atol=3e-15)

    def test_backend_owns_initial_storage_for_independent_runs(self):
        original=self.u.copy()
        v=self.h.array(self.u)
        self.h.step(v,.01)
        np.testing.assert_array_equal(self.u,original)
        self.assertGreater(np.linalg.norm(v-original),1e-5)

    def test_action_hermitian_and_step_derivative(self):
        h=self.h.action(self.u)
        self.assertLess(abs(np.vdot(self.u,h).imag),1e-13)
        dt=1e-5
        fd=(self.h.step(self.u.copy(),dt)-self.h.step(self.u.copy(),-dt))/(2*dt)
        np.testing.assert_allclose(fd,-1j*h,atol=2e-9)

    def test_second_order_time_convergence(self):
        def run(dt,n):
            u=self.u.copy()
            for _ in range(n):u=self.h.step(u,dt)
            return u
        ref=run(.00125,32)
        ratio=np.linalg.norm(run(.02,2)-ref)/np.linalg.norm(run(.01,4)-ref)
        self.assertGreater(ratio,3.8)
        self.assertLess(ratio,4.2)

    def test_vacuum_number(self):
        q=np.linspace(-8,8,96,endpoint=False)
        h=Hamiltonian(self.p,q,.1,0.)
        v=np.ones((20,12,1))*vacuum(q)[None,None,:]
        v=v/np.sqrt(np.sum(abs(v)**2)*self.p['dx']*self.p['dR']*(q[1]-q[0]))
        ob=h.observe(v.astype(complex))
        self.assertLess(abs(ob['photon_number']),1e-13)

    def test_tree_roundtrip_without_pickle(self):
        tree={'a':np.arange(4),'nested':{'z':np.array([1+2j])},'text':'yes','none':None}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.npz';save_tree(path,tree); out=load_tree(path)
        np.testing.assert_array_equal(out['nested']['z'],tree['nested']['z'])
        self.assertEqual(out['text'],'yes')

    def test_compact_fields_are_explicit_and_do_not_store_T_maps(self):
        out=compact_fields({'a':np.ones((2,3)), 'T1':np.ones((2,3)),
                            'epsilon2':np.ones(2),'S_q':np.ones((2,3))})
        self.assertEqual(out['a'].dtype,np.float32)
        self.assertNotIn('T1',out)
        self.assertEqual(out['epsilon2'].dtype,np.float64)
        self.assertGreater(memory_estimate((20,12,16))['gpu_reservation_bytes'],0)


if __name__=='__main__': unittest.main()
