import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from multi_component_exact_factorization.core import derivative, apply_electronic_hamiltonian, covariant_square
from multi_component_exact_factorization.coupled_actions import action_frame, NAMES, CoupledActionRecorder


def fixture():
    nx, nq, nr = 9, 12, 13
    x = np.arange(1,nx+1)*np.pi/(nx+1)
    q, R = np.arange(nq)*2*np.pi/nq, np.arange(nr)*2*np.pi/nr
    m = SimpleNamespace(x=x,q=q,R=R,dx=x[1]-x[0],dq=q[1]-q[0],dR=R[1]-R[0],
                        proton_mass=7.,heavy_mass=19.,potential=np.zeros((nx,nq,nr)),
                        heavy_trap_alpha=0.,heavy_trap_center=0.)
    psi = (np.sin(x)[:,None,None]+.15*np.sin(2*x)[:,None,None]*np.cos(q)[None,:,None])
    psi = psi*(1+.1*np.cos(R))[None,None,:]*np.exp(.3j*np.sin(q)[:,None]*np.cos(R)[None,:])[None]
    psi /= np.sqrt(np.sum(abs(psi)**2)*m.dx*m.dq*m.dR)
    hpsi = apply_electronic_hamiltonian(psi,m)-derivative(psi,m.dq,1,2)/(2*m.proton_mass)-derivative(psi,m.dR,2,2)/(2*m.heavy_mass)
    return m,psi,hpsi


class ActionTests(unittest.TestCase):
    def test_blocks_and_no_mutation(self):
        m,p,h = fixture()
        original=p.copy()
        def run(block):
            return action_frame(lambda ids:p[:,:,ids],lambda ids:h[:,:,ids],m,block,map_stride=1)
        a,b=run(1),run(5)
        for key in ('magnitude','delta','rms','mean_real','mean_imag'):
            np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-10)
        np.testing.assert_array_equal(p,original)
        self.assertLess(a['native_action_difference'],1e-12)
        self.assertAlmostEqual(a['norm'],1.)

    def test_direct_six_actions_and_complex_sum(self):
        m,p,h=fixture()
        rho=np.sum(abs(p)**2,axis=0)*m.dx;F=np.sqrt(rho)
        chi=np.sqrt(rho.sum(0)*m.dq);lam=F/chi
        phi=p/F
        a=np.sum(phi.conj()*(-1j*derivative(phi,m.dq,1)),axis=0).real*m.dx
        b=np.sum(phi.conj()*(-1j*derivative(phi,m.dR,2)),axis=0).real*m.dx
        alpha=np.sum(lam**2*b,axis=0)*m.dq
        terms=[]
        for axis,spacing,mass,v in [(1,m.dq,m.proton_mass,a),(2,m.dR,m.heavy_mass,b)]:
            D=-1j*derivative(phi,spacing,axis)-v*phi
            terms.extend([covariant_square(phi,v,spacing,axis,-1)/(2*mass),
                          (v-1j*derivative(F,spacing,axis-1)/F)*D/mass])
        delta=b-alpha;D=-1j*derivative(lam,m.dR,1)+delta*lam
        pterms=[covariant_square(lam,delta,m.dR,1,1)/(2*m.heavy_mass),
                (alpha-1j*derivative(chi,m.dR,0)/chi)*D/m.heavy_mass]
        result=action_frame(lambda ids:p[:,:,ids],lambda ids:h[:,:,ids],m,3,map_stride=1)
        for j,val in enumerate(terms):
            np.testing.assert_allclose(result['magnitude'][j],np.sqrt(np.sum(abs(val)**2,axis=0)*m.dx),atol=1e-12)
        for j,val in enumerate(pterms,4):
            np.testing.assert_allclose(result['magnitude'][j],abs(val/lam),atol=1e-12)
        np.testing.assert_allclose(result['magnitude'][6],np.sqrt(np.sum(abs(sum(terms))**2,axis=0)*m.dx),atol=1e-12)
        self.assertTrue(np.all(result['magnitude'][6] <= result['magnitude'][:4].sum(0)+1e-12))

    def test_recorder(self):
        m,p,h=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            r=CoupledActionRecorder(tmp,1,m,block_R=3,map_stride=2)
            r.save(0.,lambda ids:p[:,:,ids],lambda ids:h[:,:,ids])
            r.finish()
            with np.load(Path(tmp)/'coupled_action_diagnostics.npz') as d:
                self.assertEqual(d['magnitude'].shape,(1,len(NAMES),6,7))
                self.assertTrue(np.isfinite(d['rms']).all())
            self.assertFalse(list(Path(tmp).glob('*.partial.npy')))

    def test_projection_requires_explicit_consent(self):
        from multi_component_exact_factorization.postprocess_coupled_actions import run
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'multi_component_discrete_tdse_gpu.npz'
            np.savez(path,args=np.array([dict(tdse_propagator='spectral_split')],dtype=object),
                     times_fs=[0.],bo_states_count=2)
            with self.assertRaisesRegex(ValueError,'Full-grid Psi history'):
                run(SimpleNamespace(run=str(path),allow_bo_projection=False))
