"""CPU-side backend and I/O tests; actual CUDA checks run explicitly on server."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.linalg import eigh,expm
from multi_component_exact_factorization.vsc_polariton.phase6_gpu_backend import PFBackend
from multi_component_exact_factorization.vsc_polariton.run_phase6_gpu import load_packet,save_npz,absolute_failures


def fixture():
    # Deliberately small algebraic fixture, NOT a literature/production model.
    nr,nx,nf=4,6,3;dx=.4;dr=.6
    r=(np.arange(nr)-(nr-1)/2)*dr;x=(np.arange(nx)-(nx-1)/2)*dx
    displacement,rotation=eigh(np.diag(np.sqrt(np.arange(1,nf)),1)+np.diag(np.sqrt(np.arange(1,nf)),-1))
    rng=np.random.default_rng(123);u=rng.normal(size=(nr,nx,nf))+1j*rng.normal(size=(nr,nx,nf))
    u/=np.sqrt(np.sum(abs(u)**2)*dx*dr)
    phi=np.tile(np.linalg.qr(rng.normal(size=(nx,3)))[0][None,:,:],(nr,1,1))/np.sqrt(dx)
    mu=r[:,None]-x[None,:];g=.004;omega=.03
    return dict(psi=u,R=r,x=x,dx=dx,dR=dr,mass=1836.,omega=omega,g_chi=g,
        tx=(2*np.pi*np.fft.fftfreq(nx,dx))**2/2,tr=(2*np.pi*np.fft.fftfreq(nr,dr))**2/(2*1836),
        potential=.1*mu**2,dse=g*g*mu**2/omega,mu=mu,photon=omega*(np.arange(nf)+.5),
        rotation=rotation,displacement=displacement,phi=phi,dt=.01)


class GPUAdapterTests(unittest.TestCase):
    def test_failed_hardware_gate_cannot_start_propagation(self):
        from multi_component_exact_factorization.vsc_polariton import run_phase6_gpu as driver
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json';path.write_text(json.dumps(dict(status='FAIL')))
            with patch.object(driver,'PFBackend'),patch.object(driver,'environment',return_value={}):
                with self.assertRaises(RuntimeError):
                    driver.propagate({},'hash',.25,0,Path(tmp),path,None)
            self.assertFalse((Path(tmp)/'restart.npz').exists())

    def test_hermitian_dense_and_fourth_order(self):
        data=fixture();b=PFBackend(data,.01);n=data['psi'].size
        eye=np.eye(n,dtype=complex)
        matrix=np.column_stack([b.action(eye[:,j].reshape(b.shape)).ravel() for j in range(n)])
        np.testing.assert_allclose(matrix,matrix.conj().T,atol=1e-13)
        exact=expm(-.04j*matrix)@data['psi'].ravel();errors=[]
        for dt in (.02,.01,.005):
            engine=PFBackend(data,dt);u=data['psi'].copy()
            for _ in range(round(.04/dt)):u=engine.step(u)
            errors.append(np.linalg.norm(u.ravel()-exact))
            self.assertLess(abs(np.sum(abs(u)**2)*b.volume-1),1e-12)
        self.assertGreater(errors[0]/errors[1],12)
        self.assertGreater(errors[1]/errors[2],12)

    def test_observable_energy_and_population_accounting(self):
        data=fixture();b=PFBackend(data,.01);row=b.observe(data['psi'])
        self.assertAlmostEqual(row['energy'],row['energy_parts'].sum(),places=12)
        self.assertAlmostEqual(row['BO_populations'].sum()+row['BO_projection_remainder'],row['norm'],places=13)

    def test_checkpoint_continuation_exact(self):
        data=fixture();b=PFBackend(data,.01);u=data['psi']
        direct=b.step(b.step(u.copy()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wave.npz';save_npz(path,psi=b.step(u.copy()))
            with np.load(path) as a:resumed=b.step(a['psi'])
            np.testing.assert_array_equal(direct,resumed)

    def test_transfer_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.npz'
            save_npz(p,**fixture(),metadata=json.dumps(dict(format_version=1,legacy_adapter_validation=dict(status='PASS'))))
            p.with_suffix('.json').write_text(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
            packet,_=load_packet(p);self.assertEqual(packet['psi'].shape,(4,6,3))
            p.with_suffix('.json').write_text(json.dumps(dict(sha256='wrong')))
            with self.assertRaises(ValueError):load_packet(p)

    def test_failure_limits_not_relaxed(self):
        row=dict(norm=1,energy=0,electron_edge=1e-8,edge=0,top=np.nan,continuity_error=0)
        self.assertEqual(set(absolute_failures(row,0)),{'electron_edge','Fock_tail'})
