"""Small synthetic fixtures test numerics, NOT new literature parameters."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tempfile
import json
import numpy as np
from scipy.linalg import eigh,expm
from multi_component_exact_factorization.vsc_polariton.photon_real_grid import RealGridPF,initial_grid,hermite_basis
from multi_component_exact_factorization.vsc_polariton.run_photon_real_grid import failures
from multi_component_exact_factorization.vsc_polariton import run_photon_real_grid as runner
from multi_component_exact_factorization.vsc_polariton import run_photon_grid_campaign as campaign


def fixture(nf=8):
    R=(np.arange(4)-1.5)*.5;x=R.copy();dr=dx=.5;mass=20.;omega=.5;g=.03
    mu=R[:,None]-x[None,:];phi=np.ones((4,4,1))/np.sqrt(4*dx)
    psi=np.zeros((4,4,nf),complex);psi[:,:,0]=phi[:,:,0]/np.sqrt(4*dr)
    return dict(R=R,x=x,dR=dr,dx=dx,mass=mass,omega=omega,g_chi=g,psi=psi,
        tx=(2*np.pi*np.fft.fftfreq(4,dx))**2/2,tr=(2*np.pi*np.fft.fftfreq(4,dr))**2/(2*mass),
        potential=.2*mu**2+.1*R[:,None]**2,mu=mu,dse=g*g*mu*mu/omega,phi=phi)


class RealGridTests(unittest.TestCase):
    def test_transform_and_marginals(self):
        p=fixture();Q=np.linspace(-10,10,96,endpoint=False)
        u,e=initial_grid(p,Q)
        self.assertLess(e['backprojection_L2'],1e-12)
        np.testing.assert_allclose(np.sum(abs(u)**2,axis=2)*(Q[1]-Q[0]),
                                   np.sum(abs(p['psi'])**2,axis=2),atol=1e-14)

    def test_full_dse_and_hermiticity(self):
        h=RealGridPF(fixture(),8,64,.01)
        expected=h.vm[:,:,None]+h.omega*h.Q[None,None,:]**2/2+np.sqrt(2)*h.g*h.mu[:,:,None]*h.Q+h.dse[:,:,None]
        np.testing.assert_allclose(h.V,expected,atol=2e-14)
        rng=np.random.default_rng(4)
        a=rng.normal(size=h.shape)+1j*rng.normal(size=h.shape)
        b=rng.normal(size=h.shape)+1j*rng.normal(size=h.shape)
        lhs=np.vdot(a,h.action(b));rhs=np.vdot(h.action(a),b)
        self.assertLess(abs(lhs-rhs)/max(abs(lhs),abs(rhs)),1e-13)

    def test_fock_coordinate_action_same_H(self):
        p=fixture();h=RealGridPF(p,10,96,.01)
        rng=np.random.default_rng(2);c=p['psi'].copy()
        c[...,:3]=rng.normal(size=c[...,:3].shape)+1j*rng.normal(size=c[...,:3].shape)
        B=hermite_basis(h.Q,c.shape[-1]);u=c@B
        kinetic=p['tr'][:,None,None]+p['tx'][None,:,None]
        hc=np.fft.ifftn(np.fft.fftn(c,axes=(0,1))*kinetic,axes=(0,1))
        hc+=(p['potential']+p['dse'])[:,:,None]*c
        hc+=p['omega']*(np.arange(c.shape[-1])+.5)*c
        weights=np.sqrt(np.arange(1,c.shape[-1]))
        hc[...,:-1]+=p['g_chi']*p['mu'][:,:,None]*weights*c[...,1:]
        hc[...,1:]+=p['g_chi']*p['mu'][:,:,None]*weights*c[...,:-1]
        np.testing.assert_allclose(h.action(u),hc@B,atol=5e-12)

    def test_observables_energy_photon_and_current(self):
        p=fixture();h=RealGridPF(p,10,96,.01)
        c=p['psi'].copy();c[:,:,0]=0;c[:,:,2]=p['psi'][:,:,0]
        u=c@hermite_basis(h.Q,c.shape[-1]);row=h.observe(u)
        self.assertAlmostEqual(row['norm'],1,places=12)
        self.assertAlmostEqual(row['nph'],2,places=12)
        self.assertAlmostEqual(row['energy'],np.vdot(u,h.action(u)).real*h.volume,places=11)
        self.assertAlmostEqual(row['q'],0,places=12)
        self.assertAlmostEqual(row['p'],0,places=12)
        self.assertLess(np.max(abs(row['current'])),1e-12)

    def test_fourth_order_and_norm(self):
        p=fixture();h=RealGridPF(p,7,24,.02)
        size=np.prod(h.shape);eye=np.eye(size,dtype=complex)
        H=np.column_stack([h.action(v.reshape(h.shape)).ravel() for v in eye])
        np.testing.assert_allclose(H,H.conj().T,atol=1e-12)
        u,_=initial_grid(p,h.Q);reference=expm(-.08j*H)@u.ravel()
        errors=[]
        for dt in (.02,.01):
            hh=RealGridPF(p,7,24,dt);v=u.copy()
            for _ in range(round(.08/dt)):v=hh.step(v)
            errors.append(np.linalg.norm(v.ravel()-reference))
            self.assertLess(abs(np.vdot(v,v)-np.vdot(u,u)),1e-11)
        self.assertGreater(errors[0]/errors[1],12)

    def test_stationary_eigenstate_density(self):
        p=fixture();h=RealGridPF(p,7,24,.002)
        size=np.prod(h.shape);eye=np.eye(size,dtype=complex)
        H=np.column_stack([h.action(v.reshape(h.shape)).ravel() for v in eye])
        e,c=eigh(H,subset_by_index=(0,0));u=c[:,0].reshape(h.shape)/np.sqrt(h.volume)
        self.assertLess(np.linalg.norm(h.action(u)-e[0]*u),1e-11)
        v=u.copy()
        for _ in range(10):v=h.step(v)
        self.assertLess(np.sum(abs(abs(v)**2-abs(u)**2))*h.volume,1e-9)

    def test_validation_rejects_bad_norm_and_boundary(self):
        h=RealGridPF(fixture(),10,96,.01);u,_=initial_grid(fixture(),h.Q)
        row=h.observe(u);row.update(electron_edge=0.,edge=0.,continuity_error=0.)
        self.assertFalse(failures(row,row['energy']))
        row['photon_edge']=1e-6
        self.assertIn('photon_edge',failures(row,row['energy']))
        row['norm']=.99
        self.assertIn('norm',failures(row,row['energy']))

    def test_restart_identity_and_wave_format(self):
        # CPU substitutes for GPU ONLY to exercise checkpoint orchestration.
        # Boundary limits are separately tested above; tiny fixture is not a box benchmark.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=fixture()
            args=SimpleNamespace(out=root,validation=root/'gate.json',half_Q=10.,nq=64,dt=.01,
                device=0,end_au=.08,wave_policy='endpoints',wave_every_au=0.,
                observe_every_au=.02,checkpoint_every_au=.04,max_steps=3)
            info=dict(gpu='CPU test double',cupy='none',driver=0,runtime=0)
            gate=dict(identity=runner.identity(args,'fixture'),gpu_propagation_allowed=True,environment=info)
            args.validation.write_text(json.dumps(gate))
            factory=lambda packet,half,nq,dt,*unused:RealGridPF(packet,half,nq,dt)
            with patch.object(runner,'RealGridPF',factory),patch.object(runner,'environment',return_value=info),patch.object(runner,'failures',return_value={}):
                self.assertTrue(runner.propagate(args,p,'fixture'))
                self.assertEqual(json.loads((root/'status.json').read_text())['status'],'INTERRUPTED')
                args.max_steps=None
                self.assertTrue(runner.propagate(args,p,'fixture'))
                with np.load(root/'wave_0000008.npz') as z:
                    self.assertEqual(str(z['representation']),'Psi_Q(R,x,Q)')
                    final=z['psi']
                h=RealGridPF(p,10,64,.01);u,_=initial_grid(p,h.Q)
                for _ in range(8):u=h.step(u)
                np.testing.assert_allclose(final,u,atol=1e-13)
                args.end_au=.16
                with self.assertRaises(ValueError):runner.propagate(args,p,'fixture')

    def test_campaign_comparison_stops_bad_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,*_ in campaign.SETTINGS:
                out=root/name/'full';out.mkdir(parents=True)
                (out/'status.json').write_text(json.dumps(dict(status='COMPLETE_NOT_CERTIFIED',failures={})))
                for i,t in enumerate((0.,4.,8.)):
                    np.savez(out/f'observable_{i:07d}.npz',time_au=t,product=0.,flux=0.,nph=0.,mean_R=0.)
            self.assertEqual(campaign.compare(root,8.)['status'],'PASS')
            np.savez(root/'spacing_Q512/full/observable_0000002.npz',time_au=8.,product=.1,flux=0.,nph=0.,mean_R=0.)
            self.assertEqual(campaign.compare(root,8.)['status'],'FAIL')


if __name__=='__main__':unittest.main()
