"""Small reproducible tests; no original MCEF/VSC file is written."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.linalg import expm
from .model import Config,potential,initial,Propagator,observables,derivative
from .factorization import analyze,diagnostics,compact
from .run import simulate


class ModelATest(unittest.TestCase):
    def test_potential_and_author_reference(self):
        c=Config();v=potential(c,[2.,4.],[0.,1.5])
        self.assertAlmostEqual(v[0,0,0,0],.16)
        self.assertAlmostEqual(v[0,1,0,1],.01*np.exp(-3*(2-3.875)**2)-.17*.01*1.5)
        np.testing.assert_allclose(v,v.swapaxes(-1,-2))
        # Frozen q=0 removes LM, unlike the literal printed Eq28.
        np.testing.assert_allclose(v[:,0],potential(c,[2.,4.],[0.],0)[:,0])

    def test_normalization_and_vacuum(self):
        c=Config(nr=128,nq=256,qmin=-16,qmax=16);p=Propagator(c);u=initial(c)
        o=observables(u,p)
        self.assertAlmostEqual(o['norm'],1,places=13);self.assertAlmostEqual(o['P_S1'],1,places=13)
        self.assertLess(abs(o['n_ph']),1e-7)

    def test_hermiticity(self):
        c=Config(nr=12,nq=16);p=Propagator(c);rng=np.random.default_rng(7)
        u=rng.normal(size=(12,16,2))+1j*rng.normal(size=(12,16,2));v=rng.normal(size=u.shape)+1j*rng.normal(size=u.shape)
        self.assertLess(abs(np.vdot(u,p.action(v))-np.vdot(p.action(u),v)),1e-11)

    def test_photon_fock_and_coherent_quadratures(self):
        c=Config(nr=64,nq=256,qmin=-20,qmax=20,sigma_q=1/np.sqrt(.17))
        p=Propagator(c);q=c.grids()[1];vac=initial(c)
        one=vac*np.sqrt(2*c.omega)*q[None,:,None]
        for wave,n in [(vac,0),(one,1)]:
            o=observables(wave,p)
            self.assertAlmostEqual(o['n_ph'],n,places=12)
            self.assertAlmostEqual(o['n_q'],(n+.5)/2,places=12)
            self.assertAlmostEqual(o['n_p'],(n+.5)/2,places=12)
        # Identical q-density, different phase gradient -> different photon number.
        momentum=.3;boost=vac*np.exp(1j*momentum*q)[None,:,None]
        o=observables(boost,p)
        np.testing.assert_allclose(abs(boost)**2,abs(vac)**2,atol=1e-15)
        self.assertAlmostEqual(o['n_ph'],momentum**2/(2*c.omega),places=12)
        self.assertAlmostEqual(o['n_q'],.25,places=12)

    def test_vsc_photon_adapter(self):
        from .photon_number import vsc_series
        with tempfile.TemporaryDirectory() as temp:
            import json
            root=Path(temp);run=root/'full';run.mkdir();w=.17
            (root/'campaign_plan.json').write_text(json.dumps(dict(omega=w,input_sha256='test')))
            (run/'status.json').write_text(json.dumps(dict(status='TEST',identity=dict(half_Q=10,nq=256,input_sha256='test'))))
            Q=np.linspace(-10,10,256,endpoint=False);rho=np.exp(-Q**2)/np.sqrt(np.pi)
            np.savez(run/'observable_000.npz',time_au=0,norm=1.,rho_Q=rho,nph=0.,energy_parts=[0,0,0,w/2,0,0])
            d,_=vsc_series(run)
            self.assertAlmostEqual(d['N'][0],0,places=13)
            self.assertAlmostEqual(d['p2'][0],w/2,places=13)

    def test_split_vs_dense_and_order(self):
        c=Config(nr=8,nq=8,dt=.1);p=Propagator(c);n=128
        eye=np.eye(n,dtype=complex);H=np.stack([p.action(eye[:,j].reshape(8,8,2)).ravel() for j in range(n)],axis=1)
        u=initial(c);exact=expm(-1j*.1*H)@u.ravel()
        coarse=p.step(u);fine=Propagator(replace(c,dt=.05));fineu=fine.step(fine.step(u))
        e=np.linalg.norm(coarse.ravel()-exact);ef=np.linalg.norm(fineu.ravel()-exact)
        self.assertGreater(e/ef,3.8);self.assertLess(abs(np.linalg.norm(coarse)-np.linalg.norm(u)),1e-12)

    def test_stationary_eigenstate_density(self):
        c=Config(nr=8,nq=8,dt=.002);p=Propagator(c);n=128;eye=np.eye(n,dtype=complex)
        H=np.stack([p.action(eye[:,j].reshape(8,8,2)).ravel() for j in range(n)],axis=1)
        e,v=np.linalg.eigh(H);u=v[:,0].reshape(8,8,2);before=u.copy()
        self.assertLess(np.linalg.norm(p.action(u)-e[0]*u),1e-12)
        for _ in range(20):u=p.step(u)
        self.assertLess(np.max(abs(abs(u)**2-abs(before)**2)),1e-9)

    def test_ef_pnc_routes_and_continuity(self):
        c=Config(nr=128,nq=256,qmin=-16,qmax=16);p=Propagator(c);u=initial(c)
        for _ in range(20):u=p.step(u)
        f=analyze(u,p);d=diagnostics(f)
        self.assertLess(d['reconstruction_L2'],1e-12)
        for v in d['supports'].values():
            for k in ['epsilon1_route','epsilon2_route','epsilon2_nested','PNC_e','PNC_q','alpha_nested']:
                self.assertLess(v[k],1e-10)
        self.assertLess(d['supports']['0.0001']['continuity'],1e-7)

    def test_decoupled_photon_connection(self):
        # A true vacuum fixture, not the paper's rounded width 2.425, which
        # has a small genuine breathing current even with g=0.
        c=Config(nr=128,nq=256,qmin=-16,qmax=16,g=0,sigma_q=1/np.sqrt(.17),dt=.005);p=Propagator(c);u=initial(c)
        for _ in range(40):u=p.step(u)
        f=analyze(u,p);mask=f['rho_qR']>1e-5*f['rho_qR'].max()
        self.assertLess(np.max(abs(f['a'][mask])),1e-6)

    def test_constant_unitary_electronic_basis_invariance(self):
        c=Config(nr=128,nq=128,qmin=-16,qmax=16);p=Propagator(c);u=initial(c)
        f=analyze(u,p);U=np.array([[1,1j],[1j,1]])/np.sqrt(2)
        class Rotated:
            pass
        rp=Rotated();rp.c=c;rp.dr=p.dr;rp.dq=p.dq
        rp.v=np.einsum('ij,...jk,lk->...il',U,p.v,U.conj())
        rp.action=lambda z:np.einsum('ij,...j->...i',U,p.action(np.einsum('ji,...j->...i',U.conj(),z)))
        g=analyze(np.einsum('ij,...j->...i',U,u),rp);ok=f['rho_qR']>1e-5*f['rho_qR'].max()
        for k in ['epsilon1','a','b','rho_qR']:np.testing.assert_allclose(f[k][ok],g[k][ok],atol=1e-10)

    def test_compact_native_cuts(self):
        c=Config(nr=64,nq=64);p=Propagator(c);f=analyze(initial(c),p);s=compact(f)
        np.testing.assert_equal(s['cut_R0_cond'],f['cond'][16].real)
        self.assertEqual(s['rho_qR'].shape,(32,32))

    def test_positive_gauge_scalar_not_potential_expectation(self):
        c=Config(nr=128,nq=256,qmin=-16,qmax=16);p=Propagator(c);f=analyze(initial(c),p)
        ir=int(np.argmin(abs(f['R']-2)));iq=int(np.argmin(abs(f['q'])))
        expected=-.5/c.sigma_q**2-.5/(c.mass*c.sigma_r**2)
        self.assertAlmostEqual(f['epsilon1'][ir,iq].real,expected,places=8)
        self.assertGreater(f['GI'][ir,iq].real-f['epsilon1'][ir,iq].real,.2)

    def test_instantaneous_alpha_time_derivative(self):
        c=Config(nr=128,nq=256,qmin=-16,qmax=16);p=Propagator(c);u=initial(c)
        for _ in range(20):u=p.step(u)
        ut=-1j*p.action(u);h=1e-4
        f=analyze(u,p);plus=analyze(u+h*ut,p);minus=analyze(u-h*ut,p)
        mask=f['rho_R']>1e-5*f['rho_R'].max()
        np.testing.assert_allclose((plus['alpha'][mask]-minus['alpha'][mask])/(2*h),f['alpha_t'][mask],atol=1e-7)

    def test_restart_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'run';c=Config(nr=32,nq=32)
            a=simulate(c,out,end=.1,every=.05,fields=False)
            b=simulate(c,out,end=.1,every=.05,fields=False,resume=True)
            self.assertEqual(a['errors'],b['errors'])
            with self.assertRaises(FileExistsError):simulate(c,out,end=.1,every=.05)
            with self.assertRaises(ValueError):simulate(replace(c,g=0),out,end=.1,every=.05,fields=False,resume=True)


if __name__=='__main__':unittest.main()
