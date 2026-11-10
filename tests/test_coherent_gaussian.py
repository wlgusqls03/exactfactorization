"""Formula-level gates for the isolated full-complex Gaussian branch."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from multi_component_exact_factorization.coherent_gaussian.core import (
    State, overlap, gaussian, tangent_rate, evaluate, split_actual_rate)
from multi_component_exact_factorization.coherent_gaussian.models import Pen, RealPen, McefLocal
from multi_component_exact_factorization.coherent_gaussian.experiment import (
    quadrature, initial_state, Reference)


class CoherentTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(71)
        self.z=np.array([-.8+.3j,.6-.4j,1.1+.7j])
        d=self.rng.normal(size=(2,3))+1j*self.rng.normal(size=(2,3))
        d/=np.sqrt(np.einsum('ki,ij,kj',d.conj(),overlap(self.z),d).real)
        self.s=State(self.z,d)

    def test_overlap_and_derivative(self):
        q,w=quadrature(1.7,0,96)
        g,dg,_=gaussian(q,self.z,1.7)
        assert_allclose(g.conj().T@(w[:,None]*g),overlap(self.z),atol=2e-14)
        gp=gaussian(q+1e-5,self.z,1.7)[0]
        gm=gaussian(q-1e-5,self.z,1.7)[0]
        assert_allclose((gp-gm)/2e-5,dg,atol=5e-11)

    def test_analytic_vs_pair_quadrature(self):
        for basis in ('bo','diabatic'):
            for dse in (False,True):
                m=Pen(representation=basis,dse=dse)
                a,b=m.matrices(self.s)
                aq,bq=m.matrices(self.s,quadrature=True)
                assert_allclose(a,aq,atol=3e-14)
                assert_allclose(b,bq,atol=3e-14)
                assert_allclose(a,a.transpose(2,3,0,1).conj(),atol=1e-14)

    def test_tdvp_norm_energy_and_tangent(self):
        m=Pen(g=.03)
        H0,H1=m.matrices(self.s)
        rate,a=tangent_rate(self.s,H0,H1)
        self.assertLess(abs(a['norm_rate']),1e-11)
        self.assertLess(abs(a['energy_rate']),1e-11)
        eps=1e-5
        sp=State(self.z+eps*rate.z,self.s.d+eps*rate.d)
        sm=State(self.z-eps*rate.z,self.s.d-eps*rate.d)
        def energy(s):
            h,_=m.matrices(s)
            return np.einsum('ki,ikjl,lj',s.d.conj(),h,s.d).real
        self.assertLess(abs((energy(sp)-energy(sm))/(2*eps)),1e-9)

    def test_energy_constraint_is_not_post_step_renormalization(self):
        m=Pen(g=.03)
        H0,H1=m.matrices(self.s)
        r,a=tangent_rate(self.s,H0,H1,conserve=True)
        self.assertLess(abs(a['energy_rate']),1e-11)
        self.assertLess(a['constraint_correction_norm'],1e-9)

    def test_free_vacuum_and_ghost_components(self):
        m=Pen(g=0.)
        for K in (1,2,4,8):
            s,err=initial_state(m,K)
            r,a=tangent_rate(s,*m.matrices(s))
            self.assertLess(err,1e-9)
            q,w=quadrature(m.sigma,0,96)
            g=gaussian(q,s.z,m.sigma)[0]
            gd=g*((q[:,None]/m.sigma-s.z)*r.z-np.real(s.z.conj()*r.z))
            Gdot=g@r.d.T+gd@s.d.T
            self.assertLess(np.sqrt(np.sum(abs(Gdot)**2*w[:,None])),1e-9)

    def test_cbo_derivative_and_kinetic(self):
        m=Pen(g=.03,representation='cbo',nquad=48)
        q=np.linspace(-6,6,37)
        U,Up=m.basis(q)
        assert_allclose(Up,(m.basis(q+1e-4)[0]-m.basis(q-1e-4)[0])/2e-4,atol=1e-10)
        H0,H1=m.matrices(self.s)
        assert_allclose(H0,H0.transpose(2,3,0,1).conj(),atol=2e-14)
        rate,a=tangent_rate(self.s,H0,H1)
        self.assertLess(abs(a['energy_rate']),1e-10)
        # Independent dense integration of physical derivative validates all Uq
        # cross terms, not merely the potential eigenvalues.
        q,w=quadrature(m.sigma,0,128)
        G,Gq,_,_=evaluate(self.s,q,m.sigma,basis=m.basis(q))
        e=np.sum(abs(Gq)**2*w[:,None])/2+np.einsum('nk,nkl,nl,n',G.conj(),m.potential(q),G,w).real
        em=np.einsum('ki,ikjl,lj',self.s.d.conj(),H0,self.s.d).real+m.energy_shift
        assert_allclose(e,em,atol=1e-11)

    def test_C_quotient_rule_uses_actual_rate(self):
        G=self.rng.normal(size=(19,2))+1j*self.rng.normal(size=(19,2))
        F=self.rng.normal(size=G.shape)+1j*self.rng.normal(size=G.shape)
        lam,C,ld,Cd,good=split_actual_rate(G,F)
        assert_allclose(lam[:,None]*Cd+ld[:,None]*C,F,atol=1e-14)
        assert_allclose(2*np.real(np.sum(C.conj()*Cd,axis=1)),0,atol=1e-14)

    def test_real_electron_PEN_is_not_two_state(self):
        m=RealPen(nx=9,g=.001,nquad=32)
        s,_=initial_state(m,2)
        self.assertEqual(s.d.shape,(9,2))
        H0,H1=m.matrices(s);Q0,Q1=m.matrices(s,quadrature=True)
        assert_allclose(H0,Q0,atol=1e-13);assert_allclose(H1,Q1,atol=1e-13)
        ref=Reference(m,nf=12)
        assert_allclose(ref.H,m.reference(12),atol=1e-12)
        m.representation='cbo';q=np.linspace(-2,2,11)
        U,Uq=m.basis(q)
        assert_allclose(Uq,(m.basis(q+1e-4)[0]-m.basis(q-1e-4)[0])/2e-4,atol=1e-7)

    def test_mcef_local_BO_and_primitive(self):
        for basis in ('primitive','bo'):
            m=McefLocal(nx=9,representation=basis,nquad=48)
            s,err=initial_state(m,3)
            H0,H1=m.matrices(s)
            assert_allclose(H0,H0.transpose(2,3,0,1).conj(),atol=2e-12)
            _,a=tangent_rate(s,H0,H1)
            self.assertLess(abs(a['energy_rate']),1e-9)
            q=np.linspace(-5,-3,7)
            U,Up=m.basis(q)
            assert_allclose(Up,(m.basis(q+1e-4)[0]-m.basis(q-1e-4)[0])/2e-4,atol=1e-7)

    def test_extra_rate_not_energy_frozen(self):
        m=Pen()
        from multi_component_exact_factorization.coherent_gaussian.core import fock_coefficients
        F=fock_coefficients(self.s.z,40)
        T=np.zeros_like(F);T[1:]=np.sqrt(np.arange(1,40))[:,None]*F[:-1]
        source=.2*(self.s.d@F.T)  # Intentional norm source, not CTMQC.
        bc=source@F.conj();bs=np.sum(self.s.d.conj()*(source@T.conj()),axis=0)
        r,a=tangent_rate(self.s,*m.matrices(self.s),extra_rate=(bc,bs))
        self.assertAlmostEqual(a['norm_rate'],.4,places=9)
        with self.assertRaises(ValueError):
            tangent_rate(self.s,*m.matrices(self.s),extra_rate=(bc,bs),conserve=True)

    def test_gaussian_POL_initial_and_tail_guard(self):
        from multi_component_exact_factorization.coherent_gaussian.compare import GaussianPOL
        from multi_component_exact_factorization.pen_ctmqc_2026.core import Settings
        m=Pen()
        from dataclasses import replace
        e=GaussianPOL(replace(m.c,nf=8,nr=32),Settings(perspective='pol',ntraj=4,initial='bare'),4)
        y=e.initial();native=e.engine.initial();state=e.native_state(y)
        assert_allclose(state['coeff'],native['coeff'],atol=1e-12)
        assert_allclose(state['X'],native['X'])
        f=e.rhs(0,y)
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertLess(e.max_norm_rate,1e-10)
        bad=y.copy().reshape(e.n,e.width);bad[0,2:6]=8
        with self.assertRaises(ValueError):e.native_state(bad.ravel())

    def test_driver_render_and_no_overwrite(self):
        from multi_component_exact_factorization.coherent_gaussian.experiment import run
        from multi_component_exact_factorization.coherent_gaussian.render import render_fixed
        with tempfile.TemporaryDirectory() as tmp:
            m=Pen(g=0)
            result=run(m,2,tmp,end_fs=.01,every_fs=.005,max_step=.2)
            self.assertEqual(result['status'],'COMPLETE_NOT_CERTIFIED')
            self.assertLess(result['max_state_l2'],1e-10)
            render_fixed(tmp)
            self.assertTrue((Path(tmp)/'plots/conditional_summary.pdf').exists())
            with self.assertRaises(FileExistsError):run(m,2,tmp)

    def test_underresolved_CBO_is_rejected_before_propagation(self):
        from multi_component_exact_factorization.coherent_gaussian.experiment import run
        with tempfile.TemporaryDirectory() as tmp:
            r=run(RealPen(nx=17,representation='cbo',nquad=16),4,tmp,end_fs=.01)
            self.assertEqual(r['status'],'FAILED_INITIALIZATION')
            self.assertIn('quadrature',r['error'])
            self.assertFalse((Path(tmp)/'states.npz').exists())

    def test_local_quadrature_converges_for_supported_two_state_CBO(self):
        from multi_component_exact_factorization.coherent_gaussian.experiment import quadrature_check
        m=Pen(representation='cbo',nquad=32)
        s,_=initial_state(m,4)
        self.assertLess(quadrature_check(m,s),1e-8)


if __name__=='__main__':
    unittest.main()
