import unittest
import numpy as np
from ..outer_zero_connection import link_jet,integrate_anchor,fixed_outer_fields,validate_wave
from ..qhj_saved_wave import outer_qhj
from ..real_grid_mcef_fields import action
from .test_real_grid_mcef_preview import fixture


class OuterGaugeTests(unittest.TestCase):
    def test_large_phase_no_alias(self):
        q=np.arange(0,4.01,.1);S=np.exp(1j*np.diff(q))
        g,k=link_jet(S,1j*np.diff(q)*S)
        np.testing.assert_allclose(g,np.exp(-1j*q),atol=1e-14)
        np.testing.assert_allclose(k,-q,atol=1e-14)

    def test_small_overlap_rejected(self):
        with self.assertRaises(ValueError):link_jet(np.array([0j]),np.array([1j]))

    def test_no_node_bridge(self):
        R=np.linspace(-5,5,101);rho=np.ones(101);rho[50]=0
        work,_,_,support=integrate_anchor(R,np.ones(101),rho,20)
        self.assertFalse(support[80]);self.assertTrue(np.isnan(work[80]))
        self.assertAlmostEqual(work[20],0)

    def test_fixed_anchor_derivative(self):
        R=np.linspace(-4,4,161);rho=np.exp(-R**2);anchor=80
        g,_=fixed_outer_fields(R,rho,2*R,3*R,np.zeros(161),np.zeros(161),anchor)
        s=g['support'];np.testing.assert_allclose(g['eta'][s],-R[s]**2,atol=1e-13)
        np.testing.assert_allclose(g['kappa'][s],-1.5*R[s]**2,atol=1e-13)

    def test_full_unitary_covariance(self):
        rng=np.random.default_rng(9);n=8
        h=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));h=h+h.conj().T
        u=rng.normal(size=n)+1j*rng.normal(size=n);g=np.exp(1j*rng.normal(size=n));k=rng.normal(size=n)
        up=g.conj()*u;utp=g.conj()*(-1j*h@u)-1j*k*up
        hp=g.conj()[:,None]*h*g[None,:]+np.diag(k)
        np.testing.assert_allclose(1j*utp,hp@up,atol=1e-13)

    def test_complex_g_time_jet_with_fixed_anchor(self):
        R=np.linspace(-4,4,161);rho=np.exp(-R**2);t=3.;h=1e-5
        def at(time):
            return fixed_outer_fields(R,rho,time*time*R,2*time*R,np.zeros(len(R)),np.zeros(len(R)),80)[0]
        f=at(t);gp=at(t+h);gm=at(t-h);s=f['support']
        np.testing.assert_allclose((gp['g'][s]-gm['g'][s])/(2*h),1j*f['kappa'][s]*f['g'][s],atol=2e-6)

    def test_wave_reconstruction_and_force(self):
        psi,p,Q=fixture(boost=.3);q=Q/np.sqrt(p['omega']);u=psi*p['omega']**.25
        h=action(u,p,q);vol=p['dx']*(q[1]-q[0])
        outer=outer_qhj(u,h,p['dR'],vol,p['mass']);R=p['R'];anchor=np.argmin(abs(R))
        g,_=fixed_outer_fields(R,outer['rho_R'],outer['alpha'],outer['alpha_t'],outer['epsilon2'],outer['EF_force'],anchor)
        report=validate_wave(u,-1j*h,R,vol,p['mass'],outer,g)
        self.assertLess(report['reconstruction'],1e-12)
        self.assertLess(report['rate_reconstruction'],1e-12)
        self.assertLess(report['fd6']['force_absolute'],1e-5)


if __name__=='__main__':unittest.main()
