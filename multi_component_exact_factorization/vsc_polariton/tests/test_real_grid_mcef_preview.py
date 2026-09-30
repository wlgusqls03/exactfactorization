"""Synthetic numerical fixtures only, not additional physical model settings."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from multi_component_exact_factorization.vsc_polariton.real_grid_mcef_fields import analyze,action,derivative,diagnostics


def fixture(boost=.3):
    R=np.linspace(-8,8,64,endpoint=False);x=R.copy();Q=R.copy()
    M=2.;w=.7;dr=dx=.25
    norm=lambda z,h:z/np.sqrt(np.sum(abs(z)**2)*h)
    nuclear=norm(np.exp(-R**2/2+1j*boost*R),dr)
    el=norm(np.exp(-x**2/2),dx)
    photon=norm(np.exp(-Q**2/2),Q[1]-Q[0])
    u=nuclear[:,None,None]*el[None,:,None]*photon[None,None,:]
    p=dict(R=R,x=x,mass=M,omega=w,g_chi=0.,dR=dr,dx=dx,
           potential=.5*R[:,None]**2+.5*x[None,:]**2,
           phi=np.broadcast_to(el[None,:,None],(len(R),len(x),1)))
    return u,p,Q


class RealGridMCEFTests(unittest.TestCase):
    def test_fft_derivative(self):
        q=np.arange(32)*2*np.pi/32;u=np.exp(3j*q)
        np.testing.assert_allclose(derivative(u,q[1],0),3j*u,atol=4e-14)

    def test_reconstruction_routes_and_physical_force(self):
        u,p,Q=fixture();f=analyze(u,p,Q);d=diagnostics(f)
        self.assertAlmostEqual(d['norm'],1,places=12)
        self.assertLess(d['reconstruction_L2'],1e-13)
        mask=f['rho_R']>1e-5*f['rho_R'].max()
        np.testing.assert_allclose(f['alpha'][mask],.3,atol=1e-8)
        np.testing.assert_allclose(f['force'][mask],-p['R'][mask],atol=2e-7)
        self.assertLess(max(d['budgets']['1e-08']['weighted_errors'].values()),1e-8)
        # Positive-gauge scalar slope alone is not the bare force.
        self.assertGreater(np.max(abs(-f['epsilon2_R'][mask]+p['R'][mask])),.1)
        np.testing.assert_allclose(f['conditional_q'][mask],0,atol=1e-12)

    def test_H_hermiticity(self):
        u,p,Q=fixture();q=Q/np.sqrt(p['omega']);p['g_chi']=.02
        v=u*np.exp(.2j*p['R'][:,None,None]*q[None,None,:])
        a=np.vdot(u,action(v,p,q));b=np.vdot(action(u,p,q),v)
        self.assertLess(abs(a-b),1e-10)

    def test_alpha_time_derivative_independent_probe(self):
        u,p,Q=fixture();q=Q/np.sqrt(p['omega']);f=analyze(u,p,Q)
        # Two independently evaluated infinitesimal full-wave Taylor probes.
        dt=1e-5;ut=-1j*action(u*p['omega']**.25,p,q)/p['omega']**.25
        plus=analyze(u+dt*ut,p,Q);minus=analyze(u-dt*ut,p,Q)
        mask=f['rho_R']>1e-4*f['rho_R'].max()
        np.testing.assert_allclose((plus['alpha'][mask]-minus['alpha'][mask])/(2*dt),f['alpha_t'][mask],atol=1e-6)

    def test_real_coordinate_quadrature_normalization(self):
        u,p,Q=fixture();f=analyze(u,p,Q)
        dq=f['q'][1]-f['q'][0]
        np.testing.assert_allclose(f['rho_qR'].sum(axis=1)*dq,f['rho_R'],atol=1e-13)
        mask=f['rho_R']>1e-8
        np.testing.assert_allclose(f['lambda_density'][mask].sum(axis=1)*dq,1,atol=1e-12)

    def test_snapshot_mathtext_and_files(self):
        from multi_component_exact_factorization.vsc_polariton.real_grid_mcef_plotting import snapshots
        u,p,Q=fixture();f=analyze(u,p,Q);f['time_au']=0.
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);snapshots([f],out,'Synthetic numerical fixture')
            for name in ('figure2_outer_000','figure3_joint_000'):
                for ext in ('png','pdf'):self.assertTrue((out/(name+'.'+ext)).is_file())


if __name__=='__main__':unittest.main()
