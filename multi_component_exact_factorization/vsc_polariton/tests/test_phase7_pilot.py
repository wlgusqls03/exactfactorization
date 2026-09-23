"""Synthetic algebra tests; not a literature-parameter convergence claim."""
import unittest
import numpy as np
from ..phase7_fock_to_q import hermite_grid,transform,backproject
from ..phase7_native import full_action
from ..phase7_nested_fields import outer_fields,nested_fields
from ..phase7_support import budget_support
from ..phase6_gpu_backend import PFBackend
from .test_phase6_completion import packet


class PilotTests(unittest.TestCase):
    def test_transform_and_derivative_headroom(self):
        nf=7;omega=.02;g=hermite_grid(nf,32,omega)
        u=np.eye(nf,dtype=complex)
        psi=transform(u,g);d=transform(u,g,1)
        np.testing.assert_allclose(backproject(psi,g),u,atol=1e-13)
        energy=np.sum((abs(d)**2+omega**2*g['q']**2*abs(psi)**2)*g['weights'],axis=1)/2
        np.testing.assert_allclose(energy,omega*(np.arange(nf)+.5),atol=1e-13)

    def test_native_action(self):
        p=packet();rng=np.random.default_rng(7)
        u=rng.normal(size=p['psi'].shape)+1j*rng.normal(size=p['psi'].shape)
        np.testing.assert_allclose(full_action(u,p),PFBackend(p,.125).action(u),atol=1e-12)

    def test_vacuum_stationary_scalars(self):
        p=packet();g=hermite_grid(p['psi'].shape[-1],32,float(p['omega']))
        outer,ur,urr,ut=outer_fields(p['psi'],p)
        fields,checks=nested_fields(p['psi'],p,g,outer,ur,urr,ut)
        expected=.5*float(p['omega'])**2*g['q']**2-float(p['omega'])/2
        np.testing.assert_allclose(fields['epsilon1_A'],np.broadcast_to(expected,fields['epsilon1_A'].shape),atol=1e-12)
        np.testing.assert_allclose(fields['epsilon1_A'],fields['epsilon1_B'],atol=1e-12)
        np.testing.assert_allclose(outer['epsilon2_A'],outer['epsilon2_B'],atol=1e-12)
        np.testing.assert_allclose(fields['epsilon2_nested'],outer['epsilon2_B'],atol=1e-12)
        self.assertLess(checks['reconstruction_L2'],1e-12)
        self.assertLess(np.max(fields['electronic_eom_norm']),1e-12)

    def test_entangled_factor_routes(self):
        p=packet();rng=np.random.default_rng(12)
        u=rng.normal(size=p['psi'].shape)+1j*rng.normal(size=p['psi'].shape)
        u/=np.sqrt(np.sum(abs(u)**2)*float(p['dx']*p['dR']))
        g=hermite_grid(u.shape[-1],40,float(p['omega']))
        outer,ur,urr,ut=outer_fields(u,p);fields,checks=nested_fields(u,p,g,outer,ur,urr,ut)
        np.testing.assert_allclose(fields['epsilon1_A'],fields['epsilon1_B'],atol=1e-11)
        np.testing.assert_allclose(outer['epsilon2_A'],outer['epsilon2_B'],atol=1e-11)
        np.testing.assert_allclose(fields['epsilon2_nested'],outer['epsilon2_A'],atol=1e-11)
        self.assertLess(np.max(fields['electronic_eom_norm']),1e-10)
        self.assertLess(checks['electronic_PNC'],1e-12)
        np.testing.assert_allclose(fields['alpha_nested'],outer['alpha'],atol=1e-12)

    def test_support_budget_and_components(self):
        rho=np.array([0.,.2,.3,0.,.5,1e-12])
        mask,info,labels=budget_support(rho,1.,1e-8)
        self.assertLessEqual(info['excluded'],1e-8)
        self.assertEqual(info['components'],2)
        self.assertFalse(mask[3]);self.assertNotEqual(labels[2],labels[4])

    def test_fock_cutoff_defect_is_not_hidden(self):
        p=packet();p['g_chi']=np.array(.013)
        p['dse']=p['g_chi']**2*p['mu']**2/p['omega']
        rng=np.random.default_rng(29)
        u=rng.normal(size=p['psi'].shape)+1j*rng.normal(size=p['psi'].shape)
        u/=np.sqrt(np.sum(abs(u)**2)*float(p['dx']*p['dR']))
        nf=u.shape[-1];g=hermite_grid(nf,40,float(p['omega']))
        outer,ur,urr,ut=outer_fields(u,p)
        f,c=nested_fields(u,p,g,outer,ur,urr,ut)
        # H in the finite Fock space omits only the LM raising action from
        # n=N-1 to n=N. No tolerance or Hamiltonian is adjusted to remove it.
        extra=hermite_grid(nf+1,40,float(p['omega']))['basis'][-1]
        defect=-float(p['g_chi'])*np.sqrt(nf)*p['mu'][:,:,None]*u[:,:,-1,None]*extra
        psi=transform(u,g)
        projected=np.sum(psi.conj()*defect,axis=1)*float(p['dx'])/f['rho_qR']
        np.testing.assert_allclose(f['epsilon1_A']-f['epsilon1_B'],-projected,atol=1e-11)

    def test_outer_gauge_stationary(self):
        from ..phase7_outer_gauge import derivative_interior
        x=np.linspace(-2,2,101)
        np.testing.assert_allclose(derivative_interior(x**3,x[1]-x[0],6),3*x[3:-3]**2,atol=1e-12)

    def test_historical_reconstruction_compatibility(self):
        from multi_component_exact_factorization.core import reconstruct_psi
        p=packet();g=hermite_grid(p['psi'].shape[-1],32,float(p['omega']))
        psi=transform(p['psi'],g);rho=np.sum(abs(psi)**2,axis=1)*float(p['dx'])
        chi=np.sqrt(np.sum(rho*g['weights'],axis=1));lam=np.sqrt(rho)/chi[:,None]
        phi=psi/np.sqrt(rho)[:,None,:]
        rebuilt=reconstruct_psi(phi.transpose(1,2,0),lam.T,chi).transpose(2,0,1)
        np.testing.assert_allclose(rebuilt,psi,atol=1e-14)
