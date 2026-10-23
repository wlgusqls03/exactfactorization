"""Small real-space cavity model checks; no production certification."""
import unittest

import numpy as np

from multi_component_exact_factorization_cavity.core import (
    build_molecular, photon_grid, initial_wave, vacuum, sine_kinetic, gaussian,
)
from multi_component_exact_factorization_cavity.calibration import prepare, product_matrix
from multi_component_exact_factorization_cavity.projections import (
    build_analysis, project, references, oscillator_functions, projected_vibrational_matter,
    project_cbo,
)


class CavityModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet=prepare(dict(nx=48,nR=128,nQ=64,local_wall_R=0.,local_state_method='local_wall',
                               reference_R_points=7,reference_Q_points=9))
        cls.Q=photon_grid(cls.packet['config'])

    def test_dst_kinetic_eigenvalues(self):
        matrix,values=sine_kinetic(18,.2)
        np.testing.assert_allclose(np.linalg.eigvalsh(matrix),values,atol=2e-13)

    def test_historical_gaussian_density_sigma(self):
        R=np.linspace(-10,10,1001);dx=R[1]-R[0]
        g=gaussian(R,dx,-4,.3208418789467482)
        self.assertAlmostEqual(np.sum(abs(g)**2*(R+4)**2)*dx,.3208418789467482**2,places=13)

    def test_vacuum_norm_and_number(self):
        Q=self.Q; dQ=Q[1]-Q[0]; v=vacuum(Q)
        k=2*np.pi*np.fft.fftfreq(len(Q),dQ)
        dv=np.fft.ifft(1j*k*np.fft.fft(v))
        number=.5*np.sum(Q**2*v**2+abs(dv)**2)*dQ-.5
        self.assertAlmostEqual(np.sum(v**2)*dQ,1.,places=13)
        self.assertLess(abs(number),1e-12)

    def test_initial_conditions_and_actual_projections(self):
        p=self.packet;Q=self.Q;cell=p['dx']*p['dR']*(Q[1]-Q[0])
        for campaign,state in [('pen',1),('vsc',0)]:
            u=initial_wave(p,Q,campaign)
            self.assertAlmostEqual(np.sum(abs(u)**2)*cell,1.,places=12)
            a=build_analysis(p,p['omega_'+campaign],p['g_'+campaign]['strong'],Q)
            z=project(u,p,a,Q)
            self.assertAlmostEqual(z['bo_population'][state],1.,places=12)
            self.assertAlmostEqual(z['bo_outside'],0.,places=12)
            if campaign=='vsc':
                self.assertAlmostEqual(z['vib_population'][1],1.,places=12)
                self.assertLess(abs(np.imag(p['chi_vsc'])).max(),1e-15)
            self.assertAlmostEqual(z['pol_population'].sum()+z['pol_outside'],1.,places=12)

    def test_unavailable_vsc_does_not_invent_double_well(self):
        p=prepare(dict(nx=48,nR=128,nQ=32,local_state_method=None))
        self.assertFalse(p['vsc_available'])
        self.assertFalse(p['calibration']['topology']['has_double_well_barrier'])
        self.assertEqual(p['calibration']['dividing_surface_kind'],
                         'chosen transfer plane, NOT a BO barrier')

    def test_single_well_is_user_selected_default(self):
        p=prepare(dict(nx=48,nR=128,nQ=32))
        self.assertTrue(p['vsc_available'])
        self.assertEqual(p['calibration']['vsc']['preparation']['state_method'],'single_well')
        self.assertGreater(p['calibration']['vsc']['preparation']['left_of_zero_probability_v1'],.9)

    def test_dse_uses_projected_mu_squared(self):
        e=np.array([0.,1.]);mu=np.array([[1.,.2],[.2,0.]])
        mu2=mu@mu+np.eye(2)*.7
        on=product_matrix(e,mu,mu2,1.,.3,3,True)
        off=product_matrix(e,mu,mu2,1.,.3,3,False)
        np.testing.assert_allclose(on-off,.5*.3**2*np.kron(mu2,np.eye(3)),atol=1e-15)

    def test_displaced_vsc_doublet_is_not_mislabeled_ground_state(self):
        p=self.packet
        a=build_analysis(p,p['omega_vsc'],p['g_vsc']['strong'],self.Q)
        # Large permanent-dipole displacement changes bare Fock weights. The
        # genuine single-excitation doublet is followed from g=0, not selected
        # by the two largest final bare weights (which incorrectly include 0).
        self.assertEqual(a['vib_pol_pair_indices'].tolist(),[1,2])
        e=a['vib_pol_energy']; gap=(e[2]-e[1])/p['omega_vsc']
        self.assertLess(abs(gap-.1),.002)
        self.assertLess(p['calibration']['vsc']['coupling']['measurements']['strong']
                        ['fock_doubling_gap_difference']/p['omega_vsc'],1e-3)

    def test_vsc_matter_projection_uses_native_full_nuclear_kinetic(self):
        p=self.packet
        matrix,checks=projected_vibrational_matter(p)
        basis=p['bo_phi'][:,:,0,None]*p['vib_phi'][:,None,:]
        k=2*np.pi*np.fft.fftfreq(len(p['R']),p['dR'])
        tr=np.fft.ifft(np.fft.fft(basis,axis=0)*(k**2/(2*p['mass']))[:,None,None],axis=0)
        tx=np.einsum('xy,Ryv->Rxv',p['Tx'],basis)
        action=tr+tx+p['V'][:,:,None]*basis
        direct=np.einsum('Rxv,Rxw->vw',basis.conj(),action)*p['dx']*p['dR']
        np.testing.assert_allclose(matrix,direct,atol=2e-14)
        self.assertLess(checks['hermiticity_defect'],1e-13)
        self.assertLess(checks['basis_gram_error'],1e-13)
        self.assertGreater(checks['bo_reduced_difference_norm'],1e-8)

    def test_product_matrix_accepts_projected_offdiagonal_matter(self):
        matter=np.array([[.2,.07],[.07,.9]])
        mu=np.array([[1.,.2],[.2,0.]])
        h=product_matrix(matter,mu,mu@mu,.1,0.,3)
        exact=np.kron(matter,np.eye(3))+np.kron(np.eye(2),np.diag(.1*(np.arange(3)+.5)))
        np.testing.assert_allclose(h,exact)

    def test_fock_basis_and_cbo_reference(self):
        p=self.packet;Q=self.Q
        f=oscillator_functions(Q,8)
        np.testing.assert_allclose(f.T@f*(Q[1]-Q[0]),np.eye(8),atol=1e-12)
        a=build_analysis(p,p['omega_pen'],0.,Q)
        r=references(p,Q,p['omega_pen'],0.,analysis=a)
        self.assertEqual(r['cbo'].shape,(len(p['R']),len(Q),4))
        self.assertTrue(np.isfinite(r['cbo']).all())
        # At stored sample points g=0 reduces to BO + harmonic photon potential.
        ir=np.searchsorted(p['R'],r['cbo_sample_R'])
        exact=p['bo_energy'][ir,None,:]+.5*p['omega_pen']*r['cbo_sample_Q'][None,:,None]**2
        np.testing.assert_allclose(r['cbo_sample_energy'],exact,atol=3e-13)
        self.assertLess(float(r['cbo_projection_sample_max_error']),3e-13)
        # Unsampled photon coordinates must retain the EXACT harmonic shape,
        # not the secant line produced by interpolating Q^2 directly.
        native_difference=r['cbo']-r['cbo'][:,0:1,:]
        expected_difference=np.broadcast_to(
            .5*p['omega_pen']*(Q[None,:,None]**2-Q[0]**2),native_difference.shape)
        np.testing.assert_allclose(native_difference,expected_difference,atol=3e-13)
        # Likewise restore the analytic nuclear scalar at every native R.
        smooth=p['bo_energy']-r['cbo_nuclear_scalar'][:,None]
        remainder=np.column_stack([np.interp(p['R'],p['R'][ir],smooth[ir,j])
                                   for j in range(smooth.shape[1])])
        native_expected=(remainder[:,None,:]+r['cbo_nuclear_scalar'][:,None,None]
                         +r['cbo_photon_harmonic'][None,:,None])
        np.testing.assert_allclose(r['cbo'],native_expected,atol=3e-13)

    def test_cbo_character_reduces_to_bare_bo_at_zero_coupling(self):
        p=self.packet;Q=self.Q
        a=build_analysis(p,p['omega_pen'],0.,Q)
        u=initial_wave(p,Q,'pen')
        z=project_cbo(u,p,a,Q)
        joint=np.sum(abs(u)**2,axis=1)*p['dx']
        support=joint>joint.max()*1e-12
        np.testing.assert_allclose(z['cbo_character'][support,1],1.,atol=2e-13)
        np.testing.assert_allclose(z['cbo_character'][support,0],0.,atol=2e-13)
        self.assertAlmostEqual(z['cbo_population'][1],1.,places=12)
        self.assertAlmostEqual(z['cbo_omitted'],0.,places=12)
        reused=project(u,p,a,Q,include_cbo=True)
        np.testing.assert_allclose(reused['cbo_character'],z['cbo_character'],equal_nan=True)

    def test_cbo_population_and_omission_are_not_renormalized(self):
        p=self.packet;Q=self.Q
        a=build_analysis(p,p['omega_pen'],p['g_pen']['strong'],Q)
        rng=np.random.default_rng(123)
        u=rng.normal(size=(len(p['R']),len(p['x']),len(Q))).astype(complex)
        u[0]=0.
        cell=p['dx']*p['dR']*(Q[1]-Q[0]);u/=np.sqrt(np.sum(abs(u)**2)*cell)
        z=project_cbo(u,p,a,Q)
        self.assertTrue(np.isnan(z['cbo_character'][0]).all())
        self.assertGreater(z['cbo_omitted'],.5)
        self.assertAlmostEqual(z['cbo_population'].sum()+z['cbo_omitted'],1.,places=12)
        np.testing.assert_allclose(z['cbo_character'][1:].sum(axis=2)+z['cbo_outside'][1:],1.,atol=2e-15)

    def test_invalid_coulomb_grid_rejected(self):
        with self.assertRaises(ValueError):
            build_molecular(dict(R_left=-9.5,nR=64,nx=32))


if __name__=='__main__':
    unittest.main()
