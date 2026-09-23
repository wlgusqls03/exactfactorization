import unittest
import numpy as np
from ..phase6_control_flux import SpectralFluxBackend
from .test_phase6_completion import packet


class SpectralFluxTests(unittest.TestCase):
    def test_plane_wave_current(self):
        p=packet();h=SpectralFluxBackend(p,.125)
        k=2*np.pi/(len(p['R'])*float(p['dR']))
        u=p['psi']*np.exp(1j*k*p['R'])[:,None,None]
        expected=k/(float(p['mass'])*len(p['R'])*float(p['dR']))
        for x in (-.1,0,.3):self.assertAlmostEqual(h.point_current(u,x),expected,places=13)

    def test_superposition_off_grid(self):
        p=packet();h=SpectralFluxBackend(p,.125)
        k=2*np.pi/(len(p['R'])*float(p['dR']))
        u=p['psi']*(np.exp(1j*k*p['R'])+.3*np.exp(-2j*k*p['R']))[:,None,None]
        x=.071;f=np.exp(1j*k*x)+.3*np.exp(-2j*k*x)
        df=1j*k*np.exp(1j*k*x)-.6j*k*np.exp(-2j*k*x)
        expected=(f.conjugate()*df).imag/(float(p['mass'])*len(p['R'])*float(p['dR']))
        self.assertAlmostEqual(h.point_current(u,x),expected,places=13)

    def test_no_propagation_or_density_change(self):
        from ..phase6_gpu_backend import PFBackend
        p=packet();a=PFBackend(p,.125);b=SpectralFluxBackend(p,.125)
        np.testing.assert_array_equal(a.step(p['psi']),b.step(p['psi']))
        old=a.observe(p['psi']);new=b.observe(p['psi'])
        for key in ('rho_R','norm','energy','product','dproduct_exact'):
            np.testing.assert_array_equal(old[key],new[key])
        self.assertEqual(old['flux'],new['flux_linear'])
