"""Synthetic rendering/current tests; not a physical parameter scan."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from .test_real_grid_mcef_preview import fixture
from ..real_grid_mcef_fields import analyze
from ..vsc_mcef_gallery import quantities,gallery
from ..vsc_mcef_photon_barrier import force_path,photon_marginal,vsc_diagnostics
from ..real_grid_mcef_plotting import prepared


class GalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        u,cls.p,Q=fixture();cls.f=analyze(u,cls.p,Q);cls.f['time_au']=0.

    def test_currents_and_coordinate_scaling(self):
        f=self.f;v=quantities(f,self.p['mass'],self.p['omega']);q=quantities(f,self.p['mass'],self.p['omega'],'q')
        dq=f['q'][1]-f['q'][0]
        np.testing.assert_allclose(v['j_R'].sum(axis=1)*dq,f['current'],atol=1e-13)
        np.testing.assert_allclose(v['j_R_relative'].sum(axis=1)*dq,0,atol=1e-13)
        np.testing.assert_allclose(v['v_y'],np.sqrt(self.p['omega'])*q['v_y'],atol=1e-13)

    def test_photon_marginal_jacobian(self):
        rq,rQ,s=photon_marginal(self.f)
        self.assertAlmostEqual(s['norm'],1,places=12)
        self.assertAlmostEqual(s['variance_Q'],.5,places=10)
        np.testing.assert_allclose(rQ,rq/np.sqrt(self.p['omega']),atol=1e-13)

    def test_force_path_does_not_bridge_nodes(self):
        R=np.arange(-3.,4.);mask=np.ones(7,bool);mask[3]=False
        w=force_path(R,-R,mask,1)
        np.testing.assert_allclose(w[:3],.5*(R[:3]**2-R[1]**2))
        self.assertTrue(np.isnan(w[3:]).all())
        self.assertTrue(np.isnan(force_path(R,-R,mask,3)).all())

    def test_gallery_png_pdf_and_vsc_panels(self):
        f=self.f;p=self.p.copy();x=p['x'];p['tx']=(2*np.pi*np.fft.fftfreq(len(x),p['dx']))**2/2
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);r=gallery([f],out,p['mass'],p['omega'],'Synthetic fixture')
            self.assertFalse(r['phase7_pass'])
            vsc_diagnostics([f],[prepared(f,1e-8,1e-5)],p,out)
            for name in ('vsc_connections_000','vsc_transport_000','vsc_epsilon1_components_000','vsc_barrier_diagnostics','vsc_photon_marginal_and_lag'):
                for ext in ('png','pdf'):self.assertTrue((out/(name+'.'+ext)).is_file())


if __name__=='__main__':unittest.main()
