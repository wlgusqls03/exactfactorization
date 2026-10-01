import unittest
import tempfile
from pathlib import Path
import numpy as np
from .test_real_grid_mcef_preview import fixture
from ..real_grid_mcef_fields import analyze
from ..free_vacuum_fields import analyze_vacuum


class FreeVacuumTests(unittest.TestCase):
    def test_independent_full_grid_agreement(self):
        u,p,Q=fixture();vac=np.pi**(-.25)*np.exp(-Q**2/2)
        uv=u[:,:,len(Q)//2]/vac[len(Q)//2]
        a=analyze_vacuum(uv,p,Q);b=analyze(u,p,Q)
        mr=a['rho_R']>1e-5*a['rho_R'].max();mq=abs(Q)<3
        for key in ('alpha','force','epsilon2_A','epsilon2_B','alpha_t'):
            np.testing.assert_allclose(a[key][mr],b[key][mr],atol=2e-7)
        for key in ('epsilon1_A','epsilon1_B','a','b','rho_qR'):
            np.testing.assert_allclose(a[key][mr][:,mq],b[key][mr][:,mq],atol=2e-7)
        self.assertLess(a['reconstruction_L2'],1e-14)
        self.assertLess(a['photon_PNC_error'].max(),1e-13)

    def test_reject_coupled(self):
        u,p,Q=fixture();p['g_chi']=.01
        with self.assertRaises(ValueError):analyze_vacuum(u[:,:,0],p,Q)

    def test_three_movie_families_even_encoder_dimensions(self):
        from ..render_free_coupled_mcef import render
        from ..vsc_movie_only import compact_fields
        u,p,Q=fixture();f=analyze(u,p,Q)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pairs=[]
            for i in range(2):
                f['time_au']=32.*i;path=root/f'fields_{i}.npz'
                np.savez(path,**compact_fields(f));pairs.append((path,path))
            render(pairs,root,dict(free_omega_au=float(p['omega'])))
            for family in ('nuclear','epsilon1','connections'):
                self.assertGreater((root/f'mcef_free_coupled_{family}.mp4').stat().st_size,0)
