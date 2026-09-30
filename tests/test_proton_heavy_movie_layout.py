import tempfile
import unittest
from pathlib import Path
import numpy as np
from matplotlib.colors import SymLogNorm
from tests import test_proton_heavy_terms as fixtures
from multi_component_exact_factorization.proton_heavy_report import GROUPS,display_norm,render_proton_heavy
from multi_component_exact_factorization.render_final_visualizations import parse_args


class MovieLayoutTests(unittest.TestCase):
    def test_six_panels_and_small_sources(self):
        self.assertEqual(GROUPS['state'],('lambda_density','joint_density','heavy_density','b','alpha','delta'))
        self.assertEqual(GROUPS['density'],('density_dt','S_q','S_adv','S_rel','alpha','delta'))
        norm=display_norm('density_rate',False,.03)
        self.assertIsInstance(norm,SymLogNorm)
        self.assertAlmostEqual(float(norm(0)),.5)
        self.assertGreater(float(norm(.0003)),.65)

    def test_movie_only_no_still_images(self):
        q,R,dx,rho,h=fixtures.ProtonHeavyTermTests().fixture(12)
        z=np.zeros((3,*rho.shape))
        obs=dict(times_fs=np.array([0.,.1,.2]),q=q,R=R,dq=dx,dR=dx,
                 joint_density=np.array([rho]*3),heavy_density=np.array([h]*3),
                 options=dict(proton_mass=2.,heavy_mass=10.))
        ef=dict(a=z,b=z,alpha=np.zeros((3,len(R))),gauge='positive_density')
        args=parse_args(['unused','--only','proton_heavy','--ph-movies-only','--ph-groups','state','density',
                         '--format','gif','--animation-dpi','40','--max-frames','3'])
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);render_proton_heavy(obs,ef,out,args,[0])
            self.assertEqual(len(list(out.glob('*.gif'))),2)
            self.assertFalse(list(out.rglob('*.png')))
            self.assertFalse(list(out.rglob('*.pdf')))
            self.assertTrue((out/'proton_heavy_terms_diagnostics.json').exists())


if __name__=='__main__':unittest.main()
