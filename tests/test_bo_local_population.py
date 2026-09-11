from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from multi_component_exact_factorization.bo_local_population import population_frame
from multi_component_exact_factorization import render_final_visualizations as render


class LocalPopulationTests(unittest.TestCase):
    def test_local_and_proton_averaged_not_global_population(self):
        rho=np.array([[1.,3.],[3.,1.]])
        p0=np.array([[1.,.2],[0.,.8]])
        obs={'joint_density':rho[None]}
        ef={'bo_channel_density_qR':np.array([[rho*p0,rho*(1-p0)]])}
        local, conditional=population_frame(obs,ef,0)
        np.testing.assert_allclose(local[0],p0)
        np.testing.assert_allclose(conditional[0],[.25,.35])
        np.testing.assert_allclose(conditional.sum(axis=0),1)

    def test_two_channels_are_not_renormalized_and_nodes_undefined(self):
        obs={'joint_density':np.array([[[1.,0.],[1.,0.]]])}
        ef={'bo_channel_density_qR':np.array([[[[.2,0.],[.2,0.]],[[.3,0.],[.3,0.]]]])}
        local,p=population_frame(obs,ef,0)
        np.testing.assert_allclose(local.sum(axis=0)[:,0],.5)
        self.assertTrue(np.isnan(local[:,:,1]).all())
        np.testing.assert_allclose(p[:,0],[.2,.3])

    def test_only_command_movie_and_snapshots(self):
        from tests.test_tdse_report import TDSEReportTests
        with TemporaryDirectory() as directory:
            TDSEReportTests()._write_archive(directory)
            output=Path(directory)/'report'/'local'
            render.run(render.parse_args([directory,'--only','bo_local','--outdir',str(output),
                '--format','gif','--max-frames','2','--snapshot-count','2',
                '--dpi','35','--animation-dpi','30','--fps','2']))
            for name in ('bo_local_population_movie.gif','bo_local_population_snapshots.png'):
                self.assertTrue((output/name).is_file())
            self.assertEqual(len(list((output/'bo_local_population_frames').glob('*.png'))),2)


if __name__=='__main__':
    unittest.main()
