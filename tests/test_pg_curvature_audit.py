import unittest

import numpy as np

from multi_component_exact_factorization.audit_pg_curvature import curvature_terms, heavy_curvature_terms


class PGCurvatureTests(unittest.TestCase):
    def test_heavy_signed_terms_and_amplitude_scaling(self):
        R = np.linspace(-6, 6, 601)
        rho = np.exp(-R**2)
        Q, K = heavy_curvature_terms(rho, np.full_like(R, 3.), .02, 9.)
        np.testing.assert_allclose(Q[200:400], (R[200:400]**2-1)/18, atol=1e-7)
        np.testing.assert_allclose(K, -.5)
        scaled = heavy_curvature_terms(7*rho, np.full_like(R, 3.), .02, 9.)
        np.testing.assert_allclose(Q[200:400], scaled[0][200:400], atol=1e-10)

    def test_final_curvature_movies(self):
        import json
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from tests.test_tdse_report import TDSEReportTests
        from multi_component_exact_factorization.render_final_visualizations import run, parse_args, FINAL_PRODUCTS
        self.assertIn('curvature', FINAL_PRODUCTS)
        with TemporaryDirectory() as directory:
            TDSEReportTests()._write_archive(directory)
            run(parse_args([directory, '--only', 'curvature', '--format', 'gif',
                            '--max-frames', '2', '--snapshot-count', '2',
                            '--dpi', '30', '--animation-dpi', '30']))
            output = Path(directory)/'report/final_visualizations'
            for level in (1, 2):
                self.assertTrue((output/f'tdpes{level}_pg_curvature_movie.gif').is_file())
                self.assertTrue((output/f'tdpes{level}_pg_curvature_snapshots.png').is_file())
                self.assertEqual(len(list((output/f'tdpes{level}_pg_curvature_frames').glob('*.png'))), 2)
            report = json.loads((output/'pg_curvature_movies_diagnostics.json').read_text())
            self.assertEqual(len(report['records']), 2)
            self.assertEqual(len(report['records'][0]['tdpes1_clipped_fraction']), 6)
            self.assertEqual(len(report['records'][0]['tdpes2_clipped_fraction']), 4)

    def test_gaussian_ratio_and_density_scaling(self):
        q = np.linspace(-8, 8, 801)
        R = np.linspace(-6, 6, 601)
        rho = np.exp(-q[:, None]**2-R[None, :]**2)
        z = np.zeros_like(rho)
        terms = curvature_terms(rho, z, z, .02, .02, 2., 3.)
        # F=exp(-(q^2+R^2)/2): F''/F = coordinate^2-1.
        np.testing.assert_allclose(terms[0][300:500, 300], (q[300:500]**2-1)/4, atol=1e-7)
        np.testing.assert_allclose(terms[1][400, 200:400], (R[200:400]**2-1)/6, atol=1e-7)
        scaled = curvature_terms(7*rho, z, z, .02, .02, 2., 3.)
        np.testing.assert_allclose(terms[0][300:500,200:400], scaled[0][300:500,200:400], atol=1e-10)

    def test_constant_amplitude_and_momentum_sign(self):
        rho = np.ones((8, 9))
        Qq, QR, Kq, KR = curvature_terms(rho, 2*rho, -3*rho, .1, .2, 4, 9)
        np.testing.assert_array_equal(Qq, 0)
        np.testing.assert_array_equal(QR, 0)
        np.testing.assert_array_equal(Kq, -.5)
        np.testing.assert_array_equal(KR, -.5)


if __name__ == '__main__':
    unittest.main()
