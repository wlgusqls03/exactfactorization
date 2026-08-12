import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_component_exact_factorization_discrete import report
from multi_component_exact_factorization_discrete_gpu import diagnose


class DiscreteReportTests(unittest.TestCase):
    def test_static_native_report(self):
        nt, nq, nR, states = 3, 5, 6, 3
        q = np.linspace(-2.0, 2.0, nq, endpoint=False)
        R = np.linspace(-2.0, 2.0, nR, endpoint=False)
        lam = np.exp(-0.5*q[:, None]**2)*np.ones((nt, nq, nR), complex)
        chi = np.exp(-0.5*(R-0.5)**2)[None, :]*np.ones((nt, nR), complex)
        joint = np.abs(lam)**2*np.abs(chi[:, None, :])**2
        link = np.clip(1.0-1.0e-3*joint/np.max(joint), 0.0, 1.0)
        fields = np.sin(q[None, :, None]+R[None, None, :])
        fields = np.repeat(fields, nt, axis=0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root/"trajectory.npz"
            payload = dict(
                times_fs=np.linspace(0.0, 0.2, nt), q=q, R=R,
                lambda_wavefunction=lam, chi=chi,
                norm=np.ones(nt), pnc_error=np.full(nt, 1e-13),
                pnc_projection_correction=np.full(nt, 2e-13),
                bo_populations=np.tile([0.9, 0.09, 0.01], (nt, 1)),
                epsilon_1=fields, epsilon_2=np.sin(R)[None]*np.ones((nt, 1)),
                a=0.1*fields, b=-0.05*fields,
                alpha=np.cos(R)[None]*np.ones((nt, 1)),
                sphi_q1_magnitude=link, sphi_R1_magnitude=link,
                sgamma_R1_magnitude=np.mean(link, axis=1),
                relative_unexplained_residual=np.full(nt, 1e-14),
                rk_product_local_defect_relative=np.array([0.0, 2e-12, 3e-12]),
                pnc_product_change_l2=np.full(nt, 1e-15),
                outer_probability_q=np.full(nt, 1e-12),
                outer_probability_R=np.full(nt, 2e-12),
                kind=np.array("discrete_born_huang_multi_component_exact_factorization"),
                propagation_completed=np.array(True),
                requested_final_time_fs=np.array(0.2),
                failure_reason=np.array(""),
            )
            np.savez(archive, **payload)
            report.run(
                archive, root/"report", no_animation=False, dpi=45,
                fps=2, max_frames=3, animation_dpi=30, fmt="gif",
            )
            self.assertTrue((root/"report/06_discrete_mcef_consistency.png").is_file())
            self.assertTrue((root/"report/07_discrete_link_geometry.png").is_file())
            self.assertTrue((root/"report/discrete_mcef_native_geometry.gif").is_file())
            self.assertTrue(diagnose.run(archive))


if __name__ == "__main__":
    unittest.main()
