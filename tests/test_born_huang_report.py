from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from multi_component_exact_factorization import born_huang_report


class BornHuangReportTests(unittest.TestCase):
    def synthetic_data(self):
        nt, ns, nq, nR = 3, 3, 7, 5
        q = np.linspace(-2.0, 2.0, nq, endpoint=False)
        R = np.linspace(3.0, 5.0, nR, endpoint=False)
        dq, dR = q[1]-q[0], R[1]-R[0]
        lam = np.ones((nt, nq, nR), complex)/np.sqrt(nq*dq)
        chi = np.ones((nt, nR), complex)/np.sqrt(nR*dR)
        norm = np.array([1.0, 1.0001, 0.9999])
        populations = norm[:, None]*np.array([
            [0.0, 1.0, 0.0], [0.01, 0.98, 0.01], [0.02, 0.95, 0.03],
        ])
        energies = np.empty((ns, nq, nR))
        for state in range(ns):
            energies[state] = state+0.01*q[:, None]+0.02*R[None, :]
        return dict(
            times_fs=np.array([0.0, 0.5, 1.0]), q=q, R=R,
            lambda_wavefunction=lam, chi=chi, norm=norm,
            bo_populations=populations, bo_energies=energies,
            args=np.array([{"electron_excitation": 1}], dtype=object),
        )

    def test_observables_are_normalized_and_population_corrected(self):
        data = self.synthetic_data()
        obs = born_huang_report.calculate_observables(data)
        dq, dR = obs["q"][1]-obs["q"][0], obs["R"][1]-obs["R"][0]
        self.assertTrue(np.allclose(
            np.sum(obs["nuclear_joint_density"], axis=(1, 2))*dq*dR, 1.0
        ))
        self.assertTrue(np.allclose(
            np.sum(obs["normalized_state_populations"], axis=1), 1.0
        ))
        self.assertEqual(obs["mean_bo_energies"].shape, (3, 3))

    def test_static_report_does_not_require_coefficients_or_basis_states(self):
        data = self.synthetic_data()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root/"multi_component_born_huang_ef_gpu.npz"
            np.savez(archive, **data)
            report = root/"report"
            born_huang_report.run(
                archive, report, no_animation=True, dpi=50
            )
            for name in (
                "01_born_huang_nuclear_motion.png",
                "02_born_huang_state_populations.png",
                "03_born_huang_energy_ladder.png",
                "04_born_huang_numerical_reliability.png",
                "report_observables.npz",
            ):
                self.assertTrue((report/name).is_file(), name)

    def test_state_ladder_animation_writes_without_electronic_grid(self):
        obs = born_huang_report.calculate_observables(self.synthetic_data())
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            born_huang_report.make_dynamics_animation(
                obs, root, fps=2, max_frames=3, dpi=35, fmt="gif"
            )
            self.assertGreater(
                (root/"born_huang_state_ladder_dynamics.gif").stat().st_size,
                0,
            )


if __name__ == "__main__":
    unittest.main()
