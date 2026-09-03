import tempfile
from pathlib import Path
import unittest

import numpy as np

from multi_component_exact_factorization_discrete_gpu.compare_tdse import compare
from multi_component_exact_factorization.tdse_electron import (
    electron_marginal_from_bo,
    electron_proton_density_from_bo,
)


class DiscreteTDSEComparisonTests(unittest.TestCase):
    def test_electron_marginal_matches_direct_grid_contraction(self):
        rng = np.random.default_rng(12)
        states = rng.normal(size=(3, 7, 5, 6))
        y = rng.normal(size=(3, 5, 6))+1j*rng.normal(size=(3, 5, 6))
        dq, dR = 0.4, 0.3
        actual = electron_marginal_from_bo(y, states, dq, dR, block_R=2)
        psi = np.einsum("nqR,nxqR->xqR", y, states)
        norm = np.sum(np.abs(y)**2)*dq*dR
        expected = np.sum(np.abs(psi)**2, axis=(1, 2))*dq*dR/norm
        self.assertTrue(np.allclose(actual, expected, rtol=2e-15, atol=2e-15))

    def test_electron_proton_density_retains_coherent_bo_cross_terms(self):
        rng = np.random.default_rng(29)
        states = rng.normal(size=(3, 7, 5, 6))
        y = rng.normal(size=(3, 5, 6))+1j*rng.normal(size=(3, 5, 6))
        dq, dR = 0.4, 0.3
        actual = electron_proton_density_from_bo(
            y, states, dq, dR, block_R=2,
        )
        psi = np.einsum("nqR,nxqR->xqR", y, states)
        norm = np.sum(np.abs(y)**2)*dq*dR
        expected = np.sum(np.abs(psi)**2, axis=2)*dR/norm
        self.assertTrue(np.allclose(actual, expected, rtol=2e-15, atol=2e-15))
        self.assertTrue(np.allclose(
            np.sum(actual, axis=1)*dq,
            electron_marginal_from_bo(y, states, dq, dR, block_R=2),
            rtol=2e-15, atol=2e-15,
        ))

    def test_identical_factorized_trajectory_has_unit_fidelity(self):
        rng = np.random.default_rng(71)
        states, nq, nR, frames = 3, 5, 6, 2
        q = np.linspace(-1.0, 1.0, nq, endpoint=False)
        R = np.linspace(-2.0, 2.0, nR, endpoint=False)
        times = np.array([0.0, 0.1])
        c = rng.normal(size=(frames, states, nq, nR))+1j*rng.normal(
            size=(frames, states, nq, nR)
        )
        lam = rng.normal(size=(frames, nq, nR))+1j*rng.normal(
            size=(frames, nq, nR)
        )
        chi = rng.normal(size=(frames, nR))+1j*rng.normal(size=(frames, nR))
        y = c*(lam*chi[:, None, :])[:, None, :, :]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root/"multi_component_discrete_tdse_gpu.npz"
            mcef = root/"multi_component_born_huang_ef_gpu.npz"
            output = root/"comparison.npz"
            np.savez_compressed(
                reference, times_fs=times, q=q, R=R, tdse_coefficients=y
            )
            np.savez_compressed(
                mcef, times_fs=times, q=q, R=R,
                electronic_coefficients=c,
                lambda_wavefunction=lam, chi=chi,
            )
            compare(
                reference, mcef, tempdir=root, output=output,
                progress_every=0,
            )
            with np.load(output) as result:
                self.assertLess(np.max(np.abs(result["fidelity"]-1.0)), 1e-14)
                self.assertLess(np.max(result["joint_density_l1"]), 1e-14)
                self.assertLess(
                    np.max(result["max_bo_population_difference"]), 1e-14
                )


if __name__ == "__main__":
    unittest.main()
