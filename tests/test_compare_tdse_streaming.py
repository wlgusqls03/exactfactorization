import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_component_exact_factorization_discrete_gpu.compare_tdse import (
    compare,
    compare_streaming,
)


class StreamingTDSEComparisonTests(unittest.TestCase):
    def test_streaming_matches_extracted_comparison_exactly(self):
        rng = np.random.default_rng(173)
        nt, nbo, nq, nR = 5, 3, 7, 8
        q = np.linspace(-2.0, 2.0, nq, endpoint=False)
        R = np.linspace(-1.0, 3.0, nR, endpoint=False)
        times = np.arange(nt, dtype=float)*0.25
        coefficients = (
            rng.normal(size=(nt, nbo, nq, nR))
            +1j*rng.normal(size=(nt, nbo, nq, nR))
        )
        coefficients /= np.sqrt(
            np.sum(np.abs(coefficients)**2, axis=1)
        )[:, None, :, :]
        lam = (
            rng.normal(size=(nt, nq, nR))
            +1j*rng.normal(size=(nt, nq, nR))
        )
        chi = rng.normal(size=(nt, nR))+1j*rng.normal(size=(nt, nR))
        y = coefficients*(lam*chi[:, None, :])[:, None, :, :]
        y = y.copy()
        y[2, 1, 3, 4] += 2.0e-4-3.0e-4j

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tdse = root/"tdse"
            mcef = root/"mcef"
            tdse.mkdir()
            mcef.mkdir()
            np.savez_compressed(
                tdse/"multi_component_discrete_tdse_gpu.npz",
                times_fs=times, q=q, R=R, tdse_coefficients=y,
            )
            np.savez_compressed(
                mcef/"multi_component_born_huang_ef_gpu.npz",
                times_fs=times, q=q, R=R,
                electronic_coefficients=coefficients,
                lambda_wavefunction=lam, chi=chi,
            )
            extracted_path = root/"extracted.npz"
            streaming_path = root/"streaming.npz"
            compare(
                tdse, mcef, tempdir=root, output=extracted_path,
                progress_every=0,
            )
            compare_streaming(
                tdse, mcef, output=streaming_path, progress_every=0,
            )
            with np.load(extracted_path, allow_pickle=False) as extracted:
                with np.load(streaming_path, allow_pickle=False) as streaming:
                    self.assertEqual(set(extracted.files), set(streaming.files))
                    for key in extracted.files:
                        self.assertTrue(
                            np.array_equal(extracted[key], streaming[key]), key
                        )


if __name__ == "__main__":
    unittest.main()
