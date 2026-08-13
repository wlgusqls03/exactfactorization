import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_component_exact_factorization_discrete_gpu.diagnose_factor_norms import (
    diagnose,
)


class FactorNormDiagnosticTests(unittest.TestCase):
    def test_post_retraction_support_audit_without_raw_history(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            c = np.ones((2, 2, 3, 2), dtype=np.complex128)
            c[:, 0] *= 3.0
            lam = np.ones((2, 3, 2), dtype=np.complex128)
            chi = np.ones((2, 2), dtype=np.complex128)
            path = directory/"multi_component_born_huang_ef_gpu.npz"
            np.savez_compressed(
                path, times_fs=np.array([0.0, 0.1]),
                q=np.arange(3.0), R=np.arange(2.0),
                electronic_coefficients=c,
                lambda_wavefunction=lam, chi=chi,
                deep_tail_zero_threshold=np.array(1.0e-2),
            )
            output = diagnose(path, tempdir=directory, progress_every=0)
            with np.load(output) as audit:
                self.assertEqual(audit["max_support_c_norm"].shape, (2,))
                self.assertGreater(audit["max_support_c_norm"][0], 1.0)
                self.assertNotIn("max_pre_pnc_c_norm", audit.files)


if __name__ == "__main__":
    unittest.main()
