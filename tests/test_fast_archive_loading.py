from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from multi_component_exact_factorization.visualize import (
    LoadedArchive,
    load_archive,
    reduced_frame,
)


class FastArchiveLoadingTests(unittest.TestCase):
    def test_materializes_members_once_and_skips_redundant_psi(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"result.npz"
            phi = np.ones((2, 3, 2, 2), dtype=np.complex64)
            np.savez_compressed(
                path,
                x=np.arange(3.0), q=np.arange(2.0), R=np.arange(2.0),
                times_fs=np.arange(2.0), phi=phi,
                lambda_wavefunction=np.ones((2, 2, 2), complex),
                chi=np.ones((2, 2), complex), epsilon_2=np.zeros((2, 2)),
                psi=np.ones_like(phi),
            )
            data = load_archive(path)
            self.assertIsInstance(data, LoadedArchive)
            self.assertIn("phi", data.files)
            self.assertNotIn("psi", data.files)

    def test_reduced_frames_are_cached(self):
        data = LoadedArchive({
            "x": np.arange(3.0), "q": np.arange(2.0), "R": np.arange(2.0),
            "phi": np.ones((1, 3, 2, 2), complex),
            "lambda_wavefunction": np.ones((1, 2, 2), complex),
            "chi": np.ones((1, 2), complex),
        })
        first = reduced_frame(data, 0)
        second = reduced_frame(data, 0)
        self.assertIs(first, second)
        self.assertEqual(len(data.reduced_frames), 1)


if __name__ == "__main__":
    unittest.main()
