import io
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np

from multi_component_exact_factorization.compare import run


class CompareLowMemoryTests(unittest.TestCase):
    def _write_archive(self, path, phase=0.0, include_psi=False):
        nx, nq, nR = 5, 6, 7
        x = np.arange(nx)*0.2
        q = np.arange(nq)*0.3
        R = np.arange(nR)*0.4
        phi = np.ones((2, nx, nq, nR), complex)/np.sqrt(nx*0.2)
        lam = np.ones((2, nq, nR), complex)/np.sqrt(nq*0.3)
        chi = np.ones((2, nR), complex)/np.sqrt(nR*0.4)
        phi[1] *= np.exp(1j*phase)
        payload = dict(
            x=x, q=q, R=R, times_fs=np.array([0.0, 0.1]),
            phi=phi, lambda_wavefunction=lam, chi=chi,
        )
        if include_psi:
            payload["psi"] = phi*lam[:, None, :, :]*chi[:, None, None, :]
        np.savez_compressed(path, **payload)

    def test_default_mmap_mode_extracts_once_and_compares(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            reference = temporary/"reference.npz"
            direct = temporary/"direct.npz"
            self._write_archive(reference, include_psi=True)
            self._write_archive(direct, phase=0.2)
            args = Namespace(
                reference=str(reference), direct=str(direct),
                time_tolerance_fs=1.0e-12, in_memory=False,
                tempdir=str(temporary), progress_every=0,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                run(args)
            self.assertIn("비교 frame 수:                 2", output.getvalue())
            self.assertIn("최소 full-Psi fidelity:        1.0000000000", output.getvalue())

    def test_in_memory_mode_uses_same_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            reference = temporary/"reference.npz"
            direct = temporary/"direct.npz"
            self._write_archive(reference)
            self._write_archive(direct)
            args = Namespace(
                reference=str(reference), direct=str(direct),
                time_tolerance_fs=1.0e-12, in_memory=True,
                tempdir=None, progress_every=1,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                run(args)
            self.assertIn("공통 frame 2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
