import io
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np

from multi_component_exact_factorization_gpu.compare_born_huang_kernels import (
    run,
)


class CompareBornHuangKernelsTests(unittest.TestCase):
    def _write_archive(self, path, *, perturbation=0.0, kernel="reference"):
        times = np.array([0.0, 0.1])
        q = np.linspace(-1.0, 1.0, 5)
        R = np.linspace(-2.0, 2.0, 7)
        x = np.linspace(-3.0, 3.0, 9)
        populations = np.tile(np.array([0.7, 0.3]), (len(times), 1))
        density_q = np.ones((len(times), 2, len(q)))
        density_R = np.ones((len(times), 2, len(R)))
        electron = np.ones((len(times), len(x)))
        populations[-1, 0] += perturbation
        density_q[-1, 0, 2] += perturbation
        density_R[-1, 0, 3] += perturbation
        electron[-1, 4] += perturbation
        np.savez_compressed(
            path,
            times_fs=times,
            q=q,
            R=R,
            x=x,
            norm=np.ones(len(times))+perturbation,
            bo_populations=populations,
            bo_state_density_q=density_q,
            bo_state_density_R=density_R,
            electron_density=electron,
            bo_link_kernel=np.array(kernel),
            wall_seconds=np.array(10.0 if kernel == "reference" else 5.0),
        )

    @staticmethod
    def _args(reference, fused, tolerance=1.0e-8):
        return Namespace(
            reference=str(reference),
            fused=str(fused),
            time_tolerance_fs=1.0e-12,
            norm_tolerance=tolerance,
            population_tolerance=tolerance,
            density_tolerance=tolerance,
        )

    def test_identical_archives_pass_and_report_speedup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root/"reference.npz"
            fused = root/"fused.npz"
            self._write_archive(reference, kernel="reference")
            self._write_archive(fused, kernel="fused")
            output = io.StringIO()
            with redirect_stdout(output):
                run(self._args(reference, fused))
            report = output.getvalue()
            self.assertIn("speedup=2.000x", report)
            self.assertIn("end-to-end kernel comparison: PASS", report)

    def test_difference_above_tolerance_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root/"reference.npz"
            fused = root/"fused.npz"
            self._write_archive(reference, kernel="reference")
            self._write_archive(
                fused, perturbation=1.0e-4, kernel="fused"
            )
            with self.assertRaises(SystemExit):
                run(self._args(reference, fused, tolerance=1.0e-8))


if __name__ == "__main__":
    unittest.main()
