import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_component_exact_factorization_discrete_gpu.checkpoint import (
    load_checkpoint,
    validate_state_shapes,
    write_checkpoint_atomic,
)


class DiscreteCheckpointTests(unittest.TestCase):
    def _state(self, seed):
        rng = np.random.default_rng(seed)
        coefficients = rng.normal(size=(3, 5, 7))
        coefficients = coefficients+1j*rng.normal(size=coefficients.shape)
        lam = rng.normal(size=(5, 7))+1j*rng.normal(size=(5, 7))
        chi = rng.normal(size=7)+1j*rng.normal(size=7)
        return (
            coefficients.astype(np.complex128),
            lam.astype(np.complex128),
            chi.astype(np.complex128),
        )

    def test_round_trip_is_bitwise_exact_and_atomic_replacement_is_latest(self):
        metadata = {"dt_au": 0.025, "cache_key": "abc", "nq": 5}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"checkpoint.npz"
            first = self._state(3)
            write_checkpoint_atomic(
                path, completed_step=20, coefficients=first[0],
                lam=first[1], chi=first[2], metadata=metadata,
            )
            loaded = load_checkpoint(path, expected_metadata=metadata)
            self.assertEqual(loaded["completed_step"], 20)
            for name, expected in zip(
                ("electronic_coefficients", "lambda_wavefunction", "chi"),
                first,
            ):
                self.assertTrue(np.array_equal(loaded[name], expected))

            second = self._state(4)
            write_checkpoint_atomic(
                path, completed_step=40, coefficients=second[0],
                lam=second[1], chi=second[2], metadata=metadata,
            )
            loaded = load_checkpoint(path, expected_metadata=metadata)
            self.assertEqual(loaded["completed_step"], 40)
            self.assertTrue(np.array_equal(
                loaded["electronic_coefficients"], second[0]
            ))
            self.assertEqual(list(Path(tmp).glob(".*.tmp")), [])

    def test_metadata_and_shape_mismatch_are_rejected(self):
        metadata = {"dt_au": 0.025, "cache_key": "abc"}
        state = self._state(8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"checkpoint.npz"
            write_checkpoint_atomic(
                path, completed_step=12, coefficients=state[0],
                lam=state[1], chi=state[2], metadata=metadata,
            )
            with self.assertRaisesRegex(ValueError, "different metadata"):
                load_checkpoint(
                    path,
                    expected_metadata={"dt_au": 0.0125, "cache_key": "abc"},
                )
            loaded = load_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                validate_state_shapes(
                    loaded, coefficients_shape=(3, 6, 7),
                    lam_shape=(5, 7), chi_shape=(7,),
                )

    def test_continuation_from_round_trip_matches_uninterrupted_bits(self):
        metadata = {"dt_au": 0.025, "cache_key": "same-algebra"}

        def step(state):
            c, lam, chi = state
            return (
                c+(0.001-0.002j)*np.roll(c, 1, axis=1),
                lam+(0.003+0.001j)*np.roll(lam, -1, axis=0),
                chi+(0.002-0.001j)*np.roll(chi, 1),
            )

        initial = self._state(13)
        uninterrupted = initial
        for _ in range(8):
            uninterrupted = step(uninterrupted)

        split = initial
        for _ in range(4):
            split = step(split)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"checkpoint.npz"
            write_checkpoint_atomic(
                path, completed_step=4, coefficients=split[0],
                lam=split[1], chi=split[2], metadata=metadata,
            )
            loaded = load_checkpoint(path, expected_metadata=metadata)
            resumed = (
                loaded["electronic_coefficients"],
                loaded["lambda_wavefunction"],
                loaded["chi"],
            )
            for _ in range(4):
                resumed = step(resumed)
        for actual, expected in zip(resumed, uninterrupted):
            self.assertTrue(np.array_equal(actual, expected))

if __name__ == "__main__":
    unittest.main()
