"""Atomic state checkpoints for the discrete MCEF GPU propagator.

The checkpoint contains only the committed factor state at one completed
global RK4 step.  It deliberately contains no derived fields or RHS caches:
resuming therefore reconstructs every functional from exactly the same
``C``, ``Lambda`` and ``chi`` values as an uninterrupted calculation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import numpy as np


CHECKPOINT_KIND = "discrete_mcef_gpu_state_checkpoint"
CHECKPOINT_VERSION = 1


def _host_array(values, dtype):
    """Return a contiguous host array without changing floating-point bits."""
    if hasattr(values, "get"):
        values = values.get()
    return np.ascontiguousarray(np.asarray(values, dtype=dtype))


def write_checkpoint_atomic(
    path,
    *,
    completed_step,
    coefficients,
    lam,
    chi,
    metadata,
):
    """Write one uncompressed checkpoint and atomically replace the old one.

    The temporary file lives beside the target, so ``os.replace`` is atomic
    on the target filesystem.  A kill or full filesystem during a new write
    consequently leaves the preceding checkpoint intact.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "kind": np.array(CHECKPOINT_KIND),
        "version": np.array(CHECKPOINT_VERSION, dtype=np.int64),
        "completed_step": np.array(int(completed_step), dtype=np.int64),
        "metadata_json": np.array(json.dumps(
            metadata, sort_keys=True, separators=(",", ":"),
        )),
        "electronic_coefficients": _host_array(coefficients, np.complex128),
        "lambda_wavefunction": _host_array(lam, np.complex128),
        "chi": _host_array(chi, np.complex128),
    }
    try:
        with temporary.open("wb") as stream:
            # Uncompressed NPZ is intentional: checkpoint latency is much
            # lower, while the file is bounded to a single factor state.
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_checkpoint(path, *, expected_metadata=None):
    """Load and strictly validate a discrete-MCEF state checkpoint."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        kind = str(np.asarray(archive["kind"]).item())
        version = int(np.asarray(archive["version"]).item())
        if kind != CHECKPOINT_KIND:
            raise ValueError(
                f"checkpoint kind mismatch: {kind!r} != {CHECKPOINT_KIND!r}"
            )
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"checkpoint version mismatch: {version} != {CHECKPOINT_VERSION}"
            )
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        completed_step = int(np.asarray(archive["completed_step"]).item())
        coefficients = np.ascontiguousarray(
            np.asarray(archive["electronic_coefficients"], dtype=np.complex128)
        )
        lam = np.ascontiguousarray(
            np.asarray(archive["lambda_wavefunction"], dtype=np.complex128)
        )
        chi = np.ascontiguousarray(
            np.asarray(archive["chi"], dtype=np.complex128)
        )

    if completed_step < 0:
        raise ValueError(f"checkpoint completed_step must be nonnegative: {completed_step}")
    if expected_metadata is not None and metadata != expected_metadata:
        keys = sorted(set(metadata) | set(expected_metadata))
        differences = [
            key for key in keys if metadata.get(key) != expected_metadata.get(key)
        ]
        detail = ", ".join(differences[:8])
        if len(differences) > 8:
            detail += ", ..."
        raise ValueError(
            "checkpoint is incompatible with the requested calculation; "
            f"different metadata: {detail or 'unknown'}"
        )
    return {
        "path": path,
        "completed_step": completed_step,
        "metadata": metadata,
        "electronic_coefficients": coefficients,
        "lambda_wavefunction": lam,
        "chi": chi,
    }


def validate_state_shapes(checkpoint, *, coefficients_shape, lam_shape, chi_shape):
    """Reject a corrupt or incompatible state before allocating it on CUDA."""
    expected = {
        "electronic_coefficients": tuple(coefficients_shape),
        "lambda_wavefunction": tuple(lam_shape),
        "chi": tuple(chi_shape),
    }
    for name, shape in expected.items():
        actual = checkpoint[name].shape
        if actual != shape:
            raise ValueError(
                f"checkpoint {name} shape mismatch: {actual} != {shape}"
            )
