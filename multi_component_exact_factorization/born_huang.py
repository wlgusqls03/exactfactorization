"""Electronic-only Born--Huang representation for nested MCEF.

The implementation follows Eqs. (71)--(86) of ``paper/MCEF_revised.pdf``.
Only the conditional electronic factor is expanded; ``Lambda`` and ``chi``
remain on their q/R grids.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np

from .core import (
    derivative,
    harmonic_density_sigma,
    independent_surface_curvatures,
    local_electronic_basis,
    normalized_gaussian,
)


@dataclass
class BornHuangBasis:
    energies: np.ndarray                 # (state,q,R)
    states: np.ndarray                   # (state,x,q,R)
    d_q: np.ndarray                      # (left,right,q,R)
    D_q: np.ndarray
    d_R: np.ndarray
    D_R: np.ndarray
    # Forward BO-overlap links <phi(g)|phi(g+s)>, s=1,2.  These define
    # projected finite differences without invoking a discrete product rule.
    link_q1: np.ndarray | None = None
    link_q2: np.ndarray | None = None
    link_R1: np.ndarray | None = None
    link_R2: np.ndarray | None = None


def _project_basis_derivative(states, spacing, coordinate_axis, order, dx):
    """Project a q/R derivative of every right BO state on every left state."""
    n_states = states.shape[0]
    shape = (n_states, n_states)+states.shape[2:]
    result = np.empty(shape, dtype=np.result_type(states.dtype, np.float64))
    # Removing the state axis makes q/R axes 1/2 in one electronic state.
    state_axis = coordinate_axis-1
    for right in range(n_states):
        changed = derivative(
            states[right], spacing, axis=state_axis, order=order
        )
        result[:, right] = np.einsum(
            "lxqr,xqr->lqr", np.conj(states), changed, optimize=True
        )*dx
    return result


def build_born_huang_basis(model, n_states):
    """Build smooth local BO states and first/second q/R NAC matrices."""
    energies, states = local_electronic_basis(model, n_states)
    return BornHuangBasis(
        energies=energies,
        states=states,
        d_q=_project_basis_derivative(states, model.dq, 2, 1, model.dx),
        D_q=_project_basis_derivative(states, model.dq, 2, 2, model.dx),
        d_R=_project_basis_derivative(states, model.dR, 3, 1, model.dx),
        D_R=_project_basis_derivative(states, model.dR, 3, 2, model.dx),
    )


def forward_overlap_links(states, coordinate_axis, offset, dx):
    """Return ``<phi(g)|phi(g+offset)>`` on a periodic coordinate grid."""
    n_states, _, nq, nR = states.shape
    output = np.empty(
        (n_states, n_states, nq, nR),
        dtype=np.result_type(states.dtype, np.float64),
    )
    _fill_forward_links(output, states, coordinate_axis, offset, dx)
    return output


def _fill_forward_links(output, states, coordinate_axis, offset, dx):
    """Fill a RAM array or open_memmap without a full shifted-state copy."""
    _, _, nq, nR = states.shape
    block_R = max(1, min(nR, 8))
    q_shift = (np.arange(nq)+int(offset)) % nq
    for start in range(0, nR, block_R):
        stop = min(start+block_R, nR)
        left = np.asarray(states[:, :, :, start:stop])
        if coordinate_axis == 2:
            right = np.take(left, q_shift, axis=2)
        elif coordinate_axis == 3:
            indices = (np.arange(start, stop)+int(offset)) % nR
            right = np.take(states, indices, axis=3)
        else:
            raise ValueError("BO link coordinate axis must be q(2) or R(3)")
        output[:, :, :, start:stop] = np.einsum(
            "axqr,bxqr->abqr", np.conj(left), right, optimize=True
        )*dx


def _backward_links(forward, coordinate_axis, offset):
    """Reverse a forward link using S(g,g-s)=S(g-s,g)^H."""
    link_axis = coordinate_axis+1
    return np.roll(
        np.swapaxes(np.conj(forward), 0, 1), int(offset), axis=link_axis
    )


def projected_link_derivatives(
    coefficients, link1, link2, spacing, coordinate_axis, vector=None,
):
    """Exactly adjoint-compatible projected periodic five-point D1/D2.

    This is ``Phi(g)^H D[Phi C](g)`` evaluated through neighbor overlaps.
    It does not use the continuum Leibniz rule, which a finite-difference
    stencil does not satisfy exactly.
    """
    backward1 = _backward_links(link1, coordinate_axis, 1)
    backward2 = _backward_links(link2, coordinate_axis, 2)
    plus1 = np.einsum(
        "abqR,bqR->aqR", link1,
        np.roll(coefficients, -1, axis=coordinate_axis), optimize=True,
    )
    plus2 = np.einsum(
        "abqR,bqR->aqR", link2,
        np.roll(coefficients, -2, axis=coordinate_axis), optimize=True,
    )
    minus1 = np.einsum(
        "abqR,bqR->aqR", backward1,
        np.roll(coefficients, 1, axis=coordinate_axis), optimize=True,
    )
    minus2 = np.einsum(
        "abqR,bqR->aqR", backward2,
        np.roll(coefficients, 2, axis=coordinate_axis), optimize=True,
    )
    if vector is not None:
        vector_axis = coordinate_axis-1
        vector = np.asarray(vector)
        minus_vector = np.roll(vector, 1, axis=vector_axis)
        plus_vector = np.roll(vector, -1, axis=vector_axis)
        plus2_vector = np.roll(vector, -2, axis=vector_axis)
        integral1 = spacing*(
            -minus_vector+13.0*vector+13.0*plus_vector-plus2_vector
        )/24.0
        phase1 = np.exp(-1j*integral1)
        phase2 = phase1*np.roll(phase1, -1, axis=vector_axis)
        plus1 = phase1[None, :, :]*plus1
        plus2 = phase2[None, :, :]*plus2
        minus1 = np.conj(np.roll(
            phase1, 1, axis=vector_axis
        ))[None, :, :]*minus1
        minus2 = np.conj(np.roll(
            phase2, 2, axis=vector_axis
        ))[None, :, :]*minus2
    first = (minus2-8.0*minus1+8.0*plus1-plus2)/(12.0*spacing)
    second = (
        -minus2+16.0*minus1-30.0*coefficients+16.0*plus1-plus2
    )/(12.0*spacing**2)
    return first, second


_BO_CACHE_VERSION = 1
_BO_LINK_VERSION = 1
_BO_CACHE_ARRAYS = ("energies", "states", "d_q", "D_q", "d_R", "D_R")
_BO_LINK_ARRAYS = ("link_q1", "link_q2", "link_R1", "link_R2")


def _hash_array(digest, values):
    """Update a digest without constructing another full-sized byte string."""
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    raw = array.view(np.uint8).reshape(-1)
    chunk = 64*1024**2
    for start in range(0, raw.size, chunk):
        digest.update(raw[start:start+chunk])


def born_huang_cache_key(model, n_states):
    """Fingerprint every quantity defining the static electronic problem."""
    digest = hashlib.sha256()
    digest.update(f"mcef-bo-cache-v{_BO_CACHE_VERSION};N={n_states}".encode())
    for values in (model.x, model.q, model.R, model.potential):
        _hash_array(digest, values)
    return digest.hexdigest()


def born_huang_system_key(model):
    """Fingerprint the BO Hamiltonian independently of the stored state count."""
    digest = hashlib.sha256()
    digest.update(f"mcef-bo-system-v{_BO_CACHE_VERSION}".encode())
    for values in (model.x, model.q, model.R, model.potential):
        _hash_array(digest, values)
    return digest.hexdigest()


def _load_cached_basis(path, expected_key):
    metadata = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
    if metadata.get("version") != _BO_CACHE_VERSION:
        raise ValueError("BO cache version mismatch")
    if metadata.get("key") != expected_key:
        raise ValueError("BO cache fingerprint mismatch")
    arrays = {
        name: np.load(path/f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in _BO_CACHE_ARRAYS
    }
    link_version_matches = (
        metadata.get("overlap_link_version") == _BO_LINK_VERSION
    )
    arrays.update({
        name: (
            np.load(path/f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if link_version_matches and (path/f"{name}.npy").exists()
            else None
        )
        for name in _BO_LINK_ARRAYS
    })
    return BornHuangBasis(**arrays)


def _write_forward_link(path, states, coordinate_axis, offset, dx):
    """Stream one large overlap-link tensor from an existing mmap basis."""
    n_states, _, nq, nR = states.shape
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp.npy"
    )
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.result_type(states.dtype, np.float64),
        shape=(n_states, n_states, nq, nR),
    )
    try:
        _fill_forward_links(
            output, states, coordinate_axis, offset, dx
        )
        output.flush()
        del output
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_cached_overlap_links(path, basis, dx):
    """Lazily augment a legacy BO cache with Hermitian overlap links."""
    metadata_path = path/"metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version_matches = (
        metadata.get("overlap_link_version") == _BO_LINK_VERSION
    )
    specifications = {
        "link_q1": (2, 1), "link_q2": (2, 2),
        "link_R1": (3, 1), "link_R2": (3, 2),
    }
    for name, (axis, offset) in specifications.items():
        target = path/f"{name}.npy"
        if not version_matches or not target.exists():
            print(
                f"BO cache overlap link 생성: {name} "
                f"(기존 eigenstate 재사용; diagonalization 없음)"
            )
            _write_forward_link(target, basis.states, axis, offset, dx)
    if not version_matches:
        metadata["overlap_link_version"] = _BO_LINK_VERSION
        metadata.setdefault("arrays", {})
        for name in _BO_LINK_ARRAYS:
            values = np.load(path/f"{name}.npy", mmap_mode="r", allow_pickle=False)
            metadata["arrays"][name] = {
                "shape": list(values.shape), "dtype": str(values.dtype),
            }
        temporary = metadata_path.with_name(
            f".{metadata_path.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True)+"\n",
            encoding="utf-8",
        )
        os.replace(temporary, metadata_path)
    return _load_cached_basis(path, metadata["key"])


def _write_cached_basis(path, key, system_key, basis, n_states):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{key[:16]}.tmp-", dir=path.parent,
    ))
    try:
        for name in _BO_CACHE_ARRAYS:
            np.save(temporary/f"{name}.npy", np.asarray(getattr(basis, name)),
                    allow_pickle=False)
        metadata = {
            "version": _BO_CACHE_VERSION,
            "key": key,
            "system_key": system_key,
            "n_states": int(n_states),
            "arrays": {
                name: {
                    "shape": list(np.asarray(getattr(basis, name)).shape),
                    "dtype": str(np.asarray(getattr(basis, name)).dtype),
                }
                for name in _BO_CACHE_ARRAYS
            },
        }
        (temporary/"metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)+"\n", encoding="utf-8"
        )
        if path.exists():
            # A concurrent process may have completed the same immutable key.
            shutil.rmtree(temporary)
        else:
            temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _truncate_basis(basis, n_states):
    """Return mmap-backed leading BO/NAC blocks without copying them."""
    return BornHuangBasis(
        energies=basis.energies[:n_states],
        states=basis.states[:n_states],
        d_q=basis.d_q[:n_states, :n_states],
        D_q=basis.D_q[:n_states, :n_states],
        d_R=basis.d_R[:n_states, :n_states],
        D_R=basis.D_R[:n_states, :n_states],
        link_q1=(None if basis.link_q1 is None else
                 basis.link_q1[:n_states, :n_states]),
        link_q2=(None if basis.link_q2 is None else
                 basis.link_q2[:n_states, :n_states]),
        link_R1=(None if basis.link_R1 is None else
                 basis.link_R1[:n_states, :n_states]),
        link_R2=(None if basis.link_R2 is None else
                 basis.link_R2[:n_states, :n_states]),
    )


def _find_cached_superset(cache_dir, system_key, n_states, dx):
    """Find the smallest compatible cache containing at least ``n_states``."""
    candidates = []
    for metadata_path in Path(cache_dir).glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored = int(metadata["n_states"])
            if (
                metadata.get("version") == _BO_CACHE_VERSION
                and metadata.get("system_key") == system_key
                and stored >= n_states
            ):
                candidates.append((stored, metadata_path.parent, metadata["key"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    stored, path, key = min(candidates, key=lambda item: item[0])
    basis = _load_cached_basis(path, key)
    basis = _ensure_cached_overlap_links(path, basis, dx)
    return _truncate_basis(basis, n_states), stored, path, key


def load_or_build_born_huang_basis(
    model, n_states, *, cache_dir=None, rebuild=False,
):
    """Load an immutable BO/NAC cache or build and atomically store it.

    The key includes the complete grids and electronic potential, so timestep,
    propagation length and regularization options reuse the same basis while
    any Hamiltonian change necessarily creates another cache entry.
    """
    started = time.perf_counter()
    if cache_dir is None:
        basis = build_born_huang_basis(model, n_states)
        basis.link_q1 = forward_overlap_links(basis.states, 2, 1, model.dx)
        basis.link_q2 = forward_overlap_links(basis.states, 2, 2, model.dx)
        basis.link_R1 = forward_overlap_links(basis.states, 3, 1, model.dx)
        basis.link_R2 = forward_overlap_links(basis.states, 3, 2, model.dx)
        return basis, {
            "hit": False, "enabled": False, "key": "", "path": "",
            "seconds": time.perf_counter()-started,
        }

    cache_root = Path(cache_dir).expanduser().resolve()
    key = born_huang_cache_key(model, n_states)
    path = cache_root/key[:24]
    if rebuild and path.exists():
        shutil.rmtree(path)
    if path.exists():
        try:
            basis = _load_cached_basis(path, key)
            basis = _ensure_cached_overlap_links(path, basis, model.dx)
            return basis, {
                "hit": True, "enabled": True, "key": key,
                "path": str(path), "stored_states": n_states,
                "seconds": time.perf_counter()-started,
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # A corrupt/incomplete entry is never silently reused.
            shutil.rmtree(path)

    system_key = born_huang_system_key(model)
    if not rebuild:
        superset = _find_cached_superset(
            cache_root, system_key, n_states, model.dx
        )
        if superset is not None:
            basis, stored_states, source_path, source_key = superset
            return basis, {
                "hit": True, "enabled": True, "key": source_key,
                "path": str(source_path), "stored_states": stored_states,
                "seconds": time.perf_counter()-started,
            }

    basis = build_born_huang_basis(model, n_states)
    _write_cached_basis(path, key, system_key, basis, n_states)
    # Reload as read-only mmap arrays.  This avoids retaining a second copy
    # while the GPU transfer and optional electron-density basis are prepared.
    basis = _load_cached_basis(path, key)
    basis = _ensure_cached_overlap_links(path, basis, model.dx)
    return basis, {
        "hit": False, "enabled": True, "key": key,
        "path": str(path), "stored_states": n_states,
        "seconds": time.perf_counter()-started,
    }


def initial_born_huang_factors(model, args, basis):
    """One-hot BO coefficient and the same harmonic Lambda/chi initialization."""
    excitation = int(args.electron_excitation)
    if excitation < 0 or excitation >= basis.energies.shape[0]:
        raise ValueError("initial BO state must be contained in --bo-states")
    curvature = independent_surface_curvatures(
        basis.energies[excitation], model, args.q0, args.R0
    )
    kq = (
        curvature["k_q"] if args.proton_force_constant == 0.0
        else args.proton_force_constant
    )
    kR = (
        curvature["k_R"] if args.heavy_force_constant == 0.0
        else args.heavy_force_constant
    )
    if kq <= 0.0 or kR <= 0.0:
        raise ValueError("Born--Huang initial surface curvature must be positive")
    args.proton_sigma = harmonic_density_sigma(model.proton_mass, kq)
    args.heavy_sigma = harmonic_density_sigma(model.heavy_mass, kR)
    args.initial_proton_force_constant = kq
    args.initial_heavy_force_constant = kR
    args.initial_gradient_q = curvature["gradient_q"]
    args.initial_gradient_R = curvature["gradient_R"]
    args.electron_initial_state = "born_huang_one_hot"

    coefficients = np.zeros(
        (basis.energies.shape[0], len(model.q), len(model.R)), complex
    )
    coefficients[excitation] = 1.0
    proton_line = normalized_gaussian(
        model.q, model.dq, args.q0, args.proton_sigma, args.proton_momentum
    )
    lam = np.repeat(proton_line[:, None], len(model.R), axis=1)
    chi = normalized_gaussian(
        model.R, model.dR, args.R0, args.heavy_sigma, args.heavy_momentum
    )
    return coefficients, lam, chi


def reconstruct_electronic_grid(coefficients, states):
    """Reconstruct Phi(x,q,R) from C_j(q,R) for output/validation only."""
    return np.einsum("jqR,jxqR->xqR", coefficients, states, optimize=True)


def basis_connection_action(connection, coefficients):
    return np.einsum("ljqR,jqR->lqR", connection, coefficients, optimize=True)


def projected_gradient(coefficients, connection, spacing, axis):
    """Projection of ``partial Phi``: ``partial C + d C``."""
    return (
        derivative(coefficients, spacing, axis=axis)
        +basis_connection_action(connection, coefficients)
    )


def projected_residual_momentum(coefficients, connection, vector, spacing, axis):
    """Eq. (79)/(81): projection of ``(-i partial-vector) Phi``."""
    return -1j*projected_gradient(
        coefficients, connection, spacing, axis
    )-vector[None, :, :]*coefficients


def projected_residual_square(
    coefficients, first_connection, second_connection, vector, spacing, axis,
):
    """Eq. (80)/(82), including the explicitly projected second NAC."""
    first = derivative(coefficients, spacing, axis=axis)
    second = derivative(coefficients, spacing, axis=axis, order=2)
    d_first = basis_connection_action(first_connection, first)
    D_value = basis_connection_action(second_connection, coefficients)
    d_value = basis_connection_action(first_connection, coefficients)
    vector_axis = axis-1
    vector_derivative = derivative(vector, spacing, axis=vector_axis)
    return (
        -second-2.0*d_first-D_value
        +1j*vector_derivative[None, :, :]*coefficients
        +2j*vector[None, :, :]*(first+d_value)
        +vector[None, :, :]**2*coefficients
    )


def coefficient_vector_potential(coefficients, connection, spacing, axis):
    """Eq. (84): first-level BO-coefficient vector potential."""
    gradient = projected_gradient(coefficients, connection, spacing, axis)
    value = -1j*np.sum(np.conj(coefficients)*gradient, axis=0)
    return value.real


def projected_plain_second(coefficients, connection, second_connection, spacing, axis):
    """Project ``partial^2 sum_j C_j phi_j`` on the retained BO basis."""
    first = derivative(coefficients, spacing, axis=axis)
    return (
        derivative(coefficients, spacing, axis=axis, order=2)
        +2.0*basis_connection_action(connection, first)
        +basis_connection_action(second_connection, coefficients)
    )
