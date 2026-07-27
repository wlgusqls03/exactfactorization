"""Shared numerical operations for direct and reference EF calculations.

Shape notation used throughout this package
-------------------------------------------
``nr``
    Number of electronic-coordinate grid points, ``r``.
``nR``
    Number of nuclear-coordinate grid points, ``R``.
``ns``
    Number of BO electronic basis states retained.
``nt``
    Number of saved time frames (not the number of integration steps).

The real-space convention is always ``(nr, nR)``: the electronic coordinate
is the first axis and the nuclear coordinate is the second. BO-basis fields
use ``(ns, nR)``. Saved trajectories add time as the leading axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from shin_metiu_1d import compute_bo_states, shin_metiu_potential


@dataclass(frozen=True)
class Model:
    r: np.ndarray              # (nr,) electronic coordinate grid
    R: np.ndarray              # (nR,) nuclear coordinate grid
    dr: float
    dR: float
    potential: np.ndarray      # (nr,nR) V(r,R)
    bo_energies: np.ndarray    # (ns,nR) E_s(R)
    bo_states: np.ndarray      # (ns,nR,nr) phi_s(r;R)
    mass: float


def build_model(args) -> Model:
    """Construct grids, the Shin--Metiu potential, and a finite BO basis.

    Returns a :class:`Model` containing ``V(r,R)`` with shape ``(nr,nR)``,
    BO energies with shape ``(ns,nR)``, and BO wavefunctions with shape
    ``(ns,nR,nr)``. The unusual order of the BO-state array is inherited from
    :func:`shin_metiu_1d.compute_bo_states` and makes a fixed-``R`` electronic
    vector equal to ``bo_states[s, J, :]``.
    """
    grid_model = build_grid_model(args)
    energies, states = compute_bo_states(
        grid_model.r, grid_model.R, grid_model.potential, args.n_states
    )
    return Model(
        grid_model.r, grid_model.R, grid_model.dr, grid_model.dR,
        grid_model.potential, energies, states, grid_model.mass,
    )


def build_grid_model(args) -> Model:
    """Construct only the coordinate-space model, without a BO calculation.

    This is the model builder for the pure real-space EF implementation.
    ``bo_energies`` and ``bo_states`` are deliberately empty arrays, making it
    difficult to accidentally introduce a hidden BO projection into that
    workflow.

    Returns
    -------
    Model
        ``r(nr)``, ``R(nR)``, and ``potential(nr,nR)`` are populated;
        ``bo_energies`` has shape ``(0,nR)`` and ``bo_states`` has shape
        ``(0,nR,nr)``.
    """
    r = np.linspace(args.r_min, args.r_max, args.nr, endpoint=False)
    if args.boundary == "periodic":
        R = np.linspace(args.R_min, args.R_max, args.nR, endpoint=False)
    else:
        R = np.linspace(args.R_min, args.R_max, args.nR, endpoint=True)
    dr = float(r[1] - r[0])
    dR = float(R[1] - R[0])
    potential_args = SimpleNamespace(
        L=args.L, Rf=args.Rf, Rl=args.Rl, Rr=args.Rr
    )
    potential = shin_metiu_potential(r, R, potential_args)
    energies = np.empty((0, len(R)))
    states = np.empty((0, len(R), len(r)))
    return Model(r, R, dr, dR, potential, energies, states, args.mass)


def gaussian_nuclear_state(
    R: np.ndarray, dR: float, center: float, sigma: float, momentum: float
) -> np.ndarray:
    """Return a normalized complex Gaussian ``chi(R)`` with shape ``(nR,)``."""
    chi = np.exp(
        -0.5 * ((R - center) / sigma) ** 2
        + 1j * momentum * (R - center)
    )
    return normalize_initial_chi(chi.astype(complex), dR)


def gaussian_conditional_electronic_state(
    model: Model,
    center: float,
    sigma: float,
    momentum: float = 0.0,
    follow_nucleus: bool = False,
) -> np.ndarray:
    """Construct a BO-free normalized conditional electronic Gaussian.

    Parameters
    ----------
    center
        Fixed electronic center, or an offset from ``R`` when
        ``follow_nucleus=True``.
    follow_nucleus
        If true, the center at nuclear grid point ``R[J]`` is ``R[J]+center``.

    Returns
    -------
    np.ndarray
        ``Phi_R(r,0)`` with shape ``(nr,nR)`` and
        ``sum_r |Phi[:,J]|^2 dr = 1`` for every ``J``.
    """
    centers = model.R + center if follow_nucleus else np.full_like(model.R, center)
    displacement = model.r[:, None] - centers[None, :]
    phi = np.exp(
        -0.5 * (displacement / sigma) ** 2 + 1j * momentum * displacement
    )
    local_norm = np.sqrt(np.sum(np.abs(phi) ** 2, axis=0) * model.dr)
    return phi.astype(complex) / local_norm[None, :]


def derivative(values: np.ndarray, spacing: float, axis: int = -1,
               boundary: str = "dirichlet") -> np.ndarray:
    """Differentiate one array axis without changing its shape.

    ``boundary='periodic'`` uses an FFT spectral derivative. Any other value
    uses a fourth-order, five-point finite-difference stencil in the interior
    and one-sided fourth-order stencils at the two ends.
    """
    if boundary == "periodic":
        n = values.shape[axis]
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=spacing)
        # The Nyquist mode of an even real grid has no unique signed first
        # derivative. Setting it to zero preserves a real derivative for real
        # input and avoids a spurious imaginary Berry connection.
        if n % 2 == 0:
            k[n // 2] = 0.0
        shape = [1] * values.ndim
        shape[axis] = n
        return np.fft.ifft(1j * k.reshape(shape) * np.fft.fft(values, axis=axis), axis=axis)

    x = np.moveaxis(np.asarray(values), axis, -1)
    n = x.shape[-1]
    if n < 5:
        return np.moveaxis(np.gradient(x, spacing, axis=-1, edge_order=2), -1, axis)
    out = np.empty_like(x, dtype=np.result_type(x, np.float64))
    out[..., 2:-2] = (
        x[..., :-4] - 8.0 * x[..., 1:-3]
        + 8.0 * x[..., 3:-1] - x[..., 4:]
    ) / (12.0 * spacing)
    out[..., 0] = (
        -25.0 * x[..., 0] + 48.0 * x[..., 1] - 36.0 * x[..., 2]
        + 16.0 * x[..., 3] - 3.0 * x[..., 4]
    ) / (12.0 * spacing)
    out[..., 1] = (
        -3.0 * x[..., 0] - 10.0 * x[..., 1] + 18.0 * x[..., 2]
        - 6.0 * x[..., 3] + x[..., 4]
    ) / (12.0 * spacing)
    out[..., -2] = -(
        -3.0 * x[..., -1] - 10.0 * x[..., -2] + 18.0 * x[..., -3]
        - 6.0 * x[..., -4] + x[..., -5]
    ) / (12.0 * spacing)
    out[..., -1] = -(
        -25.0 * x[..., -1] + 48.0 * x[..., -2] - 36.0 * x[..., -3]
        + 16.0 * x[..., -4] - 3.0 * x[..., -5]
    ) / (12.0 * spacing)
    return np.moveaxis(out, -1, axis)


def coefficients_to_phi(coefficients: np.ndarray, model: Model) -> np.ndarray:
    """Reconstruct ``Phi(r,R)`` with shape ``(nr,nR)`` from ``C_s(R)``.

    Parameters
    ----------
    coefficients
        Complex BO coefficients with shape ``(ns,nR)``.
    """
    return np.einsum("sR,sRr->rR", coefficients, model.bo_states, optimize=True)


def project_electronic(field: np.ndarray, model: Model) -> np.ndarray:
    """Project ``field(nr,nR)`` and return BO coefficients ``(ns,nR)``."""
    return np.einsum(
        "sRr,Rr->sR", model.bo_states.conj(), field.T, optimize=True
    ) * model.dr


def density_mask(chi: np.ndarray, relative_floor: float) -> tuple[np.ndarray, float]:
    """Return a smooth ``(nR,)`` occupation mask and its density floor.

    Exact factorization is physically undefined where ``|chi|^2`` vanishes.
    The mask tends to one in occupied regions and smoothly tends to zero in
    tails, preventing division by numerical noise.
    """
    density = np.abs(chi) ** 2
    floor = max(float(relative_floor) * float(density.max()), np.finfo(float).tiny)
    return density / (density + floor), floor


def coupling_from_phi(
    phi: np.ndarray,
    chi: np.ndarray,
    model: Model,
    relative_floor: float,
    boundary: str,
    norm_correction: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the electron--nuclear coupling terms on one time slice.

    Parameters
    ----------
    phi
        Conditional electronic field ``Phi_R(r)``; shape ``(nr,nR)``.
    chi
        Nuclear wavefunction; shape ``(nR,)``.

    Returns
    -------
    A
        Berry/vector potential ``A(R)``; real array ``(nR,)``.
    u_phi
        Action ``U_en[Phi,chi] Phi``; complex array ``(nr,nR)``. ``U_en`` is
        a differential operator, so its action is the useful stored object.
    mask
        Low-density regularization mask; real array ``(nR,)``.
    ratio
        Regularized logarithmic derivative ``(d_R chi)/chi``; ``(nR,)``.
    """
    # d_R Phi couples conditional electronic states at neighboring R points.
    dphi = derivative(phi, model.dR, axis=1, boundary=boundary)
    mask, floor = density_mask(chi, relative_floor)
    # chi* d_R chi / (|chi|^2 + floor) is a stable form of (d_R chi)/chi.
    ratio = derivative(chi, model.dR, boundary=boundary) * chi.conj()
    ratio = ratio / (np.abs(chi) ** 2 + floor)

    # A(R) = <Phi_R|-i d_R|Phi_R>_r. Only the real part is physical when the
    # partial-normalization condition is satisfied exactly.
    A = np.sum(phi.conj() * (-1j * dphi), axis=0) * model.dr
    A = A.real * mask
    # D = -i d_R - A is the covariant derivative appearing in U_en.
    Dphi = -1j * dphi - A[None, :] * phi
    D2phi = -1j * derivative(Dphi, model.dR, axis=1, boundary=boundary)
    D2phi -= A[None, :] * Dphi
    q = -1j * ratio + A
    u_phi = (0.5 * D2phi + q[None, :] * Dphi) / model.mass

    if norm_correction:
        # Discretization can leave a small anti-Hermitian expectation value.
        # Removing it improves preservation of <Phi_R|Phi_R>=1.
        expectation = np.sum(phi.conj() * u_phi, axis=0) * model.dr
        u_phi -= 1j * expectation.imag[None, :] * phi
    return A, u_phi, mask, ratio


def apply_hbo(field: np.ndarray, model: Model) -> np.ndarray:
    """Apply ``H_BO(R)`` to ``field(nr,nR)`` and return ``(nr,nR)``."""
    kr = 2.0 * np.pi * np.fft.fftfreq(len(model.r), d=model.dr)
    kinetic = np.fft.ifft(
        0.5 * kr[:, None] ** 2 * np.fft.fft(field, axis=0), axis=0
    )
    return kinetic + model.potential * field


def normalize_initial_chi(chi: np.ndarray, dR: float) -> np.ndarray:
    """Normalize ``chi(nR)`` so that ``sum_R |chi|^2 dR = 1``."""
    return chi / np.sqrt(np.sum(np.abs(chi) ** 2) * dR)


def enforce_partial_normalization(
    coefficients: np.ndarray, chi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Restore partial normalization while preserving the product ``C*chi``.

    Input/output shapes are ``coefficients(ns,nR)`` and ``chi(nR)``. The
    returned scalar is the maximum PNC error *before* projection.
    """
    norm = np.sqrt(np.sum(np.abs(coefficients) ** 2, axis=0))
    error = float(np.max(np.abs(norm**2 - 1.0)))
    safe = np.where(norm > 1.0e-14, norm, 1.0)
    return coefficients / safe[None, :], chi * safe, error


def phase_of_chi(chi: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Return unwrapped nuclear phase ``S(R)`` with shape ``(nR,)``."""
    phase = np.unwrap(np.angle(chi))
    if mask is not None:
        phase = np.where(mask > 0.5, phase, np.nan)
    return phase
