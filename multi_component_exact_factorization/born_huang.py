"""Electronic-only Born--Huang representation for nested MCEF.

The implementation follows Eqs. (71)--(86) of ``paper/MCEF_revised.pdf``.
Only the conditional electronic factor is expanded; ``Lambda`` and ``chi``
remain on their q/R grids.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def _project_basis_derivative(states, spacing, coordinate_axis, order, dx):
    """Project a q/R derivative of every right BO state on every left state."""
    n_states = states.shape[0]
    shape = (n_states, n_states)+states.shape[2:]
    result = np.empty(shape, dtype=complex)
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
