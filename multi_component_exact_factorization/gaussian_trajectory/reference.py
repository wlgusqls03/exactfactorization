"""Equation-audit utilities, NOT a closed EF-G/MLS propagation backend.

R1/R4 need independently supplied chi'/chi, epsilon2 and a matched outer
equation. These routines deliberately do not manufacture them from a density.
All electronic arrays end in k; derivative matrices end in (k,l).
"""
import numpy as np


def _field(value, G):
    """Broadcast a trajectory scalar (N,) across q and electronic axes."""
    value = np.asarray(value)
    return value.reshape(value.shape+(1,)*(G.ndim-value.ndim))


def covariant_derivatives(G, first, second, d, D):
    """F1: D is the actual second basis derivative, never elementwise d**2."""
    mv = lambda A, v: np.einsum('...kl,...l->...k', A, v)
    return first + mv(d, G), second + 2*mv(d, first) + mv(D, G)


def outer_expanded(G, internal_action, G_R, G_RR, alpha, alpha_R,
                   chi_log_R, epsilon2, mass_R):
    """R4 after grouping ALL internal q/electronic physics in H_xq G.

    G_R/G_RR here are *physical* basis-covariant derivatives. In a moving
    basis obtain them using covariant_derivatives. Result is i partial_t G.
    """
    if not np.isfinite(mass_R) or mass_R <= 0:
        raise ValueError('mass_R must be positive')
    a, da, L, e = (_field(x, G) for x in
                    (alpha, alpha_R, chi_log_R, epsilon2))
    return (internal_action-e*G-G_RR/(2*mass_R)-L*G_R/mass_R
            +(1j*da/(2*mass_R)+1j*a*L/mass_R-a*a/(2*mass_R))*G)


def outer_compact(G, internal_action, derivative_R, alpha, chi_log_R,
                  epsilon2, mass_R):
    """R1 independently applying (-i partial_R - alpha) twice."""
    a, L, e = (_field(x, G) for x in
              (alpha, chi_log_R, epsilon2))
    residual = lambda u: -1j*derivative_R(u)-a*u
    return internal_action-e*G+residual(residual(G))/(2*mass_R)+(-1j*L+a)*residual(G)/mass_R


def material_rate(fixed_i_rhs, partial_R_G, velocity_R):
    """R3: exactly one R advection; no Gaussian-center/q advection."""
    return -1j*fixed_i_rhs+_field(velocity_R, partial_R_G)*partial_R_G


def project_internal(G, basis, quadrature, *, tolerance=1e-8):
    """B3/B4: test round trip BEFORE handing state to a truncated engine.

    G[q,k], basis[q,k,nu]. Never replace the input G by its projection.
    """
    w = np.asarray(quadrature)
    gram = np.einsum('q, qkn,qkm->nm', w, basis.conj(), basis)
    if not np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10, rtol=1e-10):
        raise ValueError('Internal basis is not orthonormal in this quadrature')
    A = np.einsum('q,qkn,qk->n', w, basis.conj(), G)
    reconstructed = np.einsum('qkn,n->qk', basis, A)
    norm = np.sum(w[:, None]*abs(G)**2)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Nonpositive internal-state norm')
    relative_error = float(np.sqrt(np.sum(w[:, None]*abs(G-reconstructed)**2)/norm))
    if relative_error > tolerance:
        raise ValueError('Internal-subspace round-trip error %.6g exceeds %.6g; expand basis' %
                         (relative_error, tolerance))
    return A, dict(round_trip_relative_error=relative_error,
                   outside_population=float(np.sum(w[:, None]*abs(G-reconstructed)**2)))


def lift_internal_rate(A, A_dot, basis, basis_R, velocity_R):
    """B5 includes moving internal-basis derivative; caller supplies both."""
    return np.einsum('qkn,n->qk', basis, A_dot)+velocity_R*np.einsum('qkn,n->qk', basis_R, A)


def require_backend(name):
    """No silent Ehrenfest fallback for TSH/CTMQC or unfinished EF closures."""
    if name == 'mte':
        return
    reasons = {
        'tsh': 'No validated internal-state hopping/rescaling/frustrated-hop adapter is exposed.',
        'ctmqc': 'A verified matched full-q coefficient/force/history/gauge adapter is missing; native finite-Fock methods remain separate.',
        'ef-g-mls': 'R1/R4 evaluation is implemented, but a matched scalar, force and outer-gauge closure is unspecified.',
    }
    raise NotImplementedError(reasons.get(name, 'Unknown backend: '+str(name)))
