"""State-derived bridge fields; no unprovided force or scalar closure.

F2--F5 are evaluated in a declared *fixed primitive* electronic representation
(zero basis derivative couplings). A BO/channel amplitude must first be mapped
to that common representation with its actual basis phases retained. The
reference-only F6/F8 helpers below are not autonomous trajectory equations.
"""
import numpy as np

from .mls import build_mls_operator


def periodic_q_derivative(values, q, order=1, *, q_operator='fft'):
    """Uniform periodic-grid derivative along axis 1, retaining complex phase."""
    q = np.asarray(q, dtype=float)
    data = np.asarray(values)
    if q.ndim != 1 or len(q) < 3 or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite uniform periodic grid with >= 3 points")
    spacing = np.diff(q)
    if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1.e-10, atol=1.e-13):
        raise ValueError("FFT q derivative requires increasing uniform grid")
    if data.ndim < 2 or data.shape[1] != len(q) or not np.all(np.isfinite(data)):
        raise ValueError("values must have finite q samples on axis 1")
    if order not in (1, 2):
        raise ValueError("Only derivative orders 1 and 2 are supported")
    if q_operator != 'fft':
        from .q_operators import periodic_derivative
        return periodic_derivative(data, spacing[0], operator=q_operator,
                                   order=order, axis=1)
    k = 2 * np.pi * np.fft.fftfreq(len(q), d=spacing[0])
    multiplier = ((1j * k) ** order).reshape((1, len(q)) + (1,) * (data.ndim - 2))
    return np.fft.ifft(multiplier * np.fft.fft(data, axis=1), axis=1)


def analyze_fields(R, G, quadrature, q, *, basis_convention, gauge_convention,
                   mls_config=None, support_relative=1.e-10, pnc_tolerance=1.e-6,
                   q_operator='fft'):
    """Return a,b,alpha and bridge density in a common primitive basis.

    ``G.shape = (Ntrajectory, Nq, Nelectronic)``. The bridge is positive,
    Lambda=sqrt(sum_k |G_k|²); an outer chi phase is NOT presumed zero.
    ``basis_convention``/``gauge_convention`` are caller declarations and must
    identify common conventions, not independent phase choices per trajectory.

    q uses the declared periodic derivative with rectangle dq quadrature.
    Finite-difference local fields are continuum estimators; their product rule
    is not exact at finite spacing, and they are not native bond currents.
    Tail mass is reported;
    the caller must check finite-domain convergence. a,b are NaN off support,
    while alpha includes the *full* integrated numerator, including tails.
    No per-trajectory normalization is silently applied.
    """
    state = np.asarray(G, dtype=np.complex128)
    weights = np.asarray(quadrature, dtype=float)
    q = np.asarray(q, dtype=float)
    if state.ndim != 3 or state.shape[0] != len(R) or state.shape[1] != len(q) or state.shape[2] < 1:
        raise ValueError("G must have shape (Ntrajectory, Nq, Nelectronic)")
    if not np.all(np.isfinite(state)):
        raise ValueError("G contains NaN/Inf")
    if weights.shape != q.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Positive q quadrature vector required")
    if not 0 <= support_relative < 1 or not np.isfinite(pnc_tolerance) or pnc_tolerance < 0:
        raise ValueError("Invalid support threshold or PNC tolerance")
    Gq = periodic_q_derivative(state, q, q_operator=q_operator)
    if not np.allclose(weights, q[1] - q[0], rtol=1.e-10, atol=1.e-13):
        raise ValueError("Periodic FFT q requires constant rectangle quadrature dq")
    rho = np.sum(np.abs(state) ** 2, axis=2)
    norm = rho @ weights
    if np.any(norm <= 0) or np.max(np.abs(norm - 1.)) > pnc_tolerance:
        raise ValueError("Conditional G PNC fails: max error=%g; no hidden renormalization" % np.max(np.abs(norm - 1.)))
    operator = build_mls_operator(R, mls_config)
    GR, GRR, derivative_audit = operator.apply(
        state, basis_convention=basis_convention, gauge_convention=gauge_convention)
    amplitude = np.sqrt(rho)
    support = (rho > 0) & (rho > support_relative * np.max(rho, axis=1)[:, None])
    C = np.full_like(state, np.nan)
    C[support] = state[support] / amplitude[support, None]
    fields = {}
    for name, derivative in (("a", Gq), ("b", GR)):
        contraction = np.sum(state.conj() * derivative, axis=2)
        numerator = contraction.imag
        connection = np.full_like(rho, np.nan)
        connection[support] = numerator[support] / rho[support]
        # Product/quotient rule: remove the *real* radial derivative only.
        # This is not phase alignment and does not erase the imaginary current.
        Cderivative = np.full_like(state, np.nan)
        Cderivative[support] = (derivative[support] / amplitude[support, None] -
                                C[support] * (contraction.real[support] / rho[support])[:, None])
        complex_connection = -1j * np.sum(C.conj() * Cderivative, axis=2)
        fields[name] = connection
        fields[name + "_numerator"] = numerator
        fields[name + "_imaginary_defect"] = complex_connection.imag
        fields[name + "_quotient_residual"] = complex_connection.real - connection
        fields["C_q" if name == "a" else "C_R"] = Cderivative
        fields["Lambda_q" if name == "a" else "Lambda_R_derivative"] = np.divide(
            contraction.real, amplitude, out=np.full_like(rho, np.nan), where=amplitude > 0)
    alpha = fields["b_numerator"] @ weights
    supported_alpha = np.sum(np.where(support, fields["b"] * rho, 0.) * weights[None, :], axis=1)
    excluded_alpha = np.sum(np.where(support, 0., fields["b_numerator"]) * weights[None, :], axis=1)
    excluded_mass = np.sum(np.where(support, 0., rho) * weights[None, :], axis=1)
    # This independently measured imaginary defect is not discarded by taking
    # the real connection: exact conditional normalization requires it to vanish.
    alpha_complex = -1j * np.sum(state.conj() * GR * weights[None, :, None], axis=(1, 2))
    edge_count = min(2, len(q) // 2)
    edge_mass = np.sum(rho[:, :edge_count] * weights[None, :edge_count], axis=1)
    edge_mass += np.sum(rho[:, -edge_count:] * weights[None, -edge_count:], axis=1)
    fields.update(rho=rho, Lambda=amplitude, C=C, support=support,
                  alpha=alpha, delta_b=fields["b"] - alpha[:, None],
                  alpha_from_b_supported=supported_alpha,
                  alpha_excluded_contribution=excluded_alpha,
                  alpha_bridge_residual=alpha - supported_alpha - excluded_alpha,
                  alpha_imaginary_defect=alpha_complex.imag,
                  G_q=Gq, G_R=GR, G_RR=GRR,
                  excluded_mass=excluded_mass, q_edge_mass=edge_mass,
                  pnc_error=norm - 1., derivative_audit=derivative_audit,
                  basis_representation="fixed_primitive_dq_dR_zero",
                  basis_convention=basis_convention, gauge_convention=gauge_convention,
                  bridge_gauge="positive", outer_phase_assumed_zero=False,
                  q_boundary="periodic_"+q_operator, support_relative=float(support_relative),
                  scalar_potentials_available=False)
    return fields


def bridge_scalar_reference(epsilon1, rho, a, b, alpha, Lambda_q, Lambda_R,
                            quadrature, *, mq, mass_R, gauge_convention, Vq=0.):
    """F6 reference identity ONLY in full-positive-marginal gauge.

    All quantities, including epsilon1, must be supplied in that gauge on the
    same grid. This evaluator does not provide a missing trajectory scalar.
    Inputs must be finite over the full quadrature (no silently masked tails).
    """
    if gauge_convention != "full_positive_marginals":
        raise ValueError("F6 requires explicit full_positive_marginals gauge")
    density = np.asarray(rho, dtype=float)
    if density.ndim != 2:
        raise ValueError("rho must have shape (Ntrajectory, Nq)")
    arrays = [np.broadcast_to(np.asarray(x, dtype=float), density.shape)
              for x in (epsilon1, a, b, Lambda_q, Lambda_R, Vq)]
    eps, aq, br, lq, lr, potential = arrays
    ar = np.asarray(alpha, dtype=float)
    w = np.asarray(quadrature, dtype=float)
    if (ar.shape != (len(density),) or w.shape != (density.shape[1],) or np.any(w <= 0)
            or np.any(density < 0) or not np.all(np.isfinite(density))
            or not all(np.all(np.isfinite(x)) for x in arrays + [ar, w])):
        raise ValueError("F6 needs finite full-support fields and positive quadrature")
    if mq <= 0 or mass_R <= 0 or not np.isfinite(mq + mass_R):
        raise ValueError("Masses must be finite and positive")
    if not np.allclose(density @ w, 1., rtol=0., atol=1.e-8):
        raise ValueError("F6 needs a normalized conditional bridge")
    average = (density * (eps + potential)) @ w
    kinetic = ((lq ** 2 + aq ** 2 * density) / (2 * mq)) @ w
    geometry = ((lr ** 2 + (br - ar[:, None]) ** 2 * density) / (2 * mass_R)) @ w
    return {"epsilon2_reconstructed": average + kinetic + geometry,
            "epsilon1_plus_Vq_average": average, "bridge_q_kinetic": kinetic,
            "bridge_R_geometry": geometry, "reference_identity": "F6",
            "gauge_convention": gauge_convention, "propagation_closure": False}


def bridge_continuity_reference(rho, chi_density, chi_phase_gradient, a, b, alpha, *,
                                derivative_q, derivative_R, mq, mass_R,
                                bridge_gauge):
    """F8 reference source terms with an explicitly supplied complex-chi phase.

    Derivative callables act on arrays (Ntrajectory,Nq). It is the caller's
    responsibility to supply matched spatial operators. This evaluates an
    identity on given fields; it is not the MTE/TSH/CTMQC propagation RHS.
    """
    if bridge_gauge != "positive":
        raise ValueError("F8 requires positive bridge gauge")
    rho, a, b = [np.asarray(x, dtype=float) for x in (rho, a, b)]
    chi_density, phase, alpha = [np.asarray(x, dtype=float)
                                 for x in (chi_density, chi_phase_gradient, alpha)]
    if (rho.ndim != 2 or a.shape != rho.shape or b.shape != rho.shape
            or any(x.shape != (len(rho),) for x in (chi_density, phase, alpha))):
        raise ValueError("F8 grid shapes do not match")
    if not all(np.all(np.isfinite(x)) for x in (rho, a, b, chi_density, phase, alpha)):
        raise ValueError("F8 needs finite full-support fields")
    if np.any(rho < 0) or np.any(chi_density <= 0) or mq <= 0 or mass_R <= 0 or not np.isfinite(mq + mass_R):
        raise ValueError("F8 needs positive outer support and masses")
    def real_derivative(function, values):
        result = np.asarray(function(values))
        if result.shape != rho.shape or not np.all(np.isfinite(result)):
            raise ValueError("Invalid reference derivative")
        if np.iscomplexobj(result) and np.max(np.abs(result.imag)) > 1.e-10:
            raise ValueError("Reference real-field derivative has an imaginary defect")
        return result.real
    sq = -real_derivative(derivative_q, a*rho) / mq
    adv = -(alpha + phase)[:, None] * real_derivative(derivative_R, rho) / mass_R
    relative_flux = chi_density[:, None] * rho * (b-alpha[:, None])
    relative = -real_derivative(derivative_R, relative_flux) / (mass_R*chi_density[:, None])
    return {"S_q": sq, "S_adv": adv, "S_rel": relative,
            "density_rate_reference": sq+adv+relative,
            "reference_identity": "F8", "propagation_closure": False,
            "target": "partial_t_abs_Lambda_squared"}
