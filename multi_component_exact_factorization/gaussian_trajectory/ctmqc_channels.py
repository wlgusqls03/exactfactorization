"""Matched CTMQC equations in a declared internal adiabatic channel basis.

Atomic units; one classical heavy coordinate, quantum internal x--q space.
This is an independent equation implementation, NOT the pinned G-CTMQC
executable.  Its coefficient/force factors follow Agostini et al., JCTC 12,
2127 (2016), Eqs. (28),(29), DOI 10.1021/acs.jctc.5b01180, equivalently
Villaseco Arribas et al., Molecules 27,4002 (2022), Eqs. (14)--(18),
DOI 10.3390/molecules27134002.  In one dimension, with p_n=|A_n|^2,

  dot A_n = -i E_n A_n - (P/M) sum_m d_nm A_m
            + A_n sum_m p_m Q_nm (f_n-f_m)/M,
  F = Re(A^dagger F_matrix A) + 2 sum_(n<m) p_n p_m Q_nm (f_n-f_m)^2/M,
  dot f_n = (F_matrix)_nn = -partial_R E_n.

For one common Q this force is 2 Q Var_p(f)/M plus Ehrenfest.  The pair
form is the published state-pair modified-Q prescription, not a universal
scalar chi'/chi.  The zero-transfer condition constrains the CT-only channel
population rate, not the NAC term or bare electronic/photon Rabi exchange.

Density reconstruction reuses contract Q1--Q6.  Modified Q uses the common
pair intercept of 2022 Eqs. (26),(27), generalized to normalized ensemble
weights eta.  With a state-resolved mixture its slope is the algebraic
Gaussian-mixture slope sum_j posterior_j/(2 sigma_j^2); with fixed-width KDE
it is 1/(2 h^2).  Cancellation defaults to an error.  An explicitly selected
density fallback follows the alternative discussed after Eq. (27), and is
audited; it DOES NOT retain the zero-net-transfer condition.

No native prefactor patches, coefficient renormalization, decoherence
threshold resets, intercept clipping, MLS correction, or quantum-potential
force are performed here.  The caller must integrate the complete ensemble
vector at every stage.  Adding the user's P3/P8 Gaussian projection yields
an EXPERIMENTAL 'CTMQC equations + Gaussian bridge projection' method, not
uncompressed CTMQC and not an exact EF-G/MLS closure.
"""

import numpy as np

from .heavy_density import (DensityFailure, InterceptCancellationError,
                            reconstruct_heavy_density, reconstruct_kde_density)


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(name + " must be finite")
    return result


def _populations(amplitudes, normalization_tolerance):
    amplitudes = np.asarray(amplitudes, dtype=complex)
    if amplitudes.ndim != 2 or min(amplitudes.shape) < 1:
        raise ValueError("amplitudes must have shape (Ntrajectory,Nchannel)")
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("Nonfinite channel amplitudes")
    if not np.isfinite(normalization_tolerance) or normalization_tolerance < 0:
        raise ValueError("Invalid normalization_tolerance")
    norm = np.sum(abs(amplitudes) ** 2, axis=1)
    defect = float(np.max(abs(norm - 1.)))
    if np.any(norm <= 0) or defect > normalization_tolerance:
        raise ValueError("Channel norm defect %.6g exceeds tolerance %.6g; "
                         "increase internal basis or reduce integration step" %
                         (defect, normalization_tolerance))
    # Only expectation/probability normalization within the explicit tolerance.
    # Never change the supplied amplitudes or hide their pre-correction norm.
    return amplitudes, abs(amplitudes) ** 2 / norm[:, None], norm, defect


def pair_quantum_momentum(R, eta, populations, force_history, raw, slope, *,
                          mode="density", intercept_policy="raise",
                          cancellation_rtol=1.e-12):
    """Return Q[I,pair] and an explicit pair/intercept audit (never one field).

    Inactive means the CT source is identically zero, not that a nonzero
    signed denominator is small.  A cancelling denominator is not decoherence.
    The relative cancellation test is scale invariant in the force histories.
    """
    R, eta = _real(R, "R"), _real(eta, "eta")
    p = _real(populations, "populations")
    f, raw, slope = (_real(force_history, "force_history"),
                     _real(raw, "raw quantum momentum"), _real(slope, "slope"))
    if R.ndim != 1 or not len(R) or eta.shape != R.shape or p.ndim != 2 or p.shape[0] != len(R):
        raise ValueError("Expected R/eta[N], populations[N,state]")
    if f.shape != p.shape or raw.shape != R.shape or slope.shape != R.shape:
        raise ValueError("History, quantum momentum, or slope shape mismatch")
    if np.any(eta < 0) or abs(eta.sum()-1) > 1.e-10:
        raise ValueError("eta must be nonnegative and normalized")
    if np.any(p < 0) or np.max(abs(p.sum(axis=1)-1)) > 1.e-10 or np.any(slope < 0):
        raise ValueError("Normalized nonnegative populations and nonnegative slopes required")
    if mode not in ("density", "pair_intercept"):
        raise ValueError("Unknown quantum-momentum mode: " + str(mode))
    if intercept_policy not in ("raise", "density_fallback"):
        raise ValueError("Unknown intercept policy: " + str(intercept_policy))
    if not np.isfinite(cancellation_rtol) or not 0 <= cancellation_rtol < 1:
        raise ValueError("Invalid cancellation_rtol")
    pairs = [(n, m) for n in range(p.shape[1]) for m in range(n+1, p.shape[1])]
    values = np.empty((len(R), len(pairs)))
    records = []
    origin = float(np.dot(eta, R))
    for index, (n, m) in enumerate(pairs):
        source = eta * p[:, n] * p[:, m] * (f[:, n]-f[:, m])
        weight = slope * source
        absolute_sum = float(np.sum(abs(weight)))
        denominator = float(np.sum(weight))
        record = {"states": [n, m], "denominator": denominator,
                  "denominator_absolute_sum": absolute_sum, "intercept": None,
                  "status": "density"}
        if not np.isfinite(absolute_sum) or not np.isfinite(denominator):
            raise DensityFailure("Overflow in pair-intercept source")
        if mode == "density":
            values[:, index] = raw
        elif absolute_sum == 0:
            # f_n=f_m, no jointly occupied pair, or a zero slope: CT term is
            # exactly zero.  R_star is undefined and must not be manufactured.
            values[:, index] = 0.
            record["status"] = "zero_source"
        elif abs(denominator) <= cancellation_rtol * absolute_sum:
            if intercept_policy == "raise":
                raise InterceptCancellationError(
                    "Pair (%d,%d) intercept denominator cancels (%.6g / %.6g); "
                    "no silent cap; select density_fallback explicitly" %
                    (n, m, denominator, absolute_sum))
            values[:, index] = raw
            record["status"] = "density_fallback_cancellation"
        else:
            offset = float(np.dot(R-origin, weight) / denominator)
            center = origin + offset
            values[:, index] = slope * ((R-origin) - offset)
            record.update(status="pair_intercept", intercept=center)
        if not np.all(np.isfinite(values[:, index])):
            raise DensityFailure("Nonfinite pair quantum momentum; no clipping applied")
        record["raw_transfer_numerator"] = float(np.dot(raw, source))
        record["used_transfer_numerator"] = float(np.dot(values[:, index], source))
        records.append(record)
    return values, pairs, records


def ct_channel_rates(amplitudes, energies, derivative_couplings, force_matrix,
                     force_history, P, mass, pair_values, *,
                     normalization_tolerance=1.e-7, matrix_tolerance=1.e-8):
    """Pure current-state CTMQC coefficient, force and history rates.

    d_nm=<n|partial_R m>; F_nm=-<n|partial_R H_internal|m>.
    Off-diagonal F_nm=(E_n-E_m)d_nm is checked, including complex bases.
    External common heavy forces can be added by the caller, once only.
    pair_values is either common Q[N] or Q[N,pair] in n<m lexicographic order.
    The common-Q path uses vectorized mean/variance, avoiding O(Nstate^2)
    Python loops and pair-array allocation for a complete internal basis.
    """
    A, p, norm, norm_defect = _populations(amplitudes, normalization_tolerance)
    E, f, P = _real(energies, "energies"), _real(force_history, "force_history"), _real(P, "P")
    d, force = np.asarray(derivative_couplings, complex), np.asarray(force_matrix, complex)
    ntraj, nstate = A.shape
    if E.shape != A.shape or f.shape != A.shape or P.shape != (ntraj,):
        raise ValueError("E/f must match amplitudes and P must have shape (N,)")
    if d.shape != (ntraj, nstate, nstate) or force.shape != d.shape:
        raise ValueError("d/force_matrix must have shape (N,state,state)")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(force)):
        raise ValueError("Nonfinite derivative coupling/force matrix")
    if not np.isscalar(mass) or not np.isfinite(mass) or mass <= 0:
        raise ValueError("One positive finite heavy mass required")
    if not np.isfinite(matrix_tolerance) or matrix_tolerance < 0:
        raise ValueError("Invalid matrix_tolerance")
    anti_defect = float(np.max(abs(d + d.conj().transpose(0, 2, 1))))
    hermitian_defect = float(np.max(abs(force-force.conj().transpose(0, 2, 1))))
    off_diagonal = ~np.eye(nstate, dtype=bool)
    force_relation = force - (E[:, :, None]-E[:, None, :])*d
    force_relation_defect = float(np.max(abs(force_relation[:, off_diagonal]))) if nstate > 1 else 0.
    if max(anti_defect, hermitian_defect, force_relation_defect) > matrix_tolerance:
        raise ValueError("Inconsistent adiabatic d/force matrices: anti-Hermitian=%.6g, "
                         "Hermitian=%.6g, F_nm=(E_n-E_m)d_nm defect=%.6g" %
                         (anti_defect, hermitian_defect, force_relation_defect))
    qpair = _real(pair_values, "pair quantum momentum")
    if qpair.shape == (ntraj,):
        mean_history = np.sum(p*f, axis=1)
        centered = f-mean_history[:, None]
        growth = (qpair/mass)[:, None]*centered
        ct_force = (2.*qpair/mass)*np.sum(p*centered**2, axis=1)
    elif qpair.shape == (ntraj, nstate*(nstate-1)//2):
        pairs = [(n, m) for n in range(nstate) for m in range(n+1, nstate)]
        growth, ct_force = np.zeros_like(p), np.zeros(ntraj)
        for index, (n, m) in enumerate(pairs):
            difference = f[:, n]-f[:, m]
            coupling = qpair[:, index]*difference/mass
            growth[:, n] += p[:, m]*coupling
            growth[:, m] -= p[:, n]*coupling
            ct_force += 2.*p[:, n]*p[:, m]*coupling*difference
    else:
        raise ValueError("Quantum momentum must have shape (N,) or (N,Nstate*(Nstate-1)/2)")
    eh_rate = -1j*E*A - (P/mass)[:, None]*np.einsum("inm,im->in", d, A)
    ct_rate = growth*A
    mean_force = np.einsum("in,inm,im->i", A.conj(), force, A).real/norm
    amplitude_dot = eh_rate + ct_rate
    # Literal Eq.(18): no hidden subtraction, population threshold, or reset.
    fdot = np.diagonal(force, axis1=1, axis2=2).real.copy()
    audit = {"input_channel_norm_max_error": norm_defect,
             "coefficient_renormalized": False,
             "probabilities_normalized_for_expectations": True,
             "d_antihermitian_defect": anti_defect,
             "force_hermitian_defect": hermitian_defect,
             "force_nac_relation_defect": force_relation_defect,
             "norm_rate_max_abs": float(np.max(abs(2.*np.real(np.sum(A.conj()*amplitude_dot, axis=1))))),
             "force_history_policy": "literal_integrated_adiabatic_force_no_resets",
             "coefficient_ct_prefactor": 1., "unordered_pair_force_prefactor": 2.,
             "phase_convention": "adiabatic_schrodinger_phase_with_material_NAC"}
    return {"amplitude_dot": amplitude_dot, "heavy_force": mean_force + ct_force,
            "force_history_dot": fdot, "ehrenfest_amplitude_dot": eh_rate,
            "ct_amplitude_dot": ct_rate, "ehrenfest_force": mean_force,
            "ct_force": ct_force, "populations": p, "audit": audit}


def ctmqc_channel_rhs(amplitudes, energies, derivative_couplings, force_matrix,
                      force_history, R, P, mass, eta, *, channel_basis,
                      quantum_momentum_mode="density", density_mode="state_resolved",
                      width_floor=None, bandwidth=None, intercept_policy="raise",
                      cancellation_rtol=1.e-12, normalization_tolerance=1.e-7,
                      matrix_tolerance=1.e-8):
    """Whole-ensemble density reconstruction plus the matched channel RHS.

    Requires globally aligned internal eigenchannels across trajectories and
    history: eigenvalues sorted afresh at each step are not a history policy.
    The returned pair array must NOT be substituted for a single chi'/chi.
    Density mode returns an empty pair array: its common Q is in
    raw_quantum_momentum, without allocating redundant O(Nstate^2) copies.
    Norm errors are checked before any probability expectation normalization.
    """
    A, p, _, _ = _populations(amplitudes, normalization_tolerance)
    if not isinstance(channel_basis, str) or not channel_basis.strip():
        raise ValueError("Declare the common internal adiabatic channel basis")
    if quantum_momentum_mode not in ("density", "pair_intercept"):
        raise ValueError("Unknown quantum-momentum mode: " + str(quantum_momentum_mode))
    if intercept_policy not in ("raise", "density_fallback"):
        raise ValueError("Unknown intercept policy: " + str(intercept_policy))
    if not np.isfinite(cancellation_rtol) or not 0 <= cancellation_rtol < 1:
        raise ValueError("Invalid cancellation_rtol")
    if density_mode == "state_resolved":
        density = reconstruct_heavy_density(R, eta, p, channel_basis=channel_basis,
                                            width_floor=width_floor)
    elif density_mode == "kde":
        if bandwidth is None:
            raise ValueError("KDE requires an explicit bandwidth")
        density = reconstruct_kde_density(R, eta, bandwidth=bandwidth)
    else:
        raise ValueError("Unknown heavy-density mode: " + str(density_mode))
    active = density.channel_weights > 0
    slope = np.sum(density.posterior[:, active]/(2.*density.widths[active]**2), axis=1)
    if quantum_momentum_mode == "density":
        # Every RK stage reconstructs the density, but no pairs or intercept
        # records are needed for common raw Q.  This matters for full x--q
        # eigenbases with tens/hundreds of channels, not a 2-state truncation.
        pair_values = np.empty((len(A), 0))
        pairs, records = [], []
        used_q = density.quantum_momentum
    else:
        pair_values, pairs, records = pair_quantum_momentum(
            R, eta, p, force_history, density.quantum_momentum, slope,
            mode=quantum_momentum_mode, intercept_policy=intercept_policy,
            cancellation_rtol=cancellation_rtol)
        used_q = pair_values
    result = ct_channel_rates(A, energies, derivative_couplings, force_matrix,
                              force_history, P, mass, used_q,
                              normalization_tolerance=normalization_tolerance,
                              matrix_tolerance=matrix_tolerance)
    ct_population_rate = 2.*np.real(A.conj()*result["ct_amplitude_dot"])
    result.update(raw_quantum_momentum=density.quantum_momentum.copy(),
                  pair_quantum_momentum=pair_values, pair_indices=np.asarray(pairs, dtype=int).reshape(-1, 2),
                  density=density, quantum_momentum_slope=slope)
    result["audit"].update(
        channel_basis=channel_basis, quantum_momentum_mode=quantum_momentum_mode,
        quantum_momentum_representation=("common_density_no_pair_allocation" if
                                        quantum_momentum_mode == "density" else "state_pair_array"),
        density_mode=density_mode, density_audit=density.audit,
        pair_intercepts=records, intercept_policy=intercept_policy,
        fallback_pair_count=sum(v["status"] == "density_fallback_cancellation" for v in records),
        ct_ensemble_population_rate=np.asarray(eta) @ ct_population_rate,
        source_equations="Agostini2016(28,29); VillasecoArribas2022(14-18,26-27)",
        experimental_when_gaussian_projected=True,
        energy_conservation_guaranteed=False)
    return result
