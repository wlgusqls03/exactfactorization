"""Instantaneous positive-marginal-gauge nested EF on a full real-space wave.

The input axes are ``(R, x, Q)`` with ``Q = sqrt(omega) q_c``.  The returned
maps have axes ``(R, q_c)`` and use *physical* photon-coordinate quadrature.
No two-electronic-state approximation is made here.  The supplied action must
be the action of the SAME Hamiltonian used to propagate this wave.

Formula provenance (existing implementations, not a new formalism):
``vsc_polariton/PHASE7_NESTED_EF_DERIVATION.md``;
``vsc_polariton/real_grid_mcef_fields.py::analyze``; and the expanded bridge
operator in ``vsc_polariton/perspective_bridge_data.py::model_a_bridge``.
They are reproduced in this isolated namespace to avoid dependencies on old
research scripts or modifying their behavior.

Time derivatives come from ``dot(Psi) = -i H Psi``, never sparse-frame phase
unwrapping.  Quotient derivatives are evaluated analytically using native
wave derivatives.  Their algebraic closure is NOT a spatial-convergence
certificate: finite spectral grids do not obey a product rule for arbitrary
factor quotients.  Complex residuals and undefined nodes are retained.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dst, fft, ifft


def derivative(values, spacing, axis, order=1):
    """Native periodic FFT derivative for R and Q (not the Dirichlet x axis)."""
    values = np.asarray(values)
    shape = [1] * values.ndim
    shape[axis] = values.shape[axis]
    k = 2 * np.pi * np.fft.fftfreq(values.shape[axis], float(spacing))
    spectrum = fft(values, axis=axis)
    spectrum *= ((1j * k) ** order).reshape(shape)
    return ifft(spectrum, axis=axis, overwrite_x=True)


def ratio(numerator, density):
    """Divide on positive density only; exact nodes are NaN, never floored."""
    numerator, density = np.asarray(numerator), np.asarray(density)
    shape = np.broadcast_shapes(numerator.shape, density.shape)
    result = np.full(shape, np.nan, dtype=np.result_type(numerator, density, float))
    return np.divide(numerator, density, out=result,
                     where=np.broadcast_to(density, shape) > 0)


def electronic_kinetic(values, dx):
    """Dirichlet DST-I electronic kinetic action on axis 1, mass one."""
    nx = values.shape[1]
    energy = .5 * (np.pi * np.arange(1, nx + 1) / ((nx + 1) * dx)) ** 2
    coefficients = dst(values, type=1, axis=1, norm='ortho')
    coefficients *= energy[None, :, None]
    return dst(coefficients, type=1, axis=1, norm='ortho', overwrite_x=True)


def _q_integral(values, joint, dq):
    """Integrate occupied cells, omitting ONLY exact zero-probability nodes.

    A nonfinite value at any strictly positive-density cell propagates to the
    result.  This deliberately does not use nansum or a density cutoff.
    """
    return np.sum(np.where(joint > 0, values, 0), axis=1) * dq


def analyze(psi_Q, packet, Q, omega, g_cavity, action_Q, *, support=1e-5, block=4):
    """Extract full-wave fields and bridge identities in positive gauge.

    Parameters
    ----------
    psi_Q, action_Q : complex arrays, shape (nR, nx, nQ)
        Wave normalized in dR dx dQ and its native full Hamiltonian action.
    packet : mapping
        R, x, dx, dR, mass are required.  Optional V(R,x), mu(R,x), and
        dse (default True) enable a physical electronic/LM/DSE budget.
        Optional bo_phi(R,x,nBO) enables BO-projection diagnostics.
    Q : increasing uniform dimensionless photon grid
    omega, g_cavity : float
        H_cav = omega*(P_Q**2+Q**2)/2 + sqrt(omega)*g*Q*mu
        + g**2*mu**2/2.  Here g is NOT the old g_chi parameter.
    support : float
        Relative-density threshold for diagnostic masks only, not division.
    block : int
        Number of nuclear slices processed together on host RAM.

    Returns
    -------
    dict
        NPZ-safe scalars/arrays.  epsilon1/epsilon2 are real plotting views;
        epsilon*_A/B retain complex expectation/inversion values.  S_q,
        S_adv, S_rel sum to density_dt = d_t |Lambda_R|**2, NOT i d_t Lambda.
        lambda_dt and T1..T8 separately check the complex bridge equation.
    """
    psi_Q, action_Q = np.asarray(psi_Q), np.asarray(action_Q)
    R, x, Q = (np.asarray(v, dtype=float) for v in (packet['R'], packet['x'], Q))
    omega, g_cavity = float(omega), float(g_cavity)
    dr, dx, mass = (float(packet[k]) for k in ('dR', 'dx', 'mass'))
    if psi_Q.shape != (len(R), len(x), len(Q)) or action_Q.shape != psi_Q.shape:
        raise ValueError('psi_Q and action_Q must both have axes (R,x,Q)')
    if min(len(R), len(x), len(Q)) < 3 or omega <= 0 or mass <= 0:
        raise ValueError('Positive frequency/mass and at least three points per axis required')
    if not 0 <= support < 1 or block < 1:
        raise ValueError('support must be in [0,1), block must be positive')
    for grid, spacing, label in ((R, dr, 'R'), (x, dx, 'x'), (Q, Q[1]-Q[0], 'Q')):
        if spacing <= 0 or not np.allclose(np.diff(grid), spacing, rtol=1e-9, atol=1e-12):
            raise ValueError(f'{label} must be an increasing uniform grid')
    if not np.all(np.isfinite(psi_Q)) or not np.all(np.isfinite(action_Q)):
        raise ValueError('Nonfinite wave or Hamiltonian action')

    q = Q / np.sqrt(omega)
    dq = float(q[1] - q[0])
    u = psi_Q * omega**.25
    ur, urr = derivative(u, dr, 0), derivative(u, dr, 0, 2)
    ut = -1j * action_Q * omega**.25
    inner = lambda v: np.sum(u.conj() * v, axis=(1, 2)) * dx * dq
    rho = np.sum(abs(u)**2, axis=(1, 2)) * dx * dq
    rho_r = 2 * inner(ur).real
    rho_rr = 2 * (inner(urr).real + np.sum(abs(ur)**2, axis=(1, 2)) * dx * dq)
    rho_t = 2 * inner(ut).real
    alpha = ratio(inner(ur).imag, rho)
    alpha_r = ratio(inner(urr).imag, rho) - alpha * ratio(rho_r, rho)
    log_chi_r, log_chi_t = ratio(rho_r, 2*rho), ratio(rho_t, 2*rho)
    chi_curvature = ratio(rho_rr, 2*rho) - log_chi_r**2
    geo = (ratio(np.sum(abs(ur)**2, axis=(1, 2))*dx*dq, rho)
           - log_chi_r**2 - alpha**2) / (2*mass)
    conditional = ratio(1j*inner(ut) + inner(urr)/(2*mass), rho)
    gd = -1j * (ratio(inner(ut), rho) - log_chi_t)
    e2 = conditional + geo + gd
    e2_inv = ((chi_curvature-alpha**2)/(2*mass)
              + 1j*(log_chi_t+(alpha_r+2*alpha*log_chi_r)/(2*mass)))
    utr = derivative(ut, dr, 0)
    alpha_t = (ratio(np.sum((ut.conj()*ur+u.conj()*utr).imag, axis=(1, 2))*dx*dq, rho)
               - alpha*ratio(rho_t, rho))
    del utr
    urrr = derivative(u, dr, 0, 3)
    rho_rrr = 2*inner(urrr).real + 6*np.sum((ur.conj()*urr).real, axis=(1, 2))*dx*dq
    del urrr
    curvature_r = (ratio(rho_rrr, 2*rho)-ratio(rho_r*rho_rr, rho**2)
                   + .5*ratio(rho_r**3, rho**3))
    e2_r = curvature_r/(2*mass)-alpha*alpha_r/mass
    out = dict(R=R, x=x, q=q, Q=Q, omega=np.asarray(omega),
               g_cavity=np.asarray(g_cavity), mass=np.asarray(mass),
               gauge=np.asarray('positive_density_marginals'),
               derivative_scheme=np.asarray('native-wave FFT quotient; Hpsi instantaneous time derivative'),
               source_target=np.asarray('partial_t |Lambda_R(q_c)|^2'),
               source_units=np.asarray('probability density per physical q_c per atomic time'),
               rho_R=rho, rho_R_t=rho_t, chi=np.sqrt(rho),
               current=rho*alpha/mass, alpha=alpha, alpha_R=alpha_r,
               alpha_t=alpha_t, epsilon2=e2.real, epsilon2_A=e2,
               epsilon2_B=e2_inv, epsilon2_GI=(conditional+geo).real,
               epsilon2_GD=gd.real, epsilon2_GD_complex=gd,
               epsilon2_cond=conditional, epsilon2_geo=geo,
               epsilon2_R=e2_r, force=-e2_r+alpha_t)
    reconstruction = 0.
    derivative_reconstruction = 0.
    derivative_at_undefined_nodes = 0.
    for first in range(0, len(R), block):
        s = slice(first, min(first+block, len(R)))
        z, zr, zrr, zt = u[s], ur[s], urr[s], ut[s]
        zq, zqq = derivative(z, dq, 2), derivative(z, dq, 2, 2)
        ix = lambda v: np.sum(z.conj()*v, axis=1)*dx
        joint = np.sum(abs(z)**2, axis=1)*dx
        amplitude = np.sqrt(joint)
        fq, fr, ft = (ratio(ix(v).real, joint) for v in (zq, zr, zt))
        fqq = ratio(ix(zqq).real+np.sum(abs(zq)**2, axis=1)*dx, joint)-fq**2
        frr = ratio(ix(zrr).real+np.sum(abs(zr)**2, axis=1)*dx, joint)-fr**2
        phi = ratio(z, amplitude[:, None, :])
        phiq = ratio(zq-z*fq[:, None, :], amplitude[:, None, :])
        phir = ratio(zr-z*fr[:, None, :], amplitude[:, None, :])
        phit = ratio(zt-z*ft[:, None, :], amplitude[:, None, :])
        a, b = ratio(ix(zq).imag, joint), ratio(ix(zr).imag, joint)
        aq, br = ratio(ix(zqq).imag, joint)-2*a*fq, ratio(ix(zrr).imag, joint)-2*b*fr
        gq = (np.sum(abs(phiq)**2, axis=1)*dx-a**2)/2
        gr = (np.sum(abs(phir)**2, axis=1)*dx-b**2)/(2*mass)
        gd1 = -1j*np.sum(phi.conj()*phit, axis=1)*dx
        # h = H - T_R - T_q, with exactly the action used in propagation.
        hz = 1j*zt+zrr/(2*mass)+zqq/2
        cond1 = ratio(ix(hz), joint)
        e1 = cond1+gq+gr+gd1
        e1_inv = (.5*(fqq-a**2)+(frr-b**2)/(2*mass)
                  + 1j*(ft+a*fq+aq/2+(b*fr+br/2)/mass))

        chi = np.sqrt(rho[s, None])
        lam = ratio(amplitude, chi)
        lr = lam*(fr-log_chi_r[s, None])
        lq = lam*fq
        lt = lam*(ft-log_chi_t[s, None])
        lrr = lam*(frr-2*fr*log_chi_r[s, None]+2*log_chi_r[s, None]**2-chi_curvature[s, None])
        lqq = lam*fqq
        al = alpha[s, None]
        delta = b-al
        t1, t2 = -lrr/(2*mass), -log_chi_r[s, None]*lr/mass
        t3, t4 = delta**2*lam/(2*mass), al*delta*lam/mass
        t5, t6 = -1j*delta*lr/mass, -.5j*(br-alpha_r[s, None])*lam/mass
        t7, t8 = -1j*al*lr/mass, -1j*log_chi_r[s, None]*delta*lam/mass
        u_quad, u_lin = t1+t3+t5+t6, t2+t4+t7+t8
        kq_action = .5*(-lqq+a*a*lam)-1j*(a*lq+.5*aq*lam)
        lambda_rhs = kq_action+e1*lam+u_quad+u_lin-e2[s, None]*lam
        density_dt = 2*lam*lt
        sq = -(aq*lam**2+2*a*lam*lq)
        sadv = -2*al*lam*lr/mass
        srel = -(2*fr*lam**2*delta+lam**2*(br-alpha_r[s, None]))/mass
        e1_mean = _q_integral(lam**2*e1, joint, dq)
        kq = _q_integral((lq**2+a**2*lam**2)/2, joint, dq)
        g_r_lambda = _q_integral((lr**2+delta**2*lam**2)/(2*mass), joint, dq)
        c_t = _q_integral(-1j*lam*lt, joint, dq)
        alpha_reconstructed = _q_integral(lam**2*b, joint, dq)
        e2_reconstructed = e1_mean+kq+g_r_lambda+c_t
        rebuilt = phi*lam[:, None, :]*chi[:, None, :]
        rebuilt_t = (phit*lam[:, None, :]*chi[:, None, :]
                     + phi*lt[:, None, :]*chi[:, None, :]
                     + phi*lam[:, None, :]*chi[:, None, :]*log_chi_t[s, None, None])
        occupied = np.broadcast_to(joint[:, None, :] > 0, z.shape)
        reconstruction += float(np.sum(abs(rebuilt[occupied]-z[occupied])**2)*dx*dq*dr)
        derivative_reconstruction += float(np.sum(abs(rebuilt_t[occupied]-zt[occupied])**2)*dx*dq*dr)
        derivative_at_undefined_nodes += float(np.sum(abs(zt[~occupied])**2)*dx*dq*dr)
        fields = dict(rho_qR=joint, lambda_density=lam**2, Lambda=lam,
                      lambda_dt=lt, density_dt=density_dt, a=a, b=b,
                      a_q=aq, b_R=br, delta_b=delta,
                      epsilon1=e1.real, epsilon1_A=e1, epsilon1_B=e1_inv,
                      epsilon1_GI=(cond1+gq+gr).real, epsilon1_GD=gd1.real,
                      epsilon1_GD_complex=gd1, epsilon1_cond=cond1,
                      epsilon1_qgeo=gq, epsilon1_Rgeo=gr,
                      alpha_reconstructed=alpha_reconstructed,
                      alpha_nested=alpha_reconstructed,
                      epsilon2_reconstructed=e2_reconstructed,
                      epsilon2_nested=e2_reconstructed,
                      mean_e1=e1_mean, Kq=kq, G_R_lambda=g_r_lambda, Ct=c_t,
                      G_R_lambda_amplitude=_q_integral(lr**2/(2*mass), joint, dq),
                      G_R_lambda_variance=_q_integral(delta**2*lam**2/(2*mass), joint, dq),
                      S_q=sq, S_adv=sadv, S_rel=srel,
                      Sq=sq, Sadv=sadv, Srel=srel,
                      continuity_residual=density_dt-sq-sadv-srel,
                      J_rel=joint*delta/mass,
                      T1=t1, T2=t2, T3=t3, T4=t4,
                      T5=t5, T6=t6, T7=t7, T8=t8,
                      U_quad=u_quad, U_lin=u_lin, Kq_action=kq_action,
                      lambda_eom_rhs=lambda_rhs,
                      lambda_eom_residual=1j*lt-lambda_rhs,
                      electronic_PNC_error=abs(np.sum(abs(phi)**2, axis=1)*dx-1),
                      photon_PNC_error=abs(_q_integral(lam**2, joint, dq)-1),
                      conditional_q=_q_integral(lam**2*q, joint, dq),
                      berry=2*ratio(np.sum((zq.conj()*zr).imag, axis=1)*dx, joint)-2*b*fq+2*a*fr)
        if 'V' in packet and 'mu' in packet:
            bare = electronic_kinetic(z, dx)+np.asarray(packet['V'])[s, :, None]*z
            mu = np.asarray(packet['mu'])[s]
            lm = np.sqrt(omega)*g_cavity*mu[:, :, None]*Q
            dse = .5*g_cavity**2*mu**2 if bool(packet.get('dse', True)) else np.zeros_like(mu)
            fields.update(epsilon1_bare=ratio(ix(bare), joint),
                          epsilon1_LM=ratio(ix(lm*z), joint),
                          epsilon1_DSE=ratio(ix(dse[:, :, None]*z), joint),
                          h_action_budget_residual=np.sqrt(np.sum(abs(hz-bare-(lm+dse[:, :, None]+.5*omega*Q**2)*z)**2,
                                                                        axis=1)*dx))
        if 'bo_phi' in packet or 'phi' in packet:
            basis = packet.get('bo_phi', packet.get('phi'))
            bo = np.einsum('rxj,rxq->rjq', np.asarray(basis)[s].conj(), z)*dx
            bo_density = np.moveaxis(abs(bo)**2, 1, 2)
            fields.update(BO_density=bo_density,
                          BO_character=ratio(bo_density, joint[:, :, None]))
        for key, value in fields.items():
            if key not in out:
                out[key] = np.empty((len(R),)+value.shape[1:], dtype=value.dtype)
            out[key][s] = value
    out['reconstruction_L2'] = np.asarray(np.sqrt(reconstruction))
    out['time_derivative_reconstruction_L2'] = np.asarray(np.sqrt(derivative_reconstruction))
    out['time_derivative_undefined_nodes_L2'] = np.asarray(np.sqrt(derivative_at_undefined_nodes))
    out['support_relative'] = np.asarray(support)
    out['support_R'] = rho > (support*np.max(rho))
    out['support_qR'] = out['rho_qR'] > (support*np.max(out['rho_qR']))
    out['alpha_bridge_residual'] = alpha-out['alpha_reconstructed']
    out['epsilon2_bridge_residual'] = e2-out['epsilon2_reconstructed']
    # Independent native differentiation of reconstructed probability currents.
    # Unlike quotient-rule closure above, finite FFT differentiation of products
    # is not exact. Retain this defect: a tiny algebraic residual alone cannot
    # certify resolution. Exact-zero density has exactly zero probability flux.
    joint = out['rho_qR']
    j_q = np.where(joint > 0, joint*out['a'], 0.)
    j_r = np.where(joint > 0, joint*out['b']/mass, 0.)
    joint_t = out['density_dt']*rho[:, None]+out['lambda_density']*rho_t[:, None]
    joint_t = np.where(joint > 0, joint_t, 0.)
    outer_defect = rho_t+derivative(np.where(rho > 0, rho*alpha/mass, 0.), dr, 0).real
    joint_defect = joint_t+derivative(j_q, dq, 1).real+derivative(j_r, dr, 0).real
    out['outer_continuity_native_residual'] = outer_defect
    out['continuity_native_residual'] = (ratio(joint_defect, rho[:, None])
                                       -out['lambda_density']*ratio(outer_defect, rho)[:, None])
    return out


def _weighted_metrics(values, weight, mask):
    """Strict support-weighted RMS, max and invalid probability fraction."""
    values, weight, mask = np.asarray(values), np.asarray(weight), np.asarray(mask, dtype=bool)
    selected = mask & (weight > 0)
    total = float(weight[selected].sum())
    invalid = selected & ~np.isfinite(values)
    if total == 0:
        return dict(rms=None, max_abs=None, invalid_weight_fraction=None)
    if invalid.any():
        return dict(rms=None, max_abs=None,
                    invalid_weight_fraction=float(weight[invalid].sum()/total))
    return dict(rms=float(np.sqrt(np.sum(weight[selected]*abs(values[selected])**2)/total)),
                max_abs=float(np.max(abs(values[selected]))), invalid_weight_fraction=0.)


def metrics(fields):
    """JSON-safe field diagnostics; identity closure is not production PASS."""
    f = fields
    dr, dq = float(np.diff(f['R'])[0]), float(np.diff(f['q'])[0])
    weight_r, weight_j = f['rho_R']*dr, f['rho_qR']*dr*dq
    mr, mj = f['support_R'], f['support_qR']
    total = float(weight_r.sum())
    result = dict(norm=total, reconstruction_L2=float(f['reconstruction_L2']),
                  time_derivative_reconstruction_L2=float(f['time_derivative_reconstruction_L2']),
                  time_derivative_undefined_nodes_L2=float(f.get('time_derivative_undefined_nodes_L2', 0.)),
                  time_derivative_reconstruction_scope='strictly positive joint density; undefined-node action reported separately',
                  support_relative=float(f['support_relative']),
                  excluded_outer_probability=float(weight_r[~mr].sum()),
                  excluded_joint_probability=float(weight_j[~mj].sum()),
                  scope='instantaneous PG quotient identities; not a spatial/gauge-convergence certificate',
                  phase7_pass=False, residuals={})
    joint_values = dict(epsilon1_route=f['epsilon1_A']-f['epsilon1_B'],
                        epsilon1_imag=f['epsilon1_A'].imag,
                        continuity=f['continuity_residual'],
                        lambda_eom=f['lambda_eom_residual'],
                        electronic_PNC=f['electronic_PNC_error'])
    if 'continuity_native_residual' in f:
        joint_values['continuity_native_product_rule'] = f['continuity_native_residual']
    if 'h_action_budget_residual' in f:
        joint_values['hamiltonian_budget'] = f['h_action_budget_residual']
    outer_values = dict(epsilon2_route=f['epsilon2_A']-f['epsilon2_B'],
                        epsilon2_imag=f['epsilon2_A'].imag,
                        epsilon2_bridge=f['epsilon2_bridge_residual'],
                        alpha_bridge=f['alpha_bridge_residual'],
                        photon_PNC=f['photon_PNC_error'])
    if 'outer_continuity_native_residual' in f:
        outer_values['outer_continuity_native_product_rule'] = f['outer_continuity_native_residual']
    for name, value in joint_values.items():
        result['residuals'][name] = _weighted_metrics(value, weight_j, mj)
    for name, value in outer_values.items():
        result['residuals'][name] = _weighted_metrics(value, weight_r, mr)
    result['support_scan'] = {}
    for threshold in (1e-4, 1e-5, 1e-6):
        mask_r = f['rho_R'] > threshold*np.max(f['rho_R'])
        mask_j = f['rho_qR'] > threshold*np.max(f['rho_qR'])
        result['support_scan'][str(threshold)] = dict(
            excluded_joint_probability=float(weight_j[~mask_j].sum()),
            excluded_outer_probability=float(weight_r[~mask_r].sum()),
            continuity=_weighted_metrics(f['continuity_residual'], weight_j, mask_j),
            epsilon2_bridge=_weighted_metrics(f['epsilon2_bridge_residual'], weight_r, mask_r))
    return result
