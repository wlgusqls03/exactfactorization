"""Single-set frozen coherent Gaussians, complex Dirac--Frenkel TDVP.

References: Shalashilin/Burghardt JCP 129, 084104 (2008), Eqs. 33,35;
Vacher et al. TCA 135,187 (2016), Eqs. 37,42--46; Wang et al.
arXiv:2103.05307, Eq. 9 and SI. Normalized coherent-state gauge is used.
All propagated variables are z[K], d[electronic,K], NOT C[electronic,q].
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, svd


@dataclass
class State:
    z: np.ndarray
    d: np.ndarray

    def packed(self):
        return np.concatenate((self.z, self.d.ravel())).astype(complex)

    @classmethod
    def unpack(cls, y, ne, K):
        return cls(np.asarray(y[:K]), np.asarray(y[K:]).reshape(ne, K))


def overlap(z, other=None):
    """<z_i|w_j>, including the coherent-state phase."""
    w = z if other is None else other
    return np.exp(-abs(z[:, None])**2/2-abs(w[None, :])**2/2
                  + z[:, None].conj()*w[None, :])


def gaussian(q, z, sigma, origin=0.):
    """Values, first and second analytic q derivatives (evaluation only)."""
    x = np.asarray(q)[:, None]-origin
    Q, P = 2*sigma*z.real, z.imag/sigma
    g = (2*np.pi*sigma*sigma)**(-.25)*np.exp(
        -(x-Q)**2/(4*sigma*sigma)+1j*P*(x-Q/2))
    logq = -(x-Q)/(2*sigma*sigma)+1j*P
    return g, logq*g, (logq**2-1/(2*sigma*sigma))*g


def fock_coefficients(z, nf):
    """Only for reference/projection diagnostics; no Fock variables propagate."""
    f = np.empty((nf, len(z)), complex)
    f[0] = np.exp(-abs(z)**2/2)
    for n in range(1, nf):
        f[n] = f[n-1]*z/np.sqrt(n)
    return f


def hermitian_inverse(A, rtol=1e-11, atol=1e-14):
    """Rank-revealing Hermitian pseudoinverse; never adds a hidden ridge."""
    A = (A+A.conj().T)/2
    e, U = eigh(A, check_finite=False)
    threshold = max(atol, rtol*max(float(e[-1]), 0.))
    if e[0] < -max(1e-10, 1e-7*abs(e[-1])):
        raise FloatingPointError('Tangent metric is not positive semidefinite')
    keep = e > threshold
    inv = (U[:, keep]/e[keep]) @ U[:, keep].conj().T
    return inv, dict(rank=int(keep.sum()), size=len(e),
                     min_eigenvalue=float(e[0]), threshold=threshold,
                     condition=float(e[-1]/e[keep][0]) if keep.any() else None)


def tangent_rate(state, H00, H10, *, rtol=1e-10, conserve=False,
                 shape_regularization=1e-7, extra_rate=None):
    """Solve i M v = <B|H psi> using the coefficient/shape Schur complement.

    B_c=|k>|z_j>, B_z=sum_k d_kj |k>a^dagger|z_j>.
    v=(u,zdot); ddot=u+d*Re(z* zdot). H10[i,k,j,l] is
    <a^dagger z_i,k|H|z_j,l>. H00 uses the same [i,k,j,l] order.
    The complex tangent space makes action-TDVP and complex residual
    projection equivalent here; this is NOT the old real-amplitude metric.
    """
    z, d = state.z, state.d
    K = len(z)
    S = overlap(z)
    # An auxiliary analytic coherent-state Fock expansion is used ONLY for
    # a square-root factorization of the metric (never as propagated state).
    # Forming ss-cs^dagger S^-1 cs directly catastrophically cancels near
    # coalescing Gaussians. Residual vectors below give a PSD Schur metric.
    nmetric = int(np.ceil(max(abs(z))**2+14*np.sqrt(max(abs(z))**2+1)+40))
    F = fock_coefficients(z, nmetric)
    Uf, sf, Vh = svd(F, full_matrices=False, check_finite=False)
    keep = sf > max(1e-14, np.sqrt(rtol)*sf[0])
    pinvF = (Vh[keep].conj().T/sf[keep])@Uf[:,keep].conj().T
    Sinv = pinvF@pinvF.conj().T
    sa = dict(rank=int(keep.sum()),size=K,condition=float((sf[0]/sf[keep][-1])**2),
              square_root_fock_terms=nmetric,
              fock_metric_tail=float(np.max(abs(np.diag(S)-np.sum(abs(F)**2,axis=0)))))
    bc = np.einsum('ikjl,lj->ki', H00, d)
    bs = np.einsum('ki,ikjl,lj->i', d.conj(), H10, d)
    bc_physical, bs_physical = bc.copy(), bs.copy()
    if extra_rate is not None:
        if conserve:
            raise ValueError('Do not freeze internal energy for a CTMQC/driven RHS')
        # Inner products <B|F_extra>, e.g. an explicitly supplied native CT
        # source. Projecting it defines a NEW approximation, not native CTMQC.
        bc = bc+1j*extra_rate[0]
        bs = bs+1j*extra_rate[1]
    # M_cs[k,i,j] = d[k,j] z_i* S[i,j].
    cs = d[:, None, :]*z.conj()[None, :, None]*S[None]
    ss = (d.conj().T@d)*(1+z.conj()[:, None]*z[None, :])*S
    T = np.zeros_like(F)
    T[1:] = np.sqrt(np.arange(1,nmetric))[:,None]*F[:-1]
    Lt = pinvF@T
    L = d[:,None,:]*Lt[None]
    u0 = -1j*(bc@Sinv.T)
    Tperp = T-F@Lt
    schur = (d.conj().T@d)*(Tperp.conj().T@Tperp)
    rhs = -1j*bs-np.einsum('kij,ki->j', cs.conj(), u0)
    # Absolute cutoff is relative to the original shape metric: cancellation
    # in a nearly zero Schur complement must not be amplified into motion.
    cutoff = max(1e-14, rtol*float(np.max(abs(np.diag(ss)))))
    inv, za = hermitian_inverse(schur, rtol=rtol, atol=cutoff)
    if shape_regularization < 0:
        raise ValueError('Shape regularization must be nonnegative')
    if shape_regularization:
        # Explicit smooth Tikhonov filter of the SHAPE Schur complement only.
        # Abrupt rank switches cause tiny adaptive timesteps at empty initial
        # Gaussians. The coefficient equation is unchanged. The resulting
        # physical generator remains Hermitian, so continuous norm/energy
        # conservation survives, but the exact TDVP minimum is modified.
        ev, U = eigh((schur+schur.conj().T)/2, check_finite=False)
        tau = shape_regularization*max(1., float(np.max(abs(np.diag(ss)))))
        filt = ev/(ev*ev+tau*tau)
        inv = (U*filt)@U.conj().T
        za['smooth_filter_scale'] = tau
        za['smooth_filter_active'] = True
    zdot = inv@rhs
    u = u0-np.einsum('kij,j->ki', L, zdot)
    norm = float(np.einsum('ki,ij,kj', d.conj(), S, d).real)
    E = float(np.vdot(d, bc_physical).real/norm)

    def rates(uc):
        nr = 2*np.real(np.vdot(d, uc@S.T)
            + np.einsum('ki,kij,j->', d.conj(), cs, zdot))
        er = 2*np.real(np.vdot(bc_physical, uc)+np.vdot(bs_physical, zdot))-E*nr
        return np.array([nr, er/norm])

    raw = rates(u)
    correction_norm = 0.
    if conserve:
        # Optional continuous-rate constraints, NOT a post-step energy reset.
        # Minimal correction in the coefficient subspace span{|k>g_j}.
        evec = bc@Sinv.T-E*d
        directions = np.stack((d, evec))
        A = np.array([[2*np.real(np.vdot(d, x@S.T)) for x in directions],
                      [2*np.real(np.vdot(bc-E*(d@S.T), x))/norm
                       for x in directions]])
        scales = np.linalg.lstsq(A, -raw, rcond=rtol)[0]
        delta = np.einsum('a,aki->ki', scales, directions)
        u = u+delta
        correction_norm = float(np.sqrt(max(0.,
            np.einsum('ki,ij,kj', delta.conj(), S, delta).real)))
    ddot = u+d*np.real(z.conj()*zdot)[None]
    audit = dict(norm=norm, energy=E, coefficient_metric=sa, shape_metric=za,
                 raw_norm_rate=float(raw[0]), raw_energy_rate=float(raw[1]),
                 norm_rate=float(rates(u)[0]), energy_rate=float(rates(u)[1]),
                 constraint_correction_norm=correction_norm,
                 max_z_speed=float(max(abs(zdot))),
                 shape_equation_defect=float(np.linalg.norm(schur@zdot-rhs)))
    return State(zdot, ddot), audit


def evaluate(state, q, sigma, origin=0., basis=None):
    """Reconstruct physical G,dqG and positive Lambda,C on requested points.

    Points can be plot points or local quadrature nodes; they are not state
    variables. basis=(U,dU/dq) maps electronic channels to primitive space.
    """
    g, dg, _ = gaussian(q, state.z, sigma, origin)
    channels, deriv = g@state.d.T, dg@state.d.T
    if basis is not None:
        U, Uq = basis
        deriv = np.einsum('nkl,nl->nk', U, deriv)+np.einsum('nkl,nl->nk', Uq, channels)
        channels = np.einsum('nkl,nl->nk', U, channels)
    lam = np.sqrt(np.sum(abs(channels)**2, axis=1))
    C = np.zeros_like(channels)
    good = lam > np.finfo(float).tiny**.25
    C[good] = channels[good]/lam[good, None]
    return channels, deriv, lam, C


def split_actual_rate(G, Gdot):
    """Quotient rule for the ACTUAL projected Gdot, not the unprojected RHS."""
    lam = np.sqrt(np.sum(abs(G)**2, axis=-1))
    good = lam > np.finfo(float).tiny**.25
    C, Cdot = np.zeros_like(G), np.zeros_like(G)
    C[good] = G[good]/lam[good, None]
    ldot = np.real(np.sum(C.conj()*Gdot, axis=-1))
    Cdot[good] = (Gdot[good]-C[good]*ldot[good, None])/lam[good, None]
    return lam, C, ldot, Cdot, good
