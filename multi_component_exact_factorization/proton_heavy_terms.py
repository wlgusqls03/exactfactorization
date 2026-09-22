"""PG proton-heavy diagnostic, using historical central-five operators.

These are differential diagnostics of cached fields, NOT an exact native-link
or spectral TDSE factorization. Keep product-rule, time-sampling and cache
closure defects explicit; never absorb them in a displayed physical term.
Axes: (q,R); atomic units throughout; Lambda and chi are positive real.
"""
from dataclasses import dataclass
import numpy as np
from .core import derivative, covariant_square, AU_PER_FS


@dataclass(frozen=True)
class TermConfig:
    density_floor: float = 1e-3
    heavy_floor: float = 1e-12
    q_split: float = 0.
    connection_location: str = 'bond'

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.density_floor, self.heavy_floor, self.q_split)):
            raise ValueError('Finite thresholds/q_split required')
        if min(self.density_floor, self.heavy_floor) <= 0:
            raise ValueError('Density thresholds must be positive')
        if self.connection_location not in ('bond', 'site'):
            raise ValueError('Unknown connection location')


def conditional_density(rho, heavy, floor):
    return np.divide(rho, heavy[None, :], out=np.full_like(rho, np.nan, dtype=float),
                     where=np.isfinite(heavy[None, :]) & (heavy[None, :] > floor))


def time_rate(get_frame, times_fs, frame, stride=1):
    """Same endpoint/secant and nonuniform central rule as tdse_report.

    Operates on <=3 frames, unlike a whole-trajectory np.gradient allocation.
    stride=2 is an independent coarse saved-time sampling diagnostic.
    """
    t = np.asarray(times_fs, float)*AU_PER_FS
    if len(t) < 2:
        return np.full_like(get_frame(frame), np.nan, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('Strictly increasing finite saved times required')
    left, right = max(0, frame-stride), min(len(t)-1, frame+stride)
    if frame in (0, len(t)-1):
        return (get_frame(right)-get_frame(left))/(t[right]-t[left])
    hl, hr = t[frame]-t[left], t[right]-t[frame]
    return (-hr*get_frame(left)/(hl*(hl+hr))+(hr-hl)*get_frame(frame)/(hl*hr)
            +hl*get_frame(right)/(hr*(hl+hr)))


def frame_terms(rho, heavy, a, b, alpha, dq, dR, proton_mass, heavy_mass,
                density_rate, config=TermConfig()):
    """T1..T8 are ACTIONS (Ha*a0^-1/2), not action/Lambda ratios.

    U_operator uses the existing Hermitian anticommutator covariant_square.
    Expanded T1..T8 use the supplied continuum product rule, so their sum is
    separately labeled U_total_expanded. A residual records their difference.
    """
    rho, heavy = np.asarray(rho, float), np.asarray(heavy, float)
    if rho.ndim != 2 or heavy.shape != (rho.shape[1],) or min(rho.shape) < 5:
        raise ValueError('Need (q,R) density and >=5 sites per periodic axis')
    if min(dq, dR, proton_mass, heavy_mass) <= 0:
        raise ValueError('Positive spacings and physical masses required')
    if np.shape(a) != rho.shape or np.shape(b) != rho.shape or np.shape(alpha) != heavy.shape:
        raise ValueError('Connection/density shape mismatch')
    if np.shape(density_rate) != rho.shape:
        raise ValueError('Density time derivative shape mismatch')
    if np.any(rho < 0) or np.any(heavy < 0):
        raise ValueError('Negative probability density')
    a, b, alpha = (np.asarray(v, float) for v in (a, b, alpha))
    if config.connection_location == 'bond':
        a = (a+np.roll(a, 1, axis=0))/2
        b = (b+np.roll(b, 1, axis=1))/2
        alpha = (alpha+np.roll(alpha, 1))/2
    den = conditional_density(rho, heavy, config.heavy_floor)
    lam, chi = np.sqrt(den), np.sqrt(heavy)
    al = np.broadcast_to(alpha, rho.shape)
    delta = b-al
    D = lambda v: derivative(v, dR, axis=1)
    dl = D(lam)
    logchi = np.divide(derivative(chi, dR, 0), chi,
                      out=np.full_like(chi, np.nan), where=heavy > config.heavy_floor)[None, :]
    M = heavy_mass
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        fields = dict(lambda_density=den, joint_density=rho, a=a, b=b, alpha=al,
                      delta=delta, delta_squared=delta**2,
                      T1=-derivative(lam, dR, 1, order=2)/(2*M),
                      T2=-logchi*dl/M, T3=delta**2*lam/(2*M), T4=al*delta*lam/M,
                      T5=-1j*delta*dl/M, T6=-.5j*D(delta)*lam/M,
                      T7=-1j*al*dl/M, T8=-1j*logchi*delta*lam/M)
        fields['U_quad'] = sum(fields[k] for k in ('T1','T3','T5','T6'))
        fields['U_lin'] = sum(fields[k] for k in ('T2','T4','T7','T8'))
        fields['U_total'] = fields['U_quad']+fields['U_lin']
        pD = -1j*dl+delta*lam
        fields['U_quad_operator'] = covariant_square(lam, delta, dR, 1, +1)/(2*M)
        fields['U_lin_operator'] = (al-1j*logchi)*pD/M
        fields['U_operator'] = fields['U_quad_operator']+fields['U_lin_operator']
        fields['residual_product_rule'] = fields['U_operator']-fields['U_total']
        fields['J_rel'] = rho*delta/M
        fields['div_J_rel'] = D(fields['J_rel'])
        fields['S_q'] = -derivative(a*den/proton_mass, dq, 0)
        fields['S_adv'] = -al*D(den)/M
        fields['S_rel'] = -conditional_density(fields['div_J_rel'], heavy, config.heavy_floor)
        fields['S_U'] = fields['S_adv']+fields['S_rel']
        fields['S_U_operator'] = 2*np.imag(lam*fields['U_operator'])
        fields['S_U_expanded'] = 2*np.imag(lam*fields['U_total'])
        fields['density_dt'] = np.asarray(density_rate)
        fields['residual_density'] = density_rate-fields['S_q']-fields['S_U']
        fields['residual_source_operator'] = fields['S_U_operator']-fields['S_U']
        fields['residual_source_expanded'] = fields['S_U_expanded']-fields['S_U']
    valid = (rho >= config.density_floor) & (heavy[None, :] > config.heavy_floor)
    for value in fields.values():
        valid &= np.isfinite(value)
    return dict(fields=fields, valid=valid, heavy_density=heavy,
                alpha_line=alpha, lambda_amplitude=lam)


def summarize_frame(result, dq, dR):
    """Common support rho-weighted RMS of raw actions/sources; no tail filling."""
    fields, valid = result['fields'], result['valid']
    weight = np.where(valid, fields['joint_density'], 0)
    total = weight.sum()
    rms = {key: (float(np.sqrt(np.sum(weight*np.abs(np.where(valid,value,0))**2)/total))
                 if total > 0 else None) for key,value in fields.items()}
    peak = {key: float(np.max(abs(value[valid]))) if valid.any() else None
            for key,value in fields.items()}
    return dict(support_mass=float(total*dq*dR), rms=rms, max_abs=peak)


def peak_integrals(result, q, dq, split):
    """Integrate FULL right half, not the display mask; undefined columns stay NaN.

    A coordinate partition, not automatic peak tracking. No invented source
    values outside support. Column probability normalization is also recorded.
    """
    if not q[0] < split < q[-1]:
        raise ValueError('q_split must be strictly inside the proton grid')
    fields = result['fields']
    right = q > split
    out = {key: np.sum(fields[key][right], axis=0)*dq
           for key in ('lambda_density','density_dt','S_q','S_adv','S_rel','residual_density')}
    out['P_right'] = out.pop('lambda_density')
    out['conditional_norm'] = fields['lambda_density'].sum(axis=0)*dq
    out['P_left'] = np.sum(fields['lambda_density'][~right], axis=0)*dq
    return out
