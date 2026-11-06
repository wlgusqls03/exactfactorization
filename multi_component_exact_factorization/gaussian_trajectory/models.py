"""Fixed-primitive internal Hamiltonians for the explicitly chosen MTE B1.

The common array is G[trajectory, q, electronic_primitive].  A PEN model uses
the existing 2026 two-diabatic-state Hamiltonian, but a REAL photon grid, not
the old two-Fock propagation.  ``mcef`` instead retains all electronic grid
points of the inherited electron--proton--heavy model.  These are matched
Hamiltonian/force providers, not CTMQC or a closed exact-EF trajectory force.
"""
from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np
from scipy.linalg import eigh
from scipy.special import erf

from ..pen_perspectives_2026.model import electronic
from ..pen_perspectives_2026.resonance import paper_config
from ..pen_ctmqc_2026.core import hamiltonian
from multi_component_exact_factorization_cavity.core import (
    frozen_potential, sine_kinetic)


@dataclass(frozen=True)
class ModelConfig:
    name: str = 'pen-nonadiabatic'
    nq: int = 64
    qmin: float = -14.
    qmax: float = 14.
    g: float = .01
    dse: bool = False
    omega: Optional[float] = None
    resonance: str = 'on'
    # MCEF-only numerical sizes.  Coarse defaults are a pilot, not a converged
    # replacement of the historical nx=359/nq=750 full quantum calculation.
    nx: int = 33
    xmin: float = -22.
    xmax: float = 30.
    excitation: int = 1


def erf_inverse_gradient(distance, radius):
    """Derivative of inherited erf(|r|/s)/|r|, regular including r=0.

    The small-r series prevents subtraction of two almost equal 1/r terms.
    This is a derivative of the physical kernel, not a spatial-fit force.
    """
    d = np.asarray(distance, dtype=float)
    s = float(radius)
    if not np.isfinite(s) or s <= 0:
        raise ValueError('Need a positive finite smoothing radius')
    z = d/s
    result = np.empty_like(d)
    small = abs(z) < 1e-3
    result[small] = (2/np.sqrt(np.pi)/s**2
                     * (-2*z[small]/3+2*z[small]**3/5-z[small]**5/7))
    a = z[~small]
    result[~small] = (2*a*np.exp(-a*a)/np.sqrt(np.pi)-erf(a))/(s*s*a*a)
    return result


class InternalModel:
    """Current-state action, operator derivative force and initial wavepacket.

    q uses a periodic FFT kinetic and rectangle quadrature.  MCEF x uses the
    existing Dirichlet DST-I kinetic.  Electronic primitive coefficients in
    MCEF equal sqrt(dx)*Phi(x), so sum_k |C_k|^2=1 without a hidden dx weight.
    Heavy trap energy is external and is NOT counted again in action().
    """
    def __init__(self, config=ModelConfig()):
        self.config = config
        self.name = config.name
        if self.name not in ('pen-nonadiabatic', 'pen-rabi', 'mcef'):
            raise ValueError('Supported models: pen-nonadiabatic, pen-rabi, mcef')
        if (isinstance(config.nq, bool) or config.nq != int(config.nq)
                or config.nq < 8 or config.qmax <= config.qmin
                or not np.all(np.isfinite([config.qmin, config.qmax, config.g]))):
            raise ValueError('Need finite ordered q bounds and integer nq>=8')
        if not isinstance(config.dse, bool):
            raise ValueError('dse must be a boolean, not a string')
        if config.excitation not in (0, 1):
            raise ValueError('Initial electronic excitation must be 0 or 1')
        self.q = np.linspace(config.qmin, config.qmax, config.nq, endpoint=False)
        self.dq = float(self.q[1]-self.q[0])
        self.quadrature = np.full(config.nq, self.dq)
        self.kq = 2*np.pi*np.fft.fftfreq(config.nq, self.dq)
        self.gauge = 'B1 material Schrodinger phase; bridge positive only'
        self.is_photon = self.name.startswith('pen-')
        if self.is_photon:
            self.c = paper_config(self.name[4:], config.resonance,
                                 g=config.g, dse=config.dse)
            if config.omega is not None:
                self.c = replace(self.c, omega=float(config.omega))
                self.c.grid()
            self.mass_R, self.mass_q = self.c.mass, 1.
            self.nk = 2
            self.omega = self.c.omega
            self.initial_R_center = self.c.r2
            self.initial_R_sigma = self.c.sigma_r/np.sqrt(2)
            self.bridge_center, self.bridge_sigma = 0., 1/np.sqrt(2*self.omega)
            self.basis_convention = 'fixed two-state diabatic electronic basis; real photon q grid'
        else:
            if config.nx != int(config.nx) or config.nx < 4 or config.xmax <= config.xmin:
                raise ValueError('Need integer nx>=4 and ordered electronic bounds')
            if not np.all(np.isfinite([config.xmin, config.xmax])):
                raise ValueError('Nonfinite electronic bounds')
            self.mass_R, self.mass_q = 12000., 1836.
            self.nk, self.omega = config.nx, None
            self.dx = (config.xmax-config.xmin)/(config.nx+1)
            self.x = config.xmin+self.dx*np.arange(1, config.nx+1)
            self.Tx, _ = sine_kinetic(config.nx, self.dx)
            self.initial_R_center, self.initial_R_sigma = 9.5, .13650604313545334
            self.bridge_center, self.bridge_sigma = -4., .3208418789467482
            self.basis_convention = 'fixed electronic Dirichlet grid, sqrt(dx)-weighted coefficients'
        self.initial_P_mean = 0.
        self.initial_P_sigma = 1/(2*self.initial_R_sigma)
        self.kinetic_q = self.kq**2/(2*self.mass_q)

    def _inputs(self, R, G=None):
        R = np.asarray(R, dtype=float)
        if R.ndim != 1 or len(R) == 0 or not np.all(np.isfinite(R)):
            raise ValueError('R must be a nonempty finite trajectory vector')
        if G is not None:
            G = np.asarray(G, dtype=np.complex128)
            if G.shape != (len(R), len(self.q), self.nk) or not np.all(np.isfinite(G)):
                raise ValueError('G must be finite with shape [trajectory,q,electronic_primitive]')
        return R, G

    def potential(self, R):
        """Local electronic matrix [I,q,k,l] (PEN) or diagonal [I,q,x] (MCEF)."""
        R, _ = self._inputs(R)
        if self.is_photon:
            X = np.column_stack((np.repeat(R, len(self.q)), np.tile(self.q, len(R))))
            v, gradient = hamiltonian(self.c, 'el', X)
            return v.reshape(len(R), len(self.q), 2, 2)
        # This reuses the original interaction; no BO projection in propagation.
        return np.array([frozen_potential(self.x, self.q,
                         dict(heavy_fixed=float(r), heavy_trap_alpha=0.)) for r in R])

    def potential_gradient(self, R):
        R, _ = self._inputs(R)
        if self.is_photon:
            X = np.column_stack((np.repeat(R, len(self.q)), np.tile(self.q, len(R))))
            _, gradient = hamiltonian(self.c, 'el', X)
            return gradient[:, 0].reshape(len(R), len(self.q), 2, 2)
        rr = R[:, None, None]
        if np.any(R == -9.5):
            raise ValueError('Heavy trajectory hit the bare left-ion Coulomb pole')
        return (-np.sign(rr+9.5)/abs(rr+9.5)**2
                -erf_inverse_gradient(rr-self.x[None, None, :], 4.)
                -erf_inverse_gradient(self.q[None, :, None]-rr, 1.))

    def external_potential(self, R):
        R, _ = self._inputs(R)
        return np.zeros_like(R) if self.is_photon else .03*(R-9.5)**2

    def derivative_q(self, G, order=1):
        G = np.asarray(G)
        if G.ndim != 3 or G.shape[1] != len(self.q):
            raise ValueError('Expected [trajectory,q,component] for q derivative')
        return np.fft.ifft(np.fft.fft(G, axis=1)
                           *(1j*self.kq[None, :, None])**order, axis=1)

    def action(self, R, G):
        R, G = self._inputs(R, G)
        kinetic = np.fft.ifft(np.fft.fft(G, axis=1)
                              * self.kinetic_q[None, :, None], axis=1)
        v = self.potential(R)
        if self.is_photon:
            return kinetic+np.einsum('iqkl,iql->iqk', v, G)
        return kinetic+np.einsum('kl,iql->iqk', self.Tx, G)+v*G

    def force(self, R, G):
        """B1 full-internal mean force; input is not silently normalized."""
        R, G = self._inputs(R, G)
        dv = self.potential_gradient(R)
        action = np.einsum('iqkl,iql->iqk', dv, G) if self.is_photon else dv*G
        value = -np.einsum('iqk,iqk,q->i', G.conj(), action, self.quadrature).real
        if not self.is_photon:
            value -= .06*(R-9.5)
        return value

    def initial(self, R):
        """Existing bare excited electronic preparation and real bridge Gaussian."""
        states = self.initial_coefficients(R)
        envelope = np.exp(-.25*((self.q-self.bridge_center)/self.bridge_sigma)**2)
        envelope /= np.sqrt(np.sum(envelope**2*self.quadrature))
        return envelope[None, :, None]*states

    def initial_coefficients(self, R):
        """Pointwise-normalized C without division by underflowed bridge tails.

        PEN S1 signs use a deterministic largest-component-positive convention.
        MCEF initializes BO states only; q and ordered-R signs are transported
        exactly as in core.local_electronic_basis (never propagated states).
        All subsequent evolution is in the fixed primitive representation.
        """
        R, _ = self._inputs(R)
        if self.is_photon:
            vectors = np.linalg.eigh(electronic(self.c, R)[0])[1][:, :, self.config.excitation]
            pivot = np.argmax(abs(vectors), axis=1)
            vectors *= np.where(vectors[np.arange(len(R)), pivot] < 0, -1., 1.)[:, None]
            return np.broadcast_to(vectors[:, None, :], (len(R), len(self.q), self.nk)).astype(complex).copy()
        v = self.potential(R)
        states = np.empty((len(R), len(self.q), self.nk))
        previous = None
        for i in np.argsort(R, kind='stable'):
            for j in range(len(self.q)):
                _, vec = eigh(self.Tx+np.diag(v[i, j]),
                              subset_by_index=[self.config.excitation, self.config.excitation])
                state = vec[:, 0]
                reference = states[i, j-1] if j else (None if previous is None else states[previous, 0])
                sign = state[np.argmax(abs(state))] if reference is None else np.dot(reference, state)
                states[i, j] = state*(-1. if sign < 0 else 1.)
            previous = i
        return states.astype(complex)

    def observables(self, R, G):
        """Per-trajectory observables, retaining raw norm defects and phases."""
        R, G = self._inputs(R, G)
        density = np.sum(abs(G)**2, axis=2)
        norm = density@self.quadrature
        if np.any(norm <= 0):
            raise ValueError('Cannot report observables of a zero-norm state')
        dG = self.derivative_q(G)
        p2 = np.einsum('iqk,iqk,q->i', dG.conj(), dG, self.quadrature).real/norm
        energy = np.einsum('iqk,iqk,q->i', G.conj(), self.action(R, G), self.quadrature).real/norm
        result = dict(norm=norm, internal_energy=energy,
                      external_heavy_energy=self.external_potential(R),
                      bridge_kinetic=p2/(2*self.mass_q),
                      q_mean=(density*self.q[None])@self.quadrature/norm,
                      q2=(density*self.q[None]**2)@self.quadrature/norm,
                      p_mean=np.einsum('iqk,iqk,q->i', G.conj(), -1j*dG, self.quadrature).real/norm,
                      p2=p2, primitive_population=np.einsum('iqk,q->ik', abs(G)**2, self.quadrature)/norm[:, None],
                      q_edge_probability=np.sum(density[:, [0, -1]], axis=1)*self.dq/norm)
        if self.is_photon:
            vectors = np.linalg.eigh(electronic(self.c, R)[0])[1]
            amplitudes = np.einsum('ikl,iqk->iql', vectors, G)
            result['bo_population'] = np.einsum('iqk,q->ik', abs(amplitudes)**2, self.quadrature)/norm[:, None]
            result['photon_number'] = .5*(self.omega*result['q2']+p2/self.omega)-.5
            result['photon_energy'] = .5*(self.omega**2*result['q2']+p2)
        return result

    def metadata(self):
        return dict(configuration=asdict(self.config), basis=self.basis_convention,
                    gauge=self.gauge, mass_R=self.mass_R, mass_q=self.mass_q,
                    omega=self.omega, q_operator='periodic FFT kinetic; rectangle quadrature',
                    initial_R_center=self.initial_R_center, initial_R_sigma=self.initial_R_sigma,
                    initial_P_mean=self.initial_P_mean, initial_P_sigma=self.initial_P_sigma,
                    bridge_center=self.bridge_center, bridge_sigma=self.bridge_sigma,
                    q_boundary_policy='report endpoint probability; no absorbing potential',
                    model_source=('pen_perspectives_2026/model.py:75-84; pen_ctmqc_2026/core.py:56-70'
                                  if self.is_photon else 'core.py:379-393; cavity/core.py:75-94'),
                    # Do not copy Config.nr/nf/dt: those are numerical settings
                    # of the old TDSE/Fock code, NOT this q-grid trajectory run.
                    physical_parameters=({key: getattr(self.c, key) for key in
                        ('mass', 'omega', 'g', 'dse', 'k', 'a', 'b', 'r1', 'r2',
                         'r3', 'delta', 'z', 'mu12')} if self.is_photon else dict(
                        left_fixed=-9.5, heavy_trap_center=9.5, heavy_trap_alpha=.03,
                        erf_r_lx=3.1, erf_r_qx=5., erf_r_Rx=4., erf_r_qR=1.)),
                    limitation=('Existing two-diabatic-state matter model; q-grid photon, not two Fock states.'
                                if self.is_photon else 'Real electronic grid. Pilot grids and MTE heavy motion are not the old converged quantum calculation.'),
                    historical_initial_source=(None if self.is_photon else
                        'results/20260916/erf_fullgrid_BOcurvature_compact_qR_analysisBO2_100fs/multi_component_discrete_tdse_gpu.npz:args'))


def make_model(config=ModelConfig()):
    return InternalModel(config)
