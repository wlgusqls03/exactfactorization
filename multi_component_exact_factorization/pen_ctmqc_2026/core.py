"""EL/POL CTMQC for the 2026 PEN model; all propagation arrays are trajectories.

EL: X=(R,q_c), masses=(M,1), two quantum electronic coefficients.
POL: X=(R,), mass=M, 2*N_Fock quantum electron-photon coefficients.
No full wavefunction, no exact MCEF scalars, and no nuclear quantum potential
are inferred from this classical Wigner ensemble.

The quantum coefficients are integrated in the fixed diabatic/product basis.
The mean-field equation -i V(X)c is exactly the moving adiabatic equation
including NAC; the CT term is transformed from instantaneous eigenstates.
This avoids divisions by small gaps and arbitrary eigenvector sign derivatives.
"""
from dataclasses import dataclass
from itertools import permutations
import numpy as np
from ..pen_perspectives_2026.model import electronic, local_hamiltonian, photon_operators
from .quantum_momentum import coupled_terms


@dataclass(frozen=True)
class Settings:
    perspective: str = 'pol'
    ntraj: int = 10000
    seed: int = 2026
    algorithm: str = 'ctmqc'
    initial: str = 'paper'
    quantum_momentum: str = 'corrected'
    bandwidth_scale: float = 1.
    population_threshold: float = 1e-3
    max_norm_step_error: float = 1e-5
    intercept_limit_sigma: float = 10.

    def validate(self):
        if self.perspective not in ('el', 'pol') or self.algorithm not in ('ctmqc', 'ehrenfest'):
            raise ValueError('Invalid perspective/algorithm')
        if self.initial not in ('paper', 'bare') or self.quantum_momentum not in ('corrected', 'density'):
            raise ValueError('Invalid preparation/quantum momentum')
        if self.ntraj < 2 or int(self.ntraj) != self.ntraj or self.seed < 0 or int(self.seed) != self.seed:
            raise ValueError('Need integer ntraj>=2 and seed>=0')
        if not np.all(np.isfinite([self.bandwidth_scale, self.population_threshold, self.max_norm_step_error, self.intercept_limit_sigma])):
            raise ValueError('Nonfinite CTMQC settings')
        if self.bandwidth_scale <= 0 or not 0 <= self.population_threshold < .5 or self.max_norm_step_error <= 0 or self.intercept_limit_sigma <= 0:
            raise ValueError('Invalid bandwidth/threshold/norm tolerance')
        if self.quantum_momentum == 'density' and self.ntraj > 1024:
            raise ValueError('O(N^2) density-Q is a <=1024-trajectory reference; use corrected for production')


def electronic_gradient(c, R):
    R = np.asarray(R)
    grad = np.zeros((len(R), 2, 2))
    grad[:, 0, 0], grad[:, 1, 1] = c.k*(R-c.r1), c.k*(R-c.r2)
    grad[:, 0, 1] = grad[:, 1, 0] = -2*c.a*(R-c.r3)*c.b*np.exp(-c.a*(R-c.r3)**2)
    return grad


def hamiltonian(c, perspective, X):
    """V and analytic derivatives dV/dX [N,D,L,L], no finite-difference NAC."""
    R = X[:, 0]
    h, dip = electronic(c, R)
    grad = electronic_gradient(c, R)
    eye = np.eye(2)
    if perspective == 'el':
        q = X[:, 1]
        v = h+c.omega*c.g*q[:, None, None]*dip+.5*c.omega**2*q[:, None, None]**2*eye
        vr = grad+c.omega*c.g*c.z*q[:, None, None]*eye
        vq = c.omega*c.g*dip+c.omega**2*q[:, None, None]*eye
        if c.dse:
            v += .5*c.g**2*(dip@dip)
            vr += c.g**2*c.z*dip
        return v, np.stack((vr, vq), axis=1)
    if perspective != 'pol':
        raise ValueError('Unknown perspective')
    if c.dse:
        grad += c.g**2*c.z*dip
    vr = np.einsum('nij,ab->niajb', grad, np.eye(c.nf)).reshape(len(R), 2*c.nf, 2*c.nf)
    vr += c.omega*c.g*c.z*np.kron(eye, photon_operators(c)['q'])[None]
    return local_hamiltonian(c, R), vr[:, None]


def align_bands(energies, vectors, anchor):
    """Track eigenprojectors between RK stages, including exact g=0 crossings.

    Signs are irrelevant to the fixed-basis propagation. Only f_k's labels
    must follow the same projectors. Overlap assignment is exact for small
    (<=8-state) ambiguous blocks, Hungarian otherwise. Reported Pol populations
    are always re-sorted by energy and are NOT these internal tracked labels.
    """
    if anchor is None:
        return energies, vectors
    score = abs(np.einsum('nil,nij->nlj', anchor, vectors))**2
    order = score.argmax(axis=2)
    ambiguous = np.flatnonzero(np.any(np.diff(np.sort(order, axis=1), axis=1) == 0, axis=1))
    size = energies.shape[1]
    if len(ambiguous) and size <= 8:
        perm = np.array(list(permutations(range(size))))
        # Work trajectory by trajectory ONLY for genuinely ambiguous assignment.
        for i in ambiguous:
            merit = score[i, np.arange(size)[None], perm].sum(axis=1)
            order[i] = perm[np.argmax(merit)]
    elif len(ambiguous):
        from scipy.optimize import linear_sum_assignment
        for i in ambiguous:
            order[i] = linear_sum_assignment(-score[i])[1]
    return np.take_along_axis(energies, order, 1), np.take_along_axis(vectors, order[:, None, :], 2)


class Engine:
    def __init__(self, config, settings):
        config.grid()
        settings.validate()
        self.c, self.s = config, settings
        self.masses = np.array([config.mass, 1.] if settings.perspective == 'el' else [config.mass])
        self.bandwidths = np.array([config.sigma_r/np.sqrt(2), 1/np.sqrt(2*config.omega)][:len(self.masses)])*settings.bandwidth_scale

    def initial(self):
        """Paper Wigner widths, zero mean momentum, reproducible common R/P draws.

        paper: EL upper local CBO state (Eq.25), POL bare S1|0> (Eq.27).
        bare: bare S1 for both, for a matched comparison to our existing TDSE.
        These preparations are distinguished in provenance, never equated.
        """
        c, s = self.c, self.s
        rng = np.random.default_rng(s.seed)
        rn = rng.normal(size=(s.ntraj, 2))
        X = np.zeros((s.ntraj, len(self.masses)))
        P = X.copy()
        X[:, 0] = c.r2+c.sigma_r/np.sqrt(2)*rn[:, 0]
        P[:, 0] = rn[:, 1]/(np.sqrt(2)*c.sigma_r)
        if s.perspective == 'el':
            qp = rng.normal(size=(s.ntraj, 2))
            X[:, 1] = qp[:, 0]/np.sqrt(2*c.omega)
            P[:, 1] = qp[:, 1]*np.sqrt(c.omega/2)
        v, _ = hamiltonian(c, s.perspective, X)
        _, anchor = np.linalg.eigh(v)
        bare = np.linalg.eigh(electronic(c, X[:, 0])[0])[1][:, :, 1]
        if s.perspective == 'el':
            coeff = (anchor[:, :, 1] if s.initial == 'paper' else bare).astype(complex)
        else:
            coeff = np.zeros((s.ntraj, 2*c.nf), dtype=complex)
            coeff[:, ::c.nf] = bare
        f = np.zeros((s.ntraj, coeff.shape[1], len(self.masses)))
        return dict(X=X, P=P, coeff=coeff, integrated_forces=f, anchor=anchor)

    def evaluate(self, state, anchor=None):
        X, P, coeff, f = (state[k] for k in ('X', 'P', 'coeff', 'integrated_forces'))
        v, grad = hamiltonian(self.c, self.s.perspective, X)
        energy, vectors = np.linalg.eigh(v)
        energy, vectors = align_bands(energy, vectors, anchor)
        ad = np.einsum('nik,ni->nk', vectors, coeff)
        norms = np.sum(abs(ad)**2, axis=1)
        pop = abs(ad)**2/norms[:, None]
        surfaces_force = -np.einsum('nik,ndij,njk->nkd', vectors, grad, vectors)
        mean = np.einsum('nk,nkd->nd', pop, surfaces_force)
        mf = -np.einsum('ni,ndij,nj->nd', coeff.conj(), grad, coeff).real/norms[:, None]
        df = surfaces_force-mean[:, None]
        active = (pop > self.s.population_threshold) & (pop < 1-self.s.population_threshold)
        df *= active[:, :, None]
        if self.s.algorithm == 'ctmqc':
            rate, ct, info = coupled_terms(X, pop, f, self.masses, self.bandwidths,
                                           self.s.population_threshold, self.s.quantum_momentum, self.s.intercept_limit_sigma)
        else:
            rate, ct = np.zeros_like(pop), np.zeros_like(X)
            info = dict(q_pair_weighted_rms=np.zeros(len(self.masses)), q_pair_weighted_local_rms=np.zeros_like(X), q_disabled_directions=0, q_distant_directions=0,
                        q_attempted_directions=0, ct_population_rate=np.zeros(pop.shape[1]), coefficient_norm_rate_max=0.)
        dc = -1j*np.einsum('nij,nj->ni', v, coeff)+np.einsum('nik,nk->ni', vectors, rate*ad)
        deriv = dict(X=P/self.masses, P=mf+ct, coeff=dc, integrated_forces=df)
        detail = dict(energies=energy, vectors=vectors, population=pop, mean_energy=np.sum(pop*energy, axis=1),
                      mf_force=mf, ct_force=ct, rates=rate, active=active, **info)
        return deriv, detail

    def step(self, state):
        """Coupled RK4 with ensemble reevaluation at ALL four stages.

        One common step advances nuclei, photons (EL), coefficients and force
        integrals. No per-trajectory asynchronous update. Tiny numerical norm
        errors are recorded before normalization; excessive error stops the run.
        """
        dt, anchor = self.c.dt, state['anchor']
        keys = ('X', 'P', 'coeff', 'integrated_forces')
        k1, d1 = self.evaluate(state, anchor)
        shifted = lambda k, scale: {key: state[key]+scale*dt*k[key] for key in keys}
        k2, d2 = self.evaluate(shifted(k1, .5), anchor)
        k3, d3 = self.evaluate(shifted(k2, .5), anchor)
        k4, d4 = self.evaluate(shifted(k3, 1.), anchor)
        self.last_step_guard_counts = np.array([
            sum(d[key] for d in (d1, d2, d3, d4))
            for key in ('q_attempted_directions', 'q_disabled_directions', 'q_distant_directions')], dtype=np.int64)
        new = {key: state[key]+dt*(k1[key]+2*k2[key]+2*k3[key]+k4[key])/6 for key in keys}
        if not all(np.all(np.isfinite(v)) for v in new.values()):
            raise FloatingPointError('Nonfinite CTMQC state; reduce dt/check quantum momentum')
        norm = np.sum(abs(new['coeff'])**2, axis=1)
        error = float(np.max(abs(norm-1)))
        if error > self.s.max_norm_step_error:
            raise FloatingPointError(f'Unnormalized per-step coefficient norm error {error:g}; reduce dt')
        new['coeff'] /= np.sqrt(norm[:, None])
        energy, vectors = np.linalg.eigh(hamiltonian(self.c, self.s.perspective, new['X'])[0])
        _, new['anchor'] = align_bands(energy, vectors, anchor)
        pop = abs(np.einsum('nik,ni->nk', new['anchor'], new['coeff']))**2
        active = (pop > self.s.population_threshold) & (pop < 1-self.s.population_threshold)
        # Common-force subtraction is harmless for pair differences; resetting
        # inactive states is the documented CTMQC population-threshold policy.
        new['integrated_forces'] *= active[:, :, None]
        return new, error


def observables(engine, state):
    c, s = engine.c, engine.s
    _, d = engine.evaluate(state, state['anchor'])
    X, P, coeff = state['X'], state['P'], state['coeff']
    bare = np.linalg.eigh(electronic(c, X[:, 0])[0])[1]
    kinetic = np.sum(P*P/(2*engine.masses), axis=1)
    norms = np.sum(abs(coeff)**2, axis=1)
    if s.perspective == 'pol':
        elph = coeff.reshape(s.ntraj, 2, c.nf)
        bo = np.sum(abs(np.einsum('iek,ien->ikn', bare, elph))**2, axis=2)
        _, polv = np.linalg.eigh(local_hamiltonian(c, X[:, 0]))
        pol = abs(np.einsum('nik,ni->nk', polv, coeff))**2
        ops = photon_operators(c)
        ev = lambda op: np.einsum('ien,nm,iem->i', elph.conj(), op, elph).real
        number, q2, p2 = ev(ops['number']), ev(ops['q2']), ev(ops['p2'])
        qmean, pmean = ev(ops['q']), ev(ops['p'])
    else:
        bo = abs(np.einsum('nik,ni->nk', bare, coeff))**2
        cbov = np.linalg.eigh(hamiltonian(c, 'el', X)[0])[1]
        cbop = abs(np.einsum('nik,ni->nk', cbov, coeff))**2
        pol = None  # No quantum electron-photon vector in EL; do not fake it.
        qmean, pmean = X[:, 1], P[:, 1]
        q2, p2 = qmean*qmean, pmean*pmean
        number = .5*(c.omega*q2+p2/c.omega)-.5
    se = lambda v: float(np.std(v, ddof=1)/np.sqrt(s.ntraj))
    kernel = np.exp(-.5*((X[:, 0]-4)/engine.bandwidths[0])**2)/(np.sqrt(2*np.pi)*engine.bandwidths[0])
    row = dict(norm=float(norms.mean()), coefficient_norm_max_error=float(np.max(abs(norms-1))),
               energy=float(np.mean(kinetic+d['mean_energy'])), energy_se=se(kinetic+d['mean_energy']),
               kinetic_R=float(np.mean(P[:, 0]**2/(2*c.mass))),
               photon_number=float(number.mean()), photon_number_se=se(number),
               photon_energy=float(c.omega*(number.mean()+.5)),
               q_mean=float(qmean.mean()), p_mean=float(pmean.mean()), q2=float(q2.mean()), p2=float(p2.mean()),
               bo_population=bo.mean(axis=0).tolist(), basis_population=d['population'].mean(axis=0).tolist(),
               R_mean=float(X[:, 0].mean()), R_sd=float(X[:, 0].std()),
               product_R_ge_4=float(np.mean(X[:, 0] >= 4)),
               signed_flux_R4=float(np.mean(kernel*P[:, 0]/c.mass)),
               edge_probability=float(np.mean((X[:, 0] < c.rmin)|(X[:, 0] > c.rmax))),
               mf_force_rms=np.sqrt(np.mean(d['mf_force']**2, axis=0)).tolist(),
               ct_force_rms=np.sqrt(np.mean(d['ct_force']**2, axis=0)).tolist(),
               q_pair_weighted_rms=d['q_pair_weighted_rms'].tolist(),
               q_disabled_directions=d['q_disabled_directions'], q_distant_directions=d['q_distant_directions'], q_attempted_directions=d['q_attempted_directions'],
               ct_population_rate=d['ct_population_rate'].tolist(),
               coefficient_norm_rate_max=d['coefficient_norm_rate_max'])
    if pol is not None:
        row['pol_population'] = pol.mean(axis=0).tolist()
        row['top_fock_population'] = float(np.sum(abs(elph[:, :, -1])**2)/s.ntraj)
    else:
        row['cbo_population'] = cbop.mean(axis=0).tolist()
    return row
