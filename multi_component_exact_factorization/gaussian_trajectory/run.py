"""Isolated B1 MTE / MTE+Gaussian pilot, with checkpoint and automatic reports.

Only R,P are classical samples. q is always a common quantum grid. No exact
TDSE future data, independent classical q trajectories, or fitted scalar EF
potentials enter this driver. RK4 re-evaluates the complete state every stage.
MLS and heavy-density estimators are diagnostics, NOT extra MTE force terms.
"""
import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import DOP853

from .gaussian import build_bridge, pack_parameters, split_and_project
from .heavy_density import reconstruct_heavy_density, reconstruct_kde_density
from .mls import MLSConfig
from .reference import require_backend

AU_FS = .024188843265857
GAUGE = 'common fixed-primitive Schrodinger phase; positive bridge only; chi not reconstructed'


class StateGuardError(FloatingPointError):
    """A rejected candidate, with evidence distinct from the last checkpoint."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass
class RunConfig:
    model: dict = field(default_factory=dict)
    method: str = 'mte'
    compression: str = 'gaussian'
    components: int = 1
    ntraj: int = 12
    sample_weights: list = None
    seed: int = 2026
    dt_au: float = .02
    end_fs: float = .1
    every_fs: float = .025
    support_relative: float = 0.
    max_excluded_mass: float = 1e-8
    rank_rtol: float = 1e-12
    rank_policy: str = 'raise'
    stage_pnc_tolerance: float = .01
    accepted_norm_tolerance: float = 1e-5
    edge_mass_tolerance: float = 1e-6
    integrator: str = 'dop853'
    rtol: float = 1e-10
    atol: float = 1e-12
    max_rhs_evaluations_per_segment: int = 100000
    sphere_max_auxiliary_defect: float = .05
    sphere_tangent_tolerance: float = 1e-10
    fit_center_separation: float = .65
    mls: dict = field(default_factory=lambda: dict(neighbors=7, multistream_policy='report'))
    heavy_density: str = 'state_resolved'
    heavy_width_floor: float = None
    kde_bandwidth: float = None

    def validate(self):
        require_backend(self.method)
        if self.compression not in ('native', 'gaussian'):
            raise ValueError('compression must be native or gaussian')
        if self.integrator not in ('rk4', 'dop853', 'sphere-rk4') or min(self.rtol, self.atol) <= 0:
            raise ValueError('Select rk4, dop853 or sphere-rk4 with positive tolerances')
        if self.integrator == 'sphere-rk4' and self.compression != 'gaussian':
            raise ValueError('sphere-rk4 requires Gaussian C,z state; use RK4 for native G')
        if (not np.isfinite(self.sphere_max_auxiliary_defect)
                or not 0 < self.sphere_max_auxiliary_defect < 1
                or not np.isfinite(self.sphere_tangent_tolerance)
                or self.sphere_tangent_tolerance <= 0):
            raise ValueError('Invalid homogeneous-coordinate integrator guards')
        if self.ntraj < 3 or int(self.ntraj) != self.ntraj or self.components < 1 or int(self.components) != self.components:
            raise ValueError('Need integer ntraj>=3 and components>=1')
        for value in (self.dt_au, self.end_fs, self.every_fs, self.stage_pnc_tolerance,
                      self.accepted_norm_tolerance, self.edge_mass_tolerance, self.fit_center_separation):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Time steps, tolerances and initialization separation must be positive')
        if self.heavy_density not in ('state_resolved', 'kde'):
            raise ValueError('Unknown heavy density estimator')
        if self.heavy_density == 'kde' and (self.kde_bandwidth is None or self.kde_bandwidth <= 0):
            raise ValueError('KDE requires an explicit positive bandwidth')
        MLSConfig(**self.mls).validate()
        if self.sample_weights is not None:
            weights = np.asarray(self.sample_weights, float)
            if weights.shape != (self.ntraj,) or np.any(weights < 0) or not np.all(np.isfinite(weights)) or abs(weights.sum()-1) > 1e-12:
                raise ValueError('sample_weights must be normalized nonnegative eta_I')
        if self.max_rhs_evaluations_per_segment <= 0:
            raise ValueError('Need a positive RHS evaluation limit')


def fit_initial_bridge(q, quadrature, rho, components, separation=.65):
    """Deterministic distinct mixture fit to SAME physical density (S3).

    For K>1 nonoverlapping center windows prevent identical Gaussian columns.
    This constrained initialization is a numerical policy, not exact equality
    to the original state. Its amplitude/density errors are always returned.
    """
    rho = np.asarray(rho, float)
    rho = rho/(quadrature @ rho)
    mean = quadrature @ (q*rho)
    sigma = np.sqrt(quadrature @ ((q-mean)**2*rho))
    K = components
    offsets = np.linspace(-separation, separation, K) if K > 1 else np.zeros(1)
    centers = mean+sigma*offsets
    widths = np.full(K, sigma*np.sqrt(max(.3, 1-np.mean(offsets**2))))
    z = pack_parameters(np.full(K, 1/K), centers, widths, coordinates='transformed')
    def bridge(z):
        return build_bridge(q, quadrature, z, components=K, coordinates='transformed')
    def residual(z):
        return np.sqrt(quadrature)*(bridge(z).amplitude-np.sqrt(rho))
    if K > 1:
        half_window = .2*np.min(np.diff(centers))
        low = np.r_[np.full(K-1, -8.), centers-half_window, np.full(K, np.log(.2*sigma))]
        high = np.r_[np.full(K-1, 8.), centers+half_window, np.full(K, np.log(2*sigma))]
        fit = least_squares(residual, z, bounds=(low, high), max_nfev=300,
                            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if not fit.success:
            raise ValueError('Initial distinct-mixture fit did not converge: '+fit.message)
        z = fit.x
    b = bridge(z)
    # Match select_support's representable-density domain. An underflowed
    # Gaussian tail is not a zero singular value and must never yield 0/0.
    # No density floor or modification of the physical initial state is used.
    represented = (b.rho > 0) & (b.amplitude > 0)
    tangent = (np.sqrt(quadrature[represented, None])*b.jacobian[represented]
               / b.amplitude[represented, None])
    rank = np.linalg.matrix_rank(tangent)
    return z, dict(amplitude_l2_error=float(np.linalg.norm(residual(z))),
                   density_l1_error=float(quadrature @ abs(b.rho-rho)),
                   tangent_rank=int(rank), parameter_count=int(len(z)),
                   rank_excluded_underflow_points=int(np.count_nonzero(~represented)),
                   initialization='moment Gaussian' if K == 1 else 'distinct constrained amplitude fit',
                   center_separation_sigma=separation)


class Engine:
    """B1 paired physical force/RHS with optional literal P3/P8 projection."""
    def __init__(self, config):
        from .models import InternalModel, ModelConfig
        config.validate()
        self.c = config
        self.model = InternalModel(ModelConfig(**config.model))
        self.eta = np.full(config.ntraj, 1/config.ntraj) if config.sample_weights is None else np.asarray(config.sample_weights)
        self.audit = dict(max_accepted_norm_pnc_error=0., max_edge_mass=0.,
                          max_projection_rate_residual=0., max_gaussian_condition=0.,
                          max_excluded_mass=0., minimum_gaussian_rank=3*config.components-1)

    def initial(self):
        m, c = self.model, self.c
        draws = np.random.default_rng(c.seed).normal(size=(c.ntraj, 2))
        R = m.initial_R_center+m.initial_R_sigma*draws[:, 0]
        P = m.initial_P_mean+m.initial_P_sigma*draws[:, 1]
        G = m.initial(R)
        if c.compression == 'native':
            return dict(R=R, P=P, G=G), []
        z, C, fits = [], [], []
        initial_C = m.initial_coefficients(R)
        for Gi, Ci in zip(G, initial_C):
            rho = np.sum(abs(Gi)**2, axis=1)
            zi, audit = fit_initial_bridge(m.q, m.quadrature, rho, c.components, c.fit_center_separation)
            z.append(zi); C.append(Ci); fits.append(audit)
        return dict(R=R, P=P, z=np.array(z), C=np.array(C)), fits

    def reconstruct(self, state):
        if self.c.compression == 'native':
            return state['G'], []
        bridges = [build_bridge(self.model.q, self.model.quadrature, z,
                               components=self.c.components, coordinates='transformed') for z in state['z']]
        return np.array([b.amplitude[:, None]*C for b, C in zip(bridges, state['C'])]), bridges

    def evaluate(self, state):
        G, bridges = self.reconstruct(state)
        G_rhs_eom = -1j*self.model.action(state['R'], G)
        heavy_force = self.model.force(state['R'], G)
        derivative = dict(R=state['P']/self.model.mass_R, P=heavy_force)
        if not bridges:
            derivative['G'] = G_rhs_eom
            return derivative, []
        rates = [split_and_project(F, C, b, support_relative=self.c.support_relative,
                 max_excluded_mass=self.c.max_excluded_mass, rank_rtol=self.c.rank_rtol,
                 rank_policy=self.c.rank_policy, pnc_tolerance=self.c.stage_pnc_tolerance)
                 for F, C, b in zip(G_rhs_eom, state['C'], bridges)]
        for rate in rates:
            self.audit['max_projection_rate_residual'] = max(self.audit['max_projection_rate_residual'], rate.audit['actual_vs_eom_rate_l2'])
            self.audit['max_gaussian_condition'] = max(self.audit['max_gaussian_condition'], rate.audit['condition_L'])
            self.audit['minimum_gaussian_rank'] = min(self.audit['minimum_gaussian_rank'], rate.audit['rank'])
            self.audit['max_excluded_mass'] = max(self.audit['max_excluded_mass'], rate.audit['excluded_mass'])
        derivative.update(z=np.array([r.parameter_dot for r in rates]), C=np.array([r.C_dot for r in rates]))
        return derivative, [r.audit for r in rates]

    def step(self, state):
        dt = self.c.dt_au
        if self.c.integrator == 'sphere-rk4':
            from .sphere_integrator import sphere_rk4_step
            new, audit = sphere_rk4_step(state, dt, self.evaluate,
                max_auxiliary_defect=self.c.sphere_max_auxiliary_defect,
                tangent_tolerance=self.c.sphere_tangent_tolerance)
            self.last_integrator_audit = audit
            for name in ('max_auxiliary_stage_norm_error', 'candidate_auxiliary_norm_error',
                         'max_raw_tangent_rate_error', 'max_relative_tangent_rate_error'):
                key = 'sphere_'+name
                self.audit[key] = max(self.audit.get(key, 0.), audit[name])
            self.guard(new)
            return new
        shift = lambda rate, factor: {k: state[k]+factor*dt*rate[k] for k in state}
        k1, _ = self.evaluate(state)
        k2, _ = self.evaluate(shift(k1, .5))
        k3, _ = self.evaluate(shift(k2, .5))
        k4, _ = self.evaluate(shift(k3, 1.))
        new = {k: state[k]+dt*(k1[k]+2*k2[k]+2*k3[k]+k4[k])/6 for k in state}
        self.guard(new)
        return new

    def advance(self, state, start_step, end_step):
        """Adaptive DOP853; guard EVERY accepted step, no state renormalization.

        Real packing keeps Gaussian parameters real while retaining both parts
        of every complex coefficient. Restart reinitializes the adaptive solver
        at a saved state; agreement is tolerance-based (RK4 is bit reproducible).
        """
        if self.c.integrator in ('rk4', 'sphere-rk4'):
            for current in range(start_step, end_step):
                # run() saves only at segment boundaries. Keep the actual
                # rejected step separate from the last durable checkpoint.
                self.advance_location = dict(last_accepted_step=current,
                    attempted_step=current+1, last_accepted_fs=current*self.c.dt_au*AU_FS,
                    attempted_fs=(current+1)*self.c.dt_au*AU_FS)
                state = self.step(state)
            return state, 4*(end_step-start_step)
        keys = list(state)
        shapes = {k: state[k].shape for k in keys}
        complex_keys = {k for k in keys if np.iscomplexobj(state[k])}
        sizes = [state[k].size*(2 if k in complex_keys else 1) for k in keys]
        def pack(s):
            return np.concatenate([np.r_[s[k].real.ravel(), s[k].imag.ravel()] if k in complex_keys
                                   else s[k].ravel() for k in keys])
        def unpack(y):
            parts = np.split(y, np.cumsum(sizes)[:-1]); out = {}
            for k, v in zip(keys, parts):
                if k in complex_keys:
                    a, b = np.split(v, 2); v = a+1j*b
                out[k] = v.reshape(shapes[k])
            return out
        def rhs(t, y):
            self.advance_location['last_rhs_evaluation_fs'] = t*AU_FS
            return pack(self.evaluate(unpack(y))[0])
        self.advance_location = dict(last_accepted_fs=start_step*self.c.dt_au*AU_FS,
                                     attempted_step=None, attempted_fs=None)
        solver = DOP853(rhs, start_step*self.c.dt_au, pack(state), end_step*self.c.dt_au,
                        max_step=self.c.dt_au, rtol=self.c.rtol, atol=self.c.atol)
        while solver.status == 'running':
            if solver.nfev > self.c.max_rhs_evaluations_per_segment:
                raise FloatingPointError('Adaptive RHS budget exceeded; inspect stiffness/projection rank, do not run blindly longer')
            message = solver.step()
            if solver.status == 'failed': raise FloatingPointError('DOP853: '+str(message))
            self.guard(unpack(solver.y))
            self.advance_location['last_accepted_fs'] = solver.t*AU_FS
        return unpack(solver.y), solver.nfev

    def guard(self, state):
        if any(not np.all(np.isfinite(v)) for v in state.values()):
            raise FloatingPointError('Nonfinite proposed state; checkpoint remains unchanged')
        G, bridges = self.reconstruct(state)
        norm = np.einsum('q,iqk->i', self.model.quadrature, abs(G)**2)
        err = float(np.max(abs(norm-1)))
        diagnostics = dict(integrated_norm_error=err,
                           pointwise_pnc_error=None, worst_pnc_location=None)
        if bridges:
            diagnostics['pointwise_pnc_error'] = 0.
            diagnostics['density_weighted_pnc_error'] = 0.
            for i, (b, C) in enumerate(zip(bridges, state['C'])):
                mask = b.rho > self.c.support_relative*np.max(b.rho)
                defect = abs(np.sum(abs(C)**2, axis=1)-1)
                where = np.flatnonzero(mask)[np.argmax(defect[mask])]
                local = float(defect[where])
                if local >= diagnostics['pointwise_pnc_error']:
                    diagnostics['pointwise_pnc_error'] = local
                    diagnostics['worst_pnc_location'] = dict(trajectory=i,
                        q_index=int(where), q=float(self.model.q[where]),
                        relative_density=float(b.rho[where]/np.max(b.rho)),
                        density=float(b.rho[where]))
                diagnostics['density_weighted_pnc_error'] = max(
                    diagnostics['density_weighted_pnc_error'],
                    float(self.model.quadrature@(b.rho*defect)))
                err = max(err, local)
        if err > self.c.accepted_norm_tolerance:
            raise StateGuardError('Raw norm/PNC error %.6g; inspect timestep, q discretization '
                'and tail conditioning (no renormalization performed)' % err, diagnostics)
        edge = np.sum(self.model.quadrature[[0, -1]][None, :, None]*abs(G[:, [0, -1]])**2, axis=(1, 2))
        diagnostics['q_boundary_mass'] = float(np.max(edge))
        if np.max(edge) > self.c.edge_mass_tolerance:
            raise StateGuardError('q boundary mass %.6g exceeds %.6g; enlarge box' %
                                     (np.max(edge), self.c.edge_mass_tolerance), diagnostics)
        self.last_guard_diagnostics = diagnostics
        self.audit['max_accepted_norm_pnc_error'] = max(self.audit['max_accepted_norm_pnc_error'], err)
        self.audit['max_edge_mass'] = max(self.audit['max_edge_mass'], float(np.max(edge)))
        return err


def json_value(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=json_value)+'\n')
    os.replace(tmp, path)


def atomic_npz(path, **data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **data)
    os.replace(tmp, path)


def snapshot(engine, state, t_fs):
    from .fields import analyze_fields
    m, c = engine.model, engine.c
    G, bridges = engine.reconstruct(state)
    _, rates = engine.evaluate(state)
    fields = analyze_fields(state['R'], G, m.quadrature, m.q,
                  basis_convention=m.basis_convention, gauge_convention=GAUGE,
                  mls_config=MLSConfig(**c.mls), support_relative=c.support_relative,
                  pnc_tolerance=c.accepted_norm_tolerance,
                  q_operator=m.config.q_operator)
    obs = m.observables(state['R'], G)
    pop = np.einsum('q,iqk->ik', m.quadrature, abs(G)**2)
    # Conditional probabilities ONLY for density estimation; no state update.
    # Record normalization defect and keep raw populations in saved arrays.
    norm = pop.sum(axis=1)
    cond_pop = pop/norm[:, None]
    rg = np.linspace(np.min(state['R'])-3*m.initial_R_sigma,
                     np.max(state['R'])+3*m.initial_R_sigma, 128)
    args = dict(channel_basis=m.basis_convention, width_floor=c.heavy_width_floor)
    if c.heavy_density == 'state_resolved':
        heavy = reconstruct_heavy_density(state['R'], engine.eta, cond_pop, **args)
        heavy_grid = reconstruct_heavy_density(state['R'], engine.eta, cond_pop, rg, **args)
    else:
        heavy = reconstruct_kde_density(state['R'], engine.eta, bandwidth=c.kde_bandwidth)
        heavy_grid = reconstruct_kde_density(state['R'], engine.eta, rg, bandwidth=c.kde_bandwidth)
    kinetic = state['P']**2/(2*m.mass_R)
    internal = np.einsum('q,iqk,iqk->i', m.quadrature, G.conj(), m.action(state['R'], G)).real
    external = m.external_potential(state['R'])
    derivative_audit = fields['derivative_audit']
    result = dict(t_fs=t_fs, R=state['R'], P=state['P'], eta=engine.eta, q=m.q,
                  quadrature=m.quadrature, G=G, population=pop, kinetic_R=kinetic,
                  energy=internal+external+kinetic, heavy_eval_R=rg, heavy_density=heavy_grid.density,
                  Q_raw=heavy.quantum_momentum, z=state.get('z', np.empty((c.ntraj, 0))),
                  projection_residual=np.array([r['actual_vs_eom_rate_l2'] for r in rates]) if rates else np.zeros(c.ntraj),
                  pnc_error=engine.guard(state), normalization_defect=norm-1,
                  MLS_condition=derivative_audit['condition'],
                  photon_number=obs.get('photon_number', np.full(c.ntraj, np.nan)))
    for key in ('rho', 'a', 'b', 'alpha', 'delta_b', 'excluded_mass'):
        result[key] = fields[key]
    if 'bo_population' in obs:
        result['bo_population'] = obs['bo_population']
    for key in ('q_p_commutator_over_i', 'q_p_commutator_defect',
                'q_p_commutator_imaginary_defect'):
        if key in obs:
            result[key] = obs[key]
    audit = dict(projection=rates, derivative=derivative_audit,
                 guard=engine.last_guard_diagnostics,
                 q_operator=m.config.q_operator,
                 integrator=getattr(engine, 'last_integrator_audit', {'integrator': c.integrator}),
                 field_checks={key: float(np.nanmax(abs(fields[key]))) for key in
                               ('a_imaginary_defect', 'b_imaginary_defect', 'alpha_imaginary_defect',
                                'a_quotient_residual', 'b_quotient_residual', 'alpha_bridge_residual')},
                 q_edge_mass=fields['q_edge_mass'],
                 density=heavy.audit, approximate_field=True,
                 scalar_potentials='unavailable: no reconstructed complex chi or matched EF gauge',
                 corrected_quantum_momentum='not applied by MTE',
                 bare_population_normalization_defect=(norm-1).tolist())
    return result, audit


def refresh_timeseries(root):
    names = ('energy', 'kinetic_R', 'photon_number', 'population')
    data = {k: [] for k in ('times_fs',)+names+('projection_residual', 'pnc_error', 'mls_condition')}
    for p in sorted((root/'frames').glob('frame_*.npz')):
        with np.load(p) as f:
            data['times_fs'].append(float(f['t_fs']))
            for key in names: data[key].append(np.einsum('i,i...->...', f['eta'], f[key]))
            data['projection_residual'].append(np.max(f['projection_residual']))
            data['pnc_error'].append(float(f['pnc_error']))
            data['mls_condition'].append(np.max(f['MLS_condition']))
    atomic_npz(root/'timeseries.npz', **{k: np.array(v) for k, v in data.items()})


def source_hashes():
    """Include reused Hamiltonians, not just the new adapter namespace."""
    base = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [base/p for p in ('multi_component_exact_factorization/core.py',
              'multi_component_exact_factorization/pen_ctmqc_2026/core.py',
              'multi_component_exact_factorization/pen_perspectives_2026/model.py',
              'multi_component_exact_factorization/pen_perspectives_2026/resonance.py',
              'multi_component_exact_factorization_cavity/core.py')]
    return {str(p.resolve().relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def resource_estimate(config, model):
    complex_bytes = config.ntraj*len(model.q)*model.nk*16
    frames = int(np.ceil(config.end_fs/config.every_fs))+2
    return dict(independent_complex_array_MiB=complex_bytes/2**20,
                approximate_peak_work_arrays_MiB=20*complex_bytes/2**20,
                uncompressed_frame_series_MiB=frames*(complex_bytes+6*config.ntraj*len(model.q)*8)/2**20,
                estimated_frames=frames, note='Planning bound, not measured RSS; NPZ compression and movie size vary')


def run(config, out, *, resume=False, movies=False, render=True):
    """Checkpoint contains sole independent state, config and deterministic RNG seed.

    Resume permits only extending end_fs; exact dt/model/policies must match.
    Existing non-resume directories are refused, never silently overwritten.
    """
    config.validate()
    root = Path(out)
    if root.exists() and not resume:
        raise FileExistsError('Use a new output directory, or --resume for this same configuration')
    engine = Engine(config)
    estimate = resource_estimate(config, engine.model)
    print('RESOURCE ESTIMATE '+json.dumps(estimate), flush=True)
    hashes = source_hashes()
    root.mkdir(parents=True, exist_ok=resume)
    (root/'frames').mkdir(exist_ok=True)
    start = time.perf_counter()
    checkpoint = root/'checkpoint.npz'
    if resume:
        with np.load(checkpoint, allow_pickle=False) as f:
            old = json.loads(str(f['config_json']))
            new = asdict(config)
            old.pop('end_fs'); new.pop('end_fs')
            if old != new: raise ValueError('Resume config mismatch; only end_fs may change')
            if 'source_hash_json' not in f or json.loads(str(f['source_hash_json'])) != hashes:
                raise ValueError('Resume source hash mismatch; do not mix algorithm versions in one run')
            step = int(f['step'])
            state = {key[6:]: f[key].copy() for key in f.files if key.startswith('state_')}
    else:
        atomic_json(root/'config.json', asdict(config))
        try:
            state, fit = engine.initial(); step = 0
        except Exception as exc:
            atomic_json(root/'failure.json', dict(error=repr(exc), phase='initialization',
                        instruction='No propagation/checkpoint written; inspect initial grid and Gaussian fit'))
            atomic_json(root/'status.json', dict(status='FAILED', phase='initialization', error=repr(exc)))
            raise
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        metadata = dict(commit=commit, source_sha256=hashes, python=platform.python_version(),
                        numpy=np.__version__, hostname=platform.node(), model=engine.model.metadata(),
                        integrator=config.integrator,
                        integrator_policy=('homogeneous C=Y/||Y||, raw auxiliary defects guarded; not an accuracy proof'
                            if config.integrator == 'sphere-rk4' else 'no renormalization'),
                        precision='float64/complex128',
                        gauge=GAUGE, initializer_fit=fit, backend='B1 full-internal MTE',
                        resource_estimate=estimate,
                        method_label='MTE + Gaussian bridge projection' if config.compression == 'gaussian' else 'Uncompressed full-internal MTE',
                        histories='none required by MTE', units='atomic units except labeled fs')
        atomic_json(root/'metadata.json', metadata)
    final_step = int(np.ceil(config.end_fs/AU_FS/config.dt_au-1e-12))
    if final_step < step: raise ValueError('Resume target precedes checkpoint')
    stride = max(1, int(round(config.every_fs/AU_FS/config.dt_au)))
    atomic_json(root/'config.json', asdict(config))
    last_saved_step = step if resume else None
    phase = 'initial_guard'
    def save():
        nonlocal phase, last_saved_step
        phase = 'snapshot_and_save'
        frame, audit = snapshot(engine, state, step*config.dt_au*AU_FS)
        atomic_npz(root/'frames'/('frame_%010d.npz' % step), **frame)
        atomic_json(root/'frames'/('audit_%010d.json' % step), audit)
        atomic_npz(checkpoint, step=step, config_json=json.dumps(asdict(config)), source_hash_json=json.dumps(hashes),
                   **{'state_'+k: v for k, v in state.items()})
        last_saved_step = step  # durable only after the atomic replacement
        status = dict(status='RUNNING', step=step, t_fs=frame['t_fs'], wall_seconds=time.perf_counter()-start)
        atomic_json(root/'status.json', status)
        print('SAVED step=%d t=%.6f fs norm/PNC=%.3g projection=%.3g' %
              (step, frame['t_fs'], frame['pnc_error'], np.max(frame['projection_residual'])), flush=True)
    try:
        engine.guard(state)
        if not resume: save()
        evaluations = 0
        while step < final_step:
            target = min(final_step, (step//stride+1)*stride)
            phase = 'propagation'
            state, nfev = engine.advance(state, step, target)
            evaluations += nfev; step = target
            save()
        phase = 'postprocessing'
        refresh_timeseries(root)
        propagation_seconds = time.perf_counter()-start
        if render:
            phase = 'rendering'
            from .renderer import render as render_outputs
            render_outputs(root, movies=movies)
        phase = 'final_status'
        atomic_json(root/'status.json', dict(status='COMPLETE_NOT_CERTIFIED', step=step,
            t_fs=step*config.dt_au*AU_FS, propagation_seconds=propagation_seconds,
            rhs_evaluations_this_session=evaluations,
            numerical_audit_this_session=engine.audit,
            wall_seconds=time.perf_counter()-start, max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            warning='Short pilot only; Gaussian projection and classical heavy dynamics are approximations'))
    except Exception as exc:
        location = getattr(engine, 'advance_location', {}) if phase == 'propagation' else {}
        saved_fs = None if last_saved_step is None else last_saved_step*config.dt_au*AU_FS
        atomic_json(root/'failure.json', dict(error=repr(exc),
                    phase=phase, attempted_step=location.get('attempted_step'),
                    propagation_location=location, last_saved_step=last_saved_step,
                    last_saved_fs=saved_fs,
                    rejected_candidate=getattr(exc, 'diagnostics', None),
                    integrator_rejection=getattr(exc, 'audit', None),
                    instruction='Last saved checkpoint preserved; diagnose before resume'))
        atomic_json(root/'status.json', dict(status='FAILED', step=step, error=repr(exc),
                    phase=phase, last_saved_step=last_saved_step, last_saved_fs=saved_fs,
                    numerical_audit=engine.audit,
                    propagation_location=location, wall_seconds=time.perf_counter()-start))
        raise
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Full RunConfig JSON; explicit CLI options override it')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--model', choices=['pen-nonadiabatic', 'pen-rabi', 'mcef'])
    p.add_argument('--method', choices=['mte', 'ctmqc', 'tsh', 'ef-g-mls'])
    p.add_argument('--compression', choices=['gaussian', 'native'])
    p.add_argument('--integrator', choices=['dop853', 'rk4', 'sphere-rk4'])
    for name in ('ntraj', 'seed', 'components'): p.add_argument('--'+name, type=int)
    for name in ('dt-au', 'end-fs', 'every-fs'): p.add_argument('--'+name, type=float)
    for name in ('nq', 'nx'): p.add_argument('--'+name, type=int)
    for name in ('qmin', 'qmax', 'g', 'omega'): p.add_argument('--'+name, type=float)
    p.add_argument('--dse', choices=['on', 'off'])
    p.add_argument('--q-operator', choices=['fft', 'fd2', 'fd4'],
                   help='Explicit q discretization; FFT remains the historical default')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--movies', action='store_true')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--estimate', action='store_true', help='Read-only resource estimate; no propagation/output')
    a = p.parse_args()
    raw = json.loads(a.config.read_text()) if a.config else (
        json.loads((a.out/'config.json').read_text()) if a.resume else {})
    model = raw.setdefault('model', {})
    for key in ('nq', 'nx', 'qmin', 'qmax', 'g', 'omega'):
        if getattr(a, key) is not None: model[key] = getattr(a, key)
    if a.model: model['name'] = a.model
    if a.dse: model['dse'] = a.dse == 'on'
    if a.q_operator: model['q_operator'] = a.q_operator
    for key in ('method', 'compression', 'integrator', 'ntraj', 'seed', 'components', 'dt_au', 'end_fs', 'every_fs'):
        if getattr(a, key) is not None: raw[key] = getattr(a, key)
    config = RunConfig(**raw)
    if a.estimate:
        engine = Engine(config)
        print(json.dumps(resource_estimate(config, engine.model), indent=2))
        return
    run(config, a.out, resume=a.resume, movies=a.movies, render=not a.no_render)


if __name__ == '__main__':
    main()
