"""Explicit CTMQC/FSSH pilots, optionally with the P3/P8 Gaussian projection.

Separate CLI: old MTE/native G-CTMQC/TDSE defaults are unchanged. These are
full-internal adiabatic-channel equations, not the paper's classical-photon
electronic perspective. Only R,P are classical. Every q-grid channel is kept.
No exact scalar TDPES is inferred from a heavy-density reconstruction.
"""
import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .run import (AU_FS, GAUGE, RunConfig, Engine, atomic_json, atomic_npz,
                  snapshot, refresh_timeseries, source_hashes, resource_estimate)
from .gaussian import split_and_project
from .internal_channels import InternalChannels
from .ctmqc_channels import ctmqc_channel_rhs
from .heavy_density import reconstruct_heavy_density, reconstruct_kde_density
from .surface_hopping import (sample_active_states, hopping_probabilities,
                              draw_hop_targets, apply_hops)

CHANNEL_BASIS = 'complete x-q internal eigenchannels, overlap continued from common R center'


@dataclass
class ChannelConfig(RunConfig):
    method: str = 'ctmqc'
    integrator: str = 'rk4'
    max_internal_dimension: int = 256
    minimum_channel_overlap: float = .25
    quantum_momentum_mode: str = 'density'
    intercept_policy: str = 'raise'
    frustrated_policy: str = 'keep'

    def base_config(self):
        args = {f.name: getattr(self, f.name) for f in fields(RunConfig)}
        args['method'] = 'mte'
        return RunConfig(**args)

    def validate(self):
        self.base_config().validate()
        if self.method not in ('ctmqc', 'tsh'):
            raise ValueError('Channel driver supports ctmqc or tsh; old MTE CLI is unchanged')
        if self.integrator not in ('rk4', 'sphere-rk4'):
            raise ValueError('Channel pilots require RK4 or sphere-rk4/event steps')
        if self.quantum_momentum_mode not in ('density', 'pair_intercept'):
            raise ValueError('Select density or pair_intercept quantum momentum')
        if self.intercept_policy not in ('raise', 'density_fallback'):
            raise ValueError('Unknown intercept cancellation policy')
        if self.frustrated_policy not in ('keep', 'reverse'):
            raise ValueError('Unknown frustrated-hop policy')
        if not 0 < self.minimum_channel_overlap <= 1 or self.max_internal_dimension < 2:
            raise ValueError('Invalid internal channel controls')


class ChannelEngine(Engine):
    def __init__(self, config):
        config.validate()
        super().__init__(config.base_config())
        self.c = config
        self.channels = InternalChannels(self.model, config.max_internal_dimension,
                                         minimum_overlap=config.minimum_channel_overlap)
        self.anchor = None
        self._channel_cache = None
        self.active = np.full(config.ntraj, -1, dtype=int)
        # Independent stream: Wigner sampling must not depend on hop events.
        self.rng = np.random.default_rng(np.random.SeedSequence([config.seed, 947]))
        self.hops = dict(attempted=0, accepted=0, frustrated=0)
        self.audit.update(max_roundtrip_error=0., minimum_channel_overlap=1.,
                          max_hop_energy_error=0., max_ct_force=0.)

    def initial(self):
        state, fit = super().initial()
        channels = self.channels.initialize(state['R'])
        self.anchor = channels.vectors
        G, _ = self.reconstruct(state)
        A = channels.project(G, self.model.quadrature)
        if self.c.method == 'ctmqc':
            state['f'] = np.zeros_like(A.real)
        else:
            self.active = sample_active_states(A, self.rng)
        return state, fit

    def channel_data(self, state, G=None):
        # Snapshot fields/rate audits reuse identical R and channel reference;
        # avoid rediagonalizing without altering any propagation stage.
        cache = self._channel_cache
        if cache is not None and cache[1] is self.anchor and np.array_equal(cache[0], state['R']):
            channels = cache[2]
        else:
            channels = self.channels.evaluate(state['R'], self.anchor)
            self._channel_cache = (state['R'].copy(), self.anchor, channels)
        if G is None:
            G = self.reconstruct(state)[0]
        A = channels.project(G, self.model.quadrature)
        back = channels.lift(A, self.model.quadrature, self.model.nk)
        error = float(np.max(np.sqrt(np.einsum('q,iqk->i', self.model.quadrature, abs(back-G)**2))))
        if error > 1e-10:
            raise ValueError('B3/B4 internal basis round-trip failure; no implicit truncation')
        self.audit['max_roundtrip_error'] = max(self.audit['max_roundtrip_error'], error)
        self.audit['minimum_channel_overlap'] = min(self.audit['minimum_channel_overlap'], channels.minimum_overlap)
        return channels, A

    def native_rates(self, state, G):
        m, c = self.model, self.c
        channels, A = self.channel_data(state, G)
        F = -1j*m.action(state['R'], G)
        extra = {}
        if c.method == 'ctmqc':
            result = ctmqc_channel_rhs(A, channels.energy, channels.nac, channels.force_matrix,
                state['f'], state['R'], state['P'], m.mass_R, self.eta,
                channel_basis=CHANNEL_BASIS, quantum_momentum_mode=c.quantum_momentum_mode,
                density_mode=c.heavy_density, width_floor=c.heavy_width_floor,
                bandwidth=c.kde_bandwidth, intercept_policy=c.intercept_policy,
                normalization_tolerance=c.stage_pnc_tolerance)
            # B5: U[-v d A] + v (partial_R U) A cancel for this COMPLETE basis.
            # Keeping G in its common primitive gauge avoids differentiating
            # phase-aligned eigenvectors numerically or losing the q kinetic.
            F += channels.lift(result['ct_amplitude_dot'], m.quadrature, m.nk)
            force = result['heavy_force']
            extra['f'] = result['force_history_dot']
            self.audit['max_ct_force'] = max(self.audit['max_ct_force'], float(np.max(abs(result['ct_force']))))
        else:
            force = channels.force_matrix[np.arange(c.ntraj), self.active, self.active].real
            result = dict(ct_force=np.zeros(c.ntraj), audit={'decoherence': 'none'})
        if not m.is_photon:
            force = force-.06*(state['R']-9.5)  # inherited heavy trap, ONCE
        return F, force, extra, channels, A, result

    def evaluate(self, state):
        G, bridges = self.reconstruct(state)
        F, force, extra, _, _, _ = self.native_rates(state, G)
        derivative = dict(R=state['P']/self.model.mass_R, P=force, **extra)
        if not bridges:
            derivative['G'] = F
            return derivative, []
        rates = [split_and_project(rhs, C, b, support_relative=self.c.support_relative,
                 max_excluded_mass=self.c.max_excluded_mass, rank_rtol=self.c.rank_rtol,
                 rank_policy=self.c.rank_policy, pnc_tolerance=self.c.stage_pnc_tolerance)
                 for rhs, C, b in zip(F, state['C'], bridges)]
        for r in rates:
            self.audit['max_projection_rate_residual'] = max(self.audit['max_projection_rate_residual'], r.audit['actual_vs_eom_rate_l2'])
            self.audit['max_excluded_mass'] = max(self.audit['max_excluded_mass'], r.audit['excluded_mass'])
            self.audit['max_gaussian_condition'] = max(self.audit['max_gaussian_condition'], r.audit['condition_L'])
            self.audit['minimum_gaussian_rank'] = min(self.audit['minimum_gaussian_rank'], r.audit['rank'])
        derivative.update(z=np.array([r.parameter_dot for r in rates]), C=np.array([r.C_dot for r in rates]))
        return derivative, [r.audit for r in rates]

    def step(self, state):
        # Native NAC flux only, evaluated at the start of the event interval.
        # This is first-order event sampling with RK4 continuous evolution;
        # event-step/statistical convergence must be tested independently.
        probabilities = None
        if self.c.method == 'tsh':
            channels, A = self.channel_data(state)
            probabilities = hopping_probabilities(A, channels.nac, state['P']/self.model.mass_R,
                                                   self.active, self.c.dt_au)
        new = super().step(state)
        channels, _ = self.channel_data(new)
        if probabilities is not None:
            targets = draw_hop_targets(probabilities, self.active, self.rng)
            event = apply_hops(new['P'], self.active, targets, channels.energy, self.model.mass_R,
                               frustrated_policy=self.c.frustrated_policy)
            new['P'], self.active = event.momentum, event.active
            for key in self.hops:
                self.hops[key] += int(np.count_nonzero(getattr(event, key)))
            self.audit['max_hop_energy_error'] = max(self.audit['max_hop_energy_error'], float(np.max(abs(event.energy_residual))))
        self.anchor = channels.vectors
        self.guard(new)
        return new


def channel_snapshot(engine, state, t_fs):
    frame, audit = snapshot(engine, state, t_fs)
    _, _, _, channels, A, rates = engine.native_rates(state, frame['G'])
    frame.update(channel_population=abs(A)**2, channel_energy=channels.energy,
                 active_state=engine.active.copy(), ct_force_correction=rates['ct_force'],
                 **{'hops_'+key: value for key, value in engine.hops.items()})
    if engine.c.method == 'ctmqc':
        frame.update(Q_raw=rates['raw_quantum_momentum'], Q_pairs=rates['pair_quantum_momentum'],
                     Q_pair_indices=rates['pair_indices'], force_history=state['f'])
        audit['channel_backend'] = rates['audit']
        audit['corrected_quantum_momentum'] = engine.c.quantum_momentum_mode
        # Q_raw and the displayed heavy density MUST use the same channels.
        if engine.c.heavy_density == 'state_resolved':
            heavy_grid = reconstruct_heavy_density(state['R'], engine.eta, rates['populations'],
                frame['heavy_eval_R'], channel_basis=CHANNEL_BASIS, width_floor=engine.c.heavy_width_floor)
        else:
            heavy_grid = reconstruct_kde_density(state['R'], engine.eta, frame['heavy_eval_R'],
                                                 bandwidth=engine.c.kde_bandwidth)
        frame['heavy_density'] = heavy_grid.density
        audit['density'] = rates['density'].audit
    else:
        # The meaningful FSSH conserved diagnostic is active-surface energy,
        # not the electronic expectation value saved by the generic renderer.
        frame['active_surface_energy'] = (frame['kinetic_R']+engine.model.external_potential(state['R'])
                         +channels.energy[np.arange(engine.c.ntraj), engine.active])
        audit['channel_backend'] = dict(method='FSSH', event_rule='start-of-step native NAC flux',
            frustrated_policy=engine.c.frustrated_policy, decoherence='none', collapse=False)
    audit['channel_basis'] = CHANNEL_BASIS
    audit['retained_channels'] = engine.channels.dimension
    # Compression may change channel populations even if the native pair-Q
    # correction has zero ensemble transfer. Save this contribution separately.
    derivative, _ = engine.evaluate(state)
    G, bridges = engine.reconstruct(state)
    if bridges:
        actual = []
        for b, C, dz, dC in zip(bridges, state['C'], derivative['z'], derivative['C']):
            drho = b.jacobian@dz
            dlam = np.zeros_like(b.amplitude)
            np.divide(drho, 2*b.amplitude, out=dlam, where=b.amplitude>0)
            actual.append(dlam[:, None]*C+b.amplitude[:, None]*dC)
        rhs = engine.native_rates(state, G)[0]
        difference = channels.project(np.array(actual)-rhs, engine.model.quadrature)
        frame['compression_channel_population_rate'] = 2*np.real(A.conj()*difference)
    else:
        frame['compression_channel_population_rate'] = np.zeros_like(A.real)
    return frame, audit


def run(config, out, *, resume=False, movies=False, render=True):
    config.validate()
    root = Path(out)
    if root.exists() and not resume:
        raise FileExistsError('Use a new directory or --resume; no existing results overwritten')
    engine = ChannelEngine(config)
    hashes = source_hashes()
    estimate = resource_estimate(config, engine.model)
    estimate['complete_internal_dimension'] = engine.channels.dimension
    estimate['channel_work_arrays_MiB'] = 10*config.ntraj*engine.channels.dimension**2*16/2**20
    print('RESOURCE ESTIMATE '+json.dumps(estimate), flush=True)
    root.mkdir(parents=True, exist_ok=resume)
    (root/'frames').mkdir(exist_ok=True)
    checkpoint = root/'checkpoint.npz'
    start = time.perf_counter()
    step = 0
    last_saved_step = None
    phase = 'initialization'
    final = int(np.ceil(config.end_fs/AU_FS/config.dt_au-1e-12))
    stride = max(1, int(round(config.every_fs/AU_FS/config.dt_au)))
    # Refusing an incompatible resume must not relabel a completed old run as
    # FAILED or touch its checkpoint/config/status. Validate before any writes.
    if resume:
        with np.load(checkpoint, allow_pickle=False) as f:
            old, new = json.loads(str(f['config_json'])), asdict(config)
            old.pop('end_fs'); new.pop('end_fs')
            if old != new or json.loads(str(f['source_hash_json'])) != hashes:
                raise ValueError('Resume config/source mismatch; only end_fs may change')
            if final < int(f['step']):
                raise ValueError('Resume target precedes checkpoint')
    try:
        if resume:
            with np.load(checkpoint, allow_pickle=False) as f:
                state = {k[6:]: f[k].copy() for k in f.files if k.startswith('state_')}
                step = int(f['step'])
                last_saved_step = step
                engine.anchor, engine.active = f['anchor'].copy(), f['active'].copy()
                engine.rng.bit_generator.state = json.loads(str(f['rng_json']))
                engine.hops = json.loads(str(f['hops_json']))
                engine.audit = json.loads(str(f['engine_audit_json']))
        else:
            atomic_json(root/'config.json', asdict(config))
            state, fit = engine.initial()
            label = ('CTMQC equations' if config.method == 'ctmqc' else 'FSSH')
            label += (' + Gaussian bridge projection (experimental)' if config.compression == 'gaussian'
                      else ' full-internal, uncompressed pilot')
            atomic_json(root/'metadata.json', dict(method_label=label, gauge=GAUGE,
                backend='independent published channel equations; NOT authors G-CTMQC executable',
                source_sha256=hashes, commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                hostname=platform.node(), initializer_fit=fit, model=engine.model.metadata(),
                resource_estimate=estimate, channel_basis=CHANNEL_BASIS,
                integrator=config.integrator+' whole ensemble; TSH events at each dt boundary',
                integrator_policy=('homogeneous C=Y/||Y||; raw auxiliary defects guarded; not an accuracy proof'
                    if config.integrator == 'sphere-rk4' else 'no renormalization'),
                force_history='integrated internal adiabatic force, no reset',
                low_density='outside support C frozen, no renormalization; threshold and mass budget explicit',
                scalar_potentials='unavailable without a matched complex chi/gauge closure'))
        if final < step:
            raise ValueError('Resume target precedes checkpoint')
        atomic_json(root/'config.json', asdict(config))

        def save():
            nonlocal phase, last_saved_step
            phase = 'snapshot_and_save'
            frame, audit = channel_snapshot(engine, state, step*config.dt_au*AU_FS)
            atomic_npz(root/'frames'/('frame_%010d.npz' % step), **frame)
            atomic_json(root/'frames'/('audit_%010d.json' % step), audit)
            atomic_npz(checkpoint, step=step, config_json=json.dumps(asdict(config)),
                       source_hash_json=json.dumps(hashes), anchor=engine.anchor, active=engine.active,
                       rng_json=json.dumps(engine.rng.bit_generator.state), hops_json=json.dumps(engine.hops),
                       engine_audit_json=json.dumps(engine.audit),
                       **{'state_'+k:v for k,v in state.items()})
            last_saved_step = step
            atomic_json(root/'status.json', dict(status='RUNNING',step=step,t_fs=frame['t_fs']))
            print('SAVED %s t=%.6f fs norm/PNC=%.3g projection=%.3g hops=%s' %
                  (config.method,frame['t_fs'],frame['pnc_error'],max(frame['projection_residual']),engine.hops),flush=True)
        phase = 'initial_guard'
        engine.guard(state)
        if not resume:
            save()
        while step < final:
            phase = 'propagation'
            state = engine.step(state)
            step += 1
            if step % stride == 0 or step == final:
                save()
        phase = 'postprocessing'
        refresh_timeseries(root)
        propagation_seconds = time.perf_counter()-start
        if render:
            phase = 'rendering'
            from .renderer import render as base_render
            from .channel_render import render as channel_render
            base_render(root, movies=movies)
            channel_render(root, movies=movies)
        phase = 'final_status'
        atomic_json(root/'status.json', dict(status='COMPLETE_NOT_CERTIFIED',step=step,
            t_fs=step*config.dt_au*AU_FS,propagation_seconds=propagation_seconds,
            wall_seconds=time.perf_counter()-start,numerical_audit=engine.audit,hops=engine.hops,
            warning='Implementation pilot, not convergence or physical validation; inspect projection and timestep'))
    except Exception as exc:
        saved_fs = None if last_saved_step is None else last_saved_step*config.dt_au*AU_FS
        atomic_json(root/'failure.json',dict(error=repr(exc),phase=phase,
                    attempted_step=step+1 if phase == 'propagation' else None,
                    attempted_fs=(step+1)*config.dt_au*AU_FS if phase == 'propagation' else None,
                    last_accepted_step=step,last_accepted_fs=step*config.dt_au*AU_FS,
                    last_saved_step=last_saved_step,last_saved_fs=saved_fs,
                    rejected_candidate=getattr(exc, 'diagnostics', None),
                    integrator_rejection=getattr(exc, 'audit', None),
                    instruction='Last atomic checkpoint preserved; inspect guard, do not hide norm errors'))
        atomic_json(root/'status.json',dict(status='FAILED',step=step,error=repr(exc),
                    phase=phase,last_saved_step=last_saved_step,last_saved_fs=saved_fs,
                    last_accepted_fs=step*config.dt_au*AU_FS,numerical_audit=engine.audit,
                    wall_seconds=time.perf_counter()-start))
        raise
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--method',choices=['ctmqc','tsh'])
    p.add_argument('--model',choices=['mcef','pen-nonadiabatic','pen-rabi'])
    p.add_argument('--compression',choices=['gaussian','native'])
    p.add_argument('--integrator',choices=['rk4','sphere-rk4'])
    for key in ('ntraj','seed','components','max-internal-dimension'):
        p.add_argument('--'+key,type=int)
    for key in ('dt-au','end-fs','every-fs','support-relative','max-excluded-mass','heavy-width-floor'):
        p.add_argument('--'+key,type=float)
    for key in ('nq','nx'):
        p.add_argument('--'+key,type=int)
    for key in ('qmin','qmax','g'):
        p.add_argument('--'+key,type=float)
    p.add_argument('--dse',choices=['on','off'])
    p.add_argument('--q-operator',choices=['fft','fd2','fd4'])
    p.add_argument('--quantum-momentum-mode',choices=['density','pair_intercept'])
    p.add_argument('--intercept-policy',choices=['raise','density_fallback'])
    p.add_argument('--frustrated-policy',choices=['keep','reverse'])
    p.add_argument('--movies',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-render',action='store_true')
    a = p.parse_args()
    raw = json.loads(a.config.read_text()) if a.config else (
        json.loads((a.out/'config.json').read_text()) if a.resume else {})
    model = raw.setdefault('model',{})
    for key in ('nq','nx','qmin','qmax','g'):
        if getattr(a,key) is not None:
            model[key]=getattr(a,key)
    if a.model: model['name']=a.model
    if a.dse: model['dse']=a.dse=='on'
    if a.q_operator: model['q_operator']=a.q_operator
    for f in fields(ChannelConfig):
        if f.name!='model' and getattr(a,f.name,None) is not None:
            raw[f.name]=getattr(a,f.name)
    run(ChannelConfig(**raw),a.out,resume=a.resume,movies=a.movies,render=not a.no_render)


if __name__ == '__main__':
    main()
