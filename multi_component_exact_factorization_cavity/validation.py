"""Short same-grid timestep validation; not a full scientific convergence gate."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import time
import numpy as np
from .core import configuration,photon_grid,initial_wave
from .backend import Hamiltonian
from .workflow import cached_prepare,write_json,fingerprint,source_record,AU_PER_FS


def validate(config,out,*,campaign='pen',case='strong',gpu=False,duration_fs=.02):
    """Compare dt and dt/2 at the same exact time with full-grid wave fidelity."""
    c=configuration(config);out=Path(out)
    packet=cached_prepare(c,out/'cache');Q=photon_grid(c)
    if campaign=='vsc' and not packet.get('vsc_available',False):
        raise ValueError('VSC preparation undecided; inspect calibration and choose explicitly')
    omega=packet['omega_'+campaign]*(1+c['detuning'])
    g=c.get('g_cavity',packet['g_'+campaign][case])
    dt=float(c.get('dt_au',.025));n=max(2,int(np.ceil(duration_fs*AU_PER_FS/dt)))
    initial=initial_wave(packet,Q,campaign);wall=time.monotonic()
    waves=[];obs=[]
    for fraction,steps in ((1.,n),(.5,2*n)):
        h=Hamiltonian(packet,Q,omega,g,gpu=gpu,dse=c['dse'],block=c.get('block',8))
        u=h.array(initial);before=h.observe(u)
        for _ in range(steps):u=h.step(u,dt*fraction)
        after=h.observe(u);after['energy_drift']=after['energy']-before['energy']
        waves.append(h.host(u).copy());obs.append(after)
        del h,u
        if gpu:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
    cell=packet['dx']*packet['dR']*(Q[1]-Q[0])
    overlap=np.vdot(waves[0],waves[1])*cell
    fidelity=float(abs(overlap)**2/(obs[0]['norm']*obs[1]['norm']))
    l1=float(np.sum(abs(obs[0]['rho_R']-obs[1]['rho_R']))*packet['dR'])
    checks=dict(finite=all(np.isfinite(o['energy']) for o in obs),
        norm=max(abs(o['norm']-1) for o in obs)<1e-8,
        timestep_fidelity=1-fidelity<1e-5,nuclear_density_L1=l1<1e-3,
        energy=max(abs(o['energy_drift']) for o in obs)<c.get('energy_tolerance',1e-5),
        boundary=max(max(o['edge_R'],o['edge_Q'],o['edge_x']) for o in obs)<1e-5)
    record=dict(status='SHORT_TIME_CHECK_PASS' if all(checks.values()) else 'SHORT_TIME_CHECK_FAIL',
        scope='same spatial grid dt/dt2 only; NOT spatial/basis/long-time convergence',checks=checks,
        campaign=campaign,case=case,omega=omega,g_cavity=g,dt_au=dt,time_fs=n*dt/AU_PER_FS,
        fidelity=fidelity,nuclear_density_L1=l1,
        energy_drifts=[o['energy_drift'] for o in obs],norms=[o['norm'] for o in obs],
        config_hash=fingerprint(c),source_hash=fingerprint(source_record()['physics_sources']),
        wall_seconds=time.monotonic()-wall)
    write_json(out/f'validation_{campaign}_{case}.json',record)
    print(json.dumps(record,indent=2))
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--campaign',choices=['pen','vsc'],required=True)
    p.add_argument('--case',choices=['zero','weak','strong','all'],default='all')
    p.add_argument('--gpu',action='store_true');p.add_argument('--duration-fs',type=float,default=.02)
    a=p.parse_args();c=configuration(json.loads(a.config.read_text()))
    cases=['zero','weak','strong'] if a.case=='all' else [a.case]
    results=[validate(c,a.out,campaign=a.campaign,case=x,gpu=a.gpu,duration_fs=a.duration_fs) for x in cases]
    if not all(r['status']=='SHORT_TIME_CHECK_PASS' for r in results):raise SystemExit(2)


if __name__=='__main__':main()
