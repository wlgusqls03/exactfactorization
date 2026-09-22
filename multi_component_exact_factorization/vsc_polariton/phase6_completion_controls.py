"""Portable GPU 2D-A/B/C controls and exact free-photon stationary sector.

No physical parameters changed. Arrays are (R,x,n), reduced x dimension=1;
all units atomic. Reuses the SAME tested fourth-order PFBackend map.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .phase6_gpu_backend import PFBackend
from .run_phase6_gpu import load_packet, save_json, save_npz, absolute_failures


def reduced_packet(packet, molecular, model):
    """Project ground BO: A scalar DSE; B adds variance; C also adds DBOC.

    DBOC = ||d_R phi0||²/(2M) - |<phi0|d_R phi0>|²/(2M), historical
    central-five derivative in the interior. Periodic boundary stencil is
    localized to the edge strips (whose probability is gated). A global FFT
    derivative of a nonperiodic aligned BO gauge is NOT used for this field.
    """
    p = dict(packet)
    dx, dr, mass = map(float, (p['dx'], p['dR'], p['mass']))
    phi = p['phi'][:, :, 0]
    u = np.einsum('rx,rxn->rn', phi.conj(), p['psi'])*dx
    if abs(np.sum(abs(u)**2)*dr-1) > 1e-10:
        raise RuntimeError('Initial BO projection mismatch')
    np.testing.assert_allclose(molecular['R'], p['R'], rtol=0, atol=1e-12)
    mu = np.sum(abs(phi)**2*p['mu'], axis=1)*dx
    mu2 = np.sum(abs(phi)**2*p['mu']**2, axis=1)*dx
    variance = mu2-mu**2
    dp = (np.roll(phi,2,axis=0)-8*np.roll(phi,1,axis=0)
          +8*np.roll(phi,-1,axis=0)-np.roll(phi,-2,axis=0))/(12*dr)
    connection = np.sum(phi.conj()*dp, axis=1)*dx
    dboc = (np.sum(abs(dp)**2, axis=1)*dx-abs(connection)**2)/(2*mass)
    correction = np.zeros(len(phi))
    if model in ('B', 'C'):
        correction += float(p['g_chi'])**2/float(p['omega'])*variance
    if model == 'C':
        correction += dboc
    if model not in ('A', 'B', 'C'):
        raise ValueError(model)
    p.update(psi=u[:, None, :], x=np.array([0.]), dx=np.array(1.),
             tx=np.array([0.]), phi=np.ones((len(phi), 1, 1)),
             potential=(molecular['E']+correction)[:, None],
             mu=molecular['mu'][:, None],
             dse=(float(p['g_chi'])**2/float(p['omega'])*molecular['mu']**2)[:, None])
    return p, dict(variance=variance, DBOC=dboc, correction=correction)


def control_run(packet, dt, out, device):
    """128-step CPU/GPU equivalence then 1652 au; compact observables only."""
    # dt strings contain a dot: with_suffix would collapse .125 and .0625!
    final = Path(str(out)+'.npz')
    meta = Path(str(out)+'.json')
    if meta.exists():
        result = json.loads(meta.read_text())
        if result['status'] != 'PASS' or not final.exists():
            raise RuntimeError('Preserve failed/incomplete control: '+str(out))
        return result
    cpu, gpu = PFBackend(packet, dt), PFBackend(packet, dt, True, device)
    a, b = packet['psi'].copy(), gpu.xp.asarray(packet['psi'])
    for _ in range(128):
        a, b = cpu.step(a), gpu.step(b)
    error = float(np.linalg.norm(a-gpu.host(b))*np.sqrt(cpu.volume))
    if error >= 1e-8:
        raise RuntimeError(('Reduced CPU/GPU', error))
    u = gpu.xp.asarray(packet['psi']); records = []; e0 = None
    total, every = round(1652/dt), round(4/dt)
    failures = {}
    for step in range(total+1):
        if step % every == 0 or step == total:
            row = gpu.observe(u)
            if e0 is None:
                e0 = row['energy']
            failures = absolute_failures(row, e0)
            records.append(dict(time_au=step*dt, **row))
            if failures:
                break
        if step < total:
            u = gpu.step(u)
    save_npz(final, **{k:np.array([r[k] for r in records]) for k in records[0]})
    result = dict(status='FAIL' if failures else 'PASS', failures=failures,
                  backend_wave_error=error, dt=dt, time_au=step*dt,
                  interpretation='Reduced diagnostic model, not full3D')
    save_json(meta, result)
    if failures:
        raise RuntimeError(result)
    return result


def stationary(packet, out, device, gpu=True):
    """Exact invariant n=0 free-photon sector on FULL production x,R grids.

    Not a coupled-cavity stationary-state certification. This avoids any
    spatial truncation, while checking the identical full-H free-case map.
    """
    if out.exists():
        result = json.loads(out.read_text())
        if result['status'] != 'PASS':
            raise RuntimeError(result)
        return
    if float(packet['g_chi']) != 0:
        raise ValueError('Requires decoupled photon')
    if gpu:
        from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
    else:
        from scipy.sparse.linalg import LinearOperator, eigsh
    p = dict(packet)
    p.update(psi=packet['psi'][:, :, :1], photon=packet['photon'][:1],
             rotation=np.ones((1, 1)), displacement=np.zeros(1))
    h = PFBackend(p, .125, gpu, device); xp = h.xp
    size = p['psi'].size
    op = LinearOperator((size, size), matvec=lambda v:h.action(v.reshape(h.shape)).ravel(),
                        dtype=xp.complex128)
    values, vec = eigsh(op, k=1, which='SA', tol=1e-11, maxiter=20000,
                        v0=xp.asarray(p['psi']).ravel())
    u = vec[:, 0].reshape(h.shape)/np.sqrt(h.volume)
    residual = float(h.host(xp.linalg.norm(h.action(u)-values[0]*u)))*np.sqrt(h.volume)
    initial = h.observe(u); density = abs(u)**2
    for _ in range(128):
        u = h.step(u)
    final = h.observe(u)
    l1 = float(h.host(xp.sum(abs(abs(u)**2-density))))*h.volume
    result = dict(status='PASS' if residual < 1e-9 and l1 < 1e-6
                  and abs(final['norm']-1) < 1e-9
                  and abs(final['energy']-initial['energy']) < 1e-6 else 'FAIL',
                  eigen_residual=residual, density_L1=l1, final_norm=final['norm'],
                  energy_drift=final['energy']-initial['energy'], steps=128, dt=.125,
                  scope='Full production spatial grid, exact eta=0 n=0 invariant sector; not coupled GS')
    save_json(out, result)
    if result['status'] != 'PASS':
        raise RuntimeError(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    with np.load(a.inputs/'molecular.npz') as z:
        molecular = {k:z[k] for k in z.files}
    free, _ = load_packet(a.inputs/'free_F120.npz')
    stationary(free, a.out/'stationary.json', a.device)
    for case in ('free', 'resonant', 'barrier'):
        packet, _ = load_packet(a.inputs/f'{case}_F120.npz')
        for model in ('A', 'B', 'C'):
            reduced, fields = reduced_packet(packet, molecular, model)
            fields_path = a.out/f'{case}_{model}_projection.npz'
            if not fields_path.exists():
                save_npz(fields_path, R=packet['R'], **fields)
            for dt in (.125, .0625):
                print('CONTROL', case, model, dt, flush=True)
                control_run(reduced, dt, a.out/f'{case}_{model}_dt{dt}', a.device)
            fine_packet, _ = load_packet(a.inputs/f'{case}_F160.npz')
            reduced_fine, _ = reduced_packet(fine_packet, molecular, model)
            control_run(reduced_fine, .125, a.out/f'{case}_{model}_F160_dt0.125', a.device)


if __name__ == '__main__':
    main()
