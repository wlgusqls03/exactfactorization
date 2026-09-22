"""Re-export immutable GPU packets with a more orthogonal Fock eigenbasis.

Same truncated operator Q=a+a†, same eigenvalues to roundoff, same PF H.
No Psi renormalization, no cutoff/tolerance change. Psi shape (NR,Nx,NF).
All units remain atomic. Main Li Eq.(1); only numerical diagonalization changes.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from .phase6_gpu_backend import PFBackend
from .run_phase6_gpu import load_packet


def orthogonal_rotation(n):
    q = np.diag(np.sqrt(np.arange(1, n)), 1)
    q = q + q.T
    values, vectors = eigh(q, driver='evd')
    diagnostics = dict(orthogonality=float(np.max(abs(vectors.T @ vectors-np.eye(n)))),
                       eigen_residual=float(np.max(abs(q @ vectors-vectors*values))))
    if diagnostics['orthogonality'] >= 1e-14 or diagnostics['eigen_residual'] >= 1e-12:
        raise RuntimeError(diagnostics)
    return values, vectors, diagnostics


def convert(source, target):
    source, target = Path(source), Path(target)
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError(target)
    packet, parent_sha = load_packet(source)
    meta = json.loads(str(packet['metadata']))
    n = packet['psi'].shape[-1]
    values, vectors, diagnostic = orthogonal_rotation(n)
    old = PFBackend(packet, .125)
    old_step = old.step(packet['psi'].copy())
    old_orth = float(np.max(abs(packet['rotation'].T @ packet['rotation']-np.eye(n))))
    updated = dict(packet, rotation=vectors, displacement=values, dt=np.array(.125))
    new = PFBackend(updated, .125)
    error = float(np.linalg.norm(new.step(packet['psi'].copy())-old_step)*np.sqrt(old.volume))
    if not np.isfinite(error) or error >= 1e-10:
        raise RuntimeError(('old/new step mismatch', error))
    # H.action depends on physical arrays, NOT rotation: these are byte-identical.
    changed = [k for k in packet if not np.array_equal(packet[k], updated[k])]
    if set(changed)-{'rotation', 'displacement', 'dt'}:
        raise RuntimeError(('unexpected change', changed))
    diagnostic.update(parent_input_sha256=parent_sha, old_orthogonality=old_orth,
                      old_new_step_L2=error, changed_arrays=changed,
                      method='scipy.linalg.eigh(driver=evd)', status='PASS',
                      full_interval_gpu_validation='PENDING')
    meta['orthogonal_rotation_validation'] = diagnostic
    # Preserve historical export validation as provenance, not a new GPU claim.
    meta['config']['setting'][-1] = .125
    updated['metadata'] = json.dumps(meta)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        np.savez_compressed(stream, **updated)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    report = dict(path=str(target), sha256=digest, bytes=target.stat().st_size, metadata=meta)
    with target.with_suffix('.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(path=str(target), sha256=digest, **diagnostic)), flush=True)


def main():
    root = Path(__file__).resolve().parents[2] / 'results/vsc_polariton/phase6_gpu'
    for n in (120, 160):
        convert(root/'inputs'/f'resonant_F{n}.npz',
                root/'orthogonal_transfer/inputs'/f'resonant_F{n}.npz')


if __name__ == '__main__':
    main()
