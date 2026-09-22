"""Version-aware stationary check; same full Hamiltonian and tolerances.

Psi shape (R,x,1), atomic units, exact decoupled photon vacuum sector.
Only the eigensolver initialization API differs from the historical helper.
"""
import inspect
import json
import numpy as np
from .phase6_gpu_backend import PFBackend
from .run_phase6_gpu import save_json


def ground_pair(eigsh, negative_operator, initial):
    """Largest algebraic eigenpair of -H gives the ground eigenpair of H.

    LA is supported by old and new CuPy. LM is NOT equivalent: the largest
    magnitude eigenvalue may correspond to a high-energy state of H.
    Only eigensolver initialization changes when v0 is unavailable.
    """
    options = dict(k=1, which='LA', tol=1e-11, maxiter=20000)
    supported = 'v0' in inspect.signature(eigsh).parameters
    if supported:
        options['v0'] = initial
    values, vectors = eigsh(negative_operator, **options)
    return -values, vectors, supported


def stationary(packet, out, device=0, gpu=True):
    if out.exists():
        if json.loads(out.read_text())['status'] != 'PASS':
            raise RuntimeError('Preserved stationary failure: '+str(out))
        return
    if float(packet['g_chi']) != 0:
        raise ValueError('Only exact eta=0 vacuum sector')
    if gpu:
        from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
    else:
        from scipy.sparse.linalg import LinearOperator, eigsh
    p = dict(packet, psi=packet['psi'][:,:,:1], photon=packet['photon'][:1],
             rotation=np.ones((1,1)), displacement=np.zeros(1))
    h = PFBackend(p,.125,gpu,device); xp=h.xp
    op = LinearOperator((p['psi'].size,)*2,
        matvec=lambda v:-h.action(v.reshape(h.shape)).ravel(), dtype=xp.complex128)
    eigenvalues, vectors, used_v0 = ground_pair(eigsh,op,xp.asarray(p['psi']).ravel())
    u = vectors[:,0].reshape(h.shape)/np.sqrt(h.volume)
    residual = float(h.host(xp.linalg.norm(h.action(u)-eigenvalues[0]*u)))*np.sqrt(h.volume)
    initial=h.observe(u); density=abs(u)**2
    for _ in range(128):
        u=h.step(u)
    final=h.observe(u)
    l1=float(h.host(xp.sum(abs(abs(u)**2-density))))*h.volume
    passed=(residual<1e-9 and l1<1e-6 and abs(final['norm']-1)<1e-9
            and abs(final['energy']-initial['energy'])<1e-6)
    save_json(out,dict(status='PASS' if passed else 'FAIL',eigen_residual=residual,
        density_L1=l1,final_norm=final['norm'],energy_drift=final['energy']-initial['energy'],
        steps=128,dt=.125,eigensolver_uses_v0=used_v0,
        eigensolver_target='LA of -H; eigenvalue sign restored; residual and propagation use H',
        scope='Full selected production x,R grid; exact eta=0 n=0 sector, not coupled GS'))
    if not passed:
        raise RuntimeError('Stationary validation failed; results preserved')
