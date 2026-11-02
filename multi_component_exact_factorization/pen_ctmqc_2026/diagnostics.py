"""Kernel estimates of DIRECT ensemble observables, not a reconstructed Psi.

POL uses a local *mixed density matrix*: sum_I K(R-R_I)|c_I><c_I|/N.
EL uses a phase-space ensemble and local electronic density matrices.
Kernel density is normalized on the display domain only up to tail loss;
that loss is recorded, not renormalized away. No positive-gauge wave is made.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from ..pen_perspectives_2026.model import electronic, local_hamiltonian, oscillator, cbo


def deposit(points, grids, weights, widths):
    """Histogram then fixed Gaussian smoothing; complex/vector weights allowed.

    This is display/postprocessing only. It NEVER changes the CTMQC Q or force.
    Use finer display grids/bandwidth sensitivity tests for local fine features.
    """
    points = np.asarray(points)
    weights = np.asarray(weights)
    shape = tuple(len(g) for g in grids)
    spacing = np.array([g[1]-g[0] for g in grids])
    index = np.floor((points-np.array([g[0] for g in grids]))/spacing+.5).astype(int)
    valid = np.all((index >= 0) & (index < np.array(shape)), axis=1)
    flat = np.ravel_multi_index(index[valid].T, shape)
    tail = weights.shape[1:]
    target = np.zeros((int(np.prod(shape)), int(np.prod(tail)) if tail else 1), dtype=weights.dtype)
    values = weights[valid].reshape(valid.sum(), target.shape[1])
    for k in range(target.shape[1]):
        if np.iscomplexobj(weights):
            target[:, k] = (np.bincount(flat, weights=values[:, k].real, minlength=len(target))
                            +1j*np.bincount(flat, weights=values[:, k].imag, minlength=len(target)))
        else:
            target[:, k] = np.bincount(flat, weights=values[:, k], minlength=len(target))
    target = target.reshape(shape+tail)/len(points)/np.prod(spacing)
    sigma = tuple(np.asarray(widths)/spacing)+(0,)*len(tail)
    smooth = lambda z: gaussian_filter(z, sigma, mode='constant', truncate=5.)
    return smooth(target.real)+1j*smooth(target.imag) if np.iscomplexobj(target) else smooth(target)


def ratio(x, y):
    x, y = np.broadcast_arrays(x, y)
    return np.divide(x, y, out=np.full_like(x, np.nan, dtype=np.result_type(x, float)), where=y > 1e-300)


def prepare_frame(c, s, frame, nr=241, nq=161):
    R = np.linspace(c.rmin, c.rmax, nr)
    # This is a display domain, not a photon propagation box. EL Wigner
    # samples broaden under KDE, so retain eight initial density deviations.
    qmax = max(10., 8/np.sqrt(2*c.omega))
    q = np.linspace(-qmax, qmax, nq)
    widths = np.array([c.sigma_r/np.sqrt(2), 1/np.sqrt(2*c.omega)])*s.bandwidth_scale
    X, P, coeff = frame['X'], frame['P'], frame['coeff']
    one = np.ones(len(X))
    rhoR = deposit(X[:, :1], [R], one, widths[:1])
    meanR = lambda weights: ratio(deposit(X[:, :1], [R], weights, widths[:1]), rhoR[(slice(None),)+(None,)*(weights.ndim-1)])
    density_matrix = coeff[:, :, None]*coeff[:, None, :].conj()
    if s.perspective == 'pol':
        dm = deposit(X[:, :1], [R], density_matrix, widths[:1]).reshape(nr, 2, c.nf, 2, c.nf)
        h = oscillator(c, q)[0]
        edm = np.einsum('renfm,nq,mq->rqef', dm, h, h)
        _, polv = np.linalg.eigh(local_hamiltonian(c, X[:, 0]))
        population = abs(np.einsum('nik,ni->nk', polv, coeff))**2
        state_density = deposit(X[:, :1], [R], population, widths[:1])
        labels = [fr'$|\mathrm{{Pol}}_{k}\rangle$' for k in range(2*c.nf)]
    else:
        edm = deposit(X, [R, q], density_matrix, widths)
        from .core import hamiltonian
        ev = np.linalg.eigh(hamiltonian(c, 'el', X)[0])[1]
        population = abs(np.einsum('nik,ni->nk', ev, coeff))**2
        state_density = deposit(X[:, :1], [R], population, widths[:1])
        labels = [fr'$\mathrm{{CBO}}_{k}$' for k in range(2)]
    joint = np.trace(edm, axis1=-2, axis2=-1).real
    ce, cv = cbo(c, R, q)
    local_population_density = np.einsum('rqek,rqef,rqfk->rqk', cv, edm, cv).real
    local_energy = ratio(np.sum(local_population_density*ce, axis=-1), joint)
    local_population = ratio(local_population_density, joint[:, :, None])
    rhoq = np.trapz(joint, R, axis=0)
    mass = float(np.trapz(np.trapz(joint, q, axis=1), R))
    return dict(R=R, q=q, joint=joint, rho_R=rhoR, rho_q=rhoq,
        conditional_probability=ratio(joint, rhoR[:, None]),
        local_CBO_energy=local_energy, local_CBO_population=local_population,
        mean_energy_R=meanR(frame['mean_energy']),
        mean_P_R=meanR(P[:, 0]), mf_force_R=meanR(frame['mf_force'][:, 0]),
        ct_force_R=meanR(frame['ct_force'][:, 0]),
        q_pair_R=meanR(frame['q_pair_rms'][:, 0]),
        state_density=state_density, state_labels=labels,
        cbo=ce, pol_energy=np.linalg.eigvalsh(local_hamiltonian(c, R)),
        t_fs=float(frame['t_fs']), display_mass=mass,
        interpretation='Mixed local ensemble density/conditional probabilities, NOT a pure wavefunction or exact EF.')
