"""Read-only saved-state photon-grid audit; NOT a new TDSE convergence claim.

Use Q=sqrt(omega)*q, dimensionless Hermite functions, and the reduced Gram
matrix G_nm=integral dx dR c_n* c_m. This exactly contracts global photon
quadratic diagnostics without storing an (R,x,Q) array. Periodic FFT
derivatives are compared against analytic Hermite derivatives of the SAME
finite-Fock state. No masking or renormalization is applied.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import simpson


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def hermite(Q, nf):
    """Return phi_n(Q), first/second derivatives: three arrays (nf,NQ)."""
    b = np.empty((nf + 2, len(Q)))
    b[0] = np.pi**(-.25) * np.exp(-Q**2 / 2)
    b[1] = np.sqrt(2) * Q * b[0]
    for n in range(1, nf + 1):
        b[n+1] = np.sqrt(2/(n+1))*Q*b[n] - np.sqrt(n/(n+1))*b[n-1]
    d = np.empty_like(b[:nf])
    dd = np.empty_like(d)
    for n in range(nf):
        d[n] = (np.sqrt(n)*b[n-1] if n else 0.) / np.sqrt(2) - np.sqrt((n+1)/2)*b[n+1]
        dd[n] = (Q**2 - (2*n+1))*b[n]
    return b[:nf], d, dd


def quadratic_density(G, B):
    """integral dx dR |sum c_n B_n|^2 at each Q; shape (NQ,)."""
    raw = np.einsum('ni,ni->i', B.conj(), G @ B).real
    if raw.min() < -1e-12:
        raise ValueError('Non-positive quadratic form beyond roundoff')
    return np.maximum(raw, 0.)


def grid_probe(G, half, nq):
    """Endpoint-excluded periodic grid; FFT vs exact sampled derivatives."""
    Q = np.linspace(-half, half, nq, endpoint=False)
    h = 2*half/nq
    B, d, dd = hermite(Q, len(G))
    S = B @ B.T * h
    back = S - np.eye(len(G))
    k = 2*np.pi*np.fft.fftfreq(nq, d=h)
    bf = np.fft.fft(B, axis=1)
    result = dict(half_Q=half, NQ=nq, dQ=h,
                  norm_error=abs(float(quadratic_density(G, B).sum()*h - np.trace(G).real)),
                  basis_gram_max_error=float(np.max(abs(back))),
                  backprojection_L2=float(np.sqrt(quadratic_density(G, back).sum())),
                  edge_probability=float(quadratic_density(G, B) [abs(Q) >= half-1].sum()*h))
    for order, exact in ((1,d),(2,dd)):
        numerical = np.fft.ifft(bf*(1j*k)**order, axis=1)
        error2 = quadratic_density(G, numerical-exact).sum()*h
        scale2 = quadratic_density(G, exact).sum()*h
        result[f'derivative{order}_relative_L2'] = float(np.sqrt(error2/scale2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path,
                        default=Path('results/vsc_polariton/phase7/photon_grid_audit_v1'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path('results/vsc_polariton')
    r7 = root/'phase7/results'
    jobs = []
    for name, directory, packet in (
        ('barrier_F240', r7/'phase7_barrier_f240_results/full',
         root/'phase6_gpu/barrier_f240_transfer_v1/inputs/barrier_F240.npz'),
        ('resonant_F200', r7/'phase7_f200_results/resonant/full',
         root/'phase6_gpu/f200_transfer_v1/inputs/resonant_F200.npz')):
        with np.load(packet) as z:
            omega, dx, dr = (float(z[k]) for k in ('omega','dx','dR'))
        for path in sorted(directory.glob('wave_*.npz')):
            jobs.append((name,path,omega,dx,dr))
    Q = np.linspace(-30,30,6001)
    probes = [(L,N) for L,N in ((8,256),(10,256),(12,256),(16,256),
               (20,256),(24,256),(24,384),(24,512),(24,768),(28,448),(28,560))]
    report = dict(scope='Saved-state representation audit only; NOT propagated grid/MCEF convergence.',
                  units='Q=sqrt(omega)*q; omega in Ha, hbar=1',
                  predeclared_representation_targets=dict(norm=1e-10,backprojection_L2=1e-8,
                        derivative_relative_L2=1e-6,edge_probability=1e-8),
                  frames=[], inputs={})
    densities = {'Q':Q}
    for name,path,omega,dx,dr in jobs:
        before = digest(path)
        with np.load(path) as z:
            psi = z['psi']
            t = float(z['time_au'])
        C = psi.reshape(-1,psi.shape[-1])
        G = C.conj().T @ C * dx*dr
        del C,psi
        B,d,dd = hermite(Q,len(G))
        rho = quadratic_density(G,B)
        d2rho = quadratic_density(G,dd)
        d2norm = simpson(d2rho,x=Q)
        row = dict(case=name,path=str(path),time_fs=t*.024188843265857,
                   omega=omega,nf=len(G),norm=float(np.trace(G).real),
                   fine_grid_norm_error=abs(float(simpson(rho,x=Q)-np.trace(G).real)),
                   tails={},grids=[])
        for L in (4,6,8,10,12,16,20,24,28):
            left=Q<=-L;right=Q>=L
            tail = lambda f: float(simpson(f[left],x=Q[left])+simpson(f[right],x=Q[right]))
            row['tails'][str(L)] = dict(probability=tail(rho),
                                  second_derivative_norm_fraction=tail(d2rho)/d2norm)
        for budget in (1e-6,1e-8,1e-10):
            # Locate minimum symmetric half box on the 0.01-Q analysis grid.
            radial = (rho[:3000]+rho[:3000:-1])*.01
            tails = np.cumsum(radial)
            idx = np.flatnonzero(tails<=budget)
            row[f'half_Q_for_tail_{budget:g}'] = float(30 - .01*idx[-1]) if len(idx) else 30.
        small = abs(Q)<=10*np.sqrt(omega)
        row['probability_inside_literal_q_minus10_plus10'] = float(simpson(rho[small],x=Q[small]))
        for L,N in probes:
            probe=grid_probe(G,L,N)
            probe.update(half_q=L/np.sqrt(omega),dq=2*L/N/np.sqrt(omega))
            row['grids'].append(probe)
        if before != digest(path):
            raise RuntimeError('Input modified: '+str(path))
        report['inputs'][str(path)]=before
        report['frames'].append(row)
        densities[name+'_'+path.stem]=rho
        print(name,path.stem,'tail Q10/Q16/Q24',
              [row['tails'][str(L)]['probability'] for L in (10,16,24)],flush=True)
        (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    report['input_hashes_unchanged']=True
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    np.savez_compressed(args.out/'photon_densities.npz',**densities)


if __name__ == '__main__':
    main()
