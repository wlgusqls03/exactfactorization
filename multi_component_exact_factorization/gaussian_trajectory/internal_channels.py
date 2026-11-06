"""Complete internal eigenbasis adapter for B3--B5 (not a 4-state truncation).

G[I,q,k] remains in the common primitive representation. Eigenvectors are used
only for native CTMQC/FSSH rates; they include the quantum q kinetic operator.
Overlap continuation fixes channel labels/phases, including on restart. It is
not an extra nonadiabatic force. Degenerate, derivative-coupled subspaces need
a separate derivation and are rejected rather than divided by a small gap.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, eigh
from scipy.optimize import linear_sum_assignment


@dataclass
class Channels:
    energy: np.ndarray
    vectors: np.ndarray
    force_matrix: np.ndarray
    nac: np.ndarray
    minimum_overlap: float

    def project(self, G, quadrature):
        weighted = (G*np.sqrt(quadrature)[None, :, None]).reshape(len(G), -1)
        return np.einsum('ijn,ij->in', self.vectors.conj(), weighted)

    def lift(self, A, quadrature, nk):
        weighted = np.einsum('ijn,in->ij', self.vectors, A)
        return weighted.reshape(len(A), len(quadrature), nk)/np.sqrt(quadrature)[None, :, None]


class InternalChannels:
    def __init__(self, model, max_dimension=256, gap_tolerance=1e-10,
                 derivative_tolerance=1e-9, minimum_overlap=.25):
        self.model = model
        self.dimension = len(model.q)*model.nk
        if self.dimension > max_dimension:
            raise ValueError('Complete internal dimension %d exceeds dense-pilot limit %d; '
                             'no implicit state truncation. Use an explicit smaller pilot '
                             'or increase max_internal_dimension after a resource check.' %
                             (self.dimension, max_dimension))
        self.gap_tolerance = gap_tolerance
        self.derivative_tolerance = derivative_tolerance
        self.minimum_overlap = minimum_overlap
        ident = np.eye(len(model.q))
        Tq = np.fft.ifft(model.kinetic_q[:, None]*np.fft.fft(ident, axis=0), axis=0)
        if np.max(abs(Tq.imag)) > 1e-10:
            raise ValueError('Internal kinetic matrix is unexpectedly complex')
        self.kinetic = np.kron(Tq.real, np.eye(model.nk))
        if not model.is_photon:
            self.kinetic += np.kron(ident, model.Tx)

    def matrices(self, R):
        potential = self.model.potential(np.atleast_1d(R))
        gradient = self.model.potential_gradient(np.atleast_1d(R))
        if self.model.is_photon:
            V = np.array([block_diag(*v) for v in potential])
            dV = np.array([block_diag(*dv) for dv in gradient])
        else:
            V = np.array([np.diag(v.ravel()) for v in potential])
            dV = np.array([np.diag(dv.ravel()) for dv in gradient])
        return V+self.kinetic[None], dV

    def evaluate(self, R, reference=None):
        H, dH = self.matrices(R)
        energies, vectors, forces, nacs = [], [], [], []
        min_overlap = 1.
        for i, (h, dh) in enumerate(zip(H, dH)):
            E, U = eigh(h, check_finite=True)
            if reference is not None:
                overlap = reference[i].conj().T@U
                row, col = linear_sum_assignment(-abs(overlap)**2)
                order = col[np.argsort(row)]
                E, U = E[order], U[:, order]
                diagonal = np.diag(reference[i].conj().T@U)
                min_overlap = min(min_overlap, float(np.min(abs(diagonal))))
                if np.any(abs(diagonal) < self.minimum_overlap):
                    raise ValueError('Ambiguous internal-channel continuation: reduce dt '
                                     'or resolve the degenerate subspace; no relabeling guess')
                U = U.astype(complex)*np.exp(-1j*np.angle(diagonal))[None]
            derivative = U.conj().T@dh@U
            gap = E[None, :]-E[:, None]
            near = abs(gap) <= self.gap_tolerance
            np.fill_diagonal(near, False)
            if np.any(abs(derivative[near]) > self.derivative_tolerance):
                raise ValueError('Degenerate channels have nonzero derivative coupling; '
                                 'a subspace treatment is required')
            d = np.zeros_like(derivative, dtype=complex)
            resolved = abs(gap)>self.gap_tolerance
            d[resolved] = derivative[resolved]/gap[resolved]
            np.fill_diagonal(d, 0.)
            energies.append(E); vectors.append(U); forces.append(-derivative); nacs.append(d)
        return Channels(np.array(energies), np.array(vectors), np.array(forces),
                        np.array(nacs), min_overlap)

    def initialize(self, R, max_spacing=.05):
        """Common channel labels continued from the ensemble's model center.

        Independently sorting each trajectory would give incompatible CTMQC
        history labels when levels cross. All paths start at the same basis.
        """
        center = self.model.initial_R_center
        base = self.evaluate([center]).vectors[0]
        refs = []
        for r in R:
            reference = base[None]
            count = max(1, int(np.ceil(abs(r-center)/max_spacing)))
            for point in np.linspace(center, r, count+1)[1:]:
                reference = self.evaluate([point], reference).vectors
            refs.append(reference[0])
        return self.evaluate(R, np.array(refs))
