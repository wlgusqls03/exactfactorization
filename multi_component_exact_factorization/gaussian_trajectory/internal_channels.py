"""Complete internal eigenbasis adapter for B3--B5 (not a 4-state truncation).

G[I,q,k] remains in the common primitive representation. Eigenvectors are used
only for native CTMQC/FSSH rates; they include the quantum q kinetic operator.
Overlap continuation fixes channel labels/phases, including on restart. It is
not an extra nonadiabatic force. Degenerate, derivative-coupled subspaces need
a separate derivation and are rejected rather than divided by a small gap.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
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
        Tq = model.kinetic_q_matrix()
        self.kinetic = np.kron(Tq, np.eye(model.nk))
        if not model.is_photon:
            self.kinetic += np.kron(ident, model.Tx)

    def matrices(self, R):
        """Dense audit matrices, preserving the public reference interface.

        Propagation uses the same local operator through ``_local_action``:
        constructing a dense derivative matrix is unnecessary when it is
        diagonal (MCEF) or block diagonal in q (PEN).
        """
        potential = self.model.potential(np.atleast_1d(R))
        gradient = self.model.potential_gradient(np.atleast_1d(R))
        return self._local_matrix(potential)+self.kinetic[None], self._local_matrix(gradient)

    def _local_matrix(self, values):
        """Embed the unchanged local potential without Python block loops."""
        values = np.asarray(values)
        matrix = np.zeros((len(values), self.dimension, self.dimension), dtype=values.dtype)
        if self.model.is_photon:
            nq, nk = len(self.model.q), self.model.nk
            rows = np.arange(nq)[:, None, None]*nk + np.arange(nk)[None, :, None]
            cols = np.arange(nq)[:, None, None]*nk + np.arange(nk)[None, None, :]
            matrix[:, rows, cols] = values
        else:
            index = np.arange(self.dimension)
            matrix[:, index, index] = values.reshape(len(values), -1)
        return matrix

    def _local_action(self, values, vectors):
        """Exactly apply local dH/dR to all columns; no state truncation.

        The final U^dagger product is still dense, but dH itself need not be.
        This changes only algebraic evaluation cost, not the force/NAC rule.
        """
        if self.model.is_photon:
            shaped = vectors.reshape(len(self.model.q), self.model.nk, self.dimension)
            return np.einsum('qkl,qln->qkn', values, shaped).reshape(vectors.shape)
        return values.reshape(-1, 1)*vectors

    def evaluate(self, R, reference=None):
        coordinates = np.atleast_1d(R)
        potential = self.model.potential(coordinates)
        gradient = self.model.potential_gradient(coordinates)
        H = self._local_matrix(potential)+self.kinetic[None]
        if not np.all(np.isfinite(gradient)):
            raise ValueError('Nonfinite internal Hamiltonian derivative')
        energies, vectors, forces, nacs = [], [], [], []
        min_overlap = 1.
        for i, (h, dh) in enumerate(zip(H, gradient)):
            E, U = eigh(h, check_finite=True)
            if reference is not None:
                overlap = reference[i].conj().T@U
                row, col = linear_sum_assignment(-abs(overlap)**2)
                order = col[np.argsort(row)]
                E, U = E[order], U[:, order]
                # Reuse the already computed overlap. A second dense matrix
                # product only to extract its diagonal has identical content.
                diagonal = overlap[np.arange(self.dimension), order]
                min_overlap = min(min_overlap, float(np.min(abs(diagonal))))
                if np.any(abs(diagonal) < self.minimum_overlap):
                    raise ValueError('Ambiguous internal-channel continuation: reduce dt '
                                     'or resolve the degenerate subspace; no relabeling guess')
                if np.isrealobj(U) and np.isrealobj(reference[i]):
                    # For the actual real Hamiltonians, parallel transport is
                    # exactly a sign. Avoid artificial complex arithmetic from
                    # exp(-i*pi). G and its coefficients remain fully complex.
                    U = U*np.where(diagonal < 0., -1., 1.)[None]
                else:
                    # Preserve genuine complex gauge references, including
                    # checkpoints produced by the previous complex path.
                    U = U.astype(complex)*np.exp(-1j*np.angle(diagonal))[None]
            derivative = U.conj().T@self._local_action(dh, U)
            gap = E[None, :]-E[:, None]
            near = abs(gap) <= self.gap_tolerance
            np.fill_diagonal(near, False)
            if np.any(abs(derivative[near]) > self.derivative_tolerance):
                raise ValueError('Degenerate channels have nonzero derivative coupling; '
                                 'a subspace treatment is required')
            d = np.zeros_like(derivative)
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
