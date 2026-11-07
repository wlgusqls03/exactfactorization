"""Exact-algebra performance paths versus the original dense channel adapter.

No timing assertions: CPU contention must not turn a numerical regression test
into a flaky speed test. The independent dense oracle retains the previous
matrix products and complex phase alignment, including arbitrary phase gauges.
"""
from types import MethodType
import unittest

import numpy as np
from scipy.linalg import block_diag, eigh
from scipy.optimize import linear_sum_assignment

from multi_component_exact_factorization.gaussian_trajectory.channel_run import ChannelConfig, ChannelEngine
from multi_component_exact_factorization.gaussian_trajectory.internal_channels import Channels, InternalChannels
from multi_component_exact_factorization.gaussian_trajectory.models import InternalModel, ModelConfig


def dense_reference(adapter, R, reference=None):
    """Pre-optimization dense evaluation, deliberately separate from helpers."""
    potential = adapter.model.potential(np.atleast_1d(R))
    gradient = adapter.model.potential_gradient(np.atleast_1d(R))
    local_matrix = (lambda value: block_diag(*value)) if adapter.model.is_photon else (
        lambda value: np.diag(value.ravel()))
    H = np.array([local_matrix(v) for v in potential])+adapter.kinetic[None]
    dH = np.array([local_matrix(v) for v in gradient])
    energies, vectors, forces, nacs = [], [], [], []
    minimum = 1.
    for i, (h, dh) in enumerate(zip(H, dH)):
        E, U = eigh(h, check_finite=True)
        if reference is not None:
            overlap = reference[i].conj().T@U
            row, col = linear_sum_assignment(-abs(overlap)**2)
            order = col[np.argsort(row)]
            E, U = E[order], U[:, order]
            diagonal = np.diag(reference[i].conj().T@U)
            minimum = min(minimum, float(np.min(abs(diagonal))))
            U = U.astype(complex)*np.exp(-1j*np.angle(diagonal))[None]
        derivative = U.conj().T@dh@U
        gap = E[None, :]-E[:, None]
        d = np.zeros_like(derivative, dtype=complex)
        resolved = abs(gap)>adapter.gap_tolerance
        d[resolved] = derivative[resolved]/gap[resolved]
        np.fill_diagonal(d, 0.)
        energies.append(E); vectors.append(U); forces.append(-derivative); nacs.append(d)
    return Channels(np.array(energies), np.array(vectors), np.array(forces), np.array(nacs), minimum)


def models():
    return [InternalModel(ModelConfig(name='pen-nonadiabatic', nq=16, dse=True)),
            InternalModel(ModelConfig(name='pen-rabi', nq=16)),
            InternalModel(ModelConfig(name='mcef', nq=8, nx=8, qmin=-6., qmax=-2.))]


class OptimizedChannelTests(unittest.TestCase):
    def test_dense_matrix_interface_matches_legacy_block_assembly(self):
        for model in models():
            with self.subTest(model=model.name):
                adapter = InternalChannels(model)
                R = model.initial_R_center+np.array([-.13, .05])
                H, dH = adapter.matrices(R)
                expected = dense_reference(adapter, R)
                local = (lambda value: block_diag(*value)) if model.is_photon else (
                    lambda value: np.diag(value.ravel()))
                np.testing.assert_array_equal(H, np.array([local(v) for v in model.potential(R)])+adapter.kinetic[None])
                np.testing.assert_array_equal(dH, np.array([local(v) for v in model.potential_gradient(R)]))
                np.testing.assert_allclose(np.linalg.eigvalsh(H), expected.energy, atol=4e-14)

    def test_local_derivative_action_matches_dense_for_complex_vectors(self):
        rng = np.random.default_rng(120)
        for model in models():
            with self.subTest(model=model.name):
                adapter = InternalChannels(model)
                R = np.array([model.initial_R_center+.08])
                gradient = model.potential_gradient(R)[0]
                U = rng.normal(size=(adapter.dimension, adapter.dimension))
                U = U+1j*rng.normal(size=U.shape)
                expected = adapter.matrices(R)[1][0]@U
                np.testing.assert_allclose(adapter._local_action(gradient, U), expected, atol=2e-14)

    def test_real_continuation_matches_previous_complex_formula(self):
        for model in models():
            with self.subTest(model=model.name):
                adapter = InternalChannels(model)
                R = model.initial_R_center+np.array([-.13, .05])
                reference = adapter.evaluate(R).vectors
                reference[:, :, ::2] *= -1.
                actual = adapter.evaluate(R+1e-4, reference)
                expected = dense_reference(adapter, R+1e-4, reference)
                self.assertTrue(np.isrealobj(actual.vectors))
                self.assertTrue(np.isrealobj(actual.force_matrix))
                for key in ('energy', 'vectors', 'force_matrix', 'nac'):
                    np.testing.assert_allclose(getattr(actual, key), getattr(expected, key), atol=4e-13, err_msg=key)
                self.assertAlmostEqual(actual.minimum_overlap, expected.minimum_overlap, places=14)

    def test_genuine_complex_reference_gauge_is_preserved(self):
        rng = np.random.default_rng(121)
        for model in models():
            with self.subTest(model=model.name):
                adapter = InternalChannels(model)
                R = model.initial_R_center+np.array([-.13, .05])
                reference = adapter.evaluate(R).vectors
                phase = np.exp(1j*rng.uniform(-np.pi, np.pi, (2, adapter.dimension)))
                actual = adapter.evaluate(R+1e-4, reference*phase[:, None, :])
                real = adapter.evaluate(R+1e-4, reference)
                expected = dense_reference(adapter, R+1e-4, reference*phase[:, None, :])
                self.assertTrue(np.iscomplexobj(actual.vectors))
                np.testing.assert_allclose(actual.vectors, real.vectors*phase[:, None, :], atol=3e-14)
                factors = phase.conj()[:, :, None]*phase[:, None, :]
                np.testing.assert_allclose(actual.force_matrix, real.force_matrix*factors, atol=4e-14)
                np.testing.assert_allclose(actual.nac, real.nac*factors, atol=4e-13)
                for key in ('energy', 'vectors', 'force_matrix', 'nac'):
                    np.testing.assert_allclose(getattr(actual, key), getattr(expected, key), atol=4e-13, err_msg=key)

    def test_short_native_and_projected_dynamics_match_dense_oracle(self):
        for method in ('ctmqc', 'tsh'):
            for compression in ('native', 'gaussian'):
                with self.subTest(method=method, compression=compression):
                    config = ChannelConfig(model=dict(name='pen-nonadiabatic', nq=32),
                        ntraj=7, method=method, compression=compression, dt_au=.0025)
                    fast, old = ChannelEngine(config), ChannelEngine(config)
                    old.channels.evaluate = MethodType(dense_reference, old.channels)
                    a, _ = fast.initial(); b, _ = old.initial()
                    for _ in range(4):
                        a, b = fast.step(a), old.step(b)
                    for key in a:
                        np.testing.assert_allclose(a[key], b[key], atol=3e-13, rtol=2e-12, err_msg=key)
                    np.testing.assert_array_equal(fast.active, old.active)
                    self.assertEqual(fast.hops, old.hops)
                    self.assertEqual(fast.rng.bit_generator.state, old.rng.bit_generator.state)


if __name__ == '__main__':
    unittest.main()
