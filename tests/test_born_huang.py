import unittest
from unittest.mock import patch
import sys
import tempfile
from pathlib import Path

import numpy as np

from multi_component_exact_factorization.born_huang import (
    build_born_huang_basis,
    forward_overlap_links,
    load_or_build_born_huang_basis,
    coefficient_vector_potential,
    projected_plain_second,
    projected_link_derivatives,
    projected_residual_momentum,
    projected_residual_square,
    reconstruct_electronic_grid,
)
from multi_component_exact_factorization.core import (
    build_model, covariant_square, derivative,
)
from multi_component_exact_factorization.propagate import parse_args
from multi_component_exact_factorization.compare import psi_from_archive


class BornHuangOperatorTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.nx, self.nq, self.nR = 3, 5, 17, 13
        self.dq = 2.0*np.pi/self.nq
        self.dR = 2.0*np.pi/self.nR
        q = np.arange(self.nq)*self.dq
        R = np.arange(self.nR)*self.dR
        self.c = np.empty((self.ns, self.nq, self.nR), complex)
        for state in range(self.ns):
            self.c[state] = (0.3+0.1*state)*np.exp(
                1j*((state+1)*q[:, None]+0.2*np.sin(R)[None, :])
            )
        shape = (self.ns, self.ns, self.nq, self.nR)
        self.zero = np.zeros(shape, complex)
        self.vector = 0.1*np.sin(q)[:, None]+0.03*np.cos(R)[None, :]

    def test_zero_nac_matches_grid_formula(self):
        momentum = projected_residual_momentum(
            self.c, self.zero, self.vector, self.dq, 1
        )
        expected = -1j*derivative(self.c, self.dq, 1)-self.vector[None]*self.c
        self.assertTrue(np.allclose(momentum, expected, atol=2.0e-13))
        square = projected_residual_square(
            self.c, self.zero, self.zero, self.vector, self.dq, 1
        )
        expected_square = (
            -derivative(self.c, self.dq, 1, order=2)
            +1j*derivative(self.vector, self.dq, 0)[None]*self.c
            +2j*self.vector[None]*derivative(self.c, self.dq, 1)
            +self.vector[None]**2*self.c
        )
        self.assertTrue(np.allclose(square, expected_square, atol=2.0e-13))

    def test_plain_second_and_reconstruction(self):
        actual = projected_plain_second(
            self.c, self.zero, self.zero, self.dR, 2
        )
        self.assertTrue(np.allclose(
            actual, derivative(self.c, self.dR, 2, order=2), atol=2.0e-13
        ))
        states = np.zeros((self.ns, self.nx, self.nq, self.nR), complex)
        states[:, :self.ns] = np.eye(self.ns)[:, :, None, None]
        reconstructed = reconstruct_electronic_grid(self.c, states)
        self.assertTrue(np.allclose(reconstructed[:self.ns], self.c))

        lam = np.ones((1, self.nq, self.nR), complex)
        chi = np.ones((1, self.nR), complex)
        archive = {
            "electronic_coefficients": self.c[None],
            "bo_basis_states": states,
            "lambda_wavefunction": lam,
            "chi": chi,
        }
        self.assertTrue(np.allclose(
            psi_from_archive(archive, 0), reconstructed
        ))

    def test_vector_potential_is_real(self):
        norm = np.sqrt(np.sum(np.abs(self.c)**2, axis=0))
        normalized = self.c/norm[None]
        value = coefficient_vector_potential(
            normalized, self.zero, self.dq, 1
        )
        self.assertTrue(np.isrealobj(value))
        self.assertTrue(np.all(np.isfinite(value)))

    def test_projected_operators_match_complete_grid_basis(self):
        ns = nx = 2
        nq, nR = 101, 7
        dq = 2.0*np.pi/nq
        q = np.arange(nq)*dq
        angle = 0.3*np.sin(q)
        states = np.zeros((ns, nx, nq, nR), complex)
        states[0, 0] = np.cos(angle)[:, None]
        states[0, 1] = np.sin(angle)[:, None]
        states[1, 0] = -np.sin(angle)[:, None]
        states[1, 1] = np.cos(angle)[:, None]
        d = np.empty((ns, ns, nq, nR), complex)
        D = np.empty_like(d)
        for right in range(ns):
            first = derivative(states[right], dq, axis=1)
            second = derivative(states[right], dq, axis=1, order=2)
            d[:, right] = np.einsum("lxqr,xqr->lqr", np.conj(states), first)
            D[:, right] = np.einsum("lxqr,xqr->lqr", np.conj(states), second)
        c = np.empty((ns, nq, nR), complex)
        c[0] = 0.8*np.exp(0.2j*np.sin(q))[:, None]
        c[1] = 0.6*np.exp(-0.1j*np.cos(q))[:, None]
        vector = 0.07*np.cos(q)[:, None]*np.ones((1, nR))
        phi = reconstruct_electronic_grid(c, states)
        direct_first = -1j*derivative(phi, dq, axis=1)-vector[None]*phi
        projected_first = np.einsum(
            "lxqr,xqr->lqr", np.conj(states), direct_first
        )
        actual_first = projected_residual_momentum(c, d, vector, dq, 1)
        # A finite-difference stencil has no exact Leibniz rule; the projected
        # Born--Huang formula and direct-grid product agree to the expected
        # fourth-order truncation error rather than machine precision.
        self.assertLess(np.max(np.abs(actual_first-projected_first)), 5.0e-7)
        direct_second = covariant_square(
            phi, vector[None], dq, axis=1, sign=-1
        )
        projected_second = np.einsum(
            "lxqr,xqr->lqr", np.conj(states), direct_second
        )
        actual_second = projected_residual_square(c, d, D, vector, dq, 1)
        self.assertLess(np.max(np.abs(
            actual_second-projected_second
        )), 3.0e-7)

        link1 = forward_overlap_links(states, 2, 1, 1.0)
        link2 = forward_overlap_links(states, 2, 2, 1.0)
        link_first, link_second = projected_link_derivatives(
            c, link1, link2, dq, 1
        )
        direct_plain_first = derivative(phi, dq, axis=1)
        direct_plain_second = derivative(phi, dq, axis=1, order=2)
        expected_first = np.einsum(
            "lxqr,xqr->lqr", np.conj(states), direct_plain_first
        )
        expected_second = np.einsum(
            "lxqr,xqr->lqr", np.conj(states), direct_plain_second
        )
        self.assertTrue(np.allclose(
            link_first, expected_first, atol=2.0e-13
        ))
        self.assertTrue(np.allclose(
            link_second, expected_second, atol=2.0e-13
        ))
        link_cov_first, link_cov_second = projected_link_derivatives(
            c, link1, link2, dq, 1, vector=vector
        )
        self.assertLess(np.max(np.abs(
            -1j*link_cov_first-projected_first
        )), 2.0e-6)
        self.assertLess(np.max(np.abs(
            -link_cov_second-projected_second
        )), 2.0e-6)

    def test_overlap_link_derivatives_have_exact_discrete_adjoints(self):
        rng = np.random.default_rng(20260811)
        states = rng.normal(
            size=(self.ns, self.nx, self.nq, self.nR)
        )
        # Orthonormalize the electronic columns independently at every q/R.
        for iq in range(self.nq):
            for iR in range(self.nR):
                qmat, _ = np.linalg.qr(states[:, :, iq, iR].T)
                states[:, :, iq, iR] = qmat.T
        link1 = forward_overlap_links(states, 2, 1, 1.0)
        link2 = forward_overlap_links(states, 2, 2, 1.0)
        u = rng.normal(size=self.c.shape)+1j*rng.normal(size=self.c.shape)
        v = rng.normal(size=self.c.shape)+1j*rng.normal(size=self.c.shape)
        d1u, d2u = projected_link_derivatives(
            u, link1, link2, self.dq, 1
        )
        d1v, d2v = projected_link_derivatives(
            v, link1, link2, self.dq, 1
        )
        self.assertLess(abs(np.vdot(u, d1v)+np.vdot(d1u, v)), 2.0e-11)
        self.assertLess(abs(np.vdot(u, d2v)-np.vdot(d2u, v)), 2.0e-10)
        vector = 0.2*np.sin(
            np.arange(self.nq)*self.dq
        )[:, None]*np.ones((1, self.nR))
        cov1u, cov2u = projected_link_derivatives(
            u, link1, link2, self.dq, 1, vector=vector
        )
        cov1v, cov2v = projected_link_derivatives(
            v, link1, link2, self.dq, 1, vector=vector
        )
        self.assertLess(
            abs(np.vdot(u, cov1v)+np.vdot(cov1u, v)), 2.0e-11
        )
        self.assertLess(
            abs(np.vdot(u, cov2v)-np.vdot(cov2u, v)), 2.0e-10
        )

    def test_local_basis_nacs_are_finite_and_normalized(self):
        with patch.object(sys, "argv", [
            "test", "--nx", "18", "--nq", "7", "--nR", "6",
            "--electron-excitation", "1", "--heavy-trap-alpha", "0",
        ]):
            args = parse_args()
        model = build_model(args)
        basis = build_born_huang_basis(model, 2)
        overlap = np.einsum(
            "lxqr,jxqr->ljqr", np.conj(basis.states), basis.states
        )*model.dx
        identity = np.eye(2)[:, :, None, None]
        self.assertLess(np.max(np.abs(overlap-identity)), 2.0e-13)
        for values in (basis.energies, basis.d_q, basis.D_q, basis.d_R, basis.D_R):
            self.assertTrue(np.all(np.isfinite(values)))

    def test_basis_cache_round_trip_and_fingerprint(self):
        with patch.object(sys, "argv", [
            "test", "--nx", "12", "--nq", "5", "--nR", "5",
            "--electron-excitation", "1", "--heavy-trap-alpha", "0",
        ]):
            args = parse_args()
        model = build_model(args)
        with tempfile.TemporaryDirectory() as work:
            first, first_info = load_or_build_born_huang_basis(
                model, 2, cache_dir=Path(work)
            )
            second, second_info = load_or_build_born_huang_basis(
                model, 2, cache_dir=Path(work)
            )
            self.assertFalse(first_info["hit"])
            self.assertTrue(second_info["hit"])
            self.assertEqual(first_info["key"], second_info["key"])
            for name in (
                "energies", "states", "d_q", "D_q", "d_R", "D_R",
                "link_q1", "link_q2", "link_R1", "link_R2",
            ):
                self.assertTrue(np.array_equal(
                    np.asarray(getattr(first, name)),
                    np.asarray(getattr(second, name)),
                ))
            for name, axis, offset in (
                ("link_q1", 2, 1), ("link_q2", 2, 2),
                ("link_R1", 3, 1), ("link_R2", 3, 2),
            ):
                expected = forward_overlap_links(
                    np.asarray(first.states), axis, offset, model.dx
                )
                self.assertTrue(np.allclose(
                    np.asarray(getattr(first, name)), expected,
                    atol=2.0e-13,
                ))

    def test_smaller_basis_reuses_cached_superset(self):
        with patch.object(sys, "argv", [
            "test", "--nx", "12", "--nq", "5", "--nR", "5",
            "--electron-excitation", "1", "--heavy-trap-alpha", "0",
        ]):
            args = parse_args()
        model = build_model(args)
        with tempfile.TemporaryDirectory() as work:
            large, large_info = load_or_build_born_huang_basis(
                model, 3, cache_dir=Path(work)
            )
            with patch(
                "multi_component_exact_factorization.born_huang."
                "build_born_huang_basis",
                side_effect=AssertionError("superset cache was not reused"),
            ):
                small, small_info = load_or_build_born_huang_basis(
                    model, 2, cache_dir=Path(work)
                )
            self.assertFalse(large_info["hit"])
            self.assertTrue(small_info["hit"])
            self.assertEqual(small_info["stored_states"], 3)
            self.assertEqual(small.energies.shape[0], 2)
            self.assertEqual(small.d_q.shape[:2], (2, 2))
            self.assertTrue(np.array_equal(
                np.asarray(small.states), np.asarray(large.states[:2])
            ))


if __name__ == "__main__":
    unittest.main()
