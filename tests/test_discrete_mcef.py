import unittest
import argparse
from types import SimpleNamespace

import numpy as np

from multi_component_exact_factorization_discrete.core import (
    discrete_born_huang_rhs,
    pnc_retract,
    reconstruct_coefficient_wavefunction,
)
from multi_component_exact_factorization.core import (
    add_model_arguments,
    build_model,
)
from multi_component_exact_factorization.born_huang import (
    build_born_huang_basis,
    forward_overlap_links,
)


def _problem(seed=7):
    rng = np.random.default_rng(seed)
    states, nq, nR = 3, 7, 8
    model = SimpleNamespace(
        dq=0.3, dR=0.25, proton_mass=5.0, heavy_mass=11.0,
    )
    energies = rng.normal(size=(states, nq, nR))
    identity = np.eye(states)[:, :, None, None]
    identity = np.broadcast_to(identity, (states, states, nq, nR)).copy()
    basis = SimpleNamespace(
        energies=energies,
        link_q1=identity.copy(), link_q2=identity.copy(),
        link_R1=identity.copy(), link_R2=identity.copy(),
    )
    coefficients = (
        rng.normal(size=(states, nq, nR))
        +1j*rng.normal(size=(states, nq, nR))
    )
    coefficients /= np.sqrt(np.sum(np.abs(coefficients)**2, axis=0))[None]
    lam = rng.normal(size=(nq, nR))+1j*rng.normal(size=(nq, nR))
    lam /= np.sqrt(np.sum(np.abs(lam)**2, axis=0)*model.dq)[None]
    chi = rng.normal(size=nR)+1j*rng.normal(size=nR)
    chi /= np.sqrt(np.sum(np.abs(chi)**2)*model.dR)
    return model, basis, coefficients, lam, chi


class DiscreteMCEFTests(unittest.TestCase):
    def test_real_shin_metiu_overlap_links_recombine(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        args = parser.parse_args([])
        args.nx, args.nq, args.nR = 32, 7, 8
        args.q_min, args.q_max = -3.0, 3.0
        args.R_min, args.R_max = -1.0, 5.0
        model = build_model(args)
        basis = build_born_huang_basis(model, 3)
        basis.link_q1 = forward_overlap_links(basis.states, 2, 1, model.dx)
        basis.link_q2 = forward_overlap_links(basis.states, 2, 2, model.dx)
        basis.link_R1 = forward_overlap_links(basis.states, 3, 1, model.dx)
        basis.link_R2 = forward_overlap_links(basis.states, 3, 2, model.dx)
        rng = np.random.default_rng(23)
        c = rng.normal(size=(3, 7, 8))+1j*rng.normal(size=(3, 7, 8))
        c /= np.sqrt(np.sum(np.abs(c)**2, axis=0))[None]
        lam = rng.normal(size=(7, 8))+1j*rng.normal(size=(7, 8))
        lam /= np.sqrt(np.sum(np.abs(lam)**2, axis=0)*model.dq)[None]
        chi = rng.normal(size=8)+1j*rng.normal(size=8)
        chi /= np.sqrt(np.sum(np.abs(chi)**2)*model.dR)
        result = discrete_born_huang_rhs(c, lam, chi, model, basis)
        self.assertLess(
            float(result.diagnostics["relative_unexplained_residual"]),
            5.0e-13,
        )

    def test_unmasked_factor_rhs_recombines_to_discrete_tdse(self):
        model, basis, c, lam, chi = _problem()
        result = discrete_born_huang_rhs(c, lam, chi, model, basis)
        self.assertLess(
            float(result.diagnostics["relative_unexplained_residual"]),
            2.0e-13,
        )
        self.assertLess(
            float(result.diagnostics["recombination_residual_l2"]),
            2.0e-12,
        )
        self.assertLess(
            float(result.diagnostics["max_raw_horizontal_phi"]),
            2.0e-13,
        )
        self.assertLess(
            float(result.diagnostics["max_raw_horizontal_lam"]),
            2.0e-13,
        )

    def test_flat_top_residual_is_exactly_explained(self):
        model, basis, c, lam, chi = _problem(11)
        lam = lam.copy()
        lam[:2] *= 1.0e-5
        result = discrete_born_huang_rhs(
            c, lam, chi, model, basis,
            flat_top_on_phi=2.0e-2,
            flat_top_on_lam=1.0e-2,
            transition_decades=2.0,
        )
        self.assertGreater(
            float(result.diagnostics["predicted_mask_residual_l2"]), 0.0
        )
        self.assertLess(
            float(result.diagnostics["relative_unexplained_residual"]),
            3.0e-13,
        )
        self.assertTrue(np.all(np.isfinite(result.dc)))
        self.assertTrue(np.all(np.isfinite(result.dlam)))

    def test_exact_marginal_nodes_have_finite_zero_inverse(self):
        model, basis, c, lam, chi = _problem(12)
        lam = lam.copy()
        chi = chi.copy()
        lam[0, 2] = 0.0
        chi[5] = 0.0
        # Check both the explicitly unmasked limit and an active flat-top.
        for onset in (0.0, 1.0e-3):
            result = discrete_born_huang_rhs(
                c, lam, chi, model, basis,
                flat_top_on_phi=onset,
                flat_top_on_lam=onset,
            )
            self.assertTrue(np.all(np.isfinite(result.dc)))
            self.assertTrue(np.all(np.isfinite(result.dlam)))
            self.assertTrue(np.all(np.isfinite(result.dchi)))

    def test_pnc_retraction_preserves_full_product(self):
        model, _, c, lam, chi = _problem(13)
        c = c*np.linspace(0.8, 1.2, c.shape[1])[:, None][None]
        lam = lam*np.linspace(0.7, 1.3, lam.shape[1])[None]
        before = reconstruct_coefficient_wavefunction(c, lam, chi)
        c2, lam2, chi2, diagnostics = pnc_retract(
            c, lam, chi, model.dq
        )
        after = reconstruct_coefficient_wavefunction(c2, lam2, chi2)
        self.assertLess(np.max(np.abs(after-before)), 3.0e-15)
        self.assertLess(np.max(np.abs(np.sum(np.abs(c2)**2, axis=0)-1.0)), 5e-15)
        self.assertLess(
            np.max(np.abs(np.sum(np.abs(lam2)**2, axis=0)*model.dq-1.0)),
            5e-15,
        )
        self.assertLess(float(diagnostics["max_product_change"]), 3.0e-15)

    def test_time_independent_nested_gauge_covariance(self):
        model, basis, c, lam, chi = _problem(17)
        original = discrete_born_huang_rhs(
            c, lam, chi, model, basis,
            flat_top_on_phi=1.0e-3,
            flat_top_on_lam=1.0e-3,
        )
        rng = np.random.default_rng(19)
        theta1 = rng.normal(size=lam.shape)
        theta2 = rng.normal(size=chi.shape)
        phase1 = np.exp(1j*theta1)
        phase_lam = np.exp(1j*(theta2[None, :]-theta1))
        phase_chi = np.exp(-1j*theta2)
        transformed = discrete_born_huang_rhs(
            c*phase1[None], lam*phase_lam, chi*phase_chi,
            model, basis, flat_top_on_phi=1.0e-3,
            flat_top_on_lam=1.0e-3,
        )
        self.assertLess(
            np.max(np.abs(transformed.dc-original.dc*phase1[None])), 2.0e-12
        )
        self.assertLess(
            np.max(np.abs(transformed.dlam-original.dlam*phase_lam)), 2.0e-12
        )
        self.assertLess(
            np.max(np.abs(transformed.dchi-original.dchi*phase_chi)), 2.0e-12
        )


if __name__ == "__main__":
    unittest.main()
