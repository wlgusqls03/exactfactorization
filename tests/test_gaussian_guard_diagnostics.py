"""Rejected conditional-state candidates must be distinguishable from norm loss."""
import unittest

import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.run import (
    Engine, RunConfig, StateGuardError,
)


class GuardDiagnosticsTests(unittest.TestCase):
    def engine(self):
        return Engine(RunConfig(model=dict(name='pen-rabi', nq=32), ntraj=7))

    def test_tail_pnc_failure_retains_location_without_normalizing(self):
        engine = self.engine()
        state, _ = engine.initial()
        state['C'][2, 0] *= 1.00001
        original = state['C'].copy()
        with self.assertRaises(StateGuardError) as caught:
            engine.guard(state)
        diagnostic = caught.exception.diagnostics
        self.assertGreater(diagnostic['pointwise_pnc_error'], 1e-5)
        self.assertLess(diagnostic['integrated_norm_error'], 1e-13)
        self.assertLess(diagnostic['density_weighted_pnc_error'], 1e-13)
        self.assertEqual(diagnostic['worst_pnc_location']['trajectory'], 2)
        self.assertEqual(diagnostic['worst_pnc_location']['q_index'], 0)
        self.assertLess(diagnostic['worst_pnc_location']['relative_density'], 1e-12)
        np.testing.assert_array_equal(state['C'], original)

    def test_native_norm_failure_has_no_fictitious_pnc_location(self):
        engine = Engine(RunConfig(model=dict(name='pen-rabi', nq=32),
                                  ntraj=7, compression='native'))
        state, _ = engine.initial()
        state['G'] *= 1.001
        with self.assertRaises(StateGuardError) as caught:
            engine.guard(state)
        diagnostic = caught.exception.diagnostics
        self.assertGreater(diagnostic['integrated_norm_error'], 1e-3)
        self.assertIsNone(diagnostic['pointwise_pnc_error'])
        self.assertIsNone(diagnostic['worst_pnc_location'])

    def test_accepted_state_records_separate_quantities(self):
        engine = self.engine()
        state, _ = engine.initial()
        self.assertLess(engine.guard(state), 1e-13)
        self.assertIn('q_boundary_mass', engine.last_guard_diagnostics)
        self.assertLess(engine.last_guard_diagnostics['pointwise_pnc_error'], 1e-13)


if __name__ == '__main__':
    unittest.main()
