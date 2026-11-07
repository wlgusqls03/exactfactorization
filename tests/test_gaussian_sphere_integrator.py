"""Formula/integration tests only; a passing PNC test is not molecular accuracy."""
import unittest
import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.sphere_integrator import (
    SphereStepError, sphere_rk4_step,
)


class SphereIntegratorTests(unittest.TestCase):
    def initial(self):
        return dict(C=np.array([[.6, .8j]], complex), z=np.array([.3]))

    def phase_error(self, dt):
        initial = self.initial()
        state = {k: v.copy() for k, v in initial.items()}
        # z(t)=z0+t; phase includes its SAME-stage integral.
        def rhs(s):
            return dict(C=-1j*(1.1+s['z'][0])*s['C'], z=np.ones(1))
        for _ in range(round(1/dt)):
            state, audit = sphere_rk4_step(state, dt, rhs)
        expected = initial['C']*np.exp(-1j*(1.1+.3+.5))
        np.testing.assert_allclose(state['z'], [1.3], atol=3e-15)
        self.assertLess(audit['output_pnc_error'], 8e-16)
        return np.linalg.norm(state['C']-expected)

    def test_fourth_order_phase_and_same_stage_coupled_parameter(self):
        coarse, middle, fine = [self.phase_error(h) for h in (.1, .05, .025)]
        self.assertGreater(coarse/middle, 12)
        self.assertGreater(middle/fine, 12)
        self.assertLess(fine, 2e-7)

    def test_real_radial_requested_rate_cancels_in_P3_without_losing_phase(self):
        state = self.initial()
        def rhs(s):
            # Deliberately huge requested amplitude rate: correct P3 removes
            # only its REAL radial part; its phase must remain.
            requested = (1000.-.8j)*s['C']
            r = np.real(np.sum(s['C'].conj()*requested, axis=-1))
            return dict(C=requested-s['C']*r[..., None], z=np.zeros(1))
        result, audit = sphere_rk4_step(state, .01, rhs)
        np.testing.assert_allclose(result['C'], state['C']*np.exp(-.008j), atol=1e-12)
        self.assertLess(audit['max_raw_tangent_rate_error'], 2e-12)

    def test_complex_mixing_convergence(self):
        H = np.array([[.2, .6+.1j], [.6-.1j, -.3]])
        E,U = np.linalg.eigh(H)
        C0 = self.initial()['C']
        exact = ((U@np.diag(np.exp(-1j*E))@U.conj().T)@C0[0])[None]
        errors=[]
        for dt in (.1, .05, .025):
            state = dict(C=C0.copy())
            for _ in range(round(1/dt)):
                state,_ = sphere_rk4_step(state,dt,lambda s:dict(C=-1j*np.einsum('kl,ql->qk',H,s['C'])))
            errors.append(np.linalg.norm(state['C']-exact))
        self.assertGreater(errors[0]/errors[1],12)
        self.assertGreater(errors[1]/errors[2],12)

    def test_large_auxiliary_step_is_rejected_before_mapping_it_away(self):
        state = self.initial()
        with self.assertRaises(SphereStepError) as ctx:
            sphere_rk4_step(state,1.,lambda s:dict(C=-10j*s['C'],z=np.zeros(1)))
        self.assertGreater(ctx.exception.audit['max_auxiliary_stage_norm_error'],.05)

    def test_bad_initial_PNC_is_not_repaired(self):
        state = self.initial();state['C'] *= 1.001
        with self.assertRaises(SphereStepError) as ctx:
            sphere_rk4_step(state,.01,lambda s:dict(C=-1j*s['C'],z=np.zeros(1)))
        self.assertEqual(ctx.exception.audit['rhs_evaluations'],0)

    def test_non_tangent_RHS_is_refused_not_projected(self):
        with self.assertRaises(SphereStepError) as ctx:
            sphere_rk4_step(self.initial(),.01,lambda s:dict(C=(.001-1j)*s['C'],z=np.zeros(1)))
        self.assertGreater(ctx.exception.audit['max_raw_tangent_rate_error'],.001)

    def test_input_and_returned_RHS_are_not_mutated(self):
        state=self.initial();before={k:v.copy() for k,v in state.items()};rates=[]
        def rhs(s):
            result=dict(C=-1j*s['C'],z=np.ones(1));rates.append((result,{k:v.copy() for k,v in result.items()}))
            return result,{'arbitrary_rhs_audit':True}
        _,audit=sphere_rk4_step(state,.02,rhs)
        for key in state:np.testing.assert_array_equal(state[key],before[key])
        for actual,expected in rates:
            for key in actual:np.testing.assert_array_equal(actual[key],expected[key])
        self.assertEqual(audit['rhs_evaluations'],4)
        self.assertGreater(audit['max_auxiliary_stage_norm_error'],0)

    def test_invalid_shapes_nonfinite_and_configuration(self):
        for dt in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):sphere_rk4_step(self.initial(),dt,lambda s:s)
        with self.assertRaises(ValueError):sphere_rk4_step(self.initial(),.01,lambda s:dict(C=s['C']))
        with self.assertRaises(ValueError):sphere_rk4_step(self.initial(),.01,lambda s:s,max_auxiliary_defect=1.)
        bad=self.initial();bad['C'][0,0]=np.nan
        with self.assertRaises(ValueError):sphere_rk4_step(bad,.01,lambda s:s)


if __name__=='__main__':
    unittest.main()
