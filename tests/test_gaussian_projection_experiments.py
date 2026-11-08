"""Formula tests for the opt-in fixed-R experiment, not production approval."""
import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
import json
import numpy as np

from multi_component_exact_factorization.gaussian_projection_experiments.projection import (
    constrained_least_squares, energy_constrained_rate, EnergyConstraintError)
from multi_component_exact_factorization.gaussian_projection_experiments.initialization import fit_distinct
from multi_component_exact_factorization.gaussian_trajectory.gaussian import (
    build_bridge, pack_parameters, split_and_project, GaussianRankError)
from multi_component_exact_factorization.gaussian_trajectory.models import InternalModel, ModelConfig
from multi_component_exact_factorization.gaussian_projection_experiments.fixed_problem import FixedProblem
from multi_component_exact_factorization.gaussian_projection_experiments.run import ExperimentConfig,run
from multi_component_exact_factorization.gaussian_projection_experiments.integration import adaptive_interval,IntegrationResolutionError


class ProjectionFormulaTests(unittest.TestCase):
    def test_svd_matches_independent_kkt(self):
        rng=np.random.default_rng(31)
        A=rng.normal(size=(23,5));y=rng.normal(size=23);h=rng.normal(size=23)
        v,_,a=constrained_least_squares(A,y,h,power_atol=1e-13)
        e=A.T@h
        kkt=np.block([[A.T@A,e[:,None]],[e[None,:],np.zeros((1,1))]])
        exact=np.linalg.solve(kkt,np.r_[A.T@y,h@y])[:5]
        np.testing.assert_allclose(v,exact,rtol=2e-13,atol=2e-13)
        self.assertLess(abs(a['constrained_power']),1e-13)
        self.assertGreaterEqual(a['amplitude_residual'],a['least_squares_residual']-1e-13)

    def test_infeasible_and_rank_deficient_fail(self):
        with self.assertRaises(EnergyConstraintError):
            constrained_least_squares(np.array([[1.],[0.]]),np.array([0.,1.]),np.array([0.,1.]))
        with self.assertRaises(GaussianRankError):
            constrained_least_squares(np.ones((4,2)),np.ones(4),np.ones(4))

    def test_redundant_constraint_returns_original_least_squares(self):
        A=np.array([[1.,0.],[0.,1.],[0.,0.]])
        v,_,a=constrained_least_squares(A,np.array([2.,3.,4.]),np.zeros(3))
        np.testing.assert_allclose(v,[2,3])
        self.assertEqual(a['constraint_status'],'already_satisfied_within_tolerance')

    def test_current_state_energy_pnc_and_shift_invariance(self):
        q=np.linspace(-6,6,64,endpoint=False);w=np.full(64,q[1]-q[0])
        z=pack_parameters([.4,.6],[-1.,1.],[.8,1.2],coordinates='transformed')
        b=build_bridge(q,w,z,components=2,coordinates='transformed')
        C=np.column_stack([np.cos(.2*q)*np.exp(.15j*q*q),np.sin(.2*q)*np.exp(-.3j*q)])
        G=b.amplitude[:,None]*C
        rng=np.random.default_rng(11);Z=rng.normal(size=(128,128))+1j*rng.normal(size=(128,128))
        H=(Z+Z.conj().T)/128;HG=(H@G.ravel()).reshape(G.shape)
        p=energy_constrained_rate(HG,C,b)
        old=split_and_project(-1j*HG,C,b)
        np.testing.assert_array_equal(p.C_dot,old.C_dot)
        self.assertLess(abs(p.audit['power_actual']),1e-11)
        self.assertLess(abs(p.audit['norm_rate']),1e-12)
        self.assertLess(np.max(abs(2*np.real(np.sum(C.conj()*p.C_dot,axis=1)))),1e-11)
        np.testing.assert_allclose(p.G_dot_actual+1j*HG,
            (p.Lambda_dot-p.amplitude_rate_requested)[:,None]*C,atol=1e-13)
        shifted=energy_constrained_rate(HG+2.3*G,C,b)
        np.testing.assert_allclose(p.parameter_dot,shifted.parameter_dot,atol=2e-12)
        # Independent directional derivative of normalized energy.
        def energy(v):return np.real(np.vdot(v,H@v)/np.vdot(v,v))
        h=1e-6;v=G.ravel();dv=p.G_dot_actual.ravel()
        self.assertLess(abs((energy(v+h*dv)-energy(v-h*dv))/(2*h)),1e-8)

    def test_no_complex_ls_inputs(self):
        with self.assertRaises(ValueError):
            constrained_least_squares(np.ones((3,1),complex),np.zeros(3),np.zeros(3))


class InitializationTests(unittest.TestCase):
    def test_offline_single_gaussian_really_optimizes_amplitude(self):
        q=np.linspace(-8,8,128);w=np.full(128,q[1]-q[0])
        rho=.8*np.exp(-q*q/2)+.2*np.exp(-(q-3)**2/2)
        b,moments=fit_distinct(q,w,rho,1)
        b,fit=fit_distinct(q,w,rho,1,refine_single=True)
        self.assertGreater(fit['nfev'],1)
        self.assertLess(fit['amplitude_l2_error'],moments['amplitude_l2_error'])

    def test_distinct_k_and_errors_reported(self):
        m=InternalModel(ModelConfig(q_operator='fd2',g=0))
        G=m.initial(np.array([2.]))[0];rho=np.sum(abs(G)**2,axis=1)
        for K in (1,2,3,4):
            b,a=fit_distinct(m.q,m.quadrature,rho,K)
            self.assertTrue(a['success']);self.assertEqual(a['rank'],3*K-1)
            self.assertAlmostEqual(m.quadrature@b.rho,1.,places=13)
            self.assertAlmostEqual(a['amplitude_l2_error'],
                np.sqrt(m.quadrature@(b.amplitude-np.sqrt(rho))**2),places=12)
            if K>1:self.assertGreater(min(np.diff(b.centers)),0.)


class DriverTests(unittest.TestCase):
    def config(self,**kw):
        return ExperimentConfig(model=dict(name='pen-nonadiabatic',nq=32,qmin=-10.,qmax=10.,g=.01,q_operator='fd2'),
                                dt_au=.005,end_fs=.003,every_fs=.001,**kw)

    def test_reference_off_is_identical_autonomous_propagation(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.config(mode='energy');p=Path(d)
            a=run(c,p/'with_ref');b=run(replace(c,reference=False),p/'without_ref')
            self.assertEqual(a['status'],'COMPLETE_NOT_CERTIFIED');self.assertEqual(b['status'],a['status'])
            with np.load(p/'with_ref/checkpoint.npz') as x,np.load(p/'without_ref/checkpoint.npz') as y:
                np.testing.assert_array_equal(x['state_C'],y['state_C']);np.testing.assert_array_equal(x['state_z'],y['state_z'])

    def test_resume_and_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.config();p=Path(d)
            run(c,p/'part');long=replace(c,end_fs=.006)
            run(long,p/'part',resume=True);run(long,p/'full')
            with np.load(p/'part/checkpoint.npz') as x,np.load(p/'full/checkpoint.npz') as y:
                np.testing.assert_array_equal(x['state_C'],y['state_C']);np.testing.assert_array_equal(x['state_z'],y['state_z'])
            with self.assertRaises(FileExistsError):run(c,p/'part')
            with self.assertRaises(ValueError):run(replace(long,dt_au=.0025),p/'part',resume=True)

    def test_matched_reference_and_power_budget(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);s=run(self.config(components=2),p/'K2')
            self.assertEqual(s['status'],'COMPLETE_NOT_CERTIFIED')
            self.assertLess(s['max_power_budget_error_Ha'],1e-9)
            with np.load(p/'K2/diagnostics.npz') as a:
                self.assertLess(a['state_error'][0],1e-13)
                self.assertGreater(a['initial_condition_reference_distance'][0],1e-5)
                self.assertLess(s['max_state_error_minus_residual_integral'],1e-9)

    def test_cached_hamiltonians_match_legacy(self):
        for model in ('pen-nonadiabatic','pen-rabi','mcef'):
            for op in ('fd2','fd4','fft'):
                config=dict(name=model,nq=16,qmin=-5.,qmax=-3.,nx=8,q_operator=op)
                p=FixedProblem(config)
                self.assertLess(p.cached_action_error,1e-12)
                H=p.dense_hamiltonian();self.assertLess(np.max(abs(H-H.conj().T)),1e-12)

    def test_energy_infeasibility_reported_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.config(components=12);p=Path(d)
            s=run(c,p/'bad');self.assertEqual(s['status'],'FAILED_INITIALIZATION')
            self.assertFalse((p/'bad/checkpoint.npz').exists())

    def test_adaptive_step_doubling_same_linear_equation(self):
        C=np.array([[1.,1j]],complex)/np.sqrt(2);state=dict(C=C,z=np.array([.2]))
        energy=np.array([30.,-7.])
        def rhs(s):return dict(C=-1j*energy*s['C'],z=np.zeros(1))
        v,a=adaptive_interval(state,.1,rhs,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(v['C'],C*np.exp(-.1j*energy),atol=2e-8,rtol=2e-8)
        np.testing.assert_array_equal(state['C'],C)
        self.assertGreater(a['rejected_trials'],0)

    def test_adaptive_does_not_hide_unresolvable_constraint(self):
        state=dict(C=np.array([[1.+0j]]),z=np.array([0.]))
        def bad(s):raise EnergyConstraintError('test infeasible',{})
        with self.assertRaises(IntegrationResolutionError):
            adaptive_interval(state,.1,bad,max_halvings=3)
        np.testing.assert_array_equal(state['C'],[[1.]])

    def test_hermite_seed_is_explicit_and_not_an_online_fit(self):
        p=FixedProblem(dict(nq=64,q_operator='fd2',g=0))
        rho=np.sum(abs(p.physical_G0)**2,axis=1)
        errors=[]
        for K in (2,3,4):
            b,a=fit_distinct(p.q,p.w,rho,K,optimize=False)
            self.assertEqual(a['nfev'],0);self.assertFalse(a['reference_feedback'])
            errors.append(a['amplitude_l2_error'])
        self.assertTrue(np.all(np.diff(errors)<0))

    def test_render_handles_missing_reference_and_sweep_legend(self):
        from multi_component_exact_factorization.gaussian_projection_experiments.render import render_case,render_sweep
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);s=run(self.config(reference=False),p/'case')
            render_case(p/'case')
            manifest=dict(cases=[dict(path=str(p/'case'),g=.01,K=1,mode='standard',**s)])
            (p/'sweep.json').write_text(json.dumps(manifest))
            render_sweep(p)
            self.assertGreater((p/'case/plots/diagnostic_summary.png').stat().st_size,10000)
            self.assertTrue((p/'K_comparison_g_0.01.png').is_file())


if __name__=='__main__':unittest.main()
