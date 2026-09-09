import copy
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

from multi_component_exact_factorization.external_potential import (
    harmonic_potential, separate_fields, effective_scalar, EXTERNAL,
)
from multi_component_exact_factorization_discrete.core import discrete_born_huang_rhs as discrete_mcef_rhs, discrete_tdse_action
from tests.test_discrete_mcef import _problem


class ExternalHarmonicTests(unittest.TestCase):
    def test_discrete_rhs_and_total_hamiltonian_unchanged(self):
        model, basis, c, lam, chi = _problem()
        legacy = copy.copy(model)
        model.R = np.arange(chi.size)*model.dR
        model.heavy_trap_alpha = .03
        model.heavy_trap_center = .7
        V = harmonic_potential(model.R, vars(model))
        # Both models use the identical full BO matrix; only bookkeeping differs.
        old = discrete_mcef_rhs(c, lam, chi, legacy, basis)
        new = discrete_mcef_rhs(c, lam, chi, model, basis)
        for name in ('dc','dlam','dchi'):
            np.testing.assert_allclose(getattr(new,name),getattr(old,name),atol=1e-12)
        for name in ('epsilon_1','epsilon_2'):
            np.testing.assert_allclose(new.fields[name]+V,old.fields[name],atol=1e-12)
        y = c*lam[None]*chi[None,None]
        np.testing.assert_allclose(discrete_tdse_action(y,model,basis),discrete_tdse_action(y,legacy,basis),atol=1e-12)

    def test_conversion_closes_and_is_idempotent(self):
        rng = np.random.default_rng(3)
        rho = rng.uniform(.1,1,(3,7,8)); p0 = rng.uniform(0,1,rho.shape)
        obs = {'R':np.arange(8)*.2, 'options':{'heavy_trap_alpha':.03,'heavy_trap_center':.7},
               'joint_density':rho}
        ef = {}
        for level,shape in ((1,rho.shape),(2,(3,8))):
            parts = [f'tdpes{level}_{key}' for key in ('wbo_0','wbo_excited','gd','geo_q','geo_R')]
            for key in parts: ef[key]=rng.normal(size=shape)
            ef[f'tdpes{level}_total']=sum(ef[k] for k in parts)
        old = copy.deepcopy(ef)
        separate_fields(ef,obs,lambda f:rho[f]*p0[f])
        for level in (1,2):
            np.testing.assert_allclose(ef[f'tdpes{level}_total'],sum(ef[f'tdpes{level}_{k}'] for k in ('wbo_0','wbo_excited','gd','geo_q','geo_R')),atol=1e-14)
            np.testing.assert_allclose(effective_scalar(ef[f'tdpes{level}_total'],ef,obs),old[f'tdpes{level}_total'],atol=1e-14)
            np.testing.assert_array_equal(ef[f'tdpes{level}_gd'],old[f'tdpes{level}_gd'])
        already = copy.deepcopy(ef)
        separate_fields(ef,obs)
        np.testing.assert_array_equal(ef['tdpes1_total'],already['tdpes1_total'])
        self.assertEqual(ef['energy_convention'],EXTERNAL)

    def test_external_comparison_smoke(self):
        from tests.test_tdse_report import TDSEReportTests
        from multi_component_exact_factorization.render_final_visualizations import parse_args,run
        with TemporaryDirectory() as directory:
            TDSEReportTests()._write_archive(directory)
            run(parse_args([directory,'--only','external','--format','gif','--max-frames','2',
                            '--snapshot-count','2','--dpi','30','--animation-dpi','25']))
            output=Path(directory)/'report/final_visualizations'
            self.assertTrue((output/'external_harmonic_comparison_movie.gif').is_file())

    def test_force_uses_effective_scalar_after_conversion(self):
        from tests.test_tdse_report import TDSEReportTests
        from multi_component_exact_factorization import tdse_report
        with TemporaryDirectory() as directory:
            TDSEReportTests()._write_archive(directory)
            archive=Path(directory)/'multi_component_discrete_tdse_gpu.npz'
            obs=tdse_report.calculate_observables(tdse_report.load_observables(archive))
            legacy=tdse_report._load_ef_fields(obs)
            legacy['energy_convention']='harmonic_included'
            obs['options'].update(heavy_trap_alpha=.03, heavy_trap_center=.7)
            changed=copy.deepcopy(legacy)
            separate_fields(changed,obs)
            old=tdse_report._ef_frame(obs,legacy,1)
            new=tdse_report._ef_frame(obs,changed,1)
            for key in ('force_q_full','force_R_first_full','force_R_full'):
                np.testing.assert_allclose(old[key],new[key],atol=1e-12)

    def test_projected_spectral_weights_not_full_grid_density(self):
        obs={'R':np.array([0.,1.]),'options':{'heavy_trap_alpha':1.,'heavy_trap_center':0.},
             'joint_density':np.ones((1,2,2))}
        projected=np.array([[1.,1.],[3.,3.]])
        ground=np.array([[1.,1.],[0.,0.]])
        fields={'tdpes2_wbo_0':np.zeros((1,2)), 'tdpes2_wbo_excited':np.zeros((1,2))}
        separate_fields(fields,obs,lambda f:(ground,projected))
        self.assertAlmostEqual(fields['tdpes2_wbo_0'][0,1],-.25)
        self.assertAlmostEqual(fields['tdpes2_wbo_excited'][0,1],-.75)


if __name__ == '__main__':
    unittest.main()
