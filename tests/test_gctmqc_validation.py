"""Fast tests for output/convergence reporting, never long propagation."""
import json
from pathlib import Path
import tempfile
import unittest
from multi_component_exact_factorization.pen_gctmqc_2026.audit import audit, norm_intervals
from multi_component_exact_factorization.pen_gctmqc_2026.health import assess, check_campaign
from multi_component_exact_factorization.pen_gctmqc_2026.output_check import validate_boundary_log


def rows(times, raw=1e-8):
    return [dict(t_fs=t, raw_norm_error=0. if i==0 else raw, energy=.2,
        photon_number=t*.01, R_mean=2+t*.1, product_R_ge_4=0.,
        bo_population=[t*.01,1-t*.01],basis_population=[0.,.5,.5,0.])
        for i,t in enumerate(times)]


def case(root, dt, times):
    folder = Path(root)/'rabi/on/dse_off/strong/pol'
    folder.mkdir(parents=True)
    values = {'config.json':dict(dt=dt,omega=.17),
        'author_settings.json':dict(perspective='pol',initial='paper',ntraj=128,seed=2026,
                                   m_parameter=100.,algorithm='ctmqc',initial_sample_sha256='sample',adapter_safety_version=3),
        'observables.json':rows(times), 'validation.json':{},
        'status.json':dict(status='COMPLETE_NOT_CERTIFIED'),
        'output_validation.json':dict(passed=True)}
    for name,value in values.items():
        (folder/name).write_text(json.dumps(value))
    return folder


class ValidationTests(unittest.TestCase):
    def test_runtime_boundary_and_parameter_audit(self):
        log=' ADAPTER_PERIODIC_FLAGS F F\n ADAPTER_M_PARAMETER 10.0 10.0\n ADAPTER_DOUBLE_INTERCEPT F\n'
        self.assertEqual(validate_boundary_log(log,2,10)['periodic_flags'],['F','F'])
        for wrong in (log.replace('FLAGS F F','FLAGS T F'),log.replace('10.0','100.0'),log.replace('ADAPTER_PERIODIC_FLAGS','UNKNOWN')):
            with self.assertRaises(ValueError):
                validate_boundary_log(wrong,2,10)
    def test_large_raw_error_not_hidden_by_completion(self):
        result = assess(rows([0,1,2],raw=.1))
        self.assertEqual(result['verdict'],'LARGE_STEP_ERROR_DO_NOT_USE_AS_CONVERGED')
        self.assertFalse(result['certified'])
        self.assertEqual(result['raw_norm_review']['first_interval'],[0,1])

    def test_small_error_is_not_a_certificate(self):
        result = assess(rows([0,1,2]))
        self.assertEqual(result['verdict'],'NO_LARGE_ERRORS_DETECTED_NOT_CERTIFIED')
        self.assertFalse(result['certified'])
        self.assertEqual(norm_intervals(rows([0,1]))['count'],0)

    def test_nonfinite_monitor_rejected(self):
        with self.assertRaises(ValueError):
            assess(rows([0,1],float('nan')))

    def test_exact_timestamps_separate_from_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            case(root/'coarse',.05,[0,1,2])
            case(root/'fine',.025,[0,.5,1,1.5,2])
            report=audit(root/'coarse',root/'fine',root/'audit.json',require_through_fs=2)
            self.assertEqual(report['cases'][0]['exact_shared_time_count'],3)
            self.assertEqual(report['cases'][0]['exact_shared_time_max_differences']['photon_number'],0.)
            with self.assertRaisesRegex(ValueError,'must reach'):
                audit(root/'coarse',root/'fine',root/'audit_bad.json',require_through_fs=40)

    def test_different_intercept_parameter_cannot_be_dt_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            case(root/'coarse',.05,[0,1])
            fine=case(root/'fine',.025,[0,1])
            settings=json.loads((fine/'author_settings.json').read_text())
            settings['m_parameter']=10
            (fine/'author_settings.json').write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError,'m_parameter'):
                audit(root/'coarse',root/'fine',root/'audit.json')

    def test_different_executable_cannot_be_dt_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            case(root/'coarse',.05,[0,1])
            fine=case(root/'fine',.025,[0,1])
            settings=json.loads((fine/'author_settings.json').read_text())
            settings['source_manifest']={'binary_sha256':'different'}
            (fine/'author_settings.json').write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError,'executables'):
                audit(root/'coarse',root/'fine',root/'audit.json')

    def test_incomplete_or_failed_output_cannot_pass_health_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=case(root/'run',.05,[0,1])
            report=check_campaign(root/'run',root/'health.json',40)
            self.assertTrue(report['numerical_review_required'])
            self.assertEqual(report['cases'][0]['verdict'],'INCOMPLETE_TIME_COVERAGE')
            (folder/'output_validation.json').write_text(json.dumps(dict(passed=False)))
            report=check_campaign(root/'run',root/'health.json')
            self.assertEqual(report['cases'][0]['verdict'],'OUTPUT_FAILED')

    def test_legacy_initialization_is_not_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=case(root/'run',.05,[0,40])
            meta=json.loads((folder/'author_settings.json').read_text())
            meta['adapter_safety_version']=2
            (folder/'author_settings.json').write_text(json.dumps(meta))
            report=check_campaign(root/'run',root/'health.json',40)
            self.assertEqual(report['cases'][0]['verdict'],
                             'LEGACY_INITIALIZATION_UNVERIFIED_REBUILD_REQUIRED')


if __name__ == '__main__':
    unittest.main()
