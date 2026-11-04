"""Adapter tests. Optional native tests use GCTMQC_BUILD, never long dynamics."""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from multi_component_exact_factorization.pen_gctmqc_2026.build import unpack
from multi_component_exact_factorization.pen_gctmqc_2026.workflow import (
    parser, input_text, qml_input, run_case, verify_model)
from multi_component_exact_factorization.pen_gctmqc_2026.data import read_frame, frame_observables
from multi_component_exact_factorization.pen_gctmqc_2026.output_check import (
    prepare_output_dirs, validate_native_outputs)
from multi_component_exact_factorization.pen_ctmqc_2026.core import Settings, Engine
from multi_component_exact_factorization.pen_perspectives_2026.resonance import paper_config


class AdapterTests(unittest.TestCase):
    def test_explicit_model_read_and_dipole_sign(self):
        for model in ('nonadiabatic','rabi'):
            for res in ('on','off'):
                c=paper_config(model,res,dse=False)
                for view in ('el','pol'):
                    text,steps,dump=input_text(c,Settings(view,16),.1,.03,100.)
                    self.assertIn("model_potential='read_model'",text)
                    self.assertIn('read_nml=.true.',text)
                    self.assertIn('Mu12=1',text)
                    self.assertIn('nesteps=1',text)
                    self.assertGreater(steps,dump)
        with self.assertRaises(ValueError):
            input_text(replace(c,dse=True),Settings(),1,.1,100)
        with self.assertRaises(ValueError):
            input_text(replace(c,nf=3),Settings(),1,.1,100)

    def test_reject_archive_escape(self):
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            info=tarfile.TarInfo('source/../../escape')
            info.size=1
            archive.addfile(info,io.BytesIO(b'x'))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            unpack(buffer.getvalue(),Path(tmp))

    def test_no_bundled_dependency_overlay(self):
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for name in ('source/src/main.f90','source/Ext_Lib/QDUtilLib_loc/old.f90'):
                info=tarfile.TarInfo(name);info.size=1
                archive.addfile(info,io.BytesIO(b'x'))
        with tempfile.TemporaryDirectory() as tmp:
            unpack(buffer.getvalue(),Path(tmp))
            self.assertTrue((Path(tmp)/'src/main.f90').exists())
            self.assertFalse((Path(tmp)/'Ext_Lib/QDUtilLib_loc/old.f90').exists())

    def test_truncated_frame_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'broken.bin';p.write_bytes(b'x')
            with self.assertRaises(ValueError):
                read_frame(p,paper_config('rabi','on'),'pol')

    def test_native_output_directories_prepared(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepare_output_dirs(tmp)
            prepare_output_dirs(tmp)
            for name in ('adapter_frames', 'native/coeff', 'native/trajectories'):
                self.assertTrue((Path(tmp)/name).is_dir())


@unittest.skipUnless(os.environ.get('GCTMQC_BUILD'),'Set GCTMQC_BUILD to test compiled author code')
class NativeTests(unittest.TestCase):
    def test_heap_perturbation_does_not_change_dynamics(self):
        build=Path(os.environ['GCTMQC_BUILD']).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            for model in ('nonadiabatic','rabi'):
                for resonance in ('on','off'):
                    c=paper_config(model,resonance,dse=False,dt=.05)
                    for view in ('el','pol'):
                        last=[]
                        for perturb in ('0','165'):
                            out=Path(tmp)/('_'.join((model,resonance,view,perturb)))
                            args=parser().parse_args(['--build',str(build),'--out',str(out),
                                '--end-fs','.2','--every-fs','.1','--no-render'])
                            with patch.dict(os.environ,{'MALLOC_PERTURB_':perturb}):
                                run_case((c,Settings(view,16,2026),out,args))
                            last.append(read_frame(sorted((out/'adapter_frames').glob('*.bin'))[-1],c,view))
                        for key in ('X','P','coeff','k_li'):
                            np.testing.assert_allclose(last[0][key],last[1][key],atol=1e-12,rtol=1e-12)

    def test_models_all_cases_and_initial_preparation(self):
        build=Path(os.environ['GCTMQC_BUILD']).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in ('nonadiabatic','rabi'):
                for res in ('on','off'):
                    for view in ('el','pol'):
                        for coupling in (0.,.001,.01):
                            with self.subTest(model=model,res=res,view=view,g=coupling):
                                c=paper_config(model,res,g=coupling,dse=False,dt=.05)
                                out=root/f'{model}_{res}_{view}_{coupling}'
                                args=parser().parse_args(['--build',str(build),'--out',str(out),
                                    '--end-fs','.01','--every-fs','.005','--no-render'])
                                s=Settings(view,16,2026)
                                _,report=run_case((c,s,out,args))
                                self.assertLess(report['max_surface_error'],1e-12)
                                self.assertLess(report['max_mf_force_error'],1e-10)
                                self.assertLess(report['max_normalized_norm_error'],1e-12)
                                self.assertLess(report['max_open_boundary_k_bound_excess'],1e-12)
                                output = json.loads((out/'output_validation.json').read_text())
                                self.assertTrue(output['passed'])
                                self.assertEqual(output['runtime_boundary_audit']['periodic_flags'],['F']*(2 if view=='el' else 1))
                                self.assertFalse((out/'fort.128').exists())
                                self.assertNotIn('error opening', (out/'run.log').read_text())
                                z=read_frame(sorted((out/'adapter_frames').glob('*.bin'))[0],c,view)
                                ini=Engine(c,s).initial()
                                for key in ('X','P','coeff'):
                                    np.testing.assert_allclose(z[key],ini[key],atol=1e-12,rtol=1e-12)
                                r=frame_observables(z,c,view)
                                self.assertNotIn('epsilon1',r)
                                self.assertNotIn('epsilon2',r)
                                if view=='pol':
                                    self.assertLess(abs(r['photon_number']),1e-25)
                                else:
                                    self.assertNotIn('pol_population',r)
                                with self.assertRaises(FileExistsError):
                                    run_case((c,s,out,args))
                                # Reproduce the historical missing subdirectory
                                # symptom without changing any native dynamics.
                                (out/'fort.128').write_text('unexpected implicit unit\n')
                                with self.assertRaisesRegex(ValueError, 'implicit-unit'):
                                    validate_native_outputs(out)


if __name__=='__main__':
    unittest.main()
