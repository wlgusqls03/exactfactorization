"""Run server exporter with ONLY its two committed source files available."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class PortableTransferTests(unittest.TestCase):
    def test_isolated_export(self):
        source=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);package=root/'multi_component_exact_factorization/vsc_polariton'
            package.mkdir(parents=True)
            for file in ('phase7_wave_inventory.py','phase7_transfer_utils.py',
                         'phase7_barrier_comparison.py','phase7_fock_to_q.py','phase7_support.py'):
                shutil.copy2(source/file,package/file)
            run=root/'inputs/free_recovery_campaign_v1/free_L36_dx0.3/full'
            run.mkdir(parents=True)
            (run/'status.json').write_text(json.dumps(dict(status='COMPLETE_NOT_CERTIFIED',
                failures={},time_au=12,input_sha256='fixture')))
            for i,flux in enumerate((0.,1.,-1.,0.)):
                np.savez(run/f'observable_{i:07d}.npz',time_au=i*4.,product=i*.1,flux=flux)
                np.savez(run/f'wave_{i:07d}.npz',time_au=i*4.)
            env=dict(os.environ,PYTHONPATH=str(root),OPENBLAS_NUM_THREADS='1')
            result=subprocess.run([sys.executable,'-m',
                'multi_component_exact_factorization.vsc_polariton.phase7_wave_inventory',
                '--root',str(root/'inputs'),'--out',str(root/'output'),'--pack-events'],
                cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'output/inventory.json').read_text())
            self.assertEqual(report['export']['files'],2)
            self.assertTrue((root/'output/phase7_event_waves.tar.gz').is_file())
            self.assertEqual(report['barrier_F160']['status'],'MISSING_DIRECTORY')
            self.assertEqual(json.loads((root/'output/barrier_fock_comparison.json').read_text())['status'],'MISSING')

    def test_paired_barrier_saved_waves(self):
        source=Path(__file__).resolve().parents[1]
        from ..phase7_transfer_utils import digest
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);package=root/'multi_component_exact_factorization/vsc_polariton'
            package.mkdir(parents=True)
            for file in ('phase7_wave_inventory.py','phase7_transfer_utils.py',
                         'phase7_barrier_comparison.py','phase7_fock_to_q.py','phase7_support.py'):
                shutil.copy2(source/file,package/file)
            inp=root/'inputs'
            for nf in (120,160):
                packet=inp/f'completion_transfer/inputs/barrier_F{nf}.npz'
                packet.parent.mkdir(parents=True,exist_ok=True)
                np.savez(packet,dx=1.,dR=1.,g_chi=.1,omega=1.,mu=np.ones((2,2)))
                run=inp/f'free_recovery_campaign_v1/barrier_F{nf}/full'
                run.mkdir(parents=True)
                (run/'status.json').write_text(json.dumps(dict(status='COMPLETE_NOT_CERTIFIED',
                    failures={},time_au=12,input_sha256=digest(packet))))
                for i,flux in enumerate((0.,1.,-1.,0.)):
                    np.savez(run/f'observable_{i:07d}.npz',time_au=i*4.,product=i*.1,flux=flux)
                    psi=np.zeros((2,2,nf),complex);psi[:,:,0]=.5;psi[:,:,-1]=.001
                    np.savez(run/f'wave_{i:07d}.npz',time_au=i*4.,psi=psi)
            env=dict(os.environ,PYTHONPATH=str(root),OPENBLAS_NUM_THREADS='1')
            result=subprocess.run([sys.executable,'-m',
                'multi_component_exact_factorization.vsc_polariton.phase7_wave_inventory',
                '--root',str(inp),'--out',str(root/'output'),'--pack-events'],
                cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'output/inventory.json').read_text())
            self.assertEqual(report['barrier_F160']['packet_provenance']['status'],'VERIFIED')
            self.assertEqual(len(report['barrier_F160']['events']['first_forward_peak']['sha256']),64)
            paired=json.loads((root/'output/barrier_fock_comparison.json').read_text())
            self.assertEqual(paired['status'],'AVAILABLE')
            self.assertTrue(paired['comparisons'])
            self.assertGreater(paired['comparisons'][-1]['F160']['top_fock_population'],0)


if __name__=='__main__':unittest.main()
