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
            for file in ('phase7_wave_inventory.py','phase7_transfer_utils.py'):
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


if __name__=='__main__':unittest.main()
