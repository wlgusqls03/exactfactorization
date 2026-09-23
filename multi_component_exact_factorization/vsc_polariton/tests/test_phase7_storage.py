import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ..phase7_storage_audit import verified


class ArchiveSafetyTests(unittest.TestCase):
    def test_pilot_packaging_preserves_inputs(self):
        from ..phase7_export_pilot import main
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                base=Path('results/vsc_polariton/phase6_gpu')
                gate=base/'finish_campaign_v3/phase6_completion_validation.json'
                gate.parent.mkdir(parents=True)
                gate.write_text(json.dumps(dict(status='COMPUTED_GATES_PASS_REVIEW_PENDING')))
                runs=['free_recovery_campaign_v1/free_L36_dx0.3/full',
                      'orthogonal_campaign_v1/F120_dt0125/full',
                      'orthogonal_campaign_v1/F160_dt0125/full',
                      'free_recovery_campaign_v1/barrier_F120/full']
                packets=['free_recovery_transfer_v2/inputs/free_L36_dx0.3.npz',
                         'orthogonal_transfer/inputs/resonant_F120.npz',
                         'orthogonal_transfer/inputs/resonant_F160.npz',
                         'completion_transfer/inputs/barrier_F120.npz']
                R=np.arange(2);x=np.arange(3)
                for run,name in zip(runs,packets):
                    folder=base/run;folder.mkdir(parents=True)
                    (folder/'status.json').write_text(json.dumps(dict(status='COMPLETE_NOT_CERTIFIED',input_sha256='d',dt=.125)))
                    for t in (0.,1652.):np.savez(folder/f'wave_{round(t/.125):07d}.npz',time_au=t,R=R,x=x,psi=np.ones((2,3,1)))
                    packet=base/name;packet.parent.mkdir(parents=True,exist_ok=True)
                    np.savez(packet,R=R,x=x);packet.with_suffix('.json').write_text('{}')
                with patch('multi_component_exact_factorization.vsc_polariton.phase7_export_pilot.load_packet',return_value=({'R':R,'x':x},'d')),patch('builtins.print'):
                    main()
                    with self.assertRaises(FileExistsError):main()
                with tarfile.open('results/vsc_polariton/phase7/phase7_pilot_inputs.tar.gz') as tar:
                    m=json.load(tar.extractfile('phase7_pilot_manifest.json'))
                    self.assertEqual(len(m['cases']),4)
                self.assertTrue((base/runs[0]/'wave_0000000.npz').is_file())
            finally:os.chdir(cwd)

    def test_identical_and_changed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dest=root/'extract';dest.mkdir()
            path=root/'archive.tar.gz'
            with tarfile.open(path,'w:gz') as t:
                info=tarfile.TarInfo('data');info.size=3;t.addfile(info,io.BytesIO(b'abc'))
            self.assertFalse(verified(path,dest)[0])
            (dest/'data').write_bytes(b'abc');self.assertTrue(verified(path,dest)[0])
            (dest/'data').write_bytes(b'xyz');self.assertFalse(verified(path,dest)[0])
            self.assertTrue(path.exists())

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'archive.tar.gz'
            with tarfile.open(path,'w:gz') as t:
                info=tarfile.TarInfo('../outside');info.size=0;t.addfile(info,io.BytesIO())
            self.assertFalse(verified(path,root)[0])

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'archive.tar.gz'
            with tarfile.open(path,'w:gz') as t:
                info=tarfile.TarInfo('link');info.type=tarfile.SYMTYPE;info.linkname='/etc/passwd';t.addfile(info)
            self.assertFalse(verified(path,root)[0])
