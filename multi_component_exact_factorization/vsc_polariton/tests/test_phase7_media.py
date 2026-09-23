"""Temporal alignment and event semantics; no interpolated MCEF movie."""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from unittest.mock import patch
import numpy as np
from ..phase7_reaction_media import common_times,events


class MediaTests(unittest.TestCase):
    def test_common_saved_times(self):
        t,ids=common_times([dict(time_au=np.array([0.,1,2,3])),dict(time_au=np.array([0.,2,3]))])
        np.testing.assert_array_equal(t,[0,2,3])
        np.testing.assert_array_equal(ids[0],[0,2,3])

    def test_two_snapshots_cannot_make_movie(self):
        with self.assertRaises(ValueError):common_times([dict(time_au=np.array([0.,1652]))])

    def test_event_sign_and_roundoff(self):
        f=np.array([0,1e-18,0,1.,0,-.5,0])
        e=events(np.arange(7),np.array([0,0,0,.1,.3,.2,.2]),f)
        self.assertEqual(e['first_forward_peak'],3)
        self.assertEqual(e['first_backward_peak'],5)
        self.assertEqual(e['max_product'],4)

    def test_no_recrossing_is_not_fabricated(self):
        e=events(np.arange(4),np.arange(4)/4,np.array([0.,.1,.2,.1]))
        self.assertIsNone(e['first_backward_peak'])

    def test_event_export_preserves_originals(self):
        from ..phase7_wave_inventory import main,RUNS
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'runs';out=Path(temp)/'export';originals={}
            for run in RUNS.values():
                folder=root/run;folder.mkdir(parents=True)
                (folder/'status.json').write_text(json.dumps(dict(status='COMPLETE_NOT_CERTIFIED',failures={},time_au=12.,input_sha256='fixture')))
                for i,f in enumerate((0.,1.,-1.,0.)):
                    np.savez(folder/f'observable_{i:07d}.npz',time_au=i*4.,product=i*.1,flux=f)
                    path=folder/f'wave_{i:07d}.npz';np.savez(path,time_au=i*4.)
                    originals[path]=path.read_bytes()
            with patch('sys.argv',['inventory','--root',str(root),'--out',str(out),'--pack-events']),contextlib.redirect_stdout(io.StringIO()):main()
            info=json.loads((out/'inventory.json').read_text())
            self.assertEqual(info['export']['files'],6)
            with tarfile.open(out/'phase7_event_waves.tar.gz') as tar:self.assertEqual(len(tar.getnames()),7)
            for path,data in originals.items():self.assertEqual(path.read_bytes(),data)

    def test_missing_directory_is_reported(self):
        from ..phase7_wave_inventory import main
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out'
            with patch('sys.argv',['inventory','--root',str(Path(temp)/'missing'),'--out',str(out)]),contextlib.redirect_stdout(io.StringIO()):main()
            self.assertEqual(json.loads((out/'inventory.json').read_text())['free']['status'],'MISSING_DIRECTORY')
