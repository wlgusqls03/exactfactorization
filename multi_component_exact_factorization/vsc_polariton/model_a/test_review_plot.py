"""Display regression tests: no fixed ±60 range, dense contours, preserved fields."""
from pathlib import Path
import json
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from .model import Config,initial,Propagator
from .factorization import analyze,compact
from .review_plot import scan,map_axis,cuts,LEVELS,preflight
from .event_audit import sha
from . import next_review


class ReviewPlotTests(unittest.TestCase):
    def fixture(self):
        c=Config(nr=64,nq=64,qmin=-16,qmax=16)
        f=compact(analyze(initial(c),Propagator(c)));f['time_au']=1000.;f['time_fs']=24.1888
        return c,f

    def test_shared_limits_and_dense_contours(self):
        _,f=self.fixture();f['b'][:]=120;f['alpha'][:]=95
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'f.npz';np.savez(p,**f);before=p.read_bytes()
            s=scan({'coupled':[p],'free':[p]})
            self.assertGreater(s['limits']['b'][1],120)
            self.assertGreater(s['limits']['alpha'][1],95)
            self.assertGreaterEqual(len(LEVELS),20)
            self.assertEqual(before,p.read_bytes())

    def test_map_clipping_is_reported(self):
        _,f=self.fixture();f['b'][:]=120
        fig,ax=plt.subplots()
        _,mass=map_axis(ax,f,'b',[-60,60]);plt.close(fig)
        self.assertGreater(mass,.99)

    def test_cut_extrema_are_not_hidden(self):
        c,f=self.fixture();f['cut_q0_geo_q'][:]=15.
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'f.npz';np.savez(p,**f);s=scan({'coupled':[p]})
        s['limits']['cut_energy']=[-.2,.5]
        fig,axes=plt.subplots(1,2)
        ext=cuts(axes,f,c,s)
        self.assertGreaterEqual(ext[0][1],15.)
        self.assertTrue(any(line.get_marker()=='^' for line in axes[0].lines))
        plt.close(fig)

    def test_missing_fields_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):preflight(Path(temp),movies=False)

    def test_one_return_archive(self):
        # Orchestration-only fixture: physics and renderer tested separately above.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'campaign';root.mkdir();out=Path(temp)/'review'
            for name in ['coupled','free']:
                folder=root/name/'fields';folder.mkdir(parents=True)
                np.savez(folder/'fields_0.npz',test_fixture=True)
            wave=root/'wave.npz';np.savez(wave,test_fixture=True)
            original=sha(wave)
            def analysis(root,folder,pack):
                self.assertFalse(pack);folder.mkdir()
                (folder/'summary.json').write_text('{}')
                return {'inputs':{'wave.npz':{'sha256':original}}}
            def rendering(root,folder,movies):
                self.assertFalse(movies);folder.mkdir()
                (folder/'manifest.json').write_text('{}')
            with patch.object(next_review,'audit',analysis),patch.object(next_review,'render',rendering),patch('sys.argv',[
                'test','--campaign',str(root),'--out',str(out),'--no-movies']):
                next_review.main()
            with tarfile.open(out/'model_a_next_review.tar.gz') as archive:
                self.assertEqual(set(archive.getnames()),{'campaign/wave.npz','analysis/summary.json','plots/manifest.json'})
            info=json.loads((out/'transfer.json').read_text())
            self.assertEqual(info['sha256'],sha(out/info['archive']))
            self.assertEqual(original,sha(wave))


if __name__=='__main__':unittest.main()
