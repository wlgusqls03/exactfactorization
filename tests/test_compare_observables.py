import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from multi_component_exact_factorization.compare_observables import compare, _compare_loaded


class CountingArchive(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = Counter()

    @property
    def files(self):
        return tuple(self)

    def __getitem__(self, key):
        self.reads[key] += 1
        return super().__getitem__(key)


class CompareObservablesTests(unittest.TestCase):
    def payload(self):
        return dict(times_fs=np.array([0., 1., 2.]),
                    electron_density=np.ones((3, 7))/7,
                    proton_density=np.ones((3, 9))/9,
                    heavy_density=np.ones((3, 5))/5,
                    state_populations=np.ones((3, 2))*.5)

    def test_members_read_once_including_legacy_spacing(self):
        ref, test = CountingArchive(self.payload()), CountingArchive(self.payload())
        result = _compare_loaded(ref, test, 1e-9)
        self.assertEqual(result['common_frames'], 3)
        self.assertTrue(all(v == 0 for k, v in result.items() if k != 'common_frames'))
        for archive in (ref, test):
            for key in self.payload():
                self.assertEqual(archive.reads[key], 1)

    def test_compressed_archive_spacing_and_shape_mismatch(self):
        with TemporaryDirectory() as folder:
            ref = self.payload()
            test = self.payload()
            ref['q'] = np.arange(9)*.2
            test['proton_density'] += .1
            test['heavy_density'] = np.ones((3, 6))
            paths = [Path(folder)/name for name in ('a.npz', 'b.npz')]
            for path, payload in zip(paths, (ref, test)):
                np.savez_compressed(path, **payload)
            result = compare(*paths)
            self.assertAlmostEqual(result['max_l1_proton_density'], .18)
            self.assertTrue(np.isnan(result['max_l1_heavy_density']))


if __name__ == '__main__':
    unittest.main()
