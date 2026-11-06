"""Small cached-frame renderer check; no TDSE/trajectory propagation here."""
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.channel_render import (
    load_frame, plot_snapshot, prepare, render,
)


class ChannelRenderTests(unittest.TestCase):
    def _fixture(self, root):
        (root/'frames').mkdir()
        (root/'config.json').write_text(json.dumps({'method': 'tsh', 'compression': 'gaussian'}))
        for frame in range(3):
            R = np.linspace(-1, 1, 5)+frame*.1
            population = np.tile(np.array([.4, .3, .1, .07, .06, .04, .03]), (5, 1))
            np.savez(root/'frames'/('frame_%010d.npz' % frame), t_fs=frame*.01,
                     R=R, P=np.arange(5)*.02, eta=np.ones(5)/5,
                     channel_population=population, channel_energy=.1*np.arange(7)[None, :]+R[:, None]**2,
                     active_state=np.arange(5)%2, Q_raw=R*.03, ct_force_correction=np.zeros(5),
                     hops_attempted=frame, hops_accepted=frame, hops_frustrated=0,
                     projection_residual=np.full(5, 1e-6), energy=np.full(5, .1+frame*1e-8),
                     active_surface_energy=np.full(5, .1),
                     photon_number=np.full(5, .02*frame))

    def test_stable_channels_and_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            data = prepare(root)
            np.testing.assert_array_equal(data['selected'], [0, 1, 2, 3])
            np.testing.assert_allclose(data['traces']['active_surface_energy'], .1)
            self.assertGreater(data['traces']['Q_raw_rms'][0],
                               abs(data['traces']['Q_raw'][0]))
            limits = []
            for path in (data['files'][0], data['files'][-1]):
                fig = plot_snapshot(data, load_frame(path))
                limits.append([(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes])
                plt.close(fig)
            self.assertEqual(limits[0], limits[1])
            self.assertIn('Gaussian bridge projection', data['method_label'])

    def test_png_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            output = render(root, dpi=40)
            self.assertGreater((output/'summary.png').stat().st_size, 1000)
            self.assertEqual(len(list(output.glob('snapshot_*.png'))), 3)


if __name__ == '__main__':
    unittest.main()
