"""Rendering cutoffs filter data before scales/plots, never modify archives."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt

from multi_component_exact_factorization_cavity.render_time import (
    select_observables, select_field_files, report_directory, time_mask,
)
from multi_component_exact_factorization_cavity.render import render_case, render_comparison
from multi_component_exact_factorization_cavity.polaritonic_dynamics import render_polaritonic_dynamics


class RenderTimeTests(unittest.TestCase):
    def test_spatial_axis_length_collision_and_all_time_arrays(self):
        data=dict(t_fs=np.array([0.,20.,40.,50.]),R=np.arange(4.),x=np.arange(4.),
                  Q=np.arange(4.),q=np.arange(4.),norm=np.ones(4),step=np.arange(4),
                  pol_local_density=np.ones((4,4,4)),cbo_population=np.ones((4,4)))
        out=select_observables(data,40.)
        for k in ('R','x','Q','q'):np.testing.assert_array_equal(out[k],data[k])
        self.assertEqual(out['pol_local_density'].shape,(3,4,4))
        self.assertEqual(out['cbo_population'].shape,(3,4))
        np.testing.assert_array_equal(out['t_fs'],[0.,20.,40.])
        self.assertEqual(len(data['t_fs']),4)
        self.assertIs(select_observables(data),data)
        np.testing.assert_array_equal(time_mask([40.,40.0002],40.),[True,False])

    def test_default_output_and_invalid_or_empty_interval(self):
        self.assertEqual(report_directory('case',t_max_fs=40),Path('case/report_40fs'))
        self.assertEqual(report_directory('case'),Path('case/report'))
        for invalid in (-1,np.nan,np.inf):
            with self.assertRaises(ValueError):select_observables({},invalid)
        with self.assertRaisesRegex(ValueError,'No observables'):
            select_observables(dict(t_fs=[50.]),40)
        with self.assertRaisesRegex(ValueError,'times unavailable'):select_observables({},40)

    def test_whole_report_excludes_late_frames_and_late_observables(self):
        from test_cavity_render import frame
        f=frame(); captured=[]
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);(case/'fields').mkdir()
            paths=[]
            for i,t in enumerate((0.,20.,40.,50.)):
                path=case/'fields'/f'fields_{i:07d}.npz'
                g=dict(f,t_fs=t)
                # A huge late energy must not contaminate the display scan.
                if t==50:g['epsilon1']=f['epsilon1']+1e5
                np.savez(path,**g);paths.append(path)
            np.savez(case/'observables.npz',t_fs=[0.,20.,40.,50.],step=np.arange(4),
                     R=f['R'],x=f['x'],Q=f['Q'],norm=np.ones(4),
                     photon_number=[0.,.1,.2,1000.],cbo_population=np.ones((4,2))*.4,
                     cbo_omitted=np.full(4,.2),bo_population=np.ones((4,2))*.4,
                     pol_population=np.ones((4,4))*.2)
            source=paths+[case/'observables.npz']
            hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in source]
            def inspect_figure(fig,path,dpi):
                captured.append(path.name)
                for ax in fig.axes:
                    if ax.get_xlabel()=='t (fs)':
                        self.assertLessEqual(ax.get_xlim()[1],40.)
                        for line in ax.lines:
                            x=np.asarray(line.get_xdata())
                            if len(x):self.assertLessEqual(float(np.nanmax(x)),40.)
            with patch('multi_component_exact_factorization_cavity.render._save',side_effect=inspect_figure):
                result=render_case(case,movies=False,t_max_fs=40)
            self.assertEqual(result['physical_frames'],3)
            self.assertEqual(result['rendered_time_range_fs'],[0.,40.])
            self.assertEqual(result['observables_time_range_fs'],[0.,40.])
            self.assertLess(result['scales']['energy1'][1],10.)
            self.assertTrue((case/'report_40fs/render_manifest.json').exists())
            self.assertIn('dynamics_overview',captured)
            self.assertEqual(hashes,[hashlib.sha256(p.read_bytes()).hexdigest() for p in source])
            self.assertEqual(select_field_files(paths,40),paths[:3])
        plt.close('all')

    def test_standalone_polaritonic_movie_uses_filtered_density_and_population(self):
        from test_cavity_polaritonic_dynamics import cached_data
        data,e=cached_data();data['t_fs']=np.array([0.,40.,50.])
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);np.savez(case/'observables.npz',**data)
            np.savez(case/'references.npz',R=data['R'],pol_energy=e)
            with patch('multi_component_exact_factorization_cavity.polaritonic_dynamics._save'):
                manifest=render_polaritonic_dynamics(case,movies=False,t_max_fs=40)
            self.assertEqual(manifest['physical_frames'],2)
            self.assertEqual(manifest['observables_time_range_fs'],[0.,40.])
            self.assertTrue((case/'report_40fs/polaritonic_dynamics_manifest.json').exists())

    def test_coupling_comparison_filters_all_cases_and_preserves_full_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('zero','weak','strong'):
                case=root/'pen'/name;case.mkdir(parents=True)
                np.savez(case/'observables.npz',t_fs=[0.,40.,50.],
                         product=[0.,.1,.9],photon_number=[0.,.2,1.],
                         kinetic_R=[.1,.2,.3],signed_flux=[0.,.1,-.1])
            def inspect(fig,path,dpi):
                self.assertEqual(path.name,'pen_coupling_comparison_40fs')
                for ax in fig.axes:
                    self.assertLessEqual(ax.get_xlim()[1],40.)
                    # Three cases plus the horizontal zero reference.
                    for line in ax.lines[:3]:
                        self.assertLessEqual(float(np.max(line.get_xdata())),40.)
            with patch('multi_component_exact_factorization_cavity.render._save',side_effect=inspect), patch(
                'multi_component_exact_factorization_cavity.polaritonic_surfaces.render_surfaces',
                return_value={'outputs':[]}):
                result=render_comparison(root,t_max_fs=40)
            self.assertTrue(all('_40fs.' in s for s in result['outputs']))


if __name__=='__main__':unittest.main()
