"""Static coupling comparisons retain cached energies and common axes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt

from multi_component_exact_factorization_cavity.polaritonic_surfaces import (
    CASES, PRODUCTS, HA_TO_MEV, load_cases, surface_figure, pair_figure,
    render_surfaces)
from multi_component_exact_factorization_cavity.render import render_comparison, _title


def fixture(root, names=CASES):
    R=np.linspace(-2,2,33)
    bo=np.stack([.01*R**2-.2, .01*R**2-.1+.005*R],axis=1)
    for name,g in zip(CASES,(0.,.001,.01)):
        if name not in names:continue
        folder=Path(root)/'pen'/name;folder.mkdir(parents=True)
        config=dict(campaign='pen',omega=.1,g_cavity=g,R0=0.,
                    n_bo=2,n_fock_analysis=8,dse=True)
        (folder/'config.json').write_text(json.dumps(config))
        bare=np.column_stack([bo[:,j]+(n+.5)*.1 for j,n in PRODUCTS])
        energies=np.sort(bare,axis=1)
        # An explicit common shift as well as a gap: no per-case alignment.
        energies+=g*g
        energies[:,1]-=g;energies[:,2]+=g
        np.savez(folder/'references.npz',R=R,bo_energy=bo,pol_energy=energies,
                 pol_pair_indices=np.tile([1,2],(len(R),1)))


class PolaritonicSurfaceTests(unittest.TestCase):
    def test_full_curves_common_scales_and_actual_couplings(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);cases,notes=load_cases(tmp)
            self.assertFalse(notes)
            for d in cases.values():
                np.testing.assert_allclose(d['bare'][:,0],d['bo_energy'][:,0]+.05)
            original=cases['strong']['pol_energy'].copy()
            fig,info=surface_figure(cases);fig.canvas.draw()
            for row in range(2):
                axes=fig.axes[row*3:(row+1)*3]
                for ax in axes[1:]:
                    self.assertEqual(ax.get_xlim(),axes[0].get_xlim())
                    self.assertEqual(ax.get_ylim(),axes[0].get_ylim())
                    np.testing.assert_array_equal(ax.get_yticks(),axes[0].get_yticks())
            self.assertIn('0.001',fig.axes[1].get_title())
            self.assertIn('0.01',fig.axes[2].get_title())
            self.assertIn('omega_c',fig.axes[0].get_title())
            for col,name in enumerate(CASES):
                for k in range(4):
                    np.testing.assert_array_equal(fig.axes[col].lines[k].get_ydata(),cases[name]['pol_energy'][:,k])
                    np.testing.assert_array_equal(fig.axes[col].lines[k+4].get_ydata(),cases[name]['bare'][:,k])
            np.testing.assert_array_equal(cases['strong']['pol_energy'],original)
            plt.close(fig)

    def test_pair_common_baseline_preserves_splitting_and_common_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp);cases,_=load_cases(tmp)
            fig,info=pair_figure(cases);fig.canvas.draw()
            anchor=cases['strong']['anchor']
            for ax in fig.axes[1:]:self.assertEqual(ax.get_ylim(),fig.axes[0].get_ylim())
            self.assertAlmostEqual(info['pairs']['strong']['splitting_meV'],.02*HA_TO_MEV)
            strong_mean=np.mean([line.get_ydata()[anchor] for line in fig.axes[2].lines[:2]])
            zero_mean=np.mean([line.get_ydata()[anchor] for line in fig.axes[0].lines[:2]])
            self.assertAlmostEqual(strong_mean-zero_mean,.01**2*HA_TO_MEV)
            plt.close(fig)

    def test_missing_case_pair_and_render_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp,names=('weak',))
            folder=Path(tmp)/'pen'/'weak'
            with np.load(folder/'references.npz') as z:
                fields={k:z[k] for k in z.files if k!='pol_pair_indices'}
            np.savez(folder/'references.npz',**fields)
            result=render_surfaces(tmp,dpi=30)
            self.assertEqual(len(result['outputs']),4)
            self.assertEqual(len(result['notes']),2)
            self.assertIn('unavailable',result['pair_zoom']['pairs']['weak'])
            self.assertTrue((Path(tmp)/'polaritonic_surfaces/surface_manifest.json').exists())
            for path in result['outputs']:self.assertGreater(Path(path).stat().st_size,0)

    def test_incompatible_inputs_rejected_and_driver_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            path=Path(tmp)/'pen'/'strong'/'config.json'
            config=json.loads(path.read_text());config['omega']=.2
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'Incompatible'):load_cases(tmp)
            with self.assertWarnsRegex(UserWarning,'static surfaces skipped'):
                result=render_comparison(tmp)
            self.assertFalse(result['outputs'])
            self.assertTrue(any('Incompatible' in n for n in result['notes']))
            config['omega']=float('nan');path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'Invalid frequency'):load_cases(tmp)

    def test_existing_comparison_hooks_static_renderer_and_numeric_g(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            out=Path(tmp)/'pen'/'weak'
            np.savez(out/'observables.npz',t_fs=[0.,1.],product=[.1,.2],
                     photon_number=[0.,.1],kinetic_R=[.01,.02],signed_flux=[0.,.1])
            captured=[]
            def save(fig,*args):
                captured.extend(line.get_label() for ax in fig.axes for line in ax.lines)
                self.assertEqual(len(fig.legends),1)
                self.assertTrue(all(ax.get_legend() is None for ax in fig.axes))
            with patch('multi_component_exact_factorization_cavity.render._save',side_effect=save), \
                    patch('multi_component_exact_factorization_cavity.polaritonic_surfaces.render_surfaces',
                          return_value={'outputs':['static.png'],'notes':[]}) as render:
                result=render_comparison(tmp)
            render.assert_called_once_with(Path(tmp))
            self.assertIn('static.png',result['outputs'])
            self.assertTrue(any('weak' in s and '0.001' in s for s in captured))
            fig=plt.figure();_title(fig,dict(t_fs=0.,g_cavity=.001),'Example')
            self.assertIn('g_{\\mathrm{cavity}}=0.001',fig._suptitle.get_text())
            plt.close(fig)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=render_surfaces(tmp)
            self.assertFalse(result['outputs']);self.assertEqual(len(result['notes']),3)


if __name__=='__main__':unittest.main()
