"""Cache-only plots: projected densities, product ordering and omissions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import matplotlib.pyplot as plt

from multi_component_exact_factorization_cavity.polaritonic_dynamics import (
    product_composition, prepare_wavepacket, wavepacket_figure, composition_figure,
    population_panel, render_polaritonic_dynamics,
)
from multi_component_exact_factorization_cavity.render import (
    overview_figure, outer_force_figure, prepare_display, _silhouette,
)
from multi_component_exact_factorization_cavity.workflow import save_tree


def cached_data():
    R=np.linspace(-3,3,30); dR=R[1]-R[0]; t=np.array([0.,1.,2.])
    rho=np.exp(-(R[None,:]-np.array([-.8,0,.8])[:,None])**2)
    rho/=rho.sum(axis=1,keepdims=True)*dR
    local=rho[:,:,None]*np.array([.1,.3,.35,.05])[None,None,:]
    energies=.01*R[:,None]**2+.05*np.arange(4)[None,:]
    data=dict(R=R,t_fs=t,rho_R=rho,pol_local_density=local,
              pol_population=local.sum(axis=1)*dR,pol_outside=np.full(3,.2),
              norm=np.ones(3),bo_population=np.tile([.7,.2],(3,1)),
              bo_outside=np.full(3,.1))
    return data,energies


class PolaritonicDynamicsTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_complex_coefficients_product_order_and_other_weight(self):
        # 2 BO x 3 Fock, state zero contains a third-photon component outside
        # the four requested products. Its .5 weight must not be renormalized.
        v=np.eye(6,dtype=complex)
        v[:,0]=0;v[0,0]=.5;v[3,0]=.5j;v[2,0]=np.sqrt(.5)
        a=dict(pol_vectors=np.tile(v,(3,1,1)),fock=np.zeros((8,3)))
        c=product_composition(a)
        np.testing.assert_allclose(c['weights'][0,:,0],[.25,0,.25,0,.5])
        self.assertEqual(c['coefficients'][0,2,0],.5j)
        np.testing.assert_allclose(c['weights'].sum(axis=1),1)
        a['pol_vectors']*=2
        with self.assertRaisesRegex(ValueError,'normalized'):product_composition(a)

    def test_absolute_projected_density_and_common_all_frame_scales(self):
        data,e=cached_data();p=prepare_wavepacket(data,e,data['R'])
        np.testing.assert_allclose(p['populations'],data['pol_population'])
        np.testing.assert_allclose(p['outside'],.2*data['rho_R'])
        bounds=[]
        for i in (0,2):
            fig=wavepacket_figure(p,data,i);fig.canvas.draw()
            self.assertEqual(len(fig.axes),6)
            self.assertEqual(len(set(ax.get_ylim() for ax in fig.axes[:4])),1)
            bounds.append([ax.get_ylim() for ax in fig.axes])
        self.assertEqual(bounds[0],bounds[1])
        data['pol_population']=data['pol_population']*.9
        with self.assertRaisesRegex(ValueError,'disagree'):prepare_wavepacket(data,e,data['R'])

    def test_population_groups_are_separate_and_account_for_extra_states(self):
        data,_=cached_data();fig=overview_figure(data,current=1.)
        self.assertEqual(fig.axes[4].get_title(),'BO populations')
        self.assertEqual(fig.axes[5].get_title(),'POL populations')
        self.assertFalse(any('Pol' in line.get_label() for line in fig.axes[4].lines))
        self.assertFalse(any('$S_' in line.get_label() for line in fig.axes[5].lines))
        np.testing.assert_allclose(fig.axes[4].get_ylim(),fig.axes[5].get_ylim())
        data['pol_population']=np.column_stack((data['pol_population'],np.full(3,.07)))
        data['pol_outside']=np.full(3,.13)
        fig,ax=plt.subplots();population_panel(ax,data)
        remainder=next(l for l in ax.lines if l.get_label()=='Outside shown states')
        np.testing.assert_allclose(remainder.get_ydata(),.2)

    def test_silhouette_more_visible_without_changing_energy_limits(self):
        fig,ax=plt.subplots();ax.set_ylim(-.5,.5)
        _silhouette(ax,np.arange(5),np.array([0,.2,1,.2,0]))
        self.assertGreaterEqual(ax.collections[0].get_alpha(),.45)
        self.assertEqual(ax.get_ylim(),(-.5,.5))

    def test_outer_force_adds_population_without_mislabeling_mcef(self):
        from test_cavity_render import frame
        f=frame();data,_=cached_data()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'f.npz';np.savez(path,**f);d=prepare_display([path])
            d['population_observables']=data
            fig=outer_force_figure(f,d);fig.canvas.draw()
            self.assertEqual(len(fig.axes),3)
            self.assertEqual(fig.axes[2].get_xlabel(),'t (fs)')
            self.assertTrue(any('Pol' in l.get_label() for l in fig.axes[2].lines))
            panel_label=next(t for t in fig.axes[2].texts if t.get_text()=='C')
            self.assertGreater(panel_label.get_window_extent().x0,0)
            f['system']='mcef';fig=outer_force_figure(f,d)
            self.assertEqual(len(fig.axes),2)

    def test_driver_and_missing_data_fallback(self):
        data,e=cached_data();R=data['R']
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);out=case/'report'
            np.savez(case/'observables.npz',**data)
            np.savez(case/'references.npz',R=R,pol_energy=e)
            a=dict(pol_vectors=np.tile(np.eye(4),(len(R),1,1)),fock=np.zeros((8,2)),
                   pol_energy=e,omega=.1,g_cavity=.01)
            save_tree(case/'analysis_cache.npz',a)
            with mock.patch('multi_component_exact_factorization_cavity.polaritonic_dynamics._save'):
                result=render_polaritonic_dynamics(case,movies=False)
            self.assertFalse(result['notes'])
            self.assertEqual(result['physical_frames'],3)
            record=json.loads((out/'polaritonic_composition/initial_composition.json').read_text())
            np.testing.assert_allclose(np.sum(record['weights_including_other'],axis=0),1)
            self.assertTrue(any('polaritonic_wavepacket/snapshot' in s for s in result['outputs']))
        with tempfile.TemporaryDirectory() as tmp:
            result=render_polaritonic_dynamics(tmp,movies=False)
            self.assertEqual(len(result['notes']),2)
            self.assertEqual(result['physical_frames'],0)

    def test_mismatched_native_grids_are_not_silently_combined(self):
        data,e=cached_data()
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp)
            np.savez(case/'observables.npz',**data)
            np.savez(case/'references.npz',R=data['R']+.1,pol_energy=e)
            result=render_polaritonic_dynamics(case,movies=False)
            self.assertTrue(any('grids mismatch' in s for s in result['notes']))
            self.assertEqual(result['physical_frames'],0)


if __name__=='__main__':unittest.main()
