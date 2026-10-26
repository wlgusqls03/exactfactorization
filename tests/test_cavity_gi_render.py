"""GI-only plotting must not shift energies silently or change physical data."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from multi_component_exact_factorization_cavity.render import prepare_display, outer_force_figure
from multi_component_exact_factorization_cavity.gi_render import (
    prepare_gi_display, epsilon1_gi_figure, outer_gi_figure, gi_reference_zoom_figure,
    HA_TO_MEV)


def frame():
    R=np.linspace(-3,3,17);q=np.linspace(-3,3,19)
    joint=np.exp(-R[:,None]**2-q[None,:]**2);rho=joint.sum(axis=1)*(q[1]-q[0])
    e=.01*(R[:,None]**2+q[None,:]**2)-.2
    z=np.zeros_like(e); g=np.full_like(e,.001)
    f=dict(R=R,q=q,rho_qR=joint,rho_R=rho,lambda_density=joint/rho[:,None],
        epsilon1_cond=e,epsilon1_qgeo=g,epsilon1_Rgeo=2*g,epsilon1_GI=e+3*g,
        epsilon1_GD=z+100,epsilon1=e+3*g+100,epsilon2_cond=e[:,9],
        epsilon2_geo=g[:,9],epsilon2_GI=e[:,9]+g[:,9],epsilon2_GD=np.full(17,100.),
        epsilon2=e[:,9]+g[:,9]+100,a=z,b=z,alpha=np.zeros(17),
        density_dt=z,S_q=z,S_adv=z,S_rel=z,t_fs=0.)
    ref=dict(R=R,q=q,cbo=np.stack([e+k*.002 for k in range(4)],axis=-1),
             pol_energy=np.stack([e[:,9]+k*.002 for k in range(4)],axis=-1))
    return f,ref


class GIRenderTests(unittest.TestCase):
    def test_fixed_gi_scales_ignore_large_gd_and_preserve_sums(self):
        f,ref=frame()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'f.npz';np.savez(p,**f)
            d=prepare_display([p],ref);d['gi_scales']=prepare_gi_display([p],d,ref)
            self.assertLess(d['gi_scales']['energy1'][1],1.)
            self.assertLess(d['gi_scales']['energy2'][1],1.)
            later={**f,'t_fs':1.,'epsilon1_GD':f['epsilon1_GD']*1000,
                   'epsilon2_GD':f['epsilon2_GD']*1000}
            p2=Path(tmp)/'later.npz';np.savez(p2,**later)
            self.assertEqual(prepare_gi_display([p,p2],d,ref),d['gi_scales'])
            original=f['epsilon1'].copy()
            fig=epsilon1_gi_figure(f,d,0,ref);fig.canvas.draw()
            self.assertEqual(fig.axes[0].get_ylim(),fig.axes[1].get_ylim())
            self.assertEqual(fig.axes[2].get_ylim(),fig.axes[3].get_ylim())
            total=next(l for l in fig.axes[2].lines if l.get_label()=='Geometric sum')
            np.testing.assert_allclose(total.get_ydata(),.003)
            plt.close(fig)
            for fun in (outer_gi_figure,gi_reference_zoom_figure):
                fig=fun(f,d,0,ref);fig.canvas.draw();plt.close(fig)
            np.testing.assert_array_equal(f['epsilon1'],original)
            fig=outer_force_figure(f,d,ref)
            labels=[l.get_label() for l in fig.axes[0].lines]
            self.assertIn('GI',labels);self.assertIn('GD',labels)
            self.assertEqual(sum('Pol' in s for s in labels),4);plt.close(fig)

    def test_common_baseline_gap_and_unavailable(self):
        f,ref=frame()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'f.npz';np.savez(p,**f)
            d=prepare_display([p],ref);d['gi_scales']=prepare_gi_display([p],d,ref)
            fig=gi_reference_zoom_figure(f,d,0,ref)
            # order: GI, reference 0..3; same reference subtraction, not individual alignment
            np.testing.assert_allclose(fig.axes[0].lines[2].get_ydata(),.002*HA_TO_MEV)
            plt.close(fig)
            for key in list(f):
                if any(k in key for k in ('_GI','_cond','geo')):del f[key]
            np.savez(p,**f);d['gi_scales']=prepare_gi_display([p],d,None)
            for fun in (epsilon1_gi_figure,outer_gi_figure,gi_reference_zoom_figure):
                fig=fun(f,d,0,None);fig.canvas.draw();plt.close(fig)


if __name__=='__main__':unittest.main()
