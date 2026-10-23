"""Lightweight layout/cache tests; no production simulations or video encoding."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from multi_component_exact_factorization_cavity.render import (
    prepare_display, bridge_figure, source_figure, consistency_figure, slice_figure,
    component_figure,
    bridge_wavefunction_figure,
    population_figure,
    outer_force_figure,
    static_context_figure,
    cbo_comparison_figure,
)
from multi_component_exact_factorization_cavity.schematics import (
    correlation_figure, resonance_figure,
)


def frame():
    R=np.linspace(-4,4,20); q=np.linspace(-3,3,24)
    joint=np.exp(-(R[:,None]+1)**2-q[None,:]**2)
    rho=np.sum(joint,axis=1)*(q[1]-q[0])
    e1=.01*(R[:,None]**2+q[None,:]**2)
    a=.1*np.broadcast_to(q[None,:],joint.shape)
    b=.05*np.broadcast_to(R[:,None],joint.shape)
    alpha=.05*R; s=.01*q[None,:]*joint
    return dict(R=R,q=q,Q=q,x=np.linspace(-5,5,12),rho_qR=joint,rho_R=rho,
                lambda_density=joint/rho[:,None],epsilon1=e1,epsilon2=.01*R**2,
                a=a,b=b,alpha=alpha,density_dt=s,S_q=s,S_adv=s*0,S_rel=s*0,
                alpha_reconstructed=alpha,epsilon2_reconstructed=.01*R**2,
                continuity_residual=s*0,t_fs=0.,g_cavity=.02)


class CavityRenderTests(unittest.TestCase):
    def test_missing_reference_and_gigd_are_graceful(self):
        f=frame()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fields_0000000.npz'; np.savez(p,**f)
            d=prepare_display([p])
            for function in (bridge_figure,source_figure,consistency_figure,slice_figure,bridge_wavefunction_figure):
                fig=function(f,d,0,None)
                fig.canvas.draw()
                self.assertGreaterEqual(len(fig.axes),4)
                plt.close(fig)
            fig=component_figure(f,d); fig.canvas.draw(); plt.close(fig)
            fig=outer_force_figure(f,d); fig.canvas.draw(); plt.close(fig)

    def test_source_scale_is_common_and_zero_centered(self):
        f=frame()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'f.npz'; np.savez(p,**f)
            d=prepare_display([p]); fig=source_figure(f,d,0)
            source_maps=[ax.collections[0] for ax in fig.axes[:4]]
            self.assertEqual(len(set(m.get_clim() for m in source_maps)),1)
            lo,hi=source_maps[0].get_clim(); self.assertAlmostEqual(-lo,hi)
            plt.close(fig)

    def test_schematics_are_labeled_not_numerical_data(self):
        for function in (correlation_figure,resonance_figure):
            fig=function(); fig.canvas.draw()
            text=' '.join(t.get_text() for t in fig.texts)
            self.assertTrue('schematic' in text.lower() or 'explanatory' in text.lower())
            plt.close(fig)

    def test_reference_curves_and_population_basis_labels(self):
        f=frame(); ref=dict(R=f['R'],q=f['q'],cbo=np.stack([f['epsilon1']+.1*j for j in range(4)],axis=-1),
                           pol_energy=np.stack([f['epsilon2']+.1*j for j in range(4)],axis=-1))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'f.npz'; np.savez(p,**f); d=prepare_display([p],ref)
            fig=bridge_figure(f,d,0,ref); fig.canvas.draw()
            cbo=[line for ax in fig.axes for line in ax.lines if line.get_label()=='CBO 0']
            self.assertEqual(len(cbo),1)
            self.assertEqual(len(cbo[0].get_xdata()),len(f['q']))
            plt.close(fig)
        fig=population_figure(dict(t_fs=np.array([0.,1.]),norm=np.ones(2),
            bo_population=np.array([[1.,0.],[.9,.1]]),bo_outside=np.zeros(2)))
        fig.canvas.draw(); self.assertEqual(len(fig.axes),6); plt.close(fig)

    def test_empty_region_reference_poles_do_not_flatten_energy_axes(self):
        f=frame(); R=np.linspace(-12,12,len(f['R']));q=np.linspace(-10,10,len(f['q']))
        joint=np.exp(-((R[:,None]+4)/.4)**2-(q[None,:]/.6)**2)
        rho=np.sum(joint,axis=1)*(q[1]-q[0])
        f.update(R=R,q=q,Q=q,rho_qR=joint,rho_R=rho,
                 lambda_density=np.divide(joint,rho[:,None],out=np.zeros_like(joint),where=rho[:,None]>0))
        cbo=np.stack([-.2+.001*(R[:,None]+4)**2+.005*q[None,:]**2+.1*j for j in range(4)],axis=-1)
        cbo[(R[:,None]<-8)|(abs(q[None,:])>5)]=1000.
        pol=np.stack([-.2+.001*(R+4)**2+.1*j for j in range(4)],axis=-1)
        pol[R<-8]=1000.
        ref=dict(R=R,q=q,cbo=cbo,pol_energy=pol,bo_energy=pol)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'f.npz';np.savez(p,**f);d=prepare_display([p],ref)
            self.assertLess(d['scales']['energy1'][1],2.)
            self.assertLess(d['scales']['energy2'][1],2.)
            fig=bridge_figure(f,d,0,ref)
            cbo_line=next(line for ax in fig.axes for line in ax.lines if line.get_label()=='CBO 0')
            self.assertGreater(np.nanmax(cbo_line.get_ydata()),999.)
            plt.close(fig)
        fig=static_context_figure(ref,f)
        self.assertLess(fig.axes[0].get_ylim()[1],2.)
        self.assertGreater(np.max(fig.axes[0].lines[0].get_ydata()),999.)
        plt.close(fig)

    def test_same_geometry_cbo_map_and_explicit_bare_fallback(self):
        f=frame();f['epsilon1_GI']=f['epsilon1']+.2
        f['cbo_character']=np.stack([np.full_like(f['epsilon1'],.75),np.full_like(f['epsilon1'],.2)],axis=-1)
        f['cbo_outside']=np.full_like(f['epsilon1'],.05)
        ref=dict(R=f['R'],q=f['q'],cbo=np.stack([f['epsilon1']+.1*j for j in range(2)],axis=-1))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'f.npz';np.savez(p,**f);d=prepare_display([p],ref)
            fig=cbo_comparison_figure(f,d,0,ref);fig.canvas.draw()
            self.assertEqual(fig.axes[0].collections[0].get_clim(),fig.axes[1].collections[0].get_clim())
            self.assertEqual(fig.axes[3].collections[0].get_clim(),(0,1))
            self.assertIn('BO-projected',fig.axes[3].get_title());plt.close(fig)
            f['BO_character']=f.pop('cbo_character')
            fig=cbo_comparison_figure(f,d,0,None);fig.canvas.draw()
            self.assertIn('Bare BO S0',fig.axes[3].get_title());plt.close(fig)

    def test_dynamic_lines_mask_tails_but_static_curves_stay_full(self):
        f=frame();q=f['q'];joint=f['rho_qR']*np.exp(-3*q[None,:]**2)
        rho=np.sum(joint,axis=1)*(q[1]-q[0])
        f.update(rho_qR=joint,rho_R=rho,lambda_density=joint/rho[:,None],omega=.01)
        ref=dict(R=f['R'],q=q,cbo=np.stack([f['epsilon1']+.1*j for j in range(2)],axis=-1),
                 pol_energy=np.stack([f['epsilon2']+.1*j for j in range(2)],axis=-1))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'f.npz';np.savez(p,**f);d=prepare_display([p],ref)
            outer=rho>np.max(rho)*1e-5;i=d['i'];qsupport=joint[i]>np.max(joint[i])*1e-5
            self.assertTrue(np.any(~outer));self.assertTrue(np.any(~qsupport))
            fig=bridge_figure(f,d,0,ref)
            outer_line=next(line for line in fig.axes[1].lines if line.get_label()=='total')
            slice_line=next(line for line in fig.axes[3].lines if line.get_label()=='total')
            momentum=next(line for line in fig.axes[5].lines if line.get_label()==r'$b$')
            alpha=next(line for line in fig.axes[5].lines if line.get_label()==r'$\alpha$')
            reference=next(line for line in fig.axes[3].lines if line.get_label()=='CBO 0')
            self.assertTrue(np.all(np.isnan(outer_line.get_ydata()[~outer])))
            self.assertTrue(np.all(np.isnan(slice_line.get_ydata()[~qsupport])))
            self.assertTrue(np.all(np.isnan(momentum.get_ydata()[~qsupport])))
            self.assertTrue(np.all(np.isfinite(alpha.get_ydata())))
            self.assertTrue(np.all(np.isfinite(reference.get_ydata())))
            self.assertFalse(np.any(np.ma.getmaskarray(fig.axes[0].collections[0].get_array())))
            self.assertIn(r'\omega_c',fig._suptitle.get_text());plt.close(fig)
            fig=source_figure(f,d,0,ref)
            source=next(line for line in fig.axes[4].lines if line.get_label()==r'$S_q$')
            self.assertTrue(np.all(np.isnan(source.get_ydata()[~qsupport])));plt.close(fig)


if __name__=='__main__':
    unittest.main()
