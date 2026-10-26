"""Lightweight layout/cache tests; no production simulations or video encoding."""
import tempfile
import unittest
from unittest import mock
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
    CONTOUR_LEVELS, reference_focus_figure, _prepare_reference_focus, render_case,
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
    def test_slice_time_traces_and_fixed_scales(self):
        f=frame()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'f.npz'; np.savez(p,**f)
            d=prepare_display([p])
            d['slice_observables']=dict(t_fs=np.array([0.,1.,2.]),
                photon_number=np.array([0.,.3,.2]),
                cbo_population=np.array([[.8,.1],[.6,.3],[.5,.4]]),
                cbo_omitted=np.full(3,.1))
            fig=slice_figure(f,d,0); fig.canvas.draw()
            self.assertEqual(len(fig.axes),6)
            self.assertEqual(fig.axes[2].get_xlabel(),'t (fs)')
            self.assertIn('CBO',fig.axes[5].get_title())
            self.assertTrue(any(line.get_label()=='Outside basis' for line in fig.axes[5].lines))
            limits=fig.axes[2].get_ylim(); plt.close(fig)
            f['t_fs']=1.
            fig=slice_figure(f,d,0)
            self.assertEqual(fig.axes[2].get_ylim(),limits)
            np.testing.assert_array_equal(fig.axes[5].lines[0].get_ydata(),[.8,.6,.5])
            plt.close(fig)

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
            self.assertTrue(any(line.get_label()==r'$|\mathrm{Pol}_0\rangle$'
                                for ax in fig.axes for line in ax.lines))
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

    def test_mcef_orientation_dense_contours_and_fixed_union_viewport(self):
        f=frame();f['system']='mcef';f['scalar_kind']='effective TDPES with external harmonic'
        self.assertGreaterEqual(len(CONTOUR_LEVELS),22);self.assertLessEqual(len(CONTOUR_LEVELS),28)
        self.assertAlmostEqual(CONTOUR_LEVELS[0],1e-5);self.assertAlmostEqual(CONTOUR_LEVELS[-1],.95)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'f.npz';np.savez(p,**f);d=prepare_display([p])
            fig=bridge_figure(f,d,0)
            self.assertIn('$q$',fig.axes[0].get_xlabel());self.assertIn('$R$',fig.axes[0].get_ylabel())
            np.testing.assert_allclose(fig.axes[0].get_xlim(),d['windows'][0][2:])
            np.testing.assert_allclose(fig.axes[0].get_ylim(),d['windows'][0][:2])
            np.testing.assert_allclose(fig.axes[0].collections[0].get_array(),f['epsilon1'].ravel())
            np.testing.assert_allclose(fig.axes[1].get_xlim(),d['line_xlim_R'])
            np.testing.assert_allclose(fig.axes[3].get_xlim(),d['line_xlim_q'])
            self.assertIn('eff',fig.axes[0].get_title());plt.close(fig)

    def test_reference_focus_preserves_absolute_gi_and_marks_outer_subsystem(self):
        f=frame();f['epsilon1_GI']=f['epsilon1']-.22;f['epsilon2_GI']=f['epsilon2']-.22
        refs=dict(R=f['R'],q=f['q'],cbo=np.stack([f['epsilon1_GI']+.005*j for j in range(3)],axis=-1),
                  pol_energy=np.stack([f['epsilon2_GI']+.002*j for j in range(3)],axis=-1))
        original=f['epsilon2_GI'].copy()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'f.npz';np.savez(p,**f);d=prepare_display([p],refs)
            d['reference_focus_limits']=_prepare_reference_focus([p],[0],d,refs)
            fig=reference_focus_figure(f,d,refs);fig.canvas.draw()
            line=fig.axes[0].lines[0];valid=np.isfinite(line.get_ydata())
            np.testing.assert_array_equal(line.get_ydata()[valid],original[valid])
            np.testing.assert_array_equal(f['epsilon2_GI'],original)
            self.assertIn('electron + photon',fig.axes[0].get_title())
            self.assertIn('Absolute',fig.axes[0].get_ylabel());plt.close(fig)

    def test_external_references_and_output_directory_do_not_overwrite_old_report(self):
        f=frame()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=root/'case';(case/'fields').mkdir(parents=True)
            np.savez(case/'fields/fields_0000000.npz',**f)
            (case/'report').mkdir();sentinel=case/'report/keep.txt';sentinel.write_text('preserve')
            external=root/'refined.npz';np.savez(external,R=f['R'],q=f['q'],cbo=f['epsilon1'][:,:,None])
            out=root/'new_report'
            with mock.patch('multi_component_exact_factorization_cavity.render._save'):
                result=render_case(case,movies=False,output_dir=out,reference_path=external)
            self.assertEqual(sentinel.read_text(),'preserve')
            self.assertTrue((out/'render_manifest.json').exists())
            self.assertTrue(result['reference_cache']['external'])
            self.assertEqual(result['reference_cache']['path'],str(external.resolve()))
            self.assertTrue(all(str(out) in p for p in result['outputs']))

    def test_sparse_snapshot_camera_resets_instead_of_following_old_packet(self):
        f=frame();R,q=f['R'],f['q']
        with tempfile.TemporaryDirectory() as directory:
            files=[]
            for n,center in enumerate((-2.,2.)):
                joint=np.exp(-(R[:,None]+1)**2-((q[None,:]-center)/.25)**2)
                rho=joint.sum(axis=1)*(q[1]-q[0]);g=dict(f)
                g.update(t_fs=40.*n,rho_qR=joint,rho_R=rho,lambda_density=joint/rho[:,None])
                p=Path(directory)/f'f{n}.npz';np.savez(p,**g);files.append(p)
            sequence=prepare_display(files);single=prepare_display([files[-1]])
            np.testing.assert_allclose(sequence['windows'][-1],single['windows'][0])
            self.assertGreater(sequence['windows'][-1][2],.5)


if __name__=='__main__':
    unittest.main()
