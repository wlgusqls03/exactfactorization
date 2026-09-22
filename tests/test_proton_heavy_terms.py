import tempfile
import unittest
from pathlib import Path
import numpy as np
from multi_component_exact_factorization.core import AU_PER_FS
from multi_component_exact_factorization.proton_heavy_terms import (
    TermConfig, frame_terms, summarize_frame, time_rate, peak_integrals)


class ProtonHeavyTermTests(unittest.TestCase):
    def fixture(self,n=32):
        q=np.arange(n)*2*np.pi/n-np.pi;R=q.copy();dq=q[1]-q[0]
        heavy=(1+.1*np.cos(R))/(2*np.pi)
        den=(1+.2*np.cos(q[:,None]-R[None,:]))/(2*np.pi)
        return q,R,dq,den*heavy,heavy

    def test_eight_terms_and_independent_operator(self):
        q,R,dx,rho,h=self.fixture()
        a=np.zeros_like(rho);b=.3+.1*np.sin(q[:,None]+R);alpha=.3+R*0
        result=frame_terms(rho,h,a,b,alpha,dx,dx,2,10,a,
                           TermConfig(1e-5,1e-12,0,'site'))
        f=result['fields']
        np.testing.assert_allclose(sum(f[f'T{i}'] for i in range(1,9)),f['U_total'],atol=1e-15)
        np.testing.assert_allclose(f['U_lin'],f['U_lin_operator'],atol=1e-15)
        np.testing.assert_allclose(f['U_operator']-f['U_total'],f['residual_product_rule'])
        self.assertGreater(np.max(abs(f['residual_product_rule'])),1e-9)
        for i in range(1,5):self.assertTrue(np.isrealobj(f[f'T{i}']))
        for i in range(5,9):np.testing.assert_array_equal(f[f'T{i}'].real,0)

    def test_continuum_source_convergence(self):
        errors=[]
        for n in (24,48):
            q,R,dx,rho,h=self.fixture(n)
            den=rho/h;a=np.ones_like(rho)*.4;b=np.ones_like(rho)*.3;alpha=np.ones_like(R)*.3
            # delta=0; q and R transport of cos(q-R), mp=2, M=10.
            exact=(.4/2-.3/10)*.2*np.sin(q[:,None]-R)/(2*np.pi)
            f=frame_terms(rho,h,a,b,alpha,dx,dx,2,10,exact,
                           TermConfig(1e-5,1e-12,0,'site'))['fields']
            errors.append(np.max(abs(f['residual_density'])))
            np.testing.assert_allclose(f['S_U_operator'],f['S_U_expanded'],atol=1e-15)
        self.assertLess(errors[1],errors[0]/12)

    def test_constant_factors(self):
        rho=np.ones((12,16))/12;h=np.ones(16);zero=np.zeros_like(rho)
        f=frame_terms(rho,h,zero,zero+3,h,1,.1,2,10,zero)['fields']
        np.testing.assert_allclose(f['T3'],.2*np.sqrt(rho))
        np.testing.assert_allclose(f['T4'],.2*np.sqrt(rho))
        np.testing.assert_allclose(f['U_total'],.4*np.sqrt(rho),atol=1e-13)
        np.testing.assert_allclose(f['residual_density'],0,atol=1e-13)

    def test_time_units_nonuniform_and_endpoint(self):
        times=np.array([0,1,3,4.])/AU_PER_FS
        get=lambda f: np.array([0,1,9,16.])[f:f+1]
        self.assertAlmostEqual(time_rate(get,times,1)[0],2)
        self.assertAlmostEqual(time_rate(get,times,0)[0],1)
        with self.assertRaises(ValueError):time_rate(get,times[::-1],1)

    def test_mask_and_peak_integrals(self):
        q,R,dx,rho,h=self.fixture()
        h[0]=0;rho[:,0]=0;z=np.zeros_like(rho)
        result=frame_terms(rho,h,z,z,R*0,dx,dx,2,10,z,TermConfig(1e-5))
        self.assertFalse(result['valid'][:,0].any())
        self.assertTrue(result['valid'][:,5:-5].all())
        p=peak_integrals(result,q,dx,0)
        np.testing.assert_allclose((p['P_left']+p['P_right'])[1:],1,atol=1e-14)
        self.assertTrue(np.isnan(p['P_right'][0]))
        self.assertIsNotNone(summarize_frame(result,dx,dx)['rms']['S_q'])

    def test_report_smoke_all_groups(self):
        from multi_component_exact_factorization.proton_heavy_report import render_proton_heavy
        from multi_component_exact_factorization.render_final_visualizations import parse_args
        q,R,dx,rho,h=self.fixture(12)
        times=np.array([0.,.1,.2]);z=np.zeros((3,*rho.shape))
        obs=dict(times_fs=times,q=q,R=R,dq=dx,dR=dx,joint_density=np.array([rho]*3),
                 heavy_density=np.array([h]*3), options=dict(proton_mass=2.,heavy_mass=10.))
        ef=dict(a=z,b=z,alpha=np.zeros((3,len(R))),gauge='positive_density')
        args=parse_args(['unused','--only','proton_heavy','--no-animation','--dpi','40','--snapshot-count','1'])
        with tempfile.TemporaryDirectory() as tmp:
            products=render_proton_heavy(obs,ef,Path(tmp),args,[0])
            self.assertTrue(all(p.exists() for p in products))
            with np.load(Path(tmp)/'proton_heavy_terms_frame_0000.npz') as data:
                self.assertIn('T8',data.files)
                np.testing.assert_allclose(data['residual_density'][data['valid']],0,atol=1e-12)
            args.no_animation=False;args.format='gif';args.animation_dpi=40;args.max_frames=3
            args.ph_groups=['density']
            movie_products=render_proton_heavy(obs,ef,Path(tmp)/'movie',args,[0])
            self.assertTrue(any(p.suffix=='.gif' and p.stat().st_size>0 for p in movie_products))


if __name__=='__main__':unittest.main()
