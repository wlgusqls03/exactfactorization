"""Portable synthetic algebra tests, not literature production parameters."""
import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path
import numpy as np
from multi_component_exact_factorization.vsc_polariton.phase6_completion_controls import reduced_packet, stationary
from multi_component_exact_factorization.vsc_polariton.phase6_completion_audit import compare, diagnostic
from multi_component_exact_factorization.vsc_polariton.phase6_gpu_backend import PFBackend
from multi_component_exact_factorization.vsc_polariton.phase6_orthogonal_packets import orthogonal_rotation


def packet():
    nr,nx,nf=12,8,6
    R=np.arange(nr)*.2; x=np.arange(nx)*.3
    phi=np.ones((nr,nx,1))/np.sqrt(nx*.3)
    u=np.zeros((nr,nx,nf),complex);u[:,:,0]=phi[:,:,0]/np.sqrt(nr*.2)
    v,rot,_=orthogonal_rotation(nf)
    return dict(R=R,x=x,dx=.3,dR=.2,mass=1836.,omega=.01,g_chi=0.,
                psi=u,phi=phi,tx=(2*np.pi*np.fft.fftfreq(nx,.3))**2/2,
                tr=(2*np.pi*np.fft.fftfreq(nr,.2))**2/(2*1836.),
                photon=.01*(np.arange(nf)+.5),potential=np.zeros((nr,nx)),
                dse=np.zeros((nr,nx)),mu=R[:,None]-x[None,:],
                rotation=rot,displacement=v)


class CompletionTests(unittest.TestCase):
    def test_projection_and_no_coupling(self):
        p=packet();m=dict(R=p['R'],E=np.zeros(12),mu=np.zeros(12))
        for model in 'ABC':
            r,f=reduced_packet(p,m,model)
            h=PFBackend(r,.125);row=h.observe(r['psi'])
            self.assertAlmostEqual(row['norm'],1.,places=12)
            self.assertLess(np.max(abs(f['DBOC'])),1e-25)
            self.assertLess(np.max(abs(h.step(r['psi']))**2-abs(r['psi'])**2),1e-12)

    def test_stationary_invariant_sector(self):
        with tempfile.TemporaryDirectory() as d:
            stationary(packet(),Path(d)/'stationary.json',0,gpu=False)

    def test_comparison_rejects_nan(self):
        a={k:np.zeros(414) for k in ('product','flux','nph','mean_R','P_exc')}
        a['time_au']=np.arange(414)*4.
        self.assertTrue(compare(a,a)['pass_checks'])
        b={k:v.copy() for k,v in a.items()};b['product'][4]=np.nan
        self.assertFalse(compare(a,b)['pass_checks'])

    def test_absolute_gate(self):
        a={k:np.zeros(414) for k in ('product','flux','nph','mean_R','P_exc','energy','electron_edge','edge','top','continuity_error')}
        a.update(time_au=np.arange(414)*4.,norm=np.ones(414))
        self.assertTrue(diagnostic(a)['pass_checks'])
        a['norm'][3]+=2e-9
        self.assertFalse(diagnostic(a)['pass_checks'])

    def test_complete_audit_never_auto_starts_phase7(self):
        from multi_component_exact_factorization.vsc_polariton.phase6_completion_audit import main
        a={k:np.zeros(414) for k in ('product','flux','nph','mean_R','P_exc','energy','electron_edge','edge','top','continuity_error')}
        a.update(time_au=np.arange(414)*4.,norm=np.ones(414))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs=root/'inputs';inputs.mkdir();(root/'controls').mkdir()
            np.savez(inputs/'resonant_observables.npz',**a)
            (inputs/'historical_validation.json').write_text(json.dumps(dict(computed_gates_pass=True)))
            (root/'controls/stationary.json').write_text(json.dumps(dict(status='PASS')))
            for case in ('free','resonant','barrier'):
                for model in 'ABC':
                    for suffix in ('dt0.125','dt0.0625','F160_dt0.125'):
                        np.savez(root/'controls'/f'{case}_{model}_{suffix}.npz',**a)
            # Mock the directory collector only; real compact control filenames
            # are opened, exercising decimal-dt naming and all 27 comparisons.
            from multi_component_exact_factorization.vsc_polariton.phase6_completion_audit import collect
            def read(path):
                return a if path.name=='full' else collect(path)
            with patch('sys.argv',['audit','--out',str(root),'--inputs',str(inputs)]), \
                 patch('multi_component_exact_factorization.vsc_polariton.phase6_completion_audit.collect',side_effect=read), \
                 patch('builtins.print'):
                main()
            result=json.loads((root/'phase6_completion_validation.json').read_text())
            self.assertEqual(result['status'],'COMPUTED_GATES_PASS_REVIEW_PENDING')
            self.assertFalse(result['phase7_allowed'])


if __name__=='__main__':unittest.main()
