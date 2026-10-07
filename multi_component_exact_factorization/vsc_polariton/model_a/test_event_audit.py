"""Read-only audit tests; original campaign outputs are never mutated."""
from dataclasses import asdict
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import numpy as np
from .model import Config,initial
from .event_audit import integral_fourier,populations,fields,compare_fields,same_wave,sha,audit,CASES


class EventAuditTests(unittest.TestCase):
    def test_gaussian_endpoint_bias(self):
        results=[]
        for n in [500,600]:
            R=np.linspace(0,8,n,endpoint=False);h=R[1]-R[0]
            rho=np.exp(-((R-4)/.223)**2)/(.223*np.sqrt(np.pi))
            raw=float(rho[R>=4].sum()*h);exact=integral_fourier(rho,R,4,8)
            self.assertAlmostEqual(exact,.5,places=13);results.append(raw)
        self.assertGreater(abs(results[0]-results[1]),.003)

    def test_wave_population_normalization(self):
        c=Config(nr=128,nq=64,qmin=-16,qmax=16)
        p=populations(initial(c),c)
        self.assertAlmostEqual(p['norm'],1,places=12)
        self.assertAlmostEqual(p['resampled_norm'],1,places=12)
        self.assertLess(abs(p['spectral_wave2R']),1e-12)

    def test_trigonometric_integral_nonzero_origin(self):
        R=np.linspace(-1,9,128,endpoint=False);k=2*np.pi/10
        rho=1+.1*np.cos(k*(R+1))
        expected=5+.1/k*(np.sin(k*10)-np.sin(k*5))
        self.assertAlmostEqual(integral_fourier(rho,R,4,9),expected,places=12)

    def test_identity_fields_and_qhj_closure(self):
        c=Config(nr=128,nq=128,qmin=-16,qmax=16);f=fields(initial(c),c)
        report=compare_fields(f,f)
        self.assertEqual(report['supports']['1e-06']['epsilon1']['rms'],0.)
        mask=f['rho_qR']>1e-6*f['rho_qR'].max()
        residual=f['epsilon1'].real-sum(f[k] for k in ['Q_q','flow_q','Q_R','flow_R'])
        self.assertLess(np.max(abs(residual[mask])),1e-11)

    def test_same_wave_sampling_preserves_initial_fields(self):
        c=Config(nr=128,nq=128,qmin=-16,qmax=16);u=initial(c)
        f=fields(u,c);g=same_wave(u,c);report=compare_fields(f,g)
        self.assertLess(report['rho_L1'],1e-12)
        self.assertLess(report['supports']['0.0001']['epsilon1']['rms'],1e-8)

    def test_complete_package_and_input_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'campaign';root.mkdir()
            c=Config(nr=32,nq=32,qmin=-16,qmax=16)
            source=Path(__file__).parent
            ident=dict(config=asdict(c),source={f:sha(source/f) for f in ['model.py','factorization.py']})
            for name in CASES:
                folder=root/name;folder.mkdir();(folder/'waves').mkdir()
                (folder/'identity.json').write_text(json.dumps(ident))
                np.savez(folder/'waves/wave_0000.npz',psi=initial(c),time_au=0.)
            before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
            result=audit(root,base/'audit',times=(0,),refine=False)
            self.assertEqual(result['status'],'DIAGNOSTIC_COMPLETE_NOT_CERTIFIED')
            self.assertFalse(result['phase_pass'])
            self.assertEqual(before,{str(p):sha(p) for p in root.rglob('*') if p.is_file()})
            with tarfile.open(base/'audit/model_a_event_review.tar.gz') as tar:
                self.assertEqual(len([n for n in tar.getnames() if '/waves/' in n]),6)
                self.assertTrue(all('restart' not in n and '/fields/' not in n for n in tar.getnames()))
            with self.assertRaises(FileExistsError):audit(root,base/'audit',times=(0,))
            with self.assertRaises(FileNotFoundError):audit(root,base/'missing',times=(1000,))
            self.assertFalse((base/'missing').exists())


if __name__=='__main__':unittest.main()
