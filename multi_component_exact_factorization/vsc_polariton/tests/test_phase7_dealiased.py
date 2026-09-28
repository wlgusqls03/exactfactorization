"""Independent analytic checks for the event gauge-resolution audit."""
import unittest
import numpy as np
from scipy.fft import fft
from ..phase7_dealiased_gauge import evaluate,bilinears,shifted_overlap,probe


class DealiasedTests(unittest.TestCase):
    def test_native_nyquist_is_not_split(self):
        n=12;x=np.arange(n)/n;fine=np.arange(48)/48
        u=np.exp(-2j*np.pi*(n//2)*x)
        np.testing.assert_allclose(evaluate(fft(u),48),np.exp(-2j*np.pi*(n//2)*fine),atol=1e-13)
        np.testing.assert_allclose(evaluate(fft(u),48,1),-2j*np.pi*(n//2)*np.exp(-2j*np.pi*(n//2)*fine),atol=1e-11)

    def test_bilinears_match_direct_wave(self):
        rng=np.random.default_rng(7);n=12;dr=.3;dx=.2
        u=rng.normal(size=(n,3,2))+1j*rng.normal(size=(n,3,2));ut=1j*u
        cache=bilinears(u,ut,dx,dr,block=2)
        fine=evaluate(fft(u,axis=0),48)
        rho=np.sum(abs(fine)**2,axis=(1,2))*dx
        np.testing.assert_allclose(evaluate(cache,48).real[:,0],rho,atol=1e-13)
        shift=.012
        shifted=evaluate(fft(u,axis=0),48,length=n*dr,shift=shift)
        np.testing.assert_allclose(evaluate(shifted_overlap(u,dx,dr,shift,2),48),
            np.sum(fine.conj()*shifted,axis=(1,2))*dx,atol=1e-13)

    def test_free_interfering_wave_has_zero_force(self):
        n=32;R=np.arange(n)*2*np.pi/n;M=2.
        u=(np.exp(1j*R)+.4*np.exp(-2j*R))[:,None,None]
        ut=-1j*(np.exp(1j*R)/(2*M)+.4*4*np.exp(-2j*R)/(2*M))[:,None,None]
        cache=bilinears(u,ut,1.,R[1]-R[0]);factor=16;count=n*factor
        h=2*np.pi/count
        links={s:evaluate(shifted_overlap(u,1.,R[1]-R[0],s*h),count) for s in (-3,-2,-1,1,2,3)}
        fields,report=probe(cache,R,M,factor,1e-10,links)
        self.assertLess(report['force_RMS'],1e-10)
        self.assertTrue(report['force_pass']);self.assertTrue(report['gauge_pass'])
        self.assertLess(report['scalar_imag_RMS'],1e-12)


if __name__=='__main__':unittest.main()
