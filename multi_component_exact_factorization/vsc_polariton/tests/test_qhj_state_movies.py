"""Analytic reference-basis projection checks, no propagation."""
import unittest
import numpy as np
from multi_component_exact_factorization.vsc_polariton.run_qhj_state_movies import populations,curvature_terms


class ProjectionTests(unittest.TestCase):
    def test_curvature_split(self):
        R=np.linspace(-10,10,256,endpoint=False);Q=R.copy()
        psi=np.exp(-R[:,None,None]**2/2-Q[None,None,:]**2/2).astype(complex)
        cq,cr=curvature_terms(psi,Q,dict(omega=.2,mass=10,dR=R[1]-R[0]))
        m=(abs(R)<3);ix=np.ix_(m,m)
        np.testing.assert_allclose(cq[ix],np.broadcast_to(.1*(Q**2-1),cq.shape)[ix],atol=1e-9)
        np.testing.assert_allclose(cr[ix],np.broadcast_to(.05*(R[:,None]**2-1),cr.shape)[ix],atol=1e-9)

    def test_reference_ground_and_first(self):
        Q=np.linspace(-12,12,1024,endpoint=False)
        shift=np.array([-.7,.4]);y=Q[None,:]+shift[:,None]
        h=np.pi**(-.25)*np.exp(-y*y/2)
        for n,u in enumerate((h,np.sqrt(2)*y*h)):
            rho,w,gram=populations(u[:,None,:],Q,shift,1.)
            np.testing.assert_allclose(rho,1,atol=1e-13)
            np.testing.assert_allclose(w[n],1,atol=1e-13)
            np.testing.assert_allclose(w[1-n],0,atol=1e-13)
            self.assertLess(gram,1e-13)

    def test_remainder_not_discarded(self):
        Q=np.linspace(-12,12,1024,endpoint=False);y=Q[None,:]
        h=np.pi**(-.25)*np.exp(-y*y/2)
        h2=(2*y*y-1)*h/np.sqrt(2)
        rho,w,_=populations(h2[:,None,:],Q,np.zeros(1),1.)
        np.testing.assert_allclose(rho-w.sum(axis=0),1,atol=1e-13)


if __name__=='__main__':unittest.main()
