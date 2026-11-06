"""T11: offline sampling of an exact free 2D quantum Gaussian snapshot.

This analytic wave solves the free R,q TDSE and is sampled at current,
nonuniform R locations. It is a reconstruction test, not future QD data fed
into Gaussian-trajectory propagation and not an accuracy claim for a closure.
"""
import unittest

import numpy as np

from multi_component_exact_factorization.gaussian_trajectory.fields import (
    analyze_fields, periodic_q_derivative)
from multi_component_exact_factorization.gaussian_trajectory.mls import (
    MLSConfig, reconstruct_R_derivatives)


def exact_free_snapshot(R, q, time):
    """Normalized correlated complex Gaussian for H=-d_R²/6-d_q²/2."""
    inverse_mass = np.diag([1/3, 1.])
    A0 = np.array([[1.3+.1j, .2+.15j],[.2+.15j, .7-.2j]])
    p = np.array([.7, -.4])
    center = np.array([.2, -.3]) + time * inverse_mass @ p
    A = np.linalg.inv(np.linalg.inv(A0) + 1j*time*inverse_mass)
    x = np.stack(np.broadcast_arrays(np.asarray(R)[:,None]-center[0],
                                    np.asarray(q)[None,:]-center[1]), axis=-1)
    prefactor = np.linalg.det(A0.real)**.25/np.sqrt(np.pi)
    prefactor /= np.sqrt(np.linalg.det(np.eye(2)+1j*time*inverse_mass@A0))
    # Plane-wave momentum is written relative to the initial center; the
    # constant phase makes this the exact freely translated Gaussian solution.
    initial_relative_x = x + time * (inverse_mass @ p)
    phase = np.einsum('...i,i->...', initial_relative_x,p) - .5*time*(p@inverse_mass@p)
    psi = prefactor*np.exp(-.5*np.einsum('...i,ij,...j->...',x,A,x)+1j*phase)
    gamma = A.real[0,0] - A.real[0,1]**2/A.real[1,1]
    outer_density = np.sqrt(gamma/np.pi)*np.exp(-gamma*(np.asarray(R)-center[0])**2)
    G = psi / np.sqrt(outer_density[:,None])
    log_R = (gamma-A[0,0])*x[:,:,0]-A[0,1]*x[:,:,1]+1j*p[0]
    log_q = -A[1,0]*x[:,:,0]-A[1,1]*x[:,:,1]+1j*p[1]
    return dict(psi=psi, G=G[:,:,None], GR=(log_R*G)[:,:,None],
                GRR=((log_R**2+gamma-A[0,0])*G)[:,:,None],
                Gq=(log_q*G)[:,:,None], chi_density=outer_density,
                gamma=gamma, center=center, matrix=A, displacement=x, momentum=p,
                inverse_mass=inverse_mass)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.q = np.linspace(-12,12,256,endpoint=False)
        self.w = np.full(len(self.q),self.q[1]-self.q[0])
        self.kw = dict(basis_convention="one fixed primitive electronic channel",
                       gauge_convention="positive outer marginal; exact free wave phase retained")

    def test_reference_wave_really_solves_free_TDSE(self):
        R = np.linspace(-2,2,17)
        t, dt = .35, 1e-5
        exact = exact_free_snapshot(R,self.q,t)
        psi_t = (exact_free_snapshot(R,self.q,t+dt)["psi"]-
                 exact_free_snapshot(R,self.q,t-dt)["psi"])/(2*dt)
        loggrad = -np.einsum('ij,...j->...i',exact["matrix"],exact["displacement"])+1j*exact["momentum"]
        laplacian = loggrad**2-np.diag(exact["matrix"])
        Hpsi = -.5*np.sum(laplacian*np.diag(exact["inverse_mass"]),axis=-1)*exact["psi"]
        self.assertLess(np.linalg.norm(1j*psi_t-Hpsi)/np.linalg.norm(Hpsi),1e-9)

    def test_sampled_nonuniform_R_derivatives_converge(self):
        errors = []
        for n in (21,41,81):
            u = np.linspace(-1.,1.,n)
            R = u+.08*np.sin(3*u)
            R = R[np.random.default_rng(8).permutation(n)]
            exact = exact_free_snapshot(R,self.q,.35)
            d, d2, _ = reconstruct_R_derivatives(R,exact["G"],
                        config=MLSConfig(degree=4,neighbors=9),**self.kw)
            errors.append([np.linalg.norm(d-exact["GR"])/np.linalg.norm(exact["GR"]),
                           np.linalg.norm(d2-exact["GRR"])/np.linalg.norm(exact["GRR"])])
        errors = np.array(errors)
        self.assertTrue(np.all(errors[:-1,0]/errors[1:,0] > 8),errors)
        self.assertTrue(np.all(errors[:-1,1]/errors[1:,1] > 5),errors)
        self.assertLess(errors[-1,0],1e-5)
        self.assertLess(errors[-1,1],2e-4)

    def test_snapshot_fft_and_positive_gauge_fields(self):
        R = np.linspace(-1,1,121)
        exact = exact_free_snapshot(R,self.q,.35)
        np.testing.assert_allclose(periodic_q_derivative(exact["G"],self.q),exact["Gq"],atol=1e-12)
        fields = analyze_fields(R,exact["G"],self.w,self.q,
                              mls_config=MLSConfig(degree=4,neighbors=9),**self.kw)
        x = exact["displacement"]
        phase_R = -(exact["matrix"][0,0].imag*x[:,:,0]+exact["matrix"][0,1].imag*x[:,:,1])+exact["momentum"][0]
        density = np.abs(exact["G"][:,:,0])**2
        exact_alpha = (phase_R*density)@self.w
        np.testing.assert_allclose(fields["alpha"],exact_alpha,atol=5e-7)
        self.assertLess(np.max(fields["q_edge_mass"]),1e-20)
        self.assertLess(np.max(np.abs(fields["pnc_error"])),2e-14)
        # Gauge covariance: multiply G by exp(i*kR), leave positive |chi|.
        # This changes the outer EF gauge unless chi receives exp(-i*kR).
        # The reconstruction must retain, not auto-align away, that gradient.
        k = .17
        transformed = analyze_fields(R,exact["G"]*np.exp(1j*k*R)[:,None,None],
                 self.w,self.q,mls_config=MLSConfig(degree=4,neighbors=9),
                 basis_convention=self.kw["basis_convention"],
                 gauge_convention="common theta(R)=kR relative to reference gauge")
        np.testing.assert_allclose(transformed["alpha"],fields["alpha"]+k,atol=5e-7)
        np.testing.assert_allclose(transformed["a"],fields["a"],atol=1e-7,equal_nan=True)


if __name__ == "__main__":
    unittest.main()
