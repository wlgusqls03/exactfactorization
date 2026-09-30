"""Independent checks of saved-state photon coordinate-grid diagnostics."""
import numpy as np
from scipy.integrate import simpson
from multi_component_exact_factorization.vsc_polariton.photon_uniform_grid_audit import (
    hermite, quadratic_density, grid_probe,
)


def test_complex_gram_matches_explicit_wave_and_derivatives():
    rng = np.random.default_rng(831)
    c = rng.normal(size=(5,8)) + 1j*rng.normal(size=(5,8))
    G = c.conj().T @ c
    Q = np.linspace(-10,10,501)
    for B in hermite(Q,8):
        np.testing.assert_allclose(quadratic_density(G,B), np.sum(abs(c@B)**2,axis=0),
                                   atol=2e-13,rtol=2e-13)


def test_vacuum_norm_moments_and_derivative():
    Q = np.linspace(-12,12,2401)
    b,d,dd = hermite(Q,1)
    np.testing.assert_allclose(d[0],-Q*b[0],atol=1e-15)
    np.testing.assert_allclose(dd[0],(Q**2-1)*b[0],atol=1e-15)
    np.testing.assert_allclose(simpson(b[0]**2,x=Q),1.,atol=1e-14)
    np.testing.assert_allclose(simpson(Q**2*b[0]**2,x=Q),.5,atol=1e-14)


def test_fft_probe_backprojection_matches_explicit_coefficients():
    rng=np.random.default_rng(91)
    c=rng.normal(size=(4,9))+1j*rng.normal(size=(4,9))
    c/=np.linalg.norm(c)
    Q=np.linspace(-3,3,64,endpoint=False)
    B=hermite(Q,9)[0]
    explicit=c@B@B.T*(6/64)-c
    result=grid_probe(c.conj().T@c,3,64)
    np.testing.assert_allclose(result['backprojection_L2'],np.linalg.norm(explicit),atol=1e-14)


def test_coarse_grid_fails_but_resolved_grid_recovers_excited_state():
    G=np.zeros((80,80));G[-1,-1]=1.
    coarse=grid_probe(G,18,64)
    fine=grid_probe(G,18,512)
    assert coarse['derivative2_relative_L2']>1e-2
    assert fine['norm_error']<1e-12
    assert fine['derivative2_relative_L2']<1e-9
