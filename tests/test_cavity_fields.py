"""Independent full-grid manufactured checks of the isolated cavity fields."""

import json
import unittest

import numpy as np
from scipy.fft import dst

from multi_component_exact_factorization_cavity.fields import (
    analyze, derivative, electronic_kinetic, metrics, ratio,
)


def sample(correlated=False):
    """Smooth full real-space state, not a two-state propagation surrogate."""
    nr, nx, nq = 40, 31, 64
    R = np.linspace(-np.pi, np.pi, nr, endpoint=False)
    x = np.linspace(-4., 4., nx+2)[1:-1]
    Q = np.linspace(-9., 9., nq, endpoint=False)
    dr, dx, dq = R[1]-R[0], x[1]-x[0], Q[1]-Q[0]
    phi = np.sqrt(2/8)*np.sin(np.pi*(x+4)/8)
    phi1 = np.sqrt(2/8)*np.sin(2*np.pi*(x+4)/8)
    vac = np.pi**(-.25)*np.exp(-Q**2/2)
    if correlated:
        angle = .3*np.sin(R[:, None])+.2*np.sin(Q[None, :])
        electronic = (np.cos(angle)[:, None, :]*phi[None, :, None]
                      + np.sin(angle)[:, None, :]*phi1[None, :, None]
                      * np.exp(.2j*np.cos(R))[:, None, None])
        wave = (np.exp(.2*np.cos(R))[:, None, None]*electronic*vac[None, None, :]
                * np.exp(.35j*np.sin(R[:, None])*Q[None, :])[:, None, :])
    else:
        wave = np.exp(1j*R)[:, None, None]*phi[None, :, None]*vac[None, None, :]
    wave /= np.sqrt(np.sum(abs(wave)**2)*dr*dx*dq)
    packet = dict(R=R, x=x, dx=dx, dR=dr, mass=8.,
                  V=np.zeros((nr, nx)), mu=R[:, None]-x[None, :])
    return wave, packet, Q


def action(wave, packet, Q, omega=.7, coupling=0.):
    """Direct independent kinetic and PF multiplication in Q normalization."""
    nr, nx, nq = wave.shape
    kr = 2*np.pi*np.fft.fftfreq(nr, packet['dR'])
    kq = 2*np.pi*np.fft.fftfreq(nq, Q[1]-Q[0])
    out = np.fft.ifft(np.fft.fft(wave, axis=0)*(.5*kr**2/packet['mass'])[:, None, None], axis=0)
    out += np.fft.ifft(np.fft.fft(wave, axis=2)*(.5*omega*kq**2), axis=2)
    kx = np.pi*np.arange(1, nx+1)/((nx+1)*packet['dx'])
    out += dst(dst(wave, type=1, axis=1, norm='ortho')*(.5*kx**2)[None, :, None],
               type=1, axis=1, norm='ortho')
    mu = packet['mu'][:, :, None]
    out += (packet['V'][:, :, None]+.5*omega*Q**2
            + np.sqrt(omega)*coupling*Q*mu+.5*coupling**2*mu**2)*wave
    return out


def test_separable_vacuum_positive_gauge_and_momentum():
    wave, packet, Q = sample()
    f = analyze(wave, packet, Q, .7, 0., action(wave, packet, Q))
    report = metrics(f)
    assert abs(report['norm']-1.) < 1e-14
    np.testing.assert_allclose(f['alpha'], 1., atol=2e-13)
    np.testing.assert_allclose(f['b'][f['support_qR']], 1., atol=2e-13)
    np.testing.assert_allclose(f['a'][f['support_qR']], 0., atol=1e-13)
    # In positive chi gauge the uniform nuclear plane-wave phase is in Phi.
    np.testing.assert_allclose(f['epsilon2'], -1/(2*packet['mass']), atol=3e-13)
    np.testing.assert_allclose(f['force'], 0., atol=1e-11)
    for name in ('epsilon1_route', 'epsilon2_route', 'alpha_bridge',
                 'epsilon2_bridge', 'continuity', 'lambda_eom', 'hamiltonian_budget'):
        assert report['residuals'][name]['rms'] < 2e-11, (name, report['residuals'][name])
    assert report['reconstruction_L2'] < 1e-14
    assert report['time_derivative_reconstruction_L2'] < 1e-14
    json.dumps(report, allow_nan=False)


def test_correlated_state_checks_bridge_and_complex_lambda_equation():
    wave, packet, Q = sample(correlated=True)
    f = analyze(wave, packet, Q, .7, .08, action(wave, packet, Q, coupling=.08), block=3)
    report = metrics(f)
    assert np.max(abs(f['delta_b'][f['support_qR']])) > .1
    assert np.max(abs(f['S_rel'][f['support_qR']])) > 1e-4
    for name in ('epsilon1_route', 'epsilon2_route', 'alpha_bridge', 'lambda_eom', 'hamiltonian_budget'):
        assert report['residuals'][name]['rms'] < 3e-10, (name, report['residuals'][name])
    # Quadrature/product-rule convergence is tested separately from exact algebra.
    assert report['residuals']['epsilon2_bridge']['rms'] < 2e-8
    assert report['residuals']['continuity']['rms'] < 2e-8
    assert report['reconstruction_L2'] < 1e-14
    assert report['time_derivative_reconstruction_L2'] < 1e-14
    expected = f['mean_e1']+f['Kq']+f['G_R_lambda']+f['Ct']
    np.testing.assert_allclose(f['epsilon2_reconstructed'], expected)
    np.testing.assert_allclose(f['G_R_lambda'], f['G_R_lambda_amplitude']+f['G_R_lambda_variance'])


def test_no_density_floor_and_exact_nodes_remain_explicit():
    result = ratio(np.array([1., 1., 1.]), np.array([0., 1e-100, 2.]))
    assert np.isnan(result[0])
    assert np.isclose(result[1], 1e100)
    assert result[2] == .5
    wave, packet, Q = sample()
    wave[:, :, len(Q)//2] = 0
    f = analyze(wave, packet, Q, .7, 0., action(wave, packet, Q))
    assert np.all(np.isnan(f['epsilon1'][:, len(Q)//2]))
    assert not np.any(f['support_qR'][:, len(Q)//2])
    assert np.isfinite(f['reconstruction_L2'])
    # Nodes may have a nonzero instantaneous Hamiltonian action.  A zero
    # support-only derivative residual must not conceal the excluded action.
    assert f['time_derivative_undefined_nodes_L2'] > 0


def test_dirichlet_electronic_kinetic_is_not_periodic_fft():
    wave, packet, _ = sample()
    energy = .5*(np.pi/8)**2
    np.testing.assert_allclose(electronic_kinetic(wave, packet['dx']), energy*wave, atol=5e-15)


def test_physical_q_coordinate_and_probability_preserved():
    wave, packet, Q = sample()
    omega = .09
    f = analyze(wave, packet, Q, omega, 0., action(wave, packet, Q, omega))
    np.testing.assert_allclose(f['q'], Q/np.sqrt(omega))
    np.testing.assert_allclose(f['rho_R'], np.sum(abs(wave)**2, axis=(1, 2))*packet['dx']*(Q[1]-Q[0]))
    np.testing.assert_allclose(np.sum(f['lambda_density'], axis=1)*np.diff(f['q'])[0], 1., atol=1e-14)


def test_reject_mismatched_grid_and_nonfinite_action():
    wave, packet, Q = sample()
    with unittest.TestCase().assertRaisesRegex(ValueError, 'axes'):
        analyze(wave, packet, Q, .7, 0., wave[:-1])
    with unittest.TestCase().assertRaisesRegex(ValueError, 'uniform'):
        analyze(wave, packet, Q**3, .7, 0., wave)
    action_bad = wave.copy()
    action_bad[0, 0, 0] = np.nan
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Nonfinite'):
        analyze(wave, packet, Q, .7, 0., action_bad)


def test_fft_derivative_plane_wave():
    grid = np.linspace(-np.pi, np.pi, 40, endpoint=False)
    f = np.exp(3j*grid)
    np.testing.assert_allclose(derivative(f, grid[1]-grid[0], 0), 3j*f, atol=8e-14)


def test_continuity_native_derivative_and_support_budget_are_explicit():
    wave, packet, Q = sample(correlated=True)
    f = analyze(wave, packet, Q, .7, .08, action(wave, packet, Q, coupling=.08))
    report = metrics(f)
    assert 'continuity_native_product_rule' in report['residuals']
    assert report['residuals']['continuity_native_product_rule']['rms'] < 1e-10
    assert set(report['support_scan']) == {'0.0001', '1e-05', '1e-06'}
    excluded = [report['support_scan'][str(t)]['excluded_joint_probability']
                for t in (1e-4, 1e-5, 1e-6)]
    assert excluded[0] >= excluded[1] >= excluded[2]
    assert not report['phase7_pass']


def test_algebraic_closure_does_not_hide_underresolved_current_products():
    wave, packet, Q = sample()
    R = packet['R']
    # Two resolved wave modes produce a density/current beat beyond Nyquist.
    # This is deliberately NOT a converged physical calculation.
    wave *= (np.exp(14j*R)[:, None, None]
             + .3*np.exp(-11j*R)[:, None, None]*np.sqrt(2)*Q)
    wave /= np.sqrt(np.sum(abs(wave)**2)*packet['dx']*packet['dR']*(Q[1]-Q[0]))
    f = analyze(wave, packet, Q, .7, 0., action(wave, packet, Q))
    report = metrics(f)
    assert report['residuals']['continuity']['rms'] < 1e-10
    assert report['residuals']['continuity_native_product_rule']['rms'] > 1e-3


def load_tests(loader, tests, pattern):
    """Keep independent fixtures runnable with the repository's unittest CLI."""
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in
                              sorted(globals().items()) if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
