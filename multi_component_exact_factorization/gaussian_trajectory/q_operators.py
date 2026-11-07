"""Explicit periodic q discretizations for the Gaussian trajectory experiment.

FFT remains the default. FD2/FD4 are opt-in, Hermitian finite-difference
Hamiltonians on the SAME endpoint-excluded grid and rectangle quadrature.
Their local roll implementation avoids sending round-off from an occupied
point to every tail point through a global transform. This is a different
finite-grid operator, not a density mask, filter, or claim of convergence.

The first derivative is antisymmetric. Its square is NOT the second-derivative
stencil at finite h; kinetic energy and photon p^2 must use the latter.
"""
import numpy as np


OPERATORS = ('fft', 'fd2', 'fd4')


def _check(values, spacing, operator, axis):
    data = np.asarray(values)
    if operator not in OPERATORS:
        raise ValueError('q_operator must be fft, fd2, or fd4')
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Periodic q spacing must be positive and finite')
    if data.ndim < 1 or not -data.ndim <= axis < data.ndim:
        raise ValueError('Invalid q axis')
    axis %= data.ndim
    if data.shape[axis] < (5 if operator == 'fd4' else 3):
        raise ValueError('Too few q points for the selected stencil')
    return data, axis


def periodic_derivative(values, spacing, *, operator='fft', order=1, axis=1):
    """Apply D_q or D_qq on an endpoint-excluded uniform periodic grid."""
    data, axis = _check(values, spacing, operator, axis)
    if order not in (1, 2):
        raise ValueError('Only derivative orders 1 and 2 are supported')
    if operator == 'fft':
        k = 2*np.pi*np.fft.fftfreq(data.shape[axis], spacing)
        shape = [1]*data.ndim
        shape[axis] = len(k)
        return np.fft.ifft(np.fft.fft(data, axis=axis)*(1j*k.reshape(shape))**order, axis=axis)
    plus = np.roll(data, -1, axis=axis)
    minus = np.roll(data, 1, axis=axis)
    if operator == 'fd2':
        return ((plus-minus)/(2*spacing) if order == 1 else
                (plus-2*data+minus)/spacing**2)
    plus2 = np.roll(data, -2, axis=axis)
    minus2 = np.roll(data, 2, axis=axis)
    return ((-plus2+8*plus-8*minus+minus2)/(12*spacing) if order == 1 else
            (-plus2+16*plus-30*data+16*minus-minus2)/(12*spacing**2))


def kinetic_symbol(nq, spacing, mass, *, operator='fft'):
    """Fourier eigenvalues of the chosen kinetic; not used for FD action."""
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('q kinetic mass must be positive and finite')
    _check(np.empty(nq), spacing, operator, 0)
    k = 2*np.pi*np.fft.fftfreq(nq, spacing)
    if operator == 'fft':
        return k*k/(2*mass)
    s2 = np.sin(.5*k*spacing)**2
    correction = 1. if operator == 'fd2' else 1.+s2/3.
    return 2*s2*correction/(mass*spacing**2)


def kinetic_action(values, spacing, mass, *, operator='fft', axis=1):
    """Apply -D_qq/(2m), using local arithmetic for finite differences."""
    data, axis = _check(values, spacing, operator, axis)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('q kinetic mass must be positive and finite')
    if operator == 'fft':
        shape = [1]*data.ndim
        shape[axis] = data.shape[axis]
        symbol = kinetic_symbol(data.shape[axis], spacing, mass, operator=operator).reshape(shape)
        return np.fft.ifft(np.fft.fft(data, axis=axis)*symbol, axis=axis)
    return -periodic_derivative(data, spacing, operator=operator, order=2, axis=axis)/(2*mass)


def kinetic_matrix(nq, spacing, mass, *, operator='fft'):
    """Dense small-grid audit/channel matrix of precisely the action above."""
    matrix = kinetic_action(np.eye(nq), spacing, mass, operator=operator, axis=0)
    if np.max(abs(np.asarray(matrix).imag)) > 1e-10:
        raise ValueError('Periodic kinetic matrix is unexpectedly complex')
    return np.asarray(matrix).real
