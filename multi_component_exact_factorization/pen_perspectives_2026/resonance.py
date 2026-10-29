"""Bare-gap calibration, separate from the historical paper-table frequency.

Evaluate at the actual Gaussian centre, NOT its nearest FFT grid point. The
same bare frequency is used for all g/DSE controls; finite-coupling maximal
mixing and exactly equal LP/UP populations are not implied by this choice.
"""
from dataclasses import replace
import numpy as np
from .model import electronic


def frequency_report(c, selection='configured'):
    energies = np.linalg.eigvalsh(electronic(c, np.array([c.r2]))[0][0])
    gap = float(energies[1]-energies[0])
    return dict(selection=selection, reference='bare S1-S0 at initial Gaussian centre',
                R0=c.r2, bare_energies_Ha=energies.tolist(), bare_gap_Ha=gap,
                omega_au=c.omega, detuning_Ha=c.omega-gap,
                matches_bare_gap=bool(abs(c.omega-gap) < 1e-12),
                paper_table_on_omega_au=.17,
                note='Bare resonance, not finite-coupling equal mixing; not a claim of exact paper reproduction.')


def select_frequency(c, selection, preset='nonadiabatic'):
    if selection == 'bare-gap':
        c = replace(c, omega=frequency_report(c)['bare_gap_Ha'])
    elif selection in ('on', 'paper-on'):
        c = replace(c, omega=.17)
    elif selection == 'off':
        c = replace(c, omega=.12 if preset == 'rabi' else .097)
    elif selection != 'configured':
        raise ValueError(f'Unknown frequency selection: {selection}')
    c.grid()
    return c, frequency_report(c, selection)
