"""Bare-gap calibration, separate from the historical paper-table frequency.

Evaluate at the actual Gaussian centre, NOT its nearest FFT grid point. The
same bare frequency is used for all g/DSE controls; finite-coupling maximal
mixing and exactly equal LP/UP populations are not implied by this choice.
"""
from dataclasses import replace
import numpy as np
from .model import AU_FS, Config, electronic, local_hamiltonian


PAPER_FREQUENCIES = {'nonadiabatic': {'on': .17, 'off': .097},
                     'rabi': {'on': .17, 'off': .12}}
PAPER_FIGURE_WINDOWS_FS = {'nonadiabatic': {'on': 30., 'off': 40.},
                          'rabi': {'on': 30., 'off': 3.5}}


def paper_config(model, resonance, **overrides):
    """Table 2, not a fit to 50:50 mixing. Overrides are numerical/controls only."""
    omega = PAPER_FREQUENCIES[model][resonance]
    allowed = {'nr', 'dt', 'nf', 'g', 'dse'}
    if set(overrides)-allowed:
        raise ValueError('Paper presets allow only numerical, g and DSE overrides')
    c = Config(r1=2. if model == 'rabi' else 6., delta=.17 if model == 'rabi' else 0.,
               omega=omega, **overrides)
    c.grid()
    return c


def scenario_report(c, model=None, resonance=None):
    """Model labels and a LOCAL two-level RWA estimate, never a TDSE result.

    The estimate omits counter-rotating terms, diagonal dipoles, nuclear motion
    and DSE shifts. In particular its on/off periods are not fit to the paper.
    """
    if model is None:
        model = 'rabi' if c.r1 == c.r2 and c.delta == .17 else 'nonadiabatic'
    if resonance is None:
        resonance = next((key for key, value in PAPER_FREQUENCIES[model].items()
                          if abs(c.omega-value) < 1e-12), 'custom')
    h, dip = electronic(c, np.array([c.r2]))
    energies, vectors = np.linalg.eigh(h[0])
    gap = float(energies[1]-energies[0])
    mu = float((vectors.T@dip[0]@vectors)[0, 1])
    v = c.g*np.sqrt(c.omega/2)*mu
    splitting = float(np.hypot(gap-c.omega, 2*v))
    return dict(model=model, resonance=resonance, omega_au=c.omega,
                label=f'{"Rabi oscillations" if model == "rabi" else "Nonadiabatic process"}; {resonance}-resonance',
                paper_doi='10.1039/D6CP02020H', paper_parameters='Table 2; Eqs. 21-27',
                paper_figure_time_window_fs=PAPER_FIGURE_WINDOWS_FS[model].get(resonance),
                paper_baseline=bool(resonance in ('on', 'off') and c == paper_config(model, resonance)),
                initial_state='Eq.27: bare adiabatic S1(R) * real Gaussian at R2 * Fock vacuum',
                finite_fock_states=c.nf, nuclear_points=c.nr, dt_au=c.dt,
                frequency=frequency_report(c, 'paper-'+resonance),
                local_pol_energies_R0_Ha=np.linalg.eigvalsh(local_hamiltonian(c, np.array([c.r2]))[0]).tolist(),
                rwa_reference=dict(splitting_Ha=splitting,
                    period_fs=2*np.pi*AU_FS/splitting if splitting > 1e-14 else None,
                    transfer_amplitude=4*v*v/splitting**2 if splitting > 1e-14 else 0.,
                    transition_dipole_au=mu,
                    limitation='Local two-level RWA only; not full-model TDSE or fitted paper period'),
                interpretation=('R>=4 probability/flux are region diagnostics, NOT a reaction yield in this Rabi model.'
                                if model == 'rabi' else 'R>=4 defines the product-side region.'),
                limitations=['Finite-Fock TDSE; displayed q grid does not enlarge the propagation basis.',
                    'Paper gives q/R grids for QD and two Fock states for POL; exact author QD truncation is not certified.',
                    'Eq.27 bare-state preparation chosen; not a claim of reproducing the q-dependent Eq.25 preparation.',
                    'Periodic FFT endpoints and second-order splitting are explicit implementation choices.'])


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
    elif selection in ('off', 'paper-off'):
        c = replace(c, omega=PAPER_FREQUENCIES[preset]['off'])
    elif selection != 'configured':
        raise ValueError(f'Unknown frequency selection: {selection}')
    c.grid()
    return c, frequency_report(c, selection)
