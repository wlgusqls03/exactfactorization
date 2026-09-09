"""Explicit external harmonic confinement and archive energy conventions.

BO basis caches intentionally retain full energies for backward-compatible
eigenstates, initial Gaussian widths and cache keys. Solvers split these into
internal BO energy and an explicit external term; total H is unchanged.
"""
import numpy as np

EXTERNAL = 'external_harmonic_v1'
INCLUDED = 'harmonic_included'


def harmonic_center(options):
    # The erf model is implemented with Rc=L/2 in core.build_model.
    model = str(options.get('interaction_model', '')).replace('-', '_')
    if model == 'erf_shin_metiu':
        return .5*float(options.get('fixed_ion_separation', 19.0))
    return float(options.get('heavy_trap_center',
                             .5*float(options.get('fixed_ion_separation', 19.0))))


def harmonic_potential(R, options):
    alpha = float(options.get('heavy_trap_alpha', 0.0))
    if alpha == 0:
        return np.zeros_like(np.asarray(R), dtype=float)
    return alpha*(np.asarray(R)-harmonic_center(options))**2


def model_external(model):
    """CPU models carry coordinates; GPU models carry the small R array."""
    if hasattr(model, 'external_harmonic'):
        return model.external_harmonic
    if not hasattr(model, 'R'):
        return 0.0
    return harmonic_potential(model.R, vars(model))


def internal_bo_energies(basis, model):
    """CPU BO caches are historical full energies; GPU uploads are internal."""
    if getattr(basis, 'energy_convention', INCLUDED) == EXTERNAL:
        return basis.energies
    return basis.energies-model_external(model)


def separate_fields(fields, obs, ground_density=None):
    """Convert an owned plotting dictionary exactly once, preserving sums.

    ground_density(frame) yields the physical BO ground-channel joint density.
    No GD, geometry, vector potential, density or original archive is modified.
    """
    if fields.get('energy_convention') == EXTERNAL:
        return fields
    if fields.get('energy_convention', INCLUDED) != INCLUDED:
        raise ValueError('Unknown scalar energy convention; refusing to subtract the trap')
    V = harmonic_potential(obs['R'], obs['options'])
    if np.any(V):
        for key in ('epsilon_1','epsilon_1_gi','epsilon_1_wbo',
                    'epsilon_2','epsilon_2_gi','tdpes1_total','tdpes2_total'):
            if key in fields:
                fields[key] -= V
        component_keys = [k for k in ('tdpes1_wbo_0','tdpes1_wbo_excited',
                                      'tdpes2_wbo_0','tdpes2_wbo_excited') if k in fields]
        if component_keys:
            if ground_density is None:
                raise ValueError('BO ground-channel density required to separate weighted BO energies')
            for i, rho in enumerate(obs['joint_density']):
                ground = ground_density(i)
                conditional_rho = rho
                if isinstance(ground, tuple):
                    ground, conditional_rho = ground
                p0 = np.divide(ground, conditional_rho, out=np.zeros_like(rho), where=conditional_rho>0)
                p0 = np.clip(p0, 0, 1)
                # Use precisely the same conditional normalization as the EF decomposition.
                marginal = conditional_rho.sum(axis=0)
                p0_R = np.divide((conditional_rho*p0).sum(axis=0), marginal,
                                 out=np.zeros_like(marginal), where=marginal>0)
                weights = {'tdpes1_wbo_0': p0, 'tdpes1_wbo_excited': 1-p0,
                           'tdpes2_wbo_0': p0_R, 'tdpes2_wbo_excited': 1-p0_R}
                for key in component_keys:
                    fields[key][i] -= weights[key]*V
    fields['energy_convention'] = EXTERNAL
    return fields


def effective_scalar(value, fields, obs):
    """Return the full scalar for physical force calculations (same gauge)."""
    if fields.get('energy_convention') == EXTERNAL:
        return value+harmonic_potential(obs['R'], obs['options'])
    return value
