"""Read-only adapter of historical real-space MCEF caches for the new renderer.

No TDSE wave or propagation coefficient is read.  Small, previously generated
proton-heavy term snapshots supply densities, site connections and sources;
only selected scalar members of the postprocessed fields NPZ are streamed.
The source diagnostics remain cached FD5/bond-to-site diagnostics, NOT a new
native H-Psi extraction or a convergence certification.  Axes are converted
from historical (q,R) to renderer (R,q); q is proton, R is heavy nucleus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from numpy.lib import format as npformat

from multi_component_exact_factorization.external_potential import harmonic_potential


def selected_member_frames(path, key, indices):
    """One forward decompression pass for one numeric C-order NPZ member.

    Discard skipped frames immediately.  Never materialize the full time
    trajectory, and stop after the last requested frame.  No pickle here.
    """
    indices = sorted(set(int(i) for i in indices))
    if not indices or indices[0] < 0:
        raise ValueError('Need nonnegative selected frame indices')
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        shape, fortran, dtype = npformat._read_array_header(stream, npformat.read_magic(stream))
        if fortran or dtype.hasobject or len(shape) < 2 or indices[-1] >= shape[0]:
            raise ValueError('Expected numeric C-order time array and valid indices')
        count = int(np.prod(shape[1:])) * dtype.itemsize
        wanted = set(indices)
        for i in range(indices[-1] + 1):
            raw = stream.read(count)
            if len(raw) != count:
                raise ValueError('Truncated NPZ member: ' + key)
            if i in wanted:
                yield i, np.frombuffer(raw, dtype).reshape(shape[1:]).copy()


def _load_small(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def adapt_frame(source, scalars, rho_R, options):
    """Preserve cached sources; add the declared external trap exactly once.

    ``scalars`` are already sampled on source q/R, in the historical
    external_harmonic_v1 convention.  GI receives the same trap as total;
    GD=total-GI is consequently unchanged.  No interpolation or renormalizing
    the decimated density, and no finite differences of decimated data.
    """
    q, R = np.asarray(source['q']), np.asarray(source['R'])
    shape = (len(q), len(R))
    for key in ('joint_density', 'lambda_density', 'a', 'b'):
        if np.shape(source[key]) != shape:
            raise ValueError('Historical map shape mismatch: ' + key)
    alpha = np.asarray(source['alpha'])
    if alpha.shape == shape:
        np.testing.assert_allclose(alpha, np.broadcast_to(alpha[0], shape), equal_nan=True)
        alpha = alpha[0]
    if alpha.shape != (len(R),) or np.shape(rho_R) != (len(R),):
        raise ValueError('Outer line shape mismatch')
    trap = harmonic_potential(R, options)
    result = dict(system=np.array('mcef'), q=q, R=R, t_fs=float(source['time_fs']),
                  mass_q=float(options['proton_mass']), mass_R=float(options['heavy_mass']),
                  mass=float(options['heavy_mass']), gauge=np.array('positive_density_marginals'),
                  rho_R=np.asarray(rho_R), alpha=alpha,
                  source_units=np.array('conditional proton density / atomic time (a0^-1 t_au^-1)'),
                  scalar_kind=np.array('effective TDPES: cached scalar + external heavy harmonic'),
                  energy_convention=np.array('harmonic_included'), external_harmonic=trap,
                  diagnostic_scheme=np.array('Historical cached FD5/bond-to-site; sources reused unchanged; NOT native H-Psi closure'),
                  display_stride=int(source.get('map_stride', 1)))
    aliases = {'joint_density': 'rho_qR', 'delta': 'delta_b',
               'residual_density': 'continuity_residual'}
    for key in ('joint_density', 'lambda_density', 'a', 'b', 'delta',
                'density_dt', 'S_q', 'S_adv', 'S_rel', 'residual_density'):
        if key in source:
            result[aliases.get(key, key)] = np.asarray(source[key]).T.copy()
    if 'delta_b' not in result:
        result['delta_b'] = result['b'] - alpha[:, None]
    for level in (1, 2):
        key = 'epsilon_' + str(level)
        expected = shape if level == 1 else (len(R),)
        if np.shape(scalars[key]) != expected:
            raise ValueError('Scalar shape mismatch: ' + key)
        total = np.asarray(scalars[key]).real + (trap[None, :] if level == 1 else trap)
        result['epsilon' + str(level)] = total.T if level == 1 else total
        if key + '_gi' in scalars:
            gi = np.asarray(scalars[key + '_gi']).real + (trap[None, :] if level == 1 else trap)
            result['epsilon' + str(level) + '_GI'] = gi.T if level == 1 else gi
            result['epsilon' + str(level) + '_GD'] = result['epsilon' + str(level)] - result['epsilon' + str(level) + '_GI']
    return result


def adapt_bo_surfaces(bo, R, options, convention, stride=1):
    """Return ordinary BO[R,q,state], with heavy trap included exactly once."""
    if convention not in ('harmonic_included', 'external_harmonic_v1'):
        raise ValueError('Unknown BO energy convention')
    bo = np.asarray(bo)
    if bo.ndim != 3 or bo.shape[-1] != len(R):
        raise ValueError('Expected historical BO[state,q,R]')
    if convention == 'external_harmonic_v1':
        bo = bo + harmonic_potential(R, options)[None, None, :]
    return np.moveaxis(bo[:, ::stride, ::stride], 0, -1).transpose(1, 0, 2)


def prepare_preview(run_dir, out, times=(0., 40., 80.), max_time_mismatch=.3):
    """Build a new small cache; refuse overwrites and unmatched scalar times.

    The trusted historical run archive contains its options as a pickled
    dict.  Only that metadata, grids, small heavy-density lines and BO
    energies are accessed; raw TDSE coefficient/wave members are never read.
    """
    run_dir, out = Path(run_dir), Path(out)
    if out.exists():
        raise FileExistsError('Use a new output directory: ' + str(out))
    fields = run_dir / 'tdse_exact_factorization_fields.npz'
    archive = run_dir / 'multi_component_discrete_tdse_gpu.npz'
    paths = sorted((run_dir / 'report' / 'final_visualizations').glob('proton_heavy_terms_frame_*.npz'))
    candidates = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            candidates.append((float(z['time_fs']), path))
    if not candidates:
        raise ValueError('No previously generated proton-heavy source snapshots')
    selected = []
    for target in times:
        actual, path = min(candidates, key=lambda p: abs(p[0] - target))
        if abs(actual-target) > max_time_mismatch:
            raise ValueError(f'No cached source snapshot near {target:g} fs; nearest is {actual:g} fs')
        if path not in selected:
            selected.append(path)
    selected.sort()
    with np.load(fields, allow_pickle=False) as z:
        if str(z['gauge']) != 'positive_density_marginals':
            raise ValueError('Only historical positive-density marginal gauge is supported')
        if str(z['energy_convention']) != 'external_harmonic_v1':
            raise ValueError('Unknown scalar trap convention; refusing double addition')
        full_q, full_R, all_times = z['q'], z['R'], z['times_fs']
        keys = set(z.files)
    # Pickled options are read solely from this explicitly selected trusted
    # historical archive; no imported cache can execute renderer-side pickle.
    with np.load(archive, allow_pickle=True) as z:
        options = z['args'].reshape(-1)[0]
        np.testing.assert_array_equal(all_times, z['times_fs'])
        np.testing.assert_array_equal(full_q, z['q'])
        np.testing.assert_array_equal(full_R, z['R'])
        heavy = z['heavy_density']
        bo = z['bo_energies'] if 'bo_energies' in z else None
        bo_convention = str(z['bo_energy_convention']) if 'bo_energy_convention' in z else None
    sources, indices = {}, []
    for path in selected:
        source = _load_small(path)
        index = int(np.argmin(abs(all_times-float(source['time_fs']))))
        if abs(all_times[index]-float(source['time_fs'])) > 1e-8:
            raise ValueError('Source snapshot and scalar cache times disagree')
        stride = int(source.get('map_stride', 1))
        np.testing.assert_array_equal(source['q'], full_q[::stride])
        np.testing.assert_array_equal(source['R'], full_R[::stride])
        sources[index] = (source, path, stride)
        indices.append(index)
    scalar_frames = {i: {} for i in indices}
    for key in ('epsilon_1', 'epsilon_1_gi', 'epsilon_2', 'epsilon_2_gi'):
        if key not in keys:
            if key.endswith('_gi'):
                continue
            raise ValueError('Required scalar missing: ' + key)
        print('Streaming selected postprocessed member:', key, flush=True)
        for i, value in selected_member_frames(fields, key, indices):
            stride = sources[i][2]
            scalar_frames[i][key] = value[::stride, ::stride] if value.ndim == 2 else value[::stride]
    out.mkdir(parents=True)
    (out / 'fields').mkdir()
    records = []
    for index in sorted(indices):
        source, path, stride = sources[index]
        frame = adapt_frame(source, scalar_frames[index], heavy[index, ::stride], options)
        target = out / 'fields' / f'fields_{index:07d}.npz'
        np.savez_compressed(target, **frame)
        records.append(dict(index=index, time_fs=float(source['time_fs']), source=str(path),
                            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), output=str(target)))
    source, _, stride = sources[indices[0]]
    notes = ['No new TDSE; no raw wave read.',
             'Actual cached times are labeled; 80-fs request can select a nearby saved frame.',
             'Stored source terms are reused before plotting, without recomputing derivatives on decimated maps.',
             'GI/GD are cached scalar decomposition; GD = total - GI; external harmonic added once to total/GI only.',
             'Outer reconstruction omitted when no matching-time native bridge cache is available; no fitted closure.',
             'No R-only electronic BO/POL surface is invented for the two-nuclear-coordinate MCEF model.']
    if bo is not None and bo_convention in ('harmonic_included', 'external_harmonic_v1'):
        ref = adapt_bo_surfaces(bo, full_R, options, bo_convention, stride)
        # 'cbo' is the common renderer's legacy storage slot, NOT its physics
        # label. system/reference_kind ensure ordinary electronic BO labeling.
        np.savez_compressed(out / 'references.npz', R=source['R'], q=source['q'], cbo=ref,
                            system='mcef', reference_kind='BO', energy_convention='harmonic_included')
    else:
        notes.append('BO references unavailable or convention unknown; omitted.')
    manifest = dict(status='VISUAL_PREVIEW_NOT_NEW_VALIDATION', system='mcef',
                    source_fields=str(fields), source_archive=str(archive),
                    requested_times_fs=list(times), frames=records, notes=notes,
                    coordinate_meaning={'q': 'proton', 'R': 'heavy nucleus'},
                    mass_q=options['proton_mass'], mass_R=options['heavy_mass'],
                    heavy_trap_alpha=options.get('heavy_trap_alpha'),
                    bo_input_convention=bo_convention,
                    source_cache_grid_policy='Existing map_stride is retained; no renormalization')
    (out / 'preview_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--times-fs', type=float, nargs='+', default=[0., 40., 80.])
    parser.add_argument('--max-time-mismatch', type=float, default=.3)
    args = parser.parse_args(argv)
    manifest = prepare_preview(args.run_dir, args.out, args.times_fs, args.max_time_mismatch)
    print(json.dumps({'output': str(args.out), 'times_fs': [r['time_fs'] for r in manifest['frames']]}, indent=2))


if __name__ == '__main__':
    main()
