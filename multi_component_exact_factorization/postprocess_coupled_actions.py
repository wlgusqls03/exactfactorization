"""CPU blockwise six-action analysis of a saved coherent BO trajectory."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .born_huang import _load_cached_basis, BornHuangBasis
from .core import build_model
from .coupled_actions import CoupledActionRecorder
from .render_all import find_archive, resolve_run_input
from multi_component_exact_factorization_discrete.core import discrete_tdse_action
from multi_component_exact_factorization_discrete_gpu.compare_tdse import _stream_arrays


def run(args):
    archive, directory = find_archive(resolve_run_input(args.run))
    with np.load(archive, allow_pickle=True) as data:
        options = dict(data['args'].reshape(-1)[0])
        times = data['times_fs']
        nstates = int(data['bo_states_count'])
        spectral = options.get('tdse_propagator') == 'spectral_split'
        if spectral and not args.allow_bo_projection:
            raise ValueError('Full-grid Psi history is not stored. For exact diagnostics use '
                             '--save-coupled-actions during propagation. Explicit --allow-bo-projection '
                             'permits an APPROXIMATE projected-state diagnostic only.')
        cache = Path(args.basis_dir or str(data.get('bo_basis_cache_path', '')))
        key = str(data.get('bo_basis_cache_key', ''))
        grids = {k: data[k] for k in ('x','q','R')}
        has_action = 'tdse_action_coefficients' in data.files
    if not key or not (cache/'metadata.json').is_file():
        raise FileNotFoundError('Matching original BO basis cache required; supply --basis-dir. '
                                'Do not substitute unaligned eigenvectors from a different cache.')
    if spectral and not has_action:
        raise ValueError('Projected spectral analysis needs saved native action coefficients')
    basis = _load_cached_basis(cache, key)
    if basis.states.shape[0] < nstates:
        raise ValueError('Insufficient cached states')
    arrays = {}
    for name, val in vars(basis).items():
        arrays[name] = (None if val is None else val[:nstates] if name in ('states','energies')
                        else val[:nstates,:nstates])
    basis = BornHuangBasis(**arrays)
    model = build_model(SimpleNamespace(**options))
    for k, grid in grids.items():
        if not np.allclose(getattr(model,k),grid,rtol=0,atol=1e-12):
            raise ValueError('Model/archive grid mismatch: '+k)
    if basis.states.shape[1:] != (len(model.x),len(model.q),len(model.R)):
        raise ValueError('Basis/archive shape mismatch')
    ids = np.arange(len(times)) if args.max_frames == 0 else np.unique(
        np.rint(np.linspace(0,len(times)-1,min(args.max_frames,len(times)))).astype(int))
    output = Path(args.outdir) if args.outdir else directory/'coupled_action_analysis'
    output.mkdir(parents=True,exist_ok=True)
    recorder = CoupledActionRecorder(output,len(ids),model,block_R=args.R_block,
                map_stride=args.map_stride, electronic_method='spectral' if spectral else 'finite',
                source='APPROXIMATE_BO_PROJECTION_OF_FULL_GRID' if spectral else 'coherent_BO_propagated_state')
    keys = ['tdse_coefficients']+(['tdse_action_coefficients'] if has_action else [])
    with _stream_arrays(archive,keys) as readers:
        for f in ids:
            y = readers['tdse_coefficients'].read(int(f))
            hy = (readers['tdse_action_coefficients'].read(int(f)) if has_action
                  else discrete_tdse_action(y,model,basis))
            def psi(indices):
                return np.einsum('nqR,nxqR->xqR', y[:,:,indices],basis.states[:,:,:,indices],optimize=True)
            def action(indices):
                return np.einsum('nqR,nxqR->xqR', hy[:,:,indices],basis.states[:,:,:,indices],optimize=True)
            recorder.save(float(times[f]),psi,action)
            print(f'Coupled actions: frame {f+1}/{len(times)}; t={times[f]:.5f} fs',flush=True)
    recorder.finish()
    return recorder.final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--basis-dir',default='')
    p.add_argument('--outdir',default='')
    p.add_argument('--max-frames',type=int,default=8,help='0: every saved frame; default: 8-point preview')
    p.add_argument('--R-block',type=int,default=4)
    p.add_argument('--map-stride',type=int,default=2)
    p.add_argument('--allow-bo-projection',action='store_true')
    args=p.parse_args()
    if args.max_frames < 0 or min(args.R_block,args.map_stride)<1:
        p.error('Invalid frame count/block/stride')
    run(args)


if __name__ == '__main__':
    main()
