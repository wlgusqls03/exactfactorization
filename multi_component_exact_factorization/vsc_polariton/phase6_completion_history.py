"""Export compact read-only historical convergence evidence, never old outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .phase6_completion_audit import collect, diagnostic, compare


def main():
    root=Path('results/vsc_polariton')
    out=root/'phase6_gpu/completion_transfer/inputs'
    paths={}
    ortho=root/'phase6_gpu/phase6_orthogonal_results/orthogonal_campaign_v1'
    paths['resonant']=ortho/'F120_dt0125/full'
    paths['F160']=ortho/'F160_dt0125/full'
    paths['dt025']=root/'phase6_results/phase6_fock_dt_results/fock_dt_campaign_v1/F120_dt025/full'
    for name in ('xbox','xfine','Rbox','Rfine'):
        paths[name]=root/f'phase6_gpu/phase6_spatial_results/spatial_campaign_v1/{name}/full'
    arrays={k:collect(v) for k,v in paths.items()}
    runs={k:diagnostic(v) for k,v in arrays.items()}
    comparisons={k:compare(arrays['resonant'],v) for k,v in arrays.items() if k!='resonant'}
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in paths.values() for p in sorted(directory.glob('observable_*.npz'))}
    passed=all(v['pass_checks'] for v in runs.values()) and all(v['pass_checks'] for v in comparisons.values())
    with (out/'resonant_observables.npz').open('xb') as f:
        np.savez_compressed(f,**arrays['resonant'])
    with (out/'historical_validation.json').open('x') as f:
        json.dump(dict(computed_gates_pass=passed,runs=runs,comparisons=comparisons,
                       source_hashes=hashes),f,indent=2)
    if not passed:raise RuntimeError('Historical comparison failure')


if __name__=='__main__':main()
