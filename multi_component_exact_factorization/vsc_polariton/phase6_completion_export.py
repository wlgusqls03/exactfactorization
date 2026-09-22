"""Local-only portable inputs for the remaining core campaign; no old writes."""
import json
from pathlib import Path
import numpy as np
from . import phase6_resume as resume
from .phase6_gpu_export import export
from .phase6_orthogonal_packets import convert


def main():
    root = resume.ROOT/'results/vsc_polariton/phase6_gpu/completion_transfer'
    previous = resume.OUT
    try:
        resume.OUT = root/'local_build'
        for case in ('free', 'barrier'):
            for nf in (120, 160):
                key = 'completion_' + case + str(nf)
                if key in resume.SETTINGS:
                    raise RuntimeError('Preset collision')
                resume.SETTINGS[key] = (24., 160, 4.4, 352, nf, .125)
                try:
                    raw = root/'local_build'/f'{case}_F{nf}.npz'
                    if not raw.exists():
                        export(raw, case, key)
                    convert(raw, root/'inputs'/f'{case}_F{nf}.npz')
                finally:
                    del resume.SETTINGS[key]
        # Reduced controls use historical direct-FGH molecular arrays, NOT a fit.
        source = resume.ROOT/'results/vsc_polariton/phase3/molecular_N352_dx0.02500000.npz'
        with np.load(source) as a:
            print('Molecular keys:', a.files, flush=True)
        import shutil
        shutil.copy2(source, root/'inputs/molecular.npz')
        # Include already validated resonant input for 2D controls only; no rerun.
        for nf in (120, 160):
            src = resume.ROOT/f'results/vsc_polariton/phase6_gpu/orthogonal_transfer/inputs/resonant_F{nf}.npz'
            for path in (src, src.with_suffix('.json')):
                shutil.copy2(path, root/'inputs'/path.name)
    finally:
        resume.OUT = previous


if __name__ == '__main__':
    main()
