"""Local-only adapter: reuse audited export without changing historical presets.

All candidates use F120, dt=.125 au. Tuple units: a0, count, a0, count,
count, atomic time. Packet Psi shape is (NR,Nx,120). Physics and validation
come unchanged from phase6_resume.build and phase6_gpu_export.export.
"""
from pathlib import Path
from . import phase6_resume as resume
from .phase6_gpu_export import export


CANDIDATES = {
    'xbox': (28.8, 192, 4.4, 352, 120, .125),
    'xfine': (24., 192, 4.4, 352, 120, .125),
    'Rbox': (24., 160, 4.6, 368, 120, .125),
    'Rfine': (24., 160, 4.4, 440, 120, .125),
}


def main():
    root = resume.ROOT / 'results/vsc_polariton/phase6_gpu/spatial_transfer'
    # Runtime-only adapter; original source and original cached results read-only.
    previous_out = resume.OUT
    try:
        resume.OUT = root / 'local_build'
        for name, setting in CANDIDATES.items():
            key = 'spatial_F120_dt0125_' + name
            if key in resume.SETTINGS:
                raise ValueError('Preset collision: ' + key)
            resume.SETTINGS[key] = setting
            try:
                export(root / 'inputs' / (name + '.npz'), 'resonant', key)
            finally:
                del resume.SETTINGS[key]
    finally:
        resume.OUT = previous_out


if __name__ == '__main__':
    main()
