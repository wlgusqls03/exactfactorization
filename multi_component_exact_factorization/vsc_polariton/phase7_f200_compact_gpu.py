"""Storage-only adapter around unchanged Phase6 GPU propagation.

All observables and every restart are retained. Scientific waves: every64au,
the union of existing event times160/608/672/1408/1440au, and final1652au.
Failed/interrupted off-cadence diagnostic waves are also retained. No removal,
compression approximation, precision change or propagator modification.
"""
from pathlib import Path
from . import run_phase6_gpu as engine

EVENT_STEPS={1280,4864,5376,11264,11520}
FINAL_STEP=13216


def retain_wave(step):
    return step%512==0 or step in EVENT_STEPS or step==FINAL_STEP


def main():
    original=engine.save_npz
    def save(path,**data):
        path=Path(path)
        if path.name.startswith('wave_'):
            step=int(path.stem.split('_')[1])
            # Engine asks for off-cadence waves only on failures/final.
            if step%256==0 and not retain_wave(step):
                # Failed diagnostics at regular cadence must still be saved.
                obs=path.parent/f'observable_{step:07d}.npz'
                if obs.exists():
                    import numpy as np
                    with np.load(obs) as z:row={k:z[k] for k in z.files}
                    first=path.parent/'observable_0000000.npz'
                    with np.load(first) as z:e0=float(z['energy'])
                    if not engine.absolute_failures(row,e0):return
        original(path,**data)
    engine.save_npz=save
    try:return engine.main()
    finally:engine.save_npz=original


if __name__=='__main__':
    import sys
    sys.exit(main())
