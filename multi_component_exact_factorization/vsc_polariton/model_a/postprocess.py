"""Reanalyze saved event waves without TDSE propagation; writes a NEW directory."""
import argparse
import json
from pathlib import Path
import numpy as np
from .model import Config,Propagator
from .factorization import analyze,compact,diagnostics
from .run import atomic_json,atomic_npz


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'fields').mkdir()
    ident=json.loads((a.run/'identity.json').read_text());c=Config(**ident['config']);p=Propagator(c)
    atomic_json(a.out/'identity.json',ident);reports=[]
    for path in sorted((a.run/'waves').glob('wave_*.npz')):
        with np.load(path) as z:u=z['psi'];t=float(z['time_au'])
        f=analyze(u,p);d=diagnostics(f);d.update(time_au=t);reports.append(d)
        atomic_npz(a.out/'fields'/f'fields_{round(t/c.dt):07d}.npz',**compact(f),time_au=t,time_fs=t*.024188843265857)
    atomic_json(a.out/'ef_diagnostics.json',reports)


if __name__=='__main__':main()
