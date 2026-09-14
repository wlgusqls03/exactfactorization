"""CPU saved-frame benchmark, including parity against a chosen git revision.

Run from the repository root: python benchmarks/benchmark_direct_geometry.py
Each implementation runs in a separate process for comparable peak RSS.
Synthetic coherent input avoids needing GPU hardware or a large BO cache.
This measures diagnostics only, not propagation, BO reconstruction, or disk IO.
"""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='955406e')
    parser.add_argument('--shape', nargs=3, type=int, default=(359, 1250, 32))
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--block', type=int, default=8)
    parser.add_argument('--worker', choices=('baseline', 'current'))
    args = parser.parse_args()
    if not args.worker:
        for implementation in ('baseline', 'current'):
            subprocess.run([sys.executable, __file__, *sys.argv[1:],
                            '--worker', implementation], check=True, cwd=ROOT)
        return
    from multi_component_exact_factorization.direct_geometry import direct_geometry_frame
    source = subprocess.check_output([
        'git', 'show', f'{args.baseline}:multi_component_exact_factorization/direct_geometry.py'
    ], cwd=ROOT, text=True)
    namespace = {}
    exec(compile(source, '<baseline direct_geometry>', 'exec'), namespace)
    old = namespace['direct_geometry_frame']
    # Verify all four fields on a small random complex state first.
    rng = np.random.default_rng(72)
    small = rng.normal(size=(4, 13, 17))+1j*rng.normal(size=(4, 13, 17))
    inputs = (lambda ids: small[:, :, ids], small.shape, .3, .2, .4, 2., 3., args.block)
    reference = old(*inputs)
    current = direct_geometry_frame(*inputs)
    error = max(float(np.max(abs(reference[k]-current[k]))) for k in reference)
    for key in reference:
        np.testing.assert_allclose(current[key], reference[key], rtol=2e-14, atol=1e-14)
    nx, nq, nr = args.shape
    # Nonuniform, nonseparable coherent state, generated deterministically.
    x = np.arange(nx)[:, None, None]/nx
    q = np.arange(nq)[None, :, None]/nq
    r = np.arange(nr)[None, None, :]/nr
    psi = (1+.2*np.cos(2*np.pi*(x+q+r)))*np.exp(2j*np.pi*(x*q+q*r))
    function = old if args.worker == 'baseline' else direct_geometry_frame
    inputs = (lambda ids: psi[:, :, ids], psi.shape, .145, .032, .02, 1836., 12000., args.block)
    function(*inputs)  # warmup
    elapsed = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        function(*inputs)
        elapsed.append(time.perf_counter()-started)
    print(json.dumps(dict(implementation=args.worker, shape=args.shape, block=args.block,
                         seconds=elapsed, median_seconds=float(np.median(elapsed)),
                         peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                         parity_max_abs=error)), flush=True)


if __name__ == '__main__':
    main()
