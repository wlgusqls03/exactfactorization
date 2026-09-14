"""Isolated baseline/current CPU benchmarks for pipeline hot paths.

Examples (repository root, project Python environment):
  python benchmarks/benchmark_pipeline.py electronic
  python benchmarks/benchmark_pipeline.py basis
  python benchmarks/benchmark_pipeline.py compare
  python benchmarks/benchmark_pipeline.py contours
Synthetic data only; no GPU, user archives, or dynamics are modified.
"""
import argparse
import importlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('electronic', 'basis', 'compare', 'contours'))
    parser.add_argument('--baseline', default='fe6778e')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--electronic-output', choices=('both', 'marginal', 'joint'), default='both')
    parser.add_argument('--worker', choices=('baseline', 'current'))
    args = parser.parse_args()
    if not args.worker:
        for implementation in ('baseline', 'current'):
            subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker', implementation],
                           cwd=ROOT, check=True)
        return
    names = {'electronic': ('tdse_electron', 'electronic_reduced_densities_from_bo'),
             'basis': ('born_huang', '_project_basis_derivative'),
             'compare': ('compare_observables', 'compare'),
             'contours': ('render_final_visualizations', '_joint_linear_contours')}
    module_name, function_name = names[args.task]
    package = 'multi_component_exact_factorization'
    current = getattr(importlib.import_module(f'{package}.{module_name}'), function_name)
    source = subprocess.check_output(['git', 'show',
        f'{args.baseline}:{package}/{module_name}.py'], cwd=ROOT, text=True)
    namespace = {'__package__': package, '__name__': f'{package}._benchmark_baseline'}
    exec(compile(source, '<baseline>', 'exec'), namespace)
    function = namespace[function_name] if args.worker == 'baseline' else current
    kwargs = (dict(electron_marginal=args.electronic_output != 'joint',
                   electron_proton=args.electronic_output != 'marginal')
              if args.task == 'electronic' else {})
    rng = np.random.default_rng(19)
    with TemporaryDirectory(prefix='mcef-pipeline-bench-') as folder:
        if args.task == 'contours':
            class Capture:
                def contour(self, *values, **style):
                    return values, style
            q, r = np.linspace(-12, 28, 1250), np.linspace(5, 14, 450)
            density = .2*np.exp(-((q[:, None]-4)/4)**2-((r[None, :]-9.5)/.5)**2)
            inputs = (Capture(), dict(q=q, R=r), density)
            shape = [1250, 450]  # contour preparation only, no drawing
        elif args.task == 'compare':
            payload = {'times_fs': np.arange(415, dtype=float),
                       'state_populations': rng.random((415, 2))}
            for grid, key, n in [('x', 'electron_density', 359),
                                  ('q', 'proton_density', 1250), ('R', 'heavy_density', 450)]:
                payload[grid] = np.arange(n)*.02
                payload[key] = rng.random((415, n))
            paths = [Path(folder)/name for name in ('reference.npz', 'candidate.npz')]
            for path in paths:
                np.savez_compressed(path, **payload)
            inputs = tuple(paths)
            shape = [415, 359, 1250, 450]
        else:
            shape = [2, 359, 1250, 32]
            states = rng.normal(size=shape)
            if args.task == 'basis':
                inputs = (states, .032, 2, 1, .145)
            else:
                y = rng.normal(size=(2, 1250, 32))+1j*rng.normal(size=(2, 1250, 32))
                inputs = (y, states, .032, .02, 24)
        # Warmup, followed by timed calls. Outputs are discarded immediately.
        function(*inputs, **kwargs)
        elapsed = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            function(*inputs, **kwargs)
            elapsed.append(time.perf_counter()-start)
        print(json.dumps(dict(task=args.task, implementation=args.worker, shape=shape,
                              electronic_output=args.electronic_output if args.task == 'electronic' else None,
                              seconds=elapsed, median_seconds=float(np.median(elapsed)),
                              peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)),
              flush=True)


if __name__ == '__main__':
    main()
