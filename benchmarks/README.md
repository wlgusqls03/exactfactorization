# Direct geometry saved-frame optimization

Run in the project environment from the repository root:

```bash
python benchmarks/benchmark_direct_geometry.py --repeat 5
```

The baseline is commit `955406e`. The benchmark compares all four fields on
a random complex input before timing. It then uses a deterministic coherent
array of shape `(359, 1250, 32)` with R block size 8, one warmup and five
measured repetitions in separate processes. It does not include GPU
propagation, BO-wavefunction reconstruction, archive IO, or movie rendering.

Observed in the development environment on 2026-09-14:

| Measurement | Baseline | Optimized |
| --- | ---: | ---: |
| Median time | 4.8765 s | 4.4095 s |
| Peak process RSS | 912.09 MiB | 610.71 MiB |
| Small random-input maximum absolute difference | 0 | 0 |

This is about 9.6% less diagnostic time and 33% less peak process memory.
Timing varied (baseline 4.81–5.21 s, optimized 3.95–4.88 s); the shared host
was also running regression tests, so these are indicative, not a controlled
full-production speed guarantee. The R extent is deliberately smaller than
production to keep the benchmark affordable. Do not extrapolate peak RSS or
end-to-end runtime directly from these numbers.

Changes preserve the central-five-point derivative, masses, normalization,
nodal NaNs, output keys, and saved-frame cadence. q derivatives now omit the
R halo (which q differentiation cannot use); derivatives and conditional
states are allocated sequentially, and block temporaries are released before
the next block. Core source length increased from 114 to 121 lines because of
explicit lifetime management and comments, not because physics was added.
