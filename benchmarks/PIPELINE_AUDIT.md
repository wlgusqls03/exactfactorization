# Pipeline optimization audit (2026-09-14)

Baseline: `fe6778e`. This pass preserves the physical model, propagation
algorithms, precision, output keys, saved-frame cadence, plot styles, masks,
and contour levels. It is not a new validation of a 100 fs physical run.

## Scope and limits

All 103 tracked Python files at the start of this pass were parsed/compiled
and structurally inventoried. Cross-repository searches covered large array
allocation, contractions, NPZ reads/writes, CPU/GPU transfers, synchronization,
eigensolvers and periodic derivatives. Detailed manual review focused on
the active pipeline and its shared routines, rather than claiming a
line-by-line mathematical proof of every legacy solver.

| Area | Inspection / outcome |
| --- | --- |
| `exact_factorization` | Legacy single-component propagation/report entry points; retained unchanged. |
| CPU multi-component core | Initialization, potential and finite differences retained; no stencil or initial-state changes. |
| BO basis/cache | Cache/superset reuse and mmap already present; remove unnecessary real conjugates and repeated complex conjugates. |
| Discrete / continuous GPU | Fused derivatives and link workspaces already present; do not alter numerical kernels without GPU benchmarks. |
| BO TDSE save | Reuse the two CPU marginal arrays for history and boundary diagnostics. |
| Spectral TDSE | Existing bounded transforms, fused action/energy, half-kick merging retained. |
| EF postprocessing | Existing frame streaming and BO blocks retained; improve shared electronic-density reconstruction. |
| Archive comparison | Avoid repeat decompression of each density/population; ensure files close on errors too. |
| Final visualization | Existing artist reuse/selective EF loading retained; reduce contour preparation allocations. |
| Older reports/render_all | Existing shared decomposition/materialized-archive paths reviewed; do not remove legacy products. |

GPU numerical execution is not available in this environment. The GPU-file
edit only reuses already-copied CPU arrays; boundary diagnostics read these
without modifying them. CUDA speedup is not measured or asserted.

## Changes

1. `tdse_electron.py`: square the magnitude buffer in place, release coherent
   Psi before reducing probabilities, release each block before reading the
   next, and reuse the R integral when both electronic densities are requested.
   Coherent BO summation (including cross terms) and all four option
   combinations remain supported. Joint-only and marginal-only paths remain.
2. `born_huang.py`: use a real basis directly as its bra; for a complex basis
   conjugate once per derivative projection, not once per right BO state.
   Real forward-overlap blocks also no longer need a conjugate copy.
3. `compare_observables.py`: density/population members read once rather than
   for both shape checking and arithmetic; legacy no-grid spacing fallback
   reuses the already-read density. New context managers close both archives.
4. `propagate_tdse.py`: q/R marginal downloads reduced from four to two per
   saved frame. At 415 frames this eliminates 830 small transfers, not all
   GPU synchronization or all transfers. No millisecond saving is assumed.
5. `render_final_visualizations.py`: contour major-level classification is
   vectorized and the crop uses occupied rows/columns rather than a coordinate
   list for every occupied cell. Remove an unreachable relative-density branch.
   No interpolation, smoothing, contour spacing or display limit changes.

## Reproducible CPU microbenchmarks

Run in the MCEF Python environment, repository root:

```bash
python benchmarks/benchmark_pipeline.py electronic
python benchmarks/benchmark_pipeline.py basis
python benchmarks/benchmark_pipeline.py compare
python benchmarks/benchmark_pipeline.py contours --repeat 30
```

Each implementation uses a separate process, one warmup and five measured
calls (30 for contours). Timings exclude synthetic-input construction and
imports. Electronic/BO shape is `(2,359,1250,32)` with real BO states and complex
coefficients; this is a reduced R extent, not a full production calculation.
The electronic benchmark requests both reduced densities by default. Use
`--electronic-output joint` or `marginal` to time individual output paths.
Comparison uses compressed synthetic 415-frame marginals (359/1250/450 sites).
Contours measure only argument preparation, not Matplotlib drawing/encoding.

| Measured operation | Baseline median | New median | Time reduction |
| --- | ---: | ---: | ---: |
| Both electronic densities | 0.34695 s | 0.30181 s | 13.0% |
| One BO derivative projection | 0.63692 s | 0.52924 s | 16.9% |
| Compressed observable comparison | 0.11500 s | 0.06580 s | 42.8% |
| Contour preparation | 2.447 ms | 0.810 ms | 66.9% |

Peak process RSS for the electronic benchmark decreased from 588.98 to
534.12 MiB (9.3%). BO projection peak RSS was unchanged (598.43 MiB); another
temporary still determines peak memory. Comparison peak RSS increased from
52.08 to 55.90 MiB because both decoded densities are retained during
comparison instead of decoded repeatedly. These are process measurements,
not standalone tensor sizes. Timing on a shared host varies; end-to-end
speedup cannot be inferred by adding these percentages.

## Regression coverage

Added checks cover real and complex BO bases, first/second derivatives along
both axes, electronic reconstruction for all output flag combinations and
block sizes (including uneven final blocks), unchanged inputs, archive member
read counts, legacy spacing, mismatched density shapes, and exact contour
coordinates/levels/colors/linewidths against the previous crop algorithm.
The existing complete unittest suite is also run, including rendering smoke
tests: **165 tests passed in 148.54 s** for this revision. Roundoff-level
regrouping of the combined electronic marginal sum is
tested at `rtol=2e-15`; bitwise identity is not claimed for that regrouping.
An additional before/after Agg rendering of the same 1250-by-450 density
contours on a signed-color background produced identical RGBA pixels
(maximum difference 0). This is a sample visual check, not every movie frame.

## Candidates deliberately not changed

- Moving GPU work across synchronization boundaries, reusing RK4 stage buffers,
  or fusing more kernels needs real CUDA timing and numerical comparisons.
- Sharing BO reconstruction between direct geometry and electronic marginals
  could save more work, but their halo/core blocks and normalization contracts
  differ. This needs a separate API change and dedicated streaming tests.
- Final rendering is still dominated by drawing many contour paths and 3D
  surfaces plus encoding. Reducing frames, levels, or resolution would change
  the requested output; those shortcuts are not applied.
- Splitting the large renderer improves maintainability but does not itself
  improve runtime; a broad module move would obscure this performance patch.
- Cached reduced frames and loaded field arrays trade memory for repeated
  decompression. Changing that policy needs a separate memory-budget design.
