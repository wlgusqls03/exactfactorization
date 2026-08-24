# Discrete MCEF CUDA solver

This is the production implementation of the spatially discrete,
time-continuous MCEF equations for the same extended 1D Shin--Metiu model as
the existing solver.

It reuses the immutable Born--Huang cache and fused overlap-link CUDA kernel.
Per RHS evaluation it performs one q transport and one R transport.  It does
not evaluate logarithmic derivatives and does not apply the old weighted
product-rule projection.

The only tail regularization is the probability-budget flat-top generalized
inverse `W/F` and `W/chi`.  A product-preserving horizontal tangent cleanup and
support-aware PNC retraction control floating-point/RK constraint drift.

The CUDA RHS is arranged around exactly two fused BO-neighbor transport
launches (q and R) per evaluation.  It does not upload the unused full
`V(x,q,R)` tensor to the GPU.  Expensive temporal-consistency diagnostics are
computed only on saved steps.

## Validation

First compare the complete CUDA RHS (both reference and fused overlap-link
kernels) with the NumPy discrete algebra on a small physical Shin--Metiu grid:

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_discrete_gpu.validate \
  --device 0 \
  --heavy-trap-alpha 0.05
```

This checks `dC`, `dLambda`, `dChi`, both discrete scalars, exact marginal
nodes, and the recombination identity.  It must finish with `PASS`.

## Short production smoke run

The following uses the same extended 1D Shin--Metiu system and BO10 cache as
the current continuum-derived calculation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_discrete_gpu.propagate \
  --device 0 \
  --bo-states 10 \
  --bo-link-kernel fused \
  --bo-basis-cache-dir results/bo_basis_cache \
  --flat-top-budget-phi 1e-10 \
  --flat-top-budget-lam 1e-10 \
  --flat-top-transition-decades 3 \
  --deep-tail-zero-threshold 1e-12 \
  --heavy-trap-alpha 0.05 \
  --proton-force-constant 0.19245621776826924 \
  --heavy-force-constant 0.19168954579929773 \
  --electron-excitation 1 \
  --nx 300 --nq 600 --nR 800 \
  --dt-au 0.025 \
  --t-final-fs 0.1 \
  --save-every 20 \
  --progress-every 50 \
  --check-every 5 \
  --step-sleep-ms 0 \
  --verbose-diagnostics \
  --no-render-after \
  --outdir results/discrete_mcef_bh10_01fs
```

The current model defaults are the electronic Dirichlet box `x in (-22,22)`,
periodic nuclear grids `q in [-12,12)` and `R in [2,18)`, one fixed charge at
`X_L=-10`, and initial centers `q0=0,R0=10`.  The old right fixed ion is
disabled (`right_charge=0`) and the moving heavy ion is bound by
`V_trap=alpha*(R-10)^2`.  Because `alpha` fixes a physical vibrational
frequency rather than a numerical tolerance, every new propagation must pass
`--heavy-trap-alpha` explicitly.  The example value `0.05 Ha/a0^2` is only a
trial value; choose it from `alpha=M_R*omega^2/2` for the intended bond.

For a thermally constrained GPU, add (for example)
`--step-sleep-ms 20`.  A positive value synchronizes the active CUDA stream
after every completed step and then sleeps for the requested duration, so the
pause is real GPU idle time rather than merely a delay in host-side kernel
submission.  It changes neither the equations nor `dt`; it only increases
wall time.  The default is zero and preserves the original asynchronous fast
path.  The archive records `step_sleep_ms`, `throttle_sleep_seconds`, and
`throttled_steps`.

## Atomic checkpoint and exact state resume

Long calculations can keep one bounded, uncompressed state checkpoint:

```bash
--checkpoint-every 5000
```

The default path is
`RUN_FOLDER/discrete_mcef_checkpoint.npz`. Each write first creates and
`fsync`s a sibling temporary file and then atomically replaces the preceding
checkpoint. Thus an OOM kill, lost login session, or full filesystem during a
new write leaves the previous completed-step checkpoint intact. The file
contains only complex128 `C`, `Lambda`, and `chi`, their completed global step,
and a strict compatibility fingerprint. It does not change the RHS, RK4, PNC
retraction, mask, time step, or floating-point precision.

Resume with the same physical/numerical options and a final time later than
the checkpoint time:

```bash
python -m multi_component_exact_factorization_discrete_gpu.propagate \
  ...same model/grid/BO/mask/dt options... \
  --t-final-fs 50 \
  --checkpoint-every 5000 \
  --resume-from results/20260814/RUN/discrete_mcef_checkpoint.npz \
  --outdir results/20260814/RUN
```

The BO cache key, BO count/kernel, grids, masses, `dt`, masks and PNC-tail
threshold are compared before a state is uploaded to CUDA. Any mismatch is
rejected instead of silently changing the calculation. Host round trips of
complex128 state arrays are bit preserving. `SIGHUP` and `SIGTERM` request a
safe stop after the active RK4 step; `SIGKILL` and OOM cannot be caught, so the
most recent periodic atomic checkpoint is the recovery point.

A resumed result archive begins at the checkpoint time and records
`segment_start_step`/`segment_start_time_fs`; the checkpoint intentionally
does not duplicate the many-GiB trajectory history. It preserves exact future
dynamics, while frames before an ungraceful crash are available only if an
earlier partial/final trajectory archive was successfully written.
Checkpoint overhead and write count are printed and stored as
`checkpoint_seconds` and `checkpoint_writes`. An interval of 5000 steps is
normally a very small wall-time cost because only one state file is replaced.

## Direct TDSE reference in the identical BO space

`propagate_tdse` evolves the unfactorized coefficient wavefunction
`Y_j(q,R)` with the same BO cache, overlap-link Hamiltonian, complex128
precision, nuclear grids, time step and classical RK4 method as the discrete
MCEF solver.  It contains no mask, PNC projection or factor retraction.  The
validation command above checks both `H_h Y` and one TDSE RK4 step against the
NumPy oracle for the reference and fused link kernels.

Run a short smoke test first:

```bash
CUDA_VISIBLE_DEVICES=1 python -m \
  multi_component_exact_factorization_discrete_gpu.propagate_tdse \
  --device 0 \
  --bo-states 10 \
  --bo-link-kernel fused \
  --bo-basis-cache-dir results/bo_basis_cache \
  --electron-excitation 1 \
  --nx 300 --nq 450 --nR 900 \
  --dt-au 0.025 \
  --t-final-fs 0.1 \
  --save-every 20 \
  --progress-every 50 \
  --check-every 5 \
  --outdir results/discrete_tdse_bh10_01fs
```

After validation, a thermally constrained 50 fs run can use:

```bash
CUDA_VISIBLE_DEVICES=1 python -m \
  multi_component_exact_factorization_discrete_gpu.propagate_tdse \
  --device 0 \
  --bo-states 10 \
  --bo-link-kernel fused \
  --bo-basis-cache-dir results/bo_basis_cache \
  --electron-excitation 1 \
  --nx 300 --nq 450 --nR 900 \
  --dt-au 0.025 \
  --t-final-fs 50.0 \
  --save-every 1000 \
  --progress-every 2000 \
  --check-every 20 \
  --step-sleep-ms 20 \
  --outdir results/discrete_tdse_bh10_50fs
```

One saved BO10 `(450,900)` complex128 coefficient frame is about 61.8 MiB.
`--save-every 1000` stores about 84 coefficient frames (roughly 5.1 GiB
before compression) over 50 fs.  Choose a TDSE save interval that is an
integer multiple of the MCEF save interval so every TDSE frame has an exact
MCEF counterpart.

Compare the trajectories without reconstructing the electronic x grid:

```bash
python -m multi_component_exact_factorization_discrete_gpu.compare_tdse \
  results/$(date +%Y%m%d)/discrete_tdse_bh10_50fs \
  results/$(date +%Y%m%d)/discrete_mcef_bh10_50fs \
  --tempdir results/tdse_compare_tmp \
  --progress-every 5
```

For multi-tens-of-GiB archives, avoid the extraction workspace with:

```bash
python -m multi_component_exact_factorization_discrete_gpu.compare_tdse \
  TDSE_RUN MCEF_RUN \
  --low-disk \
  --progress-every 5
```

`--low-disk` sequentially decompresses each NPY member inside both NPZ files
and retains only one common-time TDSE/MCEF frame pair in RAM. It needs no
large temporary directory (roughly two coefficient frames plus factors in
RAM), and computes the same fidelity, normalized density L1 errors, BO
population errors and norms as the extraction/mmap path. The two paths are
tested for bitwise-identical comparison arrays on the same archives. It can
be slower than mmap extraction because decompression is sequential, but does
not change any metric or trajectory data.

The comparison saves `tdse_mcef_comparison.npz` beside the MCEF archive and
reports coefficient-space fidelity, joint/proton/heavy density L1 errors,
BO-population differences and both norms.  Temporary extracted arrays are
removed automatically, but `--tempdir` must have enough free space for the
required uncompressed archive members.

## Standalone TDSE figures, movies and reconstructed exact fields

The TDSE archive has its own report; it is not overlaid with an MCEF run.
The standard renderer deliberately skips the very large
`tdse_coefficients` member and reads only the saved reduced densities,
populations, energy and reliability diagnostics:

```bash
python -m multi_component_exact_factorization.render_all \
  results/YYYYMMDD/expanded_tdse_bh10_50fs \
  --format mp4 \
  --fps 12 \
  --max-frames 240 \
  --dpi 180 \
  --animation-dpi 110 \
  --marginal-ymax 1.5 \
  --marginal-xmax 12 \
  --no-3d
```

This creates raw proton/heavy marginal figures, fixed-scale q-R joint-density
snapshots, BO populations and sampled BO energies, numerical reliability and
`tdse_dynamics_overview.mp4`. When an electron marginal is available it also
creates `particle_marginals_fixed_scale.mp4`, with electron, proton and heavy
density on the same `[-12,12]` position window and `0..1.5` density window.
Those are plotting windows only: values outside them are not modified.
Densities and populations are neither smoothed
nor peak-normalized; animation axes and the joint-density color maximum are
fixed over the complete trajectory.

The two nested TDPES and vector potentials are not primary TDSE variables.
Reconstruct them once from the saved full coefficient trajectory on a GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  multi_component_exact_factorization_discrete_gpu.postprocess_tdse_ef \
  results/YYYYMMDD/expanded_tdse_bh10_50fs \
  --device 0 \
  --bo-link-kernel fused \
  --link-output full \
  --overwrite \
  --progress-every 5
```

The postprocessor sequentially decompresses one `Y_j(q,R)` frame, applies the
same discrete TDSE Hamiltonian, and factorizes it in the positive-density
gauge.  The instantaneous action `dY/dt=-i*H_h*Y` supplies the temporal terms
of both scalar potentials, avoiding a finite difference across saved frames.
It saves `tdse_exact_factorization_fields.npz` beside the TDSE archive with

- the two scalar/TDPES fields `epsilon_1(q,R)` and `epsilon_2(R)`;
- the first vector potential's q/R components `a(q,R)` and `b(q,R)`;
- the second vector potential `alpha(R)`;
- native forward overlap links `S^Phi_q`, `S^Phi_R`, and `S^Gamma_R`.
  `--link-output nearest` stores the `+1` links, while `full` also stores the
  `+2` links used by the five-point stencil; backward links follow exactly by
  shifted conjugate transpose;
- exact factorization and imaginary-scalar residual audits.
- the exact electron marginal reconstructed from the BO coefficients and
  cached electronic states.

Electron-marginal reconstruction is exact but expensive because every saved
frame must stream the large cached BO eigenstate tensor. Use
`--no-electron-density` only when the exact-potential fields are needed without
the three-particle marginal movie. New TDSE propagation archives save the
electron marginal at each output frame by default; use
`--no-bo-save-electron-density` to opt out.

Run the standard renderer again.  It automatically finds this field cache
and additionally creates
`05_tdse_exact_factorization_fields.png`,
`06_tdse_transport_and_drive.png`,
`07_tdse_discrete_link_geometry.png`,
`08_tdse_joint_density_relative_log.png`,
`tdse_exact_factorization_fields.mp4`, and
`tdse_all_exact_potentials.mp4` (six outer panels containing
`epsilon_1`, `a`, `b`, overlaid `epsilon_2`/`alpha`, the q/R magnitude and
phase of `S^Phi`, and the magnitude/phase of `S^Gamma`), and
`tdse_transport_and_drive.mp4`.  It also creates the shape-only
`tdse_joint_density_relative_log.mp4` and
`particle_marginals_relative_log.mp4`; these divide each frame by its own
peak for display and retain the absolute linear-density figures unchanged.
It also creates
`heavy_coordinate_dynamics.mp4` and `proton_coordinate_dynamics.mp4`:
each uses the existing exact-potential colors and definitions, while the
three field panels follow the instantaneous occupied coordinate interval.
The marginal panel remains on the requested fixed display window.  The
proton panels label the density-conditioned R reduction of the first TDPES
and momentum, and the R-integrated proton current.  The latter fields include mechanical
momenta, currents, and gauge-invariant drives.  Spatial derivatives and the
saved-frame connection time derivative use the same periodic five-point and
centered-time diagnostic convention as the existing MCEF report; they do not
feed back into TDSE propagation.

TDSE-derived one-dimensional connection profiles are lifted only inside
connected occupied support and their integer `2*pi/h` branch is matched to
the preceding saved frame.  Empty tails therefore cannot shift the occupied
packet by one winding.  The q and R coordinate-focus current panels use a
branch-free continuity reconstruction from the saved marginal densities,
with negligible boundary flux as the integration constant.  This current is
a saved-frame diagnostic; an exact five-point bond-current audit additionally
requires both the `+1` and `+2` native links.

All continuum-MCEF, discrete-MCEF and TDSE 2D momentum/current/drive maps use
a trajectory-wide robust color range computed only on occupied support.
Outliers outside the plotting range saturate the end color and are marked by
extended colorbars; the stored field values are not clipped or rescaled.

## Saved diagnostics

Every saved frame contains:

- the actual `i*dY-H_h*Y` recombination residual;
- the analytically predicted flat-top mask residual and the unexplained
  remainder;
- the RK4 local product defect comparing the actual `Y(n+1)-Y(n)` with the
  four-stage recombined RHS quadrature;
- the PNC retraction's full-product change, raw PNC error and correction load;
- suppressed probability, transition fractions and maximum regularized
  `F'/F`, `chi'/chi` ratios;
- native link phases, link magnitudes, weighted link defects, discrete
  scalars, BO populations and all three marginals.
- separate left/right probabilities beyond the physical fixed centers and
  probabilities in the outer five numerical-grid cells.  The former detects
  passage beyond the ions; the latter detects approach to the periodic seam.

The archive keeps `a`, `b`, and `alpha` as principal-branch `arg(S)/h`
diagnostics.  Native discrete figures unwrap those phases along their bond
coordinate before applying the occupied-support display mask.  This is
plotting-only; propagation always uses the full complex overlap links.
Together with `sphi_q1_magnitude`, `sphi_R1_magnitude`, and
`sgamma_R1_magnitude`, these fields are a lossless polar representation of
the three native nearest-neighbour links: for example
`Sphi_q1 = magnitude*exp(1j*a*dq)`.  The archive metadata records this storage
convention explicitly, so a redundant multi-GiB complex copy is unnecessary.

The factor equations use no weighted product projection.  Therefore a small
`relative_unexplained_residual` directly tests the discretize-first spatial
algebra rather than whether a later projection hid its defect.

After a run, print a max/final audit (including q/R edge probability) with:

```bash
python -m multi_component_exact_factorization_discrete_gpu.diagnose \
  results/$(date +%Y%m%d)/discrete_mcef_bh10_01fs
```

The same boundary audit works for TDSE, discrete-MCEF, and continuum-MCEF
archives and reports first threshold-crossing times:

```bash
python -m multi_component_exact_factorization_discrete_gpu.diagnose_boundaries \
  results/$(date +%Y%m%d)/expanded_tdse_bh10_50fs \
  results/$(date +%Y%m%d)/expanded_discrete_mcef_bh10_50fs \
  results/$(date +%Y%m%d)/expanded_continuum_mcef_bh10_50fs
```

All three Born--Huang propagation drivers handle a single `Ctrl+C`
gracefully.  They finish or discard the in-flight RK4 step, append the last
fully completed finite step when it was not already saved, write a partial
NPZ archive, and mark `propagation_status.log` as `interrupted`.  Wait for the
`부분 저장 완료` message after pressing `Ctrl+C`; pressing it again can abort
the archive write itself.

Audit conditional-factor norms separately in the PNC-off tail, transition,
and occupied support without loading the large trajectory into RAM:

```bash
python -m multi_component_exact_factorization_discrete_gpu.diagnose_factor_norms \
  results/$(date +%Y%m%d)/expanded_discrete_mcef_bh10_50fs \
  --tempdir results/factor_norm_tmp
```

The same command accepts a continuum Born--Huang MCEF run.  Existing archives
provide post-retraction factor norms; runs produced with the newer verbose
diagnostics additionally report exact pre-retraction min/max values and the
number/fraction below `1e-4,1e-2,1e-1` or above `10,100,1e4`.

## Figures and movies

```bash
python -m multi_component_exact_factorization.render_all \
  results/$(date +%Y%m%d)/discrete_mcef_bh10_01fs \
  --fast
```

The existing Born--Huang report still produces its standard density,
electron/proton/heavy marginal, momentum/current/force, exact-potential and BO
surface products.  In addition to its original three movies it creates a
fixed-scale three-particle marginal movie and heavy/proton coordinate movies;
the latter reuse the exact-potential fields and move only their x display
window with occupied support.  A discrete archive adds:

```text
06_discrete_mcef_consistency.png
07_discrete_link_geometry.png
discrete_mcef_native_geometry.mp4
```

These show spatial and temporal residuals, PNC/norm, `E^(1)`, `E^(2)`, the
phase and magnitude of `S^Phi,S^Gamma`, joint density and BO transfer.  Use
`--no-animation` for PNGs only, or omit `--fast` for full-resolution movies.

Run `python -m multi_component_exact_factorization_discrete_gpu.propagate
--help` for all model and grid options.
