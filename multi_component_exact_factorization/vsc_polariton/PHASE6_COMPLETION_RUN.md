# Remaining Phase6: one serial GPU batch (no Phase7)

Predeclared before server execution. Existing orthogonal resonant/spatial
results are reused, not rerun or overwritten. Physics, thresholds and the
historical GPU backend/driver are unchanged. All quantities below are au.

## Jobs and disk

| Full 3D run | Fock | dt | purpose |
|---|---:|---:|---|
| free_F120 |120|0.125|free production|
| free_F120_dt00625 |120|0.0625|meaningful free timestep refinement|
| barrier_F120 |120|0.125|161.76892211069242 meV production|
| barrier_F160 |160|0.125|barrier Fock refinement|

All: x=+-24, Nx160, R=+-4.4, NR352, target1652 au=39.959969 fs.
Free Fock refinement alone would be trivial (n=0 invariant), hence the
free second run refines time instead. Each run first performs its own
128-step CPU/GPU comparison. No automatic Phase7 import or execution.

Observables every4 au; wave snapshots every32 au plus initial/final/failure;
lossless restart every observable frame. Four runs' waves+restart occupy
23.57 GiB. Reserve another3 GiB for compact controls, diagnostics and atomic
temporary checkpoint writes. Bundle/extracted inputs approximately0.5 GiB
each; 35 decimal GB free is sufficient before transfer if no other job fills
the disk. The batch checks available space before starting; it deletes nothing.

Existing measured server hardware checks imply about4.55 h of propagation
and0.8–0.9 h of CPU/GPU checks for these four jobs. Budget **6–9 h total**
including reduced controls, regression, stationary test and I/O; not measured
end-to-end. Failed convergence or server load can require additional work.

## Additional diagnostics in the same batch

- Exact eta=0, n=0 stationary sector, FULL production spatial box/grid;
  iterative eigen-residual and128-step density/norm/energy check. This is
  not a claim of a converged coupled-cavity ground state.
- Nine reduced diagnostic trajectories: free/resonant/barrier x A/B/C.
  Each has F120 dt.125, F120 dt.0625 and F160 dt.125: **27 small 2D runs**.
- A: historical scalar DSE; B: + electronic dipole variance; C: + DBOC.
  Same matched nuclear/photon initial state. No arbitrary momentum adjustment.
- DBOC uses the historical central-five derivative, with periodic extension
  confined to boundary stencils; edge probability remains gated. A global
  FFT derivative of a nonperiodic aligned BO gauge creates spurious bulk
  contributions and is deliberately not used for this local correction.
  The propagated kinetic operator remains the unchanged FGH/FFT operator.
- All scalar/marginal data, continuity and integrated continuity; reduced
  time/Fock convergence; full3D-minus-A/B/C observable comparisons.
- MCEF regression and portable GPU/VSC tests before/after; source hash check.
  Full local VSC tests are distinct from the portable server suite (the server
  need not contain local-only historical exporter modules).

## Server command

Copy `phase6_completion_server_bundle.tar.gz` to the repository root.
The archive includes new portable source files and inputs, so git pull alone
is not needed to acquire the inputs. The installer verifies checksums and
refuses to overwrite differing existing code; identical installed code is OK.

```bash
cd /home/hbji/exactfactorization
mkdir -p results/vsc_polariton/phase6_gpu/completion_transfer
tar -xzf phase6_completion_server_bundle.tar.gz \
  -C results/vsc_polariton/phase6_gpu/completion_transfer
python results/vsc_polariton/phase6_gpu/completion_transfer/install_phase6_completion.py
tmux new -s phase6_finish
conda activate MCEF
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
python -u -m multi_component_exact_factorization.vsc_polariton.run_phase6_completion
```

Already in tmux: omit `tmux new`. Logs are saved automatically; no external
tee directory is needed. Detach with Ctrl-b,d. Graceful interruption keeps
full3D restarts; same command skips completed runs and resumes interrupted
full3D runs. An unfinished small control restarts that control, not the four
large runs. Hard kills can leave locks: inspect processes before manual
lock recovery. A FAILED_DIAGNOSTIC is preserved and never silently retried.

## Return only compact results

```bash
tar -czf phase6_completion_results.tar.gz \
  --exclude='wave_*.npz' --exclude='restart.npz' --exclude='*.pending' \
  -C results/vsc_polariton/phase6_gpu completion_campaign_v1
```

Send this archive even if a job stops. Keep full scientific wave snapshots
on the server for future Phase7. Do NOT delete existing resonant snapshots.
Final `phase6_completion_validation.json` distinguishes failed computed gates
from `COMPUTED_GATES_PASS_REVIEW_PENDING`. Scientific certification still
requires review of returned raw results and historical integrity; the batch
never fabricates a global PASS or starts Phase7 just because processes exited.

## New implementation map

- `phase6_completion_export.py`: local-only existing build/export adapter;
  same physical arrays, (352,160,NF), Ha/a0/au, validated CPU map and evd repair.
- `phase6_completion_history.py`: raw historical observable comparisons and
  SHA256 provenance; compact reference, never replacement old results.
- `phase6_completion_controls.py`: projected (NR,1,NF) control adapters and
  stationary invariant sector; input normalization and CPU/GPU checks.
- `phase6_completion_audit.py`: atomic-unit scalar comparisons, unchanged
  norm/energy/boundary/tail/continuity and observable tolerance gates.
- `run_phase6_completion.py`: sequential checks, restart, space reserve,
  test logs, protected source hashes; no new physical propagator.
- `install_phase6_completion.py`: checksum-verified new-files-only installer.
- `tests/test_phase6_completion.py`: synthetic projection, stationary,
  NaN/failure-gate tests (toy parameters, not literature production values).

Local checks: MCEF181 tests PASS; VSC73 tests PASS, including five new tests.
Full-grid free stationary CPU check: eigen residual5.70e-12 Ha,
density L1 change5.84e-8, energy drift5.91e-14 Ha. GPU counterpart and all new
full-time dynamics remain to be measured on the server.
The reduced C initial energy reproduces the resonant full3D expectation to
1.50e-12 Ha with the historical local DBOC derivative. This is an initial
projection check, NOT a claim that their time evolutions will agree.
