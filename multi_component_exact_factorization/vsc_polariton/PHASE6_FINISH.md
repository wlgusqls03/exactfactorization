# Resume after the stationary eigsh API failure

The returned recovery campaign completed the selected free box, x/R/dt
refinements and barrier F120/F160 runs. It then stopped before stationary
validation with `eigsh() got an unexpected keyword argument 'v0'`.

No historical source or result is modified. The new driver never invokes a
full 3D propagator. It verifies old source/input SHA256 values, completed-run
input identity and all compact raw observable gates before reuse. Added source
files are permitted, but changing an existing baseline source is not.

## Implementation

- `phase6_stationary_compat.py`: version-aware eigensolver initialization;
  same full PF action, (R,x,1) exact free vacuum sector, atomic units, same
  eigen-residual/128-step stationarity thresholds as the historical helper.
  Solve the largest algebraic eigenpair of **-H** with `which='LA'` and negate
  its eigenvalue. This is exactly the ground eigenpair of H, including when
  high-energy positive eigenvalues have larger magnitude than the ground
  energy. Never substitute `LM` on H. Residual and propagation use the original
  H, not -H. Only supply `v0` if present in the callable signature. Other solver failures
  remain failures; there is no catch-and-ignore fallback or tolerance change.
- `run_phase6_finish.py`: read-only reuse validation; separate stationary and
  27 reduced 2D-A/B/C runs; before/after regression and integrity checks;
  existing completion-audit adapter with explicitly labeled exact free sector.
- `tests/test_phase6_finish.py`: old/new signatures, unhidden solver error,
  CPU stationary propagation, protected-file mismatch.

GPU hardware is not available for a local production check. Server stationary
validation remains mandatory. CPU small-system stationary tests and mocked
old/new signatures are not represented as a production GPU PASS.

## Server handoff

Copy `results/vsc_polariton/phase6_gpu/phase6_finish_server_bundle.tar.gz`
to `/home/hbji/exactfactorization/` on the server. No new physics input bundle
and no git pull are required when installing this archive. Keep both previous
input directories and the whole `free_recovery_campaign_v1` intact.

```bash
cd /home/hbji/exactfactorization
conda activate MCEF
mkdir -p results/vsc_polariton/phase6_gpu/finish_transfer_v1
tar -xzf phase6_finish_server_bundle.tar.gz -C results/vsc_polariton/phase6_gpu/finish_transfer_v1
python results/vsc_polariton/phase6_gpu/finish_transfer_v1/install_phase6_completion.py
tmux new -s phase6_finish
```

Inside tmux:

```bash
cd /home/hbji/exactfactorization
conda activate MCEF
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
python -u -m multi_component_exact_factorization.vsc_polariton.run_phase6_finish
```

Scientific outputs are in `results/vsc_polariton/phase6_gpu/finish_campaign_v1`.
Regression test fixtures alone may use temporary directories. Reserve 2 GiB
for the compact controls/logs; no additional full 3D wave archive is created.
Do not run the previous recovery driver again. Completed new controls are
reused on retry; an interrupted reduced control may restart that control.
An unexpected crash leaving batch.lock requires checking the process is dead
before removing only that lock. Do not change source/input files mid-campaign.

Return results even if validation fails:

```bash
tar -czf phase6_finish_results.tar.gz -C results/vsc_polariton/phase6_gpu finish_campaign_v1
```

`COMPUTED_GATES_PASS_REVIEW_PENDING` means computed gates pass, not final
Phase6 certification. Phase7 never starts automatically. Source integrity,
stationary scope and full3D versus reduced-model physics need final review.

## Second legacy-CuPy correction (LA support)

The returned `finish_campaign_v1` verifies all seven reused full runs and all
five comparisons, then stops because legacy CuPy supports LM/LA but not SA.
No stationary or reduced-control result was generated. Preserve this failed
attempt. After obtaining the updated source via Git, use a NEW output folder
because `finish_campaign_v1/finish_baseline.json` pins the previous source:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
python -u -m multi_component_exact_factorization.vsc_polariton.run_phase6_finish \
  --out results/vsc_polariton/phase6_gpu/finish_campaign_v2
```

The previous input/full-recovery paths are unchanged. Do not edit/delete any
baseline to make the source check pass. The earlier 6KB bundle contains the
old SA implementation and must NOT be reinstalled for this retry.

```bash
tar -czf phase6_finish_v2_results.tar.gz \
  -C results/vsc_polariton/phase6_gpu finish_campaign_v2
```
