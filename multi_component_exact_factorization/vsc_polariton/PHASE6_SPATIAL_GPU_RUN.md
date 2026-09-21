# Spatial GPU refinement handoff

New numerical candidates only; historical sources/results unchanged.
All four use resonant eta=.094, omega=170.6 meV, full operator DSE,
F120, dt=.125 au, target1652 au (39.9599690752 fs).
Compare with completed F120_dt0125; never rerun historical F80 presets.

| ID | x half-box | Nx | R half-box | NR |
|---|---:|---:|---:|---:|
| xbox | 28.8 | 192 | 4.4 | 352 |
| xfine | 24 | 192 | 4.4 | 352 |
| Rbox | 24 | 160 | 4.6 | 368 |
| Rfine | 24 | 160 | 4.4 | 440 |

Numerical units: coordinates a0, dt atomic time. No physical tuning.
Criteria unchanged from PHASE6_RESUME_PLAN. No renormalization workaround
for the separate F160 norm failure. Spatial candidates may also fail;
retain their diagnostics and stop rather than relax tolerances.

Local-only input generation:
`OPENBLAS_NUM_THREADS=1 python -m multi_component_exact_factorization.vsc_polariton.phase6_spatial_gpu_export`
Uses historical export checks and a scoped runtime adapter; no old preset
or source is edited. New electronic caches are inside spatial_transfer/local_build.
Do not run exporter on server: the portable NPZ/JSON packets suffice.

Transfer phase6_gpu_spatial_inputs.tar.gz separately; git does not carry NPZ.
On server, put the tarball in repository root, activate MCEF, then:

```bash
mkdir -p results/vsc_polariton/phase6_gpu/spatial_transfer
tar -xzf phase6_gpu_spatial_inputs.tar.gz -C results/vsc_polariton/phase6_gpu/spatial_transfer
CUDA_VISIBLE_DEVICES=1 bash results/vsc_polariton/phase6_gpu/spatial_transfer/run_spatial_gpu_checks.sh
```

Run inside tmux. The script checks CPU/GPU agreement for each packet before
full propagation. It refuses an existing spatial_campaign_v1 directory.
Never delete a previous campaign to bypass this guard; inspect restart state.
Only COMPLETE_NOT_CERTIFIED permits the next run. No Phase7 execution.
Existing GPU backend/driver are sufficient; new script is also in the bundle.

Send compact results (including any failures):

```bash
tar -czf phase6_spatial_results.tar.gz --exclude='wave_*.npz' --exclude='restart.npz' --exclude='*.pending' -C results/vsc_polariton/phase6_gpu spatial_campaign_v1
```

Retain waves/restarts on server. Raw waves+restart total about25.6 GiB
for four complete runs; allow40 GB free. Runtime is not measured for these
grids: budget roughly5–9 hours serial on2080Ti, with I/O/load uncertainty.
This is a partial convergence campaign, not the complete Phase6 gate.
