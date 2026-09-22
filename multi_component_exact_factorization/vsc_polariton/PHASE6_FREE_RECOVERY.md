# Bounded free-case recovery and completion handoff

No historical source/result modified. This is NOT a promise that an unknown
convergence test will pass. The script bounds the numerical search, stops on
non-box failures, and never runs Phase7. All physical constants are unchanged.

## What was discovered before this handoff

- Server free x+-24, NR352, dt.125 stopped at168 au: electron edge1.088587e-8.
- Exact NF=1 local replay reproduces the F120 result. Halving dt changes
  the edge to1.088600e-8: time refinement does not fix this short-window tail.
- Local NF=1 probe at x+-28.8, NR352 stops at288 au: edge1.110518e-8.
- Local x+-36, NR352 reaches1576 au then continuity2.009310e-6 exceeds2e-6.
- Local x+-48, NR352 also fails continuity at1576 au. No tolerance relaxed.

Therefore do NOT blindly rerun the old completion batch. Free propagation
now starts at NR440 (dR=.02), with NR550 (dR=.016) as nuclear refinement.
The Gaussian width, momentum and target molecular energy are inherited
from the original threshold preparation. Independent electronic eigenvalues
give an initial molecular energy mismatch about1.54e-12 Ha on all new grids.

Before handoff, the new L36, NR440, dx.3, dt.125 exact-vacuum candidate was
actually propagated locally to1652 au: all absolute gates PASS. Norm error
2.62e-13; energy drift5.53e-9 Ha; electron edge9.48e-9; continuity1.29e-6;
integrated continuity1.21e-4. Raw compact data and status are preserved in
`free_recovery_transfer_v2/local_probe/L36_R440.npz` and `.json`.
This is one full-time candidate, not a replacement for the server box/grid/dt
comparisons or a declaration of Phase6 PASS.

## Exact reduction and finite numerical search

At eta=0, initial photon vacuum never couples to n>0. Thus
Psi(x,R,n,t)=psi(x,R,t) delta_n0, including photon zero-point energy.
Keeping n=0 is EXACT, not a BO approximation: electron x is still propagated.
The existing FullPF/FullSplit and portable backend action/step/observables
are cross-checked for every exported packet; errors below1e-10.

1. Test x half-boxes28.8,36,48,72 at dx=.3, NR440, dt=.125 to1652 au.
   Only a failure of the electron-edge gate alone permits the next box.
2. Require TWO adjacent completed safe boxes with converged observables.
   Select the smaller box. No pair by72 => BLOCKED, no unbounded reruns.
3. On the selected box compare dx=.24; NR550; dt=.0625 independently.
4. Only after all free gates pass: barrier-frequency F120 and F160, old
   validated resonant spatial grid, dt=.125. No resonant rerun.
5. Selected-box stationary free sector; existing27 reduced A/B/C control
   runs; regression; raw observable audit. No automatic scientific PASS.

Norm1e-9, energy1e-6 Ha, edges/tail1e-8, continuity2e-6 au and integrated
continuity2e-4 remain unchanged. Differences: product2e-4, flux2e-6,
nph2e-3, mean R2e-4. All tests cover the complete1652-au interval.

The controls remain on the historically validated 2D grid; new free nuclear
grid convergence is checked separately. The new report labels the free
propagation NF=1 explicitly; internal legacy audit aliases are not a claim
that120 photon states were propagated.

## Server (existing completion_transfer inputs must remain)

Transfer `phase6_free_recovery_server_bundle.tar.gz` to the repository root.
It contains all12 small new inputs and only NEW recovery source/test files.
The existing completion bundle's free/resonant/barrier inputs are reused for
controls and barrier runs. Neither old calculation directory is deleted.

```bash
cd /home/hbji/exactfactorization
conda activate MCEF
mkdir -p results/vsc_polariton/phase6_gpu/free_recovery_transfer_v2
tar -xzf phase6_free_recovery_server_bundle.tar.gz \
  -C results/vsc_polariton/phase6_gpu/free_recovery_transfer_v2
python results/vsc_polariton/phase6_gpu/free_recovery_transfer_v2/install_phase6_completion.py
tmux new -s phase6_recovery
```

Inside tmux:

```bash
conda activate MCEF
cd /home/hbji/exactfactorization
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
python -u -m multi_component_exact_factorization.vsc_polariton.run_phase6_free_recovery
```

All results/logs/checkpoints are below
`results/vsc_polariton/phase6_gpu/free_recovery_campaign_v1`.
Temporary regression figures may print /tmp paths, NOT scientific output.
The same command resumes interrupted jobs, skips completed jobs, preserves
failed attempts and refuses source/input identity changes. Inspect running
processes before any recovery of a stale lock after a hard kill.

Budget20 GiB free; expected output about15–18 GiB including retained failed
free candidates, waves, controls and checkpoints. No old data deletion.
Runtime expectation on2080Ti: **about4–6 h if the declared gates pass**,
dominated by two barrier runs and reduced controls. This is an estimate,
not measured end-to-end. New failures remain BLOCKED, not relaxed to finish.

Return (also on failure):

```bash
tar -czf phase6_free_recovery_results.tar.gz \
  --exclude='wave_*.npz' --exclude='restart.npz' --exclude='*.pending' \
  -C results/vsc_polariton/phase6_gpu free_recovery_campaign_v1
```

Keep full wave snapshots on the server for Phase7. Final review uses this
archive plus historical orthogonal/spatial data. Phase7 is not automatic.

## New code roles

- `phase6_free_recovery_inputs.py`: local-only full-H packet generation,
  arrays(NR,Nx,1), au; direct BO solve for initial lifting, then explicit x
  propagation. Validates target energy and existing-operator equivalence.
- `phase6_free_recovery_export.py`: preserved generator for original NR352
  diagnostic attempts, not the server production handoff.
- `run_phase6_free_recovery.py`: bounded box gate, independent refinements,
  old GPU driver/control reuse, source/input identity and final audit adapter.
- `tests/test_phase6_free_recovery.py`: exact vacuum-sector equivalence,
  reject non-boundary failures, and reject terminal-run input mismatch.

Local regression: existing MCEF181 tests PASS; VSC76 tests PASS including
three new vacuum/recovery tests. GPU equivalence remains a server gate.
