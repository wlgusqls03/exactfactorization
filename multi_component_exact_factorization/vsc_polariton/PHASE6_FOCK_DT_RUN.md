# GPU Fock/time-step campaign

`run_fock_dt_checks.sh` lines 1–50: the version-controlled copy of the
previously delivered script. It runs F120/F160 at dt=.25/.125 au in sequence,
with a 128-step CPU/GPU check before each 1652-au candidate. Failed checks,
failed diagnostics and interrupted candidates stop the sequence. Completion
does not certify convergence or Phase6 PASS. No Phase7 is started.

The script matches the copy in `phase6_gpu_fock_inputs.tar.gz`. Do not run
both copies: they use the same output directory and preserve existing runs.
Historical MCEF and Phase1–6 source/output are not modified.

## Input transfer (not included in git)

Upload `phase6_gpu_fock_inputs.tar.gz` to the repository root on the server.
This 244-MiB artifact contains F120/F160 NPZ files and checksum JSON sidecars.
Git pull alone does NOT fetch them. From the server repository root:

```bash
mkdir -p results/vsc_polariton/phase6_gpu/transfer
tar -xzf phase6_gpu_fock_inputs.tar.gz -C results/vsc_polariton/phase6_gpu/transfer
```

## Run the tracked script

After installing this commit on the server, inside tmux:

```bash
cd /home/hbji/exactfactorization
conda activate MCEF
CUDA_VISIBLE_DEVICES=1 bash multi_component_exact_factorization/vsc_polariton/run_fock_dt_checks.sh
```

Outputs: `results/vsc_polariton/phase6_gpu/fock_dt_campaign_v1/`.
Existing output is refused; do not delete it to force a rerun.
Estimated serial budget for these four candidates, including short CPU/GPU
checks: roughly 4–8 hours, extrapolated from the F80 server benchmark, not
a measured F120/F160 timing. Allow about 40GB free disk. Failures or slow
shared storage change this estimate.

Send diagnostic files without the large scientific/restart wavefunctions:

```bash
tar -czf phase6_fock_dt_results.tar.gz \
  --exclude='wave_*.npz' --exclude='restart.npz' --exclude='*.pending' \
  -C results/vsc_polariton/phase6_gpu fock_dt_campaign_v1
```

Next, compare Fock/time-step observables before choosing settings for
electron/nuclear grid refinements, other physical cases and reduced controls.
