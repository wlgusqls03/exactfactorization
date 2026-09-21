# Phase 6 GPU adapter — server handoff

Existing MCEF/Phase1–6 sources were NOT changed. This is the same full PF
Hamiltonian, complex128/float64 and fourth-order split, accelerated with
CuPy FFTs/matrix products. Full DSE g_chi²(R-x)²/omega remains included.
No Phase7 code is imported. CPU–GPU agreement does not certify 40-fs
grid/Fock/time convergence or Phase6 scientific PASS.

## What is shipped

Only new GPU source/test/documentation is committed. Historical VSC sources
were already untracked; they are not silently added to this commit.
The SERVER runtime only imports `run_phase6_gpu.py` and
`phase6_gpu_backend.py`, plus NumPy/SciPy/CuPy. It does not need the full
historical VSC source tree or its 912 protected artifacts.

The separately transferred input packet contains the historical CPU initial
Psi, molecular/operator arrays, aligned BO basis, and source/input hashes.
Export tests the new CPU backend against existing FullPF/FullSplit/observe.
For the supplied resonant_base packet: step L2 error1.16521e-15;
H-action L2 error9.00090e-17; observable errors at roundoff. Packet ~70 MiB.
SHA256: `39753d94114e084534f16246f11b22531ca7927fdf68df69814f999914de9e0c`.
Transfer BOTH NPZ and its JSON sidecar; the runtime rejects hash mismatch.

## 1. Install the commit and transfer input

An accompanying `phase6_gpu_server_bundle.tar.gz` contains a git-format
patch for this commit and `inputs/resonant_base.npz` / `.json`.
Upload that archive to `/home/hbji/exactfactorization/` on the server.
Then, from the server shell:

```bash
cd /home/hbji/exactfactorization
mkdir -p results/vsc_polariton/phase6_gpu/transfer
tar -xzf phase6_gpu_server_bundle.tar.gz -C results/vsc_polariton/phase6_gpu/transfer
git status --short
git am results/vsc_polariton/phase6_gpu/transfer/phase6_gpu.patch
```

Apply once only. If the commit is already installed through your normal git
workflow, skip `git am`. If applying conflicts with existing files, stop;
do not reset/clean or overwrite local work. No push is performed for you.

## 2. Check CUDA/CuPy first

```bash
conda activate MCEF
nvidia-smi
python -c "import cupy as cp; cp.show_config(); print('GPU count:', cp.cuda.runtime.getDeviceCount())"
```

If this works, do NOT reinstall CuPy. If CuPy is missing and this is still
the previously shown CUDA11.2/Python3.9 server, the compatible pinned wheel is:

```bash
python -m pip install 'cupy-cuda11x==13.2.0'
```

CuPy13.2 supports Python3.9 and CUDA11.2–11.8:
https://docs.cupy.dev/en/v13.2.0/install.html
The wheel requires a working compatible CUDA toolkit/runtime; nvidia-smi's
CUDA version is driver capability, not proof that NVRTC/toolkit is installed.
If CUDA is now12.x/13.x, Python differs, or another CuPy package already
exists, do not install this wheel blindly. Send show_config/error first.
Never install multiple cupy/cupy-cudaXX distributions into one environment.

## 3. Run only the short comparison first

```bash
tmux new -s vsc_gpu_check
```

Inside tmux:

```bash
cd /home/hbji/exactfactorization
conda activate MCEF
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -o pipefail
mkdir -p results/vsc_polariton/phase6_gpu/check_resonant_base
CUDA_VISIBLE_DEVICES=1 python -m multi_component_exact_factorization.vsc_polariton.run_phase6_gpu \
  --mode check --device 0 --steps 128 \
  --input results/vsc_polariton/phase6_gpu/transfer/inputs/resonant_base.npz \
  --out results/vsc_polariton/phase6_gpu/check_resonant_base \
  2>&1 | tee results/vsc_polariton/phase6_gpu/check_resonant_base/run.log
```

Physical GPU1 is remapped to device0 by CUDA_VISIBLE_DEVICES. To use physical
GPU0, change only CUDA_VISIBLE_DEVICES=0. Do not run on an occupied GPU.
Detach with Ctrl-b then d; reattach `tmux attach -t vsc_gpu_check`.
128 steps at dt=.25 correspond to .774043 fs, NOT the full40fs interval.
CPU and GPU both run all128 steps from exactly the same initial Psi.
Expect CPU reference alone roughly4minutes at current local speed; allow
5–15minutes for setup/first CUDA compilation/CPU timing. Server time is
measured rather than assumed. Old GPUFP64 throughput may limit acceleration.

The check reports synchronized kernel timing separately from observables,
warmup and initial transfer. Full-run estimates exclude checkpoint I/O and
convergence repeats. GPU memory pool reserved bytes is not peak total VRAM.

## 4. Send these results back

Send BOTH:

- `results/vsc_polariton/phase6_gpu/check_resonant_base/gpu_validation.json`
- `results/vsc_polariton/phase6_gpu/check_resonant_base/run.log`

Also include nvidia-smi output if the GPU has other jobs. No wavefunction
NPZ needs to be returned for this first check. JSON contains:

- status / gpu_propagation_allowed;
- norm, energy, direct Psi L2, overlap, product, flux, bare BO character,
  photon number and mean-R errors across sampled times;
- actual GPU/runtime/driver/CuPy information;
- CPU/GPU seconds, measured speedups, estimated39.96fs hours;
- absolute physical diagnostic failures, separately from backend agreement.

If it fails before JSON creation, send run.log and `error_*.json` in the same
directory. OOM/CUDA errors are errors, not CPU fallbacks or scientific PASS.
Do not repeat a check into an existing validation directory: use a new suffix.

## 5. Longer GPU propagation (after inspecting the comparison)

Do not launch this before the short result is checked. A matching >=128-step
PASS is enforced by input SHA, dt, GPU model, runtime/driver/CuPy and code SHA.
The run starts from the packet's initial state, not the current CPU checkpoint.
It supports its own atomic restart; the existing CPU campaign is untouched.

```bash
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 python -m \
  multi_component_exact_factorization.vsc_polariton.run_phase6_gpu \
  --mode propagate --device 0 \
  --input results/vsc_polariton/phase6_gpu/transfer/inputs/resonant_base.npz \
  --validation results/vsc_polariton/phase6_gpu/check_resonant_base/gpu_validation.json \
  --out results/vsc_polariton/phase6_gpu/resonant_base_full
```

Same command resumes an interrupted GPU run if no live lock remains.
SIGINT/SIGTERM saves a checkpoint at the next completed step. Hard kills can
leave a lock; inspect the PID before recovering it. Finished or failed runs
are preserved and cannot be overwritten. Observables every4au, scientific
wavefunctions every32au plus initial/final/failure. Event-specific snapshot
replay, stationary tests, and all full-interval comparisons are still needed
before Phase7. This is one candidate, NOT the whole convergence campaign.

## Implementation contracts

`phase6_gpu_backend.py` lines 11–100: PFBackend holds float64 operator arrays and
complex128 Psi(R,x,n) on the selected device; step is the historical Yoshida
composition; action is full PF; observe contracts on device, transfers only
small diagnostic arrays/scalars. Units and signs follow existing Phase6.
CPU unit tests check Hermiticity, fourth-order error, norm, energy partition,
population bookkeeping and lossless checkpoint continuation.

`phase6_gpu_export.py` lines 13–44: local-only adapter of existing Phase6 input builder;
production numerical arrays are exported without physical retuning.
It requires the existing local untracked VSC tree; the server does NOT
need to run the exporter. Other grids/cases require separately exported
packets and independent matching hardware checks.

`run_phase6_gpu.py` lines 19–203: checksum-protected load, CPU/GPU check, strict short
gate, synchronized timings, error JSON, and checkpointed GPU run. No PASS
is assigned to the entire Phase6 by this driver.

Actual CUDA execution is not tested on the local CPU-only environment.
That missing check is the explicit purpose of the server command above.

### Function-level map

| Location | Contract / units / validation |
|---|---|
| phase6_gpu_backend.py 11–32 | Init: float64 operator arrays; exported Li PF arrays, same mass/coupling; legacy adapter comparison |
| phase6_gpu_backend.py 34–38 | host/sync: representation transfer and stream synchronization; no physical operation |
| phase6_gpu_backend.py 40–49 | step: U4(dt) Psi(R,x,n); time au, complex128; dense expm order test + server equivalence |
| phase6_gpu_backend.py 51–66 | parts/action: Tx,TR,V,Hph,HLM,HDSE times Psi; Ha*Psi; Hermiticity and legacy action comparison |
| phase6_gpu_backend.py 68–100 | observe: marginals, currents, BO populations, energies; au; legacy observable comparison and accounting tests |
| phase6_gpu_export.py 13–44 | export: immutable packet plus SHA/provenance; (NR,Nx,NF) Psi; independent old/new CPU comparison |
| run_phase6_gpu.py 19–49 | SHA/load/code identity/atomic save: dimensionless hashes; transfer corruption and restart roundtrip tests |
| run_phase6_gpu.py 51–57 | environment: device/software identity, memory bytes; reported on server, no inference from GPU name alone |
| run_phase6_gpu.py 60–64 | absolute_failures: historical physical tolerances in au; NaN and equality-at-limit rejection |
| run_phase6_gpu.py 67–120 | compare: paired CPU/GPU time series, synchronized seconds, wave L2; predeclared backend tolerances + old absolute gates |
| run_phase6_gpu.py 123–165 | propagate: matching PASS required, same U4, sparse scientific Psi plus restart; failed-gate unit test |
| run_phase6_gpu.py 168–203 | main: CLI/locks/error JSON; output restricted to new GPU directory; unavailable-CUDA error path exercised locally |
