# Fock rotation precision repair: isolated packet revision

Same Li main Eq.(1) PF Hamiltonian, full (R-x)^2 DSE, same initial Psi,
physical parameters, grids and diagnostics. No renormalization or tolerance
relaxation. Historical MCEF, FullSplit and GPU backend/driver are unchanged.

The original truncated Q=a+a† eigenvectors had max |VᵀV-I| about4.15e-14
at N=160. Dense symmetric divide-and-conquer (`eigh`, driver `evd`) reduces
this to about1.53e-15 locally. This addresses a demonstrated source of
roundtrip drift, NOT a guarantee of 40-fs GPU norm conservation. CPU/GPU
equivalence and full-interval diagnostics must be rerun on the server.
Do not feed an old restart into the new packet campaign.

## Implementation map

`phase6_orthogonal_packets.py` lines 16–24: diagonalize the identical
dimensionless Q (NF,NF), test eigen-residual and orthogonality.
Lines 27–62: preserve physical arrays and Psi (NR,Nx,NF), check new vs old
CPU one-step L2 <1e-10, retain parent SHA and original adapter provenance,
export new NPZ/JSON without overwrite. Units remain atomic.
`run_orthogonal_gpu_checks.sh` lines 1–31: F120 then F160, dt=.125 au,
1652 au target, hardware check first, existing absolute gates unchanged.
`tests/test_phase6_orthogonal.py` lines 8–25: independent dense exponential,
Q reconstruction, and6000 unrenormalized roundtrips. Algebra tests are not
full production convergence tests.

## Server handoff

Transfer phase6_gpu_orthogonal_inputs.tar.gz separately (git has no NPZ).
Its script runs with the GPU backend already installed on the server.

```bash
cd /home/hbji/exactfactorization
mkdir -p results/vsc_polariton/phase6_gpu/orthogonal_transfer
tar -xzf phase6_gpu_orthogonal_inputs.tar.gz -C results/vsc_polariton/phase6_gpu/orthogonal_transfer
# Inside tmux, with conda MCEF active:
CUDA_VISIBLE_DEVICES=1 bash results/vsc_polariton/phase6_gpu/orthogonal_transfer/run_orthogonal_gpu_checks.sh
```

Output: results/vsc_polariton/phase6_gpu/orthogonal_campaign_v1.
F120 rerun anchors numerical-representation regression; F160 completes the
previously failed fine-dt comparison. Do not blindly rerun spatial cases.
Estimated2–5h GPU plus I/O/CPU checks; not measured for revised packets.
Complete output about12.7GiB waves+restart; allow20GB free.

```bash
tar -czf phase6_orthogonal_results.tar.gz --exclude='wave_*.npz' --exclude='restart.npz' --exclude='*.pending' -C results/vsc_polariton/phase6_gpu orthogonal_campaign_v1
```

## Storage: do not erase unique scientific archives

Compact transferred results DO NOT include full wavefunctions. Preserve
all observable NPZ, status/check JSON, logs, input packets and hashes.
Preserve ALL baseline F120_dt0125 scientific wave snapshots for prospective
Phase7, and the final failed F160_dt0125 wave as a failure diagnostic.
Historical Phase1–5 archives are outside this cleanup scope.
Redundant completed-refinement waves can be moved to external storage and
verified before removal from the server; keep their compact observables.
Completed-run restarts are not needed to resume already finished runs,
but retain failed/interrupted restarts until disposition is settled.
No deletion is performed by this change or by the shell script.
Phase6 remains uncertified; Phase7 remains gated.
