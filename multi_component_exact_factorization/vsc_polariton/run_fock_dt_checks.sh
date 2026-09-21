#!/usr/bin/env bash
# Run from /home/hbji/exactfactorization after activating the MCEF environment.
# Only NEW GPU outputs. Existing CPU campaign / historical artifacts untouched.
set -Eeuo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

readonly vsc_root="results/vsc_polariton/phase6_gpu/fock_dt_campaign_v1"
readonly vsc_inputs="results/vsc_polariton/phase6_gpu/transfer/inputs"
readonly vsc_module="multi_component_exact_factorization.vsc_polariton.run_phase6_gpu"

if [[ ! -f multi_component_exact_factorization/vsc_polariton/run_phase6_gpu.py ]]; then
    echo "Run from the exactfactorization repository root." >&2
    exit 1
fi
if [[ -e "$vsc_root" ]]; then
    echo "Preserve existing campaign: $vsc_root" >&2
    echo "Do not rerun this script over it. Inspect statuses / use per-run restart." >&2
    exit 1
fi
for vsc_n in 120 160; do
    test -f "$vsc_inputs/resonant_F${vsc_n}.npz"
    test -f "$vsc_inputs/resonant_F${vsc_n}.json"
done
python -c 'import cupy as cp; cp.cuda.Device(0).use(); print("Visible GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"])'

run_candidate() {
    local vsc_tag="$1" vsc_n="$2" vsc_dt="$3"
    local vsc_case_root="$vsc_root/$vsc_tag"
    mkdir -p "$vsc_case_root/check" "$vsc_case_root/full"
    echo "START $vsc_tag: CPU/GPU comparison, then full-interval candidate"
    python -m "$vsc_module" --mode check --device 0 --steps 128 --dt "$vsc_dt" \
        --input "$vsc_inputs/resonant_F${vsc_n}.npz" \
        --out "$vsc_case_root/check" 2>&1 | tee "$vsc_case_root/check/run.log"
    python -m "$vsc_module" --mode propagate --device 0 --dt "$vsc_dt" \
        --input "$vsc_inputs/resonant_F${vsc_n}.npz" \
        --validation "$vsc_case_root/check/gpu_validation.json" \
        --out "$vsc_case_root/full" 2>&1 | tee "$vsc_case_root/full/run.log"
    python -c 'import json,sys; s=json.load(open(sys.argv[1]))["status"]; assert s=="COMPLETE_NOT_CERTIFIED", "Not a completed candidate: "+s' \
        "$vsc_case_root/full/status.json"
}

trap 'echo "STOP: preserve all outputs and send check JSON / full status / logs. No Phase6 PASS inferred." >&2' ERR
run_candidate F120_dt025 120 0.25
run_candidate F160_dt025 160 0.25
run_candidate F120_dt0125 120 0.125
run_candidate F160_dt0125 160 0.125
echo "Four candidate runs finished. Fock/dt observable comparisons are still required."
echo "This is NOT a Phase6 PASS and does not start Phase7."
