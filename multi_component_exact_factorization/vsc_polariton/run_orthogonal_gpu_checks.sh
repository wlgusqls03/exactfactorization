#!/usr/bin/env bash
# Two new runs only, same PF model; never overwrite historical campaigns.
set -Eeuo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
readonly vsc_root=results/vsc_polariton/phase6_gpu/orthogonal_campaign_v1
readonly vsc_inputs=results/vsc_polariton/phase6_gpu/orthogonal_transfer/inputs
readonly vsc_module=multi_component_exact_factorization.vsc_polariton.run_phase6_gpu
test -f multi_component_exact_factorization/vsc_polariton/run_phase6_gpu.py
if [[ -e "$vsc_root" ]]; then
    echo "Preserve existing $vsc_root; inspect before restart." >&2
    exit 1
fi
for vsc_n in 120 160; do
    test -f "$vsc_inputs/resonant_F$vsc_n.npz"
    test -f "$vsc_inputs/resonant_F$vsc_n.json"
done
trap 'echo "STOP: preserve diagnostics; no Phase6 PASS inferred." >&2' ERR
for vsc_n in 120 160; do
    vsc_out="$vsc_root/F${vsc_n}_dt0125"
    mkdir -p "$vsc_out/check" "$vsc_out/full"
    python -m "$vsc_module" --mode check --device 0 --steps 128 --dt .125 \
        --input "$vsc_inputs/resonant_F$vsc_n.npz" --out "$vsc_out/check" \
        2>&1 | tee "$vsc_out/check/run.log"
    python -m "$vsc_module" --mode propagate --device 0 --dt .125 \
        --input "$vsc_inputs/resonant_F$vsc_n.npz" --out "$vsc_out/full" \
        --validation "$vsc_out/check/gpu_validation.json" \
        2>&1 | tee "$vsc_out/full/run.log"
    python -c 'import json,sys; s=json.load(open(sys.argv[1]))["status"]; assert s=="COMPLETE_NOT_CERTIFIED",s' "$vsc_out/full/status.json"
done
echo "Two candidates completed; compare against old runs before certification. No Phase7."
