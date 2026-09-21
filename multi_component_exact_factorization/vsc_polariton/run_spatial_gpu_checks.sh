#!/usr/bin/env bash
# Run from repository root with MCEF activated. No Phase7 or automatic PASS.
set -Eeuo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
readonly vsc_root=results/vsc_polariton/phase6_gpu/spatial_campaign_v1
readonly vsc_inputs=results/vsc_polariton/phase6_gpu/spatial_transfer/inputs
readonly vsc_module=multi_component_exact_factorization.vsc_polariton.run_phase6_gpu
test -f multi_component_exact_factorization/vsc_polariton/run_phase6_gpu.py
if [[ -e "$vsc_root" ]]; then
    echo "Preserve existing campaign: $vsc_root. Inspect before any restart." >&2
    exit 1
fi
for vsc_tag in xbox xfine Rbox Rfine; do
    test -f "$vsc_inputs/$vsc_tag.npz"
    test -f "$vsc_inputs/$vsc_tag.json"
done
python -c 'import cupy as cp; cp.cuda.Device(0).use(); print(cp.cuda.runtime.getDeviceProperties(0)["name"])'
trap 'echo "STOP: preserve outputs; send status, observables and logs. Phase6 is not certified." >&2' ERR
for vsc_tag in xbox xfine Rbox Rfine; do
    vsc_out="$vsc_root/$vsc_tag"
    mkdir -p "$vsc_out/check" "$vsc_out/full"
    echo "START $vsc_tag: F120 dt=.125, target 1652 au"
    python -m "$vsc_module" --mode check --device 0 --steps 128 --dt .125 \
        --input "$vsc_inputs/$vsc_tag.npz" --out "$vsc_out/check" \
        2>&1 | tee "$vsc_out/check/run.log"
    python -m "$vsc_module" --mode propagate --device 0 --dt .125 \
        --input "$vsc_inputs/$vsc_tag.npz" --out "$vsc_out/full" \
        --validation "$vsc_out/check/gpu_validation.json" \
        2>&1 | tee "$vsc_out/full/run.log"
    python -c 'import json,sys; s=json.load(open(sys.argv[1]))["status"]; assert s=="COMPLETE_NOT_CERTIFIED",s' "$vsc_out/full/status.json"
done
echo "Spatial runs completed. Independent observable comparisons still required; NOT Phase6 PASS."
