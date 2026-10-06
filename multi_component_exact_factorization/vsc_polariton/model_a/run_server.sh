#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
model_a_out="${1:-results/vsc_polariton/model_a/server_v1}"
mkdir -p "$model_a_out"
model_a_out="$(cd "$model_a_out" && pwd)"
export MPLCONFIGDIR="$model_a_out/mpl_cache"
mkdir -p "$MPLCONFIGDIR"
python -m unittest multi_component_exact_factorization.vsc_polariton.model_a.test_model_a -q \
  2>&1 | tee "$model_a_out/unit_tests.log"
python -u -m multi_component_exact_factorization.vsc_polariton.model_a.campaign \
  --backend gpu --out "$model_a_out/campaign" \
  2>&1 | tee -a "$model_a_out/campaign.log"
