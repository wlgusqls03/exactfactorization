#!/usr/bin/env bash
# CPU-only postprocessing. Does not launch the TDSE campaign or touch its outputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
model_a_campaign="${1:-results/vsc_polariton/model_a/server_v1/campaign}"
model_a_audit="${2:-results/vsc_polariton/model_a/server_v1/next_review_v1}"
if [[ -e "$model_a_audit" || -e "$model_a_audit.log" ]]; then
  echo "Existing audit/log preserved. Choose a NEW second argument: $model_a_audit" >&2
  exit 1
fi
mkdir -p "$(dirname "$model_a_audit")"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$(cd "$(dirname "$model_a_audit")" && pwd)/review_mpl_cache"
mkdir -p "$MPLCONFIGDIR"
python -m unittest \
  multi_component_exact_factorization.vsc_polariton.model_a.test_event_audit \
  multi_component_exact_factorization.vsc_polariton.model_a.test_review_plot -q
python -u -m multi_component_exact_factorization.vsc_polariton.model_a.next_review \
  --campaign "$model_a_campaign" --out "$model_a_audit" \
  2>&1 | tee "$model_a_audit.log"
