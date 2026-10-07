#!/usr/bin/env bash
# Model A ONLY. Historical MCEF and VSC dynamics are never run or modified.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
model_a_source="${1:-results/vsc_polariton/model_a/server_v1/campaign}"
model_a_output="${2:-results/vsc_polariton/model_a/server_2000_v1}"
mkdir -p "$(dirname "$model_a_output")"
export MPLCONFIGDIR="$(cd "$(dirname "$model_a_output")" && pwd)/mpl_cache"
mkdir -p "$MPLCONFIGDIR"
python -m unittest multi_component_exact_factorization.vsc_polariton.model_a.test_extension -q
python -u -m multi_component_exact_factorization.vsc_polariton.model_a.extend_2000 \
  --source "$model_a_source" --out "$model_a_output" --backend gpu \
  2>&1 | tee -a "$model_a_output.log"
