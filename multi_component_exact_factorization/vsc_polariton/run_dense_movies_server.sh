#!/usr/bin/env bash
# Existing data read-only. New restart/compact fields/MP4s under results only.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
movie_packet="${1:-results/vsc_polariton/phase6_gpu/barrier_f240_server_transfer_v1/inputs/barrier_F240.npz}"
movie_campaign="${2:-results/vsc_polariton/photon_real_grid/barrier_v1}"
movie_output="${3:-results/vsc_polariton/photon_real_grid/dense_movies_server_v1}"
movie_common=(--input "$movie_packet"
  --source-run "$movie_campaign/full/base_Q384/full"
  --validation "$movie_campaign/checks/base_Q384/gpu_validation.json"
  --device 0 --every-au 4 --fps 24)
test -f "$movie_packet"
test -f "$movie_campaign/full/base_Q384/full/status.json"
test -f "$movie_campaign/checks/base_Q384/gpu_validation.json"
mkdir -p "$movie_output/logs"
python -m multi_component_exact_factorization.vsc_polariton.run_vsc_dense_movies \
  --mode plan "${movie_common[@]}" --out "$movie_output/full" \
  2>&1 | tee "$movie_output/logs/plan.log"
python -m multi_component_exact_factorization.vsc_polariton.run_vsc_dense_movies \
  --mode replay "${movie_common[@]}" --end-au 8 --no-render --out "$movie_output/probe" \
  2>&1 | tee -a "$movie_output/logs/probe.log"
python -m multi_component_exact_factorization.vsc_polariton.run_vsc_dense_movies \
  --mode replay "${movie_common[@]}" --end-au 1652 --out "$movie_output/full" \
  2>&1 | tee -a "$movie_output/logs/full.log"
printf '%s\n' "Movies: $movie_output/full/movies" \
  "Send full/replay_status.json, full/movies/movie_manifest.json, logs, and MP4s." \
  "Do not send restart.npz or all fields unless requested. No Phase7 PASS is implied."
