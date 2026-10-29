#!/usr/bin/env bash
# New, explicit paper presets; no change to the legacy run_server.sh defaults.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pen-perspectives-2026-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/pen-perspectives-2026-cache}"
export PYTHONUNBUFFERED=1
python_binary="${PYTHON_BINARY:-python}"
mode="${1:?Usage: run_paper_server.sh plan|smoke|run|render OUT [options]}"
out="${2:?output root required}"
shift 2
if [[ "$mode" == plan ]]; then
    exec "$python_binary" -m multi_component_exact_factorization.pen_perspectives_2026.paper_campaign plan --out "$out" "$@"
fi
mkdir -p "$out"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"
exec > >(tee -a "$out/server.log") 2>&1
df -h "$out"
"$python_binary" -c 'import sys,numpy,scipy,matplotlib; print(sys.executable); print("NumPy",numpy.__version__); print("ffmpeg is required unless --no-movies")'
if [[ "$mode" == smoke ]]; then
    "$python_binary" -m multi_component_exact_factorization.pen_perspectives_2026.paper_campaign run \
        --out "$out" --model all --resonance both --end-fs .1 --every-fs .05 "$@"
elif [[ "$mode" == run || "$mode" == render ]]; then
    "$python_binary" -m multi_component_exact_factorization.pen_perspectives_2026.paper_campaign "$mode" --out "$out" "$@"
else
    printf 'Unknown mode: %s\n' "$mode" >&2
    exit 2
fi
