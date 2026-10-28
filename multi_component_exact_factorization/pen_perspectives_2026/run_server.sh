#!/usr/bin/env bash
# Small 4-channel CPU problem: GPU launch overhead is unnecessary. No old data removed.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pen-perspectives-2026-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/pen-perspectives-2026-cache}"
export PYTHONUNBUFFERED=1
python_binary="${PYTHON_BINARY:-python}"
mode="${1:?Usage: run_server.sh smoke|run|render OUT [extra CLI options]}"
out="${2:?new output root required}"
shift 2
mkdir -p "$out"
exec > >(tee -a "$out/server.log") 2>&1
df -h .
command -v ffmpeg
"$python_binary" -c 'import sys,numpy,scipy,matplotlib; print(sys.executable); print("NumPy",numpy.__version__)'
if [[ "$mode" == smoke ]]; then
    "$python_binary" -m unittest discover -s tests -p 'test_pen_perspectives_2026.py' -q
    "$python_binary" -m multi_component_exact_factorization.pen_perspectives_2026.workflow run \
        --out "$out" --end-fs .1 --every-fs .05 "$@"
elif [[ "$mode" == run || "$mode" == render ]]; then
    "$python_binary" -m multi_component_exact_factorization.pen_perspectives_2026.workflow "$mode" \
        --out "$out" "$@"
else
    printf 'Unknown mode: %s\n' "$mode" >&2
    exit 2
fi
