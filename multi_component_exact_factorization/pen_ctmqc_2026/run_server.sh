#!/usr/bin/env bash
# Independent trajectory dynamics. NEVER invokes the TDSE simulator.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pen-ctmqc-2026-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/pen-ctmqc-2026-cache}"
export PYTHONUNBUFFERED=1
python_binary="${PYTHON_BINARY:-python}"
mode="${1:?Usage: run_server.sh plan|smoke|run|render|compare ROOT [options]}"
out="${2:?output root required}"
shift 2
case "$mode" in
    plan) exec "$python_binary" -m multi_component_exact_factorization.pen_ctmqc_2026.workflow plan --out "$out" "$@" ;;
    smoke|run|render|compare) ;;
    *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
mkdir -p "$out" "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"
exec > >(tee -a "$out/ctmqc_server.log") 2>&1
df -h "$out"
"$python_binary" -c 'import sys,numpy,scipy,matplotlib; print(sys.executable); print("CTMQC CPU ensemble; no GPU needed",numpy.__version__)'
if [[ "$mode" == smoke ]]; then
    "$python_binary" -m multi_component_exact_factorization.pen_ctmqc_2026.workflow run \
        --out "$out" --ntraj 64 --end-fs .2 --every-fs .1 "$@"
else
    "$python_binary" -m multi_component_exact_factorization.pen_ctmqc_2026.workflow "$mode" --out "$out" "$@"
fi
