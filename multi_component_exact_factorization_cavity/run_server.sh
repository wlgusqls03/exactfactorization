#!/usr/bin/env bash
# All paths below are new outputs; never cleans historical data.
set -euo pipefail
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mcef-cavity-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mcef-cavity-cache}"
PYTHON_BINARY="${PYTHON_BINARY:-python}"
df -h .
free -h
mode="${1:?Usage: run_server.sh smoke|smoke-portable OUT | validate CONFIG OUT CAMPAIGN | production CONFIG OUT CAMPAIGN CASE JOBS}"
# The server workflow promises MP4 outputs: fail before any calculation if the
# encoder is missing instead of silently producing only PNG/PDF at the end.
if ! command -v ffmpeg >/dev/null 2>&1; then
    printf '오류: 자동 MP4 생성에 필요한 ffmpeg가 없습니다. 현재 conda 환경에 ffmpeg를 설치한 뒤 다시 실행하세요.\n' >&2
    exit 127
fi
ffmpeg -version
if [[ "$mode" == smoke || "$mode" == smoke-portable ]]; then
    out="${2:?new output directory}"
    "$PYTHON_BINARY" -m unittest discover -s tests -p 'test_cavity_*.py' -q
    if [[ "$mode" == smoke-portable ]]; then
        printf '\n검증 범위: 새 cavity 테스트 전체 + 기존 MCEF 테스트 + 기존 reference CLI smoke.\n'
        printf '주의: 이 명시적 portable 모드는 기존 VSC 연구 테스트를 검사하지 않습니다. 기존 VSC PASS를 뜻하지 않습니다.\n\n'
        "$PYTHON_BINARY" -m multi_component_exact_factorization_cavity.audit \
            --regression --suite mcef --outdir "$out/regression"
    else
        "$PYTHON_BINARY" -m multi_component_exact_factorization_cavity.audit \
            --regression --outdir "$out/regression"
    fi
    "$PYTHON_BINARY" -m multi_component_exact_factorization_cavity.workflow campaign \
        --preset smoke --campaign both --case all --out "$out"
    exit
fi
config="${2:?config.json required}"
out="${3:?new output root required}"
campaign="${4:-pen}"
case_name="${5:-all}"
jobs="${6:-1}"
if [[ "$mode" != validate && "$mode" != production ]]; then
    printf 'Unknown stage: %s\n' "$mode" >&2
    exit 2
fi
nvidia-smi
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset (CuPy default device)}"
"$PYTHON_BINARY" -c 'import sys,cupy; print(sys.executable); print("CuPy",cupy.__version__); print("free/total bytes",cupy.cuda.runtime.memGetInfo())'
"$PYTHON_BINARY" -m multi_component_exact_factorization_cavity.workflow resources \
    --config "$config" --gpu --jobs "$jobs"
if [[ "$mode" == validate ]]; then
    "$PYTHON_BINARY" -m multi_component_exact_factorization_cavity.validation \
        --config "$config" --out "$out/validation" --campaign "$campaign" --case "$case_name" --gpu
else
    # config.validation_dir must point to the validation directory above.
    # Each child checks source+configuration hashes before any long propagation.
    "$PYTHON_BINARY" -m multi_component_exact_factorization_cavity.workflow campaign \
        --config "$config" --out "$out" --campaign "$campaign" --case "$case_name" --jobs "$jobs" --gpu
fi
