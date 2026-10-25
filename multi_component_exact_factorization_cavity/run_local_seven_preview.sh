#!/usr/bin/env bash
# CPU execution-path preview, NOT the native GPU production/convergence gate.
# Run from the repository root. Preserve every existing output directory.
set -euo pipefail
out="${1:?Usage: bash run_local_seven_preview.sh NEW_OUTPUT_ROOT [JOBS]}"
jobs="${2:-1}"
[[ "$jobs" == 1 || "$jobs" == 2 || "$jobs" == 3 ]] || { echo 'JOBS must be 1, 2 or 3'; exit 2; }
[[ ! -e "$out" ]] || { echo "Existing output preserved: $out"; exit 2; }
command -v ffmpeg >/dev/null || { echo 'ffmpeg is required for automatic MP4 output'; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mcef-seven-preview-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mcef-seven-preview-cache}"
py="${PYTHON_BINARY:-python}"
pkg=multi_component_exact_factorization_cavity
"$py" -m "$pkg.workflow" init --config "$pkg/configs/local_double_well_preview.json" \
  --out "$out/vsc_input.json" \
  --set "legacy_regression_file=\"$out/regression/regression.json\"" \
  --set "validation_dir=\"$out/validation\""
"$py" -m "$pkg.workflow" init --config "$pkg/configs/local_pen_preview.json" \
  --out "$out/pen_input.json" \
  --set "legacy_regression_file=\"$out/regression/regression.json\"" \
  --set "validation_dir=\"$out/pen_validation\""
"$py" -m unittest discover -s tests -p 'test_cavity*.py' -q 2>&1 | tee "$out/cavity_tests.log"
"$py" -u -m "$pkg.audit" --regression --suite mcef --outdir "$out/regression"
"$py" -u -m "$pkg.validation" --config "$out/vsc_input.json" --out "$out/validation" \
  --campaign vsc --case all --duration-fs .1 2>&1 | tee "$out/vsc_validation.log"
"$py" -u -m "$pkg.workflow" campaign --config "$out/vsc_input.json" \
  --campaign vsc --case all --jobs "$jobs" --out "$out" 2>&1 | tee "$out/vsc_preview.log"
"$py" -u -m "$pkg.validation" --config "$out/pen_input.json" --out "$out/pen_validation" \
  --campaign pen --case all --duration-fs .1 2>&1 | tee "$out/pen_validation.log"
"$py" -u -m "$pkg.workflow" campaign --config "$out/pen_input.json" \
  --campaign pen --case all --jobs "$jobs" --out "$out/pen_5fs" 2>&1 | tee "$out/pen_5fs.log"
# Historical CPU reference uses its original FD2 electron / FD5 nuclear
# kinetic operators and sparse-time scalar extraction, not the GPU spectral
# production path. Explicitly keep its shorter 0.5 fs smoke scope.
"$py" -u -m multi_component_exact_factorization.reference --outdir "$out/mcef" \
  --compact --nx 63 --nq 128 --nR 96 --x-min -22 --x-max 30 \
  --q-min -12.012 --q-max 11.988 --R-min 6 --R-max 13 --q0 -4 --R0 9.5 \
  --electron-excitation 1 --erf-r-qr 1 --heavy-trap-alpha .03 --heavy-trap-center 9.5 \
  --proton-force-constant .01285 --heavy-force-constant .06 \
  --proton-mass 1836 --heavy-mass 12000 --fft-workers 1 \
  --dt-au .05 --t-final-fs .5 --save-every 40 --progress-every 100 2>&1 | tee "$out/mcef_preview.log"
"$py" -u -m multi_component_exact_factorization.visualize \
  "$out/mcef/multi_component_reference.npz" --outdir "$out/mcef/figures" \
  --snapshots 3 --dpi 100 --animation-dpi 80 --fps 10 --max-frames 20 --format mp4 --animation-style all
echo 'Seven execution paths completed. Coarse CPU previews only; GPU/spatial/long-time convergence not certified.'
