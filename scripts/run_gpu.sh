#!/usr/bin/env bash
# GPU launcher for train / evaluate / submission.
#
# Usage:
#   ./scripts/run_gpu.sh train         → train all models
#   ./scripts/run_gpu.sh evaluate      → offline evaluation
#   ./scripts/run_gpu.sh submission    → generate submission.csv
set -euo pipefail

SITE=$(find ./.venv/lib/ -maxdepth 2 -name "site-packages" | head -n 1)/nvidia
CU13_LIB="${SITE}/cu13/lib"
CUBLAS_LIB="${SITE}/cublas/lib"
CURAND_LIB="${SITE}/curand/lib"
CUSOLVER_LIB="${SITE}/cusolver/lib"
CUSPARSE_LIB="${SITE}/cusparse/lib"
export LD_LIBRARY_PATH="${CU13_LIB}:${CUBLAS_LIB}:${CURAND_LIB}:${CUSOLVER_LIB}:${CUSPARSE_LIB}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="."

PYTHON_BIN="./.venv/bin/python"

MODE="${1:-train}"

case "$MODE" in
  preprocess)
    exec $PYTHON_BIN scripts/preprocess.py "${@:2}"
    ;;
  train)
    exec $PYTHON_BIN scripts/train.py --use_gpu "${@:2}"
    ;;
  evaluate)
    exec $PYTHON_BIN scripts/evaluate.py "${@:2}"
    ;;
  inference)
    exec $PYTHON_BIN scripts/inference.py "${@:2}"
    ;;
  submission)
    exec $PYTHON_BIN scripts/inference.py "${@:2}"
    ;;
  python)
    exec $PYTHON_BIN "${@:2}"
    ;;
  *)
    echo "Usage: $0 {preprocess|train|evaluate|inference|submission|python <script>}"
    exit 1
    ;;
esac
