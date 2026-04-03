#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"
# Set a stable timestamp for deterministic dashboard_summary.json during reproducibility checks.
export DASHBOARD_GENERATED_AT_UTC="${DASHBOARD_GENERATED_AT_UTC:-1970-01-01T00:00:00+00:00}"

echo "Running end-to-end pipeline reproducibility flow..."
echo "Using python: $PYTHON_BIN"

"$PYTHON_BIN" src/preprocessing.py
"$PYTHON_BIN" src/feature_engineering.py
"$PYTHON_BIN" src/dimensionality_reduction.py
"$PYTHON_BIN" src/clustering.py
"$PYTHON_BIN" src/scoring.py
"$PYTHON_BIN" src/recommendation_engine.py
"$PYTHON_BIN" src/dashboard_prep.py
"$PYTHON_BIN" src/evaluation.py

echo "Reproducibility pipeline completed successfully."
