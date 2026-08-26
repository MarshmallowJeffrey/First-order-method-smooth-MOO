#!/usr/bin/env bash
# Convenience wrapper for running any project script with the project's
# venv and the OpenMP workaround needed on macOS: PyTorch's bundled
# libomp.dylib conflicts with the Homebrew libomp.dylib pulled in by
# cyipopt/IPOPT/OpenBLAS ("OMP: Error #15").
#
# Since the Aug-25 reorganisation the sources live in five subfolders
# ("Core Engine", baseline, objective, experiment_plot, sanity_check).
# You may pass either a path (experiment_plot/foo.py) or a bare script
# name (foo.py) — bare names are resolved by searching the subfolders.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/../../.venv/bin/python"

export KMP_DUPLICATE_LIB_OK=TRUE

cd "$SCRIPT_DIR"
TARGET="${1:-experiment_plot/experiments.py}"
if [[ ! -f "$TARGET" ]]; then
  for d in "Core Engine" baseline objective experiment_plot sanity_check; do
    if [[ -f "$d/$TARGET" ]]; then TARGET="$d/$TARGET"; break; fi
  done
fi
"$VENV_PYTHON" "$TARGET" "${@:2}"
