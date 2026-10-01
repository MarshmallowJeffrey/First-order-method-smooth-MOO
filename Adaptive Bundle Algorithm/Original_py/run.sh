#!/usr/bin/env bash
# Convenience wrapper for running experiments.py (or any script in this
# directory) with the project's venv and the OpenMP workaround needed on
# macOS: PyTorch's bundled libomp.dylib conflicts with the Homebrew
# libomp.dylib pulled in by cyipopt/IPOPT/OpenBLAS ("OMP: Error #15").
# See the comment at the top of experiments.py for details.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/../../.venv/bin/python"

export KMP_DUPLICATE_LIB_OK=TRUE

cd "$SCRIPT_DIR"
"$VENV_PYTHON" "${1:-experiments.py}" "${@:2}"
