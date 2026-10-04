#!/bin/sh
# Reproduces all runs and figures (serial, one numerical thread).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY scripts/run_uniform.py fishwood
$PY scripts/run_surf.py fishwood
$PY scripts/run_adaptive.py fishwood
$PY scripts/make_figure.py fishwood
$PY scripts/run_uniform.py fruittree_d6
$PY scripts/run_adaptive.py fruittree_d6
$PY scripts/make_figure.py fruittree_d6
