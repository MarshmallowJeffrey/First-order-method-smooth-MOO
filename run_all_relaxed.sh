#!/bin/sh
# Relaxed-LP variant for review (config.VARIANTS["relaxed_lp"]): Fruit Tree only; results_relaxed/ and figures_relaxed/.
# The default results and figures (run_all.sh) are not touched.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
V="--variant relaxed_lp --results results_relaxed"
$PY scripts/run_uniform.py fruittree_d6 $V
$PY scripts/run_adaptive.py fruittree_d6 $V
$PY scripts/time_repeats.py fruittree_d6 --repeats 5 $V
$PY scripts/make_figure.py fruittree_d6 $V --figures figures_relaxed
