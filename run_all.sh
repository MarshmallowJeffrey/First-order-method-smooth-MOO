#!/bin/sh
# Reproduces all runs, the Fruit Tree upper bounds and the figures.  The runs are serial with one numerical thread;
# the upper bounds are computed afterwards with several processes.  CPU-time repeats (optional):
# python scripts/time_repeats.py fishwood fruittree_d6 --repeats 5, before the upper bounds and the figures.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY scripts/run_uniform.py fishwood
$PY scripts/run_surf.py fishwood
$PY scripts/run_adaptive.py fishwood
$PY scripts/run_uniform.py fruittree_d6
$PY scripts/run_adaptive.py fruittree_d6
$PY scripts/upper_bounds.py fruittree_d6
$PY scripts/make_figure.py fishwood
$PY scripts/make_figure.py fruittree_d6
$PY scripts/make_bounds_grid.py fruittree_d6
