#!/bin/sh
# Reproduces all runs and figures (serial, one numerical thread).  Baselines first: the adaptive budget is
# computed from the farthest plotted baseline point.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for task in fishwood dst; do
  $PY scripts/run_uniform.py $task
  $PY scripts/run_surf.py $task
  $PY scripts/run_adaptive.py $task
  $PY scripts/make_figure.py $task
done
for task in bb fruittree_d6; do
  $PY scripts/run_uniform.py $task
  $PY scripts/run_adaptive.py $task
  $PY scripts/make_figure.py $task
done
