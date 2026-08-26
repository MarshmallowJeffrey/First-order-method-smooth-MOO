# Lambda adaptivity at K=3: uniform grid vs adaptive weight sequence

Produced by `Original_py/run_lambda_path_without_256_checkpoints.py`
(module docstring = full specification).  One adaptive CERTIFIED run
(epsilon=0.001, stop at own-search value <= 2*eps/3, the
30000-gradient budget kept as the fuse) on the plateau
testbed: K=3, p=6, n=30,
hidden=[4], tanh, seeds 7/8, max_inner=25,
lambda_max_starts=64 — otherwise the
configuration of `output/plateau/K3_p6_n30_h4_tanh_r6_B30000`.
Outcome: adaptive: eps=0.001, CERTIFIED at 13155 grads (of the 30000 fuse), 180 rounds, final search value 6.57e-04.

- `lambda_path_triangle.png` — the K=3 weight simplex as a triangle.
  Hollow blue circles: the baseline's weights (uniform grid,
  r=10, 66 nodes) — fixed by (K, r) before
  seeing the problem, so no baseline run is needed or performed.
  Orange dots: the weight the adaptive method's max-min search selected
  at each outer round (`lambda_history`; star = round 1, X = final
  round).  Consecutive-round movements are drawn in two styles: L1 step
  below 0.2 = thick orange segment (a LOCAL CHAIN — the
  worst weight slid to a neighbour), L1 step at or above
  0.2 = thin grey line (a GLOBAL JUMP — the budget
  teleported to a new worst region).
- `lambda_step_sizes.png` — the L1 distance between consecutive weights
  per round (orange bars = chain steps, grey = jumps), with the grid
  spacing 1/r and the chain threshold as reference lines, a rolling
  median, and each quarter's mean — the shrinking-movement trend the
  analysis quotes.
- `lambda_path_quarters.png` — the sequence split into four consecutive
  quarters, one small triangle each: coverage fills in and movement
  becomes local over time.  Panel-by-panel analysis:
  `QUARTERS_ANALYSIS.md` (English) / `QUARTERS_ANALYSIS_ZH.md` (Chinese).
- `summary.json` — the run record, the full `lambda_history`, and
  consecutive-step statistics (L1 distances between successive weights;
  the simplex L1 diameter is 2, the r=10 grid spacing is
  0.1): median
  0.475, mean
  0.615, max
  2.000, share below 0.2:
  23%.
- `EXPLANATION_ZH.md` — why the adaptive weights are non-uniform and in
  what sense they do / do not form a continuous line (mechanism
  write-up, Chinese).

Caveat: each round's weight is the argmax found by the method's own
64-start search (a heuristic maximiser); the
without-256 track only changes what checkpoints record, not which
weights the search selects.
