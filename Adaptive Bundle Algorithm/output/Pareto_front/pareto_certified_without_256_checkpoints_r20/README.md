# Certified-mode Pareto-front comparison (K=2, without-256 track)

Everything here was produced by
`Original_py/run_pareto_certified_without_256_checkpoints.py`; the module
docstring there is the full specification. Summary of what this folder is:

- Three certified runs on ONE shared problem instance
  (K=2, p=20, n=50000, hidden=[96, 96], tanh,
  seeds 7/8, r=20, budget 2000
  kept as the fuse; checkpoints are self-reported — no 256-start solves):
  - `baseline_eps0.01/` — Algorithm 1 in certification mode.
    eps 0.01 was split in norm space: node_tol = eps/4 = 0.0025
    for the grid nodes, the other half reserved for the r-grid's
    between-node degradation (h*D with h = 1/(2r) = 0.025).
    Outcome: node certification FAILED within 2000 grads (node_tol=0.0025).
  - `adaptive_eps0.01/`, `adaptive_eps0.001/` — Algorithm 2 with epsilon
    set; stops when its own lambda-search value <= 2*eps/3.
    Outcomes: certified eps=0.01 at 132 grads, m=17 /
    certified eps=0.001 at 700 grads, m=71.
- Post-run audit of the baseline's all-lambda certificate (exact for K=2):
  for every lambda the nearest node's point satisfies
  ||grad F_lambda|| <= ||grad F_lambda_i|| + h*||g1-g2||, so the certified
  level is max_i (sqrt(node value) + h*D_i)^2.
  Measured: D_max = 6.170, grid share alone (h*D_max)^2 =
  0.02379, overall audited level =
  3.805
  (does NOT certify the
  eps 0.01 target). Unserved nodes at stop: 16.
  If (h*D_max)^2 alone exceeds eps, NO node_tol can reach eps at this r —
  that is the grid floor in certified-mode language; the remedy is a finer
  grid (larger r), not a smaller node_tol.
- `pareto_front_combo1_bl0.01_a0.01.png`, `pareto_front_combo2_bl0.01_a0.001.png`
  — the requested figures. Axes: F1 and F2 (per-class cross-entropy) at
  the point each method DELIVERS for weight lambda=(t,1-t); the same
  delivery rule is used for both methods (argmin over the method's point
  set of ||grad F_lambda||^2, computed from stored/audited gradients —
  no re-optimisation). Lines trace the delivered point as t sweeps 0..1
  (1001 values); markers are the distinct delivered points,
  coloured by the mean t they serve. Lower-left is better (both losses
  smaller); a curve that lies below-left of the other dominates it.
- `pareto_data.json` — machine-readable fronts (t-segments per delivered
  point with F1/F2), the full node-by-node certificate audit, and labels.

Caveats, stated once: the adaptive stop rule trusts its own
8-start search (a heuristic lower bound of an
NP-hard max), so "certified" means "its own working search could not find
a weight above 2*eps/3" — same search strength it uses to run at all.
prune_inner=True voids the epsilon-mode convergence-TIME proof condition
(a RuntimeWarning is recorded in the summaries); the stopping rule itself
is unaffected. Wall-clock on this track is not comparable to the
256-start track.
