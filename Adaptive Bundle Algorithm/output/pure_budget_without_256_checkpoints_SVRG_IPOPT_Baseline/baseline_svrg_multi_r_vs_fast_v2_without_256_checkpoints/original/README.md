# Baseline r-sweep with the v3 Momentum-SVRG inner solver (without-256 track)

Produced by `Original_py/run_baseline_svrg_r_sweep_without_256_checkpoints.py`
(July 20, 2026).  Design record: `Note/Jul_20_note.md`.  Engine:
`Original_py/baseline_svrg_certified_without_256_checkpoints.py` (new
file; originals untouched).

Each r is ONE run of Algorithm 1 on the uniform grid at resolution r,
with the SAME inner solver, batch (b=4096), momentum,
step rule, descent safeguard and gradient-equivalent accounting as the
v3 fast trial (same instance, seeds 7/8).  Per-node
service certificate: some delivered point x has lambda^T G(x) lambda <=
node_tol = 0.02 (exact, full-gradient Grams — randomness can
never fake a certificate).  Each solved node is pushed to solve_target =
0.005 (= 0.25 x
node_tol, mirroring v3's rel_target=0.25).  share_mode =
`gram`: delivered Grams serve all still-unserved nodes by
cached lambda-reweighting (zero oracle calls — the same cache discipline
the fast method's lambda-search runs on).

Scatter y-value = GN* of the DELIVERED point set from ONE strict-tier
64-start in-family lambda-search at delivery time (metric cost excluded
from both axes; NOT the external 256-start yardstick — the track rule
holds).  x-value = the run's total grad-equivalents / wall seconds.

Fast adaptive curve reused unchanged from `/Users/shirch/vscode101/.venv/First-order-method-smooth-MOO/Adaptive Bundle Algorithm/output/fast_method_trials/v3_rel_target_two_tier` (disclosed;
its plotted values are the run's own CHEAP-tier searches — an
under-search of a maximiser can only under-report, per its README; the
user chose to keep that meter as-is on July 20).

## Results so far

| r | N nodes | delivered GN* (strict, full simplex) | grid cert end | grad-equivalents | wall s | solved | served-by-share | censored | stop |
|---|---------|--------------------------------------|---------------|------------------|--------|--------|-----------------|----------|------|
| 10 | 3,003 | 1.6352e-01 | 1.9911e-02 | 41327 | 599 | 2879 | 3003 | 0 | completed |
| 12 | 6,188 | 9.5415e-02 | 2.0000e-02 | 64428 | 986 | 4441 | 6188 | 0 | completed |
| 15 | 15,504 | 5.9456e-02 | 1.9997e-02 | 80912 | 1173 | 4758 | 15504 | 0 | completed |
| 20 | 53,130 | 6.3415e-02 | 1.9999e-02 | 241721 | 3689 | 11820 | 53130 | 0 | completed |

Dotted vertical lines on the figures: the no-reuse Algorithm-1 floor
(>= one full joint call per node, N(r) x 6 grad-equivalents resp.
N(r) x 0.1384 s at the July-11 measured pace) — what the
paper-faithful per-node-oracle baseline would pay BEFORE any solving.

## Figures

- `gn_vs_grad_evals_baseline_r_sweep_vs_fast.png`
- `gn_vs_cpu_time_baseline_r_sweep_vs_fast.png`

Each r is drawn as a trajectory LINE plus one endpoint SQUARE.  Both
are on the COMPARABLE meter (July 25 fix, Note/Jul_25_note.md §6): the
line is the strict full-simplex 64-start GN* of the delivered-set
prefix at each checkpoint, computed post-hoc on cached Grams (cost in
`metric_seconds`, excluded from both axes); the square is its final
value.  This is the same meter family as the fast curve's y-axis, so
the lines share axes without cross-meter misreads.  The run's own GRID
meter (max over grid nodes of best-known value — the July-8 lag
semantics) is NOT plotted here; it is kept in each summary
(`cov_history`) and its endpoint appears in the table above as "grid
cert end".  The gap between "grid cert end" and "delivered GN*" is the
measured between-node error of that grid.  Open squares (if any) mark
fused or censored runs: their point is a lower bound on that r's cost
at this tolerance, not a converged measurement.

## Caveats

Single instance, seeds 7/8; single machine;
delivered points are re-derivable exactly (deterministic seeds) but not
stored (memory).  The SVRG inner guarantee is expectation-type; every
certificate value is exact (full-gradient Grams).  Fresh sampler
(seed 41) per r: runs are independently reproducible.
