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
node_tol = 0.01 (exact, full-gradient Grams — randomness can
never fake a certificate).  Each solved node is pushed to solve_target =
0.0025 (= 0.25 x
node_tol, mirroring v3's rel_target=0.25).  share_mode =
`gram`: delivered Grams serve all still-unserved nodes by
cached lambda-reweighting (zero oracle calls — the same cache discipline
the fast method's lambda-search runs on).

Scatter y-value = GN* of the DELIVERED point set from ONE strict-tier
64-start in-family lambda-search at delivery time (metric cost excluded
from both axes; NOT the external 256-start yardstick — the track rule
holds).  x-value = the run's total grad-equivalents / wall seconds.

Fast adaptive curve reused unchanged from `/Users/shirch/vscode101/.venv/First-order-method-smooth-MOO/Adaptive Bundle Algorithm/output/fast_method_trials/trial_K6_d11910_h96x96_tanh_n50000_eps0.001_fast_msvrg_without_256_checkpoints_v4_strict_rel0.1` (disclosed;
its plotted values are the run's own CHEAP-tier searches — an
under-search of a maximiser can only under-report, per its README; the
user chose to keep that meter as-is on July 20).

## Results so far

| r | N nodes | delivered GN* (strict, full simplex) | grid cert end | grad-equivalents | wall s | solved | served-by-share | censored | stop |
|---|---------|--------------------------------------|---------------|------------------|--------|--------|-----------------|----------|------|
| 10 | 3,003 | 1.4993e-01 | 9.8356e-03 | 55416 | 882 | 2987 | 3003 | 0 | completed |
| 15 | 15,504 | 5.8878e-02 | 9.9970e-03 | 254197 | 3628 | 10693 | 15504 | 0 | completed |

Dotted vertical lines on the figures: the no-reuse Algorithm-1 floor
(>= one full joint call per node, N(r) x 6 grad-equivalents resp.
N(r) x 0.1384 s at the July-11 measured pace) — what the
paper-faithful per-node-oracle baseline would pay BEFORE any solving.

## Figures

- `gn_vs_grad_evals_baseline_r_sweep_vs_fast.png`
- `gn_vs_cpu_time_baseline_r_sweep_vs_fast.png`

Each r is drawn as a trajectory LINE on the baseline's NATIVE grid
meter (July 26, user decision — reverting the July 25 §6
comparable-meter lines): at each checkpoint, the max over ALL grid
nodes of the best-known value (x0-initialised lag semantics — the
July-8 baseline metric, `cov_history`).  The line ends in a small
circle = the run's own grid certificate.  The strict full-simplex
64-start GN* of the delivered set appears ONLY as a separate
delivery-time AUDIT: an x marker joined to the circle by a dotted
vertical connector.  The vertical gap IS the measured between-node
error of that grid — the quantity the baseline's own meter cannot
see.  METER CAVEAT: the baseline line (grid nodes only) and the fast
curve (its own cheap-tier lambda-search) are DIFFERENT meters; only
the x audit is a full-simplex score.  Cross-curve reads are strict at
the audit markers, indicative elsewhere.  The strict prefix trajectory
stays in each summary (`delivered_gn_strict_history`), no longer
plotted.  Open circles (if any) mark fused or censored runs: their
point is a lower bound on that r's cost at this tolerance, not a
converged measurement.

## Caveats

Single instance, seeds 7/8; single machine;
delivered points are re-derivable exactly (deterministic seeds) but not
stored (memory).  The SVRG inner guarantee is expectation-type; every
certificate value is exact (full-gradient Grams).  Fresh sampler
(seed 41) per r: runs are independently reproducible.
