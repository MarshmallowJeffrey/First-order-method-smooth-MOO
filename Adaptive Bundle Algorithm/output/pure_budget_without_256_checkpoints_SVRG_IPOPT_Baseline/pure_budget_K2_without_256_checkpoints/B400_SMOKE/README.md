# Pure fixed-budget comparison at K = 2 (exact 1-D meter; without-256 track)

Produced by `Original_py/run_pure_budget_K2_without_256_checkpoints.py`
(July 30, 2026; change record `Note/Jul_30_note.md`).  Protocol as the
K = 6 July-27 experiment: every leg spends the SAME budget
B = 400 grad-equivalents in identical work units (one
segment = 5 minibatch Momentum-SVRG steps, no trigger, + 1
full joint evaluation), s = 2 consecutive segments per
allocation decision, chain warm start for BOTH methods, stop = budget.
The ONLY difference is the next-lambda policy: snake grid order
(baseline, per r; K = 2 grid has r + 1 nodes) vs strict worst-lambda search (adaptive, 8-start IPOPT legs, warm-started — same mechanism as the K = 6 runner; decision time inside its CPU axis).

Quality meter: EXACT at K = 2 — GN(w) = min over delivered points of a
convex quadratic in w; evaluated on a 20,001-point w-grid
with closed-form polish of the winning neighbourhood, and CERTIFIED by
a per-cell slope bound: every audit stores a true-value lower bound
AND a proven upper bound (summary keys `audited_gn_history` /
`audited_gn_upper_history`; `audit_certified_gap_max` is the widest
interval of the leg), so "exact" is a checkable statement per run, not
a label.  No multistart search anywhere in the MEASUREMENT — the
adaptive's own targeting search is method-internal navigation whose
misses can only hurt its own efficiency, never the reported quality —
hence no search-limited under-reporting (the eps1e-4 lesson).  Exact
prefix audits are monotone by mathematics (asserted).  Audits are off
both cost axes; adaptive targeting is on-axis method work.

| leg | s | distinct lambdas visited | grad-equiv | wall s | exact audit |
|-----|---|--------------------------|------------|--------|-------------|
| baseline r=4 | 1 | 5 | 394 | 0 | 1.7759e-01 |
| baseline r=4 | 2 | 5 | 396 | 0 | 7.8232e-02 |
| adaptive 8-start | 2 | 17 | 396 | 16 | 9.4879e-03 |

(baseline "distinct lambdas" = grid nodes visited; at K = 2 every leg
covers its full grid many times over.)

## Discovered fronts (final budget, s = 2 legs)

Delivered set = every segment endpoint's full-batch (F1, F2); front =
its nondominated subset; reference = union of all plotted fronts (no
oracle front exists for this family).  IGD/max-dist are distances from
the union front to each method front (raw value-space Euclidean).

Metrics come in TWO variants because the raw union front is dominated
by degenerate SPECIALIST TAILS: grid legs whose vertex nodes (w = 0 or
w = 1) camp on a single-class objective for many passes drive that
class's loss toward 0 while the other explodes (F1 up to ~27 here) —
nondominated by construction, but not a genuine trade-off.  A
GN-steered method stops visiting a region once it is near-stationary
and never manufactures such tails.  So next to the raw metrics, the
"central" variant restricts the REFERENCE front to the region where
both losses are <= 1 (method fronts are never clipped); the
figure's main panel shows exactly this region, with the tails in the
full-range inset.

"Certified eps (U)" is each leg's final certified UPPER bound on
GN* = max over w of [min over its delivered points of the
lam(w)-weighted gradient-norm^2]: for EVERY trade-off weight w in
[0, 1], the leg's delivered set PROVABLY contains a point whose
weighted stationarity measure is <= U.  This is the epsilon of
"epsilon-Pareto front" in the stationarity sense — a positive claim
only an upper bound can sign (a search value is a lower bound and
cannot; the eps1e-4 false-certificate lesson).

| leg | delivered pts | front pts | certified eps (U) | IGD raw | max-dist raw | IGD central | max-dist central |
|-----|---------------|-----------|-------------------|---------|--------------|-------------|------------------|
| baseline_r4 | 77 | 47 | 7.8232e-02 | 4.5280e-02 | 4.2241e-01 | 1.0302e-02 | 8.3657e-02 |
| adaptive_ts8 | 67 | 45 | 9.4879e-03 | 1.4098e-02 | 6.5759e-02 | 1.7150e-02 | 5.6366e-02 |

`grams.npz` per leg additionally stores `seg_grads` / `seg_lams`
(cumulative grad-equivalent spend and lambda per segment), so the
delivered set and its front can be reconstructed at ANY prefix budget.

CPU-axis caveat: adaptive decision time is method work, on-axis;
audits are off-axis for everyone.  MLP torch runs are not
bit-reproducible in this environment (session-12 finding); one
realization each.
