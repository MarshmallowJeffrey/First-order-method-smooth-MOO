# Pure fixed-budget comparison at K = 2 (exact 1-D meter; without-256 track)

Produced by `Original_py/run_pure_budget_K2_without_256_checkpoints.py`
(July 30, 2026; change record `Note/Jul_30_note.md`).  Protocol as the
K = 6 July-27 experiment: every leg spends the SAME budget
B = 20,000 grad-equivalents in identical work units (one
segment = 13 minibatch Momentum-SVRG steps, no trigger, + 1
full joint evaluation), s = 5 consecutive segments per
allocation decision, chain warm start for BOTH methods, stop = budget.
The ONLY difference is the next-lambda policy: snake grid order
(baseline, per r; K = 2 grid has r + 1 nodes) vs strict worst-lambda search (adaptive, 24/64-start IPOPT legs, warm-started — same mechanism as the K = 6 runner; decision time inside its CPU axis).

Quality meter: EXACT at K = 2 — GN(w) = min over delivered points of a
convex quadratic in w; evaluated on a 200,001-point w-grid
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
| baseline r=10 | 5 | 11 | 19999 | 680 | 1.6071e-04 |
| baseline r=20 | 5 | 21 | 19999 | 547 | 6.2015e-04 |
| baseline r=40 | 5 | 41 | 19996 | 504 | 2.7010e-04 |
| baseline r=80 | 5 | 81 | 19997 | 490 | 2.9811e-04 |
| adaptive 24-start | 5 | 617 | 19994 | 1141 | 9.5186e-05 |
| adaptive 64-start | 5 | 617 | 19994 | 1093 | 9.5186e-05 |

(baseline "distinct lambdas" = grid nodes visited; at K = 2 every leg
covers its full grid many times over.)

## Discovered fronts (final budget, s = 5 legs)

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
both losses are <= 1 (method fronts are never clipped).  The
figure (SURF Fig-6 style, user revision Jul 30) shows the per-method
nondominated fronts ONLY, on log-log axes — the knee lives below 1e-2
on this instance, so linear axes cannot show the fronts.

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
| baseline_r10 | 3,405 | 35 | 1.6072e-04 | 2.0987e-01 | 4.2222e+00 | 1.9520e-02 | 2.4700e-01 |
| baseline_r20 | 3,303 | 42 | 6.2024e-04 | 7.4822e-01 | 7.7403e+00 | 1.2630e-01 | 9.0874e-01 |
| baseline_r40 | 3,248 | 62 | 2.7018e-04 | 4.1985e-01 | 2.3476e+00 | 1.2000e-01 | 9.3025e-01 |
| baseline_r80 | 3,221 | 107 | 2.9829e-04 | 3.0141e-01 | 3.3488e+00 | 4.5426e-02 | 2.1330e-01 |
| adaptive_ts24 | 3,195 | 88 | 9.5286e-05 | 1.0180e+00 | 9.4260e+00 | 4.3088e-02 | 1.7838e-01 |

SURF-paper-style metrics (their Table 1: HV / IGD / CV / Gap Ratio),
computed on each method's front restricted to the central region, with
the central corner (1, 1) as the hypervolume reference point
(conventions disclosed here; SURF's originals are computed on bounded
RL fronts, ours collapse toward the origin, hence the central
restriction).  HV higher = better; CV and Gap Ratio lower = more
uniform spacing (SURF's headline axis).  Single realization — no
mean/std yet (SURF reports 8 seeds).

| leg | central front pts | HV (ref (1,1)) | CV | Gap Ratio |
|-----|-------------------|--------------------------|----|-----------|
| baseline_r10 | 27 | 9.9862e-01 | 4.4327e+00 | 2.0734e+05 |
| baseline_r20 | 33 | 9.9737e-01 | 1.6651e+00 | 8.0611e+03 |
| baseline_r40 | 55 | 9.9705e-01 | 2.3895e+00 | 2.3205e+04 |
| baseline_r80 | 101 | 9.9300e-01 | 2.9784e+00 | 5.9796e+04 |
| adaptive_ts24 | 78 | 9.9382e-01 | 4.1261e+00 | 3.4035e+04 |

`grams.npz` per leg additionally stores `seg_grads` / `seg_lams`
(cumulative grad-equivalent spend and lambda per segment), so the
delivered set and its front can be reconstructed at ANY prefix budget.

CPU-axis caveat: adaptive decision time is method work, on-axis;
audits are off-axis for everyone.  MLP torch runs are not
bit-reproducible in this environment (session-12 finding); one
realization each.
