# Discovered epsilon-Pareto fronts (K=6, pure fixed budget B=80,912)

Produced by `Original_py/front_metrics_K6_pure_budget_without_256_checkpoints.py`
(July 30, 2026; change record `Note/Jul_30_note.md`).
Pure post-processing of the July-27 legs' stored `fvals` (the
full-batch objective vector of every delivered point); nothing was
re-run.  Delivered set = every segment endpoint (+ x0); front = its
nondominated subset (minimisation).

Metric set follows the SURF paper's Table 1 (HV, IGD) with the
deviations this track's record requires:

* No front FIGURE in value space: 6-D fronts admit no direct plot —
  the K=5 bandit recorded the same ("front_uniformity omitted ... no
  canonical 1-D ordering"), and the SURF paper itself only ever plots
  M = 2 fronts.  The figure here shows the coverage-gap CDF (the
  distribution whose mean is IGD and max is max-dist) and central-box
  hypervolume bars.
* Reference front = union of all legs' fronts (mutual reference; no
  oracle front exists for this family; SURF likewise omits IGD when no
  dense reference is available, their Table 2).
* CENTRAL variant: reference restricted to all six losses <=
  1 (genuine trade-off region; x0 sits at ~2.2 per objective).
  The raw union front is dominated by specialist tails (losses up to
  ~25), exactly the K=2 runner's rationale; method fronts are never
  clipped.
* HV is Monte-Carlo estimated (100,000 common samples,
  seed 20260730, paired across legs) as the dominated fraction
  of the box [ideal, 1]^6, ideal = per-objective minimum over all
  fronts.  A raw-box HV is deliberately not reported (tail-volume
  dominated).
* CV / Gap Ratio (SURF's spacing metrics) omitted — 1-D ordering
  does not exist at K=6.
* One realization per leg, no error bars across runs (MLP torch runs
  are not bit-reproducible in this environment; session-12 finding).

**eps labels are search LOWER bounds.**  Each leg's eps is its final
strict 64-start delivered-set audit: the best FOUND value
of GN* = max over lambda in Delta_6 of min over delivered points of
lambda' M lambda.  A search value can never sign the positive claim
"GN* <= eps" (the bandit eps1e-4 false-certificate lesson); the
certified two-sided meter exists only at K=2, where the 1-D structure
admits exact evaluation.  Quote these labels only as lower bounds.

| leg | delivered pts | front pts | front pts central | HV central (95% CI) | IGD central | max-dist central | IGD raw | max-dist raw | eps (search LB) |
|-----|---------------|-----------|-------------------|---------------------|-------------|------------------|---------|--------------|-----------------|
| adaptive (s=5) | 4,309 | 3,174 | 215 | 0.0293 +- 0.0010 | 0.0420 | 0.4198 | 11.9753 | 27.9092 | 4.6160e-02 |
| baseline r=10 s=1 | 5,558 | 3,098 | 35 | 0.0270 +- 0.0010 | 0.3302 | 0.6503 | 6.5026 | 21.8419 | 1.1144e-01 |
| baseline r=10 s=5 | 5,898 | 5,751 | 0 | 0.0000 +- 0.0000 | 1.1208 | 1.5655 | 2.4737 | 11.4873 | 3.1783e+00 |
| baseline r=12 s=5 | 5,822 | 5,747 | 0 | 0.0000 +- 0.0000 | 3.5695 | 3.7626 | 2.8472 | 10.5399 | 7.0916e+00 |
| baseline r=15 s=1 | 5,432 | 5,214 | 0 | 0.0000 +- 0.0000 | 1.2169 | 1.6258 | 4.3361 | 17.4951 | 4.9037e+00 |
| baseline r=15 s=5 | 5,847 | 5,785 | 0 | 0.0000 +- 0.0000 | 3.5695 | 3.7626 | 2.5548 | 11.4950 | 7.0916e+00 |
| baseline r=20 s=5 | 5,905 | 5,843 | 0 | 0.0000 +- 0.0000 | 3.5695 | 3.7626 | 5.2297 | 15.5817 | 7.0916e+00 |

Reading the RAW columns: the raw reference contains every leg's
nondominated cloud, and most of it lies in loss regions ABOVE the
shared initialization (the collapsed legs' wandering; specialist tails
up to ~25) — nondominated by construction, not genuine trade-offs.
The adaptive leg never visits those regions (GN-steered away once a
region is near-stationary), so its raw IGD/max-dist are the LARGEST
in the table by construction; distance to above-x0 clouds is not
front quality.  The central columns are the decision-relevant ones;
raw is kept only for completeness of the mutual-reference convention.

Union front: 29,505 pts raw, 246
central; union central-box HV = 0.0398 (the attainable
envelope under this mutual reference).  Because all legs share ONE
sample set, HV differences are paired: `hv_adaptive_minus_this` in
`front_metrics.json` carries each leg's paired delta vs the adaptive
leg with its own 95% CI (tighter than the per-leg CIs suggest).

## Per-objective minimum achieved (coverage holes)

A leg that never trains an objective cannot cover that end of the
front, whatever its GN audit says.  x0 ~= 2.2 on every objective.

| leg | F1 min | F2 min | F3 min | F4 min | F5 min | F6 min |
|-----|--------|--------|--------|--------|--------|--------|
| adaptive (s=5) | 0.0029 | 0.0021 | 0.0026 | 0.0022 | 0.0049 | 0.0030 |
| baseline r=10 s=1 | 0.0055 | 0.0028 | 0.0018 | 0.0031 | 0.0050 | 0.0205 |
| baseline r=10 s=5 | 0.9446 | 0.0009 | 0.0005 | 0.0003 | 0.0009 | 0.0040 |
| baseline r=12 s=5 | 2.1990 | 0.0765 | 0.0008 | 0.0003 | 0.0008 | 0.0053 |
| baseline r=15 s=1 | 1.1859 | 0.0080 | 0.0039 | 0.0033 | 0.0059 | 0.0147 |
| baseline r=15 s=5 | 2.1990 | 1.0445 | 0.0014 | 0.0002 | 0.0007 | 0.0053 |
| baseline r=20 s=5 | 2.1990 | 1.8532 | 0.0113 | 0.0002 | 0.0006 | 0.0022 |

## The projection figures ((F1, F2) single + all 15 pairs)

`pure_budget_K6_fronts_F1F2.png` and
`pure_budget_K6_fronts_pairwise.png` draw the two FULL-COVERAGE legs
only — adaptive and r10 s1 (the `--proj-legs` default; the other five
legs render the panels unreadable, and `--proj-legs all` restores
them).  Presentation (deliberately minimal, user request): ONLY the
two legs' fronts, each drawn as its ATTAINMENT STAIRCASE — the exact
boundary of the region its delivered set dominates in that plane
(steps-post; a straight point-to-point line would overstate what is
attained between front points).  No other overlays.  The joint-front
composition per pair (how many points of the two legs' joint
nondominated front each contributes) is NOT printed on the figures
but stays recorded in front_metrics.json under
`projection_joint_front_composition` (order adaptive / r10 s1).

Reading rules and measured context (unchanged from the 7-leg view):
fronts and dominated regions are PER PROJECTION — the four objectives
off a panel's axes are unconstrained there, and the projection front
is not the projection of the 6-D front.  The omitted legs' facts
stay on record: r10 s5 clips the (F1, F2) joint corner with one
point (0.94, 0.21); r12/r15/r20 s5's leftmost (F1, F2) front point
IS x0 (class 1 never trained); as single-class specialists they do
reach low corners in F3..F6 pairs.  The projection also compresses
the coverage story — in (F1, F2) the two drawn legs hold 8 vs 9
corner points while the 6-D record shows 215 vs 35 central front
points (IGD 0.042 vs 0.330); the 6-D table above stays the
quantitative record.

Prefix-budget front cuts are NOT possible from the July-27 artifacts
(no per-segment grad ledger was stored at K=6; the K=2 runner added
`seg_grads`/`seg_lams` for exactly that) — this analysis is
final-budget only.
