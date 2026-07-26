# K=6 FAST trial (Gram + Momentum-SVRG, eps=0.001, without-256 track)

> **PARAMETER-ITERATION NOTE (July 15, added after the run):** this folder is the
> FIRST parameter set (plan defaults: b=1024, beta=0.9, c=0.1).  Its tail
> saturated at best-so-far GN ~0.774 — a variance floor: 2% sampling plus
> heavy-ball noise amplification 1/(1-beta)=10x, diagnosed exactly by the plan's
> tuning table (safeguard silent at L_scale=1.0, 148/150 rounds at the segment
> cap, oscillating tail).  **Superseded by `..._v2`** (b=4096 -> variance /4,
> beta=0.5 -> amplification 2x; per-segment cost unchanged since p_seg=ceil(n/b)
> drops 49 -> 13).  v2 reaches 0.1526 ~= the original method's 0.1473 at ~8.5x
> less wall time.  This folder is kept as the tuning record.

Produced by `Original_py/run_trial_K6_fast_without_256_checkpoints.py`
(July 15, 2026).  The module docstring is the full specification; the
July 15 plan document (Desktop, ZH/EN) is the design reference.  All
`_fast` code files are NEW; the original files are untouched.

## What ran

The ACCELERATED adaptive method (`algorithm_adaptive_fast`), all four
plan items live:

1. **Gram-path λ-search** — exact rewrite GN(λ)=min_i λ^T M_i λ;
   measured λ-search share this run: **432.3 s
   of 1399.1 s wall (30.9%)** vs ~95% in the
   July 11 original run.
2. **Two-tier λ-search + stop-verify** — implemented and sanity-covered;
   THIS run used `lambda_tier_mode="strict"` (all-strict,
   64 starts): after the Gram rewrite the strict tier is sub-second per
   round, and all-strict keeps the self-reported metric exactly on the
   legacy 64-start yardstick.  Tier counts: {'strict': 150}.
3. **Momentum-SVRG inner loop** — segments of stratified minibatch
   variance-reduced heavy-ball steps (b=1024,
   p_seg=ceil(n/b), c=0.1,
   beta=0.9, rho=0.7,
   consec=2,
   max_segments=10); every bundle admission and
   every eps/3 acceptance test runs on FULL gradients (randomness cannot
   fake a certificate).
4. **Delivery-time pruning** — bundle 1485 -> 888
   points by lambda-activation on the r=10 simplex grid
   (+ final search winners); probe GN values bitwise unchanged.

Outcome: stop_reason=`round_fuse`, final self-reported best-so-far
GN = **7.7401e-01**, grad-equivalents used = 26775
(joint calls 1484, minibatch IFO 148922368),
wall 1399.1 s.  L_scale_final=1.0,
inner_cap_hits=148.

## Axes and accounting

Gradient axis = GRAD-EQUIVALENTS: one joint-oracle call = K (its n
per-sample gradients cover the K disjoint per-class losses once); one
minibatch Momentum-SVRG step = 2b·K/n.  The reused curves' axis is the
same unit (all their steps are full joint calls), so the axes compare
directly.  CPU axis: checkpoint bookkeeping excluded, as everywhere on
this track.

## Reused reference curves (disclosed)

Baseline (r=10) and ORIGINAL adaptive curves come from
`output/trial_K6_d11910_h96x96_tanh_n50000_B180180_without_256_checkpoints/summary.json` (July 11/12), NOT re-run: same problem
instance and x0 (seeds 7/8),
same fuse (max_outer=150), same 64-start self-reported metric.  The old
adaptive run was budget mode (epsilon=None); with epsilon=0.001 the
trajectory over the recorded range is identical (its GN never went below
0.147 >> 2eps/3 = 6.7e-4, so neither epsilon test could fire) — and the
budget-mode CPU axis, if anything, flatters the ORIGINAL method (epsilon
mode would have charged it an extra per-step GN check).  CPU comparison
is cross-run on the same machine; the July 11 folder logs machine load
and an oracle-pace calibration within 6.5%.

## Headline comparisons (self-reported meters)

vs baseline: target GN 0.7740 reached by
baseline at 2226.3945167064667 s /
94500.0 grads, by fast adaptive at
1385.0941870212555 s / 26594.26176
grad-equivalents (ratios: CPU 1.6073957551540143,
grads 3.5533981297475203).

vs original adaptive: target GN 0.7740
reached by original at 811.399516582489 s /
5400.0 grads, by fast at
1385.0941870212555 s / 26594.26176
grad-equivalents.  (In `time_to_target_vs_original_adaptive` the
"baseline_*" fields refer to the ORIGINAL adaptive method — the helper's
first-slot naming.)

## Figures

- `gn_vs_grad_evals_baseline_vs_fast.png`, `gn_vs_cpu_time_baseline_vs_fast.png`
- `gn_vs_grad_evals_adaptive_orig_vs_fast.png`, `gn_vs_cpu_time_adaptive_orig_vs_fast.png`

CPU figures: log time axis; vertical line = equal-budget point.

## Caveats, stated once

Self-reported meters (baseline never looks between its grid nodes; the
adaptive values are heuristic lower bounds of an NP-hard max — the
lambda-search is a maximiser, so an under-searched value UNDERSTATES the
criterion).  Single instance, seeds 7/8;
single machine; cross-run CPU comparison for the reused curves.  The
Momentum-SVRG inner guarantee is expectation-type; all acceptance tests
are exact (full gradients).  Inner rounds that hit the segment cap void
the Algorithm-2 termination argument for those rounds
(inner_cap_hits=148; warning recorded).
