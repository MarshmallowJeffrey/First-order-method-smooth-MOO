# Jul 20 note — SVRG-in-baseline + baseline r-sweep (scatter figure)

User request (Jul 20): (a) put the SVRG machinery into the baseline so
both methods use the same inner solver ("same algorithm on both sides,
fair comparison"); (b) replace the baseline curve by SCATTER POINTS,
one per grid resolution r, against the unchanged v3 fast curve.
Requested r list: 10, 20, 30, 40, 50 (fallback 10, 20, 25, 30, 35 if
too slow).  Constraints: new code files only (originals untouched),
results under output/, still the without-256-checkpoints track, this
note records the design.  User separately decided (Jul 20): do NOT
re-meter the v3 fast curve (the "1.5" strict-on-checkpoint-rounds fix
stays unused); its cheap-tier plotted values are kept as-is, caveat
disclosed on every figure/README.

## 1. Problems found when re-checking the Jul 19 plan against the
##    concrete r list

1. **r >= 30 is infeasible under the literal per-node design.**  The
   per-node floor is one full joint-oracle call (0.1384 s measured,
   July 11 log: 180,180 grads / 4,157.5 s).  N(r) = C(r+5,5) at K=6:
   r=30 -> 324,632 nodes -> 12.4 h FLOOR (zero solving); r=40 -> 46.8 h;
   r=50 -> 133.4 h.  The fallback list dies the same way (r=35 ->
   25.2 h).  Additionally the July-8 baseline's memory layout (one
   solution vector per node) needs N x d x 8 bytes = 31 GB at r=30 on a
   16 GB machine.
2. **SVRG alone cannot fix this.**  A faster inner solver cuts the
   per-node SOLVE cost but not the per-node VISIT floor (one oracle
   acceptance per node).  Only reusing cached Jacobians across lambda
   can — which is exactly what the fast method's own lambda-search does
   with its Gram cache.  Denying the baseline that same cache discipline
   while the adaptive method lives on it would be unfair in the
   OPPOSITE direction from the unfairness the user is trying to remove.
3. **The scatter needs a real y-metric.**  A certified run's grid-max is
   <= node_tol BY CONSTRUCTION — plotting it would plot our own knob.
   The informative y is the delivered set's GN* over the FULL simplex,
   measured once at delivery by the method-family's own strict-tier
   64-start search (in-family, NOT the external 256-start yardstick;
   the track rule is intact).  Between-node error is exactly what this
   exposes, and it shrinks with r — that is the figure's story.
4. **A single-level tolerance would disable sharing.**  If nodes are
   solved exactly to node_tol, a delivered point serves (worst-case)
   only its own node.  v3's own evidence: points pushed to
   rel_target=0.25 of the current criterion covered the whole simplex
   at 0.058 with ~500 points.  Hence two levels: solve_target =
   0.25 x node_tol (same constant as v3), service at node_tol.
5. **Boundary nodes over-pay in the naive accounting.**  ~96% of grid
   nodes at r=10 have zero coordinates; sampling rows of zero-weight
   classes adds IFO for terms that are exactly zero.  Fix: drop those
   rows BEFORE the oracle call (estimator unchanged, bit-for-bit; the
   oracle already charges by rows actually consumed).

## 2. Design implemented (new files, originals untouched)

* `Original_py/baseline_svrg_certified_without_256_checkpoints.py` —
  engine.  Algorithm 1 on the snake-ordered uniform grid; per unserved
  node, the v3 inner loop verbatim: segments of stratified minibatch
  Momentum-SVRG (b=4096, epoch = ceil(n/b) = 13, step_const=0.1,
  beta=0.5, rho=0.7, patience=2, max_segments=10, descent safeguard
  with L_scale doubling and <= 4 retries), epoch-end FULL joint call
  (charged) whose Jacobian/Gram is the acceptance test, the next
  anchor, and the delivered point.  Certificates are deterministic:
  node lambda served iff some delivered point x has
  lambda^T G(x) lambda <= node_tol on stored full-gradient Grams
  (randomness affects only runtime — the Las-Vegas structure of the
  fast method, replicated).
* Share modes: `gram` (default) — every delivered Gram is swept against
  all unserved nodes (vectorised lambda^T G lambda, zero oracle calls);
  `none` — only the chain warm-start point may serve a node at visit.
  `gram` is the headline mode; it is "Algorithm 1 + memoisation", the
  strongest honest reading of "same machinery on both sides".  The
  no-reuse Algorithm-1 floor (N(r) full joint calls) is drawn on the
  figures as dotted verticals — the paper-faithful variant's
  lower bound, no run needed.
* Accounting identical to fast: grad_equiv = joint_calls x K +
  IFO x K/n; x0 Jacobian uncharged (mirrors fast checkpoint-0); grid
  construction excluded from CPU (preprocessing); serve-sweeps INCLUDED
  in CPU (they are the baseline's service bookkeeping); delivery-time
  strict search + certificate verification excluded, timed separately
  (`metric_seconds`).
* `Original_py/run_baseline_svrg_r_sweep_without_256_checkpoints.py` —
  sweep runner.  Same instance/seeds as v3 (K=6, p=20, n=50000,
  h=[96,96], tanh, seeds 7/8), fresh StochLamOracle per r (seed 41,
  independently reproducible).  Per-r fuses: wall 4 h, grads 2M
  (fused/censored r plotted as OPEN squares = cost lower bound, never
  read as converged).  Resumable (skips r with existing summary).
  Figures + sweep_summary.json + README/README_ZH rewritten after each
  completed r.  Output:
  `output/baseline_svrg_r_sweep_without_256_checkpoints/`.
* Delivered point coordinates are NOT stored (at scale they are
  segments x d floats); certification/scoring live on K x K Grams.
  Deterministic seeds make the set exactly re-derivable; engine flag
  `return_points=True` exists for small runs.
* Counter semantics: `served_by_share` counts sweep-marked nodes
  including a solved node's self-service; `n_served` is the ground
  truth.  `served_above_target` = segment cap hit but service
  certificate still holds (the analogue of v3's cap_hits caveat:
  the termination argument does not cover those nodes' solves, the
  certificates are unaffected).

## 3. Why SVRG and not SGD (part-2 question, recorded)

Constant-step SGD converges to an O(eta sigma^2) noise ball, so the
per-node acceptance test may never fire — no certificate, ever.
Decaying-step SGD is O(1/eps^2) IFO, worse than full GD's O(n/eps) at
eps < 1/n.  Nonconvex SVRG gives O(n + n^{2/3}/eps) with an exact
full-gradient acceptance available at every epoch boundary for free
(the snapshot gradient is computed anyway).  Same conclusion as the
Jul 19 discussion; implemented accordingly.

## 4. Expected outcome, stated before running

Making the baseline stronger should SHRINK the headline ratios
(currently 138x CPU / 171x grads vs the July-11 GD baseline).  That is
the point: the surviving gap isolates the adaptive lambda-selection +
bundle reuse (the paper's actual contribution) instead of mixing in the
inner-solver difference.  The r-scatter adds the second axis: quality
attainable per budget as the grid refines, with the no-reuse floor
showing the combinatorial wall the memoised variant sidesteps.

## 5. Runtime model (basis: July-11 joint pace 0.1384 s; v3 minibatch
##    pair 13.9 ms; one epoch ~ 0.32 s)

r=10: sharing ~nil at spacing 0.2 -> ~3,003 solves x 1-4 epochs ->
**~20-70 min**, ~60k-230k grad-equivalents.  r >= 20: cost is driven by
the SOLVED-node count (a covering quantity), not N; forecast refined
from the r=10/r=20 measurements; 4 h fuse per r caps the worst case.
Full sweep estimate: **realistically 3-12 h, hard-capped at 4 h x
number of r values**; runs sequentially, results land incrementally.

## 6. Execution log

* Smoke (tiny instance, r in {2,3}, both share modes' machinery,
  certificate verification on): PASSED, ~5 s total.
* r=10 leg launched Jul 20 (background, caffeinate -i to prevent idle
  sleep).  r in {20,30,40,50} held pending the user's go after the
  r=10 read-out, per the standing launch-approval convention.

## 7. r=10 read-out (Jul 20, completed)

* FULLY CERTIFIED: 3003/3003 nodes, censored 0, verification sweep
  passed (worst served value 0.019911 <= node_tol 0.02).  No safeguard
  retries, L_scale stayed 1.0, no runtime warnings.
* Cost: 41,327 grad-equivalents / 636 s wall (metric: 5.2 s, excluded).
  2.29x the no-reuse floor (18,018).  3,154 segments for 2,879 solves
  (1.10 seg/solve), 10.8 minibatch steps/segment (early-stop bites),
  0.202 s/segment measured — FASTER than the 0.319 s model (support-
  restricted batches + lighter machine load).
* Quality: delivered strict GN* = 0.16352 (argmax lambda concentrated
  on classes 5/6).  Grid-max was 0.0199 — the full-simplex sup is 8.2x
  the grid certificate.  The between-node gap is now MEASURED, not
  argued: this is exactly problem 3.1's mechanism, and the reason the
  scatter must be scored on the full simplex (Sec. 1 problem 3).
* Sharing at spacing 0.2: 124/3003 nodes (4%) served by sharing —
  near-nil, as predicted; the r=10 point therefore doubles as the
  honest no-sharing anchor.
* Headline vs fast (both on delivered/self-reported meters, green curve
  cheap-tier as the user directed): fast reached 0.163-quality at
  2,409 grad-equivalents / 68.1 s; baseline-SVRG r=10 pays 41,327 /
  598.6 s for the same level -> **17.2x grads, 8.8x CPU**.  Down from
  171x/138x vs the July-11 GD baseline — the shrink is the intended
  effect (Sec. 4): what remains is attributable to adaptive lambda
  selection, not the inner solver.
  [CORRECTED 2026-07-25: this bullet first read "~40 s" and "~16x CPU".
  Both were wrong — the fast checkpoint that first reaches 0.1635 sits
  at t=68.1 s, not ~40 s, and the baseline side must use the cost-axis
  wall_seconds 598.6 s, not the 636 s end-to-end time (which includes
  instance build + the excluded metric solve).  The grad ratio 17.2x
  was and is correct.]
* Refined forecast from the measured pace (0.202 s/segment, sharing
  rate unknown above r=10): r=20 (N=53,130) ~1.5-3 h if sharing
  reaches 20-50% at spacing 0.1; r=30 (N=324,632) likely NEAR OR OVER
  the 4 h fuse; r=40/50 expected to hit the fuse unless sharing grows
  super-linearly with density.  Fused points are still valid scatter
  entries (open squares = cost lower bound at that tolerance).
  Decision pending user: proceed {20,30,40,50} under the 4 h fuse.
