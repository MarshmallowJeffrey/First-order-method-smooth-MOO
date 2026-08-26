# July 26, 2026 — SURF offline-bandit toy: MSVRG baseline vs MSVRG adaptive

Task (user-approved design, July 25-26 session): run the SURF paper's
offline-bandit toy example (SURF Appendix F.1 == our paper Section 5.2)
through the existing Momentum-SVRG pair on the without-256-checkpoints
track and compare the generated Pareto frontiers in epsilon-accuracy and
efficiency (CPU time, grad-equivalents).

## Design decisions locked with the user

- Parameterization: reduced logits theta in R^{A-1}, pi = softmax([theta, 0]).
- Methods: `baseline_svrg_certified` (uniform grid r=11 -> 12 nodes,
  node_tol = 2eps/3, solve_target = node_tol/4, share_mode "gram") vs
  `algorithm_adaptive_fast` (strict IPOPT 64-start lambda-search,
  rel_target 0.25, prune_grid_r 11).
- Shared inner solver: b=256 (of T=1000 rows), step_const 0.1,
  momentum 0.5, epoch_len auto (=4), trigger (0.7, 2).
- Accuracy ladder: eps in {1e-2, 1e-3, 1e-4} (the 1e-4 rung re-added by
  the user on Jul 26 after the initial two-rung plan). One run per
  (method, eps) — the driver does both methods per invocation, so 3
  invocations.
- Common meter: strict in-family lambda-search applied post hoc to each
  method's checkpoint prefixes; cost off both axes (baseline already had
  `delivered_gn_strict_history` from the Jul 25 fix; the adaptive side
  is rescored by the driver with the same instrument).
- Metrics: eps_opt (GN*), eps_PF (max point-to-oracle dist, IGD),
  eps_value (scalarized objective gap vs the closed-form optimum),
  CPU time, grad-equivalents; secondary CV / Gap Ratio; statistical
  layer ||R_hat - R||_inf and ||Phi_hat - Phi||_inf.
- The closed-form softmax solution is oracle-only (never timed, never
  used by either method).

## New files (all under `Adaptive Bundle Algorithm/Original_py/`)

- `objectives_bandit_toy.py` — problem module: balanced offline dataset
  (T=1000, noise 0.5, data_seed 7), reward-mean estimates, reduced-logit
  objectives/gradients/fused joint oracle, closed-form plug-in oracle
  (pi_star, theta_star, exact PF, scalarized optima, SURF Eq. (9) speed,
  arc-length CDF and Rule-1 weights), `BanditStochOracle` (row-minibatch
  SVRG oracle, StochLamOracle-compatible surface: exact-KL part, reward
  rows stochastic, IFO = 2*rows per grad_pair, n = T), and `calibrate_L`
  (numerical Hessian bound with x1.5 safety).
- `sanity_checks_bandit_toy.py` — 9 checks, all PASS on first run:
  analytic gradient vs finite differences (2.9e-9); fused oracle
  consistency; closed-form stationarity (6.9e-17); closed-form
  optimality vs random points and vs scipy L-BFGS-B (1.2e-16);
  minibatch unbiasedness (20k-draw mean, 1.1e-4); full-batch
  degeneration exact (7.6e-17); IFO accounting exact; SURF Eq. (9)
  speed == numerical ||d f_PF/dw|| (7.3e-11).
- `run_bandit_toy_without_256_checkpoints.py` — driver: runs both
  methods, post-hoc common scoring, value-gap machinery, 5 figures,
  summary.json + raw_histories.npz under
  `output/bandit_toy_surf_without_256_checkpoints/{smoke,eps1e-2,eps1e-3}/`.

## Edit to an existing file (minimal, user-authorised category)

`algorithm_fast_without_256_checkpoints.py` — three additions, no
behaviour change for existing callers:

1. `m_history` (bundle size at every checkpoint) recorded and returned.
2. New kwarg `return_pre_prune: bool = False`; when True the return
   dict carries a `pre_prune` snapshot {points, fvals, gram_stack}
   taken BEFORE `prune_inactive`. Default False because the points
   copy is memory-heavy on the MLP-sized problems.
3. Return dict gains keys `m_history`, `pre_prune`.

Why: the post-hoc comparable-meter rescoring and the per-checkpoint
value-gap trajectories need the bundle state at every checkpoint;
delivery-time pruning otherwise destroys that information.

Regression gate after the edit: `sanity_checks_fast.py` ALL PASS
(8/8, MSVRG degeneration still bitwise, IPOPT available).

## Calibration and statistical layer (data_seed 7)

- L calibrated: L1 = 0.1493, L2 = 0.1406 (raw max 0.0995 / 0.0937 over
  909 Hessians on the solution path + perturbations, safety 1.5).
- ||R_hat - R||_inf = 0.0593; ||Phi_hat - Phi||_inf = 0.1473.

## Parameters I adjusted on my own (user delegated, to be reported)

- `msvrg_max_segments`: 16 at eps=1e-2, 64 at eps=1e-3, 256 at
  eps=1e-4 (fuse widening for the tighter node/inner targets;
  censored_nodes will verify sufficiency).
- `eval_every_n_grads`: 10 (not the MLP track's 4500 or the plan's 50)
  — toy totals are O(10^2-10^3) grad-equivalents; finer cadence for
  readable curves, bookkeeping off both cost axes.
- Budget fuses (pure fuses, not budgets): max_grad_evals 2e5,
  max_wall 3600 s per method call.

## Smoke run (eps=1e-2 config with small caps) — plumbing validated

- Baseline: completed, 12/12 nodes served, 0 censored, 81 grad-equiv,
  0.01 s wall; strict full-simplex GN* 4.05e-3.
- Adaptive: epsilon_certified at outer 2, 30.5 grad-equiv, 0.23 s wall
  (lambda-search dominates, as predicted for K=2/d=4); strict GN*
  6.65e-3; bundle 6 -> 1 after delivery pruning.
- All 5 figures + summary.json + raw_histories.npz written.
- Notable preview: both methods pass the GN target easily at eps=1e-2,
  yet eps_value is 0.48 (baseline) / 0.31 (adaptive) — the logit-space
  first-order meter and the value-space meter separate exactly where
  the softmax Jacobian degenerates (w near 0/1). This is the expected
  complementarity of eps_opt and eps_value, and the reason both are
  reported.

## Formal results (all three rungs run July 26 with the user's go;
## serial, idle machine; data_seed 7 / sampler_seed 41)

Health: every rung clean — censored_nodes 0, safeguard_retries 0,
L_scale 1.0 end-to-end, no fuse hits, both methods stopped naturally
(baseline "completed", adaptive "epsilon_certified").

| rung | grads-to-eps bl/ad | cpu-to-eps bl/ad (s) | final GN* bl/ad | eps_value bl/ad | IGD bl/ad |
|---|---|---|---|---|---|
| 1e-2 | 85 / 30 | 0.007 / 0.081 | 3.4e-3 / 6.6e-3 | 4.76e-1 / 3.09e-1 | 0.338 / 0.480 |
| 1e-3 | 329 / 408 | 0.025 / 1.18 | 5.1e-4 / 5.0e-4 | 4.76e-1 / 3.50e-2 | 0.322 / 0.120 |
| 1e-4 | 1216 / 988 | 0.090 / 2.90 | 2.8e-5 / 5.3e-5 | 4.76e-1 / 2.68e-2 | 0.318 / 0.105 |

Findings:

1. GN axis, grad-equivalents: comparable/alternating (ad 2.8x fewer at
   1e-2, bl 1.24x fewer at 1e-3, ad 1.23x fewer at 1e-4) — consistent
   with the paper's "comparable oracle complexity" at K=2.  CPU axis:
   baseline wins 12-47x at every rung; the adaptive method's wall is
   ~97% IPOPT lambda-search (3.65 of 3.75 s at 1e-4), exactly the
   predicted K=2/d=4 behaviour.
2. HEADLINE — the value meter separates where the gradient meter is
   blind: baseline eps_value is pinned at 4.758e-1 across ALL rungs
   (85 -> 1216 grads bought zero value improvement), adaptive drops
   0.309 -> 0.035 -> 0.027 (17.7x better at the matched 1e-4 budget).
   Mechanism: near w = 1 the softmax saturates, so points with
   squared-gradient <= 6.7e-5 can still be ~0.48 above the optimum in
   value; the baseline's warm-start chain (w: 0 -> 1) must traverse
   ~25 logit units and its GN certificate triggers early in the flat
   region, so its delivered set never reaches the left end of the
   front (fig3: 232 points hug the right tail; fig4: red delta curve
   climbs monotonically to 0.48 for w > 0.5).  The adaptive worst-λ
   loop pushes points to both ends (5 delivered points span the front;
   blue curve <= 0.027 everywhere).  Where the baseline fails is
   exactly where SURF's traversal speed v(w) is large — the two
   papers' concerns meet.  Honest counterpoint: for w < 0.45 the
   baseline is actually BETTER than adaptive (both below 6e-3); the
   max is decided at the w = 1 end.
3. Negative finding: the r=11 between-node geometric floor did NOT
   appear down to GN* = 2.8e-5 — the delivered-set + gram-share
   memoisation serves between-node lambdas far better than the
   nearest-node bound.  To exhibit the floor on this toy the r-sweep
   should include SMALL r (3, 5), not just larger r; noted for
   FUTURE_WORK.

Caveats: single seed (7/41); eps_value/PF metrics use the common
solution-map rule (min over the delivered set of F_w); CPU numbers are
each method's on-curve axis (track semantics: method work only,
metric/checkpoint excluded).

## Jul 26, later: user questions -> figure-convention fixes + analysis

Figure fixes (driver edited, all three rungs re-run; trajectories
reproduced bitwise, only wall-clock jitter):

1. Shared pseudo-t0 on log-time axes: both curves' t=0 checkpoint now
   sits at (first real time of EITHER curve)/3, so the lines start
   from one shared point (both start at the same {x0} bundle, so y0
   was already equal; the per-curve pseudo-abscissa had split x0).
2. Pareto figure: headline markers are now each method's solution-map
   answers at the SAME 12 uniform query weights (equal counts by
   construction; distinct-point counts in the legend), with the full
   evaluated-point clouds kept as faint background.  The clouds turn
   out to carry the mechanism: the adaptive's 228 evaluated points
   trace the whole front end-to-end, the baseline's 232 sit off-front
   and cluster in the right tail.

Analysis recorded for the user's "why does the baseline win final GN"
question (kept for the eventual write-up):

- "Final GN" is a stopping-semantics artifact, not a quality ranking:
  the adaptive stops the moment its strict certificate <= 2eps/3
  (final value just under the stop line BY DESIGN), while the baseline
  has no global stop -- it must serve all 12 nodes at node_tol with
  solve_target = node_tol/4 undershoot, and its delivered-set minimum
  keeps deepening as a side effect.  Protocol-correct readouts are
  first-crossing and fixed-budget cuts.
- On grad-equivalents the two methods alternate leads (30/85, 408/329,
  988/1216); at matched budget the baseline's GN is better at the two
  tight rungs.  The MLP-track dominance (17x at K=6) is structurally
  absent at K=2: the grid is only r+1 = 12 nodes (no combinatorial
  blow-up), the w-chain warm start is a near-perfect 1-D continuation
  method, and the gram-share baseline is itself bundle-like
  (delivered-set memoisation).  This matches the paper's claim shape
  (advantage grows with K) and the pre-registered expectation.
- Why the GN and value meters diverge (fig4/fig5 vs fig1/fig2): in
  logit space EVERY vertex-concentrated policy is near-stationary for
  EVERY lambda (J -> 0), so eps-stationarity certificates are
  satisfiable at wrong vertices; the exact stationary point is unique
  (interior), but eps-stationary REGIONS include all vertex
  neighbourhoods.  The baseline chain signs its extreme-node
  certificates at wrong vertices and its eps_value pins at 0.476; the
  adaptive escapes partially because the T-map anchor rule is
  value-aware (argmin of F_lambda - GN/(2L)), so inner loops restart
  from the best-value point and make genuine value progress.
- Delivery pruning is GN-activation-based, not value-coverage-based:
  the adaptive's pruned 5-point delivery loses front coverage its own
  228-point evaluated cloud possessed (fig3).  If front delivery ever
  becomes the goal, prune by value-activation too (future option; not
  implemented).

## Jul 26, later still: equal-level stop + K=5 extension

Equal-level stop (user request "both stop on one line"):

- `baseline_svrg_certified` gained optional `global_stop_gn` (default
  None = unchanged): at checkpoint cadence, cheap-tier screen then
  strict-tier signature on the delivered Grams; stop when strict GN* <=
  the line.  Search time deliberately INSIDE the CPU axis (it decides
  termination, mirroring the fast method's lambda-search accounting).
  K=2 driver passes 2eps/3 for both methods.
- K=2 re-runs: stop_reason "global_stop_gn" at every rung; on THIS toy
  the level is reached only when the chain fixes the worst (last)
  region, so grads are bit-identical to the serve-all-nodes runs and
  only the CPU axis gained the check cost (0.30 / 0.47 / 0.76 s).
  Terminal values now within one progress quantum of the line
  (1e-3: 5.1e-4 vs 5.0e-4 — equal; 1e-2 / 1e-4 within 2x).  The sup
  drops discontinuously when the worst region is fixed, so "exactly on
  the line" is not achievable by any stopping rule; "first checkpoint
  <= line" is the implementable optimum, now symmetric across methods.

K=5 extension (user-approved design: centered-quartic rewards
R_k(a) = 1 - |x_a - x_k|^4, A = 5, K = 5, d = 4; r = 10 -> 1001 nodes;
eps {1e-2, 1e-3} first; fig3 -> value-gap CDF, fig4 -> 10 edge
profiles):

- `objectives_bandit_toy.py` generalised to arbitrary K: optional
  `R_true` matrix in the constructor; lambda-based closed-form API
  (`pi_star_lam` / `theta_star_lam` / `f_vec_lam` /
  `scalarized_opt_lam` / vectorised `oracle_batch`); w-based SURF layer
  guarded behind K == 2; `make_bandit_toy_K` factory; `calibrate_L`
  branches (K=2 path verbatim, K-general via vertex/centroid/Dirichlet
  lambdas).  K=2 preservation verified: sanity 9/9 unchanged AND a
  re-run of eps1e-2 reproduced every trajectory number bit-for-bit.
- `sanity_checks_bandit_toy_K5.py`: 9/9 PASS.  Check 4 was redesigned
  after a first FAIL that was a check-design flaw, not a code bug:
  cold-start L-BFGS-B stalls ~3e-5 ABOVE the closed-form optimum at
  vertex lambdas (the same flat-region early-stop phenomenon the
  experiment studies); scipy never lands BELOW the closed form, and a
  warm start matches it to 9e-15.  The check now asserts exactly that
  two-sided statement.
- `run_bandit_toy_K5_without_256_checkpoints.py`: K=5 driver reusing
  the K=2 module's scorer/figure machinery; evaluation lambda set =
  full r=10 grid (1001) + 20000 Dirichlet(1) (seed 0); chunked
  value-gap and IGD computations; front-uniformity (CV/Gap Ratio)
  omitted at K=5 (no canonical 1-D front ordering).  eval_every: 10 at
  1e-2 (share-served totals are only ~10^2 grads), 100 at 1e-3.
  L calibrated (K=5): [0.1240, 0.0545, 0.0555, 0.0919, 0.2491];
  ||R_hat - R||_inf = 0.0813.
- K=5 smoke (eps 1e-2, small caps): x0 ALONE serves 861/1001 nodes at
  node_tol 6.7e-3 (vertex flatness + gram share); baseline completes
  all 1001 nodes with 56 grads / 9 delivered points / 0.01 s; adaptive
  certifies at outer 3, 137 grads, 2.46 s (lambda-search dominant);
  eps_value 0.226 (bl) vs 0.117 (ad).  All figures render.
- Formal K=5 runs: user added the 1e-4 rung and gave the go.  Fuse
  auto-widening for 1e-4 (reported): max_outer 500 -> 2000,
  max_grad_evals 5e5 -> 2e6.  Figure-label fix: the shared figure
  functions gained a ``baseline_label`` parameter (the K=2 "r=11"
  string was hard-coded and leaked into K=5 figures); K=5 rungs re-run
  with correct labels, trajectories reproduced identically.

## K=5 formal results (July 26; serial idle machine; seeds 7/41)

Health: every rung clean — 1001/1001 nodes served, censored 0,
safeguard 0, L_scale 1.0, no fuse hits, adaptive certified at every
rung; adaptive CPU is ~98% lambda-search (19.8 of 20.3 s at 1e-4).

| rung | grads-to-eps bl/ad | cpu-to-eps bl/ad (s) | final GN* bl/ad | eps_value bl/ad | IGD bl/ad |
|---|---|---|---|---|---|
| 1e-2 | 77 / 61 | 0.007 / 0.59 | 6.3e-3 / 6.0e-3 | 2.26e-1 / 1.17e-1 | 0.171 / 0.166 |
| 1e-3 | 418 / 808 | 0.21 / 6.10 | 4.3e-4 / 4.6e-4 | 1.23e-1 / 6.42e-2 | 0.061 / 0.054 |
| 1e-4 | 3282 / 3338 | 0.92 / 15.8 | 2.6e-5 / 5.4e-5 | 1.11e-1 / 1.43e-2 | 0.053 / 0.029 |

Findings:

1. The equal-level stop delivers what it was built for: per-rung final
   GN* pairs now sit together just under the 2eps/3 line (differences
   within the last progress quantum).
2. GRID COLLAPSE — the central K=5 discovery: served_by_share =
   1001/1001 at every rung; the baseline certified the whole 1001-node
   grid with only 12 / 51 / 444 delivered points.  Gram-share
   memoisation plus vertex flatness (large eps-stationary regions)
   neutralise the combinatorial grid burden the K-advantage argument
   rests on, so no MLP-style grads dominance appears even at K=5
   (first crossings 77/61, 418/808, 3282/3338 — mixed/tie).
3. The ANYTIME picture still favours the adaptive method strongly
   (fig2 @ 1e-4): its curve rides 10-30x below the baseline's through
   the mid-budget range (e.g. ~1.7e-4 vs ~2.6e-3 at 2000 grads); the
   baseline only ties at the end via a terminal vertical plunge when
   the snake chain finally fixes the worst region.  Baseline progress
   is back-loaded (staircase), adaptive progress is anytime-smooth —
   which method "wins the grads axis" depends entirely on where the
   budget cut falls.
4. Value axis: the adaptive advantage GROWS with tighter eps —
   eps_value ratio 1.9x -> 1.9x -> 7.7x; IGD ratio 1.03x -> 1.14x ->
   1.8x; max point-to-oracle 0.64 vs 0.20 (3.2x) at 1e-4.  The
   baseline's eps_value stagnates around 0.11-0.23 across rungs (the
   wrong-vertex-certificate phenomenon persists at K=5).
5. Write-up narrative supported by both K=2 and K=5: on this bandit
   toy family the adaptive method's demonstrable advantage is
   value/coverage (eps_value, IGD, max-dist) and anytime GN quality,
   not terminal gradient counts; the terminal grads-axis K-advantage
   needs problem classes where the oracle is expensive and memoised
   grids cannot shortcut (the MLP track, and the FUTURE_WORK
   non-convex-reward variant which kills the closed-form flatness).

---

## Jul 26, part 2 (SESSION 12, MLP track): grid-meter figures, tol=0.01 leg, adaptive extension

Separate work block, same date: everything above is session 11's SURF
record; everything below is session 12 (resumed from the ledger).

### Ledger correction (reported to the user first)

The ledger said "r15/r20 pending the user's go", but the sweep had
already been run overnight Jul 25->26 (sweep_run.log, pid 36689;
r15: 1255 s, audit GN* 5.9456e-2; r20: 4114 s, audit GN* 6.3415e-2;
both completed, censored 0, node_tol 0.02) and committed in df75b55
together with that folder's READMEs and the two July-25-presentation
figures.  Evidently a parallel session with the user's go; the
session-11 ledger rewrite missed it.  Data verified healthy (log +
summaries consistent); nothing re-run.

### User requests (this session)

1. Adaptive: v3 stopped at round_fuse (max_outer=500), best 0.0581 vs
   eps=1e-3 -> raise the cap, re-run, report the expected runtime.
2. Baseline figures: intermediate checkpoints on the GRID meter
   (enumerate nodes); final point = grid endpoint + a SEPARATE
   full-simplex audit.  User rationale, recorded: a global meter on
   the baseline's trajectory over-serves its contract (its theory
   covers nodes only); the audit must EXHIBIT the between-node gap,
   not absorb it.  The MLP sweep never used the equal-level stop
   (that option exists only in the bandit drivers), so the engine is
   already native — this is a figures-only change.
3. Keep the node_tol=0.02 results AND add a lower-node_tol leg to see
   how the audit and the trajectory move.  Value chosen by the
   assistant (disclosed reasoning): 0.01 with solve_target 0.0025
   (=tol/4) — halves the certificate while keeping the inner target
   above the ~1e-3 zone where Jul_16_note.md flagged the SVRG stall,
   so the grid-geometry question stays separated from the
   inner-solver-floor question.  r legs: {10, 15} first (r10 = the
   between-node-gap anchor, 96% of nodes solver-visited; r15 = where
   the tol=0.02 audit saturates ~0.06); r12/r20 only if the result
   warrants.

### Code changes

- `run_baseline_svrg_r_sweep_without_256_checkpoints.py` (edited;
  engine `baseline_svrg_certified_without_256_checkpoints.py`
  untouched):
  * `_plot_sweep`: baseline lines now `cov_history` (native grid
    meter); endpoint circle = grid certificate end; separate x marker
    = `delivered_gn_strict` (the audit), dotted vertical connector
    between them; legend/ylabel/title carry the meter caveat.  The
    strict prefix history stays in the summaries, unplotted.
  * `_write_readmes`: figure-section text (EN+ZH) rewritten to match.
  * New args: `--out-dirname` (redirect the output folder; enables
    the v2 home and keeps the tol legs apart) and `--fast-ref`
    (adaptive reference curve from any fast-trial folder; the used
    path is disclosed in the generated READMEs as before).
- `run_trial_K6_fast_without_256_checkpoints.py`: NO changes needed —
  `--max-outer` and `--variant-tag` already exist.

### New comparison home

`output/baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/`:
`original/` = verbatim copy of the old folder (the old folder itself
stays in place untouched, so existing references remain valid);
`tol0.02/` = copied summaries (r10/r12/r15/r20) with figures REDRAWN
via --replot in the new presentation (no re-runs; output inspected —
r10 shows the largest connector, 0.02 -> 0.1635 = 8.2x; r15/r20
audits sit at ~0.06 near the fast best 0.0581); `tol0.01/` and
`adaptive_extended/` land after the runs.

### Run plan presented (awaiting the user's go)

Serial on an idle machine (CPU-axis discipline):
1. Fast adaptive, --max-outer 2000 --variant-tag v4_max_outer_2000:
   estimate ~45-60 min.  Basis: lambda-search cost/round grows
   ~linearly in bundle size m (~ round index); v3 measured 135.7 s
   over rounds with m=2..502 -> c ~ 1.1e-3 s per m-unit; rounds
   501..2000 add ~2000 s of search + ~470 s inner/oracle work +
   the 500-round replay (~290 s).  Expectation set honestly IN
   ADVANCE: v3's best_so_far was flat at 0.05811 over its last 42%
   (grad-equiv 5413 -> 9226; cheap-tier pc bottom 0.0479), and
   Jul_16_note.md already recorded "certification at eps=1e-3 is NOT
   reachable by round-count alone at this pace" — so a mere-cap
   explanation is unlikely; the run decides the hypothesis either
   way.  If the plateau persists, the next single lever is the inner
   solver (batch / segments / rel_target), not more rounds.
2. Baseline node_tol=0.01: r=10 (est 15-25 min), then r=15 (est
   35-65 min), --save-grams both, into tol0.01/.
3. Replot both tol legs with --fast-ref at the extended run; copy the
   extended run's outputs into adaptive_extended/; fill the Results
   section below.

### Results — part A: the v3 plateau diagnostic

Script: `Original_py/diag_v3_plateau_without_256_checkpoints.py`
(new, one-off; output `.../v2_.../diag_v3_plateau/diag.json`).  It
replays the v3 config/seeds with `return_pre_prune=True`, strict-audits
prefixes of the resulting bundle, and localises the strict witness.

- **REPLAY NOT BIT-IDENTICAL (open item):** same config, same seeds
  reproduced neither the grad total (9712.98 vs stored 9225.69, +5.3%)
  nor pc_history (max abs diff 0.37); the replay had multi-segment
  rounds and one segment-cap warning (stored v3: 499/500 rounds at a
  single segment, no cap).  MLP-track torch runs are therefore NOT
  bit-reproducible in the current environment (the bandit-track numpy
  runs are — session 11 verified).  Open: thread-level float
  nondeterminism vs a behavioural side effect of the session-11
  algorithm edits (sanity 8/8 passes, but that compares today-vs-today,
  not today-vs-July-15).  Conclusions below are mechanism-level, drawn
  from the replay's own internally consistent bundle/history pair.
- **Cheap meter under-reports ~2x:** strict 64-start audit of the
  final replay bundle = **0.1403** vs the replay's cheap final pc
  0.0777 (cheap min over last 50 rounds 0.0578).  The quoted "v3 best
  0.058" is a cheap-meter artifact; the honest worst-case of such a
  bundle is ~0.14.
- **TARGETING FAILURE PROVEN (dominant cause):** strict witness
  lambda* ~ [0, 0, 0.104, 0, 0, 0.896] — an e3-e6 EDGE mix; its l1
  distance to the nearest lambda the run ever targeted is **0.208**,
  and that nearest visit happened at ROUND 5; median l1 distance to
  the 500 visited lambdas is 1.58.  The cheap tier (centroid + K
  vertices + prev_lam, ~8 structured starts) never aimed at the true
  peak region again in 495 rounds — rounds ground down only the
  peaks the search could see.
- **True trajectory nearly stalled:** strict prefix audits 0.753
  (m=25) -> 0.250 (m=100) -> 0.191 (m=200) -> 0.178 (m=300) -> 0.153
  (m=400) -> 0.140 (m=450) -> 0.140 (m=528): under one halving across
  the last ~400 rounds, while the cheap meter crawled 0.096 -> 0.076.
- A per-round achieved-depth statistic was also computed but is
  **DISCARDED**: it assumed one delivered point per round, which the
  replay violates (multi-segment rounds deliver one point per segment;
  m=528 for 500 rounds), so the Gram/lambda pairing misaligns after
  the first multi-segment round.  Recorded to keep the discard honest.
- Measured strict-search pace on this problem: ~1.08e-2 s per bundle
  point per warm-started search (12 searches, sum m = 2803, 30.4 s).

**Revised adaptive plan (supersedes the max_outer=2000 idea; awaiting
the user's go):** `run_trial_K6_fast_without_256_checkpoints.py
--tier-mode strict --rel-target 0.1 --max-outer 300 --variant-tag
v4_strict_rel0.1`.  Rationale: strict per-round search fixes both the
targeting and the meter (the proven failure); rel_target 0.25 -> 0.1
deepens each cut (~2-4 segments/round expected; initial targets ~0.008,
still above the ~1e-3 inner floor v2 established with 150/150 cap-hits
at 3.3e-4 targets, so no cap storm expected at first).  max_outer=300
is a probe-sized fuse: estimated 40-60 min wall (strict lambda-search
cost ~1.08e-2 s x bundle size per round, bundle growing 2-4/round;
~15-20k grad-equivalents).  Honest expectation, set in advance: the
TRUE GN* should land well below the replay's 0.14, but certification
at eps=1e-3 is still NOT expected — when pc approaches ~3e-3 the eps/3
floor and the b=4096 variance wall should reassert; the batch lever
(8192/16384) is the next single change after this probe.

### Results — part B: the runs (executed overnight Jul 26 -> 27)

Order actually run (user decision): tol=0.01 legs first; the v4 probe
after them; then a NEW fixed-budget experiment the user designed and
approved the same evening (protocol below).  All serial on an idle
machine.

**B1. Baseline node_tol=0.01 legs (grid-geometry vs solve-depth):**

| leg | grads | wall s | grid cert end | strict audit | vs tol=0.02 |
|-----|-------|--------|---------------|--------------|-------------|
| r10 | 55,416 (+34%) | 882 | 0.0098 | **0.1499** | audit -8% (was 0.1635) |
| r15 | 254,197 (x3.1) | 3,628 | 0.0100 | **0.0589** | audit -1% (was 0.0595) |

Both clean (censored 0, completed; r10 delivered 4,156 points, r15
16,896).  VERDICT: tightening the node certificate 2x costs +34% to
x3.1 budget and buys 1-8% of global quality — the between-node error
is GRID-GEOMETRY dominated, not solve-depth dominated.  Publishable
negative result; also fixes the baseline frontier shape: knee at
r15@0.02 (81k -> 0.0595); r20@0.02 (242k -> 0.0634) and r15@0.01
(254k -> 0.0589) show the ~0.06 saturation — 3x more budget buys
nothing.

**B2. v4 probe (strict targeting + rel_target 0.1, max_outer 300):**
926 s, grads 12,346 (41.2/round vs v3's 18.5), stop=round_fuse.
Strict per-round readings (now honest): 50-round block medians 0.836
-> 0.265 -> 0.168 -> 0.136 -> 0.117 -> 0.104; no plateau.  At 300
rounds the TRUE value (0.104) beats v3's TRUE value at 500 rounds
(0.14, audited) — the targeting fix works.  NEW floor evidence:
31/300 rounds hit the 10-segment cap at targets ~0.01
(inner_cap_hits=28) — from some anchors ~0.01 is already unreachable
in-budget; the b=4096 wall is closer than the v2 bracket suggested.
lambda-search = 81.7% of wall at 64 starts (why the fixed-budget run
targets with 24).  Probe archived to v2 `adaptive_extended/`.

**B3. Fixed-budget experiment (user-designed protocol, approved):**
one budget axis, ONE instrument (strict 64-start in-family audit).
Baseline configurations (r, tol) enter as completed-run POINTS
(x = realized cost, y = stored delivery audit); the adaptive method is
ONE budget-mode run cut at B = 80,912 (r15@0.02's realized cost),
its trajectory audited post-hoc on bundle prefixes at checkpoints
(off-axis; monotone lower-bound envelope plotted — raw audits in the
summary).  NEW driver `run_fixed_budget_K6_without_256_checkpoints.py`
(+ --replot; smoke caught a README KeyError before launch, fixed).
Run: rel_target 0.05, targeting 24 starts, eval_every 2000.  Result:
48.6 min wall (lambda-search 27.5 min = 57%), grads 81,058, bundle
m=4,318, cap-hit rounds 414 (expected at these targets; budget mode),
stop=budget.  Audited (envelope) trajectory: 7.32 (x0) -> 0.203
(10.5k) -> 0.159 (31k) -> 0.137 (41.6k) -> 0.146-envelope (62k) ->
**0.102 (final, 80.9k)**; post-prune audit 0.108 (pruning costs a
little global GN — value-blind activation pruning, known property).
One audit non-monotonicity caught (52k raw 0.129 < 62k raw 0.146 —
impossible for the true prefix GN*, so the 52k audit under-reported);
the envelope repairs it and is disclosed in the README.

FIXED-BUDGET VERDICT (single meter, grads axis): mixed, baseline
ahead in the mid/high range — at 41k adaptive 0.137 vs r10@0.02
0.1635 (adaptive x1.1 better); at 55k vs r10@0.01 ~tie (x1.0); at
64k baseline r12@0.02 0.0954 vs ~0.135 (x0.7, baseline better); at
B=81k baseline r15@0.02 **0.0595 vs adaptive 0.102** (x0.6, baseline
clearly better).  Beyond budget: r20@0.02 / r15@0.01 sit at ~0.06 for
3x the cost.  On this K=6 MLP with the gram-share baseline, the
terminal grads-axis advantage at these budgets belongs to the
BASELINE at its knee; the adaptive method's remaining case is the
anytime shape below ~40k and the (still unfixed) inner floor.
Consistent with the bandit-track lesson: memoised grids neutralise
the K-advantage argument on the grads axis.

**Figures/archive (final layout of the v2 home):** `original/`,
`tol0.02/` (grid-meter presentation, adaptive reference = v4 strict
curve via --fast-ref), `tol0.01/` (same), `adaptive_extended/` (v4
probe copy), `fixed_budget_B80912/` (headline figures
fixed_budget_gn_vs_{grads,cpu}.png + summary + bundle_grams.npz),
`diag_v3_plateau/`, `fixed_budget_B600_SMOKE/`.

**Open items after session 12:** (1) MLP torch runs not
bit-reproducible in this environment — cause unresolved (threading vs
session-11 edit side effect); (2) the b=4096 inner floor now has a
measured onset (~0.01 from some anchors) — the batch lever
(8192/16384/b=n) is the designed next single change if the user wants
to chase eps=1e-3; (3) audit under-search exists even at 64 starts
(the caught non-monotonicity) — more starts or restarts are an option
where audits are load-bearing; (4) Jul_20_note.md §7 still carries
the wrong ~16x CPU figure (correct 8.8x) — fix on next MLP-track
touch.
