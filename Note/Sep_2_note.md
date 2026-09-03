# Sep 2 note — K = 2 pair campaign v2: design freeze + S0 TAG pair scan

Session of Sep 1-2, 2026 (design Q&A Sep 1 - Sep 2, sign-off and S0 run
Sep 2). The Aug-13 K = 2 pair campaign (v1) is extended to **v2**: a
SURF weight-allocation leg joins the comparison, one optimization core
is picked by a pre-experiment, GN moves to the norm scale, and the pair
itself is now selected by a TAG lookahead-affinity scan of all 45 digit
pairs. ALL new files; nothing overwritten.

## Campaign in one line

K = 2 MNIST digit-pair MOO (2-logit patch-softplus net, d = 8,098,
per-class mean CE, per_class = 5421): fixed budget B = 20,000
grad-equivalents, legs = SURF (baseline1, N swept) + uniform grid
(baseline2, r swept) + adaptive CCP, all three on the same
pre-experiment-selected optimization core; judged on best-so-far
worst GN vs total gradient evaluations, worst GN vs CPU time, and the
Pareto front figure; ridge side-line mu = 1e-4 (appendix).

## Design decisions (user-approved Sep 1-2)

* **GN scale — three-layer convention** (theory: bundle paper v2,
  Prop 3: GN(·;B) is uniformly Lipschitz in lambda on the NORM scale,
  not the squared scale; its eq. 4 defines eps_sm-stat as a norm):
  numerical kernel unchanged (Grams, val = lam' Q lam, every
  argmax/argmin identical since sqrt commutes with min/max); every
  prescribed tolerance is stated on the norm scale and squared once at
  code entry; all cross-lambda aggregation and ALL reporting on the
  norm. Fixed budget => NO threshold line on any plot, NO
  certified/uncertified marking on the front figure.
* **Metric**: worst GN(D) = max_lam min_{x in D} ||J(x)' lam||,
  computed EXACTLY on the 200,001-point w-grid (v1 convention; smoke
  20,001). CCP multistart is only the adaptive leg's internal selection
  engine, never the metric. Checkpoints log (grad_equiv, cpu_time,
  worst GN) at eval_every cadence; every leg charges one full
  evaluation at the shared x0 so curves start together.
* **Legs**: SURF as baseline1 (spec in `Adaptive Bundle
  Algorithm/BASELINE_SURF.md` + Zh twin — SURF's Phi-quantile
  allocation verbatim, our MSVRG segment machinery inside, single
  phase, no tol stopping, alpha = 0.3, eps_arc = 1e-12, Phi on 1001
  points); uniform grid as baseline2 (round-robin, s = 5 per visit);
  adaptive CCP (production config N0 = 2000, r = 10, exp sampler).
* **Resolution ladders, symmetric**: uniform r in {10, 20, 30, 40} AND
  SURF N in {10, 20, 30, 40}, each leg fights at its own best (r*, N*).
  Theory hook: v2 paper Prop 4 (uniform needs ~eK(1+C*LipGN/eps)^(K-1)
  points; constant unknown => sweep, don't guess).
* **Optimization core pre-experiment (S2)**: steppers const / bb /
  adagrad / adam per `ADAPTIVE_STEPPERS.md` (revised Sep 2: the 4-way
  switch lives ONLY in the CCP bundle engine — shared module
  `Core Engine/stepper_core.py` + engine variant
  `algorithm_ccp_stepper_without_256_checkpoints.py`; SURF/uniform legs
  import only the winner). 11 configs (const 1 + bb 1 + adagrad 3 +
  adam 6) x 3 seeds, B = 2,500, eval_every = 50, on the S0-selected
  pair; judge = worst GN vs grad_equiv, CPU secondary, ties to the
  simpler stepper. Winner frozen for every leg.
* **Gates before S2**: Gate 0 = stepper="const" bit-identical to the
  existing CCP engine; Gate 1 = bb/adagrad/adam on bandit-toy, 2 nodes
  each, no NaN / bounded retries.
* **Ridge side-line (S5)**: mu = 1e-4 (K3 Exp-6 value), i.e.
  f_k + (mu/2)||x||^2, new objective file objectives_mnist_pair_ridge
  planned; same winner core and same r*/N* as the mu = 0 main line;
  separate problem instance — figures side by side, never overlaid;
  appendix placement (SmokeA precedent).
* **Pair selection (S0)**: TAG lookahead affinity, the K3 Phase-1
  method ported to pairs (user question "is this feasible" answered
  yes — the measure is pairwise-native; K = 3 needed aggregation, K = 2
  reads the matrix directly). Criterion: MOST conflicting
  (Cbalanced-first ranking), because a non-conflicting pair collapses
  the front and de-differentiates the legs.
* Stage gating: S0 (pair) -> S1 (stepper_core + gates) -> S2
  (pre-experiment) -> S3 (SURF leg + ladders) -> S4 (main, mu = 0) ->
  S5 (ridge). Each stage ends with a user sign-off before the next.

## Artifacts created today

* `Adaptive Bundle Algorithm/BASELINE_SURF.md` + `Zh/BASELINE_SURF_ZH.md`
  — SURF leg + evaluation protocol spec (new doc pair).
* `Adaptive Bundle Algorithm/ADAPTIVE_STEPPERS.md` + Zh twin — Sep-2
  revision block (stepper home moved to the CCP engine, Gate-0 target
  changed, smoke superseded by S2, GN norm-scale note).
* `Reference_essay/A_first_order_bundle_method_for_smooth_multi_
  objective_optimization__MAnalytics_ (2).pdf` — bundle paper v2
  (Prop 3 = GN Lipschitz in lambda, Prop 4 = uniform-grid complexity,
  Thm 1 = adaptive Pascal bound, 5 inner solvers under Assumption 3.1);
  copied from Downloads next to the July v1.
* `Original_py/experiment_plot/
  run_tag_affinity_K2_mnist_pairs_without_256_checkpoints.py` — S0
  scanner: K3 A1 protocol ported to pairs (centroid chain lam = (1/2,
  1/2), throwaway probes theta - (alpha/L_i) J_i at EVERY segment
  0..15, alpha = 0.05, Z = 1 - F_j(probe)/F_j(theta), C = mean
  max(0,-Z), c_j = 0.5 * incoming, Cbalanced/Cmean ranking; per_class
  5421, seeds 8/41/7, resume-skip). Both checkpoint grids ranked from
  ONE run: scan {0,3,6,9,12,15} = headline (K3-faithful), full 0..15 =
  stability variant.

## S0 result (scan of all 45 pairs, ~7 s/pair, outputs under
## output/CCP/K2_mnist_pair_without_256_checkpoints/tag_affinity/scan/)

Headline ranking (scan grid), top 6 of 45:

| rank | pair | Cbalanced | Cmean |
|---|---|---|---|
| 1 | **4v9** | 1.5095 | 2.5484 |
| 2 | 3v5 | 1.3709 | 1.9750 |
| 3 | 2v8 | 1.0546 | 1.4975 |
| 4 | 5v8 | 1.0322 | 1.4999 |
| 5 | 7v9 | 1.0298 | 1.8384 |
| 6 | 3v8 | 1.0288 | 1.4570 |

(tail: 0v4 at 0.4005; full table in RANKING.md / ranking_scan.csv.)

* **Grid stability** (the Phase-1 worry): Spearman(scan, full) = 0.969
  over 45 pairs; top-1 = 4v9 on BOTH grids; top-2 (4v9, 3v5) identical
  on both; only the 1.03-cluster (2v8/5v8/7v9/3v8, spread 0.026)
  shuffles internally.
* **Cross-check vs the Aug-13 conflict smoke** (different score:
  interior front width S_int over favour chains): its five candidates
  3v5(.901) 7v9(.843) 3v8(.810) 4v9(.727) 5v8(.725) are EXACTLY five
  of TAG's top 6 — the only newcomer is 2v8 (#3). Two independent
  instruments agree on the head of the pack; ordering within the head
  differs (TAG: 4v9 first; S_int: 3v5 first) as expected from
  different measurands (gradient-level conflict vs solution-level
  front width).
* Consistency with K3: 4v9 and 3v5 are core pairs of the two
  top-ranked K3 triples {4,7,9} and {3,5,8}.

**Recommendation to lock: 4v9** — #1 by the mandated criterion, stable
across both grids, corroborated by the Aug-13 instrument, and FRESH
(v1 already spent full budgets on 3v5 and 7v9; a new pair avoids any
reuse concern). Runner-up if continuity with v1 data is preferred:
3v5. Awaiting the user's lock (end of S0).

## S0 sign-off + S1 (same day)

**Pair locked by the user: 4v9** (TAG top-1, both grids).

S1 built and gated (all same day):

* `Original_py/Core Engine/stepper_core.py` — the four walk rules
  (const / bb / adagrad / adam) behind one hook contract
  (on_lambda_change / start_segment / step / on_segment_result);
  const path promises the incumbent's exact float-op sequence, no
  stepper consumes randomness. Reset semantics as per the design:
  per-lambda state resets when lambda changes; on an ascent bb falls
  back to const for the next segment, adagrad re-inits G0 with the
  doubled L_scale, adam clears moments and halves alpha.
* `Original_py/experiment_plot/
  run_stepper_pre_experiment_K2_without_256_checkpoints.py` — the
  stepper-parameterized copy of the v1 pair executor (`_run_leg_pair`
  imported untouched as the Gate-0 reference) + gate drivers + the S2
  driver (11 configs x seeds {41,141,241}, B=2,500, eval_every=50,
  s=5, audit grid 20,001; judge board + norm-scale curves both axes).
  Summary gains additive fields only (stepper block,
  audited_gn_norm_history, per-segment stepper diagnostics).
  NOTE (doc deviation, recorded in the ADAPTIVE_STEPPERS Sep-2
  notice): the campaign executors live in the runner layer, so there
  is no `algorithm_ccp_stepper_*` Core-Engine file; and Gate 1 runs on
  the 4v9 smoke instance instead of the bandit toy (same intent,
  exactly the S2 machinery).
* **Gate 0 PASS** (4v9 smoke, B=800, per_class=300, adaptive-CCP leg):
  v1 executor vs stepper="const" — bit-exact on all 12 checks
  (gram_stack, fvals, lam_history, seg_grads, theta_stack, test_ce,
  grad_equiv_total, L_scale_final, safeguard_retries, ck_m, ck_grads,
  audited_gn_history). Report: stepper_pre_experiment/gates/
  gate0_report.json.
* **Gate 1 PASS** (same smoke instance): bb (L_scale 8, 3 retries,
  final GN-norm 0.113), adagrad_mult3 (L_scale 2, 1 retry, 0.074),
  adam alpha3e-4/beta2 .99 (L_scale 1, 0 retries, 0.579) — all
  finite, audits monotone, progress on every arm. Smoke-tier signal
  only (NOT the S2 judge): adagrad ahead, adam slow at tiny budget.

## S2 pre-experiment result (user go same day; 33 runs, ~52 min,
## outputs under stepper_pre_experiment/, board in s2_summary.json)

Full instance 4v9: n = 11,684 (per_class = 5,842), epoch_len = 12,
L = [1.449, 1.373]. Board, mean final worst GN (norm) over seeds
{41, 141, 241}, best first:

| config | final GN | per-seed | safeguard retries |
|---|---|---|---|
| adam a1e-3 b2 .9 | 1.057e-2 | 1.03-1.08e-2 | 27 / 32 / 33 |
| **adagrad x10** | 1.132e-2 | 1.03-1.20e-2 | **1 / 1 / 1** |
| adagrad x3 | 1.407e-2 | 1.34-1.49e-2 | 0 |
| adam a1e-3 b2 .99 | 1.454e-2 | 1.40-1.53e-2 | 17-24 |
| adam a3e-4 b2 .9 | 2.669e-2 | — | ~20 |
| adagrad x1 | 2.741e-2 | — | 0 |
| const (incumbent) | 2.827e-2 | 2.78-2.88e-2 | 0 |
| adam a3e-4 b2 .99 | 3.048e-2 | — | ~19 |
| bb | 3.126e-2 | 2.57-3.52e-2 | 3 |
| adam a1e-4 (both) | 6.7e-2 / 1.1e-1 | — | 13-27 |

**Verdict: winner = adagrad, alpha_mult = 10**, by the pre-registered
rules: (1) primary judge is the whole worst-GN-vs-grad_equiv curve —
adagrad x10 leads at every interior budget point (B=500: 7.6e-2 vs
adam 1.4e-1 vs const 2.3e-1; B=1000: 3.2e-2 vs 4.7e-2 vs 8.1e-2;
B=1500/2000 likewise), adam catches up only at the very end; (2) the
final values are a statistical tie (mean gap 7 %, per-seed ranges
overlap, adam wins 2 of 3 seeds pairwise) -> tie goes to the simpler
stepper (tax 3 vs 6); (3) CPU axis identical (all walls ~94 s,
decisions ~1 s); (4) robustness: adagrad x10 tripped the descent
safeguard once per run vs adam's ~30 (alpha overshoot repeatedly
bailed out by the safeguard; L_scale reached 2^13-2^17 in adam runs —
harmless to adam itself, but a fragility signal).

Findings worth keeping: both top arms beat the incumbent const by
~2.5x at equal budget — the pre-experiment paid for itself; **bb
underperformed const** (3.1e-2 vs 2.8e-2, widest seed spread) — under
adaptive-CCP the lambda changes every s = 5 segments, the secant
memory resets each time, so BB acts on at most 4 segments per block
with noisy s=5-tier secants (honest negative result; BB may still
shine in lambda-stable settings like the uniform leg, but it is NOT
the campaign core); adam is knife-edge alpha-sensitive (1e-4 is 10x
worse than const, 1e-3 is best) — the "no L formula" tax made visible;
the adagrad ladder is monotone up to the grid edge (x1 < x3 < x10) —
an x30 probe would be outside the pre-registered tax, noted only.

**Frozen for all legs of S3/S4/S5: adagrad with alpha_mult = 10**
(G0 warm start at (L_hat/c)^2, per-lambda reset, ascent re-init).

## S2 sign-off amendment (user, same day): TWO cores, not one

The user chose to carry BOTH top cores through the campaign as two
parallel, self-contained experiments: core A = adagrad x10, core B =
adam(alpha = 1e-3, beta2 = 0.9).  Every downstream stage doubles:
ladders pick (r*, N*) per core; S4 runs 2 cores x 3 legs x 3 seeds.

## S3 build + the SURF collapse finding (Sep 2, evening)

New files:

* `Original_py/baseline/baseline_surf_without_256_checkpoints.py` —
  the SURF leg: Phi-quantile allocation on a 1001-point w-grid, one
  MSVRG segment per slot per round (vertical warm start, per-slot
  anchors + v1 accept/reject), chord -> PCHIP -> damped Phi update
  (alpha = 0.3), stepper injected from stepper_core (one instance,
  per-visit lambda reset — the S2-validated semantics).  x0 uncharged
  (v1 convention; BASELINE_SURF.md said "charged" — corrected).
* `Original_py/experiment_plot/
  run_surf_compare_K2_without_256_checkpoints.py` — stages: ladders
  (S3: uniform r and SURF N in {10,20,30,40} x 2 cores x 3 seeds,
  B = 2,500) and main (S4: 3 legs x 2 cores x 3 seeds, B = 20,000,
  worst-GN curves both axes + the Pareto-front figure per core).

**Finding 1 — verbatim SURF collapses on the unregularized pair.**
At mu = 0 the vertex solutions diverge (ignored-class CE unbounded,
the Aug-13 "runaway arms"), violating SURF's bounded-speed
Assumptions 2/3.  The two arms swallow the measured arc length and
Phi^{-1} drives every slot onto the vertices.  Verified at B = 2,500
full instance (diag_surf_N10_full): after 38 rounds all 11 slots sat
at w < 0.001 or w > 0.999, final worst GN 0.119 = 10x worse than
adaptive at equal budget/core.  Evidence kept under
v2_campaign/SMOKE/.

**Fix — windowed chords** (minimal intervention, house precedent):
the chord measurement winsorizes each slot f-vector at
f_window = ln 2 (2-class random-guess level; the SAME [0, ln2]^2
window the v1 front figures use).  Beyond-window arm growth adds no
arc.  Dynamics/delivery/metric untouched; f_window=None restores
verbatim.  Windowed diagnostic (diag_surf_N10_full_windowed): slots
spread over w = [0, .104, .129, .203, .296, .486, .646, .693, .743,
.905, 1] — allocation healthy.

**Finding 2 — depth vs breadth (early, one config).**  Even windowed,
SURF's final worst GN at B = 2,500 (N = 10, core A, seed 41) is
0.193 vs adaptive 0.0113 at the same budget/core: SURF runs N+1
shallow parallel per-slot chains (~38 segments each) while the
chain-style legs pour ~428 segments into one warm-started sweep, and
the worst-GN metric rewards depth.  A structural, honest result —
the ladders + main run will quantify it (expect N* at the small end
of the ladder).

S3 ladders launched (48 runs, ~1.3 h): uniform r / SURF N in
{10,20,30,40} x {adagrad x10, adam 1e-3/0.9} x seeds {41,141,241},
B = 2,500 -> (r*, N*) per core, under v2_campaign/ladders/.

## Design pivot (user decision, Sep 2 late evening): RIDGE-ONLY campaign

On seeing the collapse finding the user redirected: **no windowed
chords — regularize the problem instead, and drop the mu = 0 line
entirely.**  The campaign now runs on the ridge objectives only;
verbatim SURF (f_window = None) throughout; the mu = 0 ladders were
stopped mid-run and their partial outputs abandoned (collapse
evidence under v2_campaign/SMOKE/ retained).  The user also asked for
a smoke test of the ridge coefficient itself before committing.

Build (same evening):

* `Original_py/objective/objectives_mnist_pair_ridge.py` — K = 2 port
  of the Aug-26 K3 ridge wrapper (penalty on all d = 8,098 coords,
  exact SVRG anchor-term cancellation, L + mu analytic shift, raw-CE
  test reporting; mu = 0 degenerates bit-identically).  Factory
  sanity-checked: f/J/L/grad_pair penalty terms and identical batch
  stream all verified against the base factory.
* `mu` wired through both executors (`_run_leg_pair_stepper`,
  `run_surf_leg`); SURF's `f_window` default flipped to None (the
  windowed mode remains only as the mu = 0 collapse record).

**mu smoke (running)**: verbatim SURF, N = 10, core A, seed 41,
B = 2,500, mu in {1e-4, 1e-3, 1e-2, 1e-1}.  Pass criteria: slot
allocation not collapsed onto vertices, arc length bounded/stable,
interior trade-off intact.  Prior evidence says small mu may fail
(K3 SmokeA: mu = 1e-3 "vertex not tamed"; the runaway logit scale
shrinks only logarithmically in 1/mu).

## mu smoke verdict (Sep 2/3): the mu-only window is EMPTY

Verbatim SURF, N = 10, core A, seed 41, B = 2,500, five mu values
(mu_smoke/ dirs; per-w test profiles + ||theta|| checked):

| mu | allocation | learning | verdict |
|---|---|---|---|
| 1e-4 | collapsed (all slots at vertices, arc 399) | alive | FAIL |
| 1e-3 | collapsed (arc 287) | alive | FAIL |
| 1e-2 | vertex-dominated (8/11 at edges, arc 24, still shrinking) | alive (||th||~4, w=.083 a real classifier) | FAIL |
| 3e-2 | healthy spread | DEAD (||th||~0.3, constant classifiers, mid CE = ln 2) | FAIL |
| 1e-1 | healthy spread (arc converged 9.4) | DEAD (||th_mid|| = 0.00) | FAIL |

Mechanism: the vertex arms shrink only ~log(1/mu) while ridge kills the
d = 8,098 net's learning around mu ~ 3e-2 — taming needs mu >= 3e-2,
learning needs mu <= 1e-2.  No single mu passes both.

**Dial-trim probe (PASS)**: run_surf_leg gains `w_min` (affine trim of
the weight dial to [w_min, 1-w_min]; exact vertices carry zero weight
on one objective — that zero is what permits divergence; at any w > 0
the scalarized minimizer keeps both objectives finite.  Same remedy as
the MODPO branch's ADAPTIVE_LAMBDA_MIN=0.05).  Probe mu = 1e-3,
w_min = 0.05: arc 4.9 -> 3.0 (bounded, converged); slots spread over
[0.05, .10-.15 shoulder cluster, .40, .82, .87, .95] (geometry-driven
concentration on the steep left shoulder — not collapse); learning
fully alive (||theta|| 17-19; real trade-off: test err (23%,0.5%) at
w=.10 -> (5.9%,4.5%) at w=.40 -> (0.4%,41%) at w=.95).

Recommendation to the user: campaign = ridge mu = 1e-3 + SURF dial
trim w_min = 0.05 (uniform/adaptive legs untouched — their vertex
nodes are harmless without SURF's arc feedback loop; the worst-GN
metric stays over the FULL simplex for every leg).  Awaiting sign-off.

## Sep 3 sign-off + per-slot steppers + S3 launch

The user's friend's slide deck (SURF Baseline - What It Does, MODPO
branch) corroborated two things: (1) their verbatim SURF also chases
the longest arm, but their DPO losses are BOUNDED (~1.3 cap), which is
exactly why they never collapse — confirming our diagnosis that
collapse = unbounded front x arc allocation, and supporting the
bound-the-front fix; (2) their slots keep independent optimizer state
across rounds (same-slot warm start of AdamW + scheduler), which is
the correct "slot = node" reading of our own reset rule.

**User sign-off (both items)**: campaign mu = 1e-3 + SURF dial trim
w in [0.05, 0.95]; AND the SURF leg's steppers become per-slot
instances persisted across rounds (initialised at first visit, cleared
only by that slot's own safeguard ascents).  Implemented in
baseline_surf (steppers list + init flags) and verified: adam-core
smoke shows per-slot t = 51-66 (~rounds x epoch_len) and per-slot
alpha halvings — persistence real.  Campaign runner carries
CAMPAIGN_MU = 1e-3 / SURF_W_MIN = 0.05; homes renamed
ladders_mu0.001 / main_mu0.001.

S3 ladders launched on the ridge problem: uniform r / SURF N in
{10,20,30,40} x {adagrad x10, adam 1e-3/0.9} x seeds {41,141,241},
B = 2,500 -> (r*, N*) per core.  48 runs, ~1.4 h.

## S4 descope (user decision, Sep 3)

S4 = 2 cores x 3 legs x **ONE seed (41)** = 6 runs, ~1.6 h (was 18
runs / ~5 h with 3 seeds).  Seeds 141/241 are FUTURE WORK: MAIN_SEEDS
in the campaign runner is the single switch — extending it and
re-running --stage main fills in only the missing runs (resume-skip)
and the figures aggregate whatever seeds exist.  Caveat of record:
with one seed, legs finishing within seed-noise (~5-10 % per S2)
cannot be adjudicated; the tie-break then needs the future seeds.
S3 ladders stay at 3 seeds (already nearly finished when decided).

## S3 result (Sep 3, 48/48, 2,205 s total; board in
## v2_campaign/ladders_mu0.001/ladders_summary.json + ladders.png)

Mean final worst GN (norm, 3 seeds) per rung:

| core | family | 10 | 20 | 30 | 40 | pick |
|---|---|---|---|---|---|---|
| adagrad x10 | uniform | .1497 | .1192 | .1043 | **.0872** | r* = 40 |
| adagrad x10 | surf | .1033 | .1640 | .1057 | **.0894** | N* = 40 |
| adam 1e-3/.9 | uniform | .2386 | .1836 | .1337 | **.1013** | r* = 40 |
| adam 1e-3/.9 | surf | .2016 | .2279 | **.1939** | .2091 | N* = 30 |

Patterns: uniform improves monotonically with r for both cores —
r* = 40 is a LADDER-EDGE pick (finer grids may be better still; not
extrapolated, per pre-registration; also the pick is made at the
B = 2,500 tier and carried to B = 20,000 — a known smoke-tier
limitation of record).  SURF under core A is non-monotone with a bad
N = 20 rung but essentially ties best-uniform at N = 40 (.0894 vs
.0872); under core B SURF is uniformly weak (~.19-.23).  At this
budget the S2 adaptive reference (.0113, core A) leads both baselines
by ~8x.  Word report updated in place (S2 curves + S3 results/figure
inserted into the user-edited docx via XML surgery; the user's edits
preserved).

## S4 launch (Sep 3): core A only

User sign-off: picks accepted (A: r* = 40, N* = 40) AND core B (adam
1e-3/0.9) is DROPPED from the main run — its S2/S3 results stand as
recorded; MAIN_CORES in the runner resurrects it if ever wanted
(resume-skip).  S4 = core A x 3 legs (uniform@r40 / surf@N40 /
adaptive-CCP) x seed 41 x B = 20,000 = 3 runs, ~50 min, launched on a
quiet machine (CPU axis is a headline figure).  Outputs under
v2_campaign/main_mu0.001/adagrad_x10/: three run dirs +
worst_gn_curves.png + pareto_front.png.

## S4 + core-compare RESULTS (Sep 3 evening — campaign complete)

Core A (adagrad x10), B = 20,000, seed 41, 48 min: adaptive-CCP
**7.57e-3** vs uniform@r40 5.74e-2 (7.6x) vs SURF@N40 7.85e-2 (10.4x).
Core B (adam 1e-3/0.9, resurrected by user request; adaptive shared
with core-compare, only 2 new runs, 31 min): adaptive **5.80e-3** vs
uniform@r40 8.85e-2 (15.3x) vs SURF@N30 1.81e-1 (31.2x).

Readings: (1) both cores show the same shape — adaptive dives
monotonically, both baselines PLATEAU after B~2.5-5k and never improve
again; CPU axis identical story (lambda-search ~10% of wall).
(2) Front lens differs by core: under A SURF's frontier is visibly
dominated; under B all three frontiers nearly coincide while worst GN
differs 31x — the two lenses disagree, which is exactly why both are
reported.  (3) core-compare (adaptive leg, both cores, full budget,
ridge): adagrad leads early, crossover at B~6.5k, adam finishes 23%
ahead (5.80e-3 vs 7.57e-3, single seed) — the S2 preselection
structure REPLICATES on the ridge problem; using core A for the
headline three-leg comparison is conservative for adaptive (A is the
baselines' best core).  (4) Diagnostic: SURF's global L_scale reached
2^40 (~40 ascents share one global multiplier); per-slot L_scale filed
as future work.  (5) Front figures: windowed view (v1 convention,
vertex points annotated out-of-view) + uniform frontier dashed/on top
(it hugs the same band as adaptive's and vanished underneath).

Word report finalized (XML surgery on the user-edited docx, all user
edits preserved): 2.2 methodology-closure line, double-core S4
results, 4 figures, core-compare, campaign conclusion.  Figures under
v2_campaign/main_mu0.001/{adagrad_x10,adam_1e-3_b0.9}/ and
v2_campaign/core_compare_adaptive.png.

**Campaign conclusion**: at fixed budget the adaptive bundle
(lambda-search hitting the worst direction) leads both baselines by
an order of magnitude on worst GN (7.6-31x) with front quality no
worse; SURF's arc-length allocation did not deliver a coverage
advantage on this problem.

## Sep 3 late: methodological reordering + authoritative core selection

User concern (valid): the core was screened at mu = 0 (S2), the ridge
was adopted later for SURF's sake, and ridge also affects the cores —
the chronological order is circular.  Resolution: present the campaign
in LOGICAL order — (1) fix the problem (ridge mu = 1e-3, justified by
SURF boundedness independently of the core); (2) select the core ON
that problem; (3) tune baselines (S3, done for both finalist cores);
(4) main run (S4, done for both).  The mu = 0 S2 run is demoted to a
preliminary screen (appendix mention).  Step (2) is now being run as
the AUTHORITATIVE selection: stage `s2-extend` = ALL 11 S2 configs x
3 seeds x B = 10,000 (4x the screen tier) on ridge mu = 1e-3
(user decision: every config, because this figure replaces the S2
figure in the report).  33 runs, ~4 h.  Output:
stepper_pre_experiment/extended_B10000_mu0.001/ (s2_extended_curves.png,
s2_extended_summary.json).  Contingency: if the ridge top-2 is not
{adagrad x10, adam 1e-3/0.9}, rerun S3+S4 for the newcomer (~2.5 h);
otherwise every existing result stands.

(The 3-seed core-compare extension and the mu = 0 s2-extend were
stopped mid-run at the user's request and superseded by this stage.
`--stage core-compare` remains available for the ridge full-budget
two-core head-to-head with more seeds.)

Report restructured the same evening into the LOGICAL order (lxml
block moves on the user's file, now at
~/Desktop/Internship-UCB/Week11/K2_Campaign_实验设计与进展.docx):
2.1 pairs -> 2.2 problem (ridge + trim, moved up from old 2.3) ->
2.3 core selection on the ridge problem (rewritten bullets; red
placeholder for the pending 33-run result; interim note that S3/S4
already cover both finalists) -> 2.4 ladders -> 3 main -> Appendix A
= the mu=0 screen (old board table, S2 figure, judgement moved
verbatim).  Doc-surgery lessons of record: Word resaves rename media
files and rIds (locate images via the caption's preceding r:embed);
headings carry Word bookmarks (strip on copy); whitespace-normalise
anchor text; never anchor on a phrase that also appears in a table.

## Sep 3: K = 3 migration launched (user decisions)

* Problem: ridge, mu = 1e-4 (K3 house value; the Aug-26 const-core
  campaign triple_4v7v9_B40000_mu0.0001 — adaptive 1.88e-2, uniform
  r10/20/30 = 0.101/0.089/0.098 norm — is the built-in comparison set).
  K2's mu = 1e-3 was SURF-driven; no SURF at K = 3.
* S0 done ({4,7,9}, TAG Phase 1).  Cores INHERITED from K2 (adagrad x10,
  adam 1e-3/0.9); no core scan at K3 — full 11-config scan = future
  work.  S3 = uniform r in {10,20,30} rescanned under both cores
  (3 seeds, B = 5,000); no SURF leg.  S4 = per core {uniform@r*,
  adaptive-CCP}, seed 41, B = 40,000, const-core Exp-6 curves overlaid.
* New file run_k3_stepper_campaign_without_256_checkpoints.py: the
  Aug-26 ridge executor with the four stepper hooks (stepper_core is
  K-agnostic), sampler-seed param, additive summary fields; stages
  gate0 / gate1 / ladders / main.  Gates running; ladders+main queued
  behind (a) both gates passing and (b) the K2 s2-extend finishing
  (waiter checks gate reports + process table).  Est. ~1.7 h ladders +
  ~2.5 h main.

Incident (Sep 3 night): the K2 s2-extend leg `bb seed141` aborted at
checkpoint 42 on the executor's hard assert "exact prefix audit not
monotone" — a 1e-12-level jitter of the exact meter's closed-form
polish on long (~1,600-gram) stacks, not a real violation.  Fix: the K2
stepper executor and the SURF executor now WARN and record
`audit_mono_violations` (the K3 executor's convention) instead of
asserting; the stage resumed (4/33 kept).  Second lesson: the K3
waiter keyed on "no s2-extend process" fired on the crash and started
K3 ladders early (stopped within a run); the re-armed waiter keys on
the s2-extend completion artifact (s2_extended_summary.json) instead.

## Authoritative core selection RESULT (Sep 4 morning; 33 runs, 207 min)

Ridge mu = 1e-3, B = 10,000, 3 seeds, adaptive-CCP leg, mean final
worst GN (norm): adam 1e-3/0.9 **8.00e-3** (seeds 7.88-8.09e-3) <
adagrad x10 8.82e-3 (8.64-9.10e-3) < adam 1e-3/0.99 8.89e-3 <
adam 3e-4/0.9 9.62e-3 < adagrad x3 9.71e-3 < adam 3e-4/0.99 1.09e-2 <
const 1.14e-2 = adagrad x1 1.14e-2 < adam 1e-4/0.9 1.50e-2 < bb 1.54e-2
< adam 1e-4/0.99 1.78e-2.  Readings: (1) top-2 = the screen's top-2 ->
every S3/S4 result stands as run; (2) adam's late edge PERSISTS and
grows on the ridge problem: adam/adagrad ratio 1.20 at B=2,500
(behind) -> 0.91 at B=10,000 (9 % ahead, seed ranges disjoint); the
crossover sits near B~7,000 — consistent with core-compare's ~6,500 at
B=20,000; (3) the ridge compresses the field: const trails the winners
by only ~1.4x here (2.5x in the mu=0 screen at B=2,500), bb still near
the bottom, adam alpha=1e-4 last.  K2 report 2.3 placeholder filled by
update_k2_core_selection.py (data-driven).  K3 ladders started
automatically right after (waiter keyed on the completion artifact).

## K3 migration RESULTS (Sep 4; ladders 18 runs + main 4 runs)

Ladders (B = 5,000, 3 seeds, norm): adagrad r10/20/30 = .239/.507/1.64,
adam .192/.447/.916 -> r* = 10 for both — monotone WORSE with r (the
opposite of K2): at this tier 231/496-node grids get too few visits.
Main (B = 40,000, seed 41, norm): adaptive/adagrad **1.19e-2**,
adaptive/adam 1.38e-2, const adaptive (Exp 6) 1.88e-2 -> core gain
1.59x (adagrad) / 1.37x (adam); uniform_r10/adam 7.41e-2, const
uniform r20 (Exp 6) 8.88e-2, uniform_r10/adagrad 1.74e-1 (!).  The
r*=10 pick mis-transfers to the full budget: the 66-node grid
plateaus from B~5k (coverage-limited), so the adagrad-core uniform at
r10 is WORSE than Exp 6's const r20.  Fix of record: MAIN_EXTRA_RS=[20]
— the full-budget uniform arm is also run at r=20 for both cores (2
runs, ~1.2 h); the K3 report takes each core's best uniform r for the
ratios and states the ladder-transfer lesson.  Note at K3 adagrad
beats adam at the final point (single seed), the reverse of K2's
late adam edge.

K3 main FINAL board with the r=20 arms (Sep 4): adaptive/adagrad
1.19e-2, adaptive/adam 1.38e-2, const adaptive 1.88e-2; uniform_r20/adam
7.19e-2, uniform_r10/adam 7.41e-2, uniform_r20/adagrad 8.05e-2, const
r20 8.88e-2, uniform_r10/adagrad 1.74e-1.  With the best uniform r per
core, adaptive leads uniform 6.8x (adagrad) / 5.2x (adam); both
adaptive-core uniforms now beat the const-core uniform.  K3 report
generated (data-driven docx-js, 5 pages) and deployed to
~/Desktop/Internship-UCB/Week11/K3_Campaign_实验设计与进展.docx.

Sep 4 report tweak (user request): the K2 ladder figure in the report
now shows the adam 1e-3/0.9 core only — new figure
v2_campaign/ladders_mu0.001/ladders_adam_1e-3_b0.9.png from the new
script experiment_plot/plot_ladders_single_core_K2_without_256_
checkpoints.py (reads ladders_summary.json; same 1650x600 size, so the
docx media bytes were swapped in place, caption updated; the user's
latest edits — including their relabelling of the caption to "S2" and
removal of several figures — preserved).

Sep 4 (user): "B = 5,000 per ladder run is too small at K3" — agreed
(66/231/496 nodes; the small tier inverts the pick).  Decision: the K3
uniform resolution is selected AT the main budget — r in {10,20,30},
B = 40,000, seed 41, both cores (MAIN_EXTRA_RS = [20, 30]; r10/r20
already existed, only r30 x 2 cores new, ~1.2 h).  The B = 5,000 ladder
stays in the report only as the documented lesson; K3 report 2.4
rewritten accordingly (data-driven from main_summary.json).

Sep 4 (user, minutes later): the K3 main line (full-budget r selection
+ S4) is the ADAM core only — MAIN_CORES = [adam]; the two-core run was
stopped (adagrad r30 abandoned mid-run), r30/adam is the only new run
(~35 min).  The adagrad r10/r20/adaptive runs stay on disk as reference
(the report mentions adagrad's adaptive number only as an aside).

## Future work of record

Seeds {141, 241} for S4 (one-switch resume); per-slot L_scale for the
SURF leg; finer uniform r (ladder-edge pick); K3 full 11-config core
scan; ladder tier vs full-budget transfer (K3 showed the 1/8-tier pick
can invert at full budget — pick r on a longer budget or report all r); SURF at K >= 3 (surface-density generalisation); K >= 3 migration of the
adaptive cores (plan discussed Sep 3: reuse stepper_core, wire the K3
CCP engine, gate, 3-config re-check on {4,7,9}, adaptive+uniform legs
first — SURF at K = 3 needs a surface-density generalisation).

Env reminders: launch with the venv 3.13 interpreter explicitly
(run.sh points at an inner 3.11 venv); worst-GN grids 200,001 / 20,001;
every long run only after the stage sign-off.

## Sep 3, late morning (the "Sep 4" labels above are the same working
## day by the machine clock): K3 adam main line FINAL + paper draft

r30/adam finished (1,841 s wall): final audit 8.60e-3 (norm 9.27e-2).
Adam-core board at B = 40,000, seed 41 (main/adam_main_summary.json):
  adaptive-CCP 1.375e-2 | uniform r10 7.41e-2 | r20 7.19e-2 | r30 9.27e-2
  -> r* = 20 at the main budget (the B = 5,000 ladder had picked r* = 10);
  adaptive leads the best uniform by 5.2x.  const-core (Exp 6) contrasts:
  adaptive 1.88e-2 (1.4x gain from the core swap), uniform r20 8.88e-2.
Figures (plot_k3_adam_main_without_256_checkpoints.py -> repo output
v2_stepper_mu0.0001/main/): worst_gn_curves_adam.png (adam only:
adaptive + uniform r10/20/30, no const lines) and pareto_front_adam_3d.png
(3-D frontier sheet, adaptive vs uniform r = 20, train loss space, 3 views).
K3 report (~/Desktop/Internship-UCB/Week11/K3_Campaign_实验设计与进展.docx)
updated by XML surgery on the user's 10:26 version
(scratchpad/update_k3_report_adam_v2.py; backup
k3_user_current_0903_1026.docx).  Touched ONLY: 2.4 (two protocol lines,
table = main-budget values + r* = 20, the B = 5,000 ladder caption
relabelled as the contrast), the section-3 table (adam rows + const
contrasts; adagrad rows dropped from the table, numbers stay in
main_summary.json), the 换核收益 paragraph, the main figure (bytes +
caption) and the NEW 3-D figure + caption.  validate.py PASS; rendered
and checked page by page.

Paper (Overleaf, Section 4.1 rewrite): Overleaf Essay/section_4_1_draft.tex
— Block 1 overview (one short paragraph, per user: "just say what the
two experiments are"), Block 2 preliminaries (lookahead affinity, SVRG
direction + the 4 step rules, SURF), Block 3 K = 2 (instance table,
objective with mu = 1e-3, settings, step-rule selection = full 11-row
B = 10,000 table with seed ranges, ladder, results, front), Block 4 K = 3
placeholder.  User rule: B = 2,500 must not appear in the step-rule part
-> new figure s2_extended_curves_paper.png
(plot_s2_extended_paper_K2_without_256_checkpoints.py; same data, no
2,500 marker, readable labels, chosen/competitor/incumbent highlighted).
Cite keys: fifty2021efficiently / standley2020tasks already exist in the
paper .bib; jiang2026surf, kingma2015adam, duchi2011adagrad, barzilai1988
are new (entries in the tex tail; [? ?] until added).

## Sep 3 (system clock; the "Sep 4" entries above are this same working day) — K3 adam-only main line CLOSED + paper Section 4.1 draft

K3 r=30/adam finished (1,841 s wall; L_scale=inf diagnostic as before):
uniform_r30 final worst GN (norm) 9.274e-2.  Adam-core board at
B=40,000, seed 41: adaptive 1.375e-2 | uniform r10 7.405e-2 | r20
7.186e-2 | r30 9.274e-2 -> r* = 20 selected AT the main budget (the
B=5,000 ladder had picked r=10; r=30 is worst: 496 nodes, too little
budget per node).  Adaptive leads the best uniform 5.2x; const controls
(Exp 6): adaptive 1.883e-2 (core gain 1.4x), uniform r20 8.875e-2
(1.2x).  New script experiment_plot/plot_k3_adam_main_without_256_
checkpoints.py -> main/worst_gn_curves_adam.png (adaptive + r10/20/30,
adam only, no const lines), main/pareto_front_adam_3d.png (K3 plotter's
Delaunay sheet, adaptive vs uniform r=20, train loss space, 3 views),
main/adam_main_summary.json.

K3 report updated by XML surgery on the user's edited file only (backup
scratch k3_user_current_0903_1026.docx; script update_k3_report_adam.py):
2.4 parameter + selection lines and table (main-budget values, r*=20),
section-3 config line, result table (adam + const controls only), ratio
paragraph (+ ladder lesson), main figure bytes swapped + caption, NEW
3-D sheet figure + caption after it, future-work bullet on the 3-D
figure.  Untouched: overview table (S3 row still says "1核×3档×3种子=9
run") — user's own text, flagged to them.

Paper: ~/Desktop/Internship-UCB/Overleaf Essay/section_4_1_draft.tex
(4 blocks: overview / preliminaries / K=2 / K=3 placeholder).  User
rulings: overview = one short paragraph (only what the two experiments
do); the step-rule selection part must NOT mention B=2,500 (the screen
budget is dead) -> new figure script experiment_plot/plot_s2_extended_
paper_K2_without_256_checkpoints.py -> stepper_pre_experiment/
extended_B10000_mu0.001/s2_extended_curves_paper.png (no marker;
adam(1e-3,0.9) / adagrad x10 / const highlighted); table = full 11-rule
ranking at B=10,000 with seed ranges.  Cite keys: fifty2021efficiently /
standley2020tasks already exist in the paper's .bib; jiang2026surf (use
the paper's [JHC26] key if it exists), kingma2015adam, duchi2011adagrad,
barzilai1988 are new entries (listed in the tex tail).  Labels still to
map by the user: eq:eps-sm-stat (paper eq. 7), alg:uniform (Algorithm 7).
K=3 block of the paper: not yet written (needs the numbers above).

Sep 3, ~11:30 (user, 3rd K3-report pass): (a) the 3-D sheet figure's
suptitle was too long -> shortened in plot_k3_adam_main_without_256_
checkpoints.py ("Pareto frontier sheets, MNIST 4/7/9, adam core,
B=40,000: adaptive CCP vs uniform r=20"; construction details now live
only in the report caption), figure regenerated (1912x713 px);
(b) an analysis block was added under the figure's caption (3 short
paragraphs; numbers from grams.npz fvals via the K3 plotter's
_nondominated_kd/_hv_3d): knee points adaptive (0.021,0.022,0.022) vs
uniform (0.024,0.022,0.023); in-window non-dominated 604 vs 763; HV ref
(ln3)^3 1.271 vs 1.277 (0.5%, a tie); the green spikes = uniform's
lambda_9~0 grid nodes (F9>0.1: 23 pts, max 0.98 vs adaptive 8 pts, max
0.19) bridged to the corner by the (F4,F7)-plane triangulation, not a
real surface; conclusion = same as K2 (coverage tie, worst GN 5.2x).
Surgery script update_k3_report_front3d_v3.py on the user's 11:29
resave (Word renamed media -> image via the caption's preceding blip
rId5/media/image2.png; extent cy recomputed from the new pixel aspect);
backup k3_user_current_0903_1129.docx; validate PASS, rendered OK.
Paper draft: floats changed from [t] to [!htb] (tables had floated
above the subsection heading in Overleaf).
