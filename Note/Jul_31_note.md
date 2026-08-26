# Jul 31 note — mean-variance K=2 bandit toy (nonconvex objectives)

Session of July 31, 2026.  User-directed: "on the original K=2 bandit
experiment, change the objective to a mean-variance objective, then run
the same experiment at eps 1e-2 and 1e-3, and report how long it
takes."  This note is the change record for all code written this
session.  NO existing file was modified; the July-26 convex instance,
its outputs and its bit-reproducibility are untouched.

## 1. Objective (design agreed in the July-26/27 session Q&A)

Same instance data as `objectives_bandit_toy` (A = 5 arms, T = 1000
balanced offline rows, noise std 0.5, data seed 7, plug-in per-arm
means R_hat), plus plug-in per-arm SECOND moments

    S_hat[k, a] = mean over rows t with a_t = a of r_{k,t}^2.

With pi = softmax([theta, 0]) unchanged:

    F_k(theta) = tau*KL(pi||pi_ref) - <pi, Rk_hat> + gamma*VarHat_k(pi),
    VarHat_k(pi) = <pi, Sk_hat> - <pi, Rk_hat>^2
                 = plug-in variance of the reward drawn under pi
                   (within-arm noise + between-arm spread).

gamma = 0 reproduces the July-26 objective BIT-FOR-BIT (sanity check 1).

Why the closed form dies: the scalarized objective adds
-gamma * sum_k lam_k <pi, Rk_hat>^2 — concave quadratics in pi — so
F_lam is no longer convex-plus-linear in pi, no softmax closed form
exists, and scalarizations can carry several local minima.  Stationary
points stay interior (any stationary pi solves a softmax fixed-point
equation with bounded effective logits), so the reduced-logit
parameterization still covers every optimum.

Mechanism of the multimodality: committing to ONE arm costs only that
arm's noise variance; MIXING two good arms adds between-arm spread.
The variance penalty therefore pushes toward near-single-arm policies
and the reward pull picks WHICH arm — competing "commit to arm i"
basins with comparable values.

## 2. Finite-sum / MSVRG treatment (exact-deterministic part)

    f_{k,t}(theta) = tau*KL + gamma*VarHat_k(pi)
                     - (T/N_{a_t}) * pi(a_t) * r_{k,t}

so F_k = (1/T) sum_t f_{k,t} exactly.  KL AND the variance penalty are
arm-level plug-in statistics carried exactly in every row (the July-26
oracle already did this for the KL; zero minibatch variance), the
mean-reward term is estimated from rows exactly as before.  In the
MSVRG update v = g_S(y) - g_S(anchor) + mu(anchor) the exact MV part
cancels between g_S(anchor) and mu(anchor) and survives exactly at y:
unbiased, variance-reduction identity untouched.  Full-batch estimator
== analytic gradient to 1e-16 (sanity check 6).

## 3. Ground truth: never-timed multi-start reference (best-known pool)

Per the session-12 ruling ("ground truth then = never-timed multi-start
reference solves"):

* Reference table on the dense w-grid (n = 5000): vectorised Adam over
  (every w) x (~29 structured + random starts) simultaneously, scipy
  L-BFGS-B polish (analytic gradients) of every winner, then ascending
  + descending relax sweeps (polish from the neighbour's solution,
  accept improvements) until fixpoint, so a basin found anywhere
  propagates along the whole path.  Cached
  (`reference_gamma<g>_n5000.npz`), shared by BOTH eps rungs.
* Structured starts include the gamma = 0 closed form and scaled
  one-hot logits per arm (the commit-to-arm basins).
* The table is BEST-KNOWN, not a certificate.  After ALL rungs run,
  every method-delivered point seeds an L-BFGS polish at every table +
  evaluation weight; improvements > 1e-9 update the table + cache and
  are counted in summary.json (`mean_variance.reference_refresh`).
  delta_value(w) >= -1e-9 by construction afterwards.
* The SURF Eq.-9 speed formula was closed-form-specific; the arc CDF
  (and the Rule-1 arc-uniform reference weights) now use the CHORDAL
  arc length of the reference front.  The statistical layer gains
  `second_moment_sup_gap` = ||S_hat - S_true||_inf with
  S_true = R_true^2 + noise_std^2.
* True-parameter twin table built the same way (statistical layer).

## 4. New files (nothing existing touched)

* `Original_py/objectives_bandit_toy_mv.py` — BanditToyMVProblem
  (subclass; closed-form *_lam API poisoned to fail loudly),
  BanditStochOracleMV, reference machinery, cache, refresh.
* `Original_py/run_bandit_toy_mv_without_256_checkpoints.py` — driver:
  July-26 pipeline verbatim (same engines, same eps ladder, same
  equal-level stop 2eps/3, same figure set fig1-fig5 + 4b/4c, same
  summary/raw_histories layout) with the reference-based oracle layer;
  NEW: --epsilons runs several rungs serially in ONE invocation so both
  rungs score against the same final reference; --gamma-scan mode.
* `Original_py/sanity_checks_bandit_toy_mv.py` — 13 checks.
* Output home: `output/bandit_toy_mv_without_256_checkpoints/`
  {smoke, eps1e-2, eps1e-3, gamma_scan.json, reference caches,
  record_run_console.log}.

## 5. Sanity results (July 31): ALL PASS 13/13

Highlights: gamma=0 objective/gradients/joint AND stochastic oracle
bitwise-identical to the July-26 modules; MV analytic gradient vs
finite differences worst rel err 7.6e-9; full-batch oracle gradient
exact to 1.9e-16; minibatch unbiased (4000-draw mean, rel err 1.4e-3);
reference at gamma=0 matches the closed form to 9.0e-13 (scalarized
opt) / 2.8e-8 (front); MV reference points stationary to 1.3e-9;
nonconvexity realised; chordal CDF monotone; cache round-trip bitwise.

## 6. gamma choice: scan + a metric caveat, gamma = 1.0 chosen

`--gamma-scan` (n_dense 1001, 41 anchor weights, 29-start pools):

| gamma | front jump (at w) | pool bimodal frac | L1/L2 |
|---|---|---|---|
| 0.25 | 0.012 @ 0.669 | 0.93 | 0.141/0.118 |
| 0.5  | 0.489 @ 0.670 | 1.00 | 0.143/0.109 |
| 1.0  | 0.765 @ 0.665 | 1.00 | 0.148/0.100 |
| 1.5  | 0.787 @ 0.661 | 1.00 | 0.153/0.111 |
| 2.0  | 0.801 @ 0.658 | 1.00 | 0.158/0.127 |

CAVEAT (recorded so nobody quotes the wrong column): the scan's raw
"basin counts" cluster by THETA distance and are inflated by softmax
plateau degeneracy (far-apart thetas representing the same
near-vertex pi; median value gap between "basins" ~1e-15).  The honest
multimodality evidence is (a) the front JUMP (a jump requires two
competing basins with crossing values) and (b) a pi-space recount at
gamma = 1: at w = 0.66 THREE pi-distinct minima within 0.005 of each
other (mix-on-arm-4 vs pure-arm-4 vs arm-5); between w = 0.66 and 0.68
the global minimum teleports from arm-4-mix (pi ~ [0,0,.05,.94,.01])
to arm-5 (pi ~ [0,0,0,.01,.99]); the loser basin persists to w = 1.

Decision: gamma = 1.0 — jump already 96% of the gamma=2 size, L
essentially unchanged vs the convex instance (0.148/0.100 vs
0.149/0.141), and the unit value reads cleanly.

## 7. Machine conditions (honest-reporting order)

Third-party load was present all session (WindowServer + ChatGPT/Codex
+ Messages; 1-min loadavg 3.3-11.3).  The July-30 settle gate (wait 5
min for loadavg < 3) did NOT settle before the record launch.  Per the
July-30 precedent the record runs were launched anyway with this
disclosure: CPU-AXIS readings (fig1, cpu-to-eps, wall/process seconds)
are ESTIMATES ONLY; grads-axis readings, GN*, value metrics and fronts
are load-independent and record-grade.  A clean re-run of the same
command once the machine is idle supersedes the CPU numbers (the
reference cache makes it cheap).  Runs are serial in one process under
caffeinate; smoke and scan ran under the same load.

## 8. Record runs (gamma = 1.0, eps 1e-2 + 1e-3, --eval-every 0): RESULTS

One invocation, serial under caffeinate; reference build 174.6 s
(cached, n = 5000), method work ~2 s total, refresh 23.2 s, scoring +
figures ~40 s — whole record invocation ~4 min.  Both rungs clean:
censored 0, safeguard retries 0, L_scale 1.0 end-to-end, adaptive
inner-cap hits 0.  Reference refresh: NONE of the 67 method-delivered
seed points beat the multi-start reference anywhere (the table stood).

| rung | method | stop (mechanism) | grads total | grads-to-eps | final GN* | eps_value final | front IGD |
|---|---|---|---|---|---|---|---|
| 1e-2 | baseline | global_stop_gn — 1 node solved, 12/12 by share | 32.4 | 16.2 | 2.770e-3 | 6.363e-1 | 0.408 |
| 1e-2 | adaptive | epsilon_certified (bundle 12 -> 2) | 67.1 | 18.3 | 4.499e-3 | 3.849e-1 | 0.324 |
| 1e-3 | baseline | global_stop_gn — 1 node solved, 12/12 by share | 72.9 | 56.7 | 6.297e-4 | 6.363e-1 | 0.404 |
| 1e-3 | adaptive | epsilon_certified (bundle 27 -> 2) | 158.5 | 140.2 | 6.659e-4 | 3.291e-1 | 0.302 |

CPU-axis columns exist in the summaries but carry the section-7 dirty-
machine caveat; the table above quotes load-independent axes only.

### Finding 1 — PLATEAU-INFLATED CERTIFICATES (the headline)

At BOTH rungs the baseline solved exactly ONE node (w = 0, snake
start); its whole delivered chain descends into the commit-to-arm-2
basin (final pi ~ [0.11, 0.83, ...]) and the equal-level stop
(strict full-simplex GN* <= 2eps/3, checked at the per-segment
cadence) fired from that single chain: near a softmax vertex the
Jacobian J = Diag(pi) - pi pi^T squashes the theta-gradients of EVERY
objective, so the deeper the chain commits to arm 2, the smaller ALL
lambda-scalarized gradient norms get — at eps1e-2 the lam = (1, 0)
certificate is signed by the arm-2 point with ||grad F1||^2 = 2.77e-3
whose F1 VALUE is +0.08, i.e. 0.83 above the reference F1* = -0.754
(the meter was re-verified independently: recomputed strict GN* over
the delivered Grams = 2.7696e-3 = the stored value; worst-lambda
(1, 0); no instrument bug).  A single wrong-basin chain can sign the
WHOLE simplex at ANY eps level; more eps just rides the plateau
deeper (eps1e-3: same story, GN* 6.3e-4, 18 segments, still 1 node).
This upgrades the session-13 structural fact (convex run: 2 of 12
nodes chain-processed, all by share) from a curiosity into a failure
mode: under convexity the closed-form uniqueness made share-signed
certificates value-valid; mean-variance removes that protection and
"share + plateau" signs value-catastrophic weights.  It is also the
value-side, quantitative version of the user's Q4 ruling: terminal
GN* must NEVER be cited as coverage evidence — here GN-axis
readouts (baseline 16.2 / 56.7 grads-to-eps, "beating" the adaptive)
coexist with a 0.636 value gap at the weight its own grid nominally
covers (w = 1 is a grid node; the baseline's best answer there is x0
itself, F1 = -0.118).

### Finding 2 — basin capture and the eps-invariant value gap

The baseline's eps_value final is IDENTICAL at both rungs (6.363e-1,
peak at w = 1.0): tightening stationarity 10x moved its value profile
not at all — the gap is a BASIN gap (its delivered set has no point in
the arm-4 or arm-5 basins), and no amount of within-basin polishing
shrinks it.  The adaptive halves the miss (0.385 -> 0.329; profile
peak also at w = 1) because its worst-lambda targeting plants bundle
points in TWO basins (arm-4-mix pi ~ [0,0,.07,.75,.15] and arm-2-ish
pi ~ [.13,.79,...]) — but NEITHER method ever discovers the arm-5
basin that owns the global optimum for w > 0.665 (jump w+): coverage-
direction distances to the reference front peak at the w = 1 reference
point (-0.754, 0.373): baseline 0.92, adaptive 0.67-0.71.  The
reference table holds arm-5 exclusively via its structured one-hot
starts.  Min raw value gaps are POSITIVE everywhere (bl 4.65e-2 /
2.07e-2, ad 5.74e-2 / 2.35e-2 at 1e-2 / 1e-3): at these GN levels no
delivered point matches the reference even in its own basin
(within-basin residue at the gradient tolerance, halving as eps
tightens — consistent, not a bug).

### Finding 3 — the nonconvex instance separates the methods on the
value/coverage axes, not the GN axis

GN axis: baseline reaches every stop line first (plateau assist).
Value/front axes at BOTH rungs: adaptive better everywhere — eps_value
0.33-0.38 vs 0.64 (flat), front IGD 0.30-0.32 vs 0.40-0.41, coverage
max-miss 0.67-0.71 vs 0.92.  Mirrors the convex-track lesson
(value/PF metrics, not GN, expose coverage) with a nonconvex
mechanism: the adaptive's targeting DIVERSIFIES basins, the baseline's
share machinery COLLAPSES into one.

### Follow-up candidates (recorded, NOT launched — user decides)

1. Native-protocol baseline leg: disable the equal-level stop (its
   July-26 courtesy semantics are what the plateau exploits) and let
   the baseline run its full per-node sweep at the same rungs — does
   node-by-node solving with chain warm starts find arm-4/arm-5, or
   does hysteresis trap it anyway?
2. Basin-aware GN reporting: quote GN* jointly with a pi-concentration
   readout (e.g. max_i pi_i of the signing point) so plateau-signed
   certificates are visible in tables.
3. Clean re-run for the CPU axis once the machine is idle (same
   command; reference cached; ~1 min).
4. eps1e-4 rung (cap 256) if the ladder should match the convex
   track's three rungs.
