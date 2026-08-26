# Aug 13 note — K = 2 MNIST pair campaign: design freeze + Smoke A

Session of Aug 13, 2026.  Planning sessions of Aug 12-13 froze the
design for the next experiment; today the first code landed (NEW
files only, nothing overwritten) and Smoke A (conflict probe) ran.

## Experiment in one line

K = 2 MNIST digit-pair MOO under the pure fixed-budget protocol:
adaptive(CCP) vs multi-r grid baselines (NO IPOPT leg), training
figures (GN vs grads / GN vs CPU / Pareto front) plus a NEW test-side
evaluation (front thetas re-scored on the official test split).

## Design decisions (user-approved, Aug 12-13 Q&A)

* Borrow ONLY the task/data idea from Reddi et al. 2016
  (Reference_essay/Stochastic_Variance_Reduction_for_Nonconvex_
  Optimization.pdf): MNIST, pixels /255, official train/test split,
  no augmentation.  Algorithm, protocol, seeds, init, batching all
  stay the repo's own (Aug-9 conventions).
* K = 2 via a digit PAIR (2-logit softmax head), chosen for a clean
  2-D Pareto front.  Objectives: per-class mean CE, s_k ≡ 1, NO
  regularisation (confirmed: the repo's objectives never had one).
* Architecture: patch net FIRST (patch-64 5x5 -> softplus -> dense 96
  -> softplus -> 2 logits, d = 8,098).  The paper's FC-100 (d = 78,702
  at K = 2, ~10x compute) stays a later option; conflict ladder if the
  front degenerates: confusable pair -> label-flip -> ah16_faithful
  (d = 1,794).
* Data scale (Aug-13 revision): take the balanced MAXIMUM per pair —
  per_class = min(count_a, count_b), i.e. 5,421-5,949/class (MNIST
  train holds 5,421-6,742 per digit; "more than 5,000" is capped by
  the dataset itself).  Test = ALL official t10k rows of the two
  digits (~1,000/class — that IS the full official test; a bigger
  test set would require QMNIST, deliberately not adopted).  Batch
  stays 1024 (stratified 512/512); epoch_len follows ceil(n/1024)
  (= 11-12), so the segment shape is essentially the Aug-9 one.
* Legs: adaptive_s5_ccp (CCP production defaults N0=2000, r=10, exp
  sampler, pool on, adaptive schedule off) + baselines r in
  {10, 20, 40} (Aug-13 revision: r=5 dropped as too coarse), snake
  order, cycling.  s = 5 everywhere if coverage allows; rule when B
  is tight: cut that leg's s first (coverage inequality
  (r+1)*s*c <= B), dropping a leg only as a last resort.
* Quality meter: exact GN via the 1-D lambda grid (K=2 exact meter,
  200,001-point convention) — no audit_v2 needed at K = 2.
* NEW measurement requirement: theta snapshots (chain point per
  segment end) so front points can be re-scored on test rows.  Test
  figures: train+test front overlay in per-class mean CE (main), test
  per-class error front (1-recall_A, 1-recall_B) (secondary), test
  metric vs budget/CPU curves (the paper's "test error vs effective
  passes" analogue).  Test-side GN plots ruled out (GN is a
  stationarity meter for the TRAINED problem; paper separates the
  axes the same way).
* Budget accounting reconfirmed from _Budget: joint call = K units,
  minibatch row pair = 2K/n; support drop charges only consumed rows.
  A-priori c at K=2: interior-lambda ~6.1-6.2, vertex ~4.05.

## Conflict score (Smoke A ruler)

score = [F_A(end of lam=(0,1)) - F_A(end of lam=(1,0))]
      + [F_B(end of lam=(1,0)) - F_B(end of lam=(0,1))], all / ln 2.

Each bracket = how much that class's CE rises when the optimizer
fully ignores it vs fully favours it — i.e. the front's extent along
that axis.  ln 2 (~0.693) is the guess-level CE of a balanced 2-class
problem (log 1/2), so the normalised score reads as a fraction of the
perfect-to-random scale.  Degenerate front <=> score ~ 0.  Shape
check: the five chain ends must order monotonically (F_A down, F_B up
as lam_A grows); violations counted.

## New files (Aug 13)

* ``objectives_mnist_pair.py`` — pair loader (train: first per_class
  rows per digit, per_class=None -> balanced max; test: all t10k rows),
  ``PairPatchMLP`` (2-logit head), ``make_pair_initial_point``,
  ``PairStochLamOracle`` (verbatim mirror), ``evaluate_pair`` (fixed-
  theta per-class CE + error — the test-front primitive),
  ``make_mnist_pair`` factory (L probes, meta carries _X/_y so callers
  can build fresh oracles with identical batch streams).
* ``run_conflict_smoke_K2_mnist_pairs_without_256_checkpoints.py`` —
  Smoke A: 5 pairs x 5 fixed lambdas x 15 segments, independent chains
  from the shared He x0, fresh sampler (seed 41) per lambda so all
  lambdas see identical batch streams; verbatim executor segment;
  wiring check (full-batch stoch grad == joint scalarized grad,
  <1e-10) at every pair's x0; c read off spent() diffs and asserted
  against the exact formula.  Outputs under the NEW campaign home
  ``output/K2_mnist_pair_without_256_checkpoints/conflict_smoke/``.

## Smoke A results (run 96 s; re-run after the interior-score fix)

All checks green on every pair: wiring rel err ~5e-15, c matches the
formula exactly (interior 6.13-6.21, vertex 4.07-4.10 — n-dependent),
mono violations 0, L probes in [1.00, 1.45].  Safeguard: mild and
healthy — 5 of 25 chains fired it exactly once each (L_scale ends at
2.0), all at interior lambdas (4x at (.75,.25), 1x at (.5,.5) on
7v9); vertex and near-B chains never fired.

FINDING (Aug-13): the vertex score is divergence-dominated.  With no
regularisation, the fully-ignored class DIVERGES — after only 15
segments, lam=(1,0) ends at F_B in [14.2, 17.4] and lam=(0,1) at
F_A in [26.5, 31.3] — so the vertex score lands in a narrow 63-70
band for all five pairs and ranks divergence speed, not
confusability.  The INTERIOR score (between (.75,.25) and (.25,.75),
both classes keep weight, nothing diverges) is the discriminating
ruler; the script was amended the same day so top2 uses it (vertex
kept as a diagnostic).  Consequences for the campaign: (a) NO
degeneration anywhere — every pair shows a wide, monotone interior
front (S_int 0.72-0.90, i.e. 72-90 % of the guess-level ln 2 scale),
so the conflict ladder (label-flip / ah16_faithful) is NOT needed;
(b) baseline vertex nodes will produce runaway arms in the formal
run — front figures should window to roughly [0, ln 2]^2.

| pair | S_int | A part | B part | (.75,.25) end | (.5,.5) end | (.25,.75) end |
|---|---|---|---|---|---|---|
| 3 vs 5 | 0.9007 | 0.3290 | 0.5717 | (0.135, 0.481) | (0.191, 0.169) | (0.363, 0.085) |
| 7 vs 9 | 0.8425 | 0.3334 | 0.5092 | (0.122, 0.442) | (0.219, 0.211) | (0.353, 0.089) |
| 3 vs 8 | 0.8099 | 0.3297 | 0.4802 | (0.126, 0.418) | (0.183, 0.163) | (0.354, 0.085) |
| 4 vs 9 | 0.7274 | 0.3422 | 0.3852 | (0.089, 0.362) | (0.160, 0.175) | (0.326, 0.095) |
| 5 vs 8 | 0.7247 | 0.3002 | 0.4246 | (0.087, 0.377) | (0.146, 0.159) | (0.295, 0.083) |

**top2 = 3-5 and 7-9** (disjoint digits — two independent instances).
Caveat recorded: at 15 segments the interior spread mixes true
conflict with finite-budget lag (the down-weighted class also just
trains slower); the ranking is comparative at equal budget, the real
fronts come from the campaign.  Consistent asymmetry: the B part
exceeds the A part on every pair (second digit suffers more when
down-weighted).

Wall economics measured: ~0.22 s/segment (build ~2.5 s/pair, whole
smoke 96 s), c_interior <= 6.21.

## B/r/s proposal (from measured c, awaiting user sign-off)

B = 20,000 (the K2-planted convention) for every leg; s = 5
everywhere — coverage is comfortable: passes per grid cycle at
c = 6.21 are r10 ~58, r20 ~31, r40 ~16, all >> 1.  Per-node depth
B/((r+1)c): r10 ~293, r20 ~153, r40 ~79 segments.  Adaptive: ~644
decisions, ~3,220 segments/leg.  Wall estimate: ~12 min/baseline leg,
~17 min adaptive (CCP ~0.45 s/decision), 4 legs x 2 pairs ~2 h + audit
/ test eval / figures.  eval_every = 250 (K2 convention);
theta snapshots per segment end ~200 MB/leg (float64, savez).

## Formal campaign (Aug 13, user go "Okk跑吧"; ALL DONE in 7,583 s)

New files: ``run_pure_budget_K2_mnist_pair_without_256_checkpoints.py``
(executor = K2 replica + theta snapshots per delivered point +
official-test evaluation, both off-axis; legs baseline r10/20/40 s5 +
adaptive_s5_ccp; resume-skip; --smoke = Smoke B) and
``plot_K2_mnist_pair_without_256_checkpoints.py`` (5 figures/pair +
front_metrics.json + README).  Smoke B green in 9 s (budget
conservation, seg costs on-formula, THETA ROUND-TRIP 1e-9, test eval
sane, exact audits certified, figures render).  Homes:
``output/K2_mnist_pair_without_256_checkpoints/pair_{3v5,7v9}_B20000``.

Fairness audit (all 8 legs): spent 19,994-19,998 of 20,000; every
segment cost equals the vertex/interior formula; x0 bit-identical
across legs per pair; decision_seconds 0.0 on baselines vs ~87-88 s on
CCP (~11 % of wall; 0.135 s/decision at K=2, vs 0.45 s at K=10);
L_scale 1.0, zero safeguard retries; certified audit gaps <= 1.9e-5.
NOTE the support effect REVERSED vs K10: baselines got MORE segments
(3,459-3,476 vs CCP 3,252-3,266) because the grid's two vertex nodes
buy cheap segments while CCP camps on interior worst-w.

### Results — final EXACT audit (GN*) at B = 20,000

| leg | 3v5 | 7v9 |
|---|---|---|
| baseline r10 | 1.733e-4 | 1.341e-4 |
| baseline r20 | 9.063e-5 | 7.802e-5 |
| baseline r40 | 8.900e-5 | 2.474e-5 |
| adaptive CCP | **1.234e-5 (7.2x)** | **3.838e-6 (6.4x)** |

Matched-budget and matched-CPU sweeps: CCP leads at EVERY sampled
point (B in {2k, 5k, 10k, 20k}: 6-56x; T in {100, 300, 700} s:
6-33x, decision overhead included).  3v5: CCP at B=10k already beats
every baseline's B=20k value.  The user's overnight expectation holds
on both GN axes with no caveats; fairness audit found nothing wrong.

Front / test side (the honest nuance, mechanism-consistent, NOT a
bug): central-HV train is a four-way tie (0.4802-0.4804); test-side
metrics are a statistical tie (HV test 0.4757-0.4785; best mean test
CE differs ~10 % favouring fine grids on 3v5; best (errA+errB) 1.58 %
(grids) vs 1.77 % (CCP) on 3v5, 1.76 % all round on 7v9 — a couple of
test images).  Grids win IGD-central (uniform lambda spread is a
born front-coverer at K=2, and their decisions are free).  KEY
figure-3 finding: on 3v5 the grids' train fronts run 5-10x deeper
than CCP's in the mid region, but ALL test fronts collapse onto ONE
band that coincides with CCP's train front — the extra central train
depth is pure overfitting and does not transfer; on 7v9 even the
train fronts coincide.

Figure fixes made after first render: (a) fronts figure now drops
off-window points BEFORE plotting (divergence arms were driving the
log-axis autoscale to 1e-14); (b) figure 5 redesigned from raw chain
snapshots (lambda-driven oscillation, unreadable) to PREFIX-BEST mean
per-class test CE — the correct analogue of the paper's "test error
vs effective passes".

Aug-13 evening addition (user request): ``front_{train,test}_replam
.png`` — the split front figures plus "best point per grid lambda"
circles for the best baseline (own points only, selected by train
lambda^T F; ties from underflowed favoured-class CE broken by the
complementary weighted value).  Two findings: (a) TRAIN: the 40/41
in-window representatives lie ON the front but BUNCH in the
mid-band — with chain warm start they are snapshots of one
trajectory 5 segments apart, not 41 independent optima; the full
front's spread comes from the whole history, so an endpoint-only
front would be both sparser and clustered.  (b) TEST: the
train-selected representatives sit clearly ABOVE the common test
band — picking each lambda's train-best picks its most overfitted
late-cycle snapshot; the test front is populated largely by
mid-training points.  Directly answers "why not build the front from
one converged point per lambda".

Companion variant ``front_{train,test}_replam_ccp.png`` (same
treatment applied to the adaptive leg): with 644 distinct lambdas in
653 decisions, "per-lambda best" degenerates into "best of each
5-segment visit" — grouping by lambda IS grouping by time.  TRAIN:
the 632/644 in-window representatives fill the whole region ABOVE
the front (the cloud is simply the trajectory sampled once per
visit; a CCP lambda is chosen where the chain is currently WORST, so
its own-best snapshot is typically not front-quality — the
improvement it triggers is credited to later lambdas).  TEST: the
cloud drapes over the common band, mid-era points on it, late-era
lifting off.  Lesson: the per-lambda-representative summary is
meaningful only for few, persistent, revisited lambdas (grids); for
an adaptive policy lambda is a timestamp and the whole-trajectory
front is the only honest summary.

Report: ``~/Desktop/Experiment4_report_K2_MNIST_pair.docx`` (simple
Chinese; Smoke A + formal experiment, figures embedded with
per-figure analysis, parameter tables, py-file list).

Aug-13 revisions (user-requested, on the USER-EDITED copy
``Experiment4_report_K2_MNIST_pair 2.docx``; user edits preserved,
pre-edit backup ``Experiment4_report_user_edit_backup_aug13.docx`` in
the campaign home): (1) 1.3 now states the scoring formulas (A 方向 /
B 方向 / S_int, ln-2 normalisation); (2) part 2 restructured into
协议概述 / 参与对比的算法 (legs table) / 实验参数 (参数-作用-数值-备注
table) / 结果图与分析 / 结论; (3) the fronts figure was split into
front_train.png + front_test.png showing adaptive CCP vs the SINGLE
best baseline (lowest final exact audit — r40 on both pairs), replacing
fronts_train_test.png (plot script updated; report figures renumbered
图 1-12).  Two count fixes consequential to the user's deletions:
"改了五个地方"→四个, "三个结论"→"结论".

## Still pending after Smoke A

B (budget), per-leg s via the coverage inequality, eval_every /
snapshot granularity (scale with B), whether both top-2 pairs get the
full campaign or the runner starts with #1, then the campaign runner
itself (theta snapshots + test evaluation + figures) and Smoke B
(wiring: budget conservation, c == formula, theta round-trip
re-evaluation, exact-GN meter, figure rendering, resume/skip).
