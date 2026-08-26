# Aug 9 note (2) — K = 10 MNIST patch-softplus trial

Session of Aug 9, 2026 (same session as Aug_9_note.md, separate
thread).  User-directed trial: upgrade the objective family to real
data — MNIST subset + softplus + AH16-style patch first layer — and
compare adaptive(IPOPT) vs adaptive(CCP) under the pure fixed-budget
protocol at K = 10.  NO baseline legs (user decision: the K = 10 grid
baseline needs its own design — node counts explode — deferred).
NEW FILES ONLY; outputs under the existing campaign root.

## Design (user-approved in the Aug-9 Q&A)

* Data: MNIST train, first 1000 rows per class in dataset order
  (deterministic, no rng), pixels /255 -> n = 10,000, K = 10.
  Balanced; imbalance / label-flip variants deferred until the front's
  shape is seen.  Cached in Adaptive Bundle Algorithm/data/mnist/
  (torchvision S3 mirror, one 10 MB download).
* Architecture: patch-64 (5x5 blocks, top-left corners on the
  deterministic 8x8 grid over [0,23]^2) -> softplus -> dense 96 ->
  softplus -> 10 logits; d = 8874 (same scale as the planted family's
  11.9k — structure upgraded, scale held).  ``ah16_faithful`` flag
  drops the dense layer (d = 2314).  Rationale for dense layer 2:
  locality needs spatial structure (gone after layer 1 at depth 2),
  someone must mix global information before 10-way output, and the
  parameter win of patching layer 2 is negligible (6.1k block).
* Smoothness kept: linear patch/dense maps + softplus + CE; no ReLU /
  max-pool / BatchNorm / dropout / augmentation (all would break the
  L-smooth finite-sum objective definition).
* Objectives: per-class mean CE (same definition as the planted
  family); s_k ≡ 1; L_i by random parameter-pair probes (40), executor
  L_scale safeguard as backstop.  Seeds: init 8, sampler 41.
* Protocol: pure fixed budget, verbatim K2/K6 semantics.  batch 1024
  (n/b ≈ 10 steps per epoch — shape matched to the planted runs'
  13), s = 5, B = 55,000 (targets ~360 decisions), eval_every 1500.
* IPOPT targeting cap ts = 24 (K6 convention); the FULL structured set
  at K = 10 is 67 points — the cap-vs-coverage tension is itself a
  reported finding.  CCP config: production defaults (N0 = 2000,
  r = 10, exp sampler, pool on, adaptive schedule off).
* Quality meter: post-hoc audit_v2 (max of strict-64 and CCP N = 8192)
  via the existing script with --home; no exact meter exists at K=10.

## New files

* ``objectives_mnist_patch.py`` — loader (IDX parse + cache),
  ``PatchMLP``, ``make_patch_initial_point`` (He, small positive
  hidden biases, zero output bias), ``PatchStochLamOracle`` (verbatim
  mirror of StochLamOracle: stratified b_k ∝ n_k, persistent
  current/anchor nets, ifo += 2·rows), ``make_mnist_patch`` factory.
  Verified: gradient vs central difference rel err 1.1e-8; full-batch
  stochastic gradient == joint scalarized gradient to 1e-15; L_i in
  [1.05, 1.45]; d = 8874.
* ``run_ccp_compare_K10_mnist_without_256_checkpoints.py`` — executor
  (declared replica of the K2 pure-budget loop; the originals
  hard-wire the planted factory) + 2-leg campaign
  (adaptive_s5_ts24, adaptive_s5_ccp), manifest, --smoke.
  Smoke: both legs green in 8 s; budget conserved.

## Observation already visible at smoke scale

Equal budgets buy different segment counts across legs (26 vs 62 at
smoke): ``_support_batch`` drops zero-weight classes from the
minibatch, so near-vertex lambdas make segments cheaper in
grad-equivalents.  At K = 10 this support effect is large.  Both legs
are metered by the same rule — protocol-consistent — but the effect
belongs in the results discussion.

## Results (trial complete; total campaign 3541 s + audit 1057 s)

| | adaptive (IPOPT ts24) | adaptive (CCP) |
|---|---|---|
| leg wall | 2293.5 s | **1188.7 s** (1.9x) |
| decision time | 1454.9 s = **63% of wall** (4.0 s/decision) | **214.3 s** (0.45 s/decision, 6.8x) |
| decisions / segments | 361 / 1804 | 475 / 2372 (support effect) |
| final audit_v2 | 2.5257e-1 | **7.6209e-2 (3.3x lower)** |

* The K-scaling prediction held: IPOPT targeting grows to 63% of the
  leg at K = 10 (was 52% at K2, 74% at K6 with the planted family);
  CCP decisions stay ~0.5 s.  On the CPU axis CCP reaches IPOPT's
  final quality with roughly 4-5x less CPU and then descends a further
  3.3x within the same budget.
* audit_v2 again vindicated the two-instrument ruler: on the IPOPT
  leg's stacks the CCP instrument was tighter on 35/38 checkpoints
  (final: strict-64 said 1.81e-1, true >= 2.53e-1); on the CCP leg's
  own stacks 32/38.
* softplus is well-behaved: both legs L_scale = 1.0, zero safeguard
  retries (the tanh saturation issue from the planted family did not
  reappear).  L probes landed in [1.27, 1.62].
* Support effect at K = 10 is large: equal budgets bought 2372 (CCP)
  vs 1804 (IPOPT) segments because `_support_batch` drops zero-weight
  classes and CCP visits near-vertex lambdas more often.  Same metering
  rule for both legs; discussed, not corrected.
* Pool pressure grew: n_distinct_before_cap reached 35 vs cap 30
  (5 dropped/round) — the 4r ablation flag strengthens with K.
* Figures (in the home): K10_gn_vs_grads / K10_gn_vs_cpu /
  K10_gap_vs_decision / K10_per_class_losses .png; audit_v2.json per
  leg; grams.npz keeps Q stacks, per-class fvals, lam/seg histories.

Report (simple Chinese: rationale/changes, algorithms, parameter
table, all 4 figures with analysis, py files):
``~/Desktop/Experiment3_report_Objective_MNIST_K10.docx``.

Aug-10 revisions (user-requested, mirroring the experiment-1 set):
(1) gn-vs-CPU/grads figures now start both legs from one shared origin
(identical {x0} first stack).  (2) NEW central-reference-front
evaluation (slide method) in
plot_ccp_compare_K10_mnist_without_256_checkpoints.py
(central_front_metrics; K10_front_central_metrics.png +
K10_front_metrics_ccp_compare.json).  c = 1 works directly on the
MNIST loss scale: |R_central| = 9 — and ALL 9 reference points come
from the CCP front (its Central IGD = max-distance = 0); Central HV
0.921 (CCP) vs 0.367 (IPOPT); IPOPT's 11 central front points are all
dominated.  Opposite of the K6 planted case (there the central
metrics favoured IPOPT) — the front-view advantage is
problem-dependent; on real data at K = 10 CCP wins BOTH views.  The
user-edited report (~/Desktop/"Experiment3_report_Objective_MNIST_K10
2.docx") updated in place (2 media swapped, origin note appended,
figure-3 section inserted; user edits preserved; validate PASSED);
pre-edit backup in output/ccp_compare_without_256_checkpoints/
Experiment3_report_user_edit_backup_aug10.docx.

Aug-10 revision round 2 (user): in the exp-3 report the K10 central
metrics figure was replaced by a TABLE, and a second table with the
K6 comparison recomputed WITHOUT the baseline was inserted into the
comparison paragraph.  The no-baseline K6 variant (union/reference/c
re-derived from the two adaptive fronts alone; c = median 6.69,
|R_central| = 13) is computed by the exp-1 plot script and stored as
K6_front_metrics_adaptive_only.json: IPOPT IGD 0.328 / max-dist 1.746
/ HV 0.807 vs CCP 1.412 / 3.532 / 0.739 — same direction as the
three-method table.  The user's file had been re-saved by Word
(relationship ids renumbered, bookmarks added) — the second surgery
re-mapped ids and strips cloned bookmarks; backup:
Experiment3_report_user_edit_backup_aug10_v2.docx.

Follow-ups from the trial: full-size version (per_class 5000+,
baselines with a K-appropriate small-r grid) if the trial goes into
the paper; imbalance / label-flip variants for a sharper front;
pool_cap 4r; the changeCoeff bulk-rewrite optimisation now also
matters at K = 10 (m up to ~2400).
