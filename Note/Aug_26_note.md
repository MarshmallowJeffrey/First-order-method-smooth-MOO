# Aug 26 note — K = 3 MNIST triple campaign: Smoke A, meter change, full run

Session of Aug 25-26, 2026 (design Q&A Aug 25 evening, code + runs
Aug 26).  The K = 2 pair campaign (Aug 13) carried to a digit TRIPLE:
adaptive(CCP) vs simplex-grid baselines under the pure fixed-budget
protocol, Pareto front lifted to 3-D.  ALL new files; nothing
overwritten.

## Experiment in one line

K = 3 MNIST digit-triple MOO (per-class mean CE, 3-logit
patch-softplus net, d = 8,195, s_k ≡ 1, no regularisation): pure
budget B = 40,000, legs = simplex grids r ∈ {10, 20, 30} + adaptive
CCP, θ snapshots + official-test evaluation, audit_v2 two-instrument
meter, 3-D train/test fronts.

## Design decisions (user-approved, Aug 25-26 Q&A)

* Everything not listed here is the Aug-13 K = 2 freeze verbatim
  (seeds init 8 / sampler 41 / probe 7 / CCP 0, MSVRG segment, step
  0.1/(λᵀL·L_scale), momentum 0.5, batch 1024 stratified
  (342/341/341), chain warm start, joint = K units / row pair = 2K/n,
  balanced-max per_class, test = ALL t10k rows of the digits).
* Smoke A (triple selection): 5 candidate triples built from the
  measured K = 2 pair scores; 10 fixed-λ chains × 15 segments each —
  3 vertices (diagnostic + c(h=1)), 3 edge midpoints (K=3-specific:
  c(h=2) + single-class divergence diagnostic), 3 favour points
  (0.6,0.2,0.2)-perm (the SCORING chains; 3:1 favoured:starved
  exactly as K=2's (0.75,0.25)), 1 centroid (shape anchor).
  S_int = Σ_k [mean_{j≠k} F_k(favour-j end) − F_k(favour-k end)] / ln 3.
  Shape check per axis: F_k(favour-k) ≤ F_k(centroid) ≤ F_k(favour-j).
* Baseline r-set: user revised {5,10,20} → **{10, 20, 30}** (r=5 too
  coarse; r=30 = 496 nodes ≈ 1.9 cycles at B=40,000 deliberately
  probes the coverage-breakdown end).  Node counts 66 / 231 / 496;
  at r=10 45% of nodes have a zero component (divergence arms).
* TOP-1 only (user call): the campaign runs the best triple only.
* B = 40,000, s = 5, eval_every = 500 (user sign-off after measured-c
  proposal; r=20 turns 4.1 cycles, adaptive ≈ 900 decisions).

## METER CHANGE (the one convention change vs K = 2; user-approved)

The K = 2 exact 1-D meter has NO K = 3 analogue: the audit quantity
is the coverage maximin GN*(stack) = max_{λ∈Δ₃} min_i λᵀM_iλ — a
nonconvex maximin with no affordable certified evaluation in 2-D λ.
(An earlier in-chat claim that a support-enumeration closed-form QP
could serve as the exact meter was WRONG — that QP solves the
single-point min_λ, a different quantity; corrected before the run.)
Adopted meter = the K6/K10 audit_v2 convention, in-runner, off-axis:
audit(stack) = max(IPOPT strict-64 multistart, heavy CCP N0=8192/r=20
fresh solver) at EVERY checkpoint prefix of EVERY leg, previous
argmax λ threaded into the IPOPT start set; history NOT forced
monotone (violations counted, never clipped).  NEW final-stack
cross-check: dense simplex grid res 500 (125,751 nodes; exact chunked
6-monomial BLAS evaluation, `_grid_maxmin`) — an independent lower
bound; if it beats the instruments the miss is recorded and the grid
value becomes final_audit.

## Interpreter finding (root-caused Aug 26)

The project's real environment is the venv at
`/Users/shirch/vscode101/.venv/` (arm64 native, Python 3.13.7, torch
2.9.0, numpy 2.3.2, cyipopt 1.7.0 WORKING).  A bare `python` in the
work shell resolves to miniconda base (x86_64/Rosetta, py3.9) whose
pip cyipopt is import-broken (flat-namespace `_AddIpoptIntOption`
missing; no x86_64 libipopt anywhere).  Consequence: Smoke A and the
first Smoke B ran on miniconda with the audit's NLP instrument
silently on the SLSQP fallback.  Fixed by running everything with the
venv interpreter explicitly; summaries record `audit_nlp_backend`
("ipopt" everywhere in the formal campaign).  Re-runs of Smoke A and
Smoke B on the venv reproduced every number BIT-IDENTICALLY
(S_int table, c table, smoke audits 6.724273e-01 / 7.798533e-02) —
the deterministic protocol is arch/interpreter-stable on this
problem family (contrast: session-12 MLP torch runs were not
bit-reproducible across reruns on the K6 planted family).

## New files (Aug 26)

* `objective/objectives_mnist_triple.py` — triple loader (balanced
  max; t10k full test), `TriplePatchMLP` (3-logit, d = 8,195),
  `make_triple_initial_point`, `TripleStochLamOracle` (verbatim
  PairStochLamOracle mirror), `evaluate_triple`, `make_mnist_triple`.
  The pair file's `ah16_faithful` ladder option deliberately NOT
  carried (no conflict ladder in the K=3 design).
* `experiment_plot/run_conflict_smoke_K3_mnist_triples_without_256_
  checkpoints.py` — Smoke A (checks: wiring ≤1e-10, per-chain
  c == K + epoch_len·2·rows_support·K/n, shape violations, L probes,
  safeguard counts, divergence diagnostics).
* `experiment_plot/run_pure_budget_K3_mnist_triple_without_256_
  checkpoints.py` — campaign runner (K2-replica executor + snapshots
  + test eval + the new audit stack + `--smoke` Smoke B; resume-skip;
  `audit_nlp_backend` recorded).
* `experiment_plot/plot_K3_mnist_triple_without_256_checkpoints.py`
  — figures (3-D fronts: two log10 views + three pairwise log-log
  projections; K-D nondominated filter `_nondominated_kd`; 3-D
  hypervolume `_hv_3d` by z-sweep over the 2-D staircase).

## Smoke A results (run twice: miniconda 362 s, venv 315 s — numbers identical)

All 5 triples green (wiring ~4e-15, shape violations 0, all 50
chains' c == formula, L ∈ [1.21, 1.58]).  Vertex/edge divergence
confirmed (ignored-class CE ~25 / ~15 after 15 segments) → ln-3
window rule for figures.  Vertex diagnostic scores all ~50
(divergence-dominated, as at K=2).

| rank | triple | S_int | parts (per axis / ln3) | c int / edge / vtx |
|---|---|---|---|---|
| 1 | **3-5-8** | **0.8317** | 0.331 / 0.302 / 0.280 | 9.045 / 7.030 / 5.015 |
| 2 | 4-7-9 | 0.7855 | 0.239 / 0.272 / 0.352 | 9.310 / 7.207 / 5.103 |
| 3 | 2-3-8 | 0.6294 | 0.207 / 0.234 / 0.250 | 9.300 / 7.200 / 5.100 |
| 4 | 3-8-9 | 0.5837 | 0.248 / 0.239 / 0.155 | 9.300 / 7.200 / 5.100 |
| 5 | 5-6-8 | 0.5268 | 0.225 / 0.153 / 0.201 | 9.045 / 7.030 / 5.015 |

top-1 = {3,5,8} (balanced conflict across all three axes; n=16,263,
epoch_len=16).  Measured per-segment wall (venv): interior ~0.40 s.
c matches the a-priori formula exactly at all three support sizes.

## Smoke B (venv, 23 s, SMOKE OK)

Budget conservation (790.5/800 both legs), every segment cost in the
7-support formula set, θ round-trip 1e-9, audit histories finite,
final grid ≤ instruments, test eval sane (shape (m,3)), λ simplex
checks, resume-skip, 5 figures + front_metrics render.  Adaptive
already 8.6× better than the r=4 grid at B=800 (7.80e-2 vs 6.72e-1).

## Formal campaign (Aug 26 overnight, user go; ALL DONE in 8,283 s ≈ 2.30 h)

Home: `output/CCP/K3_mnist_triple_without_256_checkpoints/
triple_3v5v8_B40000/`.  caffeinate-wrapped, venv interpreter.

Fairness audit (all 4 legs): x0 bit-identical (md5 3a0632acea…),
spent 39,991.9-39,999.2 of 40,000, decision_seconds 0.0 on grids vs
202.6 s on CCP (909 decisions, 0.222 s/decision, ~10.5% of leg wall —
between K2's 0.135 s and K10's 0.45 s), safeguard_retries ≤ 1
everywhere, `audit_nlp_backend = ipopt` everywhere.  Post-hoc
overhead per leg: audits ~210-230 s, grid check ~0.5 s, test eval
~33 s.  Segments: r10 4,981 / r20 4,729 / r30 4,604 / CCP 4,542
(grids buy MORE, cheaper segments via vertex/edge nodes — the K=2
support-effect direction, now graded by r).  Audit-history monotone
violations: r10/r30 zero; r20 3 ups of ≤ +5.1e-10 (pure noise); CCP
3 ups of ≤ +6.4e-5 (mid-run instrument wiggle at the small-value end;
final value grid-confirmed).

### RESULTS — final audit (GN*) at B = 40,000

| leg | final audit | grid cross-check |
|---|---|---|
| baseline r10 | 9.105e-4 | 8.824e-4 ✓ |
| baseline r20 | 1.308e-3 | 1.238e-3 ✓ |
| baseline r30 | 3.910e-3 | 3.749e-3 ✓ |
| **adaptive CCP** | **7.390e-5 (12.3×)** | 7.390e-5 (identical — saturated) |

Headline: **CCP beats the best grid 12.3× on the final audit** (vs
6.4-7.2× at K=2) — the grid disadvantage GROWS with K.  The baseline
ladder is monotone in r (finer = worse: 9.1e-4 < 1.31e-3 < 3.91e-3):
node explosion dilutes per-node depth; r30 (1.9 cycles) is the
designed breakdown exhibit.

Matched-budget sweep (audit at nearest checkpoint; best grid always
r10): B=5k **7.8×**, 10k **6.5×**, 20k **12.6×**, 40k **13.6×**
(sweep-grid convention: 6.703e-5 at the pre-final checkpoint).
CCP at B=10k (5.60e-3) beats r10 at B=20k (1.27e-2) — half-budget
crossover.  Matched-CPU sweep (decision overhead ON-axis): T=300s
4.9×, 700s 8.1×, 1500s 8.9×.

### Front / test side

* HV central train (ref (ln3)³): four-way tie 1.3244-1.3258 — the
  K=2 train-HV tie replicates.
* **HV central TEST: CCP WINS — 1.2942 vs r10 1.2723 / r20 1.2709 /
  r30 1.2377.**  NEW vs K=2 (where test was a statistical tie): at
  K=3 the grids' train coverage transfers WORSE, and the coarsest
  grid transfers best among grids (r30 clearly worst).
* IGD-central train: r20 0.0017 < r10 0.0033 < CCP 0.0057 < r30
  0.0164 — "grids are born front-coverers" survives at K=3 ONLY for
  r ≤ 20; at r=30 the grid can no longer even cover (worse than the
  adaptive).  Grid coverage now has a breakdown point INSIDE the
  grid family.
* Fronts: train front sizes 268-919 points, test fronts 854-1,210
  central points; test clouds drape over a common band with CCP
  reaching deeper low-CE corners in the projections; grid legs show
  the characteristic per-λ "spoke" arms.

## Figure/metric conventions (K=3 specifics)

Front figures: adaptive vs best baseline (lowest final audit = r10),
window ≤ ln 3; off-window points dropped before plotting (K=2
lesson).  RESTYLE (post-campaign user request, modelled on the
breakable-bottles reference figure the user supplied): each front
figure is ONE row of three 3-D views at fixed angles
(elev,azim) = (22,−60) / (18,−140) / (34,115), LINEAR axes autoscaled
to the windowed points, dots only, per-panel angle caption, legend
with non-dom counts; the pairwise log-log projections (the
quantitative read) moved to companion files
front_{train,test,err_test}_proj.png.  The first render (two log10
3-D views + projections in one 2×3 grid) was judged too cluttered by
the user and replaced the same day; data untouched, pure re-render.
front_err_test shows all legs.  test_ce_vs_budget = prefix-best mean
per-class test CE (K=2 fig-5 convention).  NEW supplementary figure
(user request, Aug 26 evening): ``hv_slices_test.png`` — makes the
test-HV gap visible: 4 horizontal slices at data-driven F_3 heights
showing the rasterised dominated regions (green = adaptive-only,
orange = best-baseline-only, gray = both; exact per-slice areas via
the 2-D staircase) + a panel of the area difference over z and its
running integral, which reproduces the 3-D HV gap (slice integral
0.02179 vs exact 0.02191, z-grid discretisation only; recorded in
front_metrics.json under _hv_gap_test_slice_check).  Finding the
figure surfaced: most of the gap comes from the extreme low-F_3
slice (area diff +0.41 at F_8-CE <= 0.008 — the deep digit-8 corner
is essentially adaptive-only territory), plus a steady ~+0.02 area
lead at every height above.

SECOND SUPPLEMENTARY FIGURE (user request, Aug 26 late, MODPO-style):
``front_test_surface.png`` — the test frontier rendered as a shaded
sheet like MODPO's appendix-B 3-objective figure: faint dots = full
central cloud; sheet = plot_trisurf over the LOWER ENVELOPE of the
nondominated set on a log (F1,F2) grid (min-F3 per cell — kills the
multi-valued vertical F3-arm spikes), Delaunay triangles with any 3-D
edge above a threshold dropped (no fabricated bridges across the
empty inter-arm region); baseline sheet first, adaptive on top,
alpha 0.55.  Final settings after a "simplify" iteration (user asked
for fewer points): the full-cloud faint scatter REMOVED, grid
coarsened 34x34 -> 18x18 (envelope ~97/96 pts per method), edge_max
0.30 -> 0.45, envelope nodes drawn as solid dots on the sheet.
Rendering aid only — the scatter figures and metrics remain the
ground truth.

THIRD SUPPLEMENTARY FIGURE (user request, Aug 26 late — "one glance
that CCP wins"): ``dominance_map_test.png`` — the ADVANTAGE MAP.
For each pair-of-objectives budget (u, v) on a 240x240 log grid, the
colour answers "who reaches the lower third objective": z_env(u,v) =
prefix-min of F_k over front points with the other two coords <=
(u,v); colour = z_env(baseline) - z_env(CCP), green = CCP deeper,
orange = baseline deeper, solid +/-vmax where only one method reaches
the cell at all.  Three panels = the three orientations.  RESULT:
green 59/59/79% vs orange 10/14/18% of reachable cells (overall 65%
vs 14%); the solid-green frontier band = CCP-only territory; the
orange pockets sit at the extreme single-axis corners (the grid's
vertex-node points, cf. the per-axis min analysis).  This is the
one-glance verdict figure; the sheet/scatter figures stay for
structure.

DRAW-ORDER FIX (Aug 26 evening, user caught it): the first renders
painted the adaptive series FIRST and the baseline SECOND; matplotlib
3-D does not depth-sort across scatter artists, so orange overpainted
green wherever the clouds overlap — the dense central bowl looked
all-orange, visually inverting who owns the origin corner.  Numbers
say the corner is green's: nearest-to-origin front point L2 0.0929
(CCP) vs 0.1024 (r10, beaten coordinate-wise); points with all three
test CEs <= 0.1: 297 (CCP) vs 208 (r10); inside the <=0.15 cube
267/389 orange front points are dominated by a green point vs 10/538
the other way.  Cross-domination on the full central test fronts:
61.2% of r10's points dominated by CCP vs 2.0% reverse.  Fix:
baselines plotted first, adaptive LAST (winner on top) in all 3-D
rows + alpha 0.85; figures re-rendered, data untouched.  front_metrics.json: 3-D
HV (z-sweep × 2-D staircase), IGD to union front raw + central.
No replam figures at K=3 (the K=2 per-λ-representative analysis was
a one-off user request; trivially portable later if asked).

## Report (produced same day, user request)

`~/Desktop/Experiment5_report_K3_MNIST_triple.docx` — Chinese, 11
pages, Experiment-4 report structure (Part 1 Smoke A with formulas +
ranking table + top-1 figure; Part 2 协议概述 / 计量说明 / 算法表 /
参数表 / 公平性审计 / 图 2-10 逐图分析 / 匹配预算与 CPU 表 / 前沿指标
汇总表 / 四条结论; 附录 代码与产物).  Builder:
`ledger-artifacts/exp5-report-builder/build_report.js` (docx-js,
npm-installed locally in that folder; PingFang SC).  Extra numbers
computed for it: best mean test err CCP 1.34% vs grids 1.61-1.64%;
best mean test CE 0.0530 vs 0.0588-0.0606; best err-sum 4.02% vs
4.84-4.91%.  v2 same day (user: "写的太复杂了"): full prose rewrite in
plain Chinese (short sentences, terms glossed, "图 N 怎么看" analysis
style); structure/tables/figures unchanged; rebuilt in place.

Paper section (user request, same day): `~/Desktop/section_4_2_mnist
.tex` — Overleaf-ready Section 4.2 (Multiclass classification on
MNIST) for the MAnalytics draft v2, written from the K=3 campaign
with a closing K=2-companion paragraph; matches the paper's Section-4
conventions (eps_sm-stat, two budget axes, Algorithm 1 vs uniform
discretization); needs figures/{gn_vs_grads,gn_vs_cpu,front_test,
test_ce_vs_budget}.png uploaded + 3 flagged placeholders (reddi bib
key, alg:bundle, alg:uniform) swapped to the paper's own labels.
Test-compiled clean (pdflatex, booktabs).

## Still open / next steps

* Deck for Experiment 5 — not yet requested (report docx done).
* SURF-as-baseline thread (Aug 25 analysis): SURF slots into K=2
  only (1-D arc-length machinery); K≥3 non-extension is a paper
  motivation paragraph.  CV / Gap-Ratio uniformity metrics exist in
  the K2 planted module (`_spacing_metrics`) if a K=2 SURF leg is
  ever run.
* Possible follow-ups: second triple {4,7,9} via `--triple 4 7 9`
  (resume-skip makes this a clean 2.3 h add); tolerance/N0 ablation
  of the audit instruments; miniconda cyipopt left broken (venv is
  the project interpreter — fix only if the user wants miniconda
  usable too).

---

# PART 2 (appended same day, evening session) — Experiment 6: the
# ridge-penalty campaign (mu = 1e-4)

Same campaign, ONE knob changed: every objective gains the ridge
penalty (mu/2)*||theta||^2 (all d = 8,195 parameters, biases
included), mu = 1e-4.  Everything else is the Part-1 freeze verbatim;
Experiment 5 stays untouched as the mu = 0 reference arm.

## Design (user-approved Q&A, this session)

* Form: same mu for all three objectives.  Since sum(lam) = 1, this
  equals adding the penalty once to any scalarization.  Penalty
  gradient mu*theta is deterministic, touches no data rows -> ifo
  accounting and the segment-cost formula UNCHANGED (smoke-verified).
  In the MSVRG estimator the anchor penalty terms cancel exactly ->
  estimator + mu*y.  L_k -> L_k + mu analytically (exact Hessian
  shift mu*I; no re-probing).
* mu = 1e-4 fixed by the user's direct call (no sweep; "拍脑袋"
  ruling).  Basis: ||theta||^2(init) ~ 326 -> init penalty 0.016 =
  1.5% of the ln 3 window; ~1.6/n (n = 16,263); taming pull
  mu*||theta|| reaches 1e-3 scale at norms of tens.  Contingency
  (raise to 3e-4 if vertex chains fail to plateau) NOT triggered.
* NEW-FILES ruling (user, mid-session: "不要改代码，写新的"): the
  first implementation edited the two base files with a guarded
  mu kwarg; fully REVERTED on the ruling (line counts 313/566
  restored, zero mu/ridge grep hits, compile OK, plus behavioral
  proof below).  Final implementation:
  - `objective/objectives_mnist_triple_ridge.py` — wraps the base
    factory (make_mnist_triple_ridge(triple, mu, ...)); subclass
    RidgeTripleStochLamOracle adds mu*theta in grad_pair (anchor
    theta recorded by set_anchor); nothing copied.
  - `experiment_plot/run_pure_budget_K3_mnist_triple_ridge_without_
    256_checkpoints.py` — executor/main replica of the base runner;
    audit/grid/test/cost helpers IMPORTED from it; `--mu` default
    1e-4; EVERY output home suffixed `_mu<mu>` (mu = 0 -> `_mu0`) so
    base-campaign records can never be touched.
* Reporting axes: train fvals / Gram stacks / GN* audits in
  PENALISED coordinates (the problem being solved); official-test
  evaluation stays RAW CE; the penalty adds the same (mu/2)||theta||^2
  to all three coordinates of a point, so dominance differs between
  the coordinate systems; raw train values recoverable from
  thetas.npz at plot time.  DISCIPLINE: cross-mu GN* values are
  different problems — never compared directly; only within-mu
  ratios and the fixed-meaning test axis cross campaigns.

## Pre-launch gates (all green)

1. Algebra gate (scratch): f/J/grad_pair shifts exactly mu*theta
   (max err 2.2e-16 = ulp), ifo identical, L_ridge == L_base + mu
   exact, mu=0 factory bit-identical to base.
2. Replica-fidelity gate: ridge runner `--smoke --mu 0` into scratch
   reproduced the stored Part-1 Smoke B BIT-IDENTICALLY — all 8
   arrays per leg (thetas included; seg_lams equal under
   equal_nan), all summary numerics, all CCP algorithmic stats;
   only wall-clock fields differ.  6.724273e-01 / 7.798533e-02
   reproduced exactly.  (Also serves as behavioral proof the base
   objective file's revert is exact.)
3. mu=1e-4 Smoke B: SMOKE OK -> `SMOKE/triple_3v5v8_B800_mu0.0001/`
   (final smoke audits 6.780346e-01 / 7.839086e-02; seg-cost formula
   check passed = penalty provably outside the budget meter).

## TWO INCIDENTS (both caught before data damage; full disclosure)

* INCIDENT 1 — interpreter: `run.sh` resolves `../../.venv` to the
  REPO-INNER venv `First-order-method-smooth-MOO/.venv/` (python
  3.11.5, homebrew base, created Jun 25; its cyipopt WORKS — audits
  ran ipopt, so NOT the Part-1 SLSQP failure mode).  It is NOT the
  project venv (3.13.7).  First campaign launch went through run.sh
  -> killed ~10 min in, zero disk residue; relaunched with the
  EXPLICIT venv interpreter.  Forensics: same smoke on 3.11 vs 3.13
  agrees to all 7 printed digits but differs at bit level
  (|dtheta| <= 2e-15, audits <= 3e-14) — harmless at smoke scale,
  but the mu comparison must be same-stack as Exp 5 (3.13).
  The 3.11-produced smoke was regenerated on 3.13 (--force; 3.11
  copy preserved in session scratch).  **LEDGER CORRECTION: every
  run.sh invocation (including the ledger's sanity-gate protocol)
  actually runs on the inner 3.11 venv, not the project venv; the
  run.sh header's claim is wrong.  run.sh left untouched (user's
  file) — disposition is the user's call.**
* INCIDENT 2 — battery/clamshell sleep: laptop unplugged + lid
  closed at 19:15 (13% charge); ~22 forced sleep episodes until the
  21:38 AC wake (caffeinate -dims cannot block battery-critical or
  clamshell sleeps).  pmset log shows ZERO sleeps before 19:00:
  r10/r20 finished clean (their loops + audits ended 19:00:03).
  r30 was mid-loop -> its real-time clock (ck_cpu) poisoned ->
  killed; resume-skip preserved r10/r20; r30 + CCP re-run from
  scratch on AC, clean walls.  Consequence: campaign_manifest.json
  holds only the restart's two legs — per-leg summary.json files are
  the authoritative record.

## Formal campaign — `triple_3v5v8_B40000_mu0.0001/` (venv 3.13,
## clean walls, total leg seconds 8,182)

Fairness: x0 md5 3a0632acea = same bit-identical init as Exp 5;
spent 39,992-39,999.2; audit_nlp_backend ipopt everywhere;
safeguards <= 1 (one L_scale doubling on r20); decisions 997/946/921
grids (IDENTICAL to Exp 5 — the penalty changed no grid scheduling;
segments 4,981/4,729/4,604 identical too) vs CCP 907 decisions
(Exp 5: 909), 4,533 segments (4,542), decision_seconds 179.4
(202.6), all distinct lambdas.

### RESULTS — final audit (GN*, penalised problem) at B = 40,000

| leg | final audit | grid cross-check | mono ups |
|---|---|---|---|
| baseline r10 | 1.134531e-2 | 1.116339e-2 ✓ | 3, all <= 8.3e-11 (noise) |
| baseline r20 | **6.584289e-3** (best grid) | 6.146558e-3 ✓ | 0 |
| baseline r30 | 9.229476e-3 | 9.048672e-3 ✓ | 1, +1.2e-11 (noise) |
| **adaptive CCP** | **3.438077e-4 (19.1x)** | GRID VALUE (instruments 3.308045e-4) | 1, +1.5e-5 (small-value wiggle) |

* CCP final stack = the campaign's first REAL instrument miss: dense
  grid beat strict-64+heavy-CCP by +3.9%; recorded, final takes the
  grid value per convention.  (Option on the books: raise audit
  multistarts / N0.)
* **Headline: 19.1x vs Exp 5's 12.3x.  The REGISTERED PREDICTION
  (grids benefit, ratio shrinks) is FALSIFIED — ratio grew.**
* Grid ladder now NON-monotone: r20 best, r10 WORST (Exp 5 was
  monotone r10 < r20 < r30).  All three grid lam* sit essentially at
  the digit-5 vertex (weights 0.9597 / 0.9702 / 0.9899); CCP's own
  lam* is on the 3-8 EDGE (0.31, 0, 0.69) — it has pressed the
  vertex region down and its residual bottleneck is elsewhere.
* Matched-budget sweep (nearest-checkpoint instrument convention):
  B=5k **8.8x**, 10k **8.5x**, 20k **21.4x**, 40k **19.9x**
  (Exp 5: 7.8/6.5/12.6/13.6).  QUARTER-budget crossover: CCP@10k
  (5.219e-3) beats every grid at full 40k (best 6.584e-3); Exp 5 had
  a half-budget crossover.  Matched-CPU (decisions on-axis): T=300s
  6.8x, 700s 14.4x, 1500s 19.5x (Exp 5: 4.9/8.1/8.9).

### Front / test side

* HV central TEST: **CCP 1.2989** > r20 1.2843 > r10 1.2702 > r30
  1.2601 (slice-integral check +0.0157).  Exp 5: 1.2942 vs
  1.2723/1.2709/1.2377 — CCP keeps the win, absolute values up.
* HV central train: 1.2496-1.2530 four-way tie in PENALISED
  coordinates (not comparable to Exp 5's 1.32 raw values).
* IGD central train: r20 0.0061 < r10 0.0284 < CCP 0.0319 < r30
  0.0405 — grids-as-front-coverers survives for r <= 20, r30 broken,
  same shape as Exp 5.
* Best mean test CE: CCP 0.0483 vs grids 0.0560-0.0592 (Exp 5:
  0.0530 vs 0.0588-0.0606).  Best mean test err: CCP 1.27% vs grids
  1.58-1.64% (Exp 5: 1.34% vs 1.61-1.64%).  Mild-ridge
  generalization dividend: EVERY leg improves vs mu = 0, CCP most.
* Figures: the Part-1 plotter reused verbatim on the mu home — all
  12 PNGs incl. the three supplementary (hv_slices / surface /
  dominance map).  Dominance-map percentages NOT recomputed for this
  run (figure is qualitative in the report).

### Mechanism observations (measurements + hypothesis, not proof)

* Divergence subsidy hypothesis: at mu = 0, CE gradients vanish
  along the divergent vertex arms, giving the grids free near-vertex
  coverage; the ridge restores a real finite minimizer there that
  budget-limited chains cannot reach, so the near-vertex gap opens.
  r10 took the most subsidy (deepest nodes) and loses the most.
* Taming revision (corrects an in-chat claim from the design phase):
  at this horizon mu = 1e-4 does NOT visibly cap the ignored-class
  CE peaks (r10 maxF ~123/54/5.5 in BOTH campaigns) and only mildly
  compresses norms (r10 ||theta|| max 28.9 -> 20.5).  The real
  signature is the end-state direction: mu = 0 norm still RISING at
  budget end (final = max = 28.9) vs ridge TURNED (final 20.0 <
  max 20.5).  The penalty acts through certificate geometry, not
  visible trajectory rerouting.

## Report (produced same session, user request)

`~/Desktop/Experiment6_report_K3_MNIST_triple_ridge.docx` — Chinese,
11 pages, Experiment-5-v2 structure (plain prose, "图 N 怎么看"),
every table with an Exp-5 comparison column; incidents disclosed in
2.6; mechanism section 2.10; conclusions x5.  Builder:
`ledger-artifacts/exp6-report-builder/build_report.js` (docx-js;
node_modules symlinked from exp5-report-builder; PingFang SC).
QA-rendered and page-checked.

## Still open (Part 2 additions)

* run.sh interpreter disposition (points at the inner 3.11 venv) —
  user's call: repoint / delete inner venv / leave.
* Audit multistart/N0 raise (the 3.9% CCP-stack miss).
* Part-1 open items unchanged (deck; {4,7,9}; SURF-as-baseline).

---

# PART 3 (Aug 26 late -> Aug 27 early) — second triple {4,7,9} under
# the ridge (Experiment 7)

User go: same ridge campaign, only the triple changed ({3,5,8} ->
{4,7,9}); one command (`--triple 4 7 9` on the ridge runner, mu=1e-4
default).  NO mu = 0 arm exists for this triple — comparisons are
within-campaign and cross-triple vs Part 2.

Pre-launch: disk was at 2.0 GiB free (campaign needs ~1.2 GB) — the
user personally authorized deleting five named cache dirs
(~/Library/Caches/{Google, pip, com.microsoft.VSCode.ShipIt,
com.openai.atlas, com.anthropic.claudefordesktop.ShipIt}) -> 12 GiB
free; AC power verified before launch (Part-2 lesson).

Instance: n = 17,526 (per_class 5,842 = balanced max of {4,7,9}),
epoch_len 18, L = [1.666, 1.536, 1.569]; x0 bit-identical to BOTH
{3,5,8} campaigns (same init seed + same d) — a built-in cross-triple
control.  Clean single pass: 8,917 s wall (leg seconds 8,783), venv
3.13 explicit, ipopt everywhere, safeguards 0 on ALL four legs,
spent 39,991.8-39,999.8, decisions 968/915/896/878, segments
4,840/4,572/4,476/4,387, CCP decision_seconds 170.9.

### RESULTS — final audit (GN*, penalised) at B = 40,000

| leg | final audit | grid cross-check | mono ups |
|---|---|---|---|
| baseline r10 | 1.027422e-2 | 1.008726e-2 ✓ | 0 |
| baseline r20 | **7.876759e-3** (best grid) | 7.640729e-3 ✓ | 1 (+2.0e-11) |
| baseline r30 | 9.641793e-3 | 9.060776e-3 ✓ | 1 (+1.4e-9) |
| **adaptive CCP** | **3.544621e-4 (22.2x)** | 3.508004e-4 ✓ (NO miss) | 7 (largest +2.3e-4, see caveat) |

* **Part-2 structure REPLICATES**: ladder non-monotone with the SAME
  ordering r20 < r30 < r10; all three grid lam* pinned at ONE vertex
  — digit 9 here (weights 0.9208/0.9604/0.9757) vs digit 5 for
  {3,5,8}; CCP's own lam* INTERIOR (0.2513, 0.3540, 0.3947) (was the
  3-8 edge).  CCP absolute value nearly identical across triples
  (3.54e-4 vs 3.44e-4); ratio GREW again (22.2x vs 19.1x).
* CAVEAT (worst instrument wiggle so far): CCP audit history has 7
  mono ups, largest +2.3e-4 ≈ 65% of the final value — small-value
  mid-run under-search; the FINAL value is triple-confirmed
  (instruments ≥ dense grid).  Mid-run small-value curve segments
  carry this noise.  Strengthens the standing multistart/N0-raise
  option.
* Matched-budget (nearest-ck instrument convention): B=5k 6.4x, 10k
  6.1x, 20k 16.2x, 40k 22.2x.  Matched-CPU: T=300 8.5x, 700 7.9x,
  1500 16.1x.  QUARTER-budget crossover replicates: CCP@10k
  (5.085e-3) beats every grid at full 40k (best 7.877e-3).

### Front / test side (NOT a sweep for CCP this time — honest split)

* HV central TEST: **CCP 1.3085** > r10 1.2962 > r20 1.2915 > r30
  1.2828 (slice check +0.0175 vs r20).  Train HV four-way near-tie
  1.2537-1.2594 (penalised coords).
* Best mean test CE: **CCP 0.0471** vs r10 0.0484 / r20 0.0519 / r30
  0.0583.  Best mean test err: **r10 WINS — 1.43% vs CCP 1.52%**
  (r20 1.55%, r30 1.92%) — first grid test-side win in the ridge
  campaigns.
* IGD central train: r10 0.0062 champion; r20 0.1284 and r30 0.1228
  BOTH worse than CCP 0.0331 — grid front-coverage breakdown starts
  at r20 on this triple ({3,5,8}: r20 was the IGD champion).
  Front counts: CCP 4,388 pts / 460 train-front / 1,941 test-front;
  r10 4,841/371/1,792; r20 4,573/879/2,264; r30 4,477/1,397/1,822.
* n_test = [982, 1028, 1009] (all official t10k rows of 4/7/9).

Figures: the shared plotter on the home (all 12 PNGs +
front_metrics.json).  Report: `~/Desktop/Experiment7_report_K3_
MNIST_479_ridge.docx` (Exp-6 structure; comparison columns =
{3,5,8}-ridge); builder `ledger-artifacts/exp7-report-builder/`.
Open: a mu = 0 arm for {4,7,9} (one command on the base runner,
~2.4 h) if the cross-mu story is ever wanted on this triple.

CORRECTION (Aug 27, during the vertex-pinning mechanism analysis): the
Experiment-7 report's claim "both pinned vertices are the max-conflict-
share class" was WRONG for {3,5,8}: the pinned digit 5's Smoke-A share
is 0.302 (SECOND; digit 3 leads with 0.331); only {4,7,9}'s pin (digit
9, 0.352) matches.  The report docx was corrected in place same day
(观察一 + 结论 2, marked 勘误).  What the Gram stacks actually show
(ridge {3,5,8}, r10/r20): own-vertex stationarity is excellent at ALL
three vertex nodes (lam^T M lam ~ 3e-6..8e-6), but the cross-gradient
norms at the arm endpoints are wildly asymmetric — at the e_5 endpoint
(‖∇F_3‖, ‖∇F_8‖) ≈ (55, 2.5-3.3) vs (21, 20) at e_8 and (7.7-9.5,
3.9-5.8) at e_3 — so tiny-mixture lambdas near e_5 are the least
coverable (audit lam* epsilons 0.006/0.035 sit BELOW the node
spacing); the pin location is set by this arm-end gradient geometry,
no simpler rule established (n=2).
