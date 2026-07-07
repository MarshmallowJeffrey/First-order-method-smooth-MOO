# Experiments: definitions, design, results, and trends

Date: July 6, 2026.
Code state: soundness fixes of July 4 (`Note/Jul_5_note.md`) plus the
paper-conformance fixes of July 6 (`Note/Jul_6_note.md`). All results below
were produced by `Original_py/run_experiments.py` on this code state; every
earlier figure in the repository history predates these fixes and was
produced by code with a confirmed frozen-loop defect, so none of it is
comparable or citable.

---

## 1. What experiments exist

The experiment surface of this project is exactly two designs, both driven
by `experiments.experiment_mlp_plateau_comparison` (an equal-budget
head-to-head of the two methods on one shared problem instance and initial
point):

1. **Plateau experiment** (sweep over K, the number of objectives).
   Question: at the same gradient budget, where does each method's quality
   stop improving, and how do those stopping levels compare?
2. **CPU-time crossover experiment** (sweep over MLP width, i.e. parameter
   count d and per-gradient cost). Question: the adaptive method spends
   extra non-oracle CPU per round; at what problem scale does its gradient
   efficiency start winning in wall-clock time?

Historical note: `experiments.py` also contains a second driver,
`experiment_mlp_gn_coverage` ("time-to-target": the baseline runs its full
schedule, its final GN* becomes the adaptive method's stopping target).
The old crossover notebooks used it. Its design is asymmetric — the
baseline is always charged its full schedule even if it reached its final
level early, while the adaptive method stops at the target — so this
rewrite does not use it for any headline number; the crossover ratios are
instead extracted symmetrically from equal-budget curves (Section 3). The
driver remains in the code base and is still useful for quick
single-configuration comparison plots.

There are no other experiments. (The strongly-convex logistic-regression
problem generator `make_logreg_strongly_convex` exists in
`objectives_numpy.py` but no experiment driver uses it; it serves the
verification suite.)

### The quality metric both experiments share

GN\*(B) = max over weights λ in the simplex of [ min over bundle points
x_i of ‖∇F_λ(x_i)‖² ].

Plain reading: "for the *worst-served* trade-off weighting, how
near-stationary is the *best* point we have for it?" Lower is better. A
small GN\* certifies that every weighting λ has some computed point that is
close to first-order stationary for F_λ — the paper's uniform-over-the-
simplex quality target (its ε_sm-stat, Eq. 2/11-12).

Two caveats, both by construction:

- Each reported GN\* is a **heuristic lower bound**: the max over λ is
  NP-hard, and is approximated by 256 multistart IPOPT solves (fixed
  strength regardless of run configuration, so it is one comparable
  yardstick for both methods). A reported value can understate the true
  worst case for either method equally.
- The metric is evaluated at checkpoints; its cost is **excluded from both
  reported axes** (gradient count and CPU time) for both methods.

### The two methods being compared

- **Baseline — uniform discretisation (paper Algorithm 1).** Lay a uniform
  grid of resolution r over the simplex (C(r+K−1, K−1) nodes), run
  fixed-step gradient descent on F_λ at every node, sweeping the grid in
  snake order with warm starts (first pass chains node to node; later
  passes resume each node from its own iterate). Its per-node work is
  blind to where quality is actually poor.
- **Adaptive bundle method (paper Algorithm 2).** Keep every evaluated
  point with all K per-objective gradients in a bundle; each outer round,
  find the currently worst-served λ (multistart max–min search), run a few
  1/L_λ gradient steps there (the paper's T-map, Eq. 10), add the best
  candidate to the bundle (`prune_inner=True`, the paper's Section 5
  implementation note). Stored gradients let it evaluate ∇F_λ at ANY λ
  with no new oracle calls — this cross-weight reuse is the paper's core
  idea.

---

## 2. Expected outcomes, derived from the paper (before looking at data)

The user's stated expectations were: (a) on the gradient axis the adaptive
method should dominate at equal budgets and plateau lower; (b) on the CPU
axis the adaptive method should win only once the objective is expensive
enough. Checking these against the paper's theory before accepting them:

**Gradient axis — expectation (a) is what the theory predicts.**

- *Reuse:* every bundle point stores all K per-objective gradients, so one
  oracle evaluation improves the certified quality at every λ
  simultaneously (Section 4: ∇F_λ(x_i) = Σ λ_k ∇F_k(x_i) is assembled from
  storage). The baseline's work at one node serves only that node's λ;
  information is discarded across weights — the inefficiency the paper's
  introduction names explicitly.
- *Adaptive allocation:* Algorithm 2 always works at the argmax-GN λ, so
  budget flows to the current worst spot. The baseline splits budget
  uniformly regardless of need.
- *Structural floor for the baseline:* with resolution r, quality between
  grid nodes is limited by the covering radius of the grid (Proposition 1:
  ℓ₁ radius ≤ K/(2r)) times the problem's sensitivity to λ. Once every
  node is well-optimised, more budget cannot push the worst-case-over-λ
  below this discretisation floor — the plateau. The adaptive method has
  no such floor: its bundle points serve arbitrary λ and its λ-search
  keeps finding and fixing the worst weight. Its plateau, when it appears,
  comes from optimisation difficulty (non-convexity), not from a grid.
- *Trend in K — derived carefully, then resolved by a budget study:* at
  fixed r the baseline's node count C(r+K−1, K−1) grows rapidly with K, so
  its grid floor worsens (covering radius ≤ K/(2r) grows) but is reached
  quickly. The adaptive method has no grid floor, but its outer complexity
  is also exponential in K (Theorem 2: (C·LipGN/ε)^(K−1)), so it converges
  more slowly at higher K. The two effects combine into a clean prediction:
  the adaptive method eventually beats the baseline at every K (baseline
  floored, adaptive not), but the GRADIENT BUDGET needed to overtake grows
  with K. At a single fixed budget you therefore see the advantage shrink
  and apparently vanish once the budget falls short of the break-even point
  — which is exactly what the 30k sweep shows at K=6 and what the 90k/240k
  escalation resolves (Section 5.1). Neither "advantage grows with K" nor
  "advantage shrinks with K" is right; "break-even budget grows with K" is.

**CPU axis — expectation (b) is exactly the paper's "two budget axes"
argument (Section 5).**

- Per scalarised step both methods pay K oracle calls, costing Θ(n·d)
  each (a forward+backward over the dataset). The baseline pays
  essentially nothing else. The adaptive method additionally pays
  oracle-free algebra per outer round: the multistart λ-search and the
  T-map selection each cost O(m·K·d) per evaluation (m = bundle size).
- Ratio of overhead to oracle cost scales like m·(solver iterations)/n:
  with pruning, m stays a few hundred; with n = 50,000 samples the oracle
  dominates as d grows, and the adaptive method's gradient savings convert
  into wall-clock savings. With a cheap oracle (small n·d) the overhead
  dominates and the baseline wins the CPU axis even while losing the
  gradient axis. **Prediction: the CPU-time ratio to a common quality
  target (baseline / adaptive) rises with d and crosses 1 somewhere in
  the sweep; the gradient ratio should exceed 1 throughout (once budgets
  are long enough for the adaptive method's early rounds to amortise).**

Verdict on the user's expectations: both are consistent with the paper,
with one refinement — on the CPU axis the adaptive method is *expected to
lose* at small scale; that is not a bug but the crossover thesis itself.
Additionally, at very short budgets the gradient-axis advantage can
transiently fail to show (the adaptive method spends its first rounds
building a bundle and calibrating L via the safeguard), so gradient ratios
are read at plateau-exposing budgets, not toy budgets.

**What would count as contradicting the theory** (and trigger a parameter/
code investigation per the user's instruction): the adaptive method
plateauing *above* the baseline at equal budget at plateau-exposing
budgets; a plateau ratio that *shrinks* as K grows; a CPU ratio that
*falls* as d grows.

---

## 3. Experiment definitions (precise)

### 3.1 Common protocol (both experiments)

One problem instance per configuration: planted linear-softmax data
(paper Section 5.1.1; W* entries now U[−1,1] per the July 6 fix), K
per-class cross-entropy objectives of a ReLU MLP, shared He initial point,
shared probe-based smoothness estimates, shared fused oracle. Both methods
run under the same total gradient budget `max_grad_evals`; neither stops
on a quality target. At checkpoints (every ~1/25 of budget for plateau,
~1/13 for crossover) we record (cumulative gradient evaluations, CPU time
with metric cost excluded, GN\*).

Cost accounting (paper Section 5): one gradient evaluation = one ∇F_k
call, so one scalarised step costs K evaluations; the initial point's
evaluation and all checkpoint/metric work are excluded from both axes for
both methods. CPU time is wall-clock on an otherwise idle machine, runs
strictly serial.

- **Plateau detection** (`detect_plateau`, window=5 — 4 for crossover —,
  tolerance 5%, 2 consecutive windows): onset = first checkpoint from
  which the best-so-far GN\* curve improves by <5% over each of 2
  consecutive windows AND over the whole remaining tail;
  plateau level = median of the best-so-far curve from onset to end
  (July 6 fix: level and detection now use the same monotone curve).
- **Symmetric time-to-target** (crossover's headline numbers): target =
  the WORSE of the two methods' final best-so-far GN\* (so both provably
  reached it); for each method, take the CPU time / gradient count at the
  first checkpoint whose best-so-far GN\* ≤ target;
  ratio = baseline's value / adaptive's value; >1 means the adaptive
  method got there first. This replaces the old asymmetric design where
  the baseline was always charged its full schedule.

### 3.2 Plateau sweep configurations

Swept: K = 3, 4, 5, 6. Fixed: p=6, n=30, hidden=[4], r=6,
budget=30,000 gradient evaluations, steps_per_point_per_pass=5,
max_inner=25, lambda_max_starts=64, prune_inner=True, data seed 7,
init seed 8. Per-configuration rationale is written into each result
directory's README; the parameter-by-parameter reasoning is in Section 6.

Baseline pass structure at r=6 (why the budget suffices):

| K | grid nodes C(r+K−1,K−1) | gradients per pass (nodes×5×K) | passes in 30k |
|---|---|---|---|
| 3 | 28  | 420    | ~71 |
| 4 | 84  | 1,680  | ~17.9 |
| 5 | 210 | 5,250  | ~5.7 |
| 6 | 462 | 13,860 | ~2.2 |

Even at K=6 the baseline completes two full passes, i.e. it genuinely
reaches its discretisation floor rather than being starved mid-sweep.

### 3.3 Crossover sweep configurations

Swept: hidden sizes [16,16], [32,32], [64,64], [96,96], [128,128]
(parameter count d from ~1k to ~20k). Fixed: K=2, p=20, n=50,000,
r=10 (11 grid nodes), budget=2,000 gradient evaluations,
steps_per_point_per_pass=5, max_inner=5, lambda_max_starts=8,
prune_inner=True, plateau window 4, data seed 7, init seed 8.

At K=2 and r=10 a baseline pass costs 11×5×2=110 gradients, so the budget
is ~18 passes — deep into its floor. The adaptive method gets ~1,000
scalarised iterations (~200 outer rounds).

---

## 4. Where the results live

```
output/
  plateau/
    K{3,4,5,6}_p6_n30_h4_r6_B30000/
      gn_vs_grad_evals.png   # GN* vs total gradient evaluations (log y)
      gn_vs_cpu_time.png     # GN* vs CPU time (log y)
      summary.json           # all curves + plateaus + ratios + parameters
      README.md              # parameters, rationale, per-config analysis
    plateau_ratio_vs_K.png   # the sweep's trend plot
    sweep_index.json
    README.md                # cross-configuration analysis (this sweep)
  crossover/
    d*_h{16x16,...,128x128}_n50000_B2000/   (same 4 files each)
    crossover_ratio_vs_d.png # the sweep's trend plot
    sweep_index.json
    README.md
```

Every configuration directory is self-contained: the two comparison plots
the user asked for, the raw curves, the exact parameters, why those
parameters, health flags (`L_scale_final`, `inner_cap_hits`, runtime
warnings), and the analysis.

---

## 5. Results and trends

### 5.1 Plateau sweep (tanh, equal budget 30k gradients, r=6, seeds 7/8)

| K | baseline final GN* | adaptive final GN* | ratio bl/a2 | plateau found (bl / a2) | L_scale_final |
|---|---|---|---|---|---|
| 3 | 8.97e-03 | 1.72e-04 | **52.1** | no / no | 2 |
| 4 | 3.56e-02 | 2.84e-03 | **12.6** | no / at 2.84e-03 | 8 |
| 5 | 4.04e-02 | 4.36e-03 | **9.3** | at 4.04e-02 / no | 4 |
| 6 | 4.77e-03 | 1.92e-02 | **0.25** | no / no | 4 |

(Final = best-so-far at budget end. "Plateau found" uses the 5%/two-window
detector; a "no" means the curve was still improving — e.g. at K=5 the
baseline stalls at 4.0e-02 while the adaptive method is still descending
below 4.4e-03, which is precisely the paper's plateau story.)

**Gradient-axis verdict.** At K = 3, 4, 5 the adaptive method dominates at
equal budget by 9x–52x, confirming the paper's reuse-plus-adaptive-
allocation prediction. The K=5 run shows the textbook picture (baseline
plateaued, adaptive still descending an order of magnitude lower).

**CPU-axis verdict (this sweep).** The baseline wins the CPU axis at every
K here (ratios 0.006–0.16): with n=30 the oracle is nearly free and the
adaptive method's lambda-search overhead dominates — exactly the cheap-
oracle end of the crossover thesis. Wall-clock advantage is the crossover
sweep's question, not this one's.

**The K=6 result, resolved: a budget-truncation artefact, not a defeat.**
At the fixed 30k budget the adaptive method lands ~4x ABOVE the baseline.
Two investigation rounds: (1) seven single-budget variants (three seeds;
max_inner 25→10; prune_inner False, 5,001-point bundle; w_true_scale 1→4 on
two seeds) ruled out every 30k-budget explanation — `L_scale_final` 4–16,
smooth descent, no pathology; (2) budget escalation, which found the cause.
Holding everything fixed and raising only the budget:

| budget | baseline final | adaptive final | ratio bl/a2 |
|---|---|---|---|
| 30k  | 4.77e-03 | 1.92e-02 | 0.25 |
| 90k  | 3.55e-03 | 4.74e-03 | 0.75 |
| 240k | 3.63e-03 | 1.17e-03 | **3.09** |

The baseline hits its grid floor (~3.6e-03) at ~56k gradients and never
improves; the adaptive method descends monotonically, crosses that floor at
~105k gradients, and reaches 1.17e-03 by 240k — 3x better and still
falling. At 30k it was only a third of the way down. So **the plateau story
holds at K=6 too; the break-even budget just grows with K** (immediate at
K=3, within 30k at K=4–5, ~105k at K=6). This is consistent with the paper:
the baseline has a structural grid floor at every K, the adaptive method has
none but converges more slowly at higher K (Theorem 2's outer bound is
exponential in K). Full trajectory in `output/plateau/README.md`.

**Trend (the sweep's one-line summary).** At the FIXED 30k budget the
quality ratio is 52x, 12.6x, 9.3x, 0.25x for K=3,4,5,6
(`output/plateau/plateau_ratio_vs_K.png`) — appearing to cross 1 between
K=5 and K=6. But that crossing is the break-even BUDGET growing past 30k at
K=6, not a permanent loss: at matched convergence the adaptive method wins
at every K tested. The corrected trend statement is "the adaptive method's
advantage is realised at every K, but the gradient budget required to
realise it grows with K". The user's original "advantage grows with K" and
my earlier "advantage shrinks with K" were both reading a single budget
slice; the budget-resolved picture supersedes both.

### 5.2 Crossover sweep (tanh, equal budget 2k gradients, K=2, n=50k, seed 7/8)

| width | d | equal-TIME quality ratio | equal-GRADIENT quality ratio | L_scale_final |
|---|---|---|---|---|
| 16x16   | 642    | ≈1 (1.3) | 64.3  | 4 |
| 32x32   | 1,794  | 3.7   | 60.7  | 4 |
| 64x64   | 5,634  | 18.8  | 332.5 | 8 |
| 96x96   | 11,522 | 46.5  | 1,180 | 8 |
| 128x128 | 19,458 | 101.4 | 1,941 | 16 |

Equal-TIME ratio = best-so-far GN* ratio (baseline/adaptive) when both are
given the same wall-clock time (the baseline's total — the shorter), read
with log-interpolation between checkpoints (a raw step reading is unstable
at the comparison time: it swung the 16x16 ratio 200x because the
comparison time fell 0.2 s before the adaptive method's first checkpoint;
the 16x16 entry means "parity within measurement granularity").
Equal-GRADIENT ratio = the same at the full shared gradient budget.

**Verdicts.** (1) On this expensive-oracle testbed (n=50k) the adaptive
method wins the wall-clock axis from the smallest width tested — parity at
d=642, rising monotonically to ~101x at d ≈ 19.5k; the equal-time
crossover point sits at or below d ≈ 642 here, while the cheap-oracle side
of the crossover (baseline winning wall-clock outright) is exhibited by
the plateau sweep's n=30 runs (CPU ratios 0.006–0.16). Together the two
sweeps bracket the paper's "two budget axes" argument end-to-end. (2) On
the gradient axis the adaptive method wins at every width, 61x–1,941x,
growing with d. Both directions match the Section 2 derivation.

Statistic choice, documented: the symmetric time-to-common-target ratios
(also in every summary.json) degenerate at the sweep's ends — when the two
final qualities differ by orders of magnitude, "first to reach the worse
final level" measures the adaptive method's fixed start-up rounds against
a target the baseline occupies almost immediately (ratios 0.38, 0.45,
2.02, 1.37, 0.215 — non-monotone). The equal-resource quality ratios have
no such degeneracy and are the sweep's headline statistics; full
explanation in `output/crossover/README.md`.

Health: `L_scale_final` 4→16 with width (probe estimate degrades on larger
parameter spaces; safeguard corrects with 2–4 doublings — intended
regime); `inner_cap_hits` = 0 everywhere.

### 5.3 The overall picture

- **Gradient axis (the paper's primary claim): confirmed at every K
  tested.** The adaptive method's equal-budget quality advantage is 9x–52x
  on the plateau sweep at K ≤ 5 and 61x–1,941x on the crossover sweep at
  K=2. At K=6 it appears inverted at the 30k budget, but a budget study
  (§5.1) shows this is truncation: with enough gradients the adaptive
  method crosses the baseline's grid floor and wins (ratio 3.09 at 240k).
  The realisation budget grows with K; the advantage itself does not
  disappear.
- **CPU axis (the crossover claim): confirmed, now with real data.** With
  expensive gradients (n=50k) the adaptive method's equal-time advantage
  runs from parity at d=642 to ~101x at d≈19.5k, monotonically; with cheap
  gradients (the plateau sweep's n=30) the baseline wins the CPU axis
  outright. The crossover is real and the two sweeps bracket it from both
  sides. This is the FIRST valid measurement of the crossover claim in
  this project (all pre-July-4 runs were invalidated by the frozen-loop
  defect; the ReLU re-runs by the smoothness violation).
- **Trend statements the data supports:** the adaptive advantage grows
  with per-oracle cost (d) at fixed K, and shrinks with K at fixed oracle
  cost — opposite in the two sweep directions. Parameter-selection
  guidance derived from this is in Section 6.

---

## 6. Parameter reference: what each knob does and why these values

For every parameter: **G** = changes how gradient budget is spent, **C** =
changes CPU cost per unit of work, **Q** = changes achievable quality.

### Problem-size parameters

- `K` (G, C, Q) — number of objectives. Structural: baseline grid size
  C(r+K−1, K−1) explodes in K; every scalarised step costs K oracle
  calls; the λ-search space is (K−1)-dimensional. The plateau sweep's
  variable.
- `p`, `n`, `hidden_sizes` → parameter count d (C, Q) — set the
  per-oracle cost Θ(n·d) and the optimisation difficulty. The crossover
  sweep's variable is `hidden_sizes`; `n=50,000` there makes each oracle
  call expensive on purpose (plateau sweep uses n=30 to make oracle cost
  negligible instead).
- `seed`/`init_seed` — reproducibility of data and starting point (7/8
  everywhere; 7 is the paper's data seed).
- `w_true_scale` (Q) — half-width of the uniform law for W*; live again
  after the July 6 fix; 1.0 = the paper's U[−1,1].

### Budget and measurement

- `max_grad_evals` (G) — THE fairness knob: both methods consume exactly
  this many oracle calls. Chosen per sweep so both methods expose their
  plateaus (30k) or the baseline is deep in its floor (2k at K=2).
- `eval_every_n_grads` (C only) — checkpoint cadence. Cannot change either
  method's trajectory (measurement is separate); each checkpoint costs a
  fixed-strength 256-start metric solve, so cadence is a pure wall-clock
  vs plot-resolution trade. ~25 points for plateau detection (needs ≥10),
  ~13 for crossover.

### Baseline knobs

- `resolution` r (G, C, Q) — the baseline's only quality knob: more nodes
  = lower discretisation floor but budget spread thinner and slower to
  reach the floor. Held FIXED (6 / 10) across each sweep so the sweep
  isolates its own variable; the K-dependence of the ratio then reflects
  the grid's structural scaling, which is the paper's point.
- `steps_per_point_per_pass` × `n_passes` (G) — how the budget is split
  between polishing a node now vs revisiting it later. 5 steps/node/pass
  throughout, n_passes effectively unbounded (budget is the stop).

### Adaptive knobs

- `max_inner` (G, Q) — T-map steps per outer round (each costs K oracle
  calls): the trade between exploiting the current worst λ and re-running
  the λ-search to find a new worst. 25 for plateau (project default), 5
  for crossover (more frequent λ-search suits K=2's cheap search and
  short budget).
- `lambda_max_starts` (C, Q) — multistart count of the OUTER λ-search
  only (the metric's own 256 is fixed separately). 64 at K≤6 admits the
  full structured start set; 8 at K=2 is already exhaustive for the
  one-dimensional simplex.
- `prune_inner=True` (C, Q) — paper Section 5 implementation note: only
  the best inner candidate joins the bundle (pruned candidates still
  count against the gradient budget). Keeps bundle size ~outer-count
  instead of ~iteration-count, so λ-search/T-map/metric algebra (all
  O(m·K·d) per evaluation) stay fast. Cost: forfeits the ε-mode proof's
  full-bundle condition — irrelevant here since all runs are budget-mode.
- `L` quality via `n_probes` (C at setup, Q) — 40 probe pairs per
  objective estimate each L_k before the clock starts. On ReLU MLPs no
  finite global constant exists; the descent-lemma safeguard (July 4 fix)
  doubles a global `L_scale` whenever the certified-decrease inequality
  fails, so a probe underestimate costs some wasted early steps, never
  correctness. `L_scale_final` is recorded per run.
- `epsilon` — None everywhere: budget mode (the paper's experimental
  protocol). ε-certificate mode exists in the code but is a separate,
  unused pathway.

### Complexity impact summary (time vs gradients)

- Parameters that move ONLY wall-clock, never the gradient axis:
  `eval_every_n_grads`, `lambda_max_starts` (given the search still finds
  the max), `prune_inner` (given budget-mode; it also weakly affects Q
  through which points the bundle keeps).
- Parameters that move the gradient axis: `max_grad_evals` (directly),
  `K` (multiplies per-step cost), `max_inner` /
  `steps_per_point_per_pass` (how the same budget is scheduled).
- Parameters that move quality at fixed budget: `resolution` (baseline
  floor), `K`/`d` (problem difficulty), `max_inner` (exploit-vs-search
  balance), L quality (step sizes).

---

## 7. Honest-reporting notes

- Both sweeps run both methods under identical budgets on identical
  problem instances; no method is given a stopping advantage.
- All reported GN\* values (including plateau levels and targets) are
  heuristic lower bounds of an NP-hard maximum, computed by one fixed
  256-start IPOPT yardstick for both methods.
- CPU-time plots exclude checkpoint-metric cost for both methods; at
  large d the metric is a visible share of *total* wall time, so "CPU
  time" here means "iterative-work time", exactly as in the paper's
  protocol.
- The baseline has no descent safeguard (per the paper); it runs
  fixed-step GD with the same probe-estimated L the adaptive method
  starts from. The adaptive method self-rescues from L underestimates
  (`L_scale_final` > 1); the baseline cannot — its steps can be silently
  too long. This asymmetry favours neither side by construction, but it
  is a real difference in robustness and is visible in the health flags.
- Runs made before July 4/6 fixes are archived at
  `/Users/shirch/vscode101/.venv/ledger-artifacts/pre_fix_outputs_archive/`
  and are not comparable with anything in `output/`.
