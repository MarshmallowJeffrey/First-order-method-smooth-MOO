# Variable reference: every knob, where it lives, its default, and which ones matter

Date: July 8, 2026. 

## 1. How the pieces connect

```
objectives_torch.py     make the problem: data, K objectives, gradients,
 (objectives_numpy.py)  smoothness estimates L, parameter count d
        |
        v
baseline.py             Algorithm 1: grid of weightings, gradient descent
algorithm.py            Algorithm 2: adaptive bundle method
        |
        v
experiments.py          run both on ONE shared problem under ONE budget,
                        checkpoint them, detect plateaus, plot
        |
        v
run_experiments.py      the official parameter values for the two sweeps;
                        writes summary.json / README / trend plots
```

The `*_without_256_checkpoints.py` files are A/B measurement variants of
`baseline.py` / `algorithm.py` with the same knobs; only what checkpoints
record differs.

---

## 2. Inventory by file

### 2.1 Problem generation — `objectives_torch.py`, `make_mlp_nonconvex`

| variable | meaning | default |
|---|---|---|
| `K` | number of objectives (classes) | 3 |
| `p` | input dimension of the data | 4 |
| `n` | number of data samples | 60 |
| `hidden_sizes` | widths of the MLP hidden layers, e.g. `[4]` or `[96, 96]` | `[8]` (via `h` shorthand: `h=4` means `[4]`) |
| `seed` | random seed for the planted data (W*, X) | 7 |
| `w_true_scale` | half-width of the uniform law for the planted W*: entries ~ U[−w, w] | 1.0 (the paper's U[−1,1]) |
| `n_probes` | number of probe pairs per objective used to ESTIMATE the smoothness constants L before the clock starts | 40 |
| `activation` | hidden nonlinearity: `"relu"`, `"tanh"`, `"softplus"`, `"identity"` | `"relu"` (benchmarks pass `"tanh"` — ReLU violates the paper's smoothness assumption) |

Derived, not settable directly: `d` = total parameter count of the MLP
(grows with `p` and `hidden_sizes`; e.g. `[4]` at K=5,p=6 gives d=53;
`[96,96]` at K=2,p=20 gives d=11,522).

`objectives_numpy.py` has `make_logreg_strongly_convex(K=5, p=4, n=60,
reg=0.1, seed=42, w_true_scale=1.0)` — a strongly convex logistic
regression testbed used ONLY by the verification scripts, never by the
experiments.

`experiments.make_mlp_initial_point(K, p, h=None, seed=0, hidden_sizes=None)`
builds the shared starting point x0 (benchmarks pass seed 8).

### 2.2 Baseline — `baseline.py`, `uniform_discretisation`

Required inputs: `K`, the objectives and gradients, `L`, `x0`, and
`resolution`.

| variable | meaning | default | benchmark |
|---|---|---|---|
| `resolution` (r) | grid density on the weight simplex; the grid has C(r+K−1, K−1) nodes | (required) | 6 (plateau), 10 (crossover) |
| `n_passes` | how many sweeps over the whole grid are allowed | 1 | 100,000 (never the stopper) |
| `steps_per_point_per_pass` | gradient-descent steps at each node per sweep | 20 | 5 |
| `eval_every_n_grads` | checkpoint every this many gradient evaluations (None = once per pass) | None | budget/25 (plateau), budget/13 (crossover) |
| `max_grad_evals` | hard budget on total gradient evaluations | None | 30,000 / 2,000 |
| `node_tol` | OPTIONAL certification mode: per-node acceptance level on ‖∇F_{λ_i}‖²; None = budget mode | None | None (unused) |
| `evaluate_coverage` | whether checkpoints measure GN* at all | False | True |
| `joint_oracle` | fused evaluator returning all K values+gradients in one call (a speed device, not a semantics change) | None | the fused torch oracle |
| `verbose` | progress printing | False | True |

Internal state worth knowing: `node_served` and `node_grad_sq`
(certification bookkeeping; in the A/B variant `node_grad_sq` also feeds
the self-reported checkpoint value).

### 2.3 Adaptive method — `algorithm.py`, `algorithm_adaptive`

Required inputs: `K`, `d`, the objectives and gradients, `L`, `x0`.

| variable | meaning | default | benchmark |
|---|---|---|---|
| `max_outer` | cap on outer rounds (λ-search + inner steps) | 120 | 1,000,000 (never the stopper) |
| `max_inner` | cap on T-map steps per outer round; each step costs K gradient evaluations | 25 | 25 (plateau), 5 (crossover) |
| `epsilon` | OPTIONAL certified mode: outer loop stops when the method's own search value ≤ 2ε/3; inner target ε/3 | None | None (unused) |
| `eval_every_n_grads` | checkpoint cadence (None = every outer round) | None | budget/25, budget/13 |
| `target_cov` | OPTIONAL: stop when the measured checkpoint value ≤ this (used by the legacy time-to-target driver) | None | None |
| `lambda_max_starts` | multistart count of the method's OWN per-round λ-search | 256 | 64 (plateau), 8 (crossover) |
| `lambda_solver` | `"ipopt"` or `"slsqp"` for the λ-search | `"ipopt"` | `"ipopt"` |
| `require_ipopt` | refuse to run if IPOPT is missing (instead of silently falling back to SLSQP); with the default True, an explicit `lambda_solver="slsqp"` run must also pass `require_ipopt=False` | True (was False until July 8) | True |
| `max_grad_evals` | hard budget on total gradient evaluations | None | 30,000 / 2,000 |
| `prune_inner` | keep only the best inner candidate in the bundle (paper §5 note); pruned candidates still cost budget | True | True |
| `joint_oracle`, `verbose` | as in the baseline | None / False | fused / True |

Internal state worth knowing:

- `L_scale` — the descent-lemma safeguard's multiplier on L. Starts at 1,
  doubles every time the certified-decrease inequality fails, is reported
  as `L_scale_final`, and aborts the run above 2^60 (the objective is then
  not L-smooth along the iterates). This is a HEALTH FLAG: 2–16 is the
  intended occasional-correction regime.
- `m` — the bundle size (number of stored points). Grows by 1 per outer
  round with `prune_inner=True`, by up to `max_inner` per round without.
- Fixed constant, deliberately NOT a knob: the checkpoint metric `pc_star`
  always uses 256 starts, independent of `lambda_max_starts`, so it is one
  comparable yardstick across all runs and methods.

### 2.4 Experiment driver — `experiments.py`

`experiment_mlp_plateau_comparison` bundles all of the above for one
head-to-head run. Its own defaults (development values): K=3, p=4, n=60,
seed=10, init_seed=None (meaning seed+1), coarse_resolution=10,
n_passes=1000, steps_per_point_per_pass=10,
baseline_eval_every_n_grads=None, adaptive_eval_every_n_grads=2000,
max_grad_evals=30000, max_outer=10000, max_inner=25,
lambda_max_starts=256, prune_inner=True, hidden_sizes=None,
activation="relu", w_true_scale=1.0, plus the plateau-detector knobs
below. The benchmarks override most of these (Section 2.5).

Plateau detector (`detect_plateau`) — measurement only, never changes a
trajectory:

| variable | meaning | default | benchmark |
|---|---|---|---|
| `plateau_window` | checkpoints per stability window | 5 | 5 (plateau), 4 (crossover) |
| `plateau_relative_improvement_tol` | "improving by less than this fraction" counts as flat | 0.05 | 0.05 |
| `plateau_consecutive_windows` | how many flat windows in a row are required | 2 | 2 |

Also in this file: `experiment_mlp_gn_coverage` — the LEGACY time-to-target
driver (baseline runs its full schedule, its final value becomes the
adaptive method's `target_cov`). Asymmetric by design; not used for any
headline number; kept for quick single-config comparisons.

### 2.5 The official values — `run_experiments.py`

Constants: `DATA_SEED = 7`, `INIT_SEED = 8` (identical across every
benchmark run).

Plateau sweep (`plateau_configs`): K ∈ {3,4,5,6}, p=6, n=30,
hidden_sizes=[4], activation="tanh", r=6, budget 30,000, cadence 1,200,
steps_per_point_per_pass=5, max_inner=25, lambda_max_starts=64,
prune_inner=True, n_passes=100,000, max_outer=1,000,000.

Crossover sweep (`crossover_configs`): K=2, p=20, n=50,000,
hidden_sizes ∈ {[16,16],[32,32],[64,64],[96,96],[128,128]},
activation="tanh", r=10, budget 2,000, cadence 153,
steps_per_point_per_pass=5, max_inner=5, lambda_max_starts=8,
prune_inner=True, plateau_window=4.

`--smoke` runs shrunken versions of both into separate `*_smoke` output
folders.

---

## 3. The variables that matter most

### 3.1 What actually STOPS a run

A run ends at the FIRST of the following to trigger. In every official
experiment, only item 1 is active — that is the equal-budget design.

Baseline:

1. `max_grad_evals` — the gradient budget. **The active stopper in all
   benchmark runs.**
2. `n_passes` exhausted — set to 100,000 in benchmarks, i.e. never
   reached.
3. `node_tol` all-nodes-served (certification mode) — off by default;
   when on, `max_grad_evals` remains as the fuse and a failure to certify
   within it is reported honestly.

Adaptive method:

1. `max_grad_evals` — the gradient budget. **The active stopper in all
   benchmark runs.**
2. `max_outer` exhausted — set to 1,000,000 in benchmarks, never reached.
3. `epsilon` (certified mode) — stop when the method's own λ-search value
   ≤ 2ε/3; off in all benchmarks.
4. `target_cov` — stop when the measured checkpoint value reaches a
   target; only the legacy time-to-target driver uses it.

Two more caps shape behaviour WITHIN a run without stopping it:
`max_inner` ends one inner loop (back to a fresh λ-search), and
`steps_per_point_per_pass` ends one node visit (move to the next node).

### 3.2 What drives cost and running time

The cost model in words:

- One gradient evaluation costs roughly n·d work (one forward+backward
  pass over the dataset). One scalarised step costs K gradient
  evaluations.
- **Total oracle work = max_grad_evals × (cost of one evaluation).** This
  is shared by both methods by design.
- The baseline pays essentially nothing else. Its structure knob is the
  grid: N = C(r+K−1, K−1) nodes decide how thinly the budget is spread
  (and where its quality floor sits).
- The adaptive method pays EXTRA CPU (no extra gradients) every outer
  round: the λ-search runs up to `lambda_max_starts` local solves, and
  every solver iteration costs about m·K·d arithmetic (m = bundle size);
  the T-map step selection costs the same order per step. The number of
  outer rounds in a budget is about max_grad_evals / (K · max_inner), and
  with `prune_inner=True`, m ≈ that round count.
- Checkpoints cost wall-clock only (excluded from both reported axes but
  real waiting time): each one runs the fixed 256-start metric solve,
  again m·K·d per solver iteration. Number of checkpoints =
  max_grad_evals / eval_every_n_grads. (A comment in
  `experiment_mlp_gn_coverage` records the real incident: checkpointing
  10x more often made total wall time WORSE, because each checkpoint pays
  the 256-start solve.)

Ranked by practical impact:

| rank | variable | why it dominates |
|---|---|---|
| 1 | `n`, `hidden_sizes`/`p` (i.e. n·d) | the price of EVERY gradient evaluation, for both methods; the crossover sweep is precisely a sweep of this price |
| 2 | `max_grad_evals` | multiplies everything; also decides whether the adaptive advantage is visible at all (the K=6 budget lesson) |
| 3 | `K` | triple effect: every step costs K evaluations, the baseline grid explodes as C(r+K−1,K−1), and the λ-search lives in K−1 dimensions |
| 4 | `resolution` r | baseline node count (polynomial of degree K−1 in r) and its quality floor |
| 5 | `max_inner` | sets the outer-round count for a given budget, hence bundle growth AND how many λ-searches are paid |
| 6 | `prune_inner` | bundle size m multiplies ALL adaptive algebra (λ-search, T-map, metric); True keeps m ≈ rounds, False lets it grow max_inner× faster |
| 7 | `lambda_max_starts` | linear multiplier on the adaptive method's per-round CPU overhead (never on gradients) |
| 8 | `eval_every_n_grads` | wall-clock only, via the number of 256-start metric solves |

Mode switches that change SEMANTICS rather than cost: `epsilon`,
`node_tol`, `target_cov` (budget mode → certified/targeted stopping);
`activation` (tanh keeps the smoothness assumption valid; ReLU breaks it
and invalidates benchmark use); `seed`/`init_seed` (which problem instance
you are on).

Quick lookup — "my run is too slow, which knob?":

- Waiting too long between/at checkpoints → raise `eval_every_n_grads`
  (fewer metric solves; plots get coarser, trajectories unchanged).
- Adaptive rounds too slow at large d → lower `lambda_max_starts`, keep
  `prune_inner=True`, or raise `max_inner` (fewer searches per budget).
- Everything too slow → the real levers are `n`, `hidden_sizes`, and
  `max_grad_evals`; nothing else changes the oracle bill.
