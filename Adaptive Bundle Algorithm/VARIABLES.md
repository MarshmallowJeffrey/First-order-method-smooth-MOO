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

---
---

# Part 2 — knobs added after July 8 (written Aug 25, 2026)

Part 1 above (sections 1–3) covers the original engines and is kept as
written. The later generations (`_fast`, `_ccp`, the SVRG baselines and
the campaign runners) reuse those knobs where they apply and add the
ones below. File-by-file context: `CODE_MAP.md`; run commands:
`MANUAL.md` Part 2.

## 4. Fast engine — `algorithm_fast_without_256_checkpoints.py`, `algorithm_adaptive_fast`

| variable | meaning | default |
|---|---|---|
| `lambda_tier_mode` | λ-search tiering: `"strict"` = every round pays the full multistart search; `"two_tier"` = cheap tier (centroid+vertices+prev starts) with periodic strict verification | `"strict"` — the only honest choice on the K=6 MLP: the two-tier cheap meter was proven to under-report ~2x AND mistarget (v3 diagnosis) |
| `lambda_max_starts` | starts of the strict tier | 64 |
| `cheap_tol` / `cheap_max_iter` | cheap-tier solve tolerance / iteration cap | 1e-4 / 30 |
| `strict_tol` / `strict_max_iter` | strict-tier solve tolerance / iteration cap | 1e-8 / 100 |
| `sticky_strict` | once strict fires, stay strict | True |
| `msvrg_step_const` | Momentum-SVRG inner step constant | 0.1 |
| `msvrg_momentum` | inner momentum β | 0.9 (the pure-budget runners set 0.5) |
| `msvrg_epoch_len` | minibatch steps per segment (one segment = `epoch_len` steps + 1 full joint evaluation) | None = auto rule; the K=6 pure-budget campaign resolves to 13 |
| `msvrg_max_segments` | segment cap per outer round; hitting it is the `cap_hits` health flag (in budget mode a hit is accepted, in target mode it marks an unreachable inner target) | 10 |
| `msvrg_trigger_rho` / `msvrg_trigger_patience` | early-exit trigger of the inner loop (REMOVED from the pure-budget protocol: segments run to length there) | 0.7 / 2 |
| `msvrg_rel_target` | relative inner target (fraction of the round's search value) | None (v4 probe 0.1; fixed-budget run 0.05) |
| `prune_grid_r` | probe-λ grid resolution for delivery-time pruning (bitwise-checked) | 10 |
| `epsilon`, `max_outer`, `max_grad_evals`, `eval_every_n_grads`, `require_ipopt` | as in Part 1 | 1e-3 / 150 / None / None / True |

`objectives_torch_fast.StochLamOracle` (the minibatch oracle): `batch_size`
b, stratified across classes ∝ n_k (campaign standard b = 4096; MNIST runs
use 1024), `seed`. Gradient-equivalent accounting everywhere in this
generation: one full joint call = K units; one minibatch step = 2·b·K/n
units; x0 and metric/audit work stay off-axis.

## 5. SVRG-certified baseline — `baseline_svrg_certified_without_256_checkpoints.py` via `run_baseline_svrg_r_sweep_without_256_checkpoints.py`

| flag | meaning | default |
|---|---|---|
| `--r-list` | grid resolutions swept, comma-separated | `10,12,15,20` |
| `--node-tol` | per-node certification level on ‖∇F_{λ_i}‖² | 0.02 (the tol0.01 home used 0.01) |
| `--solve-target-frac` | inner solve target as a fraction of node_tol | 0.25 |
| `--share-mode` | certificate sharing between nodes | `"gram"` (Gram-based share; all 12→N nodes may be signed by share) |
| `--ckpt-every-grads` | checkpoint cadence | 4,500 |
| `--max-wall-per-r` / `--max-grads-per-r` | per-resolution fuses | 14,400 s / 2e6 |
| `--save-grams` | store `delivery_audit.npz` (per-node Grams for the between-node audit) | off |
| `--out-dirname` | output home override (used to build the v2 homes) | old home |
| `--fast-ref` | path to an adaptive `summary.json` drawn as the reference curve | None |
| `--replot` | redraw figures from stored summaries, no runs | off |

## 6. Pure fixed-budget protocol — `run_pure_budget_{K6,K2}(_ccp)_…`, `run_fixed_budget_K6_…`

The protocol has NO tolerance parameter anywhere. Shared knobs:

| flag | meaning | values of record |
|---|---|---|
| `--run` | which leg: `baseline` or `adaptive` (one leg per invocation, serial) | — |
| `--budget` | total gradient-equivalent budget B | K6: 80,912 (= r15@0.02's realized cost); K2 and MNIST pairs: 20,000 |
| `--s` | segments spent per allocation decision | 5 main; 1 sensitivity legs |
| `--r` | baseline grid resolution | K6: 10/12/15/20; K2: 10/20/40/80; pairs: 10/20/40 |
| `--targeting-starts` | starts of the adaptive worst-λ search (its decision policy) | K6: 24; K2: 64 (ts64 leg) and 24 (ts24 leg) |
| `--eval-every` | checkpoint cadence in grad-equivalents | K6: 2,000; K2: 250 |
| `--backfill-audits` | add strict 64-start prefix audits to finished baseline legs (never-understate merge) | — |
| `--figure` / `--replot` | redraw campaign figures from stored data | — |
| `--force` | allow overwriting a completed leg | off |

K=2 exact-meter extras (`run_pure_budget_K2_…`): `--decision-mode`,
`--decision-grid` (default 2,001) and `--audit-grid` (default 200,001)
— at K=2 the simplex is 1-D, so quality is measured by an exact dense
grid, no multistart search in any measurement. The CCP leg
(`run_pure_budget_K2_ccp_…`) swaps only the next-λ policy and adds
`--ccp-N0` (2,000), `--ccp-r` (10), `--ccp-seed` (0).

`run_fixed_budget_K6_…` (protocol 5e, the precursor): `--budget 80912`,
`--rel-target 0.05`, `--targeting-starts 24`, `--eval-every 2000`,
`--max-outer 5000`, `--tag`.

## 7. CCP λ-solver — `ccp_lambda_solver.CCPConfig`

| field | meaning | default |
|---|---|---|
| `N0` | random seeds sampled per round (static mode) | 2,000 (heavy audit instrument: 8,192) |
| `r` | restarts polished by CCP per round | 10 (heavy audit: 20) |
| `pool_cap_factor` | carried cross-round pool cap = factor × r | 3 |
| `tau_rel` | relative stationarity tolerance of a CCP restart | 1e-8 |
| `tau_eps_frac` | safety cap 0.01×epsilon on tau (rarely binds) | 0.01 |
| `T_max` | CCP iteration cap per restart | 100 |
| `seed_sampler` | random-seed law: `"exp"` (Exp(1)-normalized) or `"sobol"` (scrambled) | `"exp"` — Study A found no significant difference; exp kept |
| `adaptive_seed_schedule` | rho-rule shrinking of N0 (ablation switch) | False |
| `n_new_floor_factor` | shrink floor = factor × r when the schedule is on | 10 |
| `rho_low` | schedule band edge | 0.25 |
| `screen_sep_l1` | l1 separation enforced among retained seeds | 0.05 |
| `dedup_l1_tol` / `dedup_phi_rel` | pool dedup: same-point l1 / phi proximity | 1e-3 / 1e-9 |
| `active_tol` | tolerant active-set threshold (rel) | 1e-9 |
| `collapse_frac` | pool-collapse trigger fraction | 0.5 |
| `seed` | rng / Sobol scramble seed | 0 |
| `use_highspy` | force/forbid HiGHS for the game LP | None = auto-detect |

## 8. Bandit-toy runners — `run_bandit_toy{,_K5,_mv}_without_256_checkpoints.py`

| flag | meaning | default |
|---|---|---|
| `--epsilon` | the accuracy rung (recorded rungs: 1e-2, 1e-3, 1e-4; mv: 1e-2, 1e-3) | 1e-2 |
| `--eval-every` | checkpoint cadence in gradients; **0 = per-segment recording**, the precise-readout mode (session 13) — coarse-cadence first-crossing readouts are upper-bound artifacts | 10 |
| `--max-grad-evals` / `--max-wall` | fuses | 200,000 / 3,600 s (K5: 7,200 s) |
| `--smoke` | tiny run into `smoke/` | off |
| mv only: `--gamma` | variance weight (kills the closed form) | None = value recorded by `gamma_scan.json` |
| mv only: `--gamma-scan`, `--gamma-list` | scan γ candidates | — |
| mv only: `--rebuild-reference` | rebuild the untimed multistart ground-truth table (`reference_gamma1_*.npz`) | — |
| mv only: `--epsilons` | run several rungs in one call | None |

Bandit-only semantics: the equal-level stop grants the baseline a
terminal global property its native theory does not promise — NEVER
cite terminal GN\* as baseline coverage evidence; coverage weakness
shows in value/PF metrics and audits.

## 9. MNIST runners

`run_ccp_compare_K10_mnist_…` (K=10 patch-softplus): `--budget` 55,000,
`--eval-every` 1,500, `--per-class` 1,000 (first N train images per
class), `--batch` 1,024, `--s` 5, `--ts` 24 (IPOPT leg's strict
starts), `--ah16-faithful` (ablation switch), `--ccp-seed` 0.
Problem family: `objectives_mnist_patch.py` (patch-connected softplus
MLP, d = 8,874).

`run_pure_budget_K2_mnist_pair_…` (Experiment 4): `--pair a b` (the
two digits; campaigns of record 3 5 and 7 9), `--budget` 20,000,
`--eval-every` 250, `--audit-grid` 200,001, `--s` 5, `--ccp-seed` 0.
per_class is the balanced maximum for the pair (5,421 for 3v5); batch
1,024; test values use ALL official t10k rows of the two digits.
Problem family: `objectives_mnist_pair.py` (d = 8,098).

## 10. Audit instruments (not knobs of any single run)

- The family instrument everywhere: strict 64-start λ-search.
- August campaigns: `audit_v2 = max(strict-64 IPOPT, heavy CCP with
  N0 = 8,192, r = 20, fresh solver)` per delivered stack
  (`audit_v2_K6_…py`, `--quick` = first 3 stacks per leg) — both are
  lower bounds of an NP-hard max, so the max is a tighter, method-
  symmetric lower bound.
- Where audits are load-bearing, the monotone lower-bound envelope is
  applied (prefix GN\* is non-increasing; raw values are kept
  alongside).
- Time accounting: the adaptive λ-search is ON the CPU axis (it steers
  the run); checkpoint metric and audit work are OFF both axes
  (`metric_seconds` / `audit_seconds` in the summaries).

## 11. Which of the new knobs matter most

| rank | knob | why |
|---|---|---|
| 1 | `--budget` B | the whole contest is defined at fixed B; every verdict is "at this budget" |
| 2 | `--s` | coverage collapse lever: at K=6, s=5 lets baseline grids visit only 39%/19%/7.5%/2.2% of nodes for r=10/12/15/20 |
| 3 | `--r` | the baseline's budget dial (node count C(r+K−1, K−1)) and quality floor |
| 4 | `--targeting-starts` / CCP `N0`,`r` | decision quality AND the decision-time cost on the CPU axis |
| 5 | batch b | the inner-solver floor: b = 4,096 evidence says targets ≲ 0.01 hit the segment cap from some anchors; the designed next lever is 8,192/16,384 or b = n |
| 6 | `lambda_tier_mode` | strict is the only honest measurement mode on the MLP family |
| 7 | `--eval-every` | wall-clock only (off-axis), but real waiting time |
