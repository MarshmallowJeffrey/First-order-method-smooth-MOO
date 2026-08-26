# Original_py — file map

All Python sources of the adaptive-bundle smooth-MOO project: algorithm engines,
objective families, baselines, experiment runners, post-hoc audits/plots, and
sanity gates. This document is the organizational map of the `Original_py/`
folder. (Until Aug 25, 2026 it lived at `Original_py/README.md`; it was moved
here, next to `MANUAL.md` and `VARIABLES.md`, so all project-wide documentation
sits in one place.)

## ⚠ Aug 25 reorganisation notice

Later on Aug 25 the user moved the sources into five subfolders,
overturning the flat-layout premise the warning below describes:

| subfolder | contents |
|---|---|
| `Core Engine/` | `algorithm*.py` (all four engines), `bundle.py`, `bundle_fast.py`, `ccp_lambda_solver.py` |
| `baseline/` | the three `baseline*.py` files |
| `objective/` | the seven `objectives_*.py` files |
| `experiment_plot/` | every `run_*.py`, `plot_*.py`, `audit_v2_*.py`, `diag_*.py`, `front_metrics_*.py`, `rebuild_cert_partial.py`, `experiments.py` |
| `sanity_check/` | the five `sanity_checks_*.py` gates |

`run.sh` stays at the top. The split initially broke bare-name sibling
imports and every `parent.parent / "output"` anchor; a user-approved
PATCH the same day restored operation:

- each subfolder carries `_layout.py` (appends all five subfolders to
  `sys.path`); every module with sibling imports does
  `import _layout` before them;
- output/data anchors were deepened one level
  (`HERE.parent.parent / "output"`, `parents[2]`, …);
- output-home constants now point at the reorganised locations
  (`CCP/…`, `Bandit_toy/…`, `pure_budget_…_Baseline/…`,
  `fast_method_trials_v1v2v3_IPOPT`, `with_256_checkpoints/…`, …);
- `run.sh` resolves a bare script name across the subfolders.

Verified after the patch: all 52 modules import, 16 stored-result path
constants resolve, and all five sanity gates plus the two legacy
verifiers PASS. The per-file descriptions below remain correct; prefix
each file with its subfolder. Output paths named below use the
pre-reorg names as historical identifiers — translate via
`output/README.md`.

> **Why the folder stays flat (do not move files into subdirectories).**
> 1. Every module imports its siblings by bare name (`from bundle import Bundle`),
>    and runners even import other runners' executors
>    (`run_ccp_compare_K2` → `run_pure_budget_K2` → `run_pure_budget_K6` → `run_fixed_budget_K6`).
> 2. Every script anchors its output home via
>    `Path(__file__).resolve().parent.parent / "output"` — moving a file would
>    silently relocate where it reads/writes results.
> 3. Project convention: existing files are never edited; each stage adds new
>    files. Rewriting 49 files' imports to support subfolders would break that.
>
> Run everything from this directory (see [How to run](#how-to-run)).

## Filename-suffix legend

The suffixes encode the project's evolution; a file's name tells you which
track and implementation generation it belongs to.

| Suffix | Meaning |
| --- | --- |
| *(none)* | Original reference implementation; checkpoints scored by the external fixed 256-start `pc_star` yardstick |
| `_without_256_checkpoints` | Measurement-variant track (the **active** track): identical trajectories, but checkpoints record the method's **own** most recent λ-search value. All later work builds on this track |
| `_fast` | July 15 acceleration set: Gram cache (GN evaluation O(m·K·d) → O(m·K²)), two-tier λ-search with stop-verify, Momentum-SVRG inner loop, delivery-time pruning |
| `_ccp` | Aug 9 λ-solver replacement: multistart convex–concave procedure (CCP) instead of IPOPT |

## Layer diagram (who imports whom)

```
bundle ──► algorithm                                  (original track, IPOPT/SLSQP)
bundle ──► algorithm_without_256_checkpoints          (same, self-reported checkpoints)
bundle ──► bundle_fast ──► algorithm_fast_without_256_checkpoints   (accelerated engine)
                                    │                        ▲
ccp_lambda_solver ──────────────────┴─► algorithm_ccp_without_256_checkpoints  (CCP twin)

objectives_torch ──► objectives_torch_fast            (adds the stochastic λ-scalarized oracle)
objectives_torch ──► objectives_mnist_patch           (MNIST patch-softplus, K = 10)
objectives_bandit_toy ──► objectives_bandit_toy_mv    (mean-variance variant)

runners import engines + objective factories; the pure-budget/CCP-campaign
runners also import each other's executors (see the warning above).
```

## 1 · Core engines

| File | What it is |
| --- | --- |
| `algorithm.py` | Original Algorithm 2 (adaptive bundle method). λ-search via cyipopt/IPOPT, SLSQP fallback. |
| `algorithm_without_256_checkpoints.py` | Same algorithm, A/B measurement variant: checkpoints record the run's own latest λ-search value (checkpoint 0 backfilled with the round-1 value). |
| `algorithm_fast_without_256_checkpoints.py` | Accelerated engine: Gram-path GN (exact rewrite), cheap/strict two-tier λ-search with stop-verify, Momentum-SVRG inner loop, gradient-equivalent accounting. The `adaptive_s5_ts24` (IPOPT) legs run this with `lambda_tier_mode="strict"`. |
| `algorithm_ccp_without_256_checkpoints.py` | CCP twin of the fast engine: everything except the λ-search is imported from `algorithm_fast_*`, so both arms share one implementation of every common part. Single tier, never needs IPOPT; returns `ccp_stats_history` telemetry. |
| `bundle.py` | Core bundle data structure `B_m` (points, per-objective values and gradients, λ-dependent smoothness). |
| `bundle_fast.py` | Bundle + Gram cache (`M_i = J_i J_iᵀ` per point) + delivery-time pruning (bitwise-checked against a probe-λ set). |
| `ccp_lambda_solver.py` | The multistart CCP λ-solver itself: sandwich bounds (`val(A)`, `λ_A`) → seeds (vertices + λ_A + carried pool + Exp(1)/Sobol random) → batched screening (top-r, l1-separated) → CCP polish via warm-started HiGHS game LP → cross-round pool. |

## 2 · Objective families

| File | What it is |
| --- | --- |
| `objectives_numpy.py` | NumPy single-hidden-layer MLP testbed (earliest, small instances). |
| `objectives_torch.py` | PyTorch multi-layer MLP testbed (autograd gradients); the [96, 96] tanh instances. |
| `objectives_torch_fast.py` | Re-exports the torch factory verbatim and adds `StochLamOracle`: the stratified, λ-scalarized minibatch oracle the Momentum-SVRG inner loop needs (one forward + one backward per step). |
| `objectives_bandit_toy.py` | SURF offline-bandit toy (K = 2, 5 arms, KL-regularized; closed-form softmax reference). |
| `objectives_bandit_toy_mv.py` | Mean-variance variant: adds a plug-in variance term so the closed form dies and scalarizations become genuinely nonconvex (`gamma` knob). |
| `objectives_mnist_patch.py` | MNIST per-class CE objectives on the patch-connected softplus MLP (K = 10, d = 8874) — the Experiment-2 (MNIST) problem family. |

## 3 · Baselines

| File | What it is |
| --- | --- |
| `baseline.py` | Original uniform-grid baseline (budget mode + certification mode). |
| `baseline_without_256_checkpoints.py` | Same trajectories, self-reported checkpoints: max over grid nodes of each node's own-weight latest gradient value (weights *between* nodes never enter — the gap the A/B experiment exposes). |
| `baseline_svrg_certified_without_256_checkpoints.py` | Grid baseline upgraded to the *same* Momentum-SVRG inner solver as the fast adaptive engine (fairness), certification-style stopping. |

## 4 · Experiment runners

### 4a · Early family (July)

| File | What it is |
| --- | --- |
| `experiments.py` | Experiment library (problem builders, equal-budget head-to-head protocol, plotting helpers). |
| `run_experiments.py` | Unified driver for the plateau sweep (vary K) and CPU-crossover sweep (vary width d); replaced the old notebooks. |
| `run_experiments_without_256_checkpoints.py` | Reruns one plateau (K=5) + one crossover (96×96) config on the measurement-variant track. |
| `run_trial_K6_without_256_checkpoints.py` | One-off July-11 trial: crossover problem scaled to K = 6, original method, budget mode. |
| `run_trial_K6_fast_without_256_checkpoints.py` | July-15 twin of the K=6 trial with the accelerated engine; compares against the stored July-11 curves (disclosed reuse). |
| `run_lambda_path_without_256_checkpoints.py` | K = 3 figure: baseline's fixed grid vs the adaptive method's chosen-λ sequence on the 2-simplex triangle. |
| `run_pareto_certified_without_256_checkpoints.py` | Certified-mode Pareto-front comparison at K = 2 (two ε combos). |
| `run_fixed_budget_K6_without_256_checkpoints.py` | Fixed-budget comparison: adaptive trajectory audited post-hoc vs baseline endpoint audits, one instrument. |
| `run_baseline_svrg_r_sweep_without_256_checkpoints.py` | Baseline r-sweep (SVRG-certified grids) vs the v3 fast curve. |
| `diag_v3_plateau_without_256_checkpoints.py` | Diagnostic replay of the v3 plateau: bitwise-verified rerun + strict prefix audits + witness-λ localisation. |
| `plot_between_node_gap_without_256_checkpoints.py` | Post-processing figures: certified grid nodes meet node_tol while λ between nodes does not. |
| `rebuild_cert_partial.py` | One-off (July 16): rebuilds summary/figures for the user-stopped cert run from its `run_log.txt`. |

### 4b · Pure fixed-budget protocol (late July)

Shared segment unit, shared `s`, chain warm start, stop = budget, **no
tolerance parameter anywhere**; the only difference between legs is the
next-λ policy.

| File | What it is |
| --- | --- |
| `run_pure_budget_K6_without_256_checkpoints.py` | Protocol's first implementation (July 27), K = 6, B = 80,912. |
| `run_pure_budget_K2_without_256_checkpoints.py` | K = 2 version (July 30) with the **exact 1-D quality meter** (no multistart search in any measurement). |
| `run_pure_budget_K2_ccp_without_256_checkpoints.py` | The CCP targeting leg for K = 2 (Aug 9): imports the whole K2 executor, swaps only the next-λ policy for `CCPLambdaSolver.solve`. |

### 4c · CCP comparison campaign (Aug 9–10) — the deck/report experiments

| File | What it is | Maps to |
| --- | --- | --- |
| `run_ccp_compare_K2_without_256_checkpoints.py` | Fresh same-machine serial rerun of all K=2 legs + the CCP leg → `output/ccp_compare_without_256_checkpoints/K2_B20000/`. | Experiment 1 (K = 2) |
| `run_ccp_compare_K6_without_256_checkpoints.py` | Mirror for K = 6 → `.../K6_B80912/`. | Experiment 1 (K = 6) |
| `run_ccp_compare_K10_mnist_without_256_checkpoints.py` | MNIST K = 10 trial, two adaptive legs only (IPOPT ts24 vs CCP); executor is a declared copy of the K2/K6 loop (originals hard-wire the planted-MLP factory). | Experiment 2 (MNIST) |
| `run_ccp_smoke_sampler_without_256_checkpoints.py` | Seed-sampler ablation Exp(1) vs scrambled Sobol on 19 frozen bundles; decides `CCPConfig.seed_sampler`. | Study A (sampler) |
| `run_lambda_solver_bench_without_256_checkpoints.py` | Controlled λ-solver benchmark on frozen Gram snapshots: 2a paired-starts polish comparison + 2b 60-second fixed-time race. | Study B (60-s race) |
| `audit_v2_K6_without_256_checkpoints.py` | Method-symmetric two-instrument audit: `audit_v2 = max(strict-64 IPOPT, heavy CCP N₀=8192/r=20)` per delivered stack. | K6/K10 quality meter |
| `front_metrics_K6_pure_budget_without_256_checkpoints.py` | Post-hoc ε-Pareto-front metrics (HV, IGD, central-reference set) from stored `grams.npz`; no engine imports. | Pareto tables |
| `plot_ccp_compare_without_256_checkpoints.py` | Campaign figures for K2/K6 (GN vs grads / CPU, gap per decision). | Report figures |
| `plot_ccp_compare_K10_mnist_without_256_checkpoints.py` | Same for the MNIST trial (+ per-class losses figure). | Report figures |

### 4d · Bandit-toy series (July 26–31)

| File | What it is |
| --- | --- |
| `run_bandit_toy_without_256_checkpoints.py` | K = 2 convex toy: certified MSVRG grid vs MSVRG adaptive bundle, exact plug-in Pareto oracle. |
| `run_bandit_toy_K5_without_256_checkpoints.py` | K = 5 variant (centered-quartic rewards); closed-form reference still holds. |
| `run_bandit_toy_mv_without_256_checkpoints.py` | Mean-variance variant (nonconvex); ground truth = never-timed multistart reference table. |

## 5 · Sanity gates & utilities

| File | What it is |
| --- | --- |
| `sanity_checks_fast.py` | Equivalence gates for the `_fast` set (Gram path ≡ einsum ≤ 1e-12, MSVRG degeneration ≡ original inner loop, pruning bitwise-safe, …). |
| `sanity_checks_ccp.py` | Correctness gates for `ccp_lambda_solver.py` (LP warm ≡ fresh, monotone ascent, K = 2 ≡ exact envelope, …). |
| `sanity_checks_bandit_toy.py`, `sanity_checks_bandit_toy_K5.py`, `sanity_checks_bandit_toy_mv.py` | Pre-flight checks for the corresponding bandit objectives. |
| `run.sh` | Wrapper: project venv + `KMP_DUPLICATE_LIB_OK=TRUE` (torch and IPOPT each ship a libomp on macOS; loading both aborts otherwise). |

## The active line

New work happens on the CCP line of the without-256 track:

```
bundle_fast + ccp_lambda_solver + algorithm_ccp_without_256_checkpoints
    → run_ccp_compare_{K2,K6,K10_mnist}_* → audit_v2 → plot_ccp_compare_*
```

`algorithm.py` / `bundle.py` / `baseline.py` and the July runners stay frozen as
references; their outputs are never overwritten (campaigns write to new homes).

## How to run

```bash
./run.sh run_ccp_compare_K2_without_256_checkpoints.py   # any script in this dir
```

or, manually: use the project venv's python from inside this directory with
`KMP_DUPLICATE_LIB_OK=TRUE`. Outputs land one level up in
`../output/<experiment home>/`. Before touching engine code, run the relevant
sanity gate (`sanity_checks_fast.py`, `sanity_checks_ccp.py`, …) — every check
must print PASS.
