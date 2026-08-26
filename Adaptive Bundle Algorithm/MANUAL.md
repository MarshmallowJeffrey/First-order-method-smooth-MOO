# Manual: what every file is, and how to reproduce the experiments

Date: July 7, 2026. Companion documents: `EXPERIMENTS.md` (what the
experiments are, results, how to read the figures) and its Chinese
version `EXPERIMENTS_ZH.md`. A Chinese version of this manual is kept at
`MANUAL_ZH.md`; when one file changes, change the other to match.

This manual answers two questions:

1. What is every file and directory in this project for?
2. What exactly do I run to reproduce the results in `output/`?

---

## 1. Project layout

Repository root: `First-order-method-smooth-MOO/`, branch
`mlp-comparison-results`. Everything below is relative to the root
unless stated otherwise. Paths contain spaces — always quote them in a
shell.

### 1.1 Repository root

| Path | What it is |
|---|---|
| `Adaptive Bundle Algorithm/` | The whole project: code, experiment outputs, documentation, reference papers. |
| `Note/Jul_5_note.md` | Detailed record of the July 4 soundness fixes (descent-lemma safeguard, simplex-projection consistency, epsilon-mode honesty, snake grid ordering, corrected logistic-regression smoothness constant), with the mathematical reasoning for each. |
| `Note/Jul_6_note.md` | Detailed record of the July 6 paper-conformance review and fixes (W* distribution, plateau-level definition, activation parameter, docstring corrections), plus the ReLU smoothness investigation. |
| `.venv/` | The project's Python virtual environment (Python 3.11.5, PyTorch, cyipopt/IPOPT, NumPy, SciPy, Matplotlib). All commands in this manual use its interpreter. |

### 1.2 `Adaptive Bundle Algorithm/` — documentation and assets

| Path | What it is |
|---|---|
| `EXPERIMENTS.md` / `EXPERIMENTS_ZH.md` | The experiment analysis report (English / Chinese): definitions, theory-derived expectations, precise protocols, results, parameter reference, figure-reading guide, curve-behaviour explanations, honest-reporting notes. |
| `MANUAL.md` / `MANUAL_ZH.md` | This manual (English / Chinese). |
| `Python_Change.md/PYTHON_CHANGES.md` | Chronological record of code changes made while adapting the original notebook code into the current module layout, including the July 4 soundness fixes. |
| `Python_Change.md/PYTHON_CHANGES_ZH.md` | Chinese translation of the above. |
| `Python_Change.md/PLATEAU_EXPERIMENT_CHANGES.md` | Record of the changes that built the plateau experiment machinery (`detect_plateau`, `experiment_mlp_plateau_comparison`, budget accounting, pairwise plots). |
| `Python_Change.md/PLATEAU_EXPERIMENT_CHANGES_ZH.md` | Chinese translation of the above. |
| `Reference_essay/A_first_order_bundle_method_for_smooth_multi_objective_optimization.pdf` | THE paper. Algorithm 1 (uniform-grid baseline) and Algorithm 2 (adaptive bundle method) implemented here come from it. |
| `Reference_essay/Smooth Tchebycheff Scalarization for Multi-Objective Optimization.pdf` | Related work (Lin et al. 2024), cited by the draft. |
| `Reference_essay/Beyond One-Preference-Fits-All Alignment- Multi-Objective Direct Preference Optimization.pdf` | Related work (MODPO), cited by the draft. |
| `Reference_essay/reference essay.pdf` | Additional reference material. |
| `output/` | All current experiment results. Structure in section 1.4. |

### 1.3 `Adaptive Bundle Algorithm/Original_py/` — the code

One module per responsibility; `run_experiments.py` is the only entry
point you normally need.

| File | What it does |
|---|---|
| `algorithm.py` | The paper's **Algorithm 2** (adaptive bundle method): the outer loop, the multistart max–min λ-search (`_maximise_GN`, IPOPT with SLSQP fallback), the T-map inner steps (Eq. 10, batched, least-index tie-break), the descent-lemma safeguard (adaptive `L_scale` doubling with a RuntimeWarning), epsilon mode, and the GN\* quality metric (`pc_star`, fixed 256-start yardstick). |
| `bundle.py` | The `Bundle` container: every evaluated point with all K per-objective gradients; assembles ∇F_λ at any λ from storage. |
| `baseline.py` | The paper's **Algorithm 1** (uniform discretisation baseline): simplex grid of resolution r, snake ordering with warm starts, fixed-step gradient descent per node, checkpointing, and the snapshot bundle used to score the baseline with the same GN\* metric. Also has an optional certification mode (`node_tol`, default off, unused by all current experiments): per-node ‖∇F_{λ_i}‖² acceptance checks that stop the run once every node is served (see `Note/Jul_8_note.md`). |
| `objectives_torch.py` | The MLP testbed (PyTorch): planted linear-softmax data (paper §5.1.1, W* ~ U[−1,1]), K per-class cross-entropy objectives, selectable activation (`relu`/`tanh`/`softplus`/`identity` — benchmarks use `tanh`), probe-based smoothness estimates, fused joint oracle. |
| `objectives_numpy.py` | NumPy problem generators, including the strongly-convex logistic-regression testbed used only by the verification suite (no experiment driver uses it). |
| `experiments.py` | Experiment drivers and analysis: `experiment_mlp_plateau_comparison` (THE equal-budget head-to-head both sweeps use), `experiment_mlp_gn_coverage` (legacy time-to-target design — not used for headline numbers), `detect_plateau`, and all plotting (`_plot_plateau_pair` draws every figure element described in `EXPERIMENTS.md` §8). |
| `run_experiments.py` | The unified experiment runner and the reproduction entry point: defines both sweeps' configurations, runs them, writes every `summary.json`, `README.md`, trend plot, and `sweep_index.json` under `output/`. |
| `run.sh` | Convenience wrapper: runs any script in this directory with the project venv and `KMP_DUPLICATE_LIB_OK=TRUE` set (see section 2). |

### 1.4 `Adaptive Bundle Algorithm/output/` — the results

```
output/
  README.md                  # how the tree is organised, plot-reading pointers
  plateau/                   # K-sweep at fixed 30k budget + K=6 budget study
    README.md                # cross-configuration analysis, K=6 investigation
    sweep_index.json         # machine-readable index of the 4 main configs
    plateau_ratio_vs_K.png   # trend plot
    K6_budget_study.png      # K=6 best-so-far curves at 3 budgets
    K{3..6}_p6_n30_h4_tanh_r6_B30000/    # the main sweep
    K6_p6_n30_h4_tanh_r6_B{90000,240000}/  # the K=6 budget study
  crossover/                 # width sweep at K=2, n=50k, 2k budget
    README.md, sweep_index.json, crossover_ratio_vs_d.png
    d{642,1794,5634,11522,19458}_h*_tanh_n50000_B2000/
```

Every configuration directory contains exactly four files:

- `gn_vs_grad_evals.png` — raw GN\* vs cumulative gradient evaluations.
- `gn_vs_cpu_time.png` — raw GN\* vs CPU time (log x, equal-budget
  marker). How to read every element: `EXPERIMENTS.md` §8.
- `summary.json` — the complete record: `config` (all parameters),
  `baseline` and `adaptive` blocks (checkpoint histories `cov_history`,
  `best_so_far`, `grad_evals_history`, `cpu_times`, plus health flags
  `L_scale_final`, `inner_cap_hits`), `plateaus` (detector output per
  method), `time_to_target` (final values and the symmetric
  time-to-common-target statistics), `runtime_warnings`.
- `README.md` — the parameters, why they were chosen, results, health
  flags, and per-configuration analysis.

### 1.5 Archives outside the repository

`/Users/shirch/vscode101/.venv/ledger-artifacts/` (deliberately outside
the git repository) holds: `verify_fixes.py` and `prefix_repro.py` (the
verification drivers, section 5), `orig_backup/` (the pre-July-4 code),
`pre_fix_outputs_archive/` (all pre-fix experiment outputs — not
comparable with current results), and `relu_sweep_archive/` (the ReLU
diagnostic sweeps that motivated the switch to tanh).

---

## 2. Environment

- Interpreter: `<repo root>/.venv/bin/python` (Python 3.11.5).
- Required packages are already installed in that venv: PyTorch, cyipopt
  (IPOPT), NumPy, SciPy, Matplotlib.
- **Always set `KMP_DUPLICATE_LIB_OK=TRUE`** when running anything that
  imports both PyTorch and cyipopt on macOS: PyTorch bundles its own
  OpenMP runtime and IPOPT/OpenBLAS pulls in the Homebrew one; without
  the flag the process aborts with "OMP: Error #15".
  `experiments.py` and `run_experiments.py` set it themselves;
  standalone scripts must set it explicitly (or use `run.sh`).
- IPOPT is required for the experiments (`experiment_mlp_plateau_comparison`
  refuses to run without it) so that the λ-search and the metric use the
  intended solver rather than silently falling back to SLSQP.

Quick check that the environment works:

```bash
cd "<repo root>/Adaptive Bundle Algorithm/Original_py"
KMP_DUPLICATE_LIB_OK=TRUE ../../.venv/bin/python -c \
  "import torch, cyipopt; print('environment OK')"
```

---

## 3. Reproducing the experiments

All commands below are run from
`"<repo root>/Adaptive Bundle Algorithm/Original_py"`.
`$PY` means `../../.venv/bin/python`.

### 3.0 Before you run

- **Run serially on an otherwise idle machine.** The CPU-time axis is
  wall-clock; concurrent load distorts it (the gradient axis is immune).
- **Overwrite behaviour:** `summary.json` and `README.md` are
  overwritten in place on a re-run; **PNG files are never overwritten**
  — a re-run adds `_001`, `_002`, … suffixed copies next to the old
  ones. If you want a clean directory, delete the old config directory
  first.
- **Reproducibility:** data and initialisation seeds are fixed (7/8) in
  the sweep definitions, so the gradient-axis curves and all quality
  numbers should reproduce up to floating-point/BLAS-threading noise.
  CPU times and therefore the equal-time ratios are machine-dependent:
  expect the qualitative picture (monotone growth, crossover at small
  d), not the exact numbers 1.3–101.4.

### 3.1 Smoke tests (minutes — run these first)

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY run_experiments.py plateau   --smoke
KMP_DUPLICATE_LIB_OK=TRUE $PY run_experiments.py crossover --smoke
```

Tiny versions of both sweeps (plateau: K=3,4 at a 4k budget). They
write to `output/plateau_smoke/` and `output/crossover_smoke/`, so they
never touch the real results. Use them to confirm the environment
before committing hours to the full sweeps. The smoke directories can
be deleted afterwards.

### 3.2 The plateau sweep (K = 3,4,5,6 at 30k budget)

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY run_experiments.py plateau
```

Writes the four `K*_B30000` directories, `plateau_ratio_vs_K.png`,
`sweep_index.json`, and the sweep `README.md` under `output/plateau/`.
Measured iterative-work time on the reference machine: ~36 minutes
total (baselines are seconds each; the adaptive runs are 442 s, 494 s,
531 s, 639 s for K=3,4,5,6). Real wall time is somewhat higher because
checkpoint metric evaluations (excluded from the reported axes) also
take time.

### 3.3 The K=6 budget study (90k and 240k budgets)

`run_experiments.py` deliberately keeps the main sweep fixed at 30k, so
the two budget-study directories were produced by calling its
`run_one_config` helper directly — that is the function that runs one
configuration AND writes the standard four-file layout (`summary.json`,
the two renamed plots, `README.md`). From `Original_py/`:

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY - <<'EOF'
from pathlib import Path
from run_experiments import run_one_config, PLATEAU_RATIONALE

# Checkpoint cadences as recorded in the committed summary.json files.
for budget, cadence in ((90_000, 3_600), (240_000, 8_000)):
    config = dict(
        K=6, p=6, n=30, hidden_sizes=[4], activation="tanh",
        coarse_resolution=6, n_passes=100_000,
        steps_per_point_per_pass=5,
        max_grad_evals=budget,
        baseline_eval_every_n_grads=cadence,
        adaptive_eval_every_n_grads=cadence,
        max_outer=1_000_000, max_inner=25,
        lambda_max_starts=64, prune_inner=True,
    )
    out = Path(f"../output/plateau/K6_p6_n30_h4_tanh_r6_B{budget}")
    run_one_config(config, out, PLATEAU_RATIONALE)
EOF
```

(Seeds 7/8 are supplied by `run_one_config` itself via its
`DATA_SEED`/`INIT_SEED` constants.) Measured iterative-work times:
90k ≈ 35 min, 240k ≈ 2.2 h (adaptive 2,096 s and 7,880 s; baselines
21 s and 59 s). The combined figure `K6_budget_study.png` is assembled
from the three summary.json files' `best_so_far` curves (30k/90k/240k
adaptive + 240k baseline); any small script that reads those four
curves and plots them on a log-y axis reproduces it.

### 3.4 The crossover sweep (widths 16x16 … 128x128)

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY run_experiments.py crossover
```

Writes the five `d*` directories, `crossover_ratio_vs_d.png`,
`sweep_index.json`, and the sweep `README.md` under `output/crossover/`.
**This is the expensive one**: measured iterative-work time ~4.9 h
total, dominated by the adaptive runs at the large widths (64x64:
2,467 s; 96x96: 4,632 s; 128x128: 9,040 s). With checkpoint metric
evaluations on top (large at big d), plan for the better part of a day
on an idle machine.

### 3.5 A single custom configuration

To run one configuration of your own (different K, width, budget,
activation), reuse `run_one_config` exactly as in section 3.3 with your
own `config` dict and output directory — you get the standard four-file
layout. Every parameter in section 6 of `EXPERIMENTS.md` is a `config`
key. If you also want different seeds (the constants `DATA_SEED`/
`INIT_SEED` are fixed at 7/8), call the underlying driver instead:

```bash
KMP_DUPLICATE_LIB_OK=TRUE $PY - <<'EOF'
from experiments import experiment_mlp_plateau_comparison

result = experiment_mlp_plateau_comparison(
    K=4, p=6, n=30, hidden_sizes=[4], activation="tanh",
    seed=7, init_seed=8,
    coarse_resolution=6,
    steps_per_point_per_pass=5, n_passes=100_000,
    max_grad_evals=30_000,
    baseline_eval_every_n_grads=1_200,
    adaptive_eval_every_n_grads=1_200,
    max_outer=1_000_000, max_inner=25,
    lambda_max_starts=64, prune_inner=True,
    output_dir="../output/my_custom_run",
)
print(result["plateaus"])
EOF
```

The raw driver writes the two plots (under their original
`baseline_vs_ipopt_*.png` names) and returns the result dictionary; it
does NOT write `summary.json` or the per-config `README.md` — those are
`run_one_config`'s additions.

### 3.6 Checking a reproduction against the recorded results

Compare your new `summary.json` against the committed one in the same
directory: `config` must match exactly;
`time_to_target.baseline_final_best_gn` and
`time_to_target.adaptive_final_best_gn` should match to several
significant digits on the same machine (gradient axis); `cpu_times`
will differ across machines. The headline tables live in
`EXPERIMENTS.md` §5 and each sweep `README.md`.

---

## 4. Reading the results

- Start with `EXPERIMENTS.md`: §5 for the results and trends, §8 for
  what every figure element means, §9 for the non-obvious curve
  behaviours (raw curves rising, large-d fluctuation, the 128x128
  dashed line).
- Then the sweep `README.md` (cross-configuration analysis) and the
  per-config `README.md` (parameters and rationale for that run).
- `summary.json` is the machine-readable ground truth for every number
  quoted anywhere.

---

## 5. Verifying the code itself (optional)

Two standalone drivers live OUTSIDE the repository at
`/Users/shirch/vscode101/.venv/ledger-artifacts/`:

```bash
PY="<repo root>/.venv/bin/python"
cd /Users/shirch/vscode101/.venv/ledger-artifacts
KMP_DUPLICATE_LIB_OK=TRUE "$PY" verify_fixes.py   # expect: 10 passed, 0 failed
KMP_DUPLICATE_LIB_OK=TRUE "$PY" prefix_repro.py   # expect: duplicates: 39, pc_history all 1.0
```

`verify_fixes.py` runs 10 checks against the LIVE code (safeguard
convergence, no point duplication, corrected logistic-regression L,
epsilon-mode stop, λ/value consistency, snake-ordering bound).
`prefix_repro.py` runs the ARCHIVED pre-July-4 code and reproduces its
frozen-loop defect (39 duplicated bundle points, quality pinned at 1.0)
— proof that the defect existed and is gone.

---

## 6. Things that are deliberately NOT here

- Old experiment notebooks (`run_plateau*.ipynb`, `mlp_crossover_h*.ipynb`,
  `Mlp_Compare.ipynb`, `mlp_complexity_crossover_experiment.ipynb`) and
  the standalone `gn_sample_ipopt.py` module were deleted; everything
  they did is covered by `run_experiments.py` + `experiments.py`.
- Pre-fix experiment outputs (including the old `output/plateau result/`
  tree) were removed from the repository; they were produced by code
  with a confirmed defect and are archived at
  `ledger-artifacts/pre_fix_outputs_archive/` outside the repo.
- The presentation DOCX files were removed; a future talk should be
  rebuilt from `EXPERIMENTS.md` §5.

---
---

# Part 2 — everything added after July 9 (written Aug 25, 2026)

Part 1 above (sections 1–6, dated July 7) is kept as written and still
accurately describes the ORIGINAL track: the plateau and crossover
sweeps run by `run_experiments.py`. Every later track is documented
here, in the same two-question spirit: what is each piece, and what do
I run to reproduce it.

## ⚠ Aug 25 reorganisation notice (read first)

Later on Aug 25 the user reorganised the whole repository. Every path
in this manual — Part 1 AND Part 2 — refers to the PRE-reorganisation
layout; use these maps to translate:

- **Code**: `Original_py/` is no longer flat. Files moved into
  `Core Engine/` (engines + bundle + CCP solver), `baseline/`,
  `objective/`, `experiment_plot/` (all runners, plots, audits,
  `experiments.py`), `sanity_check/`; only `run.sh` stays at the top.
  The layout was PATCHED the same day (user-approved): each subfolder
  carries a `_layout.py` sys.path bootstrap, every module with sibling
  imports loads it first, output/data anchors were deepened one level,
  and all output-home constants point at the NEW locations. All five
  sanity gates plus the two legacy verifiers pass on the patched tree.
  `./run.sh <script>.py` works again and accepts either a bare script
  name (searched across the subfolders) or a `subfolder/script.py`
  path — the Part 1 §3 and §9 commands below run as written.
- **Results**: `output/` was regrouped and renamed; the old→new mapping
  table and the list of deleted homes (Pareto_front, the old r-sweep
  home, calibration test, several logs/backups) are in
  `output/README.md`. The `*_ZH` documents moved to `Zh/`;
  `EXPERIMENTS(_ZH).md` and `Python_Change.md/` were deleted (git
  history has them).

Section 9's commands run as written (the runners now write/read the
NEW homes); §10's table keeps the pre-reorg names as historical
identifiers — resolve actual locations through `output/README.md`.

Two standing references for this part:

- `CODE_MAP.md` (this directory; until Aug 25 it was
  `Original_py/README.md`) — the complete per-file map of ALL code
  generations, the import-layer diagram, and why `Original_py/` stays
  flat.
- The authoritative record of each experiment is the `README.md` inside
  its `output/` home plus the dated `Note/` file of the session that ran
  it. This manual gives the commands and points there; it does not
  duplicate result numbers.

If `EXPERIMENTS.md` (referenced throughout Part 1) is absent from the
working tree, recover it from git history — it is the only full report
of the July plateau/crossover results.

## 7. Tracks, file generations, and the measurement rule

Filename suffixes encode the code generations (full tables in
`CODE_MAP.md`):

| suffix | generation |
|---|---|
| *(none)* | original reference implementation; checkpoints scored by the external fixed 256-start yardstick |
| `_without_256_checkpoints` | the ACTIVE measurement track (July 8 →): identical trajectories, but checkpoints record the method's own most recent λ-search value |
| `_fast` | July 15 acceleration set: Gram cache, two-tier λ-search with stop-verify, Momentum-SVRG inner loop, delivery-time pruning |
| `_ccp` | Aug 9 λ-solver replacement: multistart convex–concave procedure (CCP) instead of IPOPT |

Standing measurement rule (July 8 onward): ALL experiments live on the
without-256 track. There is no external 256-start yardstick anywhere;
the family's own strict 64-start λ-search is the instrument for
certificates, stops, audits and shared-axis curves. Where audits are
load-bearing they are reported as the monotone lower-bound envelope
(every audit is a lower bound of an NP-hard max and can under-report; a
prefix GN\* is non-increasing). The August campaigns additionally use
the method-symmetric two-instrument audit
`audit_v2 = max(strict-64 IPOPT, heavy CCP)` — see section 9.7.

## 8. Environment additions

- Same venv and `KMP_DUPLICATE_LIB_OK=TRUE` rule as Part 1 §2. The
  convenience wrapper is `Original_py/run.sh`:
  `./run.sh <script.py> [args]` runs any script in that directory with
  the venv interpreter and the flag set.
- `highspy` (HiGHS) is auto-detected by `ccp_lambda_solver.py` for its
  game LP and is strongly preferred (warm-started LP is the CCP inner
  step); without it the solver falls back to `scipy.optimize.linprog`.
  CCP legs do NOT need IPOPT. IPOPT is still required for the `ts*`
  (IPOPT strict-tier) legs and for the IPOPT half of `audit_v2`.
- Any run whose wall-clock lands on a CPU axis must run SERIALLY on an
  otherwise idle machine (checkpoint/audit time is kept off-axis by the
  runners, but the decision time on the CPU axis is real).
- Reproducibility caveat (session-12 finding, July 27): MLP torch runs
  are NOT bit-reproducible in this environment — treat each stored MLP
  trajectory as one realization, verify stored `summary.json` files
  rather than re-running for identity. Bandit numpy runs ARE
  bit-reproducible.

## 9. Reproducing the post-July-9 experiments

Conventions: every runner has `--smoke` (a tiny run into a separate
`*_SMOKE`/`smoke` home — run it first); campaign runners refuse to
overwrite a completed leg unless `--force` is given. All commands are
run from `Original_py/` as `./run.sh <script> [args]`.

### 9.1 July additions on the original engine

| experiment | command | output home |
|---|---|---|
| Certified Pareto fronts, K=2 (July 8) | `./run.sh run_pareto_certified_without_256_checkpoints.py` | `output/Pareto_front/pareto_certified_without_256_checkpoints{,_r20}/` |
| λ-path figure, K=3 (July 9) | `./run.sh run_lambda_path_without_256_checkpoints.py` | `output/lambda_path_K3/` |
| Measurement-variant A/B rerun (K5 plateau + 96×96 crossover) | `./run.sh run_experiments_without_256_checkpoints.py` | `output/without_256_checkpoints/` |
| K=6 reference trial, B=180,180 (July 11) | `./run.sh run_trial_K6_without_256_checkpoints.py` | `output/trial_K6_…_B180180_without_256_checkpoints/` — do NOT move; the fast trials compare against these stored curves |

### 9.2 Fast-engine trials (July 15–16, session 12 probe July 27)

Gate first: `./run.sh sanity_checks_fast.py` — must print 8/8 PASS.

```bash
./run.sh run_trial_K6_fast_without_256_checkpoints.py \
  --tier-mode strict --rel-target 0.1 --max-outer 300 --variant-tag v4_strict_rel0.1
```

Home: `output/fast_method_trials/`. The v1/v2/v3 folders are earlier
flag-sets of the same runner (each folder README records its exact
identity); v4 (the command above) is the honest strict-instrument
probe. `v3`'s plotted cheap-tier meter was later proven dishonest ~2x —
keep the folder, do not quote its curve.

### 9.3 Baseline r-sweep at fixed node_tol + between-node gap (July 20–26)

```bash
./run.sh run_baseline_svrg_r_sweep_without_256_checkpoints.py \
  --r-list 10,12,15,20 --node-tol 0.02 \
  --out-dirname baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/tol0.02 \
  --fast-ref "../output/fast_method_trials/trial_K6_…_v4_strict_rel0.1/summary.json" \
  --save-grams
```

(`--node-tol 0.01` for the tol0.01 home.) Old July-25-style home
`output/baseline_svrg_multi_r_vs_fast_without_256_checkpoints/` is
frozen; the v2 home `…_v2_without_256_checkpoints/` is the comparison
home of record (`original/` inside it is a verbatim copy of the old
one). Gap figures: `./run.sh plot_between_node_gap_without_256_checkpoints.py`.
The v3-stall diagnostic that motivated the strict instrument:
`./run.sh diag_v3_plateau_without_256_checkpoints.py` →
`…_v2_…/diag_v3_plateau/diag.json`.

### 9.4 Fixed-budget K=6 (July 26; protocol 5e)

```bash
./run.sh run_fixed_budget_K6_without_256_checkpoints.py \
  --budget 80912 --rel-target 0.05 --targeting-starts 24 --eval-every 2000
```

Home: `…_v2_…/fixed_budget_B80912/` (`--replot` redraws from stored
data; `--smoke` → `fixed_budget_B600_SMOKE/`). Baseline points on its
figures come from the completed r-sweep runs of 9.3.

### 9.5 PURE fixed-budget protocol (July 27 K=6, July 30 K=2) — the headline protocol

No tolerance parameter exists anywhere in this protocol: shared segment
unit, shared `s`, chain warm start, stop = budget; the ONLY difference
between legs is the next-λ policy. One leg per invocation, serial:

```bash
./run.sh run_pure_budget_K6_without_256_checkpoints.py --run adaptive --s 5 --budget 80912 --targeting-starts 24
```

```bash
./run.sh run_pure_budget_K6_without_256_checkpoints.py --run baseline --r 10 --s 1
```

Recorded legs: adaptive s5; baseline r ∈ {10,12,15,20} at s=5 plus
r10/r15 at s=1. `--backfill-audits` adds strict 64-start prefix audits
to finished baseline legs; `--figure` redraws the campaign figures.
Home: `…_v2_…/pure_budget_B80912/`.

K=2 version with the EXACT 1-D quality meter (no multistart search in
any measurement; `--decision-grid 2001`, `--audit-grid 200001`):

```bash
./run.sh run_pure_budget_K2_without_256_checkpoints.py --run adaptive --s 5 --budget 20000 --targeting-starts 24
```

baseline legs `--run baseline --r 10|20|40|80 --s 5`; the CCP targeting
leg is its own runner (same executor, only the next-λ policy swapped):

```bash
./run.sh run_pure_budget_K2_ccp_without_256_checkpoints.py --s 5 --budget 20000
```

Home: `output/pure_budget_K2_without_256_checkpoints/B20000/`.
Post-hoc ε-Pareto front metrics for the K=6 campaign:
`./run.sh front_metrics_K6_pure_budget_without_256_checkpoints.py`.

### 9.6 SURF bandit toys (July 26 K=2/K=5, July 31 mean-variance)

Gates: `./run.sh sanity_checks_bandit_toy.py` (9/9),
`sanity_checks_bandit_toy_K5.py` (9/9), `sanity_checks_bandit_toy_mv.py`.

```bash
./run.sh run_bandit_toy_without_256_checkpoints.py --epsilon 1e-2 --eval-every 0
```

`--eval-every 0` = per-segment recording, the precise-readout mode
(session 13). The eps1e-3/1e-4 folders still carry the coarse July-26
cadence; re-run with `--eval-every 0` before quoting their
first-crossing numbers. K=5: `run_bandit_toy_K5_without_256_checkpoints.py
--epsilon …`. Mean-variance (nonconvex):
`run_bandit_toy_mv_without_256_checkpoints.py --epsilon …`
(`--gamma-scan` picks γ, `--rebuild-reference` rebuilds the untimed
multistart ground-truth table). Homes:
`output/bandit_toy_{surf,K5,mv}_without_256_checkpoints/eps*/`.

### 9.7 CCP campaign (Aug 8–10): Experiment 1, the MNIST K=10 trial, studies A/B

Gate first: `./run.sh sanity_checks_ccp.py` — all checks must PASS.

- **Study A — seed-sampler ablation** (decides `CCPConfig.seed_sampler`):
  `./run.sh run_ccp_smoke_sampler_without_256_checkpoints.py --reps 50`
  → `output/ccp_smoke_sampler/`.
- **Study B — controlled λ-solver benchmark** (2a paired-start polish +
  2b 60-second fixed-time race, on the frozen Gram snapshots in
  `output/ccp_compare_…/lambda_solver_bench/snapshots/`):
  `./run.sh run_lambda_solver_bench_without_256_checkpoints.py --T 60 --batches 20`.
- **Experiment 1 (K=2 and K=6)** — each command runs ALL its legs
  serially into `output/ccp_compare_without_256_checkpoints/{K2_B20000,K6_B80912}/`:

```bash
./run.sh run_ccp_compare_K2_without_256_checkpoints.py --smoke
```

```bash
./run.sh run_ccp_compare_K2_without_256_checkpoints.py
```

  (same pattern with `run_ccp_compare_K6_without_256_checkpoints.py`).
- **Audits:** `./run.sh audit_v2_K6_without_256_checkpoints.py`
  (`--quick` for a 3-stack spot check) writes `audit_v2.json` into every
  K6 leg — the two-instrument quality meter of section 7.
- **Figures:** `./run.sh plot_ccp_compare_without_256_checkpoints.py --which K2`
  (and `--which K6`).
- **MNIST K=10 trial** (the report's "Experiment 2"; older in-repo
  documents numbered it 3 — same experiment). Reads the idx files in
  `data/mnist/`; two adaptive legs only (CCP vs IPOPT ts24):

```bash
./run.sh run_ccp_compare_K10_mnist_without_256_checkpoints.py --budget 55000 --per-class 1000 --batch 1024 --s 5 --ts 24
```

  then `./run.sh plot_ccp_compare_K10_mnist_without_256_checkpoints.py`.

### 9.8 Experiment 4 — K=2 MNIST digit-pair campaign (Aug 13)

Pair selection first (ranks 5 candidate pairs by conflict):

```bash
./run.sh run_conflict_smoke_K2_mnist_pairs_without_256_checkpoints.py
```

then one campaign per chosen pair (baseline r ∈ {10,20,40} + adaptive
CCP, B = 20,000, exact 1-D meter, train+test fronts):

```bash
./run.sh run_pure_budget_K2_mnist_pair_without_256_checkpoints.py --pair 3 5
```

(`--pair 7 9` for the second pair; `--smoke` → `SMOKE/pair_3v5_B800/`.)
Figures: `./run.sh plot_K2_mnist_pair_without_256_checkpoints.py`.
Home: `output/K2_mnist_pair_without_256_checkpoints/`.

## 10. Output map — post-July homes at a glance

| output home | experiment (record) |
|---|---|
| `Pareto_front/` | certified Pareto fronts K=2, main r=10 + r=20 re-run (its READMEs + `FINDINGS_ZH.md`) |
| `lambda_path_K3/` | K=3 λ-path figure (its README + quarters analysis) |
| `without_256_checkpoints/` | measurement-variant A/B rerun (its `FINDINGS.md`) |
| `trial_K6_…_B180180_…/` | K=6 reference trial (frozen reference curves) |
| `calibration_speed_test_B2000/` | pre-run speed calibration for the trial — not an experiment result |
| `fast_method_trials/` | fast-engine series v1/v2/v3/v4 + stopped cert attempt |
| `baseline_svrg_multi_r_vs_fast_without_256_checkpoints/` | July-25 r-sweep home (frozen figures) |
| `baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/` | comparison home of record: `original/`, `tol0.02/`, `tol0.01/`, `adaptive_extended/`, `diag_v3_plateau/`, `fixed_budget_B80912/`, `pure_budget_B80912/` (+SMOKEs) |
| `bandit_toy_surf_…/`, `bandit_toy_K5_…/`, `bandit_toy_mv_…/` | SURF bandit toys K=2 / K=5 / mean-variance, eps rungs + smoke |
| `pure_budget_K2_…/B20000/` | K=2 pure fixed budget with the exact 1-D meter (`REPORT_ZH.md`) |
| `ccp_compare_…/K2_B20000/`, `…/K6_B80912/` | Experiment 1 (CCP vs IPOPT vs grids) |
| `ccp_compare_…/K10_mnist10k_B55000/` | MNIST K=10 trial (report "Experiment 2", earlier "3") |
| `ccp_compare_…/lambda_solver_bench/` | Study B: controlled λ-solver benchmark |
| `ccp_smoke_sampler/` | Study A: Exp(1) vs Sobol seed-sampler ablation |
| `K2_mnist_pair_…/` | Experiment 4: digit-pair conflict smoke + pair_3v5 / pair_7v9 campaigns |

Per-leg file conventions: `summary.json` (full curves + parameters +
health flags) and `grams.npz` (delivered points' Gram matrices) in every
leg; `thetas.npz` (parameter vectors) only in the MNIST-pair legs;
`raw_histories.npz` in bandit runs; `audit_v2.json` in the audited K6 /
K10 legs; `campaign_manifest.json` + `*.log` at campaign level.

## 11. Verification gates — current full list

| gate | expectation |
|---|---|
| `./run.sh sanity_checks_fast.py` | 8/8 PASS (Gram path ≡ einsum, MSVRG degeneration bit-identical, pruning bitwise-safe, …) |
| `./run.sh sanity_checks_ccp.py` | all PASS (LP warm ≡ fresh, monotone ascent, K=2 ≡ exact envelope, …) |
| `./run.sh sanity_checks_bandit_toy.py` | 9/9 |
| `./run.sh sanity_checks_bandit_toy_K5.py` | 9/9 |
| `./run.sh sanity_checks_bandit_toy_mv.py` | all PASS |
| legacy `verify_fixes.py` / `prefix_repro.py` (outside the repo, Part 1 §5) | 10 passed / duplicates: 39 |

Run the relevant gate before touching any engine file; every check must
print PASS. Stored-result spot checks (exact JSON expectations) live in
the session ledger outside the repository.
