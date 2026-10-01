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
| `baseline.py` | The paper's **Algorithm 1** (uniform discretisation baseline): simplex grid of resolution r, snake ordering with warm starts, fixed-step gradient descent per node, checkpointing, and the snapshot bundle used to score the baseline with the same GN\* metric. |
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
