# Plateau Experiment Implementation

Date: July 1, 2026

## Goal

The new experiment compares the steady-state GN* behavior of three methods:

1. Uniform-discretization baseline.
2. Sample-based adaptive bundle method.
3. IPOPT-based adaptive bundle method.

The primary comparison axis is total gradient evaluations. Lower GN* and a
lower plateau level are better.

## Files Changed

### `algorithm.py`

- Added `ipopt_available()` so experiments can verify the backend before
  starting expensive runs.
- Expanded IPOPT import handling to recognize both a missing package and a
  broken binary/library installation.
- Added `lambda_solver` to select `"ipopt"` or `"slsqp"` explicitly.
- Added `require_ipopt`. When enabled, the run fails immediately instead of
  silently falling back to SLSQP.
- Added `max_grad_evals` as an optional hard gradient-evaluation budget.
- The final state is checkpointed when fewer than `K` gradient evaluations
  remain, including when the budget is not divisible by `K`.
- Added the actual selector/backend and gradient budget to the returned result
  dictionary.
- Added early validation for invalid selector, solver, strict-IPOPT, and budget
  combinations.

### `baseline.py`

- Added `max_grad_evals` as an optional hard gradient-evaluation budget.
- The gradient-descent loop stops before a step would exceed the budget.
- A final checkpoint is recorded even when the budget is reached in the middle
  of a simplex-grid pass.
- Added the configured budget to the returned result dictionary.

### `experiments.py`

- Added `detect_plateau(...)`.
- Added pairwise plateau plotting against total gradient evaluations.
- Added `experiment_mlp_plateau_comparison(...)` to run all three methods on
  the same generated MLP problem, initial point, smoothness estimates, and
  fused oracle.
- Disabled baseline-target early stopping for the two adaptive runs by passing
  `target_cov=None`.
- Added strict IPOPT preflight validation before any expensive method is run.
- Added three output plots:
  - `baseline_vs_sample.png`
  - `baseline_vs_ipopt.png`
  - `sample_vs_ipopt.png`
- Added plateau ratios and a console summary table.

## Plateau Definition

Default detector parameters are:

```python
plateau_window = 5
plateau_relative_improvement_tol = 0.05
plateau_consecutive_windows = 2
```

The detector first constructs a best-so-far GN* curve:

```python
best_so_far = np.minimum.accumulate(cov_history)
```

For each candidate onset, it examines two adjacent, non-overlapping blocks of
five checkpoints. Each block must improve by less than 5%. The total
best-so-far improvement from that candidate onset to the end of the run must
also be less than 5%, which prevents an early temporary flat region from being
reported as the final plateau.

The returned dictionary contains:

```python
{
    "found": bool,
    "onset_index": int | None,
    "onset_grad_evals": int | None,
    "onset_cpu_time": float | None,
    "plateau_level": float | None,
}
```

`plateau_level` is the median raw GN* value from the detected onset to the end
of the run.

## Experiment Outputs

`experiment_mlp_plateau_comparison(...)` returns:

```python
{
    "baseline": ...,
    "sample_adaptive": ...,
    "ipopt_adaptive": ...,
    "plateaus": ...,
    "plateau_ratios": ...,
    "plots": ...,
    "detector_options": ...,
    "max_grad_evals": ...,
}
```

The reported ratios are:

```text
baseline plateau / sample-adaptive plateau
baseline plateau / IPOPT-adaptive plateau
sample-adaptive plateau / IPOPT-adaptive plateau
```

A ratio greater than one means the numerator method has the higher, worse
plateau.

## Default Run Configuration

The new experiment defaults to:

```python
K = 3
p = 4
n = 60
h = 8
max_grad_evals = 30000
coarse_resolution = 10
steps_per_point_per_pass = 10
adaptive_eval_every_n_grads = 2000
```

The baseline checkpoints after every pass by default. Both adaptive methods
use the same gradient budget and checkpoint interval. The sample-based method
uses 512 random simplex samples per outer selection by default, while the IPOPT
method uses the structured multi-start optimizer.

## How to Run

Call the new experiment function explicitly:

```python
from experiments import experiment_mlp_plateau_comparison

result = experiment_mlp_plateau_comparison(
    output_dir="plateau_results",
)
```

The existing direct script entry point was intentionally left unchanged so the
new, more expensive three-method experiment does not start accidentally.

## Current IPOPT Environment Status

The code correctly detects the current IPOPT backend as unavailable because the
installed `cyipopt` extension fails to load with a missing IPOPT symbol:

```text
symbol not found in flat namespace '_AddIpoptIntOption'
```

Therefore, the new plateau experiment currently fails immediately with a clear
error instead of labeling an SLSQP fallback as IPOPT. The local IPOPT/cyipopt
installation must be repaired before the full three-method experiment can run.

## Verification Performed

- All changed Python files pass bytecode compilation.
- Plateau detection finds a known synthetic plateau at the expected index.
- A continuously improving synthetic curve correctly returns `found=False`.
- Pairwise plateau plotting produces a valid image.
- Baseline and sampled adaptive test runs stop at their requested gradient
  budgets.
- Strict IPOPT mode rejects the broken local backend without falling back to
  SLSQP.
