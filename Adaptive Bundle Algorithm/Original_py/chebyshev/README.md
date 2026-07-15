# Weighted Chebyshev Module

This folder contains the Chebyshev-side algorithms used to compare against the
existing uniform discretisation baseline and adaptive bundle method.

The default experiment matches the classmate crossover configuration:
`K=2`, `p=20`, `n=50000`, `r=10`, `max_grad_evals=2000`,
`max_inner=5`, `lambda_max_starts=8`, `activation="tanh"`, and a
two-hidden-layer MLP with `hidden_sizes=[96, 96]`.

Use `--classmate-crossover-width 64` to switch to the `[64, 64]` case from
the same sweep.

## Files

- `weighted.py`
  - `solve_weighted_chebyshev_dual`: solves the beta subproblem
    \[
      \min_{\beta \ge 0,\ \sum_k \lambda_k \beta_k = 1}
      \left\|\sum_k \beta_k \hat g_k\right\|^2.
    \]
  - `weighted_chebyshev_exact_gd`: the fixed-lambda inner loop.
  - `chebyshev_adaptive`: the outer loop that selects the worst lambda by
    GN*, warm-starts from the nearest anchor, runs the Chebyshev inner loop,
    and appends the new point to the bundle.

## Default Mode

The command-line default uses `--cheb-mode hybrid`, which keeps the
Chebyshev direction as a candidate but aligns accepted points with the
outer GN* metric.

To run the classmate crossover `[96, 96]` case:

```bash
python -m chebyshev.experiment
```

To run the `[64, 64]` case:

```bash
python -m chebyshev.experiment --classmate-crossover-width 64
```

For GN*-aligned experiments, use:

```python
chebyshev_adaptive(
    ...,
    line_search="gn_decrease",
    append_non_improving=False,
    fallback_update="scalarized_gd",
    use_active_best_start=True,
)
```

This keeps the Chebyshev direction as the first proposal, but each accepted
step must decrease the active-lambda scalarized gradient norm used by GN*.
If the Chebyshev direction cannot pass that check, the inner loop tries a
scalarized-gradient fallback step under the same GN-decrease condition.  The
outer loop also starts from the current bundle point with the best active
lambda score, rather than only from the nearest anchor.

The stronger experimental mode used in the comparison script is `hybrid`:

```python
chebyshev_adaptive(
    ...,
    candidate_anchor_count=2,
    normalization_powers=(1.0, 0.75, 0.5),
    line_search="gn_decrease",
    append_non_improving=False,
    fallback_update="scalarized_gd",
    use_active_best_start=True,
    use_tmap_safeguard=True,
)
```

This is the "afternoon" version: each outer iteration tests several
Chebyshev candidates and a temporary adaptive-bundle `T_map` chain at the
same active preference, then appends the candidate with the smallest
active-lambda gradient norm.
