# Branch Overview: `mlp-comparison-results`

This branch adds an MLP comparison notebook and the generated plot outputs for the GN* coverage experiments.

## Main File to Review

### `Mlp_Compare.ipynb`

This is the primary file for this branch.

It contains the experiment workflow for:

- running MLP GN* coverage experiments,
- comparing the uniform discretisation baseline with the adaptive bundle method,
- testing larger model settings such as `K=4, p=15` and `K=5, p=20`,
- comparing settings with similar gradient-evaluation budgets,
- saving the generated result figures.

Start reviewing this file first.

## Newly Added Files

### Notebook

- `Mlp_Compare.ipynb`

### Generated result figures

- `mlp_K=3_p=10_n=20_h=8_err=1e-2(r=9).png`
- `mlp_K=3_p=10_n=20_h=8_err=1e-2(r=26).png`
- `mlp_K=3_p=10_n=20_h=8_err=1e-2(r=85).png`
- `mlp_compare_K4_p15_n20_h8(r=26)_baseline_vs_adaptive.png`
- `mlp_compare_K4_p15_n20_h8(r=26)_adaptive_outer5000_sample2048.png`
- `mlp_compare_K4_p15_n20_h8(r=10)_equal_grad_budget.png`
- `mlp_compare_K5_p20_n20_h8(r=10)_same_algorithm_params.png`

These figures are outputs from the notebook experiments.

## Modified Files

### `experiments.py`

The plot formatting in `_plot_coverage(...)` was adjusted:

- removed the `symlog` x-axis scaling,
- removed hard-coded x-axis tick settings,
- saved figures with `bbox_inches="tight"`.

This makes the generated plots easier to read and closer to the intended comparison figure format.

### `objectives.py`

The missing `make_logreg_strongly_convex(...)` factory was restored.

This function is not the main focus of the MLP comparison notebook, but restoring it keeps the repository consistent with the README and older examples that import:

```python
from objectives import make_logreg_strongly_convex
```

The existing MLP objective factory remains available as:

```python
make_mlp_nonconvex(...)
```

The staged files should mainly be:

- `Mlp_Compare.ipynb`
- the new result PNG files listed above,
- `experiments.py`
- `objectives.py`
- this overview file.

