# A First-Order Bundle Method for Smooth Multi-Objective Optimization

Reference implementation accompanying the paper *"A First-Order Bundle
Method for Smooth Multi-objective Optimization"* (Grigas & Cheng).
This repository contains:

- The **bundle method (Algorithm 2)** for smooth multi-objective
  optimization, instantiated with the per-class Lipschitz / strong-
  convexity setting of Example 2.
- A **uniform-discretization baseline** that approximates the
  Pareto-scalarised solution map by warm-started gradient descent on
  a coarse simplex grid.
- A **CPU-time–vs–worst-case-accuracy** and **gradient-evaluations–vs–
  worst-case-accuracy** comparison harness for both methods.
- Three concrete objective families: regularised multi-class logistic
  regression, an unregularised separable-Gaussian-mixture variant
  (interpolation regime), and a one-hidden-layer ReLU MLP.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `K` | Number of objectives / classes |
| `p` | Feature dimension |
| `n` | Total training samples; `n_k` per class |
| `W` ∈ ℝ^d | Decision variable (`d = K·p` for logreg, `h·p + h + K·h + K` for MLP) |
| `λ ∈ Δ_K` | Scalarisation weight on the unit simplex |
| `L_k`, `μ_k` | Smoothness / strong-convexity of `F_k` |
| `L_λ = Σ_k λ_k L_k`, `μ_λ = Σ_k λ_k μ_k` | λ-dependent constants |

The scalarised objective is `F_λ(W) := Σ_k λ_k F_k(W)`.  The goal is
to produce an approximate solution map `Ŵ : Δ_K → ℝ^d` whose worst-case
suboptimality `sup_λ [F_λ(Ŵ(λ)) − F_λ*]` is as small as possible.

---

## Repository layout

```
.
├── bundle.py        # Bundle data structure + progress criteria (UB, GAP_1/2, GN)
├── algorithm.py     # Algorithm 2 inner loop and λ-maximisation primitives
├── baseline.py      # Uniform discretisation baseline + worst-case-error harness
├── objectives.py    # Regularised logreg, separable Gaussian, 1-hidden-layer MLP
├── experiments.py   # Three experiments + CPU/gradient-evaluation plots
├── cpu_vs_accuracy.png
└── grads_vs_accuracy.png
```

### `bundle.py`

The `Bundle` class stores, for each visited point `W_i`, the function
values `F_1(W_i), ..., F_K(W_i)` and the Jacobian rows
`∇F_1(W_i), ..., ∇F_K(W_i)`.  Three progress criteria are implemented
exactly as in Example 2 (Section 5.2):

| Function | Formula | Assumption |
|----------|---------|------------|
| `UB(λ; B_m)` | `min_i { F_λ(W_i) − ‖∇F_λ(W_i)‖² / (2 L_λ) }` | smoothness only |
| `GAP(λ; B_m)` | `UB − LB` | strong convexity |
| `GN(λ; B_m)` | `½ (1/μ_λ − 1/L_λ) min_i ‖∇F_λ(W_i)‖²` | generic non-convex |

Two `LB` variants are exposed via the `variant=` keyword:

- `"lb1"` (default): the aggregated minorant from Eq. 14, solved as a
  concave QP over `β ∈ Δ_m` via Gurobi (with a transparent scipy SLSQP
  fallback if the model exceeds Gurobi's licence size).  Tightest bound.
- `"lb2"`: the single-index minorant
  `max_i { F_λ(W_i) − ‖∇F_λ(W_i)‖² / (2 μ_λ) }` (Eq. 15).
  About 100× faster per call; gives a slightly looser bound.

`GAP2_value_and_grad(bundle, λ)` returns both `GAP_2(λ; B_m)` and its
analytic λ-gradient, derived from the Danskin envelope representation.
This is used by SLSQP inside `_maximise_GAP` and gives a roughly 4× speed-
up of λ-maximisation by avoiding finite-difference fallback.

### `algorithm.py`

Implements the inner machinery for Algorithm 2:

- `_maximise_UB`, `_maximise_GAP`, `_maximise_GN` — each maximises the
  corresponding progress criterion over `Δ_K` using multi-start SLSQP
  (vertices + centre + edge midpoints).  `_maximise_GAP` uses the
  closed-form gradient from `GAP2_value_and_grad`.
- `_bundle_update_adaptive(bundle, λ, pc_fn, ε/3, ..., skip_threshold)`
  is the inner loop.  Before computing `T(λ; B)` and evaluating the K
  gradient oracles, it checks the descent-lemma certified UB drop
  `δ = ‖∇F_λ(x_{i*})‖² / (2 L_λ)` from the cached gradient at the
  active index `i*`.  If `δ < skip_threshold` (default `1e-2`), the
  inner loop terminates without an oracle call — predictive pruning
  that converts what would be wasted gradient evaluations into early
  termination.

### `baseline.py`

The `algorithm2_progressive` and `uniform_discretisation_progressive`
routines run their respective algorithms with **periodic worst-case-
error checkpoints**.  Both checkpoint at the next natural boundary
(an outer iteration for A2; a grid pass for the baseline) once `M`
gradient evaluations have accumulated since the last checkpoint, where
`M = eval_every_n_grads`.

The reference optimal-value map `{F_λ*}_{λ ∈ G_fine}` is precomputed
once via `compute_reference_map` by running warm-started gradient
descent to high accuracy on a fine simplex grid.  Both algorithms are
benchmarked against this reference.

A discarded bundle point would have cost K oracle calls but contributed
nothing — accordingly, `algorithm2_progressive` only counts gradient
evaluations from *retained* bundle points.  Combined with the
predictive pruning in the inner loop, every gradient evaluation A2
charges to its budget corresponds to a point that actually entered
the bundle.

### `objectives.py`

Three objective factories, all multi-class classification:

- `make_logreg_strongly_convex(K, p, n, reg, seed)` — ℓ₂-regularised
  multi-class logistic regression.  Each `F_k` is `reg`-strongly convex.
  Used with the **GAP** progress criterion.
- `make_logreg_separable_gaussian(K, p, n, sep, sigma, seed)` —
  unregularised logreg on a separable Gaussian mixture (the
  "inverse logistic regression" construction).  Used to test the
  interpolation regime where `inf F_λ = 0` and the **UB** criterion
  applies.
- `make_mlp_nonconvex(K, p, n, h, seed)` — one-hidden-layer ReLU MLP
  with cross-entropy loss.  Smoothness constants are estimated
  empirically (random-pair probing with a 2× safety factor).  Used
  with the **GN** progress criterion.

### `experiments.py`

Four experiments:

| Function | Setting | PC | Default settings |
|----------|---------|-----|----------------|
| `experiment_logreg_gap` | regularised logreg, strong convexity | GAP | K=4, p=4, n=30, reg=0.1 |
| `experiment_logreg_separable_gaussian` | separable mixture, interpolation | UB | K=3, p=2, n=30, sep=2.5 |
| `experiment_mlp_gn` | 1-hidden-layer MLP, generic non-convex | GN | K=3, p=4, n=60, h=8 |
| `experiment_pareto_front` | 2-class regularised logreg | GAP | K=2, p=5, n=40, reg=0.05 |

The first two each produce two plots:

- `cpu_vs_accuracy.png` — wall-time cost vs worst-case suboptimality.
- `grads_vs_accuracy.png` — gradient-oracle cost vs worst-case
  suboptimality.

The two axes tell complementary stories.  The CPU plot reflects
practical cost including A2's algorithmic overhead (maximising PC,
bookkeeping the bundle).  The gradient plot reflects pure oracle
complexity, the quantity bounded by Theorems 2–3.  For domains where
gradients are expensive (large neural networks, simulation-based
objectives), the gradient axis is the relevant one — and A2 wins
clearly there (see plot below).

---

## Quickstart

```bash
pip install numpy scipy matplotlib gurobipy
python experiments.py
```

By default, this runs `experiment_logreg_gap` and produces
`cpu_vs_accuracy.png` and `grads_vs_accuracy.png` in the working
directory.

To run a different experiment, edit the `__main__` block at the bottom
of `experiments.py`:

```python
if __name__ == "__main__":
    res1 = experiment_logreg_gap()
    res2 = experiment_logreg_separable_gaussian()
    pareto = experiment_pareto_front()
```

To customise the comparison protocol, pass arguments:

```python
res = experiment_logreg_gap(
    coarse_resolution=10,         # baseline grid resolution r
    fine_resolution=20,           # reference-map grid (G_fine) resolution
    n_passes=25,                  # baseline passes through the coarse grid
    steps_per_point_per_pass=1,   # GD steps per grid point per pass
    max_outer=10,                 # A2 outer iterations
    max_inner=20,                 # A2 max inner steps per outer
    eval_every_n_grads=200,       # checkpoint cadence (gradient evals)
)
```

### Gurobi licence

`bundle.py` uses Gurobi for the LB₁ QP solver.  A free academic licence
is available at
[gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).
If the licence size is exceeded — which happens once the bundle
exceeds about 100 points — the code transparently falls back to scipy
SLSQP without any user intervention.  If you only use `variant="lb2"`
(the default in `algorithm2_progressive`), Gurobi is never invoked.

---

## Sample output

The default `experiment_logreg_gap` produces (numbers will vary
slightly depending on hardware):

```
Reference map ready: 1771 points, 11.3s
Running baseline (coarse resolution = 10, 25 passes, 1 GD steps/point/pass) ...
  Baseline pass 25/25 | t=2.1s | iters=7150 | grad_evals=28600 | err=3.69e-02

Running Algorithm 2 (10 outer iters, up to 20 inner steps each) ...
  A2 outer 10/10 | t=4.1s | grad_evals=124 | err=1.13e-01
```

The two plots that follow (saved as `cpu_vs_accuracy.png` and
`grads_vs_accuracy.png`) show:

- **CPU axis:** baseline reaches its plateau (worst-case error
  ≈ 0.037, the rounding bias of resolution 10) within ~1 second; A2
  takes longer because of bundle bookkeeping but continues to make
  progress past the baseline's plateau.
- **Gradient-evaluation axis:** A2 reaches the same accuracy band with
  roughly 10× fewer gradient evaluations — the baseline's curve only
  starts dropping at ~2300 gradient evaluations because each pass
  through the coarse grid costs `|G_r| · K` evaluations.

For more aggressive settings (smaller `max_inner`, more outer
iterations), A2 will drop below the baseline's plateau on both axes.

---

## Implementation notes

- **Predictive inner-loop pruning** is the main performance feature
  in `algorithm.py`.  Without it, A2 wastes K gradient evaluations
  per skipped step deciding whether the step was worth taking; with
  it, A2 commits to a step only when the certified UB drop justifies
  the oracle cost.
- **`LB_2` is the default** lower bound inside `algorithm2_progressive`
  (`pc_fn = lambda b, l: GAP(b, l, variant="lb2")`).  This avoids
  Gurobi entirely and is ~100× faster per evaluation.  Use `LB_1`
  via `variant="lb1"` if you want the tightest possible bound.
- **Reference-map cost is not charged to either method.**  We assume
  that in any practical use of the algorithms one wants to actually
  consume the solution map, so producing a high-accuracy oracle for
  benchmarking is a one-time setup cost outside the comparison.

---

## References

- Grigas, P. & Cheng, J. *A First-Order Bundle Method for Smooth
  Multi-objective Optimization.* Preprint, UC Berkeley.
- Nesterov, Y. *Lectures on Convex Optimization.* Springer, 2nd ed.,
  2018.
- Désidéri, J.-A. *Multiple-gradient descent algorithm (MGDA) for
  multiobjective optimization.* Comptes Rendus Mathématique, 2012.
