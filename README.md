# First-Order Bundle Method for Smooth Multi-Objective Optimization

Implementation of **Algorithm 2** (Simple Adaptive Algorithm v2) from *"A First-Order Bundle Method for Smooth Multi-objective Optimization"* (Grigas & Cheng), following the concrete instantiation in **Example 2** (Section 5.2) with λ-dependent smoothness constants.

---

## Notation

All objectives use multi-class classification with the following notation:

| Symbol | Meaning |
|--------|---------|
| K | Number of classes |
| p | Feature dimension |
| n | Total number of training samples |
| n_i | Number of samples in class i,  n_i = \|{j : y_j = i}\| |
| w^i ∈ R^p | Weight vector for class i |
| W = [w^1, …, w^K] | Flattened decision variable in R^{Kp} |
| (y_j, x_j) | Labelled sample with y_j ∈ [K] and x_j ∈ R^p |
| λ ∈ Δ_K | Scalarisation weight vector on the unit simplex |
| L_i, µ_i | Smoothness and strong-convexity constants for F_i |
| Lλ = Σ_k λ_k L_k | λ-dependent smoothness constant |
| µλ = Σ_k λ_k µ_k | λ-dependent strong-convexity constant |

Conditional probability model:

    P(Y = i | X = x; W) = exp(⟨w^i, x⟩) / Σ_{l=1}^{K} exp(⟨w^l, x⟩)

Per-class MOO objective:

    F_i(W) = (1/n_i) Σ_{j: y_j=i} { −⟨w^i, x_j⟩ + log Σ_{l=1}^K exp(⟨w^l, x_j⟩) }

---

## File Overview

### `bundle.py` — Bundle Data Structure & Progress Criteria

Implements the bundle B_m from Section 3 and the three progress criteria from Example 2 (Section 5.2).

**Bundle.** Stores, for each visited point W_i, the function values F_1(W_i), …, F_K(W_i) and Jacobian rows ∇F_1(W_i), …, ∇F_K(W_i).

**Progress criteria** (all use λ-dependent constants Lλ, µλ):

| Function | Formula | Assumption |
|----------|---------|------------|
| UB(λ; Bm) | min_i { Fλ(W_i) − (1/2Lλ) ‖∇Fλ(W_i)‖² } | Any (smoothness only) |
| GAP(λ; Bm) | UB(λ; Bm) − LB(λ; Bm) | Strong convexity |
| GN(λ; Bm) | (1/2)(1/µλ − 1/Lλ) min_i ‖∇Fλ(W_i)‖² | Generic non-convex |

**Two LB variants** are supported via a `variant` keyword argument:

- **`"lb1"` (default):** LB₁ from Eq. 14 — aggregated strongly convex minorants, solved as a concave QP over β ∈ Δ_m using Gurobi. Tightest bound; O(m²) per call.
- **`"lb2"`:** LB₂ from the GAP₂ definition on page 12 — best single-index minorant, equivalent to restricting β to vertices of Δ_m. Computed as max_i { Fλ(W_i) − (1/(2µλ))‖∇Fλ(W_i)‖² }. O(m·K·d) per call, ~100× faster in practice but gives a looser bound.

Mathematical relationship: LB₂ ≤ LB₁ always, hence GAP₂ ≥ GAP₁. Call `LB(bundle, lam)` or `GAP(bundle, lam)` for default LB₁ behaviour; pass `variant="lb2"` for the fast version.

**Gurobi with scipy fallback.** The LB₁ QP is solved via Gurobi using a shared environment (initialised once, reused across all calls). If the Gurobi license can't handle the model size, or if Gurobi is unavailable, the code transparently falls back to scipy's SLSQP solver — no manual intervention needed.

**Mapping T(λ; Bm)** (Eq. 13): one gradient-descent step from the best bundle point, used to generate the next iterate in the inner loop.


### `algorithm.py` — Algorithm 2 (Simple Adaptive Algorithm v2)

Implements the outer loop from Section 3.1 with two improvements over the generic version.

**1. PC-specific λ maximisation.** Instead of evaluating PC on a simplex grid, the code exploits each criterion's structure:

| PC | Structure in λ | Solver |
|----|----------------|--------|
| UB | Concave (pointwise min of concave quadratics, Prop. 6) | SLSQP with Danskin subgradient |
| GAP₁ | Difference-of-concave / DC (Prop. 6) | Multi-start SLSQP (vertices + uniform + edge midpoints) |
| GN | Non-concave, piecewise-rational | Multi-start SLSQP |

For UB, the subgradient is computed via Danskin's theorem: ∇UB(λ) = ∇u_{i*}(λ) where i\* achieves the minimum.

**2. Adaptive inner-loop stopping at ε/3.** From the proof of Theorem 1 (Appendix B.1), the convergence argument requires PC(λ_t; B_{t+1}) ≤ ε/3 after the inner update. Instead of precomputing a theoretical upper bound M_t on the number of inner iterations (which can be loose by orders of magnitude), the code checks the actual PC value after each step and stops as soon as the threshold is met:

```
for step = 1, 2, … :
    W_new = T(λ_t; B)
    add W_new to bundle B
    if PC(λ_t; B) ≤ ε/3:
        break
```

**Outer loop:**
```
Repeat:
  1. λ_t = argmax_{λ ∈ Δ_K} PC(λ; B_t)     [PC-specific solver]
  2. Run inner steps at λ_t until PC(λ_t; B) ≤ ε/3
Until max_λ PC(λ; B_t) ≤ ε
```

**Return dict** from `algorithm2(...)` contains:
- `bundle` — final Bundle object
- `pc_history` — list of PC* at each outer iteration
- `lam_history` — list of λ_t chosen at each iteration
- `inner_steps_history` — list of inner-loop step counts per outer iteration (used by the LaTeX table writer)
- `oracle_calls` — total number of gradient oracle evaluations
- `outer_iters` — number of outer iterations executed


### `objectives.py` — Multi-Class Classification Objectives

Three objective families, all based on multi-class classification.

**1. `make_logreg_strongly_convex`** — ℓ₂-regularised multi-class logistic regression:

    F_i(W) = (1/n_i) Σ_{j: y_j=i} { −⟨w^i, x_j⟩ + log Σ_l exp(⟨w^l, x_j⟩) } + (reg/2) ‖W‖²

Each F_i is reg-strongly convex (µ_i = reg). Used with the **GAP** progress criterion. Smoothness bound: L_i ≤ ‖X‖²_op / (4 n_i) + reg, from the softmax Hessian spectral norm.

**2. `make_logreg_standard`** — Standard multi-class logistic regression (no regulariser):

    F_i(W) = (1/n_i) Σ_{j: y_j=i} { −⟨w^i, x_j⟩ + log Σ_l exp(⟨w^l, x_j⟩) }

When `separable=True`, the data is generated so a perfect linear classifier exists (in the limit ‖W‖ → ∞). A tiny reg = 1e-4 is added solely to give a finite µ_i for the algorithm's inner-loop computation.  Used with the **UB** progress criterion.

*Note on the interpolation assumption.* Standard logistic regression does not rigorously satisfy Assumption 4.1 because F_i(W) > 0 for all finite W — the zero-loss infimum is only approached as ‖W‖ → ∞. The UB criterion still works well empirically but lacks the full theoretical guarantee of Proposition 2.

**3. `make_mlp_nonconvex`** — Single-hidden-layer neural network:

    Architecture:  x_j → σ(W_1 x_j + b_1) → W_2 a_j + b_2 → softmax
    (σ = ReLU; θ = (W_1, b_1, W_2, b_2); d = h·p + h + K·h + K)

    F_i(θ) = (1/n_i) Σ_{j: y_j=i} { −z_j^{(i)} + log Σ_l exp(z_j^{(l)}) }

Non-convex due to the composition W_2 · σ(W_1 x + b_1). Gradients are computed via backpropagation. Smoothness constants are estimated empirically by probing ‖∇F_i(θ₁) − ∇F_i(θ₂)‖ / ‖θ₁ − θ₂‖ at random pairs, with a 2× safety factor. Used with the **GN** progress criterion.


### `experiments.py` — Numerical Experiments

Four experiments + plotting + LaTeX table generation.

| Exp | Objective | PC | Default K, p, n, ε |
|-----|-----------|-----|---------------------|
| 1 | Regularised logreg (reg=0.1) | GAP | 3, 4, 60, 10⁻² |
| 2 | Standard logreg (separable) | UB | 3, 5, 45, 0.15 |
| 3 | Single-hidden-layer MLP (h=8) | GN | 3, 4, 60, 0.5 |
| 4 | 2-class regularised logreg (Pareto front) | GAP | 2, 5, 40, 0.05 |

**Each experiment function returns a result dict with a `config` key** containing the problem parameters (`K`, `p`, `n`, `d`, `reg`, `h`, `eps`). The plot titles and the LaTeX table both read from this config — change a parameter in the experiment function and both outputs update automatically. No hardcoded values anywhere.

**Outputs:**
- `experiment_results.png` — 2×2 figure with three convergence subplots and the Pareto front. Each convergence curve is labelled as "Algorithm 2 (GAP/UB/GN)" and includes the ε threshold as a dashed horizontal line. Subtitles show problem parameters (e.g., `K=3, p=4, n=60, d=12, reg=0.1`).
- `results_table.tex` — booktabs-style LaTeX table with columns: Experiment, PC, ε, Outer iters, Inner steps, Bundle size, Time (s). Problem parameters appear in a footnote-sized row below each main row. Bundle sizes use LaTeX thousand separators (`1{,}297`); ε values render as powers of ten when exact (e.g., `$10^{-2}$`) or as decimals otherwise (`$0.15$`).

---

## Dependencies

```
numpy
scipy
matplotlib
gurobipy       (optional; falls back to scipy SLSQP automatically if unavailable or license-limited)
```

**Gurobi license.** A free academic license is available at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/). If your license is expired, log into the Gurobi User Portal and request a new one from the License Request section. The scipy fallback is transparent — no code changes or environment variables needed.

## How to Run

```bash
pip install numpy scipy matplotlib gurobipy
python experiments.py
```

All four source files (`bundle.py`, `algorithm.py`, `objectives.py`, `experiments.py`) should be in the same directory. Running the script produces `experiment_results.png` and `results_table.tex` in the working directory.

---

## Sample Output

Console log from one full run:

```
=================================================================
Exp 1: Regularised multi-class logreg  (PC = GAP)
=================================================================
  K=3, p=4, n=60, reg=0.1, d=12, ε=0.01
  L = [1.0897 1.0897 1.0897],  µ = [0.1 0.1 0.1]
  outer iter   0 | PC* = 3.75e-01 | λ = [0. 0. 1.]
           inner steps =  10 | PC(λ_t; B) = 2.77e-03 after update
  outer iter   1 | PC* = 3.61e-01 | λ = [1. 0. 0.]
           inner steps =   7 | PC(λ_t; B) = 2.43e-03 after update
  ...
  outer iter  10 | PC* = 8.80e-03
  Converged at outer iteration 10.

  Outer iterations : 11
  Oracle calls     : 138
  Final PC*        : 8.80e-03
  Wall time        : 14.xxs
```

Representative metrics across the three main experiments:

| Exp | PC | Outer iters | Inner steps | Bundle size |
|-----|-----|---|---|---|
| 1 (reg logreg, GAP)  | GAP | 11 | 3–10     | 46    |
| 2 (std logreg, UB)   | UB  |  8 | 96–200   | 1,297 |
| 3 (MLP, GN)          | GN  |  4 | 2–3      | 9     |

Exp 1 is best-conditioned (small inner loop counts). Exp 2's large condition number (κ ≈ 10⁴ from tiny reg = 10⁻⁴) drives long inner loops. Exp 3 converges fastest because GN at θ = 0 is dominated by vertex λ values where a handful of gradient steps suffices.

---

## References

- Grigas, P. & Cheng, J. "A First-Order Bundle Method for Smooth Multi-objective Optimization." Preprint, UC Berkeley.
- Nesterov, Y. *Lectures on Convex Optimization.* Springer, 2nd edition, 2018.
- Désidéri, J.-A. "Multiple-gradient descent algorithm (MGDA) for multiobjective optimization." *Comptes Rendus Mathématique*, 2012.
