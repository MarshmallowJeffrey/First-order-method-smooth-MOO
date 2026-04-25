"""
experiments.py  –  Numerical experiments for the bundle MOO algorithm
=====================================================================

All experiments use multi-class classification objectives following
the notation from the paper:

    K classes, weight vectors  w^1, …, w^K ∈ R^p,
    labelled data  {(y_j, x_j)}_{j=1}^n  with  y_j ∈ [K],
    per-class loss  F_i(W) = (1/n_i) Σ_{j: y_j=i} {−log P(Y=i|x_j; W)}.

Five experiments:

  Exp 1 – Regularised multi-class logreg           (PC = GAP₁, strongly convex)
  Exp 2 – Single-hidden-layer MLP                  (PC = UB,   interpolation + PL)
  Exp 3 – Single-hidden-layer MLP                  (PC = GN,   generic non-convex)
  Exp 4 – Pareto front tracing (2-class logreg)

The algorithm uses:
  - PC-specific λ maximisation (SLSQP / multi-start) instead of grid search
  - Adaptive inner-loop stopping at ε/3 (from the Theorem 1 proof)

Output:  ``experiment_results.png``
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

#from algorithm import algorithm2
from bundle import Bundle, UB, GAP, GN, T_map
from objectives import (
    make_logreg_strongly_convex,
    make_mlp_nonconvex,
)
from baseline import *


# =====================================================================
#  Utility:  simplex grid  (for Pareto front)
# =====================================================================
def simplex_grid(K: int, resolution: int = 20) -> np.ndarray:
    """Tile the unit simplex Δ_K with a uniform grid."""
    if K == 1:
        return np.array([[1.0]])
    if K == 2:
        ts = np.linspace(0, 1, resolution + 1)
        return np.column_stack([ts, 1 - ts])
    points = []

    def _recurse(remaining, depth, current):
        if depth == K - 1:
            current.append(remaining)
            points.append(current[:])
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            _recurse(remaining - v, depth + 1, current)
            current.pop()

    _recurse(resolution, 0, [])
    return np.array(points, dtype=float) / resolution



# =====================================================================
#  Experiment 1:  Regularised multi-class logreg  (PC = GAP)
# =====================================================================
def experiment_logreg_gap(
    verbose: bool = True,
    coarse_resolution: int = 10,
    fine_resolution: int = 20,
    n_passes: int = 25,
    steps_per_point_per_pass: int = 1,
    max_outer: int = 60,
    max_inner: int = 5,
    plot_path: str = "cpu_vs_accuracy.png",
) -> Dict:
    """Compare Algorithm 2 vs the uniform-discretisation baseline on a
    CPU-time-vs-worst-case-accuracy axis.

    Protocol
    --------
    1. Precompute a reference optimal-value map  F*_λ  on a very fine
       uniform grid G_fine of Δ_K (one-time offline cost, not charged
       to either method).
    2. Run the baseline in "progressive" mode: pass through a coarse
       grid G_r doing M_pp gradient-descent steps per point per pass,
       checkpoint worst-case error (vs G_fine) after each pass.
    3. Run Algorithm 2 with a checkpoint after every outer iteration.
    4. Plot (CPU time) vs (worst-case suboptimality) for both methods
       on a log-y axis.

    Under strong convexity, the worst-case suboptimality
        err  =  sup_{λ ∈ G_fine}  [F_λ(x̂(λ)) − F*_λ]
    is the right metric — it directly measures how well the returned
    solution map approximates the true Pareto-scalarised optima
    uniformly over Δ_K.
    """
    print("=" * 65)
    print("Exp 1: Regularised multi-class logreg — CPU time vs worst-case err")
    print("=" * 65)

    K, p, n, reg = 4, 4, 30, 0.1
    d = K * p
    objs, grads, L, mu = make_logreg_strongly_convex(
        K=K, p=p, n=n, reg=reg, seed=43,
    )
    W0 = np.zeros(d)

    print(f"  K={K}, p={p}, n={n}, reg={reg}, d={d}")
    print(f"  L = {np.round(L, 4)}, µ = {mu}")

    # --- 1. Precompute reference map on the fine grid ---
    if verbose:
        print(f"\n  Precomputing reference map on fine grid "
              f"(resolution = {fine_resolution}) ...")
    ref_t0 = time.time()
    reference_map = compute_reference_map(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, fine_resolution=fine_resolution,
        n_iters=20_000, grad_tol=1e-5, verbose=False,
    )
    ref_time = time.time() - ref_t0
    print(f"  Reference map ready: {len(reference_map['fine_grid'])} points, "
          f"{ref_time:.1f}s")

    # --- 2. Run the progressive baseline ---
    if verbose:
        print(f"\n  Running baseline (coarse resolution = {coarse_resolution}, "
              f"{n_passes} passes, {steps_per_point_per_pass} GD steps/point/pass) ...")
    bl = uniform_discretisation_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, resolution=coarse_resolution,
        reference_map=reference_map,
        n_passes=n_passes,
        steps_per_point_per_pass=steps_per_point_per_pass,
        verbose=verbose,
    )

    # --- 3. Run Algorithm 2 with per-outer checkpoints ---
    if verbose:
        print(f"\n  Running Algorithm 2 ({max_outer} outer iters, "
              f"up to {max_inner} inner steps each) ...")
    a2 = algorithm2_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, reference_map=reference_map,
        mu=mu, mode="gap",
        max_outer=max_outer, max_inner=max_inner, verbose=verbose,
    )

    # --- 4. Plot ---
    _plot_cpu_vs_accuracy(
        bl=bl, a2=a2, plot_path=plot_path,
        problem_params={"K": K, "p": p, "n": n, "d": d, "reg": reg},
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
    )

    return {
        "reference_map": reference_map,
        "baseline": bl,
        "algorithm2": a2,
        "problem_params": {"K": K, "p": p, "n": n, "d": d, "reg": reg},
    }


def _plot_cpu_vs_accuracy(
    bl: Dict, a2: Dict,
    plot_path: str,
    problem_params: Dict,
    coarse_resolution: int,
    fine_resolution: int,
) -> None:
    """Plot CPU time vs worst-case suboptimality for both methods."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.semilogy(
        a2["cpu_times"], a2["worst_errs"],
        "o-", color="#2563eb", markersize=5, linewidth=1.8,
        label="Algorithm 2 (GAP)",
    )
    ax.semilogy(
        bl["cpu_times"], bl["worst_errs"],
        "s-", color="#dc2626", markersize=5, linewidth=1.8,
        label=f"Uniform discretisation (r = {coarse_resolution})",
    )

    ax.set_xlabel("CPU time (s)")
    ax.set_ylabel(r"$\sup_{\lambda \in G_{\mathrm{fine}}}\,"
                  r"[F_\lambda(\hat x(\lambda)) - F_\lambda^*]$")
    params_str = _format_params(problem_params)
    ax.set_title(
        f"CPU time vs worst-case suboptimality\n"
        f"{params_str}  |  G_fine res = {fine_resolution}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to {plot_path}")


# =====================================================================
#  Experiment 2:  Single Layer MLP  (PC = UB)
# =====================================================================



# =====================================================================
#  Experiment 3:  Single-hidden-layer MLP  (PC = GN)
# =====================================================================
def experiment_mlp_gn():
    """Generic non-convex regime: 1-hidden-layer neural network.

    Architecture:  x_j ∈ R^p  →  σ(W_1 x_j + b_1) ∈ R^h  →  W_2 a + b_2 ∈ R^K
    with σ = ReLU and softmax output.

    Parameters  θ = (W_1, b_1, W_2, b_2)  flattened,  d = h·p + h + K·h + K.

    F_i(θ) = (1/n_i) Σ_{j: y_j=i} {−z_j^{(i)} + log Σ_l exp(z_j^{(l)})}

    This is non-convex due to the composition  W_2 · σ(W_1 x + b_1).
    We use the GN (gradient norm) progress criterion.
    """
    print("=" * 65)
    print("Exp 3: Single-hidden-layer MLP  (PC = GN)")
    print("=" * 65)

    K, p, n, h = 3, 4, 60, 8
    d = h * p + h + K * h + K               # d = 67
    objs, grads, L = make_mlp_nonconvex(K=K, p=p, n=n, h=h, seed=7)
    theta0 = np.zeros(d)
    eps = 0.05

    print(f"  K={K}, p={p}, n={n}, h={h}, d={d}, ε={eps}")
    print(f"  Estimated L = {np.round(L, 2)}")

    t0 = time.time()
    res = algorithm2_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=theta0, eps=eps, mode="gn", mu=None,
        max_outer=40, max_inner=200, verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Outer iterations : {res['outer_iters']}")
    print(f"  Oracle calls     : {res['oracle_calls']}")
    print(f"  Final PC*        : {res['pc_history'][-1]:.4e}")
    print(f"  Wall time        : {elapsed:.2f}s\n")
    res["elapsed"] = elapsed
    res["config"] = {
        "name": "Exp 3: MLP neural network",
        "pc": "GN",
        "eps": eps,
        "params": {"K": K, "p": p, "n": n, "h": h, "d": d},
    }
    return res


# =====================================================================
#  Experiment 4:  Pareto front tracing  (2-class logreg)
# =====================================================================
def experiment_pareto_front():
    """Trace the Pareto front for 2-class regularised logistic regression.

    After Algorithm 2 converges, we evaluate the solution map
    Ŵ(λ) = T(λ; B_final) for a fine grid of λ ∈ Δ_2 and plot
    the corresponding (F_1(Ŵ), F_2(Ŵ)) pairs.

    The Pareto front shows the trade-off between per-class losses:
    improving class-1 accuracy comes at the cost of class-2 accuracy.
    """
    print("=" * 65)
    print("Exp 4: Pareto front  (2-class regularised logreg)")
    print("=" * 65)

    K, p, n, reg = 2, 5, 40, 0.05
    d = K * p                                # d = 10
    objs, grads, L, mu = make_logreg_strongly_convex(
        K=K, p=p, n=n, reg=reg, seed=99,
    )
    W0 = np.zeros(d)
    eps = 5e-2

    print(f"  K={K}, p={p}, n={n}, reg={reg}, d={d}, ε={eps}")

    res = algorithm2_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, eps=eps, mode="gap", mu=mu,
        max_outer=80, max_inner=200, verbose=False,
    )
    bundle = res["bundle"]

    # Evaluate the approximate solution map  Ŵ(λ) = T(λ; B_final)
    fine_grid = simplex_grid(K, 100)
    f1_vals, f2_vals = [], []
    for lam in fine_grid:
        W_hat = T_map(bundle, lam)
        f1_vals.append(objs[0](W_hat))
        f2_vals.append(objs[1](W_hat))

    print(f"  Outer iterations : {res['outer_iters']}")
    print(f"  Oracle calls     : {res['oracle_calls']}")
    print(f"  Bundle size      : {bundle.m} points\n")

    return f1_vals, f2_vals, res


# =====================================================================
#  Plotting
# =====================================================================
def _format_params(params):
    """Format a params dict as a plain-text string for plot titles.

    Examples:
        {'K': 3, 'p': 4, 'n': 60, 'd': 12, 'reg': 0.1}
          -> 'K=3, p=4, n=60, d=12, reg=0.1'
        {'K': 3, 'p': 5, 'n': 45, 'd': 15, 'separable': True}
          -> 'K=3, p=5, n=45, d=15, separable'
    """
    parts = []
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                parts.append(str(k))
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
def make_plots(res1, pareto_data, res2=None, res3=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Plot 1: GAP convergence (regularised logreg) ----
    ax = axes[0, 0]
    ax.semilogy(res1["pc_history"], "o-", color="#2563eb", markersize=4, linewidth=1.5,
                label="Algorithm 2 (GAP)")
    eps1 = res1["config"]["eps"]
    ax.axhline(y=eps1, color="grey", ls="--", lw=1, label=f"ε = {eps1}")
    ax.set_xlabel("Outer iteration t")
    ax.set_ylabel("max_λ GAP(λ; B_t)")
    ax.set_title(
        f"Exp 1: Regularised Logreg (GAP)\n"
        f"{_format_params(res1['config']['params'])}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: UB convergence (standard logreg) ----
    # ax = axes[0, 1]
    # ax.semilogy(res2["pc_history"], "s-", color="#dc2626", markersize=4, linewidth=1.5,
    #             label="Algorithm 2 (UB)")
    # eps2 = res2["config"]["eps"]
    # ax.axhline(y=eps2, color="grey", ls="--", lw=1, label=f"ε = {eps2}")
    # ax.set_xlabel("Outer iteration t")
    # ax.set_ylabel("max_λ UB(λ; B_t)")
    # ax.set_title(
    #     f"Exp 2: Standard Logreg, Separable (UB)\n"
    #     f"{_format_params(res2['config']['params'])}"
    # )
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    # ---- Plot 3: GN convergence (MLP) ----
    # ax = axes[1, 0]
    # ax.semilogy(res3["pc_history"], "^-", color="#16a34a", markersize=4, linewidth=1.5,
    #             label="Algorithm 2 (GN)")
    # eps3 = res3["config"]["eps"]
    # ax.axhline(y=eps3, color="grey", ls="--", lw=1, label=f"ε = {eps3}")
    # ax.set_xlabel("Outer iteration t")
    # ax.set_ylabel("max_λ GN(λ; B_t)")
    # ax.set_title(
    #     f"Exp 3: Single-Hidden-Layer MLP (GN)\n"
    #     f"{_format_params(res3['config']['params'])}"
    # )
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    # ---- Plot 4: Pareto front (2-class logreg) ----
    f1, f2, _ = pareto_data
    ax = axes[1, 1]
    ax.scatter(f1, f2, s=10, c="#7c3aed", alpha=0.7)
    ax.set_xlabel("F₁(Ŵ(λ))  [class 1 loss]")
    ax.set_ylabel("F₂(Ŵ(λ))  [class 2 loss]")
    ax.set_title("Exp 4: Pareto Front (2-class Logreg)\n"
                 "K=2, p=5, n=40, d=10, reg=0.05")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment_results.png", dpi=150)
    plt.close()
    print("Plots saved to experiment_results.png")




# =====================================================================
#  LaTeX table writer
# =====================================================================
def _fmt_inner_range(history):
    """Format inner-step history as a LaTeX range like '3--10' or '4'."""
    if not history:
        return "---"
    lo, hi = min(history), max(history)
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


def _fmt_int(n):
    """Format an integer with LaTeX thousand separators (1297 → '1{,}297')."""
    s = f"{n:,}"
    return s.replace(",", "{,}")


def _fmt_eps_latex(eps):
    """Format ε for LaTeX table cells.

    Exact powers of 10 render as $10^{k}$; other values as $0.15$ etc.
    """
    if eps <= 0:
        return f"${eps}$"
    log10 = np.log10(eps)
    if abs(log10 - round(log10)) < 1e-9:
        k = int(round(log10))
        return f"$10^{{{k}}}$"
    # Drop trailing zeros; typical εs are short decimals
    return f"${eps:g}$"


def _fmt_params_latex(params):
    r"""Format a params dict as the math body of a LaTeX stats line.

    Returns the raw math content (without outer $...$ delimiters) so the
    caller can wrap it in parentheses inside a single math-mode block.

    Examples:
        {'K': 3, 'p': 4, 'n': 60, 'd': 12, 'reg': 0.1}
          -> r'K\!=\!3,\; p\!=\!4,\; n\!=\!60,\; d\!=\!12,\; \mathrm{reg}\!=\!0.1'
        {'K': 3, 'p': 5, 'n': 45, 'd': 15, 'separable': True}
          -> r'K\!=\!3,\; p\!=\!5,\; n\!=\!45,\; d\!=\!15,\; \text{separable}'
    """
    parts = []
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                parts.append(rf"\text{{{k}}}")
        elif k == "reg":
            parts.append(rf"\mathrm{{reg}}\!=\!{v}")
        else:
            parts.append(rf"{k}\!=\!{v}")
    return r",\; ".join(parts)


def _write_baseline_comparison_table(
    rows: List[Dict],
    problem_params: Dict,
    path: str = "baseline_comparison.tex",
) -> None:
    """Write a LaTeX table comparing Algorithm 2 vs baseline across ε values.

    Schema:  Method | ε | Outer iters | Inner iters | PC | Runtime (s)

    Each ε produces two rows (one per method).  The problem-parameter
    footnote under the caption is pulled from ``problem_params``.
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\label{tab:baseline-comparison}",
        r"\smallskip",
        r"\begin{tabular}{@{}l c c c c c@{}}",
        r"\toprule",
        (r"\textbf{Method} & $\boldsymbol{\varepsilon}$ & "
         r"\textbf{Outer iters} & \textbf{Inner iters} & "
         r"\textbf{PC} & \textbf{Runtime (s)} \\"),
        r"\midrule",
    ]

    for idx, r in enumerate(rows):
        eps_cell = _fmt_eps_latex(r["eps"])
        lines.append(
            f"Algorithm 2 & {eps_cell} & {r['a2_outer']} & "
            f"{_fmt_int(r['a2_inner'])} & {'GAP'} & "
            f"{r['a2_time']:.2f} \\\\"
        )
        trailer = r"\\[4pt]" if idx < len(rows) - 1 else r"\\"
        lines.append(
            f"Baseline    & {eps_cell} & {_fmt_int(r['bl_outer'])} & --- & "
            f"{'GN'} & {r['bl_time']:.2f} {trailer}"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Algorithm 2 vs uniform-discretisation baseline on "
         r"regularised multi-class logistic regression "
         f"$({_fmt_params_latex(problem_params)})$.  "
         r"}"),
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nLaTeX table saved to {path}")



# =====================================================================
if __name__ == "__main__":
    res1 = experiment_logreg_gap()
    print("✓ Experiment completed.")
