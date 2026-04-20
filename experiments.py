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

from algorithm import algorithm2
from bundle import Bundle, UB, GAP, GN, T_map
from objectives import (
    make_logreg_strongly_convex,
    make_mlp_nonconvex,
)


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
def experiment_logreg_gap():
    """Strongly convex regime: ℓ₂-regularised multi-class logistic regression.

    F_i(W) = (1/n_i) Σ_{j: y_j=i} {−⟨w^i,x_j⟩ + log Σ_l exp(⟨w^l,x_j⟩)}
             + (reg/2) ‖W‖²

    The ℓ₂ term makes each F_i  reg-strongly convex (µ_i = reg = 0.1).
    We use the GAP = UB − LB progress criterion.
    """
    print("=" * 65)
    print("Exp 1: Regularised multi-class logreg  (PC = GAP)")
    print("=" * 65)

    K, p, n, reg = 4, 4, 60, 0.1
    d = K * p                                # d = 12
    objs, grads, L, mu = make_logreg_strongly_convex(
        K=K, p=p, n=n, reg=reg, seed=42,
    )
    W0 = np.zeros(d)
    eps = 5e-2

    print(f"  K={K}, p={p}, n={n}, reg={reg}, d={d}, ε={eps}")
    print(f"  L = {np.round(L, 4)},  µ = {mu}")

    t0 = time.time()
    res = algorithm2(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, eps=eps, mode="gap", mu=mu,
        max_outer=80, max_inner=200, verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Outer iterations : {res['outer_iters']}")
    print(f"  Oracle calls     : {res['oracle_calls']}")
    print(f"  Final PC*        : {res['pc_history'][-1]:.4e}")
    print(f"  Wall time        : {elapsed:.2f}s\n")
    res["elapsed"] = elapsed
    res["config"] = {
        "name": "Exp 1: Regularised logreg",
        "pc": "GAP",
        "eps": eps,
        "params": {"K": K, "p": p, "n": n, "d": d, "reg": reg},
    }
    return res


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
    res = algorithm2(
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

    res = algorithm2(
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
def make_plots(res1, res3, pareto_data, res2=None):
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
    ax = axes[1, 0]
    ax.semilogy(res3["pc_history"], "^-", color="#16a34a", markersize=4, linewidth=1.5,
                label="Algorithm 2 (GN)")
    eps3 = res3["config"]["eps"]
    ax.axhline(y=eps3, color="grey", ls="--", lw=1, label=f"ε = {eps3}")
    ax.set_xlabel("Outer iteration t")
    ax.set_ylabel("max_λ GN(λ; B_t)")
    ax.set_title(
        f"Exp 3: Single-Hidden-Layer MLP (GN)\n"
        f"{_format_params(res3['config']['params'])}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

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


def write_results_table(results, path="results_table.tex"):
    """Write a LaTeX results table summarising a list of experiments.

    Each entry in ``results`` must be a result dict with a ``config`` key
    containing:  {"name": str, "pc": str, "eps": float, "params": dict}
    in addition to the standard algorithm2 outputs (``outer_iters``,
    ``inner_steps_history``, ``bundle``, ``elapsed``).

    Schema:
        Experiment | PC | ε | Outer iters | Inner steps | Bundle size | Time (s)
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\label{tab:experiment-results}",
        r"\smallskip",
        r"\begin{tabular}{@{}l c c c c c c@{}}",
        r"\toprule",
        (r"\textbf{Experiment} & \textbf{PC} & $\boldsymbol{\varepsilon}$ & "
         r"\textbf{Outer iters} & \textbf{Inner steps} & "
         r"\textbf{Bundle size} & \textbf{Time (s)} \\"),
        r"\midrule",
    ]

    for idx, res in enumerate(results):
        cfg = res["config"]
        outers = res["outer_iters"]
        inner_range = _fmt_inner_range(res.get("inner_steps_history", []))
        bundle_size = res["bundle"].m
        elapsed = res["elapsed"]

        lines.append(
            f"{cfg['name']}  & {cfg['pc']} & {_fmt_eps_latex(cfg['eps'])} & "
            f"{outers} & {inner_range} & {_fmt_int(bundle_size)} & "
            f"{elapsed:.2f} \\\\"
        )
        trailer = r"\\[6pt]" if idx < len(results) - 1 else r"\\"
        stats_body = _fmt_params_latex(cfg["params"])
        lines.append(
            f"{{\\footnotesize (${stats_body}$)}} & & & & & & {trailer}"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Performance of the first-order bundle method under "
        r"different objective-function assumptions.}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved to {path}")



# =====================================================================
if __name__ == "__main__":
    res1 = experiment_logreg_gap()
    #res2 = experiment_logreg_ub()
    res3 = experiment_mlp_gn()
    pareto = experiment_pareto_front()
    make_plots(res1=res1, res3=res3, pareto_data=pareto)
    #make_plots(res1, res2, res3, pareto)
    write_results_table([res1, res3], path="results_table.tex")
    print("✓ All experiments completed.")
