"""Between-node-gap figures for one baseline-SVRG certification run.

NEW FILE (July 25, 2026; Note/Jul_25_note.md §7).  Post-processing only:
reads the ``delivery_audit.npz`` written by
``run_baseline_svrg_r_sweep_without_256_checkpoints.py --save-grams``
and produces TWO figures that demonstrate, without mixing meters, that
the certified grid nodes all meet node_tol while λ BETWEEN the nodes
does not:

1.  ``between_node_gap_path_r{r}.png`` — the value profile
        g(λ) = min_i λᵀM_iλ
    of the delivered set along the 1-D path
        nearest grid node → witness λ* → second-nearest grid node,
    parameterised by ℓ₁ arc length.  Dips at the grid nodes sit at or
    below node_tol (their exact certified values); the peak at λ* is
    the delivered set's strict full-simplex GN*.
2.  ``between_node_gap_nodes_r{r}.png`` — all N certified node values,
    sorted, on a log axis: the entire curve lies below the node_tol
    line, while the witness value sits a factor gap above it.

Everything is exact arithmetic on the cached full-gradient Grams — no
oracle calls, no external yardstick; the without-256 track rule is
untouched (this is delivery-time measurement, outside both cost axes).

Usage:
    python plot_between_node_gap_without_256_checkpoints.py \
        [--r 10] [--sweep-dir <path>] [--n-path 240]
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SWEEP = (HERE.parent.parent / "output"
                 / "pure_budget_without_256_checkpoints_SVRG_IPOPT_Baseline/baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/original")


def _g_values(Ms: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """g(λ) = min_i λᵀM_iλ for each row λ of ``lams`` (exact, cached)."""
    return np.einsum("pk,mkl,pl->pm", lams, Ms, lams).min(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--sweep-dir", type=str, default=str(DEFAULT_SWEEP))
    parser.add_argument("--n-path", type=int, default=240,
                        help="samples per path segment")
    args = parser.parse_args()

    sweep = Path(args.sweep_dir)
    r_dir = sweep / f"r{args.r:02d}"
    with np.load(r_dir / "delivery_audit.npz") as z:
        Ms = z["delivered_grams"]          # (m, K, K)
        best_val = z["node_best_val"]      # (N,)
        grid = z["grid"]                   # (N, K)
        lam_star = z["lambda_star"]        # (K,)
        node_tol = float(z["node_tol"])
        gn_strict = float(z["delivered_gn_strict"])

    N, K = grid.shape
    g_star = float(_g_values(Ms, lam_star[None, :])[0])

    # Witness value must reproduce the recorded strict score (same
    # arithmetic on the same Grams; the search may have found g_star
    # itself, the recorded value is the max of two searches).
    if not np.isclose(g_star, gn_strict, rtol=1e-6, atol=1e-12):
        print(f"note: g(lambda*)={g_star:.6e} vs recorded strict "
              f"{gn_strict:.6e} (max over two searches)")

    # Nearest and second-nearest grid nodes to the witness (l1).
    d1 = np.abs(grid - lam_star[None, :]).sum(axis=1)
    order = np.argsort(d1)
    ia, ib = int(order[0]), int(order[1])
    lam_a, lam_b = grid[ia], grid[ib]

    # Two straight segments inside the simplex, parameterised by l1 arc
    # length: lam_a -> lam_star -> lam_b.
    t = np.linspace(0.0, 1.0, args.n_path)
    seg1 = (1 - t)[:, None] * lam_a[None, :] + t[:, None] * lam_star[None, :]
    seg2 = (1 - t)[:, None] * lam_star[None, :] + t[:, None] * lam_b[None, :]
    len1 = float(np.abs(lam_star - lam_a).sum())
    len2 = float(np.abs(lam_b - lam_star).sum())
    x1 = t * len1
    x2 = len1 + t * len2
    g1 = _g_values(Ms, seg1)
    g2 = _g_values(Ms, seg2)

    inst = ("MLP K=6 (p=20, n=50000, h=[96, 96], tanh) — baseline-SVRG "
            f"r={args.r}, node_tol={node_tol:g} (without-256 track)")

    # ---------------- Figure 1: path profile ----------------
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x1, g1, color="#d62728", lw=1.8)
    ax.plot(x2, g2, color="#d62728", lw=1.8)
    ax.axhline(node_tol, color="dimgray", ls="--", lw=1.3,
               label=f"node_tol = {node_tol:g} (per-node contract)")
    ax.axhline(gn_strict, color="#8b1a1a", ls=":", lw=1.3,
               label=f"delivered GN* = {gn_strict:.4g} "
                     f"(= {gn_strict / node_tol:.1f} x node_tol)")
    for x_node, i_node, name in ((0.0, ia, "nearest grid node"),
                                 (len1 + len2, ib, "2nd-nearest grid node")):
        ax.scatter([x_node], [best_val[i_node]], s=70, zorder=5,
                   color="#1f77b4", edgecolor="black", linewidth=0.6)
        ax.annotate(f"{name}\ncertified {best_val[i_node]:.3g}",
                    xy=(x_node, best_val[i_node]),
                    xytext=(8, 10), textcoords="offset points", fontsize=8)
    ax.scatter([len1], [g_star], s=80, marker="*", color="#8b1a1a",
               edgecolor="black", linewidth=0.5, zorder=6)
    ax.annotate("witness $\\lambda^*$ (found by ONE delivery-time\n"
                "strict 64-start search)",
                xy=(len1, g_star), xytext=(10, -26),
                textcoords="offset points", fontsize=8, color="#8b1a1a")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell_1$ arc length along path:  node $\to$ "
                  r"$\lambda^*$ $\to$ node")
    ax.set_ylabel(r"$g(\lambda)=\min_i\,\lambda^\top M_i\lambda$"
                  "  (delivered set, exact)")
    ax.set_title("Grid nodes are certified; the λ between them is not\n"
                 + inst, fontsize=9.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out1 = sweep / f"between_node_gap_path_r{args.r:02d}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure 2: all certified nodes ----------------
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(np.arange(1, N + 1), np.sort(best_val), color="#1f77b4",
            lw=1.6, label=f"all {N:,} grid nodes, certified values (sorted)")
    ax.axhline(node_tol, color="dimgray", ls="--", lw=1.3,
               label=f"node_tol = {node_tol:g}")
    ax.axhline(gn_strict, color="#8b1a1a", ls=":", lw=1.5,
               label=(f"worst λ between nodes (witness): {gn_strict:.4g} "
                      f"= {gn_strict / node_tol:.1f} x node_tol"))
    ax.set_yscale("log")
    ax.set_xlabel("grid nodes, sorted by certified value")
    ax.set_ylabel(r"$\min_i\,\lambda^\top M_i\lambda$  (exact)")
    ax.set_title("Every grid node meets the contract; "
                 "the simplex between nodes does not\n" + inst, fontsize=9.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out2 = sweep / f"between_node_gap_nodes_r{args.r:02d}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out1}")
    print(f"wrote {out2}")
    print(f"witness lambda* = {np.round(lam_star, 4).tolist()}")
    print(f"nearest node l1-dist = {d1[ia]:.4f}, "
          f"2nd = {d1[ib]:.4f}; g(lambda*) = {g_star:.6e}; "
          f"max certified node value = {best_val.max():.6e}")


if __name__ == "__main__":
    main()
