"""plot_k3_front_alternatives_without_256_checkpoints.py — two readable
alternatives to the K = 3 3-D front sheets (user question Sep 9 2026).

(1) ``pairwise``: the non-dominated set of every delivered point (window
    <= ln 3), projected onto the three objective pairs; in each panel the
    2-D lower envelope of the projection is drawn as a curve per method,
    in the style of the K = 2 Figure 4.  Train or test space.
(2) ``certificate``: a heat map over the preference simplex of the
    per-preference certificate ‖∇F_λ(θ̂(λ))‖ = sqrt(min_i λᵀQ_iλ), one
    panel per method, shared log colour scale; the maximum is the
    worst-case gradient norm of Table 8.  This is the quantity the main
    metric summarises, shown for every λ.

Usage:
    python plot_k3_front_alternatives_without_256_checkpoints.py pairwise --space train
    python plot_k3_front_alternatives_without_256_checkpoints.py certificate --grid-res 120
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import _layout  # noqa: F401
from plot_K3_mnist_triple_without_256_checkpoints import LN3, _nondominated_kd  # noqa: E402
from plot_pareto_front_paper_K2_without_256_checkpoints import nondominated_2d  # noqa: E402
from run_k3_stepper_campaign_without_256_checkpoints import K3_HOME, TRIPLE  # noqa: E402

DEFAULT_HOME = K3_HOME / "dots_B100000" / "adam_1e-3_b0.9"
STYLE = {"adaptive": dict(color="#ff7f00", ls="-", lw=1.8, label="adaptive λ-bundle"),
         "uniform": dict(color="#377eb8", ls="--", lw=2.0, label="uniform grid")}


def load(home, r):
    out = {}
    for tag, d in (("adaptive", home / "adaptive_ccp_seed41"), ("uniform", home / f"uniform_r{r}_seed41")):
        out[tag] = np.load(d / "grams.npz")
    return out


def pairwise(a):
    home = Path(a.home)
    runs = load(home, a.uniform_r)
    key = "fvals" if a.space == "train" else "test_ce"
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=200)
    for tag, npz in runs.items():
        F = np.asarray(npz[key], dtype=float)
        F = F[np.isfinite(F).all(axis=1)]
        nd = F[_nondominated_kd(F)]
        nd = nd[(nd <= LN3).all(axis=1)]
        st = dict(STYLE[tag])
        if tag == "uniform":
            st["label"] = f"uniform grid, r = {a.uniform_r}"
        for ax, (i, j) in zip(axes, pairs):
            P = nd[:, [i, j]]
            env = P[nondominated_2d(P)]
            ax.plot(env[:, 0], env[:, 1], **st)
    unit = "training loss" if a.space == "train" else "test cross-entropy"
    for ax, (i, j) in zip(axes, pairs):
        ax.set_xlim(-0.004, a.window); ax.set_ylim(-0.004, a.window)
        ax.set_xlabel(f"{unit} of digit {TRIPLE[i]}", fontsize=10)
        ax.set_ylabel(f"{unit} of digit {TRIPLE[j]}", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle(f"MNIST {TRIPLE[0]}/{TRIPLE[1]}/{TRIPLE[2]}: non-dominated front projected on the three objective pairs "
                 f"({'training loss' if a.space == 'train' else 'test cross-entropy'} space)", fontsize=11)
    fig.tight_layout()
    out = home / f"pareto_pairwise_{a.space}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"figure -> {out}")


def simplex_grid(res):
    nodes = [np.array([i, j, res - i - j], dtype=float) / res
             for i in range(res + 1) for j in range(res + 1 - i)]
    return np.array(nodes)


def certificate(a):
    home = Path(a.home)
    runs = load(home, a.uniform_r)
    lams = simplex_grid(a.grid_res)
    x = lams[:, 1] + 0.5 * lams[:, 2]
    y = (np.sqrt(3) / 2) * lams[:, 2]
    tri = mtri.Triangulation(x, y)
    vals = {}
    for tag, npz in runs.items():
        G = np.asarray(npz["gram_stack"], dtype=float)
        best = np.full(len(lams), np.inf)
        for s in range(0, len(G), 2000):          # chunked min_i λᵀQ_iλ
            A = np.einsum("mjk,nk->mnj", G[s:s + 2000], lams)
            v = np.einsum("mnj,nj->mn", A, lams)
            best = np.minimum(best, v.min(axis=0))
        vals[tag] = np.sqrt(np.maximum(best, 0.0))
    vmin = min(v.min() for v in vals.values()); vmax = max(v.max() for v in vals.values())
    levels = np.geomspace(vmin, vmax, 30)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), dpi=200)
    for ax, (tag, v) in zip(axes, vals.items()):
        cf = ax.tricontourf(tri, v, levels=levels, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
        k = int(np.argmax(v))
        ax.plot([x[k]], [y[k]], marker="x", color="red", ms=10, mew=2.2)
        lab = "adaptive λ-bundle" if tag == "adaptive" else f"uniform grid, r = {a.uniform_r}"
        ax.set_title(f"{lab}\nmax over the grid = {v.max():.2e}", fontsize=10.5, pad=18)
        ax.text(0.0, -0.05, f"λ = e_{TRIPLE[0]} (digit {TRIPLE[0]} only)", ha="left", va="top", fontsize=8.5)
        ax.text(1.0, -0.05, f"λ = e_{TRIPLE[1]} (digit {TRIPLE[1]} only)", ha="right", va="top", fontsize=8.5)
        ax.text(0.5, np.sqrt(3) / 2 + 0.025, f"λ = e_{TRIPLE[2]} (digit {TRIPLE[2]} only)", ha="center", va="bottom", fontsize=8.5)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.14, 0.98)
        ax.set_aspect("equal"); ax.axis("off")
    ticks = [t for t in (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1) if vmin <= t <= vmax]
    cb = fig.colorbar(cf, ax=axes, shrink=0.8, pad=0.03, ticks=ticks)
    cb.ax.set_yticklabels([f"{t:g}" for t in ticks])
    cb.set_label("‖∇F_λ(θ̂(λ))‖: gradient norm of the best bundle point for λ (log scale)", fontsize=9)
    fig.suptitle(f"MNIST {TRIPLE[0]}/{TRIPLE[1]}/{TRIPLE[2]}: per-preference certificate over the simplex, "
                 f"B = 100,000 (red cross = worst preference on the grid)", fontsize=11)
    out = home / "certificate_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")
    for tag, v in vals.items():
        print(f"  {tag:9s} max {v.max():.3e}  median {np.median(v):.3e}  fraction of λ above 2x median: {(v > 2 * np.median(v)).mean():.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["pairwise", "certificate"])
    ap.add_argument("--home", default=str(DEFAULT_HOME))
    ap.add_argument("--uniform-r", type=int, default=30)
    ap.add_argument("--space", choices=["train", "test"], default="train")
    ap.add_argument("--window", type=float, default=0.3)
    ap.add_argument("--grid-res", type=int, default=120)
    a = ap.parse_args()
    (pairwise if a.kind == "pairwise" else certificate)(a)


if __name__ == "__main__":
    main()
