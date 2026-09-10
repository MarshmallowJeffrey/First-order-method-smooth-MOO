"""plot_k3_solution_map_sheets_without_256_checkpoints.py — K = 3 front
sheets in the style of the MO-Gym figures (user request Sep 9 2026):
one row per leg (adaptive bundle on top, uniform grid below), three
views per row, TRAIN loss space or TEST cross-entropy space.

Construction ("linear display interpolation on a common preference
grid"): take a uniform simplex grid of preferences λ (resolution
``--grid-res``, restricted to λ_k >= ``--lam-min`` so the diverging
vertex arms stay out of view); for every λ pick the bundle point of the
solution map, θ̂(λ) = argmin_i ‖∇F_λ(θ_i)‖ = argmin_i λᵀQ_iλ from the
saved Gram stack; plot F(θ̂(λ)) (train fvals or test_ce) as a point, and
connect neighbouring preferences of the grid with triangles.  A smooth
solution map gives a smooth sheet; a bundle that only has good points at
a few preferences gives spikes.  Also prints a roughness number: the mean
3-D distance between the images of neighbouring grid preferences.

Usage:
    python plot_k3_solution_map_sheets_without_256_checkpoints.py --space train
    python plot_k3_solution_map_sheets_without_256_checkpoints.py --space test --uniform-r 30
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _layout  # noqa: F401
from run_k3_stepper_campaign_without_256_checkpoints import K3_HOME, TRIPLE  # noqa: E402

DEFAULT_HOME = K3_HOME / "dots_B100000" / "adam_1e-3_b0.9"
VIEWS = [(22, -60), (18, -140), (34, 115)]
COLORS = {"adaptive": "#d62728", "uniform": "#1f77b4"}


def simplex_grid(res, lam_min):
    nodes, index = [], {}
    for a in range(res + 1):
        for b in range(res + 1 - a):
            c = res - a - b
            lam = np.array([a, b, c], dtype=float) / res
            if lam.min() >= lam_min - 1e-12:
                index[(a, b)] = len(nodes)
                nodes.append(lam)
    tris = []
    for (a, b), i in index.items():
        j, k = index.get((a + 1, b)), index.get((a, b + 1))
        if j is not None and k is not None:
            tris.append((i, j, k))
        l = index.get((a + 1, b + 1))
        if j is not None and k is not None and l is not None:
            tris.append((j, l, k))
    return np.array(nodes), np.array(tris, dtype=int)


def solution_map(gram_stack, lams):
    """index of argmin_i λᵀ Q_i λ for every λ (rows of lams)."""
    G = np.asarray(gram_stack, dtype=float)                 # (m, 3, 3)
    A = np.einsum("mjk,nk->mnj", G, lams)                   # (m, n, 3)
    vals = np.einsum("mnj,nj->mn", A, lams)                 # (m, n)
    return np.argmin(vals, axis=0)


def roughness(P, tris):
    edges = set()
    for t in tris:
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
            edges.add((min(u, v), max(u, v)))
    e = np.array(sorted(edges))
    return float(np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(DEFAULT_HOME))
    ap.add_argument("--uniform-r", type=int, default=30)
    ap.add_argument("--grid-res", type=int, default=40)
    ap.add_argument("--lam-min", type=float, default=0.05)
    ap.add_argument("--space", choices=["train", "test"], default="train")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--budget-label", default="B = 100,000")
    ap.add_argument("--title", default=None, help="figure title override (user request Sep 9: no 'training' once the test figure is dropped)")
    ap.add_argument("--clip-pct", type=float, default=99.0,
                    help="shared axis limits for both rows: per coordinate from the minimum to this "
                         "percentile over both legs; triangles/points beyond are not drawn")
    a = ap.parse_args()
    home = Path(a.home)
    out_dir = Path(a.out_dir) if a.out_dir else home
    key = "fvals" if a.space == "train" else "test_ce"
    lams, tris = simplex_grid(a.grid_res, a.lam_min)
    legs = [("adaptive", "Adaptive bundle", home / "adaptive_ccp_seed41"),
            ("uniform", f"Uniform grid (r = {a.uniform_r})", home / f"uniform_r{a.uniform_r}_seed41")]
    sheets, report = {}, {}
    for tag, label, d in legs:
        npz = np.load(d / "grams.npz")
        idx = solution_map(npz["gram_stack"], lams)
        P = np.asarray(npz[key], dtype=float)[idx]
        sheets[tag] = (label, P)
        report[tag] = {"distinct_bundle_points_used": int(len(np.unique(idx))),
                       "roughness_mean_edge": roughness(P, tris),
                       "max_coord": float(P.max())}
    allP = np.vstack([sheets[t][1] for t, _l, _d in legs])
    lo = allP.min(axis=0); hi = np.percentile(allP, a.clip_pct, axis=0)
    pad = 0.04 * (hi - lo); lo = lo - pad; hi = hi + pad
    fig = plt.figure(figsize=(15, 9.6))
    for row, (tag, _lbl, _d) in enumerate(legs):
        label, P = sheets[tag]
        inside = ((P >= lo) & (P <= hi)).all(axis=1)
        keep_t = inside[tris].all(axis=1)
        report[tag]["nodes_shown"] = int(inside.sum())
        for col, (elev, azim) in enumerate(VIEWS):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            if keep_t.any():
                ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=tris[keep_t], color=COLORS[tag],
                                alpha=0.45, linewidth=0.25, edgecolor=COLORS[tag], shade=True)
            ax.scatter(P[inside, 0], P[inside, 1], P[inside, 2], s=4, color=COLORS[tag], depthshade=False)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{label} — View {col + 1}", fontsize=10)
            unit = "loss" if a.space == "train" else "test CE"
            ax.set_xlabel(f"digit-{TRIPLE[0]} {unit}", fontsize=8, labelpad=2)
            ax.set_ylabel(f"digit-{TRIPLE[1]} {unit}", fontsize=8, labelpad=2)
            ax.set_zlabel(f"digit-{TRIPLE[2]} {unit}", fontsize=8, labelpad=2)
            ax.tick_params(labelsize=7, pad=1)
    space_txt = "Training loss space" if a.space == "train" else "Test cross-entropy space"
    fig.suptitle(a.title or f"MNIST {TRIPLE[0]}/{TRIPLE[1]}/{TRIPLE[2]} — {space_txt} ({a.budget_label})", fontsize=13, y=0.98)
    fig.legend(handles=[Patch(facecolor=COLORS[t], alpha=0.6, label=l) for t, l, _ in legs],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.055), frameon=True)
    fig.text(0.5, 0.030,
             f"Points are the bundle points selected by the solution map θ̂(λ) = argmin_i ‖∇F_λ(θ_i)‖ on a common "
             f"preference grid (resolution {a.grid_res}, λ_k ≥ {a.lam_min:g});",
             ha="center", fontsize=8, color="#333333")
    fig.text(0.5, 0.012,
             f"meshes are linear display interpolation on that grid. Both rows share the same axes, "
             f"clipped at the {a.clip_pct:g}th percentile of each coordinate.",
             ha="center", fontsize=8, color="#333333")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.11, wspace=0.08, hspace=0.12)
    stem = f"pareto_solution_map_{a.space}"
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    (out_dir / f"{stem}.json").write_text(json.dumps({"space": a.space, "grid_res": a.grid_res,
        "lam_min": a.lam_min, "uniform_r": a.uniform_r, "nodes": int(len(lams)), "legs": report}, indent=2))
    print(f"figure -> {out_png}")
    for tag, r in report.items():
        print(f"  {tag:9s} shown {r.get('nodes_shown', 0)}/{len(lams)} nodes; distinct bundle points used: {r['distinct_bundle_points_used']:5d}; "
              f"mean neighbour jump: {r['roughness_mean_edge']:.4f}; max coordinate {r['max_coord']:.3f}")


if __name__ == "__main__":
    main()
