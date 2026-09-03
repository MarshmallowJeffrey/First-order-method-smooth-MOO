"""plot_k3_adam_main_without_256_checkpoints.py — report figures for the
K = 3 adam-core main line (user request Sep 4):

  1. main/worst_gn_curves_adam.png — adam core only: adaptive-CCP + the
     three uniform baselines (r = 10 / 20 / 30), worst GN (norm) vs
     grad_equiv and vs CPU seconds (no const-core reference curves);
  2. main/pareto_front_adam_3d.png — 3-D Pareto frontier SHEET (the
     MODPO-style triangulated rendering of the K3 plotter) for adaptive
     vs the best uniform r (lowest final audit at B = 40,000), in the
     TRAINING loss space actually optimised (penalised per-class CE).

Reads the summaries written by run_k3_stepper_campaign; nothing rerun.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _layout  # noqa: F401
from plot_K3_mnist_triple_without_256_checkpoints import (  # noqa: E402
    LN3,
    VIEWS3,
    _nondominated_kd,
)
from run_k3_stepper_campaign_without_256_checkpoints import (  # noqa: E402
    MAIN_HOME,
    TRIPLE,
)

CORE = "adam_1e-3_b0.9"
RS = (10, 20, 30)
HOME = MAIN_HOME / CORE


def _load(name):
    sm = json.loads((HOME / name / "summary.json").read_text())
    npz = np.load(HOME / name / "grams.npz")
    return sm, np.asarray(npz["fvals"], dtype=float)


def curves_figure(legs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {"adaptive": ("-", "#ff7f0e", 2.0),
              "uniform_r10": ("--", "#1f77b4", 1.4),
              "uniform_r20": (":", "#2ca02c", 1.4),
              "uniform_r30": ("-.", "#9467bd", 1.4)}
    for leg, (sm, _f) in legs.items():
        g = np.asarray(sm["ck_grads"], dtype=float)
        v = np.sqrt(np.maximum(np.asarray(sm["audited_gn_history"]), 0.0))
        cpu = np.asarray(sm["ck_cpu"], dtype=float)
        ls, col, lw = styles[leg]
        axes[0].plot(g, v, ls, color=col, lw=lw, label=leg)
        axes[1].plot(cpu, v, ls, color=col, lw=lw, label=leg)
    axes[0].set_xlabel("total gradient evaluations (grad_equiv)")
    axes[1].set_xlabel("CPU seconds")
    for ax in axes:
        ax.set_ylabel("best-so-far worst GN (norm)")
        ax.set_yscale("log"); ax.legend(fontsize=8)
    axes[0].set_title(f"K3 main, adam core: {TRIPLE[0]},{TRIPLE[1]},"
                      f"{TRIPLE[2]}, ridge mu=1e-4, B=40000, seed 41",
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(HOME.parent / "worst_gn_curves_adam.png", dpi=150)
    plt.close(fig)


def _envelope(fr, nbins=18):
    lo = max(1e-3, 0.9 * float(fr[:, :2].min()))
    edges = np.geomspace(lo, LN3, nbins + 1)
    xi = np.clip(np.searchsorted(edges, fr[:, 0]) - 1, 0, nbins - 1)
    yi = np.clip(np.searchsorted(edges, fr[:, 1]) - 1, 0, nbins - 1)
    best = {}
    for k in range(fr.shape[0]):
        key = (int(xi[k]), int(yi[k]))
        if key not in best or fr[k, 2] < fr[best[key], 2]:
            best[key] = k
    return fr[np.array(sorted(best.values()))]


def surface_figure(series, best_r, edge_max=0.45):
    """series = [(label, color, front_pts)] baseline first, adaptive last.
    Same sheet construction as the K3 plotter's _front_surface_figure
    (lower envelope on a log (F1,F2) grid, Delaunay in (F1,F2), long
    bridging triangles dropped), retitled for the TRAIN loss space."""
    env_series = [(lbl, col, _envelope(fr)) for lbl, col, fr in series]
    fig = plt.figure(figsize=(13.4, 4.9))
    for p, (elev, azim) in enumerate(VIEWS3):
        ax = fig.add_subplot(1, 3, p + 1, projection="3d")
        for lbl, col, env in env_series:
            ax.scatter(env[:, 0], env[:, 1], env[:, 2], color=col, s=7,
                       alpha=0.9, depthshade=False)
            x, y, z = env[:, 0], env[:, 1], env[:, 2]
            key = np.round(x, 8) + 1j * np.round(y, 8)
            _, uniq = np.unique(key, return_index=True)
            xu, yu, zu = x[uniq], y[uniq], z[uniq]
            if xu.size < 4:
                continue
            tri = mtri.Triangulation(xu, yu)
            t = tri.triangles
            P = np.stack([xu, yu, zu], axis=1)
            a, b, c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
            elen = np.maximum.reduce([np.linalg.norm(a - b, axis=1),
                                      np.linalg.norm(b - c, axis=1),
                                      np.linalg.norm(a - c, axis=1)])
            keep = elen <= edge_max
            if keep.any():
                ax.plot_trisurf(xu, yu, zu, triangles=t[keep], color=col,
                                alpha=0.55, linewidth=0.15, edgecolor=col,
                                shade=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=8)
        ax.set_xlabel(f"digit-{TRIPLE[0]} loss", fontsize=7, labelpad=-1)
        ax.set_ylabel(f"digit-{TRIPLE[1]} loss", fontsize=7, labelpad=-1)
        ax.set_zlabel(f"digit-{TRIPLE[2]} loss", fontsize=7, labelpad=-1)
        ax.tick_params(labelsize=6, pad=-1)
        if p == 0:
            ax.legend(handles=[Patch(facecolor=c_, alpha=0.6,
                                     label=f"{l_} frontier sheet "
                                           f"({len(e_)} envelope pts)")
                               for l_, c_, e_ in env_series],
                      fontsize=6.5, loc="upper left")
    # short title (user request Sep 3); the construction details (train
    # loss space, window <= ln 3, bridging triangles > edge_max dropped)
    # are stated in the report caption instead
    fig.suptitle(
        f"Pareto frontier sheets, MNIST {TRIPLE[0]}/{TRIPLE[1]}/{TRIPLE[2]}, "
        f"adam core, B=40,000: adaptive CCP vs uniform r={best_r}",
        fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.05,
                        wspace=0.10)
    fig.savefig(HOME.parent / "pareto_front_adam_3d.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main():
    legs = {"adaptive": _load("adaptive_ccp_seed41")}
    for r in RS:
        legs[f"uniform_r{r}"] = _load(f"uniform_r{r}_seed41")
    curves_figure(legs)
    best_r = min(RS, key=lambda r: legs[f"uniform_r{r}"][0]["final_audit"])
    series = []
    for lbl, col, name in ((f"uniform r={best_r}", "#2ca02c",
                            f"uniform_r{best_r}"),
                           ("adaptive CCP", "#ff7f0e", "adaptive")):
        F = legs[name][1]
        fr = F[_nondominated_kd(F)]
        series.append((lbl, col, fr[(fr <= LN3).all(axis=1)]))
    surface_figure(series, best_r)
    out = {"core": CORE, "best_uniform_r": best_r,
           "final_worst_gn_norm": {k: float(np.sqrt(max(v[0]["final_audit"], 0)))
                                   for k, v in legs.items()}}
    (HOME.parent / "adam_main_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1), flush=True)
    print("[plots] ->", HOME.parent / "worst_gn_curves_adam.png",
          HOME.parent / "pareto_front_adam_3d.png", flush=True)


if __name__ == "__main__":
    main()
