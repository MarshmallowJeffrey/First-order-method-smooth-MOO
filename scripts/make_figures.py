#!/usr/bin/env python
"""The figures of Section 4.1 and Appendix C.1, drawn from results/ (no training, no audits):

    figures/mnist_worst_gn_k2.pdf    worst-case gradient norm vs gradient calls and time, {4,9}
    figures/mnist_worst_gn_k3.pdf    the same for {4,7,9}
    figures/mnist_front_k2.pdf       linear scalarization fronts, {4,9} (mean of three seeds)
    figures/mnist_front_k3.pdf       linear scalarization fronts, {4,7,9}: seed 41 and the mean of three seeds
    figures/mnist_step_rules_k2.pdf  step-rule experiment, {4,9}

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm import config as C  # noqa: E402
from abm.analysis import fit_trend, suffix_max, trend_curve  # noqa: E402
from abm.fronts import dominated_share, log_cells, mean_front_2d, mean_front_3d  # noqa: E402
from abm.labels import place  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
COL = {"adaptive": "#ff7f0e", "uniform": "#1f77b4", "surf": "#d62728"}
MARK = {"uniform": "s", "surf": "^"}
NAME = {"adaptive": "Adaptive Bundle Method", "uniform": "Unif Discrtztn", "surf": "SURF"}
YLAB = r"$\max_{\lambda\in\Delta_K}\,\mathrm{GN}(\lambda,B_t)$"


def _save(fig, stem, pdf_dpi=None, **kw):
    """PNG at 300 dpi; PDF vector (pdf_dpi: resolution of its rasterized parts), without a creation date so that
    a rerun gives the same bytes."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, **kw)
    fig.savefig(FIGURES / f"{stem}.pdf", metadata={"CreationDate": None}, **(dict(kw, dpi=pdf_dpi) if pdf_dpi else kw))
    plt.close(fig)
    print("saved", FIGURES / f"{stem}.pdf")


def adaptive_curve(res):
    """The adaptive method over the seeds: at every checkpoint (the same budgets in every seed) the geometric mean
    of the seeds' audited worst-case gradient norms (each repaired by its suffix maximum) and of their wall-clock
    times.  Returns (gradient calls, wall-clock seconds, worst-case gradient norm)."""
    runs = sorted((r for r in res["runs"].values() if r["method"] == "adaptive"), key=lambda r: r["seed"])
    if any(r["ck_grads"] != runs[0]["ck_grads"] for r in runs):
        raise ValueError("the adaptive runs have different checkpoints")
    G = np.array([suffix_max(r["audit_gn"]) for r in runs])
    W = np.array([r["ck_wall"] for r in runs], dtype=float)
    wall = np.where((W > 0).all(axis=0), np.exp(np.log(np.where(W > 0, W, 1.0)).mean(axis=0)), 0.0)
    return np.asarray(runs[0]["ck_grads"], float), wall, np.exp(np.log(G).mean(axis=0))


def worst_gn(K):
    """Markers: the drawn configurations (geometric means over the seeds); dashed: fitted trends; curve: the
    adaptive method (geometric mean over the seeds)."""
    res = json.loads((RESULTS / f"k{K}.json").read_text())
    fams = ("uniform", "surf") if K == 2 else ("uniform",)
    drawn = {"uniform": C.FIGURE_UNIFORM_R[K], "surf": C.FIGURE_SURF_N}
    stats = {(s["method"], s["param"]): s for s in res["configs"]}
    big = {(f, p): {"budget": stats[(f, p)]["x_geomean"], "wall_seconds": stats[(f, p)]["wall_geomean"],
                    "norm": stats[(f, p)]["y_geomean"]} for f in fams for p in drawn[f]}
    fits = {}
    for f in fams:
        keys = sorted(k for k in big if k[0] == f)
        for axis in ("budget", "wall_seconds"):
            fits[(f, axis)] = fit_trend([big[k][axis] for k in keys], [big[k]["norm"] for k in keys])
    ad_x, ad_w, ad_y = adaptive_curve(res)

    FS = dict(label=16, tick=14, legend=14, num=11)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    per_axis, handles = [], {}
    for ax, axis, xlabel in ((axs[0], "budget", "Gradient Calls"), (axs[1], "wall_seconds", "Time (s)")):
        axv = ad_x if axis == "budget" else ad_w
        m = ad_x > 0
        handles["adaptive"], = ax.plot(axv[m], ad_y[m], "-", color=COL["adaptive"], lw=2.6, zorder=3)
        items, lines = [], []
        for f in fams:
            keys = sorted(k for k in big if k[0] == f)
            handles[f], = ax.plot([big[k][axis] for k in keys], [big[k]["norm"] for k in keys], MARK[f],
                                  color=COL[f], ms=9, mec="white", mew=0.8, ls="", zorder=5)
            xs, ys = trend_curve(fits[(f, axis)])
            ax.plot(xs, ys, "--", color=COL[f], lw=1.8, alpha=0.9, zorder=2)
            if K == 2:
                lines.append((xs, ys))
            else:                                    # the curve the K = 3 labels avoid: 600 linear samples
                dense = np.linspace(min(xs), max(xs), 600)
                fit = fits[(f, axis)]
                lines.append((dense, fit["c"] + fit["a"] * (dense / fit["s"]) ** (-fit["p"])))
            items += [{"x": big[k][axis], "y": big[k]["norm"], "text": str(k[1]), "color": COL[f], "key": k}
                      for k in keys]
        xmax = 1.1 * max(it["x"] for it in items)
        ax.set_yscale("log")
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel, fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        mv = (axv > 0) & (axv <= xmax)
        per_axis.append((ax, xlabel, items, lines + [(axv[mv], ad_y[mv])], float(ad_y[mv].min()), float(ad_y[mv].max())))
    axs[0].set_ylabel(YLAB, fontsize=FS["label"])
    y_lo = min(p[4] for p in per_axis) / 1.35                         # below the visible adaptive curve
    y_hi = 1.9 * max(it["y"] for p in per_axis for it in p[2])         # room above the top markers for their labels
    if K == 3:
        y_hi = max(y_hi, 1.15 * max(p[5] for p in per_axis))
    axs[0].set_ylim(y_lo, y_hi)
    if K == 3:           # about 1.5 decades: label the 2x, 3x and 5x ticks as well
        axs[0].yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2.0, 3.0, 5.0)))
        axs[0].yaxis.set_minor_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False,
                                                                         minor_thresholds=(3, 3)))
        axs[0].tick_params(axis="y", which="minor", labelsize=FS["tick"] - 1)
        axs[1].tick_params(axis="y", which="both", labelleft=False)
    names = [NAME["adaptive"]] + [f"{NAME[f]} ({'r' if f == 'uniform' else 'N'})" for f in fams]
    fig.legend([handles["adaptive"]] + [handles[f] for f in fams], names, loc="upper center", ncol=len(names),
               fontsize=FS["legend"], frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.15, top=0.87, wspace=0.07)
    for ax, xlabel, items, lines, _, _ in per_axis:     # labels last: they need the final limits and layout
        info = place(ax, items, lines, fontsize=FS["num"])
        if info["hard"]:
            print(f"  warning ({xlabel}): labels with a hard clash: {info['hard']}")
    _save(fig, f"mnist_worst_gn_k{K}")


def front_k2():
    """Seed-mean fronts of the adaptive method, uniform discretization (r = 60) and SURF (N = 38)."""
    fr = json.loads((RESULTS / "k2_fronts.json").read_text())
    spec, window, xy_min = C.FRONT_LEGS[2], C.FRONT_WINDOW[2], 0.04
    styles = {"surf": dict(color="#e41a1c", ls=":", lw=2.4, zorder=4), "uniform": dict(color="#377eb8", ls="--", lw=2.2, zorder=5),
              "adaptive": dict(color="#ff7f00", ls="-", lw=1.8, zorder=6)}
    legs = {"adaptive": "adaptive_seed{}", "uniform": f"uniform_r{spec['uniform']}_seed{{}}", "surf": f"surf_N{spec['surf']}_seed{{}}"}
    labels = {"adaptive": NAME["adaptive"], "uniform": f"{NAME['uniform']} (r={spec['uniform']})",
              "surf": f"{NAME['surf']} (N={spec['surf']})"}
    fig, ax = plt.subplots(figsize=(3.1, 3.0))
    for fam in ("surf", "uniform", "adaptive"):
        grid, mean = mean_front_2d([np.asarray(fr[legs[fam].format(s)]) for s in spec["seeds"]], window)
        st = dict(styles[fam])
        st["lw"] = 0.6 * st["lw"]
        ax.plot(grid, mean, label=labels[fam], solid_capstyle="round", **st)
    ax.set_xlim(xy_min, window)
    ax.set_ylim(xy_min, window)
    ax.set_aspect("equal", adjustable="box")
    d = C.DIGITS[2]
    ax.set_xlabel(f"$F_{d[0]}$", fontsize=10, labelpad=1)
    ax.set_ylabel(f"$F_{d[1]}$", fontsize=10, labelpad=1)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index(labels[f]) for f in ("adaptive", "uniform", "surf")]
    ax.legend([h[i] for i in order], [lab[i] for i in order], fontsize=7, loc="upper right", handlelength=2.2)
    _save(fig, "mnist_front_k2", pdf_dpi=300, bbox_inches="tight", pad_inches=0.03)


def _sheet(ax, env, col, edge_max, alpha):
    """Front surface: Delaunay triangulation of the envelope points in (F_1, F_2), long triangles dropped."""
    x, y, z = env[:, 0], env[:, 1], env[:, 2]
    _, uniq = np.unique(np.round(x, 8) + 1j * np.round(y, 8), return_index=True)
    xu, yu, zu = x[uniq], y[uniq], z[uniq]
    if xu.size < 4:
        return
    t = mtri.Triangulation(xu, yu).triangles
    P = np.stack([xu, yu, zu], axis=1)
    a, b, c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
    elen = np.maximum.reduce([np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(a - c, axis=1)])
    keep = elen <= edge_max
    if keep.any():
        ax.plot_trisurf(xu, yu, zu, triangles=t[keep], color=col, alpha=alpha, linewidth=0.1, edgecolor=col,
                        shade=True, rasterized=True)


def _k3_panel(fr, seeds, r):
    """Mean front of each method over the given seeds (see abm.fronts.mean_front_3d) inside the box where both
    methods' fronts exist, the ideal point (mean over the seeds), and the boxed fronts per seed."""
    data = {"adaptive": [np.asarray(fr[f"adaptive_seed{s}"]) for s in seeds],
            "uniform": [np.asarray(fr[f"uniform_r{r}_seed{s}"]) for s in seeds]}
    allA, allU = np.vstack(data["adaptive"]), np.vstack(data["uniform"])
    lo, hi = np.maximum(allA.min(axis=0), allU.min(axis=0)), np.minimum(allA.max(axis=0), allU.max(axis=0))
    box = {k: [F[(F >= lo).all(axis=1) & (F <= hi).all(axis=1)] for F in v] for k, v in data.items()}
    edges = log_cells(box["adaptive"] + box["uniform"])
    env = {k: mean_front_3d(v, edges) for k, v in box.items()}
    ideal = np.mean([np.minimum(box["adaptive"][i].min(axis=0), box["uniform"][i].min(axis=0))
                     for i in range(len(seeds))], axis=0)
    return env, ideal, box


def front_k3():
    """Fronts of the adaptive method and uniform discretization (r = 24) inside the box where both exist: seed 41
    (left) and the mean of the three seeds (right).  Also prints, per seed, the share of each front dominated by
    the other."""
    fr = json.loads((RESULTS / "k3_fronts.json").read_text())
    spec = C.FRONT_LEGS[3]
    r, seeds = spec["uniform"], spec["seeds"]
    _, _, box = _k3_panel(fr, seeds, r)
    for i, s in enumerate(seeds):
        print(f"  K=3 fronts, seed {s}, common box: {len(box['adaptive'][i])} adaptive, {len(box['uniform'][i])} uniform "
              f"points; dominated: uniform by adaptive {100 * dominated_share(box['uniform'][i], box['adaptive'][i]):.1f} %, "
              f"adaptive by uniform {100 * dominated_share(box['adaptive'][i], box['uniform'][i]):.1f} %")
    d = C.DIGITS[3]
    fig = plt.figure(figsize=(7.2, 3.6))
    zlabels = []
    for i, (title, ss) in enumerate(((f"Seed {spec['single_seed']}", (spec["single_seed"],)),
                                     ("Mean of seeds " + ", ".join(str(v) for v in seeds), seeds))):
        env, ideal, _ = _k3_panel(fr, ss, r)
        ax = fig.add_axes([0.5 * i, 0.0, 0.47, 0.86], projection="3d")
        ax.set_box_aspect(None, zoom=0.92)
        for key in ("uniform", "adaptive"):
            pts = env[key]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=COL[key], s=2, alpha=0.25, depthshade=False,
                       linewidths=0, rasterized=True)
            _sheet(ax, pts, COL[key], 0.18, alpha=(0.35 if key == "adaptive" else 0.6))
        ax.scatter([ideal[0]], [ideal[1]], [ideal[2]], color="#2ca02c", s=22, marker="o", depthshade=False, zorder=10)
        ax.view_init(elev=24, azim=-55)
        ax.set_xlabel(f"$F_{d[0]}$", fontsize=10, labelpad=-2)
        ax.set_ylabel(f"$F_{d[1]}$", fontsize=10, labelpad=-2)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(f"$F_{d[2]}$", fontsize=10, labelpad=2, rotation=0)
        ax.tick_params(labelsize=6.5, pad=-1)
        ax.set_title(title, fontsize=9, y=0.97)
        zlabels.append(ax.zaxis.label)
    handles = [Patch(facecolor=COL["adaptive"], alpha=0.5, label=NAME["adaptive"]),
               Patch(facecolor=COL["uniform"], alpha=0.6, label=f"{NAME['uniform']} (r={r})"),
               Line2D([], [], marker="o", ls="", color="#2ca02c", ms=4, label="Ideal point")]
    leg = fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.905),
                     columnspacing=1.5, handlelength=1.8)
    # the tight bounding box of a 3-D axes misses the z label: include the z labels and the legend explicitly
    _save(fig, "mnist_front_k3", pdf_dpi=300, bbox_inches="tight", pad_inches=0.03, bbox_extra_artists=zlabels + [leg])


STEP_RULE_NAME = {"const": "Constant step", "bb": "Barzilai--Borwein"}
for _m in (1, 3, 10):
    STEP_RULE_NAME[f"adagrad_mult{_m}"] = rf"AdaGrad ($\alpha_{{\mathrm{{mult}}}}={_m}$)"
for _al, _als in (("0.001", r"10^{-3}"), ("0.0003", r"3\times10^{-4}"), ("0.0001", r"10^{-4}")):
    for _b2 in ("0.9", "0.99"):
        STEP_RULE_NAME[f"adam_alpha{_al}_beta2{_b2}"] = rf"Adam ($\alpha={_als}$, $\beta_2={_b2}$)"


def step_rules_k2():
    """Mean over the three seeds of the worst-case gradient norm, per step rule; four rules highlighted."""
    res = json.loads((RESULTS / "step_rules_k2.json").read_text())
    highlight = {"adam_alpha0.001_beta20.9": dict(color="#d62728", lw=2.6, ls="-", zorder=6, suffix=" (chosen)"),
                 "adam_alpha0.001_beta20.99": dict(color="#1f77b4", lw=1.9, ls="-", zorder=5, suffix=""),
                 "adagrad_mult10": dict(color="#2ca02c", lw=1.9, ls="-", zorder=5, suffix=""),
                 "const": dict(color="#444444", lw=1.9, ls="--", zorder=5, suffix=" (incumbent)")}
    FS = dict(label=16, tick=14, legend=13)
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    handles, labels = [], []
    for ax, key, xlabel in ((axs[0], "ck_grads", "Gradient Calls"), (axs[1], "ck_wall_mean", "Time (s)")):
        for tag in res["ranking"]:
            rule = res["rules"][tag]
            if tag in highlight:
                h = highlight[tag]
                ln, = ax.plot(rule[key], rule["gn_mean"], color=h["color"], lw=h["lw"], ls=h["ls"], zorder=h["zorder"])
                if key == "ck_grads":
                    handles.append(ln)
                    labels.append(STEP_RULE_NAME[tag] + h["suffix"])
            else:
                ln, = ax.plot(rule[key], rule["gn_mean"], color="#bbbbbb", lw=1.0, zorder=2)
                if key == "ck_grads" and "Other rules" not in labels:
                    handles.append(ln)
                    labels.append("Other rules")
        ax.set_yscale("log")
        ax.set_xlim(0, None)
        ax.set_xlabel(xlabel, fontsize=FS["label"])
        ax.tick_params(labelsize=FS["tick"])
        ax.grid(True, color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].set_ylabel(YLAB, fontsize=FS["label"])
    # legend columns fill first: rows read [chosen, Adam beta2 = 0.99, AdaGrad x10] / [constant, other rules]
    order = [next(i for i, lab in enumerate(labels) if w in lab and not (w == r"\beta_2=0.99" and "10^{-4}" in lab))
             for w in ("chosen", "Constant", r"\beta_2=0.99", "Other", "AdaGrad")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=3,
               fontsize=FS["legend"], frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.14, top=0.8, wspace=0.07)
    _save(fig, "mnist_step_rules_k2")


def main():
    worst_gn(2)
    worst_gn(3)
    front_k2()
    front_k3()
    step_rules_k2()


if __name__ == "__main__":
    main()
