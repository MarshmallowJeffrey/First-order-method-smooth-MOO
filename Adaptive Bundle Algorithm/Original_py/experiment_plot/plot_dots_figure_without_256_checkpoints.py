"""plot_dots_figure_without_256_checkpoints.py — the new Figure 3 (K = 2)
and Figure 5 (K = 3) of the paper ("dots" figure, PI review Sep 7 2026):
the best-so-far worst-case gradient norm of the adaptive λ-bundle as a
curve, and ONE DOT per baseline resolution — uniform grid r, SURF slot
count N — placed where that run stopped improving.  Two panels: total
gradient evaluations (left) and CPU seconds (right), both log-log by
default (``--linear-x`` for a linear abscissa).

Dot rule (user sign-off Sep 9): y = the run's final worst GN at the
budget B (the Table 6 / Table 8 number); x = the first checkpoint at
which the best-so-far value is within ``tol`` (5 %) of that final value,
i.e. the point after which the rest of the budget buys less than 5 %.
A run whose value still improves by more than ``tol`` over the LAST
QUARTER of the budget never flattened: it is drawn with an open marker
and flagged "still improving" in the table.

Reads every <home>/uniform_r*_seed<seed>, <home>/surf_N*_seed<seed> and
<home>/adaptive_ccp_seed<seed> summary.json (K = 2 and K = 3 summaries
both work: the norm-scale history is used when present, else the square
root of the squared-scale history).  Writes <home>/<name>.png,
<home>/<name>.json and <home>/<name>.md (the table numbers).

``--dot final`` (user request Sep 9) places every baseline dot at the END
of its run instead — x = the total gradient evaluations / CPU seconds
the run consumed, y = its final worst GN — so all dots of a family stack
at the budget; labels are then de-overlapped with leader lines.

Usage:
    python plot_dots_figure_without_256_checkpoints.py --home <core_home>
    python plot_dots_figure_without_256_checkpoints.py --home <core_home> --dot final --name worst_gn_dots_final
    python plot_dots_figure_without_256_checkpoints.py --home <core_home> --linear-x --name worst_gn_dots_linx
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

from dots_label_placement import place_labels  # noqa: E402

FAMILIES = (   # key, dir prefix, marker, colour, symbol in labels
    ("uniform", "uniform_r", "s", "#377eb8", "r"),
    ("surf", "surf_N", "^", "#e41a1c", "N"),
)
ADAPTIVE_COLOR = "#ff7f00"


def _history(sm):
    if "audited_gn_norm_history" in sm:
        return np.asarray(sm["audited_gn_norm_history"], dtype=float)
    return np.sqrt(np.maximum(np.asarray(sm["audited_gn_history"],
                                         dtype=float), 0.0))


def load_runs_from_json(path: Path, key: str):
    """runs dict from k2_inner_range_gn_without_256_checkpoints.py's JSON:
    y = the recomputed series ``key`` (e.g. 'inner_norm' or 'full_norm')."""
    J = json.loads(Path(path).read_text())
    runs = {"uniform": {}, "surf": {}, "adaptive": None}
    for name, rec in J["runs"].items():
        r = {"dir": name, "g": np.asarray(rec["ck_grads"], dtype=float),
             "c": np.asarray(rec["ck_cpu"], dtype=float), "y": np.asarray(rec[key], dtype=float),
             "wall": float(rec["ck_cpu"][-1])}
        if rec["family"] == "adaptive":
            runs["adaptive"] = r
        else:
            runs[rec["family"]][int(rec["param"])] = r
    return runs


def load_runs(home: Path, seed: int):
    runs = {"uniform": {}, "surf": {}, "adaptive": None}
    suffix = f"_seed{seed}"
    for d in sorted(home.iterdir()):
        if not d.is_dir() or not d.name.endswith(suffix):
            continue
        p = d / "summary.json"
        if not p.exists():
            continue
        sm = json.loads(p.read_text())
        rec = {"dir": d.name,
               "g": np.asarray(sm["ck_grads"], dtype=float),
               "c": np.asarray(sm["ck_cpu"], dtype=float),
               "y": _history(sm),
               "wall": float(sm.get("wall_seconds", float("nan")))}
        if d.name.startswith("uniform_r"):
            runs["uniform"][int(d.name[len("uniform_r"):-len(suffix)])] = rec
        elif d.name.startswith("surf_N"):
            runs["surf"][int(d.name[len("surf_N"):-len(suffix)])] = rec
        elif d.name.startswith("adaptive_ccp"):
            runs["adaptive"] = rec
    return runs


def dot_of(rec, tol, mode="plateau"):
    """mode 'plateau': first checkpoint within tol of the final value;
    'final': the end of the run; 'auto' (user rule Sep 9): the plateau
    onset if the run flattened, else the end of the run — a run counts as
    flattened when its improvement over the last quarter of the budget is
    at most tol."""
    g, c, y = rec["g"], rec["c"], rec["y"]
    final = float(y[-1])
    j = int(np.searchsorted(g, 0.75 * g[-1], side="right") - 1)
    drift = float(y[max(j, 0)] / final - 1.0)
    if mode == "final" or (mode == "auto" and drift > tol):
        i = len(y) - 1
    elif mode == "laststep":
        # user rule Sep 9: the last step of the staircase — the last
        # checkpoint at which the best-so-far value still decreased
        # (relative drop > 1e-6 to ignore audit noise); flat afterwards
        drops = np.where(y[1:] < y[:-1] * (1.0 - 1e-6))[0]
        i = int(drops[-1] + 1) if len(drops) else 0
    else:
        i = int(np.argmax(y <= final * (1.0 + tol)))
    return {"final": final, "x": float(g[i]), "cpu": float(c[i]),
            "ck_index": i, "drift_last_quarter": drift,
            "still_improving": bool(drift > tol)}


def value_at(rec, x, key="g"):
    xs = rec[key]
    i = int(np.searchsorted(xs, x, side="right") - 1)
    return float(rec["y"][i]) if i >= 0 else float("nan")


def budget_to_reach(rec, level, key="g"):
    idx = np.where(rec["y"] <= level)[0]
    return float(rec[key][idx[0]]) if len(idx) else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--name", default="worst_gn_dots")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--linear-x", action="store_true")
    ap.add_argument("--dot", choices=["plateau", "final", "auto", "laststep"], default="plateau")
    ap.add_argument("--connect", action="store_true",
                    help="clean style: join each family's dots with a dashed "
                         "line in order of increasing x (not of r / N)")
    ap.add_argument("--series-json", default=None,
                    help="read the curves from this inner_range_gn_K2.json instead of summary.json")
    ap.add_argument("--series-key", default="inner_norm",
                    help="series name inside --series-json (inner_norm or full_norm)")
    ap.add_argument("--numbers-only", action="store_true",
                    help="label dots with the bare number instead of r=.. / N=..")
    ap.add_argument("--uniform-rs", default=None,
                    help="comma-separated subset of r to draw (default: all present)")
    ap.add_argument("--surf-Ns", default=None,
                    help="comma-separated subset of N to draw (default: all present)")
    ap.add_argument("--clean", action="store_true",
                    help="paper style (user request Sep 9): no connector lines, "
                         "no open/filled distinction, labels with a white halo, "
                         "no top/right spines")
    ap.add_argument("--legend-style", choices=["paper", "ccp"], default="paper",
                    help="'ccp': wording of the user's Sep-9 report (Adaptive CCP / Uniform grid (r) / SURF (N), "
                         "'Gradient evaluations', 'Wall-clock seconds', 'Best-so-far worst gradient norm')")
    ap.add_argument("--label-size", type=float, default=8.5, help="font size of the dot labels in --clean mode")
    ap.add_argument("--title", default=None, help="one title centred over both panels (user request Sep 10)")
    a = ap.parse_args()
    home = Path(a.home)
    runs = load_runs_from_json(a.series_json, a.series_key) if a.series_json else load_runs(home, a.seed)
    if runs["adaptive"] is None:
        raise SystemExit(f"no adaptive_ccp_seed{a.seed} run under {home}")
    ad = runs["adaptive"]
    for fam, arg in (("uniform", a.uniform_rs), ("surf", a.surf_Ns)):
        if arg:
            keep = set() if arg.strip().lower() == "none" else {int(v) for v in arg.split(",")}
            runs[fam] = {p: r for p, r in runs[fam].items() if p in keep}

    rows, dots = [], {"uniform": {}, "surf": {}}
    for fam, _pre, _m, _col, sym in FAMILIES:
        for p in sorted(runs[fam]):
            d = dot_of(runs[fam][p], a.tol, a.dot)
            dots[fam][p] = d
            a_at = value_at(ad, d["x"])
            a_reach = budget_to_reach(ad, d["final"])
            rows.append({
                "family": fam, "param": p, "symbol": sym,
                "final_worst_gn": d["final"],
                "dot_grads": d["x"], "dot_cpu_s": d["cpu"],
                "drift_last_quarter": d["drift_last_quarter"],
                "still_improving": d["still_improving"],
                "adaptive_at_dot": a_at,
                "ratio_equal_budget": d["final"] / a_at if a_at > 0 else float("nan"),
                "adaptive_budget_to_reach": a_reach,
                "ratio_equal_gn": d["x"] / a_reach if a_reach > 0 else float("nan"),
            })

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=200)
    deferred = []   # (ax, fam, colour, [(x, y, text, k)]) for --dot final
    clean_labels = {}   # ax -> [(x, y, text, colour)] for --clean
    connectors = {}     # ax -> dashed connector segments (labels must not cover them)
    ccp = a.legend_style == "ccp"
    xlabels = ((("x", "Gradient evaluations"), ("cpu", "Wall-clock seconds")) if ccp else
               (("x", "total gradient evaluations"), ("cpu", "CPU seconds")))
    for ax, (key, xlabel) in zip(axes, xlabels):
        xa = ad["g"] if key == "x" else ad["c"]
        # the x = 0 checkpoint is the common initial point of every run; it
        # is drawn on linear axes (all curves start together) and dropped
        # only on log axes, where x = 0 cannot be shown
        keep = np.ones(len(xa), dtype=bool) if a.linear_x else xa > 0.0
        ax.plot(xa[keep], ad["y"][keep], color=ADAPTIVE_COLOR,
                lw=2.3 if a.clean else 2.0, zorder=3, label="Adaptive CCP" if ccp else "adaptive λ-bundle",
                solid_capstyle="round")
        xs_c, ys_c = xa[keep], ad["y"][keep]
        connectors.setdefault(ax, []).extend([((xs_c[k], ys_c[k]), (xs_c[k + 1], ys_c[k + 1])) for k in range(len(xs_c) - 1)])
        y_lo, y_hi = float(ad["y"][keep].min()), float(ad["y"][keep].max())
        for fam, _pre, marker, color, sym in FAMILIES:
            params = sorted(dots[fam])
            if not params:
                continue
            xs = [dots[fam][p][key] for p in params]
            ys = [dots[fam][p]["final"] for p in params]
            if not a.clean:
                ax.plot(xs, ys, ls=":", lw=0.9, color=color, zorder=2)
            elif a.connect:
                order = np.argsort(xs)
                ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], ls="--",
                        lw=1.1, color=color, alpha=0.8, zorder=2)
                xo, yo = np.asarray(xs)[order], np.asarray(ys)[order]
                connectors.setdefault(ax, []).extend([((xo[k], yo[k]), (xo[k + 1], yo[k + 1])) for k in range(len(xo) - 1)])
            lab = ({"uniform": "Uniform grid (r)", "surf": "SURF (N)"} if ccp else
                   {"uniform": "uniform grid, one dot per r", "surf": "SURF, one dot per N"})[fam]
            items = []
            for k, p in enumerate(params):
                d = dots[fam][p]
                if a.clean:
                    ax.scatter([d[key]], [d["final"]], marker=marker, s=70,
                               facecolors=color, edgecolors="white",
                               linewidths=0.9, zorder=4,
                               label=lab if k == 0 else None)
                else:
                    ax.scatter([d[key]], [d["final"]], marker=marker, s=52,
                               facecolors="white" if d["still_improving"] else color,
                               edgecolors=color, linewidths=1.4, zorder=4,
                               label=lab if k == 0 else None)
                text = f"{p}" if a.numbers_only else f"{sym}={p}"
                if a.dot == "final":
                    items.append((d[key], d["final"], text, k))
                    continue
                if a.clean:
                    clean_labels.setdefault(ax, []).append(
                        (d[key], d["final"], text, color))
                    continue
                offs = ((6, 7), (6, -13), (6, 19), (6, -25))
                ax.annotate(text, (d[key], d["final"]),
                            textcoords="offset points",
                            xytext=offs[k % 4], fontsize=7 if a.clean else 6.5,
                            color=color, zorder=5,
                            bbox=(dict(boxstyle="round,pad=0.12", fc="white",
                                       ec="none", alpha=0.85) if a.clean else None))
            if items:
                deferred.append((ax, fam, color, items))
        for fam in dots:
            for d in dots[fam].values():
                y_lo, y_hi = min(y_lo, d["final"]), max(y_hi, d["final"])
        ax.set_yscale("log")
        ax.set_ylim(0.6 * y_lo, 1.6 * y_hi)
        if not a.linear_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel("Best-so-far worst gradient norm" if ccp else "best-so-far worst-case gradient norm", fontsize=10.5)
        if a.clean:
            ax.grid(True, which="major", alpha=0.3, lw=0.6)
            ax.grid(False, which="minor")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=9)
            ax.legend(fontsize=8.5, loc="upper right" if ccp else "lower left", frameon=True,
                      framealpha=0.9, edgecolor="#dddddd")
        else:
            ax.grid(alpha=0.25, which="both", lw=0.5)
            ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    if a.title:
        fig.suptitle(a.title, fontsize=11, y=0.975)
        fig.subplots_adjust(top=0.90)
    if clean_labels:
        # user request Sep 9: numbers right next to their dots; overlapping
        # dots share one label; displaced or crowded labels get leader lines
        fig.canvas.draw()
        for ax, labs in clean_labels.items():
            info = place_labels(fig, ax, labs, fontsize=a.label_size, segments=connectors.get(ax, []))
            if info["merged"]:
                print(f"labels merged (dots overlap on the page): {info['merged']}")
    if deferred:
        # stacked labels with leader lines: dots of one family share the
        # abscissa in --dot final mode, so spread their labels vertically
        # (axes pixels) with a minimum gap and draw a thin line to each dot
        fig.canvas.draw()
        for ax, fam, color, items in deferred:
            org = ax.transAxes.transform((0.0, 0.0))
            pts = ax.transData.transform(np.array([(x, y) for x, y, _t, _k in items]))
            pts = pts - org
            order = np.argsort(pts[:, 1])
            gap = 13.0
            ys = pts[order, 1].astype(float).copy()
            for j in range(1, len(ys)):          # push up
                ys[j] = max(ys[j], ys[j - 1] + gap)
            shift = (ys.mean() - pts[order, 1].mean())
            ys -= shift                            # re-centre on the dots
            xlab = pts[:, 0].max() - 62.0 if fam == "uniform" else pts[:, 0].max() - 62.0
            side = -1.0
            for j, idx in enumerate(order):
                x_d, y_d, text, _k = items[idx]
                ax.annotate(text, xy=(x_d, y_d), xycoords="data",
                            xytext=(pts[idx, 0] + side * 70.0, ys[j]),
                            textcoords="axes pixels", fontsize=6.5,
                            color=color, ha="right", va="center",
                            arrowprops=dict(arrowstyle="-", lw=0.4,
                                            color=color, shrinkA=0, shrinkB=2))
    out_png = home / f"{a.name}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    report = {"home": str(home), "seed": a.seed, "tol": a.tol,
              "adaptive": {"dir": ad["dir"], "final_worst_gn": float(ad["y"][-1]),
                           "budget": float(ad["g"][-1]),
                           "cpu_s": float(ad["c"][-1])},
              "rows": rows}
    (home / f"{a.name}.json").write_text(json.dumps(report, indent=2))
    md = ["| family | r or N | final worst GN | dot: grad evals | dot: CPU s "
          "| improvement in last quarter | still improving | adaptive at same budget "
          "| ratio (equal budget) | adaptive budget to reach the same GN | ratio (equal GN) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['family']} | {r['symbol']}={r['param']} | {r['final_worst_gn']:.3e} "
                  f"| {r['dot_grads']:,.0f} | {r['dot_cpu_s']:.0f} "
                  f"| {100 * r['drift_last_quarter']:.1f}% | {'yes' if r['still_improving'] else 'no'} "
                  f"| {r['adaptive_at_dot']:.3e} | {r['ratio_equal_budget']:.1f}x "
                  f"| {r['adaptive_budget_to_reach']:,.0f} | {r['ratio_equal_gn']:.1f}x |")
    md.append("")
    md.append(f"adaptive λ-bundle: final {report['adaptive']['final_worst_gn']:.3e} "
              f"at B = {report['adaptive']['budget']:,.0f} "
              f"({report['adaptive']['cpu_s']:.0f} CPU s)")
    (home / f"{a.name}.md").write_text("\n".join(md) + "\n")
    print(f"figure -> {out_png}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
