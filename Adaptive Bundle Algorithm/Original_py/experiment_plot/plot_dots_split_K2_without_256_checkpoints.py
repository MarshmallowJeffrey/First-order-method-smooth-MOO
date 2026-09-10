"""K=2 inner-dial dots figure, split into a budget panel and a wall-clock panel
(user's Sep-9 report style: 'Adaptive CCP' curve, 'Uniform grid (r)' squares,
'SURF (N)' triangles, bare numbers as labels, dashed connectors per family).

Difference to the user's own rendering: every label whose text box had to be
moved away from its dot gets a thin leader line, so crowded clusters
(uniform r = 40/50/120/140 at ~19,050 grad evals, SURF N = 20..70 at
18,000-18,800) stay readable.  Dots come from worst_gn_dots_inner_all.json
(last decrease of the best-so-far staircase, y = final value); the CCP curve
from inner_range_gn_K2.json.  No training data is touched.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
from dots_label_placement import place_labels

HOME = Path("/Users/shirch/vscode101/.venv/First-order-method-smooth-MOO/Adaptive Bundle Algorithm/output/CCP/"
            "K2_mnist_pair_without_256_checkpoints/v2_campaign/main_mu0.001/adam_1e-3_b0.9")
BLUE, RED, ORANGE = "#3b7dc4", "#e3191c", "#ff7f0e"


def load(home, dots_json, series_json):
    dots = json.load(open(home / dots_json))
    ser = json.load(open(home / series_json))
    ad = [r for r in ser["runs"].values() if r["family"] == "adaptive"][0]
    return dots["rows"], ad


def panel(ax, rows, ad, key, xlabel, note=None, style="paper"):
    xkey = "ck_grads" if key == "grads" else "ck_cpu"
    # user request Sep 9 (late): wording as in the K = 3 dots figure
    names = ({"adaptive": "adaptive λ-bundle", "uniform": "uniform grid, one dot per r", "surf": "SURF, one dot per N"}
             if style == "paper" else {"adaptive": "Adaptive CCP", "uniform": "Uniform grid (r)", "surf": "SURF (N)"})
    ax.plot(ad[xkey], ad["inner_norm"], color=ORANGE, lw=2.6, zorder=3, label=names["adaptive"])
    fams = {"uniform": ("s", BLUE, names["uniform"]), "surf": ("^", RED, names["surf"])}
    labs = []
    cx, cy = list(ad[xkey]), list(ad["inner_norm"])      # labels must not cover the CCP curve either
    ax._connectors = [((cx[k], cy[k]), (cx[k + 1], cy[k + 1])) for k in range(len(cx) - 1)]
    for fam, (marker, color, lab) in fams.items():
        pts = sorted([(r[f"dot_{'grads' if key == 'grads' else 'cpu_s'}"], r["final_worst_gn"], r["param"])
                      for r in rows if r["family"] == fam])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, ls="--", lw=0.9, color=color, alpha=0.55, zorder=2)
        ax._connectors += [((xs[k], ys[k]), (xs[k + 1], ys[k + 1])) for k in range(len(xs) - 1)]
        ax.scatter(xs, ys, marker=marker, s=60, color=color, edgecolors="white", linewidths=0.8, zorder=4, label=lab)
        labs += [(x, y, str(p), color, marker) for x, y, p in pts]
    ax.set_yscale("log")
    xmax = max(max(ad[xkey]), max(l[0] for l in labs))
    ax.set_xlim(-0.03 * xmax, 1.14 * xmax)          # head-room on the right for labels of the last dots
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("best-so-far worst-case gradient norm" if style == "paper" else "Best-so-far worst gradient norm", fontsize=11)
    ax.grid(True, which="major", alpha=0.3, lw=0.6); ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9.5)
    ax.legend(fontsize=9.5, loc="upper right", frameon=True, framealpha=0.95, edgecolor="#dddddd")
    return labs


def clusters(ax, labs, thr_px):
    """single-linkage groups (same family) of dots closer than thr_px pixels"""
    pts = ax.transData.transform(np.array([(l[0], l[1]) for l in labs], dtype=float))
    parent = list(range(len(labs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            if labs[i][3] == labs[j][3] and np.hypot(*(pts[i] - pts[j])) < thr_px:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(labs)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def inset_zoom(fig, ax, labs, members, rect, fontsize):
    """zoomed inset for a crowded cluster (user request Sep 9: the five SURF
    dots at 18,000-18,900 grad evals are too close for labels); the main
    axes keep the dots, the inset carries the labels"""
    xs = [labs[i][0] for i in members]; ys = [labs[i][1] for i in members]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    padx = 0.45 * (x1 - x0) + 1e-9; ly0, ly1 = np.log10(y0), np.log10(y1); pady = 0.45 * (ly1 - ly0) + 0.01
    axins = ax.inset_axes(rect)
    axins.set_yscale("log")
    axins.set_xlim(x0 - padx, x1 + padx); axins.set_ylim(10 ** (ly0 - pady), 10 ** (ly1 + pady))
    for i in sorted(members, key=lambda i: labs[i][0]):
        x, y, _t, color, marker = labs[i]
        axins.scatter([x], [y], marker=marker, s=60, color=color, edgecolors="white", linewidths=0.8, zorder=4)
    order = sorted(members, key=lambda i: labs[i][0])
    axins.plot([labs[i][0] for i in order], [labs[i][1] for i in order], ls="--", lw=0.9, color=labs[order[0]][3], alpha=0.55, zorder=2)
    ticks = [t for t in (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15) if 10 ** (ly0 - pady) <= t <= 10 ** (ly1 + pady)]
    axins.yaxis.set_major_locator(FixedLocator(ticks)); axins.yaxis.set_minor_locator(NullLocator())
    fmt = ScalarFormatter(); fmt.set_scientific(False); axins.yaxis.set_major_formatter(fmt)
    axins.tick_params(labelsize=8); axins.grid(True, alpha=0.3, lw=0.5)
    for side in ("top", "right"):
        axins.spines[side].set_visible(False)
    axins.set_title("zoom", fontsize=8.5, color="#555555", pad=3)
    ax.indicate_inset_zoom(axins, edgecolor="#888888", alpha=0.9, lw=0.8)
    fig.canvas.draw()
    place_labels(fig, axins, [labs[i][:4] for i in members], fontsize=fontsize)
    return axins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(HOME))
    ap.add_argument("--dots-json", default="worst_gn_dots_inner_all.json")
    ap.add_argument("--series-json", default="inner_range_gn_K2.json")
    ap.add_argument("--name", default="worst_gn_dots_inner_split")
    ap.add_argument("--omit-r", default="", help="comma-separated uniform r to leave out (dots that coincide with another r)")
    ap.add_argument("--omit-on", choices=["both", "budget", "time"], default="both",
                    help="apply --omit-r / --omit-N (and draw --note) on these panels only (user decision Sep 9: budget only)")
    ap.add_argument("--omit-N", default="", help="comma-separated SURF N to leave out")
    ap.add_argument("--label-size", type=float, default=8.5)
    ap.add_argument("--note", default="", help="English note drawn under the plot")
    ap.add_argument("--prefer", default="", help="preferred label positions, e.g. 'surf:120=lower-left;uniform:90=below'")
    ap.add_argument("--legend-style", choices=["paper", "ccp"], default="paper",
                    help="'paper' (default): wording of the K = 3 dots figure; 'ccp': the user's Sep-9 report wording")
    ap.add_argument("--title", default=None, help="one title centred over the figure (user request Sep 10)")
    ap.add_argument("--figsize", default="12.6,5.6", help="side layout only: figure size in inches, e.g. '15,6.2'")
    ap.add_argument("--layout", choices=["separate", "side"], default="separate",
                    help="'side' (paper, Sep 10): both panels side by side in ONE figure like the K = 3 dots figure, "
                         "written as <name>_side.png; 'separate': one file per panel (report)")

    a = ap.parse_args()
    home = Path(a.home)
    rows_all, ad = load(home, a.dots_json, a.series_json)
    omit = {"uniform": {int(v) for v in a.omit_r.split(",") if v.strip()}, "surf": {int(v) for v in a.omit_N.split(",") if v.strip()}}
    rows_omit = [r for r in rows_all if r["param"] not in omit[r["family"]]]
    outs = []
    xl = ((("grads", "total gradient evaluations", "budget"), ("cpu", "CPU seconds", "time")) if a.legend_style == "paper"
          else (("grads", "Gradient evaluations", "budget"), ("cpu", "Wall-clock seconds", "time")))
    prefer = {}
    for item in (a.prefer.split(";") if a.prefer else []):
        fam, rest = item.split(":"); num, pos = rest.split("=")
        prefer[({"uniform": BLUE, "surf": RED}[fam.strip()], num.strip())] = pos.strip()
    if a.layout == "side":
        fw, fh = (float(v) for v in a.figsize.split(","))
        fig, axes = plt.subplots(1, 2, figsize=(fw, fh), dpi=200)
        labs_all = []
        for ax, (key, xlabel, tag) in zip(axes, xl):
            omit_here = a.omit_on in ("both", tag)
            labs_all.append((ax, panel(ax, rows_omit if omit_here else rows_all, ad, key, xlabel, style=a.legend_style)))
        if a.note:
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            note = a.note
            if a.omit_on == "budget":
                note = "Left panel: " + note[0].lower() + note[1:]
            fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=9.5, color="#333333")
        else:
            fig.tight_layout()
        if a.title:
            fig.suptitle(a.title, fontsize=11, y=0.975); fig.subplots_adjust(top=0.90)
        fig.canvas.draw()
        for ax, labs in labs_all:
            info = place_labels(fig, ax, [l[:4] for l in labs], fontsize=a.label_size, segments=ax._connectors,
                                prefer=prefer, overlap_px=14.0)
            print(f"  panel: {len(labs)} dots, {info['n_leader']} labels with leader lines, merged {info['merged']}")
        out = home / f"{a.name}_side.png"
        fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".svg")); plt.close(fig)
        print(f"{out.name} written")
        return [out]
    for key, xlabel, tag in xl:
        fig, ax = plt.subplots(figsize=(11.9, 6.0 if a.title else 5.5), dpi=200)   # the title takes ~0.5 in
        omit_here = a.omit_on in ("both", tag)
        rows = rows_omit if omit_here else rows_all
        labs = panel(ax, rows, ad, key, xlabel, style=a.legend_style)
        if a.note and omit_here:         # user request Sep 9: the note goes UNDER the plot, plain text
            fig.tight_layout(rect=[0, 0.045, 1, 1])
            fig.text(0.5, 0.012, a.note, ha="center", va="bottom", fontsize=9.5, color="#333333")
        else:
            fig.tight_layout()
        if a.title:
            fig.suptitle(a.title, fontsize=11, y=0.975); fig.subplots_adjust(top=0.91)
        fig.canvas.draw()
        info = place_labels(fig, ax, [l[:4] for l in labs], fontsize=a.label_size, segments=ax._connectors, prefer=prefer,
                            overlap_px=14.0)      # r = 50/120/140 on the time axis: 9-13 px apart, one marker blob
        out = home / f"{a.name}_{tag}.png"
        fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".svg")); plt.close(fig)
        print(f"{out.name}: {len(labs)} dots, {info['n_leader']} labels with leader lines, merged {info['merged']}")
        outs.append(out)
    return outs


if __name__ == "__main__":
    main()
