"""plot_k3_fronts_dots_without_256_checkpoints.py — K = 3 Pareto-front
sheets (paper Figure 6) for the dots campaign, in the TRAIN loss space
or the TEST cross-entropy space (user request Sep 9 2026).

Same construction as plot_k3_adam_main_without_256_checkpoints.py:
non-dominated set of every delivered point, window <= ln 3 on every
axis, lower envelope on a log (F1, F2) grid, Delaunay in (F1, F2),
bridging triangles longer than ``edge_max`` dropped, three fixed views.
Differences: the home directory is a parameter (default: the B = 100,000
dots home); the uniform representative is the r with the LARGEST EXACT
hypervolume in the TRAINING loss space among the uniform_r*_seed41 runs
present (report rule, Sep 9 2026) and the SAME r is used for the train
and the test figure; ``--space test`` uses grams.npz["test_ce"] (every delivered theta scored on all
t10k rows of the three digits, plain mean cross-entropy per digit, no
ridge term) instead of the penalised training objectives ["fvals"].
Also writes the EXACT hypervolume (reference point (ln 3, ln 3, ln 3),
full reference box, non-dominated set of all delivered points, sweep
algorithm ``_hv_3d`` of plot_K3_mnist_triple) of every leg in BOTH spaces
to <out>/<stem>_hv.json.  The display window of the sheets (<= ln 3, log
grid binning) is for drawing only and does not enter the HV.

Usage:
    python plot_k3_fronts_dots_without_256_checkpoints.py                       # train, dots home
    python plot_k3_fronts_dots_without_256_checkpoints.py --space test
    python plot_k3_fronts_dots_without_256_checkpoints.py --home <dir> --out-dir <dir> --suffix _x
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
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _layout  # noqa: F401
from plot_K3_mnist_triple_without_256_checkpoints import (  # noqa: E402
    LN3,
    VIEWS3,
    _hv_3d,
    _nondominated_kd,
)
from plot_k3_adam_main_without_256_checkpoints import _envelope  # noqa: E402
from run_k3_stepper_campaign_without_256_checkpoints import (  # noqa: E402
    K3_HOME,
    TRIPLE,
)

DEFAULT_HOME = K3_HOME / "dots_B100000" / "adam_1e-3_b0.9"
SEED = 41
COLORS = {"uniform": "#2ca02c", "adaptive": "#ff7f0e"}


def _final_norm(sm):
    v = sm.get("final_audit")
    if v is None:
        v = sm["audited_gn_history"][-1]
    return float(np.sqrt(max(float(v), 0.0)))


def _legs(home):
    """{name: (summary, points)} for adaptive and every uniform r present."""
    out = {}
    for d in sorted(home.iterdir()):
        if not d.is_dir() or not d.name.endswith(f"_seed{SEED}"):
            continue
        if not (d / "summary.json").exists() or not (d / "grams.npz").exists():
            continue
        if d.name.startswith("uniform_r") or d.name.startswith("adaptive_ccp"):
            out[d.name] = (json.loads((d / "summary.json").read_text()),
                           np.load(d / "grams.npz"))
    return out


def hypervolume_3d(front, zref):
    """EXACT hypervolume (minimisation) of a discrete point set with respect
    to ``zref``: sweep of the third coordinate with the exact 2-D staircase
    area of the accumulated (F1, F2) projections on every slab — the
    project's ``_hv_3d``.  Replaces the former 400x400 mid-point grid
    approximation (review of Sep 9 2026: its error re-ordered the uniform
    resolutions).  Points outside the box below zref contribute nothing."""
    return float(_hv_3d(np.asarray(front, dtype=float), tuple(float(v) for v in zref)))


def sheets_figure(series, title, out_png, edge_max=0.45):
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
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.05,
                        wspace=0.10)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


SPLIT_COLORS = {"uniform": "#1f77b4", "adaptive": "#d62728"}


def sheets_figure_split(series, title, out_png, edge_max=0.45, footnote="", clip_pct=100.0):
    """Same sheet construction as sheets_figure, but one row per leg (adaptive
    on top, uniform below), three views per row, both rows on the same axes
    (shared limits; clip_pct < 100 zooms to that percentile of the envelope
    coordinates so the knee fills the box; points beyond are not drawn)."""
    env_series = [(lbl, col, _envelope(fr)) for lbl, col, fr in series]
    allE = np.vstack([e for _l, _c, e in env_series])
    lo, hi = allE.min(axis=0), np.percentile(allE, clip_pct, axis=0)
    pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
    if clip_pct < 100.0:
        env_series = [(lbl, col, e[((e >= lo) & (e <= hi)).all(axis=1)]) for lbl, col, e in env_series]
    fig = plt.figure(figsize=(15, 9.6))
    rows = list(reversed(env_series))           # adaptive (last in series) on top
    for row, (lbl, col, env) in enumerate(rows):
        for p, (elev, azim) in enumerate(VIEWS3):
            ax = fig.add_subplot(2, 3, row * 3 + p + 1, projection="3d")
            ax.scatter(env[:, 0], env[:, 1], env[:, 2], color=col, s=7, alpha=0.9, depthshade=False)
            x, y, z = env[:, 0], env[:, 1], env[:, 2]
            key = np.round(x, 8) + 1j * np.round(y, 8)
            _, uniq = np.unique(key, return_index=True)
            xu, yu, zu = x[uniq], y[uniq], z[uniq]
            if xu.size >= 4:
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
                                    alpha=0.5, linewidth=0.2, edgecolor=col, shade=True)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{lbl} — View {p + 1}", fontsize=10)
            ax.set_xlabel(f"digit-{TRIPLE[0]} loss", fontsize=8, labelpad=2)
            ax.set_ylabel(f"digit-{TRIPLE[1]} loss", fontsize=8, labelpad=2)
            ax.set_zlabel(f"digit-{TRIPLE[2]} loss", fontsize=8, labelpad=2)
            ax.tick_params(labelsize=7, pad=1)
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.legend(handles=[Patch(facecolor=c_, alpha=0.6, label=f"{l_} ({len(e_)} envelope pts)")
                        for l_, c_, e_ in rows],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.045), frameon=True)
    if footnote:
        fig.text(0.5, 0.015, footnote, ha="center", fontsize=8, color="#333333")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.10, wspace=0.08, hspace=0.12)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-pct", type=float, default=100.0,
                    help="split figure only: zoom both rows to this percentile of the envelope coordinates")
    ap.add_argument("--split", action="store_true",
                    help="one row per leg (adaptive on top, uniform below), shared axes; "
                         "writes <stem>_split.png")
    ap.add_argument("--home", default=str(DEFAULT_HOME))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--space", choices=["train", "test"], default="train")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--rep-r", type=int, default=None,
                    help="force the uniform representative to this r (overrides --pick)")
    ap.add_argument("--pick", choices=["hv-train", "gn"], default="hv-train",
                    help="uniform representative: 'hv-train' (default, report rule) = largest EXACT "
                         "hypervolume in the TRAINING loss space, used for BOTH the train and the test "
                         "figure; 'gn' = lowest final worst-case gradient norm (legacy)")
    ap.add_argument("--budget-label", default=None,
                    help="text for the title, e.g. 'B=100,000'")
    a = ap.parse_args()
    home = Path(a.home)
    out_dir = Path(a.out_dir) if a.out_dir else home
    legs = _legs(home)
    if "adaptive_ccp_seed41" not in legs:
        raise SystemExit(f"no adaptive run under {home}")
    uni = {n: v for n, v in legs.items() if n.startswith("uniform_r")}
    if not uni:
        raise SystemExit(f"no uniform runs under {home}")
    key = "fvals" if a.space == "train" else "test_ce"
    zref = (LN3, LN3, LN3)
    # exact HV in BOTH spaces for every leg (from the raw fvals / test_ce
    # of grams.npz: non-dominated set, then the exact sweep); the
    # representative is chosen from the TRAIN values whatever space is drawn
    hv = {}
    for name, (sm, npz) in legs.items():
        rec = {"final_worst_gn": _final_norm(sm)}
        for sp, k in (("train", "fvals"), ("test", "test_ce")):
            F = np.asarray(npz[k], dtype=float)
            F = F[np.isfinite(F).all(axis=1)]
            fr = F[_nondominated_kd(F)]
            rec[f"hv_{sp}"] = hypervolume_3d(fr, zref)
            rec[f"delivered_points_{sp}"] = int(len(F))
            rec[f"nondominated_points_{sp}"] = int(len(fr))
            rec[f"nondominated_in_window_{sp}"] = int((fr <= LN3).all(axis=1).sum())
        rec["hypervolume"] = rec[f"hv_{a.space}"]          # value in the drawn space
        rec["nondominated_points"] = rec[f"nondominated_points_{a.space}"]
        rec["nondominated_in_window"] = rec[f"nondominated_in_window_{a.space}"]
        hv[name] = rec
    if a.rep_r is not None:
        best, rule = f"uniform_r{a.rep_r}_seed{SEED}", f"forced r = {a.rep_r}"
    elif a.pick == "hv-train":
        best = max(uni, key=lambda n: hv[n]["hv_train"])
        rule = "largest exact hypervolume in the training loss space (same r for train and test)"
    else:
        best = min(uni, key=lambda n: _final_norm(uni[n][0]))
        rule = "lowest final worst-case gradient norm"
    best_r = int(best[len("uniform_r"):-len(f"_seed{SEED}")])
    series = []
    for lbl, col, name in ((f"uniform r={best_r}", COLORS["uniform"], best),
                           ("adaptive CCP", COLORS["adaptive"], "adaptive_ccp_seed41")):
        F = np.asarray(legs[name][1][key], dtype=float)
        fr = F[_nondominated_kd(F)]
        series.append((lbl, col, fr[(fr <= LN3).all(axis=1)]))
    blabel = a.budget_label or ""
    space_txt = "training loss space" if a.space == "train" else "test cross-entropy space"
    title = (f"Pareto frontier sheets, MNIST {TRIPLE[0]}/{TRIPLE[1]}/{TRIPLE[2]}, "
             f"adam core{', ' + blabel if blabel else ''}: adaptive CCP vs uniform "
             f"r={best_r} ({space_txt})")
    stem = "pareto_front_adam_3d" if a.space == "train" else "pareto_front_adam_3d_test"
    if a.split:
        out_png = out_dir / f"{stem}_split{'' if a.clip_pct >= 100 else '_zoom'}{a.suffix}.png"
        series_split = [(f"Uniform grid (r = {best_r})", SPLIT_COLORS["uniform"], series[0][2]),
                        ("Adaptive bundle", SPLIT_COLORS["adaptive"], series[1][2])]
        sheets_figure_split(series_split,
                            f"MNIST {TRIPLE[0]}/{TRIPLE[1]}/{TRIPLE[2]} — {space_txt} ({blabel})",
                            out_png,
                            footnote="Points are the non-dominated delivered points (window ≤ ln 3) binned to a lower envelope on a log grid; "
                                     "sheets are Delaunay triangulations of that envelope (bridging triangles longer than 0.45 dropped). "
                                     "Both rows share the same axes" + (f", zoomed to the {a.clip_pct:g}th percentile of the envelope coordinates." if a.clip_pct < 100 else "."),
                            clip_pct=a.clip_pct)
    else:
        out_png = out_dir / f"{stem}{a.suffix}.png"
        sheets_figure(series, title, out_png)
    (out_dir / f"{stem}_hv{a.suffix}.json").write_text(json.dumps(
        {"home": str(home), "space": a.space, "zref": list(zref),
         "hv_method": "exact sweep over the third coordinate with 2-D staircase slabs (_hv_3d); "
                      "non-dominated set of all delivered points; reference box below zref",
         "display_window": "figure: coordinates <= ln 3 (envelope binning for the sheets only); "
                           "HV: full reference box, independent of the display",
         "representative_rule": rule, "best_uniform": best, "legs": hv}, indent=2))
    print(f"figure -> {out_png}")
    print(f"  representative: {best} ({rule})")
    for name, r in sorted(hv.items(), key=lambda kv: -kv[1]["hv_train"]):
        print(f"  {name:22s} final GN={r['final_worst_gn']:.3e}  HV train={r['hv_train']:.13f}  "
              f"HV test={r['hv_test']:.13f}  ND train/test={r['nondominated_points_train']}/{r['nondominated_points_test']}")


if __name__ == "__main__":
    main()
