"""plot_k3_figure6_sheets_without_256_checkpoints.py — the K = 3 report's
Figure 6 (two rows x three views of the non-dominated front sheets, adaptive
on top, uniform r below), in the report style: orange / green, subplot
titles '<method> — View k', legend without counts, fixed axes [0, 0.25]^3,
no footnote.  Same sheet construction as plot_k3_fronts_dots (non-dominated
set of every delivered point, window <= ln 3, lower envelope on a log grid,
Delaunay in (F1, F2), bridging triangles longer than edge_max dropped).

User request Sep 9 2026 (late): the test figure is dropped from the
report, so the title no longer says 'training'.

    python plot_k3_figure6_sheets_without_256_checkpoints.py --rep-r 40 \
        --title "MNIST 4/7/9 — loss space (B = 100,000)"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch

from plot_k3_fronts_dots_without_256_checkpoints import (DEFAULT_HOME, SEED, TRIPLE, LN3, VIEWS3,
                                                          _legs, _nondominated_kd, _envelope)

COL = {"adaptive": "#ff7f0e", "uniform": "#2ca02c"}


def draw(series, title, out_png, edge_max=0.45, lim=(0.0, 0.25)):
    fig = plt.figure(figsize=(15, 9.6))
    for row, (lbl, col, env) in enumerate(series):
        for p, (elev, azim) in enumerate(VIEWS3):
            ax = fig.add_subplot(2, 3, row * 3 + p + 1, projection="3d")
            ax.scatter(env[:, 0], env[:, 1], env[:, 2], color=col, s=7, alpha=0.9, depthshade=False)
            x, y, z = env[:, 0], env[:, 1], env[:, 2]
            key = np.round(x, 8) + 1j * np.round(y, 8)
            _, uniq = np.unique(key, return_index=True)
            xu, yu, zu = x[uniq], y[uniq], z[uniq]
            if xu.size >= 4:
                tri = mtri.Triangulation(xu, yu); t = tri.triangles
                P = np.stack([xu, yu, zu], axis=1); a, b, c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
                elen = np.maximum.reduce([np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(a - c, axis=1)])
                keep = elen <= edge_max
                if keep.any():
                    ax.plot_trisurf(xu, yu, zu, triangles=t[keep], color=col, alpha=0.5, linewidth=0.2, edgecolor=col, shade=True)
            ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_zlim(*lim)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{lbl} — View {p + 1}", fontsize=10)
            ax.set_xlabel(f"digit-{TRIPLE[0]} loss", fontsize=8, labelpad=2)
            ax.set_ylabel(f"digit-{TRIPLE[1]} loss", fontsize=8, labelpad=2)
            ax.set_zlabel(f"digit-{TRIPLE[2]} loss", fontsize=8, labelpad=2)
            ax.tick_params(labelsize=7, pad=1)
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.legend(handles=[Patch(facecolor=c_, alpha=0.6, label=l_) for l_, c_, _e in series],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.03), frameon=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.08, wspace=0.08, hspace=0.12)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    fig.savefig(Path(out_png).with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(DEFAULT_HOME))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rep-r", type=int, default=40, help="uniform representative (report rule: largest exact train HV = 40)")
    ap.add_argument("--title", default=f"MNIST {TRIPLE[0]}/{TRIPLE[1]}/{TRIPLE[2]} — loss space (B = 100,000)")
    ap.add_argument("--name", default="pareto_front_adam_3d")
    a = ap.parse_args()
    home = Path(a.home); out_dir = Path(a.out_dir) if a.out_dir else home
    legs = _legs(home)
    series = []
    for lbl, col, name in ((f"adaptive λ-bundle", COL["adaptive"], f"adaptive_ccp_seed{SEED}"),
                           (f"uniform grid (r = {a.rep_r})", COL["uniform"], f"uniform_r{a.rep_r}_seed{SEED}")):
        F = np.asarray(legs[name][1]["fvals"], dtype=float)
        F = F[np.isfinite(F).all(axis=1)]
        fr = F[_nondominated_kd(F)]
        fr = fr[(fr <= LN3).all(axis=1)]
        series.append((lbl, col, _envelope(fr)))
        print(f"{lbl}: {len(fr)} non-dominated points, {len(series[-1][2])} envelope points")
    out = out_dir / f"{a.name}.png"
    draw(series, a.title, out)
    print("figure ->", out)


if __name__ == "__main__":
    main()
