"""plot_curves_family_without_256_checkpoints.py — derived figure (user
request Sep 9 2026): the adaptive-CCP best-so-far worst-GN curve together
with the full curves of every run of one baseline family (uniform grid
r, or SURF N), as lines.  Two panels: gradient evaluations (left) and
CPU seconds (right); linear x, log y; clean style.

Reads the same summary.json files as plot_dots_figure_without_256_checkpoints.py.

Usage:
    python plot_curves_family_without_256_checkpoints.py --home <core_home> --family uniform
    python plot_curves_family_without_256_checkpoints.py --home <core_home> --family surf --params 10,20,40
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout  # noqa: F401
from plot_dots_figure_without_256_checkpoints import (  # noqa: E402
    ADAPTIVE_COLOR,
    dot_of,
    load_runs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--family", choices=["uniform", "surf"], default="uniform")
    ap.add_argument("--params", default=None, help="comma-separated subset")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--name", default=None)
    ap.add_argument("--log-x", action="store_true")
    ap.add_argument("--mark", choices=["none", "laststep", "plateau", "auto", "final"],
                    default="none", help="mark one point per curve")
    a = ap.parse_args()
    home = Path(a.home)
    runs = load_runs(home, a.seed)
    fam = runs[a.family]
    if a.params:
        keep = {int(v) for v in a.params.split(",")}
        fam = {p: r for p, r in fam.items() if p in keep}
    params = sorted(fam)
    sym = "r" if a.family == "uniform" else "N"
    cmap = plt.get_cmap("viridis")
    colors = {p: cmap(0.08 + 0.84 * k / max(1, len(params) - 1))
              for k, p in enumerate(params)}
    ad = runs["adaptive"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=200)
    for ax, key, xlabel in ((axes[0], "g", "total gradient evaluations"),
                            (axes[1], "c", "CPU seconds")):
        for p in params:
            r = fam[p]
            x = r[key]
            m = x > 0 if a.log_x else np.ones(len(x), dtype=bool)
            ax.plot(x[m], r["y"][m], color=colors[p], lw=1.3, alpha=0.95,
                    label=f"{'uniform grid' if a.family == 'uniform' else 'SURF'}, {sym} = {p}")
            if a.mark != "none":
                d = dot_of(r, 0.05, a.mark)
                xm = d["x"] if key == "g" else d["cpu"]
                ax.scatter([xm], [d["final"]], marker="o", s=46, color=colors[p],
                           edgecolors="white", linewidths=0.9, zorder=6)
                ax.annotate(f"{p}", (xm, d["final"]), textcoords="offset points",
                            xytext=(5, 5), fontsize=7, color=colors[p], zorder=7,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        xa = ad[key]
        m = xa > 0 if a.log_x else np.ones(len(xa), dtype=bool)
        ax.plot(xa[m], ad["y"][m], color=ADAPTIVE_COLOR, lw=2.4, zorder=5,
                label="adaptive λ-bundle", solid_capstyle="round")
        ax.set_yscale("log")
        if a.log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel("best-so-far worst-case gradient norm", fontsize=10.5)
        ax.grid(True, which="major", alpha=0.3, lw=0.6)
        ax.grid(False, which="minor")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=7.5, ncol=2, loc="upper right", frameon=True,
                  framealpha=0.9, edgecolor="#dddddd")
    fig.tight_layout()
    name = a.name or f"worst_gn_curves_{a.family}_all"
    out = home / f"{name}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
