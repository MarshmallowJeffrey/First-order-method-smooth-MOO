"""plot_pareto_front_paper_K2_without_256_checkpoints.py — paper version
of the K = 2 Pareto-front figure (PI review Sep 7 2026, user decision
Sep 8-9): ONE panel, knee window [0, 0.27]^2 (final choice Sep 9 after
0.3 / 0.2 / 0.15 / 0.25 trials), curves only — no per-lambda
scatter, no colorbar, no initial point, no beyond-view counter.

Each curve is the non-dominated set of ALL delivered points of one leg
(the same set that defines the hypervolume (17) in the paper), connected
in f1 order.  The script also computes the hypervolume of every leg with
reference point (ln 2, ln 2) for Table 6 and writes it to
pareto_front_paper_hv.json next to the figure.

Reads grams.npz of the seed-41 legs under MAIN_HOME/<core>/ and writes
pareto_front_paper.png there.  ``--space test`` (user request Sep 9) draws
the same legs' delivered points in the TEST cross-entropy space instead
(grams.npz["test_ce"]: every delivered theta scored on all t10k rows of
the two digits, plain mean cross-entropy per digit, no ridge term) and
writes pareto_front_paper_test.png; the non-dominated set and the
hypervolume are recomputed in that space.  Baseline representatives (user rule Sep 9):
among ALL uniform_r*_seed41 and surf_N*_seed41 runs present, the one
with the lowest final worst-case GN; the chosen r / N go into the legend
and into the JSON report.  Nothing is re-run and no
existing file is modified.

Usage:
    python plot_pareto_front_paper_K2_without_256_checkpoints.py                  # adam core
    python plot_pareto_front_paper_K2_without_256_checkpoints.py --core adagrad_x10
    python plot_pareto_front_paper_K2_without_256_checkpoints.py --window 0.15 --suffix _w015
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout  # noqa: F401
from run_surf_compare_K2_without_256_checkpoints import (  # noqa: E402
    MAIN_HOME,
    PAIR,
)

ZREF = (np.log(2.0), np.log(2.0))   # loss of uniform guessing, eq. (17)

# Draw order: SURF at the bottom, uniform dashed above it, adaptive on
# top.  The three fronts nearly coincide, so the line styles and widths
# are the only thing that keeps them apart.
STYLES = {
    "surf": dict(color="#e41a1c", ls=":", lw=2.4, zorder=4),
    "uniform": dict(color="#377eb8", ls="--", lw=2.2, zorder=5),
    "adaptive_ccp": dict(color="#ff7f00", ls="-", lw=1.6, zorder=6),
}


def nondominated_2d(F):
    """Indices of the non-dominated rows of F (minimisation, 2-D),
    sorted by f1 ascending; f2 is then strictly decreasing."""
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep, best = [], np.inf
    for i in order:
        if F[i, 1] < best - 1e-15:
            keep.append(i)
            best = F[i, 1]
    return np.asarray(keep, dtype=int)


def hypervolume_2d(front, zref):
    """Area dominated by `front` (sorted by f1) inside the box below zref."""
    P = front[(front < np.asarray(zref)).all(axis=1)]
    P = P[np.argsort(P[:, 0])]
    hv = 0.0
    for i, (x, y) in enumerate(P):
        x_next = P[i + 1, 0] if i + 1 < len(P) else zref[0]
        hv += (x_next - x) * (zref[1] - y)
    return float(hv)


def _leg_dirs(core_home):
    """The seed-41 legs of one core.  Among all uniform_r*_seed41 and
    surf_N*_seed41 runs present (the dots campaign), the representative
    of each baseline is the one with the LOWEST final worst-case GN; its
    r / N is read from the directory name.  Returns
    {leg: (dir, param, final_worst_gn)}."""
    best = {}
    for d in sorted(core_home.iterdir()):
        if not d.is_dir() or not d.name.endswith("_seed41"):
            continue
        if not (d / "summary.json").exists():
            continue
        if d.name.startswith("uniform_r"):
            leg = "uniform"
            param = int(d.name[len("uniform_r"):-len("_seed41")])
        elif d.name.startswith("surf_N"):
            leg = "surf"
            param = int(d.name[len("surf_N"):-len("_seed41")])
        elif d.name == "adaptive_ccp_seed41":
            leg, param = "adaptive_ccp", None
        else:
            continue
        final = float(json.loads((d / "summary.json").read_text())
                      ["audited_gn_norm_history"][-1])
        if leg not in best or final < best[leg][2]:
            best[leg] = (d, param, final)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="adam_1e-3_b0.9")
    ap.add_argument("--window", type=float, default=0.27)
    ap.add_argument("--space", choices=["train", "test"], default="train")
    ap.add_argument("--legend-style", choices=["paper", "ccp"], default="paper",
                    help="'ccp': legend/axis wording of the user's Sep-9 report (Adaptive CCP, Uniform r = .., "
                         "SURF N = .., 'Digit k objective', title 'Training objectives')")
    ap.add_argument("--uniform-r", type=int, default=None,
                    help="override the uniform representative (default: lowest final GN)")
    ap.add_argument("--surf-N", type=int, default=None,
                    help="override the SURF representative (default: lowest final GN)")
    ap.add_argument("--suffix", default="",
                    help="appended to the output file names (variants)")
    a = ap.parse_args()
    core_home = MAIN_HOME / a.core
    legs = _leg_dirs(core_home)
    for leg, prefix, want in (("uniform", "uniform_r", a.uniform_r),
                              ("surf", "surf_N", a.surf_N)):
        if want is None:
            continue
        d = core_home / f"{prefix}{want}_seed41"
        final = float(json.loads((d / "summary.json").read_text())
                      ["audited_gn_norm_history"][-1])
        legs[leg] = (d, want, final)
    if a.legend_style == "ccp":
        labels = {"uniform": f"Uniform r = {legs['uniform'][1]}", "surf": f"SURF N = {legs['surf'][1]}",
                  "adaptive_ccp": "Adaptive CCP"}
    else:
        labels = {
            "uniform": f"uniform grid, r = {legs['uniform'][1]}",
            "surf": f"SURF, N = {legs['surf'][1]}",
            "adaptive_ccp": "adaptive λ-bundle",
        }

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    report = {"core": a.core, "space": a.space, "window": a.window, "zref": list(ZREF),
              "legs": {}}
    for leg in ("surf", "uniform", "adaptive_ccp"):
        d, _p, _final = legs[leg]
        key = "fvals" if a.space == "train" else "test_ce"
        F = np.asarray(np.load(d / "grams.npz")[key], dtype=float)
        F = F[np.isfinite(F).all(axis=1)]
        idx = nondominated_2d(F)
        front = F[idx]
        ax.plot(front[:, 0], front[:, 1], label=labels[leg],
                solid_capstyle="round", **STYLES[leg])
        in_win = int(((front <= a.window).all(axis=1)).sum())
        report["legs"][leg] = {
            "dir": d.name,
            "param": _p,
            "final_worst_gn": _final,
            "delivered_points": int(len(F)),
            "nondominated_points": int(len(front)),
            "nondominated_in_window": in_win,
            "hypervolume": hypervolume_2d(front, ZREF),
            "min_f1": float(front[:, 0].min()),
            "min_f2": float(front[:, 1].min()),
        }
    w = a.window
    ax.set_xlim(-0.004 * w / 0.3, w)
    ax.set_ylim(-0.004 * w / 0.3, w)
    ax.set_aspect("equal", adjustable="box")
    if a.legend_style == "ccp":
        ax.set_xlabel(f"Digit {PAIR[0]} objective" if a.space == "train" else f"Digit {PAIR[0]} test cross-entropy")
        ax.set_ylabel(f"Digit {PAIR[1]} objective" if a.space == "train" else f"Digit {PAIR[1]} test cross-entropy")
        # user request Sep 9: no 'training' in the title (no test set in the report)
        ax.set_title("Non-dominated fronts" if a.space == "train" else "Non-dominated fronts (test cross-entropy)", fontsize=12)
    elif a.space == "train":
        ax.set_xlabel(f"$F_{PAIR[0]}$: training loss of digit {PAIR[0]}")
        ax.set_ylabel(f"$F_{PAIR[1]}$: training loss of digit {PAIR[1]}")
    else:
        ax.set_xlabel(f"test cross-entropy of digit {PAIR[0]}")
        ax.set_ylabel(f"test cross-entropy of digit {PAIR[1]}")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    stem = "pareto_front_paper" if a.space == "train" else "pareto_front_paper_test"
    out_png = core_home / f"{stem}{a.suffix}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    (core_home / f"{stem}_hv{a.suffix}.json").write_text(
        json.dumps(report, indent=2))
    print(f"figure -> {out_png}")
    for leg, r in report["legs"].items():
        print(f"  {leg:13s} param={r['param']} "
              f"final GN={r['final_worst_gn']:.4e} "
              f"delivered={r['delivered_points']:5d} "
              f"nondominated={r['nondominated_points']:4d} "
              f"(in window {r['nondominated_in_window']:4d})  "
              f"HV={r['hypervolume']:.4f}  "
              f"min f1={r['min_f1']:.4f} min f2={r['min_f2']:.4f}")


if __name__ == "__main__":
    main()
