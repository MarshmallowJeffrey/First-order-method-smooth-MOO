#!/usr/bin/env python
"""K = 2: GRAB with the exact envelope selection (Appendix A.4.1; the runs of the paper since 2026-10-04,
results/k2.json) against GRAB with the CCP lambda-search (its earlier runs, results/k2_ccp.json), on the same seeds,
budget and checkpoints.

    python scripts/compare_envelope.py [--runs <folder>] [--ccp-runs <folder>]

Writes results/k2_envelope.json and figures/k2_envelope_vs_ccp[_zoom].pdf/.png.  --runs / --ccp-runs read the runs
from run folders instead (for short test runs).  Values are the audited worst-case gradient norms repaired by their
suffix maximum, as in the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm import config as C  # noqa: E402
from abm.analysis import geomean, suffix_max  # noqa: E402

LEVELS = (10_000, 20_000, 50_000, 100_000, 240_000, 480_000)


def load_folder(folder):
    runs = {}
    for d in sorted(Path(folder).glob("adaptive_seed*")):
        if (d / "summary.json").exists():
            sm = json.loads((d / "summary.json").read_text())
            runs[int(sm["seed"])] = {k: sm[k] for k in ("ck_grads", "ck_wall", "audit_gn", "wall_seconds",
                                                         "decision_seconds", "segments", "rejections", "budget")}
            runs[int(sm["seed"])]["selector"] = sm.get("selector", "ccp")
    return runs


def load_record(path):
    """The adaptive runs of a results file and the file itself."""
    res = json.loads(Path(path).read_text())
    runs = {int(r["seed"]): {k: r[k] for k in ("ck_grads", "ck_wall", "audit_gn", "wall_seconds", "decision_seconds",
                                               "segments", "rejections", "budget")}
            for r in res["runs"].values() if r["method"] == "adaptive"}
    for s, r in runs.items():
        r["selector"] = next(x.get("selector", "ccp") for x in res["runs"].values()
                             if x["method"] == "adaptive" and int(x["seed"]) == s)
    return runs, res


def at(r, b):
    """(suffix-max value, wall-clock time) at the last checkpoint at or before b gradient calls."""
    x = np.asarray(r["ck_grads"])
    j = int(np.searchsorted(x, b * (1 + 1e-9), side="right") - 1)
    return float(suffix_max(r["audit_gn"])[j]), float(r["ck_wall"][j])


def first_below(r, y):
    g = suffix_max(r["audit_gn"])
    j = np.nonzero(g <= y)[0]
    return (float(r["ck_grads"][j[0]]), float(r["ck_wall"][j[0]])) if j.size else (None, None)


def summarize(runs):
    seeds = sorted(runs)
    finals = [float(suffix_max(runs[s]["audit_gn"])[-1]) for s in seeds]
    return {"seeds": seeds, "final_per_seed": finals, "final_geomean": geomean(finals),
            "wall_seconds": [runs[s]["wall_seconds"] for s in seeds],
            "decision_seconds": [runs[s]["decision_seconds"] for s in seeds],
            "decision_share": [runs[s]["decision_seconds"] / runs[s]["wall_seconds"] for s in seeds],
            "rejections": [runs[s]["rejections"] for s in seeds], "segments": [runs[s]["segments"] for s in seeds]}


def curve(runs):
    """Geometric mean over the seeds at every checkpoint (the checkpoints are the same in every run)."""
    seeds = sorted(runs)
    x = np.asarray(runs[seeds[0]]["ck_grads"], float)
    n = min(len(runs[s]["ck_grads"]) for s in seeds)
    G = np.array([suffix_max(runs[s]["audit_gn"])[:n] for s in seeds])
    W = np.array([np.asarray(runs[s]["ck_wall"][:n], float) for s in seeds])
    wall = np.exp(np.log(np.where(W > 0, W, 1.0)).mean(axis=0)) * (W > 0).all(axis=0)
    return x[:n], wall, np.exp(np.log(G).mean(axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="envelope runs from a folder (default: results/k2.json)")
    ap.add_argument("--ccp-runs", default=None, help="CCP runs from a folder (default: results/k2_ccp.json)")
    ap.add_argument("--out-json", default=str(ROOT / "results" / "k2_envelope.json"))
    ap.add_argument("--out-fig", default=str(ROOT / "figures" / "k2_envelope_vs_ccp"))
    a = ap.parse_args()

    env, res = (load_folder(a.runs), None) if a.runs else load_record(ROOT / "results" / "k2.json")
    ccp = load_folder(a.ccp_runs) if a.ccp_runs else load_record(ROOT / "results" / "k2_ccp.json")[0]
    if not env or any(r["selector"] != "envelope" for r in env.values()):
        sys.exit("the envelope runs are missing or did not use the envelope selection")
    if not ccp or any(r["selector"] != "ccp" for r in ccp.values()):
        sys.exit("the CCP runs are missing or did not use CCP")
    seeds = sorted(set(env) & set(ccp))
    env, ccp = {s: env[s] for s in seeds}, {s: ccp[s] for s in seeds}
    budget = env[seeds[0]]["budget"]

    out = {"budget": budget, "seeds": seeds, "envelope": summarize(env), "ccp": summarize(ccp), "levels": []}
    for b in [L for L in LEVELS if L <= budget] + ([budget] if budget not in LEVELS else []):
        row = {"calls": b}
        for name, runs in (("envelope", env), ("ccp", ccp)):
            vals = [at(runs[s], b) for s in seeds]
            row[name] = {"value_geomean": geomean([v for v, _ in vals]), "wall_mean": float(np.mean([w for _, w in vals]))}
        out["levels"].append(row)
    if res is not None:                                  # against the best baselines of the paper
        stats = {(s["method"], s["param"]): s for s in res["configs"]}
        out["baselines"] = {}
        for fam, p in C.BEST[2].items():
            s = stats[(fam, p)]
            reach = {name: [first_below(runs[sd], s["y_geomean"]) for sd in seeds] for name, runs in
                     (("envelope", env), ("ccp", ccp))}
            out["baselines"][f"{fam}_{p}"] = {
                "y_geomean": s["y_geomean"], "x_geomean": s["x_geomean"], "wall_geomean": s["wall_geomean"],
                "ratio_envelope": s["y_geomean"] / out["envelope"]["final_geomean"],
                "ratio_ccp": s["y_geomean"] / out["ccp"]["final_geomean"],
                "first_below": reach}
    Path(a.out_json).write_text(json.dumps(out, indent=1))

    e, c = out["envelope"], out["ccp"]
    print(f"seeds {seeds}, budget {budget:,.0f} gradient calls")
    print(f"final worst-case GN (geometric mean): envelope {e['final_geomean']:.4e}, CCP {c['final_geomean']:.4e} "
          f"(envelope / CCP = {e['final_geomean'] / c['final_geomean']:.3f})")
    for name, s in (("envelope", e), ("CCP", c)):
        print(f"  {name:8s} per seed {', '.join(f'{v:.4e}' for v in s['final_per_seed'])}; training wall "
              f"{np.mean(s['wall_seconds']):.0f} s, lambda search {np.mean(s['decision_seconds']):.0f} s "
              f"({100 * np.mean(s['decision_share']):.1f} %)")
    for row in out["levels"]:
        print(f"  at {row['calls']:>9,.0f} calls: envelope {row['envelope']['value_geomean']:.4e} "
              f"({row['envelope']['wall_mean']:.0f} s), CCP {row['ccp']['value_geomean']:.4e} ({row['ccp']['wall_mean']:.0f} s)")
    for k, v in out.get("baselines", {}).items():
        print(f"  {k}: y {v['y_geomean']:.3e}; ratio envelope {v['ratio_envelope']:.2f}x, CCP {v['ratio_ccp']:.2f}x; "
              f"first below it (calls, s): envelope {v['first_below']['envelope']}, CCP {v['first_below']['ccp']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for name, runs, col in (("GRAB (CCP)", ccp, "#ff7f0e"), ("GRAB (envelope)", env, "#6a3d9a")):
        x, w, g = curve(runs)
        m = x > 0
        axs[0].plot(x[m], g[m], color=col, lw=2.2, label=name)
        axs[1].plot(w[m], g[m], color=col, lw=2.2, label=name)
    if res is not None:
        stats = {(s["method"], s["param"]): s for s in res["configs"]}
        for fam, drawn, mk, col in (("uniform", C.FIGURE_UNIFORM_R[2], "s", "#1f77b4"), ("surf", C.FIGURE_SURF_N, "^", "#d62728")):
            pts = [stats[(fam, p)] for p in drawn if (fam, p) in stats]
            for ax, key in ((axs[0], "x_geomean"), (axs[1], "wall_geomean")):
                ax.plot([s[key] for s in pts], [s["y_geomean"] for s in pts], mk, color=col, ms=6, ls="",
                        label=("Unif Discrtztn (r)" if fam == "uniform" else "SURF (N)") if ax is axs[0] else None)
    for ax, lab in ((axs[0], "Gradient Calls"), (axs[1], "Time (s)")):
        ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel(r"$\max_{\lambda\in\Delta_K}\mathrm{GN}(\lambda,\mathcal{B}_t)$")
    axs[0].legend(fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out_fig}.{ext}", dpi=200)
    if res is not None:                                  # the view of the paper's figure: x cut at 1.1 x the last marker
        stats = {(s["method"], s["param"]): s for s in res["configs"]}
        pts = [stats[("uniform", p)] for p in C.FIGURE_UNIFORM_R[2]] + [stats[("surf", p)] for p in C.FIGURE_SURF_N]
        axs[0].set_xlim(0, 1.1 * max(s["x_geomean"] for s in pts))
        axs[1].set_xlim(0, 1.1 * max(s["wall_geomean"] for s in pts))
        for ext in ("pdf", "png"):
            fig.savefig(f"{a.out_fig}_zoom.{ext}", dpi=200)
    print("saved", a.out_json, "and", a.out_fig + "[_zoom].pdf/.png")


if __name__ == "__main__":
    main()
