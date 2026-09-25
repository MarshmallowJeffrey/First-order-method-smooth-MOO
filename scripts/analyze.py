#!/usr/bin/env python
"""From the run folders to results/k<K>.json (the numbers behind the figures and tables of Section 4.1 and
Appendix C.1) and results/k<K>_fronts.json (the front points of the legs in the front figure).

    python scripts/analyze.py --K 2 [--runs runs/k2] [--workers 6]
    python scripts/analyze.py --K 3 [--runs runs/k3]

Per run: the plateau test at B/8, B/4, B/2, B, B_run, and the marker (y, x, wall-clock time at x); for K = 2 the
marker is located exactly by bisection over the bundle prefixes (the slow part: about a minute per leg).
Per configuration: geometric means over the seeds and the number of seeds that plateau.  See abm/analysis.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm import config as C  # noqa: E402
from abm.analysis import TOL, checkpoint_marker, exact_marker_k2, geomean, plateau, suffix_max  # noqa: E402
from abm.fronts import LN3, front_2d, nondominated_kd  # noqa: E402


def _exact(args):
    name, path = args
    z = np.load(Path(path) / "grams.npz")
    return name, exact_marker_k2(z["gram_stack"], z["seg_grads"])


def load_runs(runs_dir):
    runs = {}
    for d in sorted(Path(runs_dir).iterdir()):
        if (d / "summary.json").exists():
            runs[d.name] = (d, json.loads((d / "summary.json").read_text()))
    return runs


def analyze(K, runs_dir, workers=1):
    runs = load_runs(runs_dir)
    out = {}
    for name, (d, sm) in runs.items():
        g = suffix_max(sm["audit_gn"])
        tests, B_run = plateau(sm["ck_grads"], g, sm["budget"])
        out[name] = {"method": sm["method"], "param": sm["param"], "seed": sm["seed"], "budget": sm["budget"],
                     "ck_grads": sm["ck_grads"], "ck_wall": sm["ck_wall"], "audit_gn": sm["audit_gn"],
                     "final": float(g[-1]), "levels": tests, "B_run": B_run,
                     "wall_seconds": sm["wall_seconds"], "decision_seconds": sm["decision_seconds"],
                     "audit_seconds": sm["audit_seconds"], "segments": sm["segments"],
                     "rejections": sm["rejections"], "device": sm["device"]}
    baselines = [r for r in out.values() if r["method"] != "adaptive"]
    B_main = max([r["B_run"] for r in baselines if r["B_run"] is not None], default=None)
    for name, r in out.items():
        if r["method"] != "adaptive" and B_main is not None:
            r["marker"] = checkpoint_marker(r["ck_grads"], r["ck_wall"], suffix_max(r["audit_gn"]), B_main)
    if K == 2:                                   # exact markers
        todo = [(n, str(runs[n][0])) for n, r in out.items() if r["method"] != "adaptive"]
        if workers > 1:
            with get_context("spawn").Pool(workers) as pool:
                exact = dict(pool.imap_unordered(_exact, todo))
        else:
            exact = dict(map(_exact, todo))
        for n, e in exact.items():
            r = out[n]
            r["marker_checkpoint"] = r.get("marker")
            r["marker"] = ({"y": e["y"], "x": e["x"], "m": e["m"],
                            "wall": float(np.interp(e["x"], r["ck_grads"], r["ck_wall"]))} if e["x"] is not None else None)
    configs = {}
    for r in baselines:
        if r.get("marker"):
            configs.setdefault((r["method"], r["param"]), []).append(r)
    stats = []
    for (method, param), rs in sorted(configs.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        stats.append({"method": method, "param": param, "seeds": [r["seed"] for r in rs],
                      "y_per_seed": [r["marker"]["y"] for r in rs],
                      "y_geomean": geomean([r["marker"]["y"] for r in rs]),
                      "x_geomean": geomean([r["marker"]["x"] for r in rs]),
                      "x_median": float(np.median([r["marker"]["x"] for r in rs])),
                      "wall_geomean": geomean([r["marker"]["wall"] for r in rs]),
                      "n_plateau": int(sum(r["B_run"] is not None for r in rs))})
    adaptive = {r["seed"]: r["final"] for r in out.values() if r["method"] == "adaptive"}
    return {"K": K, "digits": list(C.DIGITS[K]), "tol": TOL, "B_main": B_main, "configs": stats,
            "adaptive_final": {str(s): v for s, v in sorted(adaptive.items())},
            "adaptive_final_geomean": geomean(list(adaptive.values())) if adaptive else None,
            "runs": out}


def fronts(K, runs_dir):
    spec = C.FRONT_LEGS[K]
    names = [f"adaptive_seed{s}" for s in spec["seeds"]] + [f"uniform_r{spec['uniform']}_seed{s}" for s in spec["seeds"]]
    if K == 2:
        names += [f"surf_N{spec['surf']}_seed{s}" for s in spec["seeds"]]
    res = {}
    for n in names:
        if not (Path(runs_dir) / n / "grams.npz").exists():
            print(f"  (front: {n} not found, skipped)")
            continue
        F = np.asarray(np.load(Path(runs_dir) / n / "grams.npz")["fvals"], dtype=float)
        if K == 2:
            res[n] = front_2d(F).tolist()
        else:                                    # the front inside [0, ln 3]^3, cut to the window
            F = F[np.isfinite(F).all(axis=1)]
            fr = F[nondominated_kd(F)]
            fr = fr[(fr <= LN3).all(axis=1)]
            res[n] = fr[(fr <= C.FRONT_WINDOW[3]).all(axis=1)].tolist()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, choices=(2, 3), required=True)
    ap.add_argument("--runs", default=None, help="default: runs/k<K>")
    ap.add_argument("--workers", type=int, default=1, help="processes for the exact K = 2 markers")
    ap.add_argument("--out", default=None, help="default: results/")
    a = ap.parse_args()
    runs_dir = Path(a.runs) if a.runs else ROOT / "runs" / f"k{a.K}"
    out = Path(a.out) if a.out else ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    res = analyze(a.K, runs_dir, a.workers)
    (out / f"k{a.K}.json").write_text(json.dumps(res, indent=1))
    (out / f"k{a.K}_fronts.json").write_text(json.dumps(fronts(a.K, runs_dir)))
    ad = res["adaptive_final_geomean"]
    print(f"B_main = {res['B_main']}; adaptive final (geo-mean) " + (f"{ad:.4e}" if ad else "-"))
    for s in res["configs"]:
        print(f"  {s['method']:8s} {s['param']:3d}: y {s['y_geomean']:.3e}  x {s['x_geomean']:9.0f} / {s['x_median']:9.0f}"
              f"  plateau {s['n_plateau']}/{len(s['seeds'])}")


if __name__ == "__main__":
    main()
