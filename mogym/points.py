"""Reading the runs back: the plotted baseline points, the adaptive curve and the adaptive budget.

Results layout (created by the scripts):
  results/<task>/uniform/r<r>.json|.npz      one Uniform run per resolution r
  results/<task>/surf/N<N>.json|.npz         one SURF run per N (K=2)
  results/<task>/adaptive/adaptive.json|.npz the adaptive run; adaptive_metric.json caches the K>2 metric
"""
import json
import math
from pathlib import Path

import numpy as np

from . import config, metrics, plateau


def plotted_gn(row, K):
    """K=3: geometric midpoint of the certified interval (as in every figure); else the value itself."""
    return math.sqrt(row["gn"] * row["gn_upper"]) if K == 3 else row["gn"]


def points(results, task, K):
    """One point per plotted r (Uniform) and N (SURF): the earliest checkpoint after which the GN stays
    within 5% of its value at stopping; calls, CPU time and GN are read from that checkpoint."""
    out = []
    spec = config.TASKS[task]
    for method, sub, pre in (("Uniform", "uniform", "r"), ("SURF", "surf", "N")):
        for v in spec.get(sub, {}).get("values", []):
            f = Path(results) / task / sub / f"{pre}{v}.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            if m.get("status") != "plateau":
                continue
            cps = m["checkpoints"][1:]
            j = plateau.onset([plotted_gn(c, K) for c in cps], config.POINT_BAND)
            out.append(dict(method=method, param=v, calls=cps[j]["component_gradients"], cpu=cps[j]["train_cpu"],
                            gn=plotted_gn(cps[j], K), lower=cps[j]["gn"], upper=cps[j]["gn_upper"],
                            stop_calls=m["stop_calls"], stop_cpu=m["stop_cpu"],
                            iterations=m.get("sweeps", m.get("outer_rounds"))))
    return out


def budget(pts, task):
    """1.05 x the Gradient Calls of the farthest plotted point, rounded up."""
    step = config.TASKS[task]["budget_round"]
    return int(math.ceil(1.05 * max(p["calls"] for p in pts) / step) * step)


def adaptive_curve(results, task, K):
    """Checkpoints of the adaptive run with the reported metric (K>2: computed from the saved Jacobians
    on every nested prefix of the bundle and cached)."""
    stem = Path(results) / task / "adaptive" / "adaptive"
    meta = json.loads(stem.with_suffix(".json").read_text())
    cps = meta["checkpoints"]
    if K == 2:
        vals = [(c["gn"], c["gn"], c["gn"]) for c in cps]
    else:
        side = stem.parent / "adaptive_metric.json"
        cache = json.loads(side.read_text()) if side.exists() else {}
        need = [n for n in sorted({c["bundle_size"] for c in cps}) if str(n) not in cache]
        if need:
            J = np.load(stem.with_suffix(".npz"))["J"]
            if K == 3:
                for n in need:
                    lo, _, up = metrics.k3_interval(J[:n])
                    cache[str(n)] = [math.sqrt(lo * up), float(lo), float(up)]
            else:
                for n, v in zip(need, metrics.pool_gn_prefixes(J, need, metrics.fixed_stratified_weights())):
                    cache[str(n)] = [float(v)] * 3
            side.write_text(json.dumps(cache))
        vals = [tuple(cache[str(c["bundle_size"])]) for c in cps]
    return meta, [dict(calls=c["component_gradients"], cpu=c["train_cpu"], gn=v[0], lower=v[1], upper=v[2])
                  for c, v in zip(cps, vals)]


def best_within(curve, field, x):
    vals = [c["gn"] for c in curve if c[field] <= x + 1e-12]
    return min(vals) if vals else math.inf


def first_reach(curve, field, target):
    for c in curve:
        if c["gn"] <= target:
            return c[field]
    return None
