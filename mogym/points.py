"""Reading the runs back: the plotted baseline points and the adaptive curve.

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


def point_of(m, K):
    """The plotted point of a run that reached its plateau: on the Gradient-Call checkpoints before the stop and
    the stopping checkpoint, the earliest after which the GN stays within 5% of its value at stopping; calls,
    CPU time and GN are read from that checkpoint."""
    stop = m["stop_calls"]
    cps = [c for c in m["checkpoints"] if c.get("kind") == "calls" and c["component_gradients"] < stop]
    cps.append(m["checkpoints"][-1])  # the sweep / round checkpoint at which the rule stopped the run
    return cps[plateau.onset([plotted_gn(c, K) for c in cps], config.POINT_BAND)]


def points(results, task, K, all_values=False):
    """One point per plotted r (Uniform) and N (SURF): runs that reached their plateau with the point at
    <= B Gradient Calls (all_values=True: every run found, plotted or not, with a flag)."""
    out = []
    spec = config.TASKS[task]
    for method, sub, pre in (("Uniform", "uniform", "r"), ("SURF", "surf", "N")):
        files = sorted((Path(results) / task / sub).glob(f"{pre}*.json"), key=lambda f: int(f.stem[1:]))
        for f in files:
            v = int(f.stem[1:])
            if not all_values and v not in spec.get(sub, {}).get("values", []):
                continue
            m = json.loads(f.read_text())
            if m.get("status") != "plateau":
                if all_values:
                    out.append(dict(method=method, param=v, plotted=False, reason=m.get("status")))
                continue
            c = point_of(m, K)
            row = dict(method=method, param=v, calls=c["component_gradients"], cpu=c["train_cpu"],
                       gn=plotted_gn(c, K), lower=c["gn"], upper=c["gn_upper"], stop_calls=m["stop_calls"],
                       stop_cpu=m["stop_cpu"], iterations=m.get("sweeps", m.get("outer_rounds")),
                       plotted=c["component_gradients"] <= spec["budget"])
            if row["plotted"] or all_values:
                out.append(row)
    return out


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
