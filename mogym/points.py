"""Reading the runs back: the plotted baseline points, the GRAB curve and its upper bounds.

Results layout (created by the scripts):
  results/<task>/uniform/r<r>.json|.npz        one Uniform run per resolution r, run to its GN plateau
  results/<task>/surf/N<N>.json|.npz           one SURF run per N (K=2), run to its GN plateau
  results/<task>/adaptive/adaptive.json|.npz   the GRAB run
  results/<task>/adaptive/adaptive_metric.json K>2: the lower estimate (mogym.metrics) at every GRAB checkpoint
  results/<task>/adaptive/adaptive_upper.json  K>2: the upper bound (mogym.bounds) at every GRAB checkpoint
  results/<task>/timing.json                   CPU times of the timing repeats (scripts/time_repeats.py)
"""
import json
import math
from pathlib import Path

import numpy as np

from . import config, identity, metrics, plateau


def point_of(m):
    """The plotted point of a run that reached its plateau: on the Gradient-Call checkpoints before the stop and
    the stopping checkpoint, the earliest after which the GN stays within 5% of its value at stopping; calls,
    CPU time and GN are read from that checkpoint."""
    stop = m["stop_calls"]
    cps = [c for c in m["checkpoints"] if c.get("kind") == "calls" and c["component_gradients"] < stop]
    cps.append(m["checkpoints"][-1])  # the sweep / round checkpoint at which the rule stopped the run
    return cps[plateau.onset([c["gn"] for c in cps], config.POINT_BAND)]


def timing(results, task):
    """Training CPU time at every checkpoint over the repeats of scripts/time_repeats.py ({} if not run)."""
    f = Path(results) / task / "timing.json"
    return json.loads(f.read_text())["runs"] if f.exists() else {}


def _cpu(times, key, meta):
    """Median CPU time per checkpoint and the CPU time of every repeat (the stored run alone without repeats).  The
    timing entry must belong to the stored run: same fingerprint and checkpoint calls."""
    rows = meta["checkpoints"]
    if key in times:
        t = times[key]
        if t.get("run") != identity.fingerprint(meta) or t["calls"] != [r["component_gradients"] for r in rows]:
            raise RuntimeError(f"{key}: timing.json was measured for another run; rerun scripts/time_repeats.py")
        reps = np.asarray(t["cpu"], float)
        return np.median(reps, axis=0), reps
    reps = np.asarray([[r["train_cpu"] for r in rows]], float)
    return reps[0], reps


def points(results, task):
    """One point per configured r (Uniform) and N (SURF) whose run reached its plateau with the point within B,
    and the configured runs that are not plotted, with the reason."""
    out, skipped = [], []
    spec = config.TASKS[task]; times = timing(results, task)
    for method, sub, pre in (("Uniform", "uniform", "r"), ("SURF", "surf", "N")):
        for v in spec.get(sub, {}).get("values", []):
            m = json.loads((Path(results) / task / sub / f"{pre}{v}.json").read_text())
            if m["status"] != "plateau":
                skipped.append((method, v, m["status"]))
                continue
            c = point_of(m)
            if c["component_gradients"] > spec["budget"]:
                skipped.append((method, v, f"point at {c['component_gradients']:,} calls > B"))
                continue
            i = m["checkpoints"].index(c)
            med, reps = _cpu(times, f"{sub}/{pre}{v}", m)
            out.append(dict(method=method, param=v, calls=c["component_gradients"], cpu=float(med[i]),
                            cpu_repeats=reps[:, i].tolist(), gn=c["gn"], iterations=m.get("sweeps", m.get("rounds"))))
    return out, skipped


def adaptive_curve(results, task, K):
    """Checkpoints of the GRAB run with the reported metric (K>2: computed from the saved Jacobians on every
    nested prefix of the bundle and cached; the cache is tied to the .npz and to the evaluator and recomputed
    when either differs)."""
    stem = Path(results) / task / "adaptive" / "adaptive"
    meta = json.loads(stem.with_suffix(".json").read_text())
    cps = meta["checkpoints"]
    if K == 2:
        vals = [c["gn"] for c in cps]
    else:
        side = stem.parent / "adaptive_metric.json"
        key = dict(npz_sha256=identity.file_sha256(stem.with_suffix(".npz")), evaluator=metrics.evaluator_sha256())
        cache = json.loads(side.read_text()) if side.exists() else {}
        if cache.get("key") != key:
            cache = dict(key=key, values={})
        need = [n for n in sorted({c["bundle_size"] for c in cps}) if str(n) not in cache["values"]]
        if need:
            J = np.load(stem.with_suffix(".npz"))["J"]
            for n, v in zip(need, metrics.pool_ccp_prefixes(J, need, metrics.fixed_stratified_weights())):
                cache["values"][str(n)] = float(v)
            side.write_text(json.dumps(cache))
        vals = [cache["values"][str(c["bundle_size"])] for c in cps]
    med, reps = _cpu(timing(results, task), "adaptive/adaptive", meta)
    return meta, [dict(calls=c["component_gradients"], cpu=float(med[i]), cpu_repeats=reps[:, i].tolist(), gn=v)
                  for i, (c, v) in enumerate(zip(cps, vals))]


def bounds_key(results, task):
    """What the upper bounds of the GRAB run belong to: its arrays (checked against the stored run), the evaluator of
    the lower estimates and the bound settings."""
    stem = Path(results) / task / "adaptive" / "adaptive"
    meta = json.loads(stem.with_suffix(".json").read_text())
    sha = identity.file_sha256(stem.with_suffix(".npz"))
    if meta["npz_sha256"] != sha:
        raise RuntimeError(f"{task}: adaptive.npz does not match adaptive.json")
    return dict(npz_sha256=sha, evaluator=metrics.evaluator_sha256(), settings=config.BOUNDS)


def upper_curve(results, task, K):
    """The GRAB curve with the upper bound of scripts/upper_bounds.py in place of the lower estimate ("lower" keeps
    the latter).  The stored bounds must belong to the stored run and the current settings, checkpoint by checkpoint."""
    meta, curve = adaptive_curve(results, task, K)
    f = Path(results) / task / "adaptive" / "adaptive_upper.json"
    if not f.exists():
        raise RuntimeError(f"{task}: no upper bounds; run scripts/upper_bounds.py {task}")
    ub = json.loads(f.read_text())
    if ub.get("key") != bounds_key(results, task) or [(r["calls"], r["bundle_size"]) for r in ub["rows"]] != \
            [(c["component_gradients"], c["bundle_size"]) for c in meta["checkpoints"]]:
        raise RuntimeError(f"{task}: adaptive_upper.json belongs to another run or settings; rerun scripts/upper_bounds.py")
    return meta, [dict(c, lower=c["gn"], gn=r["upper"]) for c, r in zip(curve, ub["rows"])]


def best_within(curve, field, x):
    vals = [c["gn"] for c in curve if c[field] <= x + 1e-12]
    return min(vals) if vals else math.inf


def first_reach(curve, field, target):
    for c in curve:
        if c["gn"] <= target:
            return c[field]
    return None
