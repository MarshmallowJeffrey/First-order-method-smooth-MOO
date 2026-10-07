"""Upper bounds on GRAB's worst-case gradient norm at every checkpoint of a K>2 task (mogym.bounds, settings
mogym.config.BOUNDS), written to results/<task>/adaptive/adaptive_upper.json.  Stored bounds are reused if they
belong to the stored run and the current settings.  Runs after the GRAB run; uses BOUNDS["workers"] processes.

    python scripts/upper_bounds.py fruittree_d6 [--results results]
"""
import argparse
import json
from pathlib import Path

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import bounds, config, envs, points

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=tuple(config.TASKS))
    ap.add_argument("--results", default=str(_setup.ROOT / "results"))
    a = ap.parse_args()
    K = envs.build(a.task)["K"]
    if K <= 2:
        raise SystemExit(f"{a.task}: K = {K}; the metric is exact, no bounds are needed")
    meta, curve = points.adaptive_curve(a.results, a.task, K)   # lower estimates (computed and cached if needed)
    key = points.bounds_key(a.results, a.task)
    out = Path(a.results) / a.task / "adaptive" / "adaptive_upper.json"
    if out.exists() and json.loads(out.read_text()).get("key") == key:
        print(f"{a.task}: stored upper bounds belong to the stored run, reused")
        raise SystemExit
    s = config.BOUNDS; lower = [c["gn"] for c in curve]
    rows = bounds.upper_bounds(str(out.with_name("adaptive.npz")), meta["checkpoints"], lower, seconds=s["seconds"],
                               gap=s["gap"], workers=s["workers"], max_leaves=s["max_leaves"], log=print)
    out.write_text(json.dumps(dict(key=key, rows=[
        dict(calls=c["component_gradients"], bundle_size=c["bundle_size"], cpu=c["train_cpu"], lower=lo, **r)
        for c, lo, r in zip(meta["checkpoints"], lower, rows)]), indent=1))
    ratio = [r["upper"] / lo for r, lo in zip(rows[1:], lower[1:])]
    print(f"{a.task}: upper / lower estimate - 1 over checkpoints 2..{len(rows)}: min {min(ratio) - 1:.2%}, "
          f"max {max(ratio) - 1:.2%}; {sum(r['seconds'] for r in rows):.0f} s")
