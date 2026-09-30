"""Adaptive bundle method with the reported settings.

The budget is 1.05 x the Gradient Calls of the farthest plotted baseline point (rounded up), computed from
the baseline runs in --results; without them (or with --budget) the reported budget is used.  For K>2 the
reported metric of every checkpoint is then computed from the saved bundle (adaptive_metric.json).

    python scripts/run_adaptive.py fishwood [--budget 81000] [--results results]
"""
import argparse

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, points
from mogym.adaptive import adaptive

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=tuple(config.TASKS))
ap.add_argument("--budget", type=int)
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
a = ap.parse_args()
spec = config.TASKS[a.task]
model = envs.build(a.task); K = model["K"]
pts = points.points(a.results, a.task, K)
expected = sum(len(spec[m]["values"]) for m in ("uniform", "surf") if m in spec)
if a.budget:
    budget = a.budget
elif len(pts) == expected:
    budget = points.budget(pts, a.task)
else:
    budget = spec["budget"]
    print(f"{a.task}: {len(pts)} of {expected} baseline points found; using the reported budget {budget:,}")
if budget != spec["budget"]:
    print(f"{a.task}: budget {budget:,} differs from the reported {spec['budget']:,}")
stem = _setup.Path(a.results) / a.task / "adaptive" / "adaptive"
kw = dict(spec["adaptive"])
adaptive(model, config.smoothness(a.task), budget, stem, adam_keep_tol=config.ADAM_KEEP_TOL, **kw)
meta, curve = points.adaptive_curve(a.results, a.task, K)  # K>2: reported metric of every checkpoint
print(f"{a.task} adaptive: {budget:,} calls, final GN {curve[-1]['gn']:.4e}, CPU {curve[-1]['cpu']:.2f} s")
