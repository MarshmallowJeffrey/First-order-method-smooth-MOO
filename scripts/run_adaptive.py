"""GRAB (adaptive bundle method) with the reported settings and the fixed budget of the task (mogym.config);
--budget overrides it.  For K>2 the reported metric of every checkpoint is then computed from the saved bundle
(adaptive_metric.json).

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
budget = a.budget or spec["budget"]
stem = _setup.Path(a.results) / a.task / "adaptive" / "adaptive"
adaptive(model, config.smoothness(a.task), budget, stem, adam_keep_tol=config.ADAM_KEEP_TOL, **spec["adaptive"])
meta, curve = points.adaptive_curve(a.results, a.task, K)  # K>2: reported metric of every checkpoint
print(f"{a.task} adaptive: {budget:,} calls, final GN {curve[-1]['gn']:.4e}, CPU {curve[-1]['cpu']:.2f} s")
