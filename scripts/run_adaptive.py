"""The GRAB run of one task, to the budget B.

    python scripts/run_adaptive.py fishwood [--results results]
"""
import argparse

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, identity, points
from mogym.adaptive import adaptive

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=tuple(config.TASKS))
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
a = ap.parse_args()
task = config.TASKS[a.task]
model = envs.build(a.task); K = model["K"]
stem = _setup.Path(a.results) / a.task / "adaptive" / "adaptive"
run_spec = dict(task=a.task, method="GRAB", budget=task["budget"], state_tol=config.STATE_TOL,
                L=config.smoothness(a.task).tolist(), **task["adaptive"])
if identity.reusable(stem.with_suffix(".json"), run_spec, model):
    print(f"{a.task} GRAB: stored run with the same identity, reused")
else:
    adaptive(model, config.smoothness(a.task), task["budget"], stem, state_tol=config.STATE_TOL, run_spec=run_spec,
             **task["adaptive"])
meta, curve = points.adaptive_curve(a.results, a.task, K)  # K>2: reported metric of every checkpoint
print(f"{a.task} GRAB: {task['budget']:,} calls, final GN {curve[-1]['gn']:.4e}, CPU {curve[-1]['cpu']:.2f} s")
