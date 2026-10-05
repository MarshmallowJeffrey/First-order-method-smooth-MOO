"""Uniform discretization runs (each to its GN plateau) of one task, one per configured resolution r.

    python scripts/run_uniform.py fishwood [--values 2 4 ...] [--results results]
"""
import argparse

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, identity, metrics
from mogym.uniform import uniform

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=tuple(config.TASKS))
ap.add_argument("--values", type=int, nargs="*", help="resolutions r (default: the configured ones)")
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
a = ap.parse_args()
task = config.TASKS[a.task]; spec = task["uniform"]
model = envs.build(a.task)
pool = metrics.fixed_stratified_weights() if model["K"] > 3 else None
for r in a.values or spec["values"]:
    stem = _setup.Path(a.results) / a.task / "uniform" / f"r{r}"
    run_spec = dict(task=a.task, method="uniform", r=r, M=spec["M"], lr=spec["lr"], budget=task["budget"],
                    every=task["every"], rule=config.RULE, max_sweeps=config.MAX_SWEEPS, state_tol=config.STATE_TOL,
                    L=config.smoothness(a.task).tolist())
    if identity.reusable(stem.with_suffix(".json"), run_spec, model):
        print(f"{a.task} uniform r={r}: stored run with the same identity, reused")
        continue
    uniform(model, r, stem, lr=spec["lr"], steps=spec["M"], rule=config.RULE, every=task["every"],
            budget=task["budget"], L=config.smoothness(a.task), state_tol=config.STATE_TOL,
            max_sweeps=config.MAX_SWEEPS, pool=pool, run_spec=run_spec)
