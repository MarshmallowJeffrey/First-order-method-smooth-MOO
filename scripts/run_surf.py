"""SURF runs (each to its GN plateau) of one task (K=2), one per configured number of segments N.

    python scripts/run_surf.py fishwood [--values 2 4 ...] [--results results]
"""
import argparse

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, identity
from mogym.surf import surf

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=[t for t in config.TASKS if "surf" in config.TASKS[t]])
ap.add_argument("--values", type=int, nargs="*", help="numbers of segments N (default: the configured ones)")
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
a = ap.parse_args()
task = config.TASKS[a.task]; spec = task["surf"]
model = envs.build(a.task)
for n in a.values or spec["values"]:
    stem = _setup.Path(a.results) / a.task / "surf" / f"N{n}"
    run_spec = dict(task=a.task, method="surf", N=n, K_S=spec["K_S"], lr=spec["lr"], budget=task["budget"],
                    every=task["every"], rule=config.SURF_RULE, max_rounds=config.MAX_ROUNDS, state_tol=config.STATE_TOL)
    if identity.reusable(stem.with_suffix(".json"), run_spec, model):
        print(f"{a.task} SURF N={n}: stored run with the same identity, reused")
        continue
    surf(model, n, stem, rounds=config.MAX_ROUNDS, inner_steps=spec["K_S"], inner_lr=spec["lr"], every=task["every"],
         budget=task["budget"], rule=config.SURF_RULE, state_tol=config.STATE_TOL, run_spec=run_spec)
