"""Uniform discretization: one run per plotted resolution r, each until its GN plateaus, with the
Gradient-Call checkpoints of mogym.config (every grid weight end past each mark).

    python scripts/run_uniform.py fishwood [--values 2 4 8] [--results results]
"""
import argparse

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, metrics
from mogym.uniform import uniform_plateau

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=tuple(config.TASKS))
ap.add_argument("--values", type=int, nargs="*", help="resolutions r (default: the plotted ones)")
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
a = ap.parse_args()
spec = config.TASKS[a.task]["uniform"]
model = envs.build(a.task)
pool = metrics.fixed_stratified_weights() if model["K"] > 3 else None
for r in a.values or spec["values"]:
    stem = _setup.Path(a.results) / a.task / "uniform" / f"r{r}"
    if stem.with_suffix(".json").exists():
        print(f"{a.task} uniform r={r}: exists, skipped")
        continue
    uniform_plateau(model, r, stem, lr=spec["lr"], steps=spec["M"], rule=config.RULE, pool=pool,
                    max_sweeps=config.MAX_SWEEPS, every=config.TASKS[a.task]["every"],
                    budget=config.TASKS[a.task]["budget"], L=config.smoothness(a.task))
