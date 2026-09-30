"""SURF (K=2): one run per plotted N, each until its GN plateaus.

    python scripts/run_surf.py fishwood [--values 2 4 8] [--results results]
"""
import argparse

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs
from mogym.surf import surf

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=[t for t in config.TASKS if "surf" in config.TASKS[t]])
ap.add_argument("--values", type=int, nargs="*", help="numbers of segments N (default: the plotted ones)")
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
a = ap.parse_args()
spec = config.TASKS[a.task]["surf"]
model = envs.build(a.task)
for n in a.values or spec["values"]:
    stem = _setup.Path(a.results) / a.task / "surf" / f"N{n}"
    if stem.with_suffix(".json").exists():
        print(f"{a.task} SURF N={n}: exists, skipped")
        continue
    surf(model, n, stem, rounds=config.MAX_ROUNDS, inner_steps=spec["K_S"], inner_lr=spec["lr"],
         plateau=config.SURF_RULE, adam_keep_tol=config.ADAM_KEEP_TOL)
