"""Convergence check of the plotted points: every plotted Uniform / SURF run is repeated with the same
settings but without the stopping rule, for twice as many sweeps / rounds as it used.  The runs are
deterministic, so the first half repeats the stored run; reported: the GN at the stop, after doubling, and
the lowest GN in the extension.

    python scripts/check_doubling.py fishwood [--results results]
"""
import argparse
import json

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, metrics, points
from mogym.surf import surf
from mogym.uniform import uniform_plateau

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=tuple(config.TASKS))
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
a = ap.parse_args()
task = a.task
model = envs.build(task); K = model["K"]
pool = metrics.fixed_stratified_weights() if K > 3 else None
out = _setup.Path(a.results) / task / "doubling"
rows = []
for p in points.points(a.results, task, K):
    m, v = p["method"], p["param"]
    meta = json.loads((_setup.Path(a.results) / task / m.lower() / f"{'r' if m == 'Uniform' else 'N'}{v}.json").read_text())
    stem = out / f"{m}_{v}"
    if not stem.with_suffix(".json").exists():
        if m == "Uniform":
            spec = config.TASKS[task]["uniform"]
            uniform_plateau(model, v, stem, lr=spec["lr"], steps=spec["M"], rule=dict(config.RULE, tol=-1.0),
                            pool=pool, max_sweeps=2 * meta["sweeps"], save_arrays=False)
        else:
            spec = config.TASKS[task]["surf"]
            surf(model, v, stem, rounds=2 * meta["outer_rounds"], inner_steps=spec["K_S"], inner_lr=spec["lr"],
                 plateau=None, adam_keep_tol=config.ADAM_KEEP_TOL, save_arrays=False)
    ext = json.loads(stem.with_suffix(".json").read_text())["checkpoints"][1:]
    n0 = len(meta["checkpoints"]) - 1
    y = points.plotted_gn(meta["checkpoints"][-1], K); y2 = points.plotted_gn(ext[-1], K)
    low = min(points.plotted_gn(c, K) for c in ext[n0:])
    rows.append(dict(method=m, param=v, gn_at_stop=y, gn_after_doubling=y2, change=y2 / y - 1, lowest_change=low / y - 1))
    print(f"{task} {m} {v}: GN at stop {y:.4e}, after doubling {y2:.4e} ({y2 / y - 1:+.2%}), "
          f"lowest in the extension {low / y - 1:+.2%}", flush=True)
(out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
