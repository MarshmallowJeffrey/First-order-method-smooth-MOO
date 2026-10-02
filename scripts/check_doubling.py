"""Convergence check of the plotted points: every plotted Uniform / SURF run is repeated with the same
settings but without the stopping rule, for twice as many sweeps / rounds as it used.  The runs are
deterministic, so the first half repeats the stored run; reported: the GN at the stop, after doubling, and
the lowest and highest GN after the stop (every checkpoint).

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
spec_t = config.TASKS[task]
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
            spec = spec_t["uniform"]
            uniform_plateau(model, v, stem, lr=spec["lr"], steps=spec["M"], rule=dict(config.RULE, tol=-1.0),
                            pool=pool, max_sweeps=2 * meta["sweeps"], every=spec_t["every"], budget=spec_t["budget"],
                            save_arrays=False)
        else:
            spec = spec_t["surf"]
            surf(model, v, stem, rounds=2 * meta["outer_rounds"], inner_steps=spec["K_S"], inner_lr=spec["lr"],
                 plateau=None, adam_keep_tol=config.ADAM_KEEP_TOL, every=spec_t["every"], budget=spec_t["budget"],
                 save_arrays=False)
    ext = json.loads(stem.with_suffix(".json").read_text())["checkpoints"]
    stop = meta["stop_calls"]
    y = points.plotted_gn(meta["checkpoints"][-1], K)
    after = [points.plotted_gn(c, K) for c in ext if c["component_gradients"] > stop]
    rows.append(dict(method=m, param=v, gn_at_stop=y, gn_after_doubling=after[-1], change=after[-1] / y - 1,
                     lowest_change=min(after) / y - 1, highest_change=max(after) / y - 1))
    print(f"{task} {m} {v}: GN at stop {y:.4e}, after doubling {after[-1]:.4e} ({after[-1] / y - 1:+.2%}); "
          f"after the stop between {min(after) / y - 1:+.2%} and {max(after) / y - 1:+.2%}", flush=True)
(out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
