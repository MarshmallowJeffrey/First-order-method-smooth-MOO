"""CPU time over repeated runs.  The GN of every run is deterministic, its CPU time is not: every configured run
(each Uniform r, each SURF N, GRAB) of the given tasks is run again `repeats - 1` times, interleaved (repeat k runs
every configuration of every task before repeat k + 1), serially with one numerical thread.  Each repeat must
reproduce the stored run exactly (checkpoint Gradient Calls and GN; GRAB: also its weights lambda_t).  Writes
results/<task>/timing.json with the training CPU time at every checkpoint of every repeat (the stored run is
repeat 1), together with the fingerprint of the stored run (mogym.identity.fingerprint) and its checkpoint calls;
make_figure.py then plots the median CPU time and reports the range, and stops if an entry does not belong to the
stored run.

    python scripts/time_repeats.py fishwood fruittree_d6 --repeats 5
"""
import argparse
import json
import shutil

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, identity, metrics
from mogym.adaptive import adaptive
from mogym.surf import surf
from mogym.uniform import uniform

ap = argparse.ArgumentParser()
ap.add_argument("tasks", nargs="+", choices=tuple(config.TASKS))
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
ap.add_argument("--variant", choices=tuple(config.VARIANTS), help="a variant of config.VARIANTS")
a = ap.parse_args()
if a.variant:
    config.apply_variant(a.variant)
res = _setup.Path(a.results)


def runs(task):
    """(key, function that writes the run to a stem) for every configured run of a task."""
    t = config.TASKS[task]; model = envs.build(task); L = config.smoothness(task)
    pool = metrics.fixed_stratified_weights() if model["K"] > 3 else None
    out = []
    for r in t["uniform"]["values"]:
        out.append((f"uniform/r{r}", lambda stem, r=r: uniform(
            model, r, stem, lr=t["uniform"]["lr"], steps=t["uniform"]["M"], rule=config.RULE, every=t["every"],
            budget=t["budget"], L=L, state_tol=config.STATE_TOL, max_sweeps=config.MAX_SWEEPS, pool=pool,
            save_arrays=False)))
    for n in t.get("surf", {}).get("values", []):
        out.append((f"surf/N{n}", lambda stem, n=n: surf(
            model, n, stem, rounds=config.MAX_ROUNDS, inner_steps=t["surf"]["K_S"], inner_lr=t["surf"]["lr"],
            every=t["every"], budget=t["budget"], rule=config.SURF_RULE, state_tol=config.STATE_TOL, save_arrays=False)))
    out.append(("adaptive/adaptive", lambda stem: adaptive(model, L, t["budget"], stem, state_tol=config.STATE_TOL,
                                                           **t["adaptive"])))
    return out


def signature(meta):
    rows = [(c["component_gradients"], c["gn"]) for c in meta["checkpoints"]]
    return json.dumps(rows) + json.dumps(meta.get("lambdas"))


timing = {}
for task in a.tasks:
    timing[task] = {}
    for key, _ in runs(task):
        meta = json.loads((res / task / f"{key}.json").read_text())
        timing[task][key] = dict(run=identity.fingerprint(meta), calls=[c["component_gradients"] for c in meta["checkpoints"]],
                                 cpu=[[c["train_cpu"] for c in meta["checkpoints"]]], signature=signature(meta))
for k in range(2, a.repeats + 1):
    for task in a.tasks:
        tmp = res / task / f"repeat{k}"
        for key, run in runs(task):
            meta = run(tmp / key)
            if signature(meta) != timing[task][key]["signature"]:
                raise RuntimeError(f"{task} {key}: repeat {k} does not reproduce the stored run")
            timing[task][key]["cpu"].append([c["train_cpu"] for c in meta["checkpoints"]])
            print(f"repeat {k} {task} {key}: CPU at B {meta['checkpoints'][-1]['train_cpu']:.2f} s", flush=True)
        shutil.rmtree(tmp)
for task in a.tasks:
    for v in timing[task].values():
        v.pop("signature")
    (res / task / "timing.json").write_text(json.dumps(dict(repeats=a.repeats, runs=timing[task])) + "\n")
print("timing written")
