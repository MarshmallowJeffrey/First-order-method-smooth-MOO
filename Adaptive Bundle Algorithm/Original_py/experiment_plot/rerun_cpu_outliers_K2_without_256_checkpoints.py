"""rerun_cpu_outliers_K2_without_256_checkpoints.py — CPU-axis hygiene for
the K = 2 dots campaign (user instruction Sep 9 2026: fix contaminated
CPU times BEFORE the K = 3 campaign starts).

The CPU axis of the paper's figures is wall-clock time of the leg
(``wall_seconds`` = the segment loop, audit excluded).  Every baseline
leg of the dots campaign runs the same budget B = 20,000 with the same
segment cost per gradient unit, so a clean leg takes the same wall time
up to a few percent; a leg whose wall time exceeds the fastest leg by
more than ``--tol`` (5 %) was slowed by other processes (e.g. the macOS
daemon burst of Sep 9 19:10-19:23) and is re-run.  The trajectory is
seed-determined, so a re-run reproduces the gradient-axis numbers and
only the CPU numbers change.

Procedure per flagged leg (serial, and only when the machine is quiet:
no foreign process above 60 % CPU and foreign total below 150 %, waiting
up to ``--max-wait`` minutes for that):
  1. rename <leg> to <leg>__cpu_contaminated_<k>  (nothing is deleted);
  2. run the leg again into <leg> with the identical call;
  3. if the re-run is SLOWER than the contaminated one, swap them back
     (the faster measurement is the less contaminated one).

Usage:
    python rerun_cpu_outliers_K2_without_256_checkpoints.py --dry-run   # table only
    python rerun_cpu_outliers_K2_without_256_checkpoints.py             # re-run flagged legs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import _layout  # noqa: F401
from baseline_without_256_checkpoints import (  # noqa: E402
    _sort_grid_for_warmstart,
    _uniform_simplex_grid,
)
from run_pure_budget_K6_without_256_checkpoints import (  # noqa: E402
    _baseline_policy,
)
from run_stepper_pre_experiment_K2_without_256_checkpoints import (  # noqa: E402
    PAIR,
    S2_CFG,
    _run_leg_pair_stepper,
)
from baseline_surf_without_256_checkpoints import run_surf_leg  # noqa: E402
from run_surf_compare_K2_without_256_checkpoints import (  # noqa: E402
    CAMPAIGN_MU,
    CORES,
    MAIN_HOME,
    SURF_W_MIN,
    _main_args,
)
from run_dots_K2_without_256_checkpoints import (  # noqa: E402
    CORE_TAG,
    DOT_NS,
    DOT_RS,
    SEED,
)


def _wall(d):
    p = d / "summary.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["wall_seconds"])


def _legs(core_home):
    legs = []
    for r in DOT_RS:
        legs.append(("uniform", r, core_home / f"uniform_r{r}_seed{SEED}"))
    for N in DOT_NS:
        legs.append(("surf", N, core_home / f"surf_N{N}_seed{SEED}"))
    return [(fam, p, d, _wall(d)) for fam, p, d in legs]


def _foreign_cpu():
    """(max single foreign %CPU, foreign total %CPU) — 'foreign' = not this
    process tree."""
    me = os.getpid()
    out = subprocess.run(["ps", "-Ao", "pid,ppid,%cpu,comm", "-r"],
                         capture_output=True, text=True).stdout.splitlines()[1:]
    mx, tot = 0.0, 0.0
    for line in out:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, cpu = int(parts[0]), int(parts[1]), float(parts[2])
        if pid == me or ppid == me:
            continue
        if cpu > 5.0:
            tot += cpu
        mx = max(mx, cpu)
    return mx, tot


def _wait_quiet(max_wait_min, log):
    t0 = time.time()
    while True:
        mx, tot = _foreign_cpu()
        if mx <= 60.0 and tot <= 150.0:
            log(f"machine quiet (max foreign {mx:.0f}%, total {tot:.0f}%)")
            return True
        if time.time() - t0 > 60.0 * max_wait_min:
            log(f"machine still busy after {max_wait_min} min (max foreign "
                f"{mx:.0f}%, total {tot:.0f}%) — proceeding anyway")
            return False
        log(f"machine busy (max foreign {mx:.0f}%, total {tot:.0f}%), waiting")
        time.sleep(60)


def _run(fam, p, out_dir, core):
    core_tag, sname, scfg = core
    args = _main_args()
    if fam == "uniform":
        grid = _sort_grid_for_warmstart(_uniform_simplex_grid(2, p))
        return _run_leg_pair_stepper(
            "baseline", _baseline_policy(grid), PAIR, dict(S2_CFG), args,
            out_dir, {"r": p, "core": core_tag, "cpu_rerun": True},
            stepper_name=sname, stepper_cfg=scfg,
            sampler_seed=SEED, mu=CAMPAIGN_MU)
    return run_surf_leg(
        PAIR, dict(S2_CFG), args, out_dir, {"core": core_tag, "cpu_rerun": True},
        N=p, stepper_name=sname, stepper_cfg=scfg,
        sampler_seed=SEED, mu=CAMPAIGN_MU, w_min=SURF_W_MIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--max-wait", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    core = next(c for c in CORES if c[0] == CORE_TAG)
    core_home = MAIN_HOME / CORE_TAG

    def log(msg):
        print(f"[cpu-rerun {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    legs = _legs(core_home)
    have = [(f, p, d, w) for f, p, d, w in legs if w is not None]
    missing = [(f, p) for f, p, d, w in legs if w is None]
    if missing:
        log(f"legs without summary.json (skipped): {missing}")
    if not have:
        log("nothing to check")
        return
    ref = min(w for _f, _p, _d, w in have)
    log(f"reference (fastest leg) wall = {ref:.1f}s; tolerance {100 * a.tol:.0f}%")
    flagged = []
    for f, p, d, w in have:
        mark = "RERUN" if w > ref * (1.0 + a.tol) else "ok"
        log(f"  {f:8s} {p:4d}  wall {w:7.1f}s  (+{100 * (w / ref - 1):5.1f}%)  {mark}")
        if mark == "RERUN":
            flagged.append((f, p, d, w))
    if a.dry_run or not flagged:
        log(f"{len(flagged)} leg(s) flagged; dry-run={a.dry_run}; done")
        return
    for f, p, d, w_old in flagged:
        k = 1
        while (d.parent / f"{d.name}__cpu_contaminated_{k}").exists():
            k += 1
        backup = d.parent / f"{d.name}__cpu_contaminated_{k}"
        _wait_quiet(a.max_wait, log)
        d.rename(backup)
        log(f"re-running {f} {p}: old wall {w_old:.1f}s -> {backup.name}")
        t0 = time.time()
        _run(f, p, d, core)
        w_new = _wall(d)
        log(f"re-run {f} {p}: new wall {w_new:.1f}s (leg {time.time() - t0:.0f}s)")
        if w_new is None or w_new > w_old:
            log(f"re-run slower than the contaminated run; keeping the old one "
                f"(the faster measurement)")
            tmp = d.parent / f"{d.name}__cpu_rerun_slower_{k}"
            d.rename(tmp)
            backup.rename(d)
    legs = _legs(core_home)
    log("after re-runs:")
    for f, p, d, w in legs:
        if w is not None:
            log(f"  {f:8s} {p:4d}  wall {w:7.1f}s  (+{100 * (w / ref - 1):5.1f}%)")
    log("done")


if __name__ == "__main__":
    main()
