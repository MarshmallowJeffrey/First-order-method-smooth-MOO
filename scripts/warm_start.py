#!/usr/bin/env python
"""Warm-start ablation (Appendix C.1): where the segments of each decision of the adaptive bundle method start, and
when the step rule's state is reset.  Adam(1e-3, beta2 = 0.9), the three sampling seeds of the step-rule experiment,
10,000 gradient calls on {4,9} (K = 2) and 20,000 on {4,7,9} (K = 3).

    A  the last accepted point              state reset when lambda changes (the runs of the paper)
    B  the last accepted point              state reset at every decision
    C  the bundle point with the lowest F_lambda                                         state reset at every decision
    D  the bundle point with the lowest F_lambda - ||grad F_lambda||^2 / (2 L_lambda)    state reset at every decision
       (the start of Algorithms 2-6, with L_lambda = lambda^T L from the estimated L)

    python scripts/warm_start.py --K 2 --device cuda
    python scripts/warm_start.py --K 3 --device cuda [--variants A,B --seeds 41]

Runs go to runs/warm_start_k<K>/<variant>_seed<s>/.  When all runs of the four variants exist, the summary (final
worst-case gradient norm per seed, mean curves, rejections, decisions that did not start at the last accepted point)
goes to results/warm_start_k<K>.json.  K = 2 uses the exact audits, K = 3 the suffix maximum of its lower bounds.
The adaptive method chooses lambda as in the paper (config.SELECTOR; recorded as `selector`): for K = 2 the exact
envelope since 2026-10-04; the earlier K = 2 runs with CCP are in results/warm_start_k2_ccp.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm import config as C  # noqa: E402
from abm.analysis import suffix_max  # noqa: E402
from abm.experiment import run_leg  # noqa: E402

VARIANTS = {"A": ("chain", "new_lambda"), "B": ("chain", "every_decision"),
            "C": ("lowest_f", "every_decision"), "D": ("paper", "every_decision")}


def board(K, runs_dir, seeds):
    out = {}
    for v, (start, reset) in VARIANTS.items():
        sms = [json.loads((runs_dir / f"{v}_seed{s}" / "summary.json").read_text()) for s in seeds]
        curves = [np.asarray(sm["audit_gn"], float) for sm in sms]
        if K == 3:
            curves = [suffix_max(c) for c in curves]
        n = min(len(c) for c in curves)
        finals = [float(c[-1]) for c in curves]
        out[v] = {"start": start, "reset": reset, "ck_grads": sms[0]["ck_grads"][:n],
                  "gn_mean": np.mean([c[:n] for c in curves], axis=0).tolist(),
                  "final_per_seed": finals, "final_mean": float(np.mean(finals)),
                  "rejections": [int(sm["rejections"]) for sm in sms],
                  "decisions": [sm.get("decisions") or -(-sm["segments"] // C.SEGMENTS) for sm in sms],  # A: s per decision
                  "start_moved": [sm.get("start_moved", 0) for sm in sms],
                  "wall_seconds": [float(sm["wall_seconds"]) for sm in sms]}
    selectors = {json.loads((runs_dir / f"{v}_seed{s}" / "summary.json").read_text()).get("selector", "ccp")
                 for v in VARIANTS for s in seeds}
    if len(selectors) != 1:
        raise ValueError(f"runs with different lambda searches in {runs_dir}: {selectors}")
    return {"K": K, "digits": list(C.DIGITS[K]), "budget": C.WARM_START_BUDGET[K], "step_rule": C.STEP_RULE,
            "seeds": list(seeds), "selector": selectors.pop(), "variants": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, choices=(2, 3), required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in C.STEP_RULE_SEEDS))
    ap.add_argument("--budget", type=float, default=None, help="default: the ablation budget of K (abm/config.py)")
    ap.add_argument("--runs", type=Path, default=None, help="default: runs/warm_start_k<K>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    budget = a.budget or C.WARM_START_BUDGET[a.K]
    runs_dir = a.runs or ROOT / "runs" / f"warm_start_k{a.K}"
    schedule = [(float("inf"), C.WARM_START_CADENCE[a.K])]
    for v in a.variants.split(","):
        start, reset = VARIANTS[v]
        for s in seeds:
            d = runs_dir / f"{v}_seed{s}"
            if not (d / "summary.json").exists():
                run_leg(a.K, "adaptive", None, s, d, budget=budget, schedule=schedule, audit_grid=C.STEP_RULE_AUDIT_GRID,
                        device=a.device, threads=a.threads, start=start, reset=reset)
    all_seeds = [int(s) for s in C.STEP_RULE_SEEDS]
    if budget == C.WARM_START_BUDGET[a.K] and all((runs_dir / f"{v}_seed{s}" / "summary.json").exists()
                                                 for v in VARIANTS for s in all_seeds):
        res = board(a.K, runs_dir, all_seeds)
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / f"warm_start_k{a.K}.json").write_text(json.dumps(res, indent=1))
        for v, r in res["variants"].items():
            print(f"{v}  {r['start']:9s} {r['reset']:15s} mean {r['final_mean']:.4e}  per seed "
                  + "  ".join(f"{x:.4e}" for x in r["final_per_seed"]) + f"  start moved {r['start_moved']}")


if __name__ == "__main__":
    main()
