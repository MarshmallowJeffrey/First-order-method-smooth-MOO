#!/usr/bin/env python
"""The step-rule experiment (Appendix C.1): the adaptive bundle method with each of the eleven step rules,
three sampling seeds, 10,000 gradient calls on {4,9}.

    python scripts/step_rules.py --device cuda

Runs go to runs/step_rules_k2/<rule>_adaptive_seed<s>/; the summary (curves and the board: mean and range over the
seeds of the final worst-case gradient norm, from the exact audits) to results/step_rules_k2.json.
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
from abm.experiment import leg_name, run_leg  # noqa: E402
from abm.steppers import STEP_RULES  # noqa: E402

K = 2


def board(runs_dir, seeds):
    rules = {}
    for tag, _, _ in STEP_RULES:
        sms = [json.loads((runs_dir / leg_name("adaptive", None, s, tag) / "summary.json").read_text()) for s in seeds]
        curves = [np.asarray(sm["audit_gn"], float) for sm in sms]
        n = min(len(c) for c in curves)
        finals = [float(c[-1]) for c in curves]
        rules[tag] = {"ck_grads": sms[0]["ck_grads"][:n],
                      "ck_wall_mean": np.mean([np.asarray(sm["ck_wall"][:n], float) for sm in sms], axis=0).tolist(),
                      "gn_mean": np.mean([c[:n] for c in curves], axis=0).tolist(),
                      "final_per_seed": finals, "final_mean": float(np.mean(finals)),
                      "rejections": [int(sm["rejections"]) for sm in sms]}
    ranked = sorted(rules, key=lambda t: rules[t]["final_mean"])
    return {"K": K, "digits": list(C.DIGITS[K]), "budget": C.STEP_RULE_BUDGET, "seeds": list(seeds),
            "ranking": ranked, "rules": rules}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=",".join(str(s) for s in C.STEP_RULE_SEEDS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    runs_dir = ROOT / "runs" / f"step_rules_k{K}"
    schedule = [(float("inf"), C.STEP_RULE_CADENCE)]
    for tag, _, _ in STEP_RULES:
        for s in seeds:
            d = runs_dir / leg_name("adaptive", None, s, tag)
            if not (d / "summary.json").exists():
                run_leg(K, "adaptive", None, s, d, budget=C.STEP_RULE_BUDGET, step_rule=tag, schedule=schedule,
                        audit_grid=C.STEP_RULE_AUDIT_GRID, device=a.device, threads=a.threads)
    res = board(runs_dir, seeds)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"step_rules_k{K}.json").write_text(json.dumps(res, indent=1))
    for i, tag in enumerate(res["ranking"], 1):
        r = res["rules"][tag]
        print(f"{i:2d}  {tag:28s} mean {r['final_mean']:.4e}  range {min(r['final_per_seed']):.4e} - "
              f"{max(r['final_per_seed']):.4e}")


if __name__ == "__main__":
    main()
