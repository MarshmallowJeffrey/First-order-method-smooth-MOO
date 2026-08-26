"""run_ccp_compare_K2_without_256_checkpoints.py — fresh same-machine
rerun of ALL K = 2 pure-budget legs + the new CCP leg (comparison
campaign, option A of the Aug-9 Q&A).

NEW FILE (Aug 9, 2026).  No existing file or output is modified: the
July outputs under pure_budget_K2_without_256_checkpoints/B20000 stay
untouched; this campaign writes to a NEW home

    output/ccp_compare_without_256_checkpoints/K2_B20000/

Legs (serial, one process, clean CPU axis):
    baseline_r10_s5, baseline_r20_s5, baseline_r40_s5, baseline_r80_s5,
    adaptive_s5_ts24          (IPOPT strict targeting; the ts=64 twin is
                               bit-identical at K=2 — Note/Jul_30 §7a —
                               and is not rerun),
    adaptive_s5_ccp           (multistart-CCP targeting).

Everything (executor, exact 1-D audits, summary format) is imported
from the original runners.  Legs skip themselves if their summary
already exists, so the campaign is resumable.

Usage:
    python run_ccp_compare_K2_without_256_checkpoints.py            # full
    python run_ccp_compare_K2_without_256_checkpoints.py --smoke    # wiring check
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from run_pure_budget_K2_without_256_checkpoints import (  # noqa: E402
    K2_HOME,
    SMOKE_INSTANCE,
    TRIAL_INSTANCE,
    _run_leg,
)
from run_pure_budget_K6_without_256_checkpoints import (  # noqa: E402
    _adaptive_policy,
    _baseline_policy,
)
from baseline_without_256_checkpoints import (  # noqa: E402
    _sort_grid_for_warmstart,
    _uniform_simplex_grid,
)
from run_experiments import _json_ready  # noqa: E402
from algorithm_fast_without_256_checkpoints import ipopt_available
from ccp_lambda_solver import CCPConfig
from run_pure_budget_K2_ccp_without_256_checkpoints import (
    _ccp_policy,
    _stats_block,
)

COMPARE_ROOT = (K2_HOME.parent.parent  # output/ (K2_HOME sits in a group dir since Aug 25)
                / "CCP" / "ccp_compare_without_256_checkpoints")


def _leg_args(smoke: bool, s: int, **over) -> argparse.Namespace:
    base = dict(budget=400.0, eval_every=25.0, audit_grid=20_001) if smoke \
        else dict(budget=20_000.0, eval_every=250.0, audit_grid=200_001)
    base.update(s=s, decision_mode="search", targeting_starts=24,
                decision_grid=None)
    base.update(over)
    return argparse.Namespace(**base)


def _run_ccp_leg(cfg, args, out_dir, ccp_cfg) -> dict:
    stats: list = []
    summary = _run_leg("adaptive_ccp", _ccp_policy(cfg["K"], ccp_cfg, stats),
                       cfg, args, out_dir, {"ccp_config": vars(ccp_cfg)})
    summary["ccp"] = _stats_block(stats)
    (out_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ccp-seed", type=int, default=0)
    args = parser.parse_args()
    if not ipopt_available():
        raise SystemExit("IPOPT is required for the adaptive ts leg.")

    smoke = args.smoke
    cfg = dict(SMOKE_INSTANCE if smoke else TRIAL_INSTANCE)
    home = COMPARE_ROOT / ("K2_B400_SMOKE" if smoke else "K2_B20000")
    home.mkdir(parents=True, exist_ok=True)
    s = 2 if smoke else 5
    baseline_rs = (4,) if smoke else (10, 20, 40, 80)
    ts = 8 if smoke else 24
    ccp_cfg = CCPConfig(N0=2000, r=10, seed=args.ccp_seed,
                        seed_sampler="exp", adaptive_seed_schedule=False)

    manifest = {"campaign": "ccp_compare K2 pure budget (Aug 9, 2026)",
                "machine": platform.platform(),
                "processor": platform.processor(),
                "smoke": smoke, "legs": []}
    t_all = time.time()

    def _do(name, fn):
        out_dir = home / name
        if (out_dir / "summary.json").exists():
            print(f"[campaign] skip {name} (summary exists)", flush=True)
            return
        print(f"[campaign] === {name} ===", flush=True)
        t0 = time.time()
        sm = fn(out_dir)
        manifest["legs"].append(
            {"leg": name, "wall_seconds": time.time() - t0,
             "decision_seconds": sm.get("decision_seconds"),
             "final_audit": sm.get("final_audit")})
        (home / "campaign_manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")

    for r in baseline_rs:
        grid = _sort_grid_for_warmstart(_uniform_simplex_grid(cfg["K"], r))
        _do(f"baseline_r{r:02d}_s{s}",
            lambda od, g=grid, rr=r: _run_leg(
                "baseline", _baseline_policy(g), cfg,
                _leg_args(smoke, s), od, {"r": rr}))
    _do(f"adaptive_s{s}_ts{ts}",
        lambda od: _run_leg(
            "adaptive", _adaptive_policy(cfg["K"], ts), cfg,
            _leg_args(smoke, s, targeting_starts=ts), od, {}))
    _do(f"adaptive_s{s}_ccp",
        lambda od: _run_ccp_leg(cfg, _leg_args(smoke, s), od, ccp_cfg))

    manifest["total_wall_seconds"] = time.time() - t_all
    (home / "campaign_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    print(f"[campaign] DONE in {manifest['total_wall_seconds']:.0f}s "
          f"-> {home}", flush=True)

    if smoke:
        legs = sorted(p.parent.name for p in home.glob("*/summary.json"))
        assert f"adaptive_s{s}_ccp" in legs and f"adaptive_s{s}_ts{ts}" in legs
        for p in home.glob("*/summary.json"):
            sm = json.loads(p.read_text())
            assert np.isfinite(sm["final_audit"])
        print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
