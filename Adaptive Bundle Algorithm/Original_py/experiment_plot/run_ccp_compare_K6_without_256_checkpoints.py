"""run_ccp_compare_K6_without_256_checkpoints.py — fresh same-machine
rerun of ALL K = 6 pure-budget legs + the new CCP leg (comparison
campaign, option A of the Aug-9 Q&A).

NEW FILE (Aug 9, 2026).  Mirror of
``run_ccp_compare_K2_without_256_checkpoints.py`` for the K = 6 home.
No existing file or output is modified; the July outputs under
``baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/pure_budget_B80912``
stay untouched.  Campaign home:

    output/ccp_compare_without_256_checkpoints/K6_B80912/

Legs (serial, one process, clean CPU axis) — the July leg set plus CCP:
    baseline_r10_s1, baseline_r10_s5, baseline_r12_s5,
    baseline_r15_s1, baseline_r15_s5, baseline_r20_s5,
    adaptive_s5_ts24  (IPOPT strict targeting),
    adaptive_s5_ccp   (multistart-CCP targeting).

In-run audits stay whatever the original K6 executor does (strict-64);
the method-symmetric two-instrument audit_v2 (max of strict-64 and a
heavy CCP solve) is computed post-hoc from grams.npz by the audit
script and is what the comparison figures read.

Usage:
    python run_ccp_compare_K6_without_256_checkpoints.py            # full (~3 h)
    python run_ccp_compare_K6_without_256_checkpoints.py --smoke    # wiring check
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from run_pure_budget_K6_without_256_checkpoints import (  # noqa: E402
    SMOKE_INSTANCE,
    TRIAL_INSTANCE,
    V2_HOME,
    _adaptive_policy,
    _baseline_policy,
    _run_leg,
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

COMPARE_ROOT = (V2_HOME.parent.parent  # output/ (V2_HOME sits in a group dir since Aug 25)
                / "CCP" / "ccp_compare_without_256_checkpoints")


def _leg_args(smoke: bool, s: int, ts: int) -> argparse.Namespace:
    base = dict(budget=1500.0, eval_every=100.0) if smoke \
        else dict(budget=80_912.0, eval_every=2000.0)
    base.update(s=s, targeting_starts=ts)
    return argparse.Namespace(**base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ccp-seed", type=int, default=0)
    args = parser.parse_args()
    if not ipopt_available():
        raise SystemExit("IPOPT is required for the adaptive ts leg.")

    smoke = args.smoke
    cfg = dict(SMOKE_INSTANCE if smoke else TRIAL_INSTANCE)
    home = COMPARE_ROOT / ("K6_B1500_SMOKE" if smoke else "K6_B80912")
    home.mkdir(parents=True, exist_ok=True)
    baseline_legs = (((4, 2),) if smoke
                     else ((10, 1), (10, 5), (12, 5),
                           (15, 1), (15, 5), (20, 5)))
    s_ad = 2 if smoke else 5
    ts = 8 if smoke else 24
    ccp_cfg = CCPConfig(N0=2000, r=10, seed=args.ccp_seed,
                        seed_sampler="exp", adaptive_seed_schedule=False)

    manifest = {"campaign": "ccp_compare K6 pure budget (Aug 9, 2026)",
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

    for r, s in baseline_legs:
        grid = _sort_grid_for_warmstart(_uniform_simplex_grid(cfg["K"], r))
        _do(f"baseline_r{r:02d}_s{s}",
            lambda od, g=grid, rr=r, ss=s: _run_leg(
                "baseline", _baseline_policy(g), cfg,
                _leg_args(smoke, ss, ts), od, {"r": rr}))
    _do(f"adaptive_s{s_ad}_ts{ts}",
        lambda od: _run_leg(
            "adaptive", _adaptive_policy(cfg["K"], ts), cfg,
            _leg_args(smoke, s_ad, ts), od, {}))

    def _ccp_leg(od):
        stats: list = []
        sm = _run_leg("adaptive_ccp",
                      _ccp_policy(cfg["K"], ccp_cfg, stats),
                      cfg, _leg_args(smoke, s_ad, ts), od,
                      {"ccp_config": vars(ccp_cfg)})
        sm["ccp"] = _stats_block(stats)
        (od / "summary.json").write_text(
            json.dumps(_json_ready(sm), indent=2), encoding="utf-8")
        return sm

    _do(f"adaptive_s{s_ad}_ccp", _ccp_leg)

    manifest["total_wall_seconds"] = time.time() - t_all
    (home / "campaign_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    print(f"[campaign] DONE in {manifest['total_wall_seconds']:.0f}s "
          f"-> {home}", flush=True)

    if smoke:
        for p in home.glob("*/summary.json"):
            sm = json.loads(p.read_text())
            assert np.isfinite(sm["final_audit"]), p
        print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
