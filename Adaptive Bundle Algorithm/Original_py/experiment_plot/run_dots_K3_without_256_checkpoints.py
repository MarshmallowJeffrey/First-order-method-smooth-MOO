"""run_dots_K3_without_256_checkpoints.py — K = 3 "dots" campaign
(PI review Sep 7 2026; user decisions Sep 9): every uniform resolution r
is run at ONE fixed budget B and becomes one dot of the new Figure 5,
next to the adaptive-CCP curve run at the same B.  Same executor, core,
seed, audit instruments and output naming as ``--stage main`` of
run_k3_stepper_campaign_without_256_checkpoints.py; only the budget, the
checkpoint cadence (B / 80) and the r list differ, so the runs live in
their own home K3_HOME/dots_B<B>/<core>/ and nothing existing is touched.

Setting: triple (4, 7, 9); ridge mu = 1e-4; core adam(1e-3, 0.9);
FULL_CFG (batch 1,024, m = 18); s = 5; sampler seed 41; SURF not run
(one-dimensional arc length); r in {10, 20, 25, 30, 35, 40}
(66 / 231 / 351 / 496 / 666 / 861 nodes); B from ``--budget``
(default 100,000; eval_every = B / 80); audit: IPOPT 64 starts + CCP
(N0 = 8192, r = 20) at every checkpoint, dense grid res 500 at the end.

NEW FILE.  Runs are strictly serial (their wall clock lands on the CPU
axis).  Order: adaptive first, then r ascending.

Usage:
    python run_dots_K3_without_256_checkpoints.py --budget 100000
    python run_dots_K3_without_256_checkpoints.py --budget 100000 --only uniform --rs 25,35,40
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

import _layout  # noqa: F401
from run_k3_stepper_campaign_without_256_checkpoints import (  # noqa: E402
    CAMPAIGN_MU,
    CORES,
    FULL_CFG,
    K,
    K3_HOME,
    TRIPLE,
    _Args,
    _baseline_policy,
    _ccp_cfg,
    _ccp_policy,
    _json_ready,
    _load_or_run,
    _run_leg_triple_stepper,
    _sort_grid_for_warmstart,
    _stats_block,
    _uniform_simplex_grid,
)

CORE_TAG = "adam_1e-3_b0.9"
DOT_RS = [10, 20, 25, 30, 35, 40]
SEED = 41
N_CHECKPOINTS = 80


def _final_norm(sm):
    v = sm.get("final_audit")
    if v is None:
        v = sm["audited_gn_history"][-1]
    return float(np.sqrt(max(float(v), 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=100_000.0)
    ap.add_argument("--only", choices=["uniform", "adaptive"], default=None)
    ap.add_argument("--rs", default=None,
                    help="comma-separated subset of resolutions")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rs = [int(v) for v in a.rs.split(",")] if a.rs else DOT_RS
    core_tag, sname, scfg = next(c for c in CORES if c[0] == CORE_TAG)
    args = _Args(budget=float(a.budget),
                 eval_every=float(a.budget) / N_CHECKPOINTS, s=5, smoke=False)
    home = K3_HOME / f"dots_B{int(a.budget)}" / core_tag
    home.mkdir(parents=True, exist_ok=True)
    print(f"[dots-K3] triple={TRIPLE} mu={CAMPAIGN_MU} core={core_tag} "
          f"B={args.budget:g} eval_every={args.eval_every:g} s={args.s} "
          f"seed={SEED} rs={rs} -> {home}", flush=True)
    t_all = time.time()

    if a.only in (None, "adaptive"):
        def _ccp(od):
            stats: list = []
            sm = _run_leg_triple_stepper(
                "adaptive_ccp", _ccp_policy(K, _ccp_cfg(), stats), TRIPLE,
                dict(FULL_CFG), args, od, {"core": core_tag},
                stepper_name=sname, stepper_cfg=scfg, sampler_seed=SEED)
            sm["ccp"] = _stats_block(stats)
            (od / "summary.json").write_text(
                json.dumps(_json_ready(sm), indent=2), encoding="utf-8")
            return sm
        t0 = time.time()
        sm = _load_or_run(home / f"adaptive_ccp_seed{SEED}", _ccp, a.force)
        print(f"[dots-K3] adaptive: final worst GN (norm) = {_final_norm(sm):.4e}"
              f"   leg {time.time() - t0:.0f}s, total {time.time() - t_all:.0f}s",
              flush=True)

    if a.only in (None, "uniform"):
        for r in rs:
            grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, r))
            out = home / f"uniform_r{r}_seed{SEED}"
            t0 = time.time()
            sm = _load_or_run(out, lambda od, g=grid, rr=r: _run_leg_triple_stepper(
                "baseline", _baseline_policy(g), TRIPLE, dict(FULL_CFG), args,
                od, {"r": rr, "core": core_tag, "nodes": int(g.shape[0])},
                stepper_name=sname, stepper_cfg=scfg, sampler_seed=SEED),
                a.force)
            print(f"[dots-K3] uniform r={r:3d} ({grid.shape[0]} nodes): final worst GN "
                  f"(norm) = {_final_norm(sm):.4e}   leg {time.time() - t0:.0f}s, "
                  f"total {time.time() - t_all:.0f}s", flush=True)
    print(f"[dots-K3] ALL DONE in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
