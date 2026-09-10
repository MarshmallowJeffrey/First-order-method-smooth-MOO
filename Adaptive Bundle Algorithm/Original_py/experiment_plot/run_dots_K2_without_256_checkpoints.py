"""run_dots_K2_without_256_checkpoints.py — K = 2 "dots" campaign
(PI review Sep 7 2026; user decisions Sep 8-9).  The baseline-resolution
pre-selection (S3 ladders, paper Figure 2) is dropped: EVERY uniform
resolution r and EVERY SURF slot count N is run at the main budget and
becomes one dot of the new Figure 3, next to the adaptive-CCP curve.

Same executors, core, seed, budget accounting and output naming as
``--stage main`` of run_surf_compare_K2_without_256_checkpoints.py, so
the three existing main legs (adaptive_ccp, uniform r=40, SURF N=30,
seed 41) are reused through resume-skip and nothing is recomputed.

Setting (user sign-off Sep 9): core adam(alpha=1e-3, beta2=0.9);
ridge mu = 1e-3; SURF w_min = 0.05 (the 1/(2N) trim failed the Sep-8
smoke, see smoke_surf_wmin_K2_without_256_checkpoints.py); B = 20,000;
eval_every = 250; audit grid 200,001; s = 5; sampler seed 41 only;
r, N in {10, 20, 30, 40, 50, 70, 90, 120, 140}.

NEW FILE.  No existing file is modified.  Runs are strictly serial
(their wall clock lands on the CPU axis).

Usage:
    python run_dots_K2_without_256_checkpoints.py                 # all legs, resume-skip
    python run_dots_K2_without_256_checkpoints.py --only uniform
    python run_dots_K2_without_256_checkpoints.py --only surf
"""

from __future__ import annotations

import argparse
import os
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
    _load_or_run,
    _main_args,
)

CORE_TAG = "adam_1e-3_b0.9"
DOT_RS = [10, 20, 30, 40, 50, 70, 90, 120, 140]
DOT_NS = [10, 20, 30, 40, 50, 70, 90, 120, 140]
SEED = 41


def _final(sm):
    return float(sm["audited_gn_norm_history"][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["uniform", "surf"], default=None)
    ap.add_argument("--force", action="store_true",
                    help="recompute even if summary.json exists")
    a = ap.parse_args()
    core_tag, sname, scfg = next(c for c in CORES if c[0] == CORE_TAG)
    args = _main_args()
    core_home = MAIN_HOME / core_tag
    core_home.mkdir(parents=True, exist_ok=True)
    print(f"[dots] core={core_tag} mu={CAMPAIGN_MU} w_min={SURF_W_MIN} "
          f"B={args.budget:g} eval_every={args.eval_every:g} "
          f"audit_grid={args.audit_grid} s={args.s} seed={SEED}",
          flush=True)
    t_all = time.time()
    if a.only in (None, "uniform"):
        for r in DOT_RS:
            grid = _sort_grid_for_warmstart(_uniform_simplex_grid(2, r))
            out = core_home / f"uniform_r{r}_seed{SEED}"
            t0 = time.time()
            sm = _load_or_run(out, lambda od, g=grid, rr=r: _run_leg_pair_stepper(
                "baseline", _baseline_policy(g), PAIR, dict(S2_CFG), args,
                od, {"r": rr, "core": core_tag},
                stepper_name=sname, stepper_cfg=scfg,
                sampler_seed=SEED, mu=CAMPAIGN_MU), a.force)
            print(f"[dots] uniform r={r:3d}: final worst GN (norm) = "
                  f"{_final(sm):.4e}   leg {time.time() - t0:.0f}s, "
                  f"total {time.time() - t_all:.0f}s", flush=True)
    if a.only in (None, "surf"):
        for N in DOT_NS:
            out = core_home / f"surf_N{N}_seed{SEED}"
            t0 = time.time()
            sm = _load_or_run(out, lambda od, NN=N: run_surf_leg(
                PAIR, dict(S2_CFG), args, od, {"core": core_tag},
                N=NN, stepper_name=sname, stepper_cfg=scfg,
                sampler_seed=SEED, mu=CAMPAIGN_MU, w_min=SURF_W_MIN),
                a.force)
            print(f"[dots] surf N={N:3d}: final worst GN (norm) = "
                  f"{_final(sm):.4e}   leg {time.time() - t0:.0f}s, "
                  f"total {time.time() - t_all:.0f}s", flush=True)
    print(f"[dots] ALL DONE in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
