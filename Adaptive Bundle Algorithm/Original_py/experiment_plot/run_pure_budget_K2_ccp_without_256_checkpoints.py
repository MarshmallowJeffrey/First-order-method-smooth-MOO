"""run_pure_budget_K2_ccp_without_256_checkpoints.py — the CCP targeting
leg of the K = 2 pure fixed-budget experiment.

NEW FILE (Aug 9, 2026).  Design: Note/Aug_8_note.md + the Aug-9
comparison Q&A.  No existing file is modified.

Same protocol as ``run_pure_budget_K2_without_256_checkpoints.py``
(shared segment unit, shared s, chain warm start, stop = budget, no
tolerance parameter anywhere) — the ENTIRE executor, instance configs,
exact 1-D audit and summary format are imported from that runner.  The
only new ingredient is the next-lambda policy:

    next lambda = CCPLambdaSolver.solve(delivered Gram stack)

i.e. the multistart convex-concave procedure of gns-ccp.pdf (vertices +
lambda_A + carried pool + Exp(1) random seeds, screen -> r polishes,
warm-started HiGHS game LPs), replacing the strict multistart IPOPT
search.  Decision time is charged to the CPU axis exactly like the
IPOPT leg's targeting.  epsilon is None here (pure budget): the CCP
stop is the relative tau only.

Leg naming: ``adaptive_s{s}_ccp`` under the same budget home, with
``policy = "adaptive_ccp"`` in the summary — the original runner's
--figure filters on policy == "adaptive"/"baseline", so old figures
ignore this leg; the three-family comparison gets its own new figure
script.

Usage:
    python run_pure_budget_K2_ccp_without_256_checkpoints.py            # B20000 leg
    python run_pure_budget_K2_ccp_without_256_checkpoints.py --smoke    # tiny check
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from run_pure_budget_K2_without_256_checkpoints import (  # noqa: E402
    K2_HOME,
    SMOKE_INSTANCE,
    TRIAL_INSTANCE,
    _run_leg,
    exact_gn_1d,
)
from run_experiments import _json_ready  # noqa: E402
from ccp_lambda_solver import CCPConfig, CCPLambdaSolver


def _ccp_policy(K: int, config: CCPConfig, stats_out: list):
    """next_lam closure: one stateful CCP solver per leg (cross-decision
    pool lives inside it; prev_lam is ignored — the pool subsumes it)."""
    solver = CCPLambdaSolver(K, config)

    def nxt(grams, fvals, prev_lam):
        _val, lam = solver.solve(np.asarray(grams, dtype=float))
        stats_out.append(solver.stats_last)
        return lam

    return nxt


def _stats_block(stats: list) -> dict:
    """Per-decision CCP telemetry -> compact JSON columns + totals."""
    cols = ("phi_best", "rho", "n_new_used", "n_restarts", "ccp_iters",
            "lp_simplex_iters", "pool_size", "n_distinct_before_cap",
            "n_dropped_by_cap", "sandwich_closed", "pool_collapse",
            "lambda_search_wall_time", "m")
    block = {c: [s.get(c) for s in stats] for c in cols}
    block["pool_cap"] = stats[-1]["pool_cap"] if stats else None
    block["n_decisions"] = len(stats)
    block["sandwich_closed_total"] = int(sum(bool(s["sandwich_closed"])
                                            for s in stats))
    block["ccp_iters_total"] = int(sum(s["ccp_iters"] for s in stats))
    block["decision_wall_total"] = float(sum(s["lambda_search_wall_time"]
                                             for s in stats))
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--budget", type=float, default=20_000.0)
    parser.add_argument("--eval-every", type=float, default=250.0)
    parser.add_argument("--audit-grid", type=int, default=200_001)
    parser.add_argument("--ccp-N0", type=int, default=2000)
    parser.add_argument("--ccp-r", type=int, default=10)
    parser.add_argument("--ccp-seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.budget, args.eval_every = 400.0, 25.0
        args.audit_grid, args.s = 20_001, 2
        cfg = dict(SMOKE_INSTANCE)
        home = K2_HOME / "B400_SMOKE"
    else:
        cfg = dict(TRIAL_INSTANCE)
        home = K2_HOME / f"B{args.budget:.0f}"
    home.mkdir(parents=True, exist_ok=True)

    out_dir = home / f"adaptive_s{args.s}_ccp"
    if (out_dir / "summary.json").exists() and not args.force:
        print(f"[skip] {out_dir} already has a summary (use --force)",
              flush=True)
        return

    ccp_cfg = CCPConfig(N0=args.ccp_N0, r=args.ccp_r, seed=args.ccp_seed,
                        seed_sampler="exp", adaptive_seed_schedule=False)
    stats: list = []
    summary = _run_leg("adaptive_ccp",
                       _ccp_policy(cfg["K"], ccp_cfg, stats),
                       cfg, args, out_dir, {"ccp_config": vars(ccp_cfg)})

    # augment the summary with the CCP telemetry (Aug-8 note log fields)
    summary["ccp"] = _stats_block(stats)
    (out_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    dec = summary["ccp"]["decision_wall_total"]
    print(f"[adaptive_ccp] decisions={summary['n_decisions']} "
          f"ccp_decision_wall={dec:.2f}s "
          f"(runner-timed decision_seconds={summary['decision_seconds']:.2f}s) "
          f"sandwich_closed={summary['ccp']['sandwich_closed_total']}"
          f"/{summary['ccp']['n_decisions']}", flush=True)

    if args.smoke:
        # cross-checks in the spirit of the original runner's --smoke
        assert summary["grad_equiv_total"] <= args.budget + 1e-6
        hist = summary["audited_gn_history"]
        assert all(hist[i + 1] <= hist[i] + 1e-12
                   for i in range(len(hist) - 1)), "audit not monotone"
        npz = np.load(out_dir / "grams.npz")
        Ms = np.asarray(npz["gram_stack"], dtype=float)
        # the chosen lambdas must lie on the simplex
        lams = np.asarray(npz["lam_history"], dtype=float)
        assert np.all(lams >= -1e-12)
        assert np.allclose(lams.sum(axis=1), 1.0, atol=1e-9)
        # targeting value vs exact meter on the final stack: CCP is a
        # lower bound of the true max
        t0 = time.time()
        v_exact, _w, v_ub = exact_gn_1d(Ms, grid_points=args.audit_grid,
                                        certify=True)
        solver = CCPLambdaSolver(2, ccp_cfg)
        v_ccp, _lam = solver.solve(Ms)
        assert v_ccp <= v_ub + 1e-9, f"CCP {v_ccp} above certified {v_ub}"
        assert v_exact >= v_ccp - 1e-9
        rel_gap = (v_exact - v_ccp) / max(1.0, abs(v_exact))
        print(f"[smoke] ccp {v_ccp:.6e} <= exact {v_exact:.6e} <= upper "
              f"{v_ub:.6e} (ccp rel gap {rel_gap:.2e}, "
              f"{time.time() - t0:.1f}s)  OK", flush=True)
        print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
