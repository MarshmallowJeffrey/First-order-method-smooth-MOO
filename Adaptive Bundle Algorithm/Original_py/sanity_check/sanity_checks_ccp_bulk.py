"""sanity_checks_ccp_bulk.py — equivalence gate for ``CCPLambdaSolverBulk``
(block LP transport) against the original ``CCPLambdaSolver``.

NEW FILE (Sep 10, 2026).  Two levels, on the recorded decision sequence
of a campaign leg (``grams.npz`` + the bundle sizes ``m`` at which the
leg's CCP solver was called):

  LP level (the PASS criterion).  Every payoff matrix Mc the ORIGINAL
  solver hands to its LP inside a window of decisions is logged and
  solved again by the bulk LP (fed the same sequence, so it is warm) and
  by a fresh original LP.  PASS iff the optimal values agree to 1e-10
  relative and every returned lambda is optimal for its LP (envelope
  min_i Mc_i·lam equals t*, sum 1, non-negative).  Whether the lambdas
  coincide is reported (they can legitimately differ on a degenerate LP
  with several optimal vertices).

  Sequence level (reported, not a pass criterion).  Both solvers replay
  the whole decision sequence from the same state.  The multistart
  heuristic contains discrete choices (seed screening order, pool
  de-duplication, the stopping test delta <= tau); a last-bit difference
  in an LP solution (8e-13 relative was measured between the two
  transport paths) eventually flips one of them, after which the two
  paths visit different, equally valid local maximisers — the same
  effect as running the original solver with another HiGHS build or
  CPU.  Reported: the first decision where the paths differ, the
  distribution of the relative phi difference over all decisions (mean,
  median, share of decisions where the bulk value is at least as high),
  and the two wall clocks.

Usage:
    python sanity_checks_ccp_bulk.py                       # K = 10 Aug-9 trial leg (475 decisions)
    python sanity_checks_ccp_bulk.py --leg <dir> --K 3 --max-decisions 600
    python sanity_checks_ccp_bulk.py --no-carry           # bulk solver cold on bundle growth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import _layout  # noqa: F401
from ccp_lambda_solver import CCPConfig, CCPLambdaSolver, _GameLP  # noqa: E402
from ccp_lambda_solver_bulk import CCPLambdaSolverBulk, _GameLPBulk  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent.parent / "output" / "CCP"
DEFAULT_LEG = OUTPUT / "ccp_compare_without_256_checkpoints" / "K10_mnist10k_B55000" / "adaptive_s5_ccp"


def _cfg_of(sm):
    c = sm.get("extra", {}).get("ccp_config", {})
    return CCPConfig(N0=int(c.get("N0", 2000)), r=int(c.get("r", 10)), seed=int(c.get("seed", 0)),
                     seed_sampler=c.get("seed_sampler", "exp"),
                     adaptive_seed_schedule=bool(c.get("adaptive_seed_schedule", False)))


def replay(solver, Q, ms):
    phis, lams, iters, walls = [], [], [], []
    for m in ms:
        t = time.perf_counter()
        phi, lam = solver.solve(Q[:m])
        walls.append(time.perf_counter() - t)
        phis.append(phi)
        lams.append(np.asarray(lam, dtype=float))
        iters.append(int(solver.stats_last["ccp_iters"]))
    return np.asarray(phis), np.asarray(lams), np.asarray(iters), np.asarray(walls)


def lp_level(Q, ms, K, cfg, window, tol):
    """Log the original solver's LP inputs on ``window`` decisions; re-solve."""
    solver = CCPLambdaSolver(K, cfg)
    log = []
    orig = solver.lp.resolve

    def logging_resolve(M):
        out = orig(M)
        log.append((np.array(M, dtype=float, copy=True), float(out[0]), np.asarray(out[1]).copy()))
        return out
    solver.lp.resolve = logging_resolve
    for m in ms[window[0]:window[1]]:
        solver.solve(Q[:m])
    bulk, cold = _GameLPBulk(K), _GameLP(K)
    d_t, n_nonopt, n_lam_diff = 0.0, 0, 0
    for M, t_o, lam_o in log:
        t_b, lam_b = bulk.resolve(M)
        t_c, lam_c = cold.resolve(M)
        d_t = max(d_t, max(abs(t_b - t_o), abs(t_c - t_o)) / max(1.0, abs(t_o)))
        for lam in (lam_o, lam_b, lam_c):
            env = float(np.min(M @ lam))
            if (abs(env - t_o) > 1e-9 * max(1.0, abs(t_o)) or abs(lam.sum() - 1.0) > 1e-9
                    or lam.min() < -1e-12):
                n_nonopt += 1
        if float(np.abs(lam_b - lam_o).sum()) > 1e-9:
            n_lam_diff += 1
    return {"n_lps": len(log), "rows_min": int(min(l[0].shape[0] for l in log)),
            "rows_max": int(max(l[0].shape[0] for l in log)),
            "max_rel_dt": float(d_t), "n_nonoptimal_lambdas": int(n_nonopt),
            "n_lps_with_different_lambda": int(n_lam_diff),
            "pass": bool(d_t <= tol and n_nonopt == 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", default=str(DEFAULT_LEG))
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--lp-window", default="150,200",
                    help="decision index range whose LPs are logged for the LP-level check")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--no-carry", action="store_true",
                    help="bulk solver: do not carry the basis across bundle growth")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    leg = Path(a.leg)
    Q = np.load(leg / "grams.npz")["gram_stack"]
    sm = json.loads((leg / "summary.json").read_text())
    ms = [int(v) for v in sm["ccp"]["m"]]
    if a.max_decisions:
        ms = ms[:a.max_decisions]
    cfg = _cfg_of(sm)
    print(f"[gate] leg={leg.name} K={a.K} decisions={len(ms)} m_max={max(ms)} "
          f"cfg N0={cfg.N0} r={cfg.r} seed={cfg.seed}", flush=True)

    # ---- LP level ------------------------------------------------------
    w = [int(v) for v in a.lp_window.split(",")]
    w[1] = min(w[1], len(ms))
    t0 = time.time()
    lp = lp_level(Q, ms, a.K, cfg, w, a.tol)
    lp["seconds"] = time.time() - t0
    print(f"[gate/LP] {'PASS' if lp['pass'] else 'FAIL'}: {lp['n_lps']} LPs (rows {lp['rows_min']}-"
          f"{lp['rows_max']}), max rel |dt*| = {lp['max_rel_dt']:.2e}, non-optimal lambdas "
          f"{lp['n_nonoptimal_lambdas']}, LPs whose lambda differs {lp['n_lps_with_different_lambda']}",
          flush=True)

    # ---- sequence level ------------------------------------------------
    t0 = time.time()
    old = replay(CCPLambdaSolver(a.K, cfg), Q, ms)
    t_old = time.time() - t0
    t0 = time.time()
    bulk = CCPLambdaSolverBulk(a.K, cfg)
    if a.no_carry:
        bulk.lp._basis_for = lambda m: (bulk.lp._basis if m == bulk.lp._basis_m else None)
    new = replay(bulk, Q, ms)
    t_new = time.time() - t0
    d_phi = np.abs(old[0] - new[0]) / np.maximum(1.0, np.abs(old[0]))
    d_lam = np.abs(old[1] - new[1]).sum(axis=1)
    diff = np.nonzero((d_phi > 1e-9) | (d_lam > 1e-9))[0]
    first = int(diff[0]) if diff.size else None
    rel = (new[0] - old[0]) / np.maximum(1e-300, np.abs(old[0]))
    seq = {"decisions": len(ms), "wall_original_s": t_old, "wall_bulk_s": t_new,
           "first_divergent_decision": first,
           "identical_prefix_fraction": float((first if first is not None else len(ms)) / len(ms)),
           "rel_phi_diff_mean": float(rel.mean()), "rel_phi_diff_median": float(np.median(rel)),
           "rel_phi_diff_max_abs": float(np.abs(rel).max()),
           "share_bulk_at_least_as_high": float((new[0] >= old[0] * (1 - 1e-12)).mean()),
           "final_phi_original": float(old[0][-1]), "final_phi_bulk": float(new[0][-1]),
           "last_decision_seconds_original": float(old[3][-1]),
           "last_decision_seconds_bulk": float(new[3][-1]),
           "warm_solves": int(bulk.lp.n_warm), "cold_solves": int(bulk.lp.n_cold)}
    print(f"[gate/seq] original {t_old:.0f}s, bulk {t_new:.0f}s; identical for the first "
          f"{first if first is not None else len(ms)}/{len(ms)} decisions; rel phi diff mean "
          f"{seq['rel_phi_diff_mean']:+.4f}, median {seq['rel_phi_diff_median']:+.4f}, max |.| "
          f"{seq['rel_phi_diff_max_abs']:.3f}; bulk >= original on {100 * seq['share_bulk_at_least_as_high']:.0f}% "
          f"of decisions; final phi {seq['final_phi_original']:.4e} vs {seq['final_phi_bulk']:.4e}; "
          f"last decision {old[3][-1]:.2f}s vs {new[3][-1]:.2f}s", flush=True)
    ok = lp["pass"]
    print(f"[gate] {'PASS' if ok else 'FAIL'} (criterion: LP level)", flush=True)
    report = {"leg": str(leg), "K": a.K, "carry_basis": not a.no_carry, "lp_level": lp,
              "sequence_level": seq, "pass": ok}
    out = Path(a.out) if a.out else leg / "gate_ccp_bulk.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[gate] report -> {out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
