"""One run ("leg") of an experiment: build the problem, train, audit, write summary.json and grams.npz."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .methods import SAME_LAMBDA_TOL, run_adaptive, run_surf, run_uniform
from .meter import CERT_GAP, audit_k3_certified, gn_k2_prefixes
from .objective import make_problem
from .steppers import STEP_RULE_BY_TAG


def leg_name(method: str, param, seed: int, step_rule: str | None = None) -> str:
    base = {"adaptive": "adaptive", "uniform": f"uniform_r{param}", "surf": f"surf_N{param}"}[method]
    return (f"{step_rule}_" if step_rule else "") + f"{base}_seed{seed}"


def run_leg(K, method, param, seed, out_dir, *, budget=C.BUDGET, step_rule=C.STEP_RULE, schedule=None,
            audit_grid=C.AUDIT_GRID_K2, device="cpu", threads=None, start="chain", reset="new_lambda",
            selector=None):
    """Runs one leg and writes <out_dir>/summary.json and grams.npz; returns the summary.  start / reset: the
    adaptive method's warm start (values other than "chain" / "new_lambda" are recorded); selector: its lambda search,
    "ccp_cg" or "envelope" (K = 2; see methods.run_adaptive), by default the one of the paper (config.SELECTOR)."""
    selector = selector or C.SELECTOR[K]
    if threads:
        torch.set_num_threads(int(threads))
    out_dir = Path(out_dir)
    t_build = time.time()
    problem = make_problem(C.DIGITS[K], C.RHO[K], C.mu_of(K), batch_size=C.BATCH_SIZE, sampler_seed=seed,
                           n_probes=C.N_PROBES, probe_seed=C.PROBE_SEED, device=device)
    rule = STEP_RULE_BY_TAG[step_rule]
    schedule = schedule or C.SCHEDULE[K]
    tag = f"K={K} {leg_name(method, param, seed)} B={budget:g}"
    print(f"[{tag}] problem built in {time.time() - t_build:.1f}s (n={problem.n}, d={problem.d}, "
          f"{problem.device_description})", flush=True)
    if method == "adaptive":
        ccp_config = C.CCP_CG_DECISIONS if selector == "ccp_cg" else None
        rec = run_adaptive(problem, rule, budget, schedule, C.SEGMENTS, ccp_config, start=start, reset=reset,
                           selector=selector)
    elif method == "uniform":
        rec = run_uniform(problem, rule, budget, schedule, int(param), C.SEGMENTS)
    elif method == "surf":
        rec = run_surf(problem, rule, budget, schedule, int(param), C.SEGMENTS)
    else:
        raise ValueError(method)
    print(f"[{tag}] trained: {len(rec.grams) - 1} segments, {rec.rejections} rejected, "
          f"{rec.wall_seconds:.0f}s (lambda decisions {rec.decision_seconds:.0f}s)", flush=True)

    t_audit = time.time()
    Ms = np.asarray(rec.grams, dtype=float)
    summary = {"K": K, "digits": list(C.DIGITS[K]), "method": method, "param": param, "seed": int(seed),
               "budget": float(budget), "step_rule": step_rule, "segments_per_visit": C.SEGMENTS,
               "rho": C.RHO[K], "mu": C.mu_of(K), "n": problem.n, "d": problem.d,
               "L": [float(v) for v in problem.L], "device": problem.device_description,
               "torch_threads": int(torch.get_num_threads()),
               "spent": float(rec.budget.spent()), "segments": len(rec.grams) - 1, "rejections": rec.rejections,
               "wall_seconds": rec.wall_seconds, "decision_seconds": rec.decision_seconds,
               "ck_grads": rec.ck_grads, "ck_wall": rec.ck_wall, "ck_m": rec.ck_m}
    if K == 2:
        res = gn_k2_prefixes(Ms, rec.ck_m, audit_grid)
        gn2 = [float(v) for v, _, _ in res]
        summary.update(audit="exact", audit_grid=int(audit_grid), audit_w=[float(w) for _, w, _ in res],
                       audit_upper=[float(u) for _, _, u in res])
    else:                                            # certified interval [lower, upper] for GNS* (abm/certify.py)
        gn2, upper, lams, certified = audit_k3_certified(Ms, rec.ck_m)
        summary.update(audit="certified", audit_gap=CERT_GAP, audit_lam=lams,
                       audit_gn_upper=[float(np.sqrt(max(v, 0.0))) for v in upper],
                       audit_uncertified=int(sum(not c for c in certified)))
    summary["audit_gn2"] = gn2
    summary["audit_gn"] = [float(np.sqrt(max(v, 0.0))) for v in gn2]
    summary["audit_seconds"] = time.time() - t_audit
    if method == "surf":
        summary["surf_rounds"] = rec.surf_rounds
    if method == "adaptive":
        summary["selector"] = selector
    if method == "adaptive" and selector == "ccp_cg":
        summary["selector_config"] = dataclasses.asdict(C.CCP_CG_DECISIONS)
        summary["same_lambda_tol"] = SAME_LAMBDA_TOL["ccp_cg"]
        summary["selector_stats"] = rec.selector_stats
    if method == "adaptive" and rec.certified:
        summary["certified"] = True
    if method == "adaptive" and (start, reset) != ("chain", "new_lambda"):
        si, ci = np.asarray(rec.start_index), np.asarray(rec.chain_index)
        summary.update(start=start, reset=reset, decisions=int(si.size),
                       start_moved=int(np.count_nonzero(si != ci)),      # decisions not starting at the last accepted point
                       start_index=si.tolist())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / "grams.npz", gram_stack=Ms, fvals=np.asarray(rec.fvals),
                        seg_grads=np.asarray(rec.seg_grads, dtype=float),
                        seg_lams=np.asarray(rec.seg_lams, dtype=float))
    print(f"[{tag}] worst-case GN {summary['audit_gn'][-1]:.4e} (audit {summary['audit_seconds']:.0f}s) -> {out_dir}",
          flush=True)
    return summary
