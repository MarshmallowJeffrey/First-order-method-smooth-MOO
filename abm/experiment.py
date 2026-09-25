"""One run ("leg") of an experiment: build the problem, train, audit, write summary.json and grams.npz."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .methods import run_adaptive, run_surf, run_uniform
from .meter import audit_k3, gn_k2_prefixes
from .objective import make_problem
from .steppers import STEP_RULE_BY_TAG


def leg_name(method: str, param, seed: int, step_rule: str | None = None) -> str:
    base = {"adaptive": "adaptive", "uniform": f"uniform_r{param}", "surf": f"surf_N{param}"}[method]
    return (f"{step_rule}_" if step_rule else "") + f"{base}_seed{seed}"


def run_leg(K, method, param, seed, out_dir, *, budget=C.BUDGET, step_rule=C.STEP_RULE, schedule=None,
            audit_grid=C.AUDIT_GRID_K2, device="cpu", threads=None):
    """Runs one leg and writes <out_dir>/summary.json and grams.npz; returns the summary."""
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
        rec = run_adaptive(problem, rule, budget, schedule, C.SEGMENTS, C.CCP_DECISIONS)
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
    else:
        gn2, lams = audit_k3(Ms, rec.ck_m, rec.ck_grads, float(budget))
        summary.update(audit="lower bound", audit_lam=lams)
    summary["audit_gn2"] = gn2
    summary["audit_gn"] = [float(np.sqrt(max(v, 0.0))) for v in gn2]
    summary["audit_seconds"] = time.time() - t_audit
    if method == "surf":
        summary["surf_rounds"] = rec.surf_rounds
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / "grams.npz", gram_stack=Ms, fvals=np.asarray(rec.fvals),
                        seg_grads=np.asarray(rec.seg_grads, dtype=float),
                        seg_lams=np.asarray(rec.seg_lams, dtype=float))
    print(f"[{tag}] worst-case GN {summary['audit_gn'][-1]:.4e} (audit {summary['audit_seconds']:.0f}s) -> {out_dir}",
          flush=True)
    return summary
