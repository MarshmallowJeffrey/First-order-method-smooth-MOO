"""run_ccp_smoke_sampler_without_256_checkpoints.py — seed-sampler
ablation for the CCP λ-solver: Exp(1)-normalised vs scrambled Sobol.

NEW FILE (Aug 9, 2026).  Protocol: Note/Aug_8_note.md §4.  Decides the
default ``CCPConfig.seed_sampler``.  Static-bundle, single-round solves
(fresh solver per repetition, empty pool, adaptive schedule off); the
two arms share screening, dedup, τ and the deterministic seeds
(vertices + λ_A) — only the random batch differs.

Instances
---------
* synthetic: K ∈ {3, 6} × m ∈ {30, 200} × 3 seeds; families alternate
  between plain Gaussian gradients ("gauss") and planted-cancellation
  geometry ("cancel", an exact per-point cancelling weight + noise —
  the late-stage envelope shape that stresses multistart).
* K = 2: m ∈ {30, 200} × 2 seeds with ``exact_gns_K2`` ground truth.
* real: prefixes (early / mid / late) of a short K = 5 bandit-toy
  adaptive run — Gram stacks Q_i only, never the CCP payoff M^(c).

Reference value per instance: exact (K = 2) or the max over every
observation including one heavy run per sampler (N = 2^15, r = 20).
miss = relative shortfall > 1e-6.  Pre-registered decision rule in
``decide()`` (miss rates + McNemar discordant-pair test at N = 2048 and
N = 128).

Outputs (output/ccp_smoke_sampler/):  results.csv, refs.json,
summary.md, instances/*.npz.

Usage:
    python run_ccp_smoke_sampler_without_256_checkpoints.py           # full
    python run_ccp_smoke_sampler_without_256_checkpoints.py --quick   # tiny
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import binomtest

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from ccp_lambda_solver import (
    CCPConfig, CCPLambdaSolver, exact_gns_K2, sample_simplex_exp,
)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "CCP/ccp_smoke_sampler"
SAMPLERS = ("exp", "sobol")
MISS_REL = 1e-6
BASE_SEED = 20260809


# =====================================================================
#  Instances
# =====================================================================
def _make_J(rng, m, K, d, family):
    J = rng.normal(size=(m, K, d))
    if family == "cancel":
        W = sample_simplex_exp(m, K, rng)
        J = J - np.einsum('ik,id->ikd', np.ones((m, K)),
                          np.einsum('ik,ikd->id', W, J))
        J = J + 0.05 * rng.normal(size=J.shape)
    return J


def synthetic_instance(K, m, seed):
    family = "cancel" if seed % 2 else "gauss"
    rng = np.random.default_rng([BASE_SEED, K, m, seed])
    J = _make_J(rng, m, K, d=8 * K, family=family)
    Q = np.einsum('ikd,ild->ikl', J, J)
    return {"name": f"syn_K{K}_m{m}_s{seed}_{family}", "Q": Q,
            "K": K, "m": m, "family": family, "source": "synthetic",
            "exact": None}


def k2_instance(m, seed):
    family = "cancel" if seed % 2 else "gauss"
    rng = np.random.default_rng([BASE_SEED, 2, m, seed, 77])
    J = _make_J(rng, m, 2, d=16, family=family)
    Q = np.einsum('ikd,ild->ikl', J, J)
    exact_val, _ = exact_gns_K2(Q)
    return {"name": f"k2_m{m}_s{seed}_{family}", "Q": Q,
            "K": 2, "m": m, "family": family, "source": "k2_exact",
            "exact": float(exact_val)}


def real_instances() -> List[Dict]:
    """Early/mid/late Gram-stack prefixes from a short K=5 bandit-toy
    adaptive run (real objective geometry; Q_i only, per Aug-8 note)."""
    from algorithm_fast_without_256_checkpoints import algorithm_adaptive_fast
    from objectives_bandit_toy import (BanditStochOracle, calibrate_L,
                                       make_bandit_toy_K)
    problem = make_bandit_toy_K(K=5, T=1000, noise_std=0.5, data_seed=7,
                                A=5, tau=0.05, alpha=4.0)
    cal = calibrate_L(problem, safety=1.5)
    L = np.asarray(cal["L"], dtype=float)
    oracle = BanditStochOracle(problem, batch_size=256, seed=41)
    res = algorithm_adaptive_fast(
        problem.K, problem.d, problem.objectives, problem.grad_objectives,
        L, np.zeros(problem.d),
        stoch_oracle=oracle, epsilon=1e-3, max_outer=12,
        msvrg_max_segments=8, msvrg_rel_target=0.25,
        return_pre_prune=True, joint_oracle=problem.joint_oracle,
    )
    gram_full = np.asarray(res["pre_prune"]["gram_stack"], dtype=float)
    m_hist = [mm for mm in res["m_history"] if mm >= 2]
    if not m_hist:
        m_hist = [gram_full.shape[0]]
    picks = sorted({m_hist[0],
                    m_hist[len(m_hist) // 2],
                    gram_full.shape[0]})
    stages = ["early", "mid", "late"][:len(picks)]
    out = []
    for stage, mm in zip(stages, picks):
        out.append({"name": f"real_K5_{stage}_m{mm}",
                    "Q": gram_full[:mm].copy(), "K": 5, "m": int(mm),
                    "family": "bandit_toy", "source": "real_K5",
                    "exact": None})
    return out


# =====================================================================
#  Solves
# =====================================================================
def one_solve(Q, K, sampler, N, seed, r=10):
    cfg = CCPConfig(N0=int(N), r=r, seed_sampler=sampler,
                    adaptive_seed_schedule=False, seed=int(seed))
    solver = CCPLambdaSolver(K, cfg)
    t0 = time.perf_counter()
    val, _lam = solver.solve(Q)
    return {"phi": float(val),
            "wall_s": time.perf_counter() - t0,
            "ccp_iters": solver.stats_last["ccp_iters"],
            "sandwich": solver.stats_last["sandwich_closed"]}


def _seed_for(inst_idx, sampler, N, rep):
    ss = np.random.SeedSequence(
        [BASE_SEED, inst_idx, SAMPLERS.index(sampler), int(N), rep])
    return int(ss.generate_state(1)[0])


# =====================================================================
#  Analysis
# =====================================================================
def mcnemar(miss_exp: np.ndarray, miss_sob: np.ndarray):
    """Paired discordant test.  Returns (b, c, p): b = exp misses where
    sobol hits, c = the reverse; p from a two-sided binomial test."""
    b = int(np.sum(miss_exp & ~miss_sob))
    c = int(np.sum(~miss_exp & miss_sob))
    p = 1.0 if (b + c) == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, p


def decide(agg: Dict[int, Dict]) -> str:
    """Pre-registered rule (Note/Aug_8_note.md §4)."""
    n_prod, n_shrink = 2048, 128
    verdicts = {}
    for N in (n_prod, n_shrink):
        a = agg[N]
        both_low = a["miss_rate_exp"] < 0.02 and a["miss_rate_sobol"] < 0.02
        sig = a["p"] < 0.05 and a["b"] != a["c"]
        sobol_better = sig and a["c"] < a["b"]
        verdicts[N] = (both_low, sig, sobol_better)
    prod_low, prod_sig, prod_sobol = verdicts[n_prod]
    shr_low, shr_sig, shr_sobol = verdicts[n_shrink]
    if prod_sobol:
        return ("sobol — Sobol is significantly better at the production "
                "size; make it the default")
    if shr_sobol and not prod_sig:
        return ("exp — Sobol wins only at N=128; keep Exp(1) and raise the "
            "adaptive-schedule floor 10r -> 20r instead")
    if prod_low and shr_low and not prod_sig and not shr_sig:
        return ("exp — indistinguishable arms at both sizes; Exp(1) wins "
                "on simplicity")
    return ("exp — no significant Sobol advantage detected; Exp(1) by "
            "default (inspect summary tables)")


# =====================================================================
#  Driver
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny validation grid (few instances, 5 reps)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()

    t_run0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "instances").mkdir(exist_ok=True)

    instances: List[Dict] = []
    for K in (3, 6):
        for m in (30, 200):
            for s in range(3):
                instances.append(synthetic_instance(K, m, s))
    for m in (30, 200):
        for s in range(2):
            instances.append(k2_instance(m, s))
    if not args.skip_real:
        try:
            instances.extend(real_instances())
        except Exception as exc:      # keep the ablation runnable
            warnings.warn(f"real-instance harvest failed ({exc!r}); "
                          "continuing with synthetic + K2 only.")
    if args.quick:
        instances = instances[:2] + [i for i in instances
                                     if i["source"] == "k2_exact"][:1]

    reps = 5 if args.quick else args.reps
    N_grid = (128, 512) if args.quick else (128, 512, 2048)
    N_heavy = 4096 if args.quick else 32768

    rows: List[Dict] = []
    refs: Dict[str, Dict] = {}
    for idx, inst in enumerate(instances):
        Q, K, name = inst["Q"], inst["K"], inst["name"]
        t_inst = time.perf_counter()
        best_seen = -np.inf
        for sampler in SAMPLERS:
            for N in N_grid:
                for rep in range(reps):
                    out = one_solve(Q, K, sampler, N,
                                    _seed_for(idx, sampler, N, rep))
                    best_seen = max(best_seen, out["phi"])
                    rows.append({"instance": name, "source": inst["source"],
                                 "family": inst["family"], "K": K,
                                 "m": inst["m"], "sampler": sampler,
                                 "N": N, "rep": rep, **out})
        heavy = {s: one_solve(Q, K, s, N_heavy,
                              _seed_for(idx, s, N_heavy, 0), r=20)["phi"]
                 for s in SAMPLERS}
        best_seen = max(best_seen, *heavy.values())
        if inst["exact"] is not None:
            ref, ref_src = inst["exact"], "exact_K2"
            overshoot = best_seen - ref
            if overshoot > 1e-8 * max(1.0, abs(ref)):
                warnings.warn(f"{name}: observed value exceeds the exact "
                              f"GNS by {overshoot:.3e} — investigate.")
        else:
            ref, ref_src = best_seen, "best_observed_incl_heavy"
        refs[name] = {"ref": float(ref), "ref_source": ref_src,
                      "heavy_exp": heavy["exp"], "heavy_sobol": heavy["sobol"],
                      "K": K, "m": inst["m"], "family": inst["family"]}
        np.savez_compressed(
            OUT_DIR / "instances" / f"{name}.npz", Q=Q,
            meta=json.dumps({"name": name, **{k: inst[k] for k in
                                              ("K", "m", "family", "source")},
                             "ref": float(ref), "ref_source": ref_src}))
        print(f"[{idx + 1:2d}/{len(instances)}] {name:28s} ref={ref:.6e} "
              f"({ref_src})  {time.perf_counter() - t_inst:.1f}s", flush=True)

    # regret / miss vs the final refs
    for row in rows:
        ref = refs[row["instance"]]["ref"]
        row["regret_rel"] = (ref - row["phi"]) / max(1.0, abs(ref))
        row["miss"] = int(row["regret_rel"] > MISS_REL)

    with open(OUT_DIR / "results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(OUT_DIR / "refs.json", "w") as fh:
        json.dump(refs, fh, indent=2)

    # ---- aggregate + decide -----------------------------------------
    agg: Dict[int, Dict] = {}
    lines = ["# CCP seed-sampler smoke test — Exp(1) vs scrambled Sobol",
             "",
             f"Generated by run_ccp_smoke_sampler_without_256_checkpoints.py "
             f"(reps={reps}, quick={args.quick}); protocol Note/Aug_8_note.md §4.",
             "",
             f"{len(instances)} instances; miss = relative shortfall vs ref "
             f"> {MISS_REL:g}; refs in refs.json.",
             ""]
    for N in N_grid:
        sub = [row for row in rows if row["N"] == N]
        keys = sorted({(row["instance"], row["rep"]) for row in sub})
        by = {(row["instance"], row["rep"], row["sampler"]): row for row in sub}
        me = np.array([bool(by[k + ("exp",)]["miss"]) for k in keys])
        ms = np.array([bool(by[k + ("sobol",)]["miss"]) for k in keys])
        re_ = np.array([by[k + ("exp",)]["regret_rel"] for k in keys])
        rs = np.array([by[k + ("sobol",)]["regret_rel"] for k in keys])
        b, c, p = mcnemar(me, ms)
        agg[N] = {"miss_rate_exp": me.mean(), "miss_rate_sobol": ms.mean(),
                  "b": b, "c": c, "p": p}
        lines += [f"## N = {N}", "",
                  "| sampler | miss rate | mean rel regret | p95 rel regret |",
                  "|---|---|---|---|",
                  f"| exp   | {me.mean():.3%} ({me.sum()}/{len(me)}) | "
                  f"{re_.mean():.3e} | {np.percentile(re_, 95):.3e} |",
                  f"| sobol | {ms.mean():.3%} ({ms.sum()}/{len(ms)}) | "
                  f"{rs.mean():.3e} | {np.percentile(rs, 95):.3e} |",
                  "",
                  f"McNemar discordant pairs: exp-only-miss b={b}, "
                  f"sobol-only-miss c={c}, two-sided p={p:.4f}", ""]

    if not args.quick and all(N in agg for N in (128, 2048)):
        decision = decide(agg)
    else:
        decision = "no decision on --quick grid (validation run only)"
    lines += ["## Decision", "", decision, "",
              f"Total wall time: {time.perf_counter() - t_run0:.1f}s"]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[-6:]), flush=True)
    print(f"\nOutputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
