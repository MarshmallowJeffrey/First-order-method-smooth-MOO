"""run_ccp_compare_K10_mnist_without_256_checkpoints.py — K = 10 MNIST
trial: adaptive (IPOPT ts24) vs adaptive (CCP) under the pure
fixed-budget protocol, on the patch-softplus objective family.

NEW FILE (Aug 9, 2026).  User-approved trial (Aug-9 Q&A): MNIST subset
(1000/class) + softplus + patch first layer; NO baseline legs; two
adaptive legs only.  No existing file is modified.

The executor replicates the K2/K6 pure-budget shared loop verbatim
(shared segment unit, shared s, chain warm start, decision time on the
CPU axis, stop = budget, no tolerance anywhere) — it must be a copy
because the original executors hard-wire the planted-MLP factory.
Differences, all declared:

* problem factory: ``objectives_mnist_patch.make_mnist_patch``
  (K = 10, d = 8874, s_k ≡ 1, data/init/sampler seeds fixed);
* no in-run audit trajectory: quality is measured post-hoc by
  ``audit_v2_K6_without_256_checkpoints.py --home <this home>``
  (two-instrument max — the K = 10 exact meter does not exist).  The
  summary carries a final CCP-instrument value for the log only.
* IPOPT targeting keeps the ts = 24 cap (K6 convention); the FULL
  structured start set at K = 10 would be 67 points — recorded here
  because the cap-vs-coverage tension is itself a finding.

Home: output/ccp_compare_without_256_checkpoints/K10_mnist10k_B<B>/

Usage:
    python run_ccp_compare_K10_mnist_without_256_checkpoints.py           # trial
    python run_ccp_compare_K10_mnist_without_256_checkpoints.py --smoke   # wiring
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from bundle import validate_oracle_output
from baseline_svrg_certified_without_256_checkpoints import _support_batch
from run_pure_budget_K6_without_256_checkpoints import (
    MAX_SAFEGUARD_RETRIES,
    _Budget,
    _adaptive_policy,
)
from run_experiments import _json_ready
from algorithm_fast_without_256_checkpoints import ipopt_available
from ccp_lambda_solver import CCPConfig, CCPLambdaSolver
from run_pure_budget_K2_ccp_without_256_checkpoints import (
    _ccp_policy,
    _stats_block,
)
from objectives_mnist_patch import make_mnist_patch, make_patch_initial_point

HERE = Path(__file__).resolve().parent
COMPARE_ROOT = (HERE.parent.parent / "output"
                / "CCP/ccp_compare_without_256_checkpoints")
DATA_SEED_NOTE = "MNIST subset, first per_class rows per class (no rng)"
INIT_SEED = 8
SAMPLER_SEED = 41


def _run_leg_k10(policy_name, next_lam, cfg, args, out_dir, extra_cfg):
    """Pure-budget shared executor (K2 replica), MNIST-patch factory."""
    t_build = time.time()
    objectives, gradients, L, joint_oracle, stoch, meta = make_mnist_patch(
        per_class=cfg["per_class"], batch_size=cfg["msvrg_batch"],
        sampler_seed=SAMPLER_SEED, init_seed=INIT_SEED,
        n_probes=cfg["n_probes"], ah16_faithful=cfg["ah16_faithful"],
    )
    K, n, d = meta["K"], meta["n"], meta["d"]
    x0 = make_patch_initial_point(INIT_SEED, cfg["ah16_faithful"])
    L_arr = np.asarray(L, dtype=float)
    epoch_len = max(1, int(np.ceil(n / float(cfg["msvrg_batch"]))))
    print(f"[{policy_name}] instance in {time.time() - t_build:.1f}s "
          f"(K={K} d={d} n={n} epoch_len={epoch_len} "
          f"L in [{L_arr.min():.3f},{L_arr.max():.3f}])", flush=True)

    f0, J0 = validate_oracle_output(*joint_oracle(x0), K, d)
    grams = [J0 @ J0.T]
    fvals = [np.asarray(f0, dtype=float)]
    chain_x, chain_J, chain_f = x0.copy(), J0, f0

    budget = _Budget(K, n, stoch, args.budget)
    L_scale = 1.0
    safeguard_retries = 0
    lam_history = []
    seg_grads, seg_lams = [0.0], [[np.nan] * K]
    ck_grads, ck_cpu, ck_m = [0.0], [0.0], [1]
    grad_at_ck = 0.0
    t0 = time.time()
    decision_seconds = 0.0

    prev_lam = None
    while budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
        t_dec = time.time()
        lam = np.asarray(next_lam(grams, fvals, prev_lam), dtype=float)
        decision_seconds += time.time() - t_dec
        prev_lam = lam
        lam_history.append(lam.copy())

        retries_here = 0
        for _k in range(args.s):
            if not budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
                break
            g_a_full = chain_J.T @ lam
            F_a = float(chain_f @ lam)
            eta = cfg["msvrg_step_const"] / (float(lam @ L_arr) * L_scale)
            stoch.set_anchor(chain_x)
            y = chain_x.copy()
            u_vec = np.zeros(d)
            for _t in range(epoch_len):
                batch = _support_batch(stoch.sample_batch(), lam)
                g_y_S, g_a_S = stoch.grad_pair(y, lam, batch)
                u_vec = cfg["msvrg_momentum"] * u_vec + (g_y_S - g_a_S
                                                         + g_a_full)
                y = y - eta * u_vec
            f_y, J_y = validate_oracle_output(*joint_oracle(y), K, d)
            budget.joint_calls += 1
            grams.append(J_y @ J_y.T)
            fvals.append(np.asarray(f_y, dtype=float))
            seg_grads.append(float(budget.spent()))
            seg_lams.append([float(t) for t in lam])
            if float(f_y @ lam) > F_a + 1e-10 * (1.0 + abs(F_a)):
                L_scale *= 2.0
                safeguard_retries += 1
                retries_here += 1
                if retries_here > MAX_SAFEGUARD_RETRIES:
                    chain_x, chain_J, chain_f = y, J_y, f_y
                    retries_here = 0
            else:
                chain_x, chain_J, chain_f = y, J_y, f_y

            if budget.spent() - grad_at_ck >= args.eval_every:
                grad_at_ck = budget.spent()
                ck_grads.append(budget.spent())
                ck_cpu.append(time.time() - t0)
                ck_m.append(len(grams))

    wall = time.time() - t0
    ck_grads.append(budget.spent())
    ck_cpu.append(wall)
    ck_m.append(len(grams))
    Ms = np.asarray(grams, dtype=float)
    print(f"[{policy_name}] budget spent: {budget.spent():.1f} of "
          f"{args.budget} | segments={len(grams) - 1} | wall={wall:.1f}s "
          f"| decision={decision_seconds:.1f}s | L_scale={L_scale}",
          flush=True)

    # quick log-only final value (CCP instrument); real audits: audit_v2
    t_a = time.time()
    probe = CCPLambdaSolver(K, CCPConfig(N0=8192, r=20, seed=1))
    final_probe, _lam = probe.solve(Ms)
    audit_seconds = time.time() - t_a

    lam_arr = np.asarray(lam_history, dtype=float)
    distinct = (np.unique(lam_arr.round(12), axis=0).shape[0]
                if lam_arr.size else 0)
    summary = {
        "protocol": ("pure fixed budget at K=10 (MNIST patch-softplus): "
                     "shared segment unit, shared s, chain warm start; "
                     "ONLY the next-lambda policy differs; no tolerance "
                     "parameter anywhere; stop = budget; quality measured "
                     "post-hoc by audit_v2 (two-instrument max)"),
        "policy": policy_name,
        "config_instance": _json_ready(cfg),
        "data": DATA_SEED_NOTE,
        "budget": args.budget, "s": args.s, "eval_every": args.eval_every,
        "meter": "audit_v2 post-hoc; summary final = CCP-instrument probe",
        "targeting_starts": (args.ts if policy_name == "adaptive" else None),
        "structured_set_full_size_K10": 67,
        "extra": _json_ready(extra_cfg),
        "K": K, "d": d, "n": n,
        "grad_equiv_total": float(budget.spent()),
        "joint_calls": int(budget.joint_calls),
        "wall_seconds": wall,
        "decision_seconds": decision_seconds,
        "segments_total": int(len(grams) - 1),
        "m_final": int(Ms.shape[0]),
        "n_decisions": int(len(lam_history)),
        "n_distinct_lambdas": int(distinct),
        "L_scale_final": L_scale,
        "L_calibrated": [float(v) for v in L_arr],
        "safeguard_retries": int(safeguard_retries),
        "ck_grads": ck_grads, "ck_cpu": ck_cpu, "ck_m": ck_m,
        "final_audit": float(final_probe),
        "audit_seconds": audit_seconds,
        "init_seed": INIT_SEED, "sampler_seed": SAMPLER_SEED,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "grams.npz", gram_stack=Ms,
                        fvals=np.asarray(fvals), lam_history=lam_arr,
                        seg_grads=np.asarray(seg_grads, dtype=float),
                        seg_lams=np.asarray(seg_lams, dtype=float))
    print(f"[{policy_name}] final probe GN = {final_probe:.6e} "
          f"-> {out_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=55_000.0)
    parser.add_argument("--eval-every", type=float, default=1500.0)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--ts", type=int, default=24)
    parser.add_argument("--ah16-faithful", action="store_true")
    parser.add_argument("--ccp-seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not ipopt_available():
        raise SystemExit("IPOPT required for the ts leg.")

    if args.smoke:
        args.budget, args.eval_every = 800.0, 100.0
        args.per_class, args.batch = 100, 256
    cfg = dict(per_class=args.per_class, msvrg_batch=args.batch,
               msvrg_step_const=0.1, msvrg_momentum=0.5,
               n_probes=(10 if args.smoke else 40),
               ah16_faithful=bool(args.ah16_faithful))
    home = COMPARE_ROOT / ("K10_mnist_SMOKE" if args.smoke
                           else f"K10_mnist{args.per_class * 10 // 1000}k"
                                f"_B{args.budget:.0f}")
    home.mkdir(parents=True, exist_ok=True)
    ccp_cfg = CCPConfig(N0=2000, r=10, seed=args.ccp_seed,
                        seed_sampler="exp", adaptive_seed_schedule=False)

    manifest = {"campaign": "ccp_compare K10 MNIST patch-softplus trial",
                "machine": platform.platform(), "smoke": args.smoke,
                "legs": []}
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
             "final_probe": sm.get("final_audit")})
        (home / "campaign_manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")

    _do(f"adaptive_s{args.s}_ts{args.ts}",
        lambda od: _run_leg_k10(
            "adaptive", _adaptive_policy(10, args.ts), cfg, args, od, {}))

    def _ccp_leg(od):
        stats: list = []
        sm = _run_leg_k10("adaptive_ccp",
                          _ccp_policy(10, ccp_cfg, stats), cfg, args, od,
                          {"ccp_config": vars(ccp_cfg)})
        sm["ccp"] = _stats_block(stats)
        (od / "summary.json").write_text(
            json.dumps(_json_ready(sm), indent=2), encoding="utf-8")
        return sm

    _do(f"adaptive_s{args.s}_ccp", _ccp_leg)

    manifest["total_wall_seconds"] = time.time() - t_all
    (home / "campaign_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    print(f"[campaign] DONE in {manifest['total_wall_seconds']:.0f}s "
          f"-> {home}", flush=True)

    if args.smoke:
        for p in home.glob("*/summary.json"):
            sm = json.loads(p.read_text())
            assert sm["grad_equiv_total"] <= args.budget + 1e-6
            assert np.isfinite(sm["final_audit"])
        print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
