"""Diagnose the v3 slow-progress plateau (session 12).

Design record: ``Note/Jul_26_note.md`` part 2.  Replays the v3 fast
trial bit-for-bit (same config, same seeds), keeps the PRE-PRUNE bundle
(points, fvals, Gram stack — the session-11 ``return_pre_prune`` hook),
then, post-hoc and off every cost axis:

1. verifies the replay against the stored v3 summary (grad-equivalent
   total and the full per-round pc_history must match bitwise);
2. strict-audits (64-start, in-family) PREFIXES of the Gram stack —
   the TRUE max-min trajectory by round, to compare against the
   cheap-tier readings the run steered by;
3. localises the final strict witness lambda*: its value and its l1
   distance to the nearest lambda the run ever targeted — separates
   "the cheap search never aims at the true peaks" from "the peaks are
   aimed at but the serving radius per round is too small";
4. reports how deep each round actually solved at its own lambda
   (achieved lambda_t' M_t lambda_t vs its inner target).

Output: ``output/baseline_svrg_multi_r_vs_fast_v2_without_256_
checkpoints/diag_v3_plateau/diag.json`` + console log.  Costs one v3
replay (~5 min) plus ~1 min of strict searches.  No committed file is
modified; the stored v3 folder is read-only here.
"""
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from experiments import make_mlp_initial_point  # noqa: E402  (torch first)
import numpy as np  # noqa: E402
from bundle import prefer_fused_joint_oracle  # noqa: E402
from objectives_torch_fast import make_mlp_nonconvex_fast  # noqa: E402
from algorithm_fast_without_256_checkpoints import (  # noqa: E402
    algorithm_adaptive_fast,
    _maximise_GN_fast,
    ipopt_available,
)
from baseline_svrg_certified_without_256_checkpoints import _GramSet  # noqa: E402
from run_experiments import DATA_SEED, INIT_SEED, _json_ready  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = (HERE.parent.parent / "output"
           / "pure_budget_without_256_checkpoints_SVRG_IPOPT_Baseline/baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints"
           / "diag_v3_plateau")
V3_DIR = HERE.parent.parent / "output" / "fast_method_trials_v1v2v3_IPOPT" / "v3_rel_target_two_tier"

# v3 TRIAL_CONFIG verbatim (run_trial_K6_fast_without_256_checkpoints.py).
CFG = dict(
    K=6, p=20, n=50_000, hidden_sizes=[96, 96], activation="tanh",
    epsilon=1e-3, max_grad_evals=180_180.0,
    adaptive_eval_every_n_grads=600.0, lambda_max_starts=64,
    max_outer=500, lambda_tier_mode="two_tier", msvrg_rel_target=0.25,
    msvrg_batch=4096, msvrg_epoch_len=None, msvrg_step_const=0.1,
    msvrg_momentum=0.5, msvrg_trigger_rho=0.7, msvrg_trigger_patience=2,
    msvrg_max_segments=10, prune_grid_r=10, sampler_seed=41,
)


def main() -> None:
    if not ipopt_available():
        raise RuntimeError("IPOPT is required but unavailable.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    objectives, gradients, L, joint, stoch = make_mlp_nonconvex_fast(
        K=CFG["K"], p=CFG["p"], n=CFG["n"],
        hidden_sizes=CFG["hidden_sizes"], seed=DATA_SEED,
        activation=CFG["activation"], w_true_scale=1.0,
        batch_size=CFG["msvrg_batch"], sampler_seed=CFG["sampler_seed"],
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(K=CFG["K"], p=CFG["p"],
                                hidden_sizes=CFG["hidden_sizes"],
                                seed=INIT_SEED)
    print(f"[build] instance ready in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    fast = algorithm_adaptive_fast(
        K=CFG["K"], d=int(x0.size), objectives=objectives,
        grad_objectives=gradients, L=L, x0=x0, stoch_oracle=stoch,
        epsilon=CFG["epsilon"], max_outer=CFG["max_outer"],
        eval_every_n_grads=CFG["adaptive_eval_every_n_grads"],
        max_grad_evals=CFG["max_grad_evals"],
        lambda_max_starts=CFG["lambda_max_starts"],
        lambda_tier_mode=CFG["lambda_tier_mode"],
        msvrg_step_const=CFG["msvrg_step_const"],
        msvrg_momentum=CFG["msvrg_momentum"],
        msvrg_epoch_len=CFG["msvrg_epoch_len"],
        msvrg_max_segments=CFG["msvrg_max_segments"],
        msvrg_trigger_rho=CFG["msvrg_trigger_rho"],
        msvrg_trigger_patience=CFG["msvrg_trigger_patience"],
        msvrg_rel_target=CFG["msvrg_rel_target"],
        prune_grid_r=CFG["prune_grid_r"],
        joint_oracle=joint_oracle, verbose=False,
        return_pre_prune=True,
    )
    print(f"[replay] run finished in {time.time() - t1:.1f}s "
          f"(stop={fast['stop_reason']})", flush=True)

    # ---- 1. replay verification --------------------------------------
    with open(V3_DIR / "summary.json", "r", encoding="utf-8") as fh:
        s3 = json.load(fh)
    pc_old = np.asarray(s3["fast_extras"]["pc_history"], dtype=float)
    pc_new = np.asarray(fast["pc_history"], dtype=float)
    same_len = pc_old.size == pc_new.size
    replay_check = {
        "grad_equiv_total_stored": s3["fast_extras"]["grad_equiv_total"],
        "grad_equiv_total_replay": float(fast["grad_equiv_total"]),
        "grad_equiv_equal": bool(
            float(fast["grad_equiv_total"])
            == float(s3["fast_extras"]["grad_equiv_total"])),
        "pc_history_bitwise_equal": bool(
            same_len and np.array_equal(pc_old, pc_new)),
        "pc_history_max_absdiff": (
            float(np.max(np.abs(pc_old - pc_new))) if same_len else None),
    }
    print("[verify]", json.dumps(replay_check), flush=True)

    pp = fast["pre_prune"]
    print("[pre_prune] keys:", sorted(pp.keys()), flush=True)
    Ms = np.asarray(pp["gram_stack"], dtype=float)
    m_total = int(Ms.shape[0])
    lam_hist = np.asarray(fast["lambda_history"], dtype=float)
    print(f"[bundle] pre-prune m={m_total}, rounds={lam_hist.shape[0]}",
          flush=True)

    # ---- 2. strict prefix audits (off-axis measurement) --------------
    t2 = time.time()
    prefix_ms = sorted({25, 50, 75, 100, 150, 200, 250, 300, 350, 400,
                        450, m_total})
    prefix_ms = [m for m in prefix_ms if m <= m_total]
    strict_prefix = []
    prev = None
    for m in prefix_ms:
        gs = _GramSet(list(Ms[:m]), CFG["K"])
        v, lam = _maximise_GN_fast(gs, prev_lam=prev, tier="strict",
                                   max_starts=CFG["lambda_max_starts"])
        prev = np.asarray(lam, dtype=float)
        # cheap reading near the same round for contrast (prefix m holds
        # x0 + the first m-1 round points -> rounds up to m-1).
        r_idx = min(max(m - 2, 0), pc_new.size - 1)
        strict_prefix.append({
            "m": m, "after_round": m - 1,
            "strict_gn": float(v),
            "cheap_pc_at_that_round": float(pc_new[r_idx]),
            "lambda": [float(t) for t in prev],
        })
        print(f"[strict prefix] m={m:4d} (round {m - 1:4d})  "
              f"GN*={v:.6f}   cheap_pc~{pc_new[r_idx]:.6f}", flush=True)
    print(f"[strict prefix] total search time {time.time() - t2:.1f}s",
          flush=True)

    # ---- 3. witness localisation -------------------------------------
    lam_star = np.asarray(strict_prefix[-1]["lambda"], dtype=float)
    d1 = np.abs(lam_hist - lam_star[None, :]).sum(axis=1)
    witness = {
        "strict_final_gn": strict_prefix[-1]["strict_gn"],
        "cheap_final_pc": float(pc_new[-1]),
        "cheap_min_last50_rounds": float(pc_new[-50:].min()),
        "lambda_star": [float(t) for t in lam_star],
        "l1_dist_to_nearest_visited_lambda": float(d1.min()),
        "nearest_visited_round": int(np.argmin(d1)) + 1,
        "median_l1_dist_to_visited": float(np.median(d1)),
    }
    print("[witness]", json.dumps(witness), flush=True)

    # ---- 4. per-round achieved depth vs inner target -----------------
    tg = np.asarray(s3["fast_extras"]["inner_target_history"], dtype=float)
    n_r = min(lam_hist.shape[0], m_total - 1, tg.size)
    achieved = np.array([
        float(lam_hist[t] @ Ms[t + 1] @ lam_hist[t]) for t in range(n_r)
    ])
    ratio = achieved[:n_r] / np.maximum(tg[:n_r], 1e-300)
    depth = {
        "rounds_measured": int(n_r),
        "achieved_median": float(np.median(achieved)),
        "achieved_over_target_median": float(np.median(ratio)),
        "achieved_over_target_q90": float(np.quantile(ratio, 0.9)),
        "rounds_meeting_target": int(np.sum(ratio <= 1.0 + 1e-12)),
    }
    print("[depth]", json.dumps(depth), flush=True)

    diag = {
        "note": ("v3 replay diagnostic (session 12, Jul 26). Strict "
                 "prefix audits and the witness localisation are "
                 "measurements off every cost axis; the replay itself "
                 "is bit-checked against the stored v3 summary."),
        "config": _json_ready(CFG),
        "replay_check": replay_check,
        "strict_prefix_audit": strict_prefix,
        "witness": witness,
        "depth": depth,
        "data_seed": DATA_SEED, "init_seed": INIT_SEED,
    }
    (OUT_DIR / "diag.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8")
    print(f"DONE -> {OUT_DIR / 'diag.json'}  "
          f"(total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
