"""run_bandit_toy_K5_without_256_checkpoints.py — K=5 offline-bandit:
Momentum-SVRG certified baseline (r=10, 1001 nodes, equal-level stop)
vs Momentum-SVRG adaptive bundle.

NEW FILE (July 26, 2026).  User-approved design (see Note/Jul_26_note.md):
centered-quartic rewards R_k(a) = 1 - |x_a - x_k|^4 on A = 5 arms
(K = 5, d = 4); everything else inherited from the K=2 pipeline.  The
closed-form softmax oracle holds for every K; the SURF arc-length layer
is bi-objective and drops out here.

Evaluation lambda set (common to all metrics, never timed): the full
r=10 simplex grid (1001 points) plus 20000 Dirichlet(1) draws with a
fixed seed.  Figures: fig1/fig2/fig5 identical to the K=2 driver
(imported); fig3 becomes the delta_value CDF over the lambda set; fig4
becomes delta_value profiles along the 10 simplex edges.

Usage:
    python run_bandit_toy_K5_without_256_checkpoints.py --epsilon 1e-2
    python run_bandit_toy_K5_without_256_checkpoints.py --epsilon 1e-3
    python run_bandit_toy_K5_without_256_checkpoints.py --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithm_fast_without_256_checkpoints import algorithm_adaptive_fast
from baseline_svrg_certified_without_256_checkpoints import (
    baseline_svrg_certified)
from bundle_fast import simplex_grid
from objectives_bandit_toy import (BanditStochOracle, calibrate_L,
                                   make_bandit_toy_K)
from run_bandit_toy_without_256_checkpoints import (
    ADAPTIVE_COLOR, BASELINE_COLOR, best_at_budget, eps_tag,
    fig_gn_curves, fig_value_convergence, first_crossing,
    strict_prefix_history)


# =====================================================================
#  Config
# =====================================================================
def build_config(args: argparse.Namespace) -> Dict:
    eps = 1e-2 if args.smoke else float(args.epsilon)
    cfg = {
        # ---- problem (user-approved K=5 design) ----
        "K": 5, "A": 5, "tau": 0.05, "alpha": 4.0, "T": 1000,
        "noise_std": 0.5, "data_seed": 7, "sampler_seed": 41,
        # ---- shared MSVRG inner solver ----
        "msvrg_batch": 256, "msvrg_step_const": 0.1, "msvrg_momentum": 0.5,
        "msvrg_epoch_len": None, "msvrg_trigger_rho": 0.7,
        "msvrg_trigger_patience": 2, "max_segment_retries": 4,
        "msvrg_max_segments": (16 if eps > 3e-3 else
                               64 if eps > 3e-4 else 256),
        # ---- accuracy ladder + equal-level stop ----
        "epsilon": eps,
        "node_tol": 2.0 * eps / 3.0,
        "solve_target": eps / 6.0,
        "global_stop_gn": 2.0 * eps / 3.0,
        # ---- baseline / adaptive ----
        "resolution": 10,                  # C(14,4) = 1001 nodes
        "share_mode": "gram",
        # max_outer widened at the 1e-4 rung (pure fuse; a round_fuse
        # stop would be a truncation artefact, the session-3 lesson).
        "max_outer": 500 if eps > 3e-4 else 2000,
        "lambda_tier_mode": "strict",
        "lambda_max_starts": 64, "msvrg_rel_target": 0.25,
        "prune_grid_r": 10,
        # ---- recording / fuses ----
        # 10 at the loose rung (its totals are only ~10^2 grads thanks to
        # the share mechanism), 100 at tighter rungs.
        "eval_every_n_grads": 10.0 if eps > 3e-3 else 100.0,
        # Grad fuse auto-widens at 1e-4: 1001 nodes x deep solves could
        # legitimately exceed 5e5 grad-equivalents.
        "max_grad_evals": (args.max_grad_evals
                           if args.max_grad_evals is not None
                           else (500_000.0 if eps > 3e-4 else 2_000_000.0)),
        "max_wall_seconds": args.max_wall,
        # ---- evaluation (off both cost axes) ----
        "n_dirichlet_lams": 20000, "lam_sample_seed": 0,
        "n_edge_points": 51,
        "L_safety": 1.5,
        "smoke": bool(args.smoke),
    }
    if args.smoke:
        cfg.update({
            "msvrg_max_segments": 8, "max_outer": 15,
            "max_grad_evals": 8000.0, "max_wall_seconds": 900.0,
            "n_dirichlet_lams": 2000, "n_edge_points": 21,
        })
    return cfg


# =====================================================================
#  K-general evaluation machinery (all off both cost axes)
# =====================================================================
def eval_lambda_set(cfg: Dict) -> np.ndarray:
    grid = simplex_grid(cfg["K"], cfg["resolution"])           # (1001, K)
    rng = np.random.RandomState(cfg["lam_sample_seed"])
    dirich = rng.dirichlet(np.ones(cfg["K"]), size=cfg["n_dirichlet_lams"])
    return np.vstack([grid, dirich])


def value_gap_machinery_K(fvals: np.ndarray, prefix_counts: List[int],
                          lams: np.ndarray, f_star_scal: np.ndarray,
                          chunk: int = 2000) -> Dict:
    """eps_value trajectory + per-checkpoint profiles, chunked over the
    lambda columns so the (m x n_lam) scalarisation never materialises.
    Returns profiles (n_ckpt, n_lam): row j = delta_value(lam) of the
    delivered-set prefix at checkpoint j."""
    idx = np.asarray(prefix_counts, dtype=int) - 1
    n_lam = lams.shape[0]
    profiles = np.empty((len(idx), n_lam))
    for s in range(0, n_lam, chunk):
        Lb = lams[s:s + chunk]
        scal = fvals @ Lb.T                                     # (m, nb)
        cm = np.minimum.accumulate(scal, axis=0)
        profiles[:, s:s + chunk] = cm[idx] - f_star_scal[s:s + chunk][None, :]
    return {"profiles": profiles,
            "eps_value_history": profiles.max(axis=1).tolist()}


def pf_metrics_K(fvals_delivered: np.ndarray, lams: np.ndarray,
                 fvecs_oracle: np.ndarray, chunk: int = 2000) -> Dict:
    """max point-to-oracle distance (over query lambdas) and IGD in R^K."""
    maxd = 0.0
    sq_del = (fvals_delivered ** 2).sum(axis=1)                 # (m,)
    igd_sum = 0.0
    n_lam = lams.shape[0]
    for s in range(0, n_lam, chunk):
        Lb = lams[s:s + chunk]
        scal = fvals_delivered @ Lb.T
        pick = np.argmin(scal, axis=0)
        f_hat = fvals_delivered[pick]
        fo = fvecs_oracle[s:s + chunk]
        maxd = max(maxd, float(np.linalg.norm(f_hat - fo, axis=1).max()))
        # IGD chunk via the squared-norm expansion (no (nb, m, K) tensor).
        G = fo @ fvals_delivered.T                              # (nb, m)
        d2 = (fo ** 2).sum(axis=1)[:, None] + sq_del[None, :] - 2.0 * G
        igd_sum += float(np.sqrt(np.maximum(d2, 0.0)).min(axis=1).sum())
    return {"max_point_to_oracle": maxd, "igd": igd_sum / n_lam}


# =====================================================================
#  K=5 figures (fig3 CDF, fig4 edge profiles)
# =====================================================================
def fig_value_cdf(prof_bl_m: np.ndarray, prof_ad_m: np.ndarray,
                  prof_bl_f: np.ndarray, prof_ad_f: np.ndarray,
                  budget_label: str, eps: float, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    floor = 1e-17
    q = np.linspace(0.0, 1.0, len(prof_bl_m), endpoint=True)
    for prof, color, ls, lab in (
        (prof_bl_m, BASELINE_COLOR, "-", f"baseline @ {budget_label}"),
        (prof_ad_m, ADAPTIVE_COLOR, "-", f"adaptive @ {budget_label}"),
        (prof_bl_f, BASELINE_COLOR, "--", "baseline @ own delivery"),
        (prof_ad_f, ADAPTIVE_COLOR, "--", "adaptive @ own delivery"),
    ):
        ax.plot(np.sort(np.maximum(prof, floor)), q, color=color, ls=ls,
                lw=1.4, alpha=0.65 if ls == "--" else 1.0, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("scalarized objective gap  delta_value(lambda)")
    ax.set_ylabel("fraction of evaluation lambdas with gap <= x")
    ax.set_title("Value-gap distribution over the common lambda set "
                 f"(K=5, eps = {eps:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_value_gap_cdf.png"), dpi=160)
    plt.close(fig)


def fig_edge_profiles(problem, cfg: Dict, bl_fvals_del: np.ndarray,
                      ad_fvals_del: np.ndarray, eps: float,
                      out_dir: str) -> None:
    K = cfg["K"]
    t = np.linspace(0.0, 1.0, cfg["n_edge_points"])
    edges = list(itertools.combinations(range(K), 2))          # 10 edges
    fig, axes = plt.subplots(2, 5, figsize=(15.0, 5.6), sharey=True)
    floor = 1e-17
    for ax, (i, j) in zip(axes.ravel(), edges):
        lams = np.zeros((len(t), K))
        lams[:, i] = t
        lams[:, j] = 1.0 - t
        _, f_star = problem.oracle_batch(lams)
        for fvals, color, lab in ((bl_fvals_del, BASELINE_COLOR, "baseline"),
                                  (ad_fvals_del, ADAPTIVE_COLOR, "adaptive")):
            delta = (fvals @ lams.T).min(axis=0) - f_star
            ax.plot(t, np.maximum(delta, floor), color=color, lw=1.3,
                    label=lab)
        ax.set_yscale("log")
        ax.set_title(f"edge e{i + 1} <-> e{j + 1}", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("t  (lambda = t*e_i + (1-t)*e_j)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("delta_value", fontsize=9)
    fig.suptitle("Value gap along the 10 simplex edges, delivered sets "
                 f"(K=5, eps = {eps:g})", y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_edge_value_profiles.png"),
                dpi=160)
    plt.close(fig)


# =====================================================================
#  Main
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon", type=float, default=1e-2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-grad-evals", type=float, default=None)
    ap.add_argument("--max-wall", type=float, default=7200.0)
    ap.add_argument("--out-root", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "output",
        "bandit_toy_K5_without_256_checkpoints"))
    args = ap.parse_args()

    cfg = build_config(args)
    sub = "smoke" if cfg["smoke"] else f"eps{eps_tag(cfg['epsilon'])}"
    out_dir = os.path.abspath(os.path.join(args.out_root, sub))
    os.makedirs(out_dir, exist_ok=True)
    eps = cfg["epsilon"]

    print(f"== K=5 offline-bandit | eps={eps:g} | out: {out_dir}", flush=True)

    problem = make_bandit_toy_K(K=cfg["K"], T=cfg["T"],
                                noise_std=cfg["noise_std"],
                                data_seed=cfg["data_seed"], A=cfg["A"],
                                tau=cfg["tau"], alpha=cfg["alpha"])
    cal = calibrate_L(problem, safety=cfg["L_safety"])
    L = np.asarray(cal["L"], dtype=float)
    x0 = np.zeros(problem.d)
    print("   L calibrated: "
          + ", ".join(f"{v:.4f}" for v in L)
          + f" ({cal['n_hessians']} Hessians)", flush=True)

    # ---- baseline -----------------------------------------------------
    oracle_bl = BanditStochOracle(problem, batch_size=cfg["msvrg_batch"],
                                  seed=cfg["sampler_seed"])
    t_cpu0, t_wall0 = time.process_time(), time.perf_counter()
    bl = baseline_svrg_certified(
        problem.K, problem.d, problem.objectives, problem.grad_objectives,
        L, x0, cfg["resolution"], oracle_bl,
        node_tol=cfg["node_tol"], solve_target=cfg["solve_target"],
        share_mode=cfg["share_mode"],
        msvrg_step_const=cfg["msvrg_step_const"],
        msvrg_momentum=cfg["msvrg_momentum"],
        msvrg_epoch_len=cfg["msvrg_epoch_len"],
        msvrg_max_segments=cfg["msvrg_max_segments"],
        msvrg_trigger_rho=cfg["msvrg_trigger_rho"],
        msvrg_trigger_patience=cfg["msvrg_trigger_patience"],
        max_segment_retries=cfg["max_segment_retries"],
        max_grad_evals=cfg["max_grad_evals"],
        max_wall_seconds=cfg["max_wall_seconds"],
        eval_every_n_grads=cfg["eval_every_n_grads"],
        lambda_max_starts=cfg["lambda_max_starts"],
        global_stop_gn=cfg["global_stop_gn"],
        joint_oracle=problem.joint_oracle,
        return_points=True, return_grams=True, verbose=True,
    )
    bl_process_s = time.process_time() - t_cpu0
    bl_wall_s = time.perf_counter() - t_wall0
    print(f"   baseline done: {bl['stop_reason']}, served "
          f"{bl['n_served']}/{bl['n_nodes']}, censored "
          f"{bl['censored_nodes']}, grads {bl['grad_equiv_total']:.1f}, "
          f"wall {bl['wall_seconds']:.2f}s", flush=True)

    # ---- adaptive -----------------------------------------------------
    oracle_ad = BanditStochOracle(problem, batch_size=cfg["msvrg_batch"],
                                  seed=cfg["sampler_seed"])
    t_cpu0, t_wall0 = time.process_time(), time.perf_counter()
    ad = algorithm_adaptive_fast(
        problem.K, problem.d, problem.objectives, problem.grad_objectives,
        L, x0,
        stoch_oracle=oracle_ad, epsilon=eps, max_outer=cfg["max_outer"],
        eval_every_n_grads=cfg["eval_every_n_grads"],
        max_grad_evals=cfg["max_grad_evals"],
        lambda_max_starts=cfg["lambda_max_starts"],
        lambda_tier_mode=cfg["lambda_tier_mode"],
        msvrg_step_const=cfg["msvrg_step_const"],
        msvrg_momentum=cfg["msvrg_momentum"],
        msvrg_epoch_len=cfg["msvrg_epoch_len"],
        msvrg_max_segments=cfg["msvrg_max_segments"],
        msvrg_trigger_rho=cfg["msvrg_trigger_rho"],
        msvrg_trigger_patience=cfg["msvrg_trigger_patience"],
        msvrg_rel_target=cfg["msvrg_rel_target"],
        prune_grid_r=cfg["prune_grid_r"], return_pre_prune=True,
        joint_oracle=problem.joint_oracle, verbose=True,
    )
    ad_process_s = time.process_time() - t_cpu0
    ad_wall_s = time.perf_counter() - t_wall0
    print(f"   adaptive done: {ad['stop_reason']}, bundle "
          f"{ad['m_history'][-1]} -> {ad['bundle'].m} after prune, grads "
          f"{ad['grad_equiv_total']:.1f}, wall {ad['cpu_times'][-1]:.2f}s",
          flush=True)

    # ---- common post-hoc scoring (off both cost axes) -----------------
    t_score = time.perf_counter()
    ad_strict = strict_prefix_history(ad["pre_prune"]["gram_stack"],
                                      ad["m_history"], problem.K,
                                      cfg["lambda_max_starts"])
    scorer_seconds = time.perf_counter() - t_score

    lams = eval_lambda_set(cfg)
    fvecs_oracle, f_star_scal = problem.oracle_batch(lams)

    bl_points = np.asarray(bl["delivered_points"], dtype=float)
    bl_fvals = np.array([problem.joint_oracle(x)[0] for x in bl_points])
    ad_fvals_full = np.asarray(ad["pre_prune"]["fvals"], dtype=float)
    ad_points_delivered = np.asarray(ad["bundle"].points, dtype=float)
    ad_fvals_delivered = np.array([problem.joint_oracle(x)[0]
                                   for x in ad_points_delivered])

    bl_val = value_gap_machinery_K(bl_fvals, bl["delivered_history"],
                                   lams, f_star_scal)
    ad_val = value_gap_machinery_K(ad_fvals_full, ad["m_history"],
                                   lams, f_star_scal)

    budget_grads = min(bl["grad_evals_history"][-1],
                       ad["grad_evals_history"][-1])
    j_bl = int(np.flatnonzero(
        np.asarray(bl["grad_evals_history"]) <= budget_grads + 1e-9)[-1])
    j_ad = int(np.flatnonzero(
        np.asarray(ad["grad_evals_history"]) <= budget_grads + 1e-9)[-1])

    bl_pf = pf_metrics_K(bl_fvals, lams, fvecs_oracle)
    ad_pf = pf_metrics_K(ad_fvals_delivered, lams, fvecs_oracle)

    # ---- protocol readouts -------------------------------------------
    readouts = {}
    for name, res, strict in (("baseline", bl,
                               bl["delivered_gn_strict_history"]),
                              ("adaptive", ad, ad_strict)):
        readouts[name] = {
            "first_cpu_to_eps": first_crossing(strict, res["cpu_times"], eps),
            "first_grads_to_eps": first_crossing(
                strict, res["grad_evals_history"], eps),
            "final_gn_strict": float(np.min(strict)),
            "total_cpu": float(res["cpu_times"][-1]),
            "total_grads": float(res["grad_evals_history"][-1]),
        }
    budget_cpu = min(bl["cpu_times"][-1], ad["cpu_times"][-1])
    readouts["fixed_budget"] = {
        "budget_grads": float(budget_grads),
        "budget_cpu": float(budget_cpu),
        "gn_at_budget_grads": {
            "baseline": best_at_budget(bl["delivered_gn_strict_history"],
                                       bl["grad_evals_history"], budget_grads),
            "adaptive": best_at_budget(ad_strict, ad["grad_evals_history"],
                                       budget_grads)},
        "gn_at_budget_cpu": {
            "baseline": best_at_budget(bl["delivered_gn_strict_history"],
                                       bl["cpu_times"], budget_cpu),
            "adaptive": best_at_budget(ad_strict, ad["cpu_times"],
                                       budget_cpu)},
        "eps_value_at_budget_grads": {
            "baseline": best_at_budget(bl_val["eps_value_history"],
                                       bl["grad_evals_history"], budget_grads),
            "adaptive": best_at_budget(ad_val["eps_value_history"],
                                       ad["grad_evals_history"],
                                       budget_grads)},
    }

    # ---- figures ------------------------------------------------------
    fig_gn_curves(bl, ad_strict, ad, eps, out_dir,
                  baseline_label="uniform discretisation r=10 "
                                 "(MSVRG, certified)")
    fig_value_cdf(bl_val["profiles"][j_bl], ad_val["profiles"][j_ad],
                  bl_val["profiles"][-1], ad_val["profiles"][-1],
                  f"matched budget {budget_grads:.0f} grads", eps, out_dir)
    fig_edge_profiles(problem, cfg, bl_fvals, ad_fvals_delivered, eps,
                      out_dir)
    fig_value_convergence(bl, bl_val["eps_value_history"], ad,
                          ad_val["eps_value_history"], eps, out_dir,
                          baseline_label="baseline (r=10, MSVRG)")

    # ---- persist ------------------------------------------------------
    try:
        commit = subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unavailable"

    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    summary = {
        "config": cfg,
        "commit": commit,
        "L_calibration": {"L": cal["L"], "L_raw_max": cal["L_raw_max"],
                          "n_hessians": cal["n_hessians"]},
        "statistical_layer": {"reward_sup_gap": problem.reward_sup_gap()},
        "front_uniformity_secondary":
            "omitted for K=5 (no canonical 1-D ordering of the front)",
        "baseline": {k: bl[k] for k in (
            "stop_reason", "global_stop_gn", "n_nodes", "n_served",
            "censored_nodes", "solved_nodes", "served_by_share",
            "served_by_chain", "n_delivered", "segments_total",
            "minibatch_steps_total", "safeguard_retries", "L_scale_final",
            "joint_calls", "ifo_minibatch_total", "grad_equiv_total",
            "wall_seconds", "metric_seconds", "delivered_gn_strict",
            "grid_worst_best_val", "epoch_len")},
        "adaptive": {k: ad[k] for k in (
            "stop_reason", "joint_calls", "ifo_minibatch_total",
            "grad_equiv_total", "L_scale_final", "inner_cap_hits",
            "lambda_search_seconds", "lambda_tier_mode")},
        "adaptive_bundle_m": {"pre_prune": int(ad["m_history"][-1]),
                              "delivered": int(ad["bundle"].m)},
        "process_time_s": {"baseline": bl_process_s, "adaptive": ad_process_s},
        "wall_time_s": {"baseline": bl_wall_s, "adaptive": ad_wall_s},
        "scorer_seconds_adaptive_prefix": scorer_seconds,
        "pf_metrics": {"baseline": bl_pf, "adaptive": ad_pf},
        "eps_value_final": {"baseline": bl_val["eps_value_history"][-1],
                            "adaptive": ad_val["eps_value_history"][-1]},
        "readouts": readouts,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)

    np.savez_compressed(
        os.path.join(out_dir, "raw_histories.npz"),
        lams=lams, fvecs_oracle=fvecs_oracle, f_star_scal=f_star_scal,
        bl_cpu=np.asarray(bl["cpu_times"]),
        bl_grads=np.asarray(bl["grad_evals_history"]),
        bl_gn_strict=np.asarray(bl["delivered_gn_strict_history"]),
        bl_eps_value=np.asarray(bl_val["eps_value_history"]),
        bl_points=bl_points, bl_fvals=bl_fvals,
        bl_delivered_history=np.asarray(bl["delivered_history"]),
        ad_cpu=np.asarray(ad["cpu_times"]),
        ad_grads=np.asarray(ad["grad_evals_history"]),
        ad_gn_strict=np.asarray(ad_strict),
        ad_eps_value=np.asarray(ad_val["eps_value_history"]),
        ad_m_history=np.asarray(ad["m_history"]),
        ad_fvals_pre_prune=ad_fvals_full,
        ad_points_delivered=ad_points_delivered,
        ad_fvals_delivered=ad_fvals_delivered,
        prof_bl_matched=bl_val["profiles"][j_bl],
        prof_ad_matched=ad_val["profiles"][j_ad],
        prof_bl_final=bl_val["profiles"][-1],
        prof_ad_final=ad_val["profiles"][-1],
    )

    print("\n== summary ==")
    for name in ("baseline", "adaptive"):
        r = readouts[name]
        print(f"   {name:9s} final GN* {r['final_gn_strict']:.3e} | "
              f"cpu-to-eps {r['first_cpu_to_eps']} | "
              f"grads-to-eps {r['first_grads_to_eps']} | "
              f"total ({r['total_cpu']:.2f}s, {r['total_grads']:.0f} grads)")
    print(f"   eps_value final: baseline "
          f"{bl_val['eps_value_history'][-1]:.3e}, adaptive "
          f"{ad_val['eps_value_history'][-1]:.3e}")
    print(f"   figures + summary.json + raw_histories.npz -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
