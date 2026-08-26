"""run_bandit_toy_without_256_checkpoints.py — SURF offline-bandit toy:
Momentum-SVRG certified baseline vs Momentum-SVRG adaptive bundle.

NEW FILE (July 26, 2026).  Design record: LEDGER + Note/Jul_26_note.md.

Pipeline (user-approved July 26):
    D (T=1000, balanced, noise 0.5) -> (R1_hat, R2_hat) -> (F1, F2)
    on reduced logits theta in R^4, then
      closed-form softmax (untimed)  -> exact plug-in PF oracle
      certified uniform grid + MSVRG -> B_uniform   (r = 11 -> 12 nodes)
      worst-lambda (IPOPT strict) + MSVRG BundleUpdate -> B_adaptive
    common evaluation (same instrument, off both cost axes):
      eps_opt (strict full-simplex GN*), eps_PF (max-dist, IGD),
      eps_value (scalarized objective gap), CPU time, grad-equivalents;
      secondary: CV / Gap Ratio; statistical: ||R_hat - R||_inf etc.

Figures (6): GN* vs CPU, GN* vs grads, Pareto front (nondominated
discovered fronts since Jul 27), value-gap per-w profile, value-gap
per-w profile at matched budget only (fig4b, Jul 27), value-gap
convergence (2 panels).

Usage:
    python run_bandit_toy_without_256_checkpoints.py --epsilon 1e-2
    python run_bandit_toy_without_256_checkpoints.py --epsilon 1e-3
    python run_bandit_toy_without_256_checkpoints.py --smoke
    # finest checkpoint cadence (every node / every round):
    python run_bandit_toy_without_256_checkpoints.py --epsilon 1e-2 --eval-every 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from algorithm_fast_without_256_checkpoints import (_maximise_GN_fast,
                                                    algorithm_adaptive_fast)
from baseline_svrg_certified_without_256_checkpoints import (
    _GramSet, baseline_svrg_certified)
from objectives_bandit_toy import (BanditStochOracle, calibrate_L,
                                   make_bandit_toy)

BASELINE_COLOR = "tab:red"
ADAPTIVE_COLOR = "tab:blue"


# =====================================================================
#  Config
# =====================================================================
def build_config(args: argparse.Namespace) -> Dict:
    eps = 1e-2 if args.smoke else float(args.epsilon)
    cfg = {
        # ---- problem (locked by both papers) ----
        "A": 5, "tau": 0.05, "alpha": 4.0, "T": 1000, "noise_std": 0.5,
        "data_seed": 7, "sampler_seed": 41,
        # ---- shared MSVRG inner solver (identical for both methods) ----
        "msvrg_batch": 256, "msvrg_step_const": 0.1, "msvrg_momentum": 0.5,
        "msvrg_epoch_len": None, "msvrg_trigger_rho": 0.7,
        "msvrg_trigger_patience": 2, "max_segment_retries": 4,
        # per-rung fuse widening (reported to the user):
        # eps 1e-2 -> 16, 1e-3 -> 64, 1e-4 -> 256 segments per inner task.
        "msvrg_max_segments": (16 if eps > 3e-3 else
                               64 if eps > 3e-4 else 256),
        # ---- accuracy ladder ----
        "epsilon": eps,
        "node_tol": 2.0 * eps / 3.0,
        "solve_target": eps / 6.0,          # node_tol / 4
        # Jul 26 (user request): both methods stop on the SAME line —
        # strict full-simplex GN* <= 2eps/3 (the adaptive's certificate
        # level), verified with the same instrument; the baseline's
        # verification time lands inside its own CPU axis.
        "global_stop_gn": 2.0 * eps / 3.0,
        # ---- baseline ----
        "resolution": 11, "share_mode": "gram",
        # ---- adaptive ----
        "max_outer": 500, "lambda_tier_mode": "strict",
        "lambda_max_starts": 64, "msvrg_rel_target": 0.25,
        "prune_grid_r": 11,
        # ---- recording / fuses ----
        # 10 (not the MLP track's 4500): toy totals are O(10^2-10^3)
        # grad-equivalents, so a fine cadence is needed for readable
        # curves; checkpoint bookkeeping stays off both cost axes.
        # Jul 27 (user request): --eval-every <= 0 means None = checkpoint
        # at EVERY processed node (baseline) / EVERY round (adaptive) —
        # the finest cadence available.  NOTE the baseline's equal-level
        # stop check rides the checkpoint cadence, so a finer cadence can
        # legitimately stop the baseline earlier (same stop line, checked
        # more often).
        "eval_every_n_grads": (None if args.eval_every <= 0
                               else float(args.eval_every)),
        "max_grad_evals": args.max_grad_evals,
        "max_wall_seconds": args.max_wall,
        # ---- evaluation (off both cost axes) ----
        "n_w_eval": 2001, "n_dense_oracle": 5000, "n_pf_ref_points": 12,
        "L_safety": 1.5,
        "smoke": bool(args.smoke),
    }
    if args.smoke:
        cfg.update({
            "msvrg_max_segments": 8, "max_outer": 20,
            "max_grad_evals": 3000.0, "max_wall_seconds": 300.0,
            "n_w_eval": 501, "n_dense_oracle": 1001,
        })
    return cfg


def eps_tag(eps: float) -> str:
    return f"{eps:.0e}".replace("e-0", "e-")


# =====================================================================
#  Post-hoc scoring helpers (all off both cost axes)
# =====================================================================
def strict_prefix_history(gram_stack: np.ndarray, prefix_counts: List[int],
                          K: int, max_starts: int) -> List[float]:
    """Family strict GN* of the point-set prefix at every checkpoint.

    Same comparable-meter recipe as the baseline's internal
    ``delivered_gn_strict_history`` (Jul 25 fix): warm-started along the
    prefix chain; the final value keeps the max of the chained and fresh
    searches (a search only ever under-reports a maximum).
    """
    out: List[float] = []
    prev = None
    for m_j in prefix_counts:
        sub = _GramSet(list(gram_stack[:m_j]), K)
        v_j, prev = _maximise_GN_fast(sub, prev_lam=prev, tier="strict",
                                      max_starts=max_starts)
        out.append(float(v_j))
    return out


def value_gap_machinery(fvals: np.ndarray, prefix_counts: List[int],
                        w_eval: np.ndarray, f_star_scal: np.ndarray) -> Dict:
    """eps_value trajectory + final/per-prefix profiles from stored fvals.

    fvals: (m, 2) objective values of the method's points in delivery
    order.  For every prefix the solution map answers a query w with the
    point minimising F_w = w*F1 + (1-w)*F2 (min over the delivered set),
    so delta_value(w) = min_prefix F_w - F_w(theta_star(w)) >= 0.
    """
    scal = fvals @ np.stack([w_eval, 1.0 - w_eval])          # (m, n_w)
    cummin = np.minimum.accumulate(scal, axis=0)
    profiles = cummin - f_star_scal[None, :]                  # (m, n_w)
    eps_value_hist = [float(np.max(profiles[m_j - 1])) for m_j in prefix_counts]
    return {"eps_value_history": eps_value_hist,
            "profile_at_prefix": lambda m_j: profiles[m_j - 1],
            "final_profile": profiles[prefix_counts[-1] - 1]}


def first_crossing(history: List[float], axis: List[float],
                   eps: float) -> Optional[float]:
    """First axis value whose BEST-SO-FAR history is <= eps (None if never)."""
    best = np.minimum.accumulate(np.asarray(history, dtype=float))
    idx = np.flatnonzero(best <= eps)
    return float(axis[int(idx[0])]) if idx.size else None


def best_at_budget(history: List[float], axis: List[float],
                   budget: float) -> float:
    a = np.asarray(axis, dtype=float)
    best = np.minimum.accumulate(np.asarray(history, dtype=float))
    idx = np.flatnonzero(a <= budget + 1e-12)
    return float(best[int(idx[-1])]) if idx.size else float("nan")


def pf_metrics(fvals_delivered: np.ndarray, w_eval: np.ndarray,
               pf_at_w_eval: np.ndarray, pf_dense: np.ndarray) -> Dict:
    """max point-to-oracle distance (over query weights) and IGD."""
    scal = fvals_delivered @ np.stack([w_eval, 1.0 - w_eval])  # (m, n_w)
    pick = np.argmin(scal, axis=0)                             # (n_w,)
    f_hat = fvals_delivered[pick]                              # (n_w, 2)
    dists = np.linalg.norm(f_hat - pf_at_w_eval, axis=1)
    d2 = ((pf_dense[:, None, :] - fvals_delivered[None, :, :]) ** 2).sum(-1)
    igd = float(np.mean(np.sqrt(d2.min(axis=1))))
    return {"max_point_to_oracle": float(np.max(dists)), "igd": igd,
            "f_hat_at_w_eval": f_hat}


def front_uniformity(fvals_delivered: np.ndarray, query_ws: np.ndarray,
                     problem) -> Dict:
    """CV and Gap Ratio of the front produced at the given query weights."""
    scal = fvals_delivered @ np.stack([query_ws, 1.0 - query_ws])
    f_pts = fvals_delivered[np.argmin(scal, axis=0)]
    seg = np.linalg.norm(np.diff(f_pts, axis=0), axis=1)
    seg = seg[seg > 0]
    if seg.size < 2:
        return {"cv": float("nan"), "gap_ratio": float("nan")}
    return {"cv": float(np.std(seg) / np.mean(seg)),
            "gap_ratio": float(np.max(seg) / np.min(seg))}


# =====================================================================
#  Figures
# =====================================================================
def _pseudo_t0_pair(t_a: np.ndarray, t_b: np.ndarray):
    """Session-3 convention, SHARED variant (Jul 26 fix): both curves'
    t=0 checkpoint is plotted at the same pseudo-abscissa, (first real
    time of EITHER curve)/3, so the two lines start from one point —
    both start at the identical x0 bundle, so their y0 is equal too.
    """
    t_a = np.asarray(t_a, dtype=float).copy()
    t_b = np.asarray(t_b, dtype=float).copy()
    pos = np.concatenate([t_a[t_a > 0.0], t_b[t_b > 0.0]])
    t0 = (pos.min() / 3.0) if pos.size else 1e-3
    if t_a.size and t_a[0] <= 0.0:
        t_a[0] = t0
    if t_b.size and t_b[0] <= 0.0:
        t_b[0] = t0
    return t_a, t_b


def fig_gn_curves(bl: Dict, ad_strict: List[float], ad: Dict, eps: float,
                  out_dir: str,
                  baseline_label: str =
                  "uniform discretisation r=11 (MSVRG, certified)") -> None:
    for xkey, xlabel, fname, logx in (
        ("cpu_times", "CPU time (s), method work only\n"
         "(node/round milestone checkpoints; curves end at the last "
         "GN improvement)", "fig1_gn_vs_cpu.png",
         True),
        ("grad_evals_history", "total gradient evaluations (grad-equivalents)\n"
         "(milestone + spaced checkpoints shown; dense per-segment data "
         "in raw_histories.npz)",
         "fig2_gn_vs_grads.png", False),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        x_bl = np.asarray(bl[xkey], dtype=float)
        x_ad = np.asarray(ad[xkey], dtype=float)
        y_bl = np.minimum.accumulate(
            np.asarray(bl["delivered_gn_strict_history"], dtype=float))
        y_ad = np.minimum.accumulate(np.asarray(ad_strict, dtype=float))
        # Milestones are the duplicate records the node/round-end blocks
        # write right after a segment record (same grads, same count); if
        # none exist (numeric cadence), every point is its own milestone
        # fallback downstream.
        def _milestones(x_grads, counts):
            g = np.asarray(x_grads, dtype=float)
            c = np.asarray(counts)
            keep = [0] + [i for i in range(1, len(g))
                          if g[i] == g[i - 1] and c[i] == c[i - 1]]
            return np.asarray(sorted(set(keep)), dtype=int)
        kb = _milestones(bl["grad_evals_history"], bl["delivered_history"])
        ka = _milestones(ad["grad_evals_history"], ad["m_history"])
        if logx:
            # Jul 27 (user request, second pass): fig1 displays only the
            # node/round MILESTONE checkpoints — with per-segment
            # recording the full point set collapses into vertical
            # bursts separated by overhead plateaus (stop screens,
            # lambda searches), which reads poorly.  The dense data
            # stays in raw_histories.npz.
            if len(kb) >= 3 and len(ka) >= 2:
                x_bl, y_bl = x_bl[kb], y_bl[kb]
                x_ad, y_ad = x_ad[ka], y_ad[ka]
            # Jul 27 (user request): the CPU picture ends at each curve's
            # LAST GN improvement — the trailing flat segment is stop-
            # verification cost only (no descent).  Full CPU totals stay
            # in summary.json.
            j_bl = int(np.flatnonzero(y_bl == y_bl[-1])[0])
            j_ad = int(np.flatnonzero(y_ad == y_ad[-1])[0])
            x_bl, y_bl = x_bl[:j_bl + 1], y_bl[:j_bl + 1]
            x_ad, y_ad = x_ad[:j_ad + 1], y_ad[:j_ad + 1]
            x_bl, x_ad = _pseudo_t0_pair(x_bl, x_ad)
            ax.set_xscale("log")
        else:
            # Jul 27 (user request, third pass): at full per-segment
            # density fig2 reads as a solid band.  DISPLAY the milestone
            # checkpoints plus segment records spaced ~1/10 of the
            # x-span (the old node/round-level look with a few points
            # added in between); duplicate-x records collapse to one
            # marker; endpoints always shown.  Display only — the dense
            # arrays stay in raw_histories.npz untouched.
            def _spaced(x, keep_ms, n_span=10):
                x = np.asarray(x, dtype=float)
                keep = set(int(i) for i in keep_ms)
                dx = ((x[-1] - x[0]) / float(n_span)
                      if x[-1] > x[0] else 0.0)
                last = None
                for i in range(len(x)):
                    if i in keep:
                        last = x[i]
                        continue
                    if last is None or (x[i] - last) >= dx:
                        keep.add(i)
                        last = x[i]
                keep.add(len(x) - 1)
                idx = sorted(keep)
                out: List[int] = []
                for i in idx:
                    if out and x[i] == x[out[-1]]:
                        out[-1] = i
                    else:
                        out.append(i)
                return np.asarray(out, dtype=int)
            sb = _spaced(x_bl, kb)
            sa = _spaced(x_ad, ka)
            x_bl, y_bl = x_bl[sb], y_bl[sb]
            x_ad, y_ad = x_ad[sa], y_ad[sa]
        ax.plot(x_bl, y_bl, marker="s", ms=3.5, color=BASELINE_COLOR,
                label=baseline_label)
        ax.plot(x_ad, y_ad, marker="o", ms=3.5,
                color=ADAPTIVE_COLOR, label="adaptive bundle (MSVRG)")
        ax.axhline(eps, color="gray", ls="--", lw=1,
                   label=f"target eps = {eps:g}")
        ax.axhline(2 * eps / 3, color="gray", ls=":", lw=1,
                   label="stop line 2eps/3")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("worst-case GN* over the simplex\n"
                      "(strict in-family search, off cost axes)")
        ax.set_title("Offline-bandit toy: common-meter stationarity "
                     f"(eps = {eps:g})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close(fig)


def nondominated_2d(fvals: np.ndarray) -> np.ndarray:
    """Indices of the nondominated subset (2-D minimisation), sorted by
    F1 ascending.  Weakly dominated points and duplicates are dropped."""
    order = np.lexsort((fvals[:, 1], fvals[:, 0]))
    keep: List[int] = []
    best_f2 = np.inf
    for i in order:
        if fvals[i, 1] < best_f2:
            keep.append(int(i))
            best_f2 = float(fvals[i, 1])
    return np.asarray(keep, dtype=int)


def pf_front_metrics(front: np.ndarray, pf_dense: np.ndarray) -> Dict:
    """Discovered-front metrics against the dense closed-form front.

    max_front_to_oracle: max over the method's front points of the
    distance to the oracle front (worst off-front error of a point the
    method presents).  igd: mean over oracle-front samples of the
    distance to the nearest front point (how much of the true front the
    discovered front covers).  Both are query-free."""
    d = np.sqrt(((pf_dense[:, None, :] - front[None, :, :]) ** 2).sum(-1))
    return {"max_front_to_oracle": float(d.min(axis=0).max()),
            "igd": float(d.min(axis=1).mean())}


def fig_pareto(problem, cfg: Dict, bl_cloud: np.ndarray, ad_cloud: np.ndarray,
               bl_front: np.ndarray, ad_front: np.ndarray,
               bl_pf: Dict, ad_pf: Dict, out_dir: str) -> None:
    w_dense = np.linspace(0.0, 1.0, cfg["n_dense_oracle"])
    curve = np.array([problem.f_pf(w) for w in w_dense])
    ref_ws = problem.arc_uniform_weights(cfg["n_pf_ref_points"],
                                         cfg["n_dense_oracle"])
    ref_pts = np.array([problem.f_pf(w) for w in ref_ws])

    # Jul 27 (user request): the headline is each method's NONDOMINATED
    # DISCOVERED FRONT — the nondominated subset of ALL its evaluated
    # points (pre-prune), no query-weight selection involved.  This
    # compares the methods' ability to DISCOVER the front; the delivery
    # policy is out of the picture.
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.plot(curve[:, 0], curve[:, 1], "-", color="0.35", lw=1.5,
            label="exact plug-in PF (closed form)")
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], "*", color="0.6", ms=10, mew=0,
            label=f"SURF arc-uniform reference ({cfg['n_pf_ref_points']} pts)")
    ax.plot(bl_cloud[:, 0], bl_cloud[:, 1], "s", color=BASELINE_COLOR, ms=2.5,
            alpha=0.18, mew=0, label=f"baseline evaluated ({len(bl_cloud)})")
    ax.plot(ad_cloud[:, 0], ad_cloud[:, 1], "o", color=ADAPTIVE_COLOR, ms=2.5,
            alpha=0.18, mew=0, label=f"adaptive evaluated ({len(ad_cloud)})")
    ax.plot(bl_front[:, 0], bl_front[:, 1], "-s", color=BASELINE_COLOR,
            ms=6, lw=1.0, mec="white", mew=0.5,
            label=f"baseline nondominated front "
                  f"({len(bl_front)} of {len(bl_cloud)})")
    ax.plot(ad_front[:, 0], ad_front[:, 1], "-o", color=ADAPTIVE_COLOR,
            ms=6, lw=1.0, mec="white", mew=0.5,
            label=f"adaptive nondominated front "
                  f"({len(ad_front)} of {len(ad_cloud)})")
    ax.set_xlabel("F1(theta)")
    ax.set_ylabel("F2(theta)")
    ax.set_title("Offline-bandit toy: nondominated discovered fronts "
                 f"(eps = {cfg['epsilon']:g})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    note = (
        f"baseline:  max front-to-oracle dist = "
        f"{bl_pf['max_front_to_oracle']:.3e},  IGD = {bl_pf['igd']:.3e}\n"
        f"adaptive:  max front-to-oracle dist = "
        f"{ad_pf['max_front_to_oracle']:.3e},  IGD = {ad_pf['igd']:.3e}"
    )
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.12, 0.02, note, fontsize=9, family="monospace")
    fig.savefig(os.path.join(out_dir, "fig3_pareto_front.png"), dpi=160)
    plt.close(fig)


def fig_value_profile(w_eval: np.ndarray, prof_bl: np.ndarray,
                      prof_ad: np.ndarray, prof_bl_final: np.ndarray,
                      prof_ad_final: np.ndarray, node_ws: np.ndarray,
                      budget_label: str, eps: float, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    floor = 1e-17
    for w in node_ws:
        ax.axvline(w, color="0.85", lw=0.8, zorder=0)
    ax.plot(w_eval, np.maximum(prof_bl, floor), color=BASELINE_COLOR, lw=1.4,
            label=f"baseline @ {budget_label}")
    ax.plot(w_eval, np.maximum(prof_ad, floor), color=ADAPTIVE_COLOR, lw=1.4,
            label=f"adaptive @ {budget_label}")
    ax.plot(w_eval, np.maximum(prof_bl_final, floor), color=BASELINE_COLOR,
            lw=1.0, ls="--", alpha=0.6, label="baseline @ own delivery")
    ax.plot(w_eval, np.maximum(prof_ad_final, floor), color=ADAPTIVE_COLOR,
            lw=1.0, ls="--", alpha=0.6, label="adaptive @ own delivery")
    ax.set_yscale("log")
    ax.set_xlabel("query weight w   (lambda = (w, 1-w); "
                  "light verticals = the 12 grid nodes)")
    ax.set_ylabel("scalarized objective gap  delta_value(w)")
    ax.set_title("Value gap per query weight vs the closed-form optimum "
                 f"(eps = {eps:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_value_gap_profile.png"), dpi=160)
    plt.close(fig)


def fig_value_profile_matched(w_eval: np.ndarray, prof_bl: np.ndarray,
                              prof_ad: np.ndarray, node_ws: np.ndarray,
                              budget_label: str, eps: float,
                              out_dir: str) -> None:
    """Jul 27 (user request): companion to fig4 showing ONLY the two
    matched-budget (solid) profiles — the equal-spend comparison.
    Second pass: method-name legend labels; the budget lives in the
    title."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    floor = 1e-17
    for w in node_ws:
        ax.axvline(w, color="0.85", lw=0.8, zorder=0)
    ax.plot(w_eval, np.maximum(prof_bl, floor), color=BASELINE_COLOR, lw=1.6,
            label="Momentum-SVRG baseline")
    ax.plot(w_eval, np.maximum(prof_ad, floor), color=ADAPTIVE_COLOR, lw=1.6,
            label="Momentum-SVRG adaptive bundle method")
    ax.set_yscale("log")
    ax.set_xlabel("query weight w   (lambda = (w, 1-w); "
                  "light verticals = the 12 grid nodes)")
    ax.set_ylabel("scalarized objective gap  delta_value(w)")
    ax.set_title(f"Value gap per query weight at {budget_label} "
                 f"(eps = {eps:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,
                             "fig4b_value_gap_profile_matched.png"), dpi=160)
    plt.close(fig)


def fig_value_profile_owndelivery(w_eval: np.ndarray, prof_bl: np.ndarray,
                                  prof_ad: np.ndarray, node_ws: np.ndarray,
                                  bl_grads: float, ad_grads: float,
                                  eps: float, out_dir: str) -> None:
    """Jul 27 (user request, eps1e-3 pass): third fig4 companion — each
    method's FULL-delivery (final) profile only, both drawn solid (user
    request, second pass).  Own-endpoint comparison; the budgets differ
    and are quoted in the legend."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    floor = 1e-17
    for w in node_ws:
        ax.axvline(w, color="0.85", lw=0.8, zorder=0)
    # Jul 27 (user request, eps1e-4 pass): plain method names in the
    # legend; the per-curve grad totals stay in summary.json.
    ax.plot(w_eval, np.maximum(prof_bl, floor), color=BASELINE_COLOR, lw=1.6,
            label="momentum-SVRG baseline")
    ax.plot(w_eval, np.maximum(prof_ad, floor), color=ADAPTIVE_COLOR, lw=1.6,
            label="momentum-SVRG adaptive bundle method")
    ax.set_yscale("log")
    ax.set_xlabel("query weight w   (lambda = (w, 1-w); "
                  "light verticals = the 12 grid nodes)")
    ax.set_ylabel("scalarized objective gap  delta_value(w)")
    ax.set_title("Value gap per query weight at own delivery "
                 f"(eps = {eps:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,
                             "fig4c_value_gap_profile_owndelivery.png"),
                dpi=160)
    plt.close(fig)


def fig_value_convergence(bl: Dict, bl_vh: List[float], ad: Dict,
                          ad_vh: List[float], eps: float,
                          out_dir: str,
                          baseline_label: str = "baseline (r=11, MSVRG)"
                          ) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, xkey, xlabel, logx in (
        (axes[0], "cpu_times", "CPU time (s), method work only", True),
        (axes[1], "grad_evals_history",
         "total gradient evaluations (grad-equivalents)", False),
    ):
        x_bl = np.asarray(bl[xkey], dtype=float)
        x_ad = np.asarray(ad[xkey], dtype=float)
        if logx:
            x_bl, x_ad = _pseudo_t0_pair(x_bl, x_ad)
            ax.set_xscale("log")
        ax.plot(x_bl, np.minimum.accumulate(bl_vh), marker="s", ms=3.5,
                color=BASELINE_COLOR, label=baseline_label)
        ax.plot(x_ad, np.minimum.accumulate(ad_vh), marker="o", ms=3.5,
                color=ADAPTIVE_COLOR, label="adaptive bundle (MSVRG)")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("eps_value = max_w delta_value(w)  (best so far)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Value-gap convergence against the closed-form oracle "
                 f"(eps = {eps:g})", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_value_gap_convergence.png"),
                dpi=160)
    plt.close(fig)


# =====================================================================
#  Main
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon", type=float, default=1e-2)
    ap.add_argument("--eval-every", type=float, default=10.0,
                    help="checkpoint cadence in grad-equivalents; "
                         "<= 0 means every node/round")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-grad-evals", type=float, default=200_000.0)
    ap.add_argument("--max-wall", type=float, default=3600.0)
    ap.add_argument("--out-root", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "output",
        "Bandit_toy/bandit_toy_surf_without_256_checkpoints"))
    args = ap.parse_args()

    cfg = build_config(args)
    sub = "smoke" if cfg["smoke"] else f"eps{eps_tag(cfg['epsilon'])}"
    out_dir = os.path.abspath(os.path.join(args.out_root, sub))
    os.makedirs(out_dir, exist_ok=True)
    eps = cfg["epsilon"]

    print(f"== Offline-bandit toy | eps={eps:g} | out: {out_dir}", flush=True)

    # ---- problem + calibration (setup, untimed) ----------------------
    problem = make_bandit_toy(T=cfg["T"], noise_std=cfg["noise_std"],
                              data_seed=cfg["data_seed"], A=cfg["A"],
                              tau=cfg["tau"], alpha=cfg["alpha"])
    cal = calibrate_L(problem, safety=cfg["L_safety"])
    L = np.asarray(cal["L"], dtype=float)
    x0 = np.zeros(problem.d)
    print(f"   L calibrated: {L[0]:.4f}, {L[1]:.4f} "
          f"({cal['n_hessians']} Hessians, safety {cfg['L_safety']})",
          flush=True)

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
        record_segment_checkpoints=True,
        joint_oracle=problem.joint_oracle,
        return_points=True, return_grams=True, verbose=cfg["smoke"],
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
        record_segment_checkpoints=True,
        joint_oracle=problem.joint_oracle, verbose=cfg["smoke"],
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

    w_eval = np.linspace(0.0, 1.0, cfg["n_w_eval"])
    f_star_scal = np.array([problem.scalarized_opt(w) for w in w_eval])
    pf_at_w_eval = np.array([problem.f_pf(w) for w in w_eval])
    pf_dense = np.array([problem.f_pf(w) for w in
                         np.linspace(0.0, 1.0, cfg["n_dense_oracle"])])

    bl_points = np.asarray(bl["delivered_points"], dtype=float)
    bl_fvals = np.array([problem.joint_oracle(x)[0] for x in bl_points])
    ad_fvals_full = np.asarray(ad["pre_prune"]["fvals"], dtype=float)
    ad_points_delivered = np.asarray(ad["bundle"].points, dtype=float)
    ad_fvals_delivered = np.array([problem.joint_oracle(x)[0]
                                   for x in ad_points_delivered])

    bl_val = value_gap_machinery(bl_fvals, bl["delivered_history"],
                                 w_eval, f_star_scal)
    ad_val = value_gap_machinery(ad_fvals_full, ad["m_history"],
                                 w_eval, f_star_scal)

    # matched grad budget (equal-budget marker of the shorter run)
    budget_grads = min(bl["grad_evals_history"][-1],
                       ad["grad_evals_history"][-1])
    j_bl = int(np.flatnonzero(
        np.asarray(bl["grad_evals_history"]) <= budget_grads + 1e-9)[-1])
    j_ad = int(np.flatnonzero(
        np.asarray(ad["grad_evals_history"]) <= budget_grads + 1e-9)[-1])
    prof_bl_matched = bl_val["profile_at_prefix"](bl["delivered_history"][j_bl])
    prof_ad_matched = ad_val["profile_at_prefix"](ad["m_history"][j_ad])

    bl_pf = pf_metrics(bl_fvals, w_eval, pf_at_w_eval, pf_dense)
    ad_pf = pf_metrics(ad_fvals_delivered, w_eval, pf_at_w_eval, pf_dense)

    # Jul 27 (user request): query-free discovered-front comparison —
    # nondominated subset of ALL evaluated points (pre-prune) per method.
    bl_front_nd = bl_fvals[nondominated_2d(bl_fvals)]
    ad_front_nd = ad_fvals_full[nondominated_2d(ad_fvals_full)]
    bl_pf_front = pf_front_metrics(bl_front_nd, pf_dense)
    ad_pf_front = pf_front_metrics(ad_front_nd, pf_dense)

    ref_ws = problem.arc_uniform_weights(cfg["n_pf_ref_points"],
                                         cfg["n_dense_oracle"])
    uni_ws = np.linspace(0.0, 1.0, cfg["n_pf_ref_points"])
    uniformity = {
        "baseline": {"at_uniform_w": front_uniformity(bl_fvals, uni_ws, problem),
                     "at_arc_uniform_w": front_uniformity(bl_fvals, ref_ws,
                                                          problem)},
        "adaptive": {"at_uniform_w": front_uniformity(ad_fvals_delivered,
                                                      uni_ws, problem),
                     "at_arc_uniform_w": front_uniformity(ad_fvals_delivered,
                                                          ref_ws, problem)},
    }

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
    node_ws = (np.asarray(bl["grid"])[:, 0] if bl["grid"] is not None
               else np.linspace(0, 1, cfg["resolution"] + 1))
    fig_gn_curves(bl, ad_strict, ad, eps, out_dir)
    fig_pareto(problem, cfg, bl_fvals, ad_fvals_full,
               bl_front_nd, ad_front_nd, bl_pf_front, ad_pf_front, out_dir)
    fig_value_profile(w_eval, prof_bl_matched, prof_ad_matched,
                      bl_val["final_profile"], ad_val["final_profile"],
                      node_ws, f"matched budget {budget_grads:.0f} grads",
                      eps, out_dir)
    fig_value_profile_matched(w_eval, prof_bl_matched, prof_ad_matched,
                              node_ws,
                              f"matched budget {budget_grads:.0f} grads",
                              eps, out_dir)
    fig_value_profile_owndelivery(w_eval, bl_val["final_profile"],
                                  ad_val["final_profile"], node_ws,
                                  float(bl["grad_evals_history"][-1]),
                                  float(ad["grad_evals_history"][-1]),
                                  eps, out_dir)
    fig_value_convergence(bl, bl_val["eps_value_history"], ad,
                          ad_val["eps_value_history"], eps, out_dir)

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
        "statistical_layer": {
            "reward_sup_gap": problem.reward_sup_gap(),
            "cdf_sup_gap": problem.cdf_sup_gap(cfg["n_dense_oracle"]),
        },
        "baseline": {k: bl[k] for k in (
            "stop_reason", "global_stop_gn", "n_nodes", "n_served",
            "censored_nodes",
            "solved_nodes", "served_by_share", "served_by_chain",
            "n_delivered", "segments_total", "minibatch_steps_total",
            "safeguard_retries", "L_scale_final", "joint_calls",
            "ifo_minibatch_total", "grad_equiv_total", "wall_seconds",
            "metric_seconds", "delivered_gn_strict", "grid_worst_best_val",
            "epoch_len")},
        "adaptive": {k: ad[k] for k in (
            "stop_reason", "joint_calls", "ifo_minibatch_total",
            "grad_equiv_total", "L_scale_final", "inner_cap_hits",
            "lambda_search_seconds", "lambda_tier_mode")},
        "adaptive_bundle_m": {"pre_prune": int(ad["m_history"][-1]),
                              "delivered": int(ad["bundle"].m)},
        "process_time_s": {"baseline": bl_process_s, "adaptive": ad_process_s},
        "wall_time_s": {"baseline": bl_wall_s, "adaptive": ad_wall_s},
        "scorer_seconds_adaptive_prefix": scorer_seconds,
        "pf_metrics": {"baseline": {k: bl_pf[k] for k in
                                    ("max_point_to_oracle", "igd")},
                       "adaptive": {k: ad_pf[k] for k in
                                    ("max_point_to_oracle", "igd")}},
        # Jul 27: query-free discovered-front metrics (fig3 annotation).
        # Sets: nondominated subset of ALL evaluated points per method
        # (baseline: all delivered; adaptive: pre-prune bundle).
        "pf_front_nondominated": {
            "baseline": {**bl_pf_front,
                         "n_front": int(len(bl_front_nd)),
                         "n_evaluated": int(len(bl_fvals))},
            "adaptive": {**ad_pf_front,
                         "n_front": int(len(ad_front_nd)),
                         "n_evaluated": int(len(ad_fvals_full))}},
        "front_uniformity_secondary": uniformity,
        "eps_value_final": {"baseline": bl_val["eps_value_history"][-1],
                            "adaptive": ad_val["eps_value_history"][-1]},
        "readouts": readouts,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)

    np.savez_compressed(
        os.path.join(out_dir, "raw_histories.npz"),
        w_eval=w_eval, f_star_scal=f_star_scal, pf_at_w_eval=pf_at_w_eval,
        bl_cpu=np.asarray(bl["cpu_times"]),
        bl_grads=np.asarray(bl["grad_evals_history"]),
        bl_gn_strict=np.asarray(bl["delivered_gn_strict_history"]),
        bl_eps_value=np.asarray(bl_val["eps_value_history"]),
        bl_points=bl_points, bl_fvals=bl_fvals,
        bl_delivered_history=np.asarray(bl["delivered_history"]),
        bl_grid=np.asarray(bl["grid"]),
        bl_node_best_val=np.asarray(bl["node_best_val"]),
        ad_cpu=np.asarray(ad["cpu_times"]),
        ad_grads=np.asarray(ad["grad_evals_history"]),
        ad_gn_strict=np.asarray(ad_strict),
        ad_eps_value=np.asarray(ad_val["eps_value_history"]),
        ad_m_history=np.asarray(ad["m_history"]),
        ad_points_pre_prune=np.asarray(ad["pre_prune"]["points"]),
        ad_fvals_pre_prune=ad_fvals_full,
        ad_points_delivered=ad_points_delivered,
        ad_fvals_delivered=ad_fvals_delivered,
        bl_front_nd=bl_front_nd, ad_front_nd=ad_front_nd,
        prof_bl_matched=prof_bl_matched, prof_ad_matched=prof_ad_matched,
        prof_bl_final=bl_val["final_profile"],
        prof_ad_final=ad_val["final_profile"],
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
