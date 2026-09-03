"""run_surf_compare_K2_without_256_checkpoints.py — campaign-v2 stages
S3 (r/N ladders) and S4 (main comparison) on the locked pair 4v9, run as
TWO parallel experiments (user decision Sep 2): core A = adagrad x10,
core B = adam(alpha=1e-3, beta2=0.9).  Design: BASELINE_SURF.md +
ADAPTIVE_STEPPERS.md (Sep-2 revisions) + Note/Sep_2_note.md.

NEW FILE (Sep 2, 2026).  No existing file is modified.  Legs:

* uniform (baseline2): the S2 stepper executor with the v1 grid
  round-robin policy (_baseline_policy, s = 5 per visit);
* SURF (baseline1): baseline_surf_without_256_checkpoints.run_surf_leg;
* adaptive CCP (main method, S4 only): the S2 stepper executor with the
  CCP policy.

Stages:

* ``--stage ladders`` (S3): per core, uniform r in {10,20,30,40} and
  SURF N in {10,20,30,40}, 3 sampler seeds {41,141,241}, B = 2,500,
  eval_every = 50, audit grid 20,001.  48 runs; resume-skip.  Produces
  v2_ladders/ladders_summary.json (r*, N* per core) + ladder figure.
* ``--stage main`` (S4): per core, three legs at the chosen r*/N*,
  seeds = MAIN_SEEDS (descoped to seed 41 on Sep 3; more seeds =
  future work via resume-skip), B = 20,000, eval_every = 250,
  audit grid 200,001.  Produces
  the campaign figures per core: worst GN (norm) vs grad_equiv, worst GN
  vs CPU seconds, and the Pareto-front figure (best-per-lambda scatter +
  non-dominated frontier + color = lambda_1 + initial point; no
  threshold lines — fixed budget).  Run only after the S3 sign-off.

Usage:
    python run_surf_compare_K2_without_256_checkpoints.py --stage ladders
    python run_surf_compare_K2_without_256_checkpoints.py --stage main
"""

from __future__ import annotations

import argparse
import json
import time

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout  # noqa: F401
from baseline_without_256_checkpoints import (  # noqa: E402
    _sort_grid_for_warmstart,
    _uniform_simplex_grid,
)
from run_pure_budget_K6_without_256_checkpoints import (  # noqa: E402
    _baseline_policy,
)
from run_pure_budget_K2_ccp_without_256_checkpoints import (  # noqa: E402
    _ccp_policy,
    _stats_block,
)
from run_pure_budget_K2_mnist_pair_without_256_checkpoints import (  # noqa: E402
    CAMPAIGN_ROOT,
)
from run_stepper_pre_experiment_K2_without_256_checkpoints import (  # noqa: E402
    PAIR,
    S2_CFG,
    S2_SEEDS,
    _Args,
    _ccp_cfg,
    _run_leg_pair_stepper,
)
from run_experiments import _json_ready  # noqa: E402
from baseline_surf_without_256_checkpoints import run_surf_leg  # noqa: E402

V2_HOME = CAMPAIGN_ROOT / "v2_campaign"

# Campaign problem (user sign-off Sep 3): ridge mu = 1e-3 on BOTH
# objectives + SURF dial trim w in [0.05, 0.95].  The mu smoke showed
# the mu-only window is empty (taming the vertex arms needs mu >= 3e-2,
# learning survives only for mu <= 1e-2); the trim removes the
# zero-weight vertices that permit divergence, and mu = 1e-3 + trim
# passed all three criteria (Note/Sep_2_note.md).
CAMPAIGN_MU = 1e-3
SURF_W_MIN = 0.05

LADDER_HOME = V2_HOME / f"ladders_mu{CAMPAIGN_MU:g}"
MAIN_HOME = V2_HOME / f"main_mu{CAMPAIGN_MU:g}"

CORES = [
    ("adagrad_x10", "adagrad", {"adagrad_alpha_mult": 10.0}),
    ("adam_1e-3_b0.9", "adam", {"adam_alpha": 1e-3, "adam_beta2": 0.9}),
]
LADDER_RS = [10, 20, 30, 40]

# S4 descope (user decisions Sep 3): the main run uses ONE seed (the
# house sampler seed 41); seeds 141/241 are future work — extending
# MAIN_SEEDS and re-running --stage main fills in only the missing
# runs (resume-skip) and the figures aggregate whatever seeds exist.
# Caveat of record: with one seed, legs that finish within seed-noise
# of each other (~5-10 % per S2) cannot be adjudicated.
# Second descope (same day): S4 ran core A first; core B was then
# resurrected by user request the same evening (its adaptive run is
# shared with core-compare via resume-skip, so only the two baseline
# legs are new).
MAIN_SEEDS = (41,)
MAIN_CORES = CORES


def _ladder_args():
    return _Args(budget=2_500.0, eval_every=50.0, audit_grid=20_001, s=5,
                 smoke=False)


def _main_args():
    return _Args(budget=20_000.0, eval_every=250.0, audit_grid=200_001,
                 s=5, smoke=False)


def _load_or_run(out_dir, fn, force=False):
    if (out_dir / "summary.json").exists() and not force:
        print(f"[skip] {out_dir.name} (resume)", flush=True)
        return json.loads((out_dir / "summary.json").read_text())
    return fn(out_dir)


def _final_norm(sm):
    return sm["audited_gn_norm_history"][-1]


def stage_ladders(force=False):
    args = _ladder_args()
    t_all = time.time()
    board = {}
    for core_tag, sname, scfg in CORES:
        core_home = LADDER_HOME / core_tag
        board[core_tag] = {"uniform": {}, "surf": {}}
        for r in LADDER_RS:
            grid = _sort_grid_for_warmstart(_uniform_simplex_grid(2, r))
            finals = []
            for seed in S2_SEEDS:
                out = core_home / f"uniform_r{r}_seed{seed}"
                sm = _load_or_run(out, lambda od, g=grid: _run_leg_pair_stepper(
                    "baseline", _baseline_policy(g), PAIR, dict(S2_CFG),
                    args, od, {"r": r, "core": core_tag},
                    stepper_name=sname, stepper_cfg=scfg,
                    sampler_seed=seed, mu=CAMPAIGN_MU), force)
                finals.append(_final_norm(sm))
            board[core_tag]["uniform"][str(r)] = {
                "mean": float(np.mean(finals)),
                "per_seed": [float(v) for v in finals]}
        for N in LADDER_RS:
            finals = []
            for seed in S2_SEEDS:
                out = core_home / f"surf_N{N}_seed{seed}"
                sm = _load_or_run(out, lambda od, NN=N: run_surf_leg(
                    PAIR, dict(S2_CFG), args, od, {"core": core_tag},
                    N=NN, stepper_name=sname, stepper_cfg=scfg,
                    sampler_seed=seed, mu=CAMPAIGN_MU,
                    w_min=SURF_W_MIN), force)
                finals.append(_final_norm(sm))
            board[core_tag]["surf"][str(N)] = {
                "mean": float(np.mean(finals)),
                "per_seed": [float(v) for v in finals]}

    picks = {}
    for core_tag, _s, _c in CORES:
        b = board[core_tag]
        r_star = min(b["uniform"], key=lambda k: b["uniform"][k]["mean"])
        n_star = min(b["surf"], key=lambda k: b["surf"][k]["mean"])
        picks[core_tag] = {"r_star": int(r_star), "N_star": int(n_star)}
    LADDER_HOME.mkdir(parents=True, exist_ok=True)
    (LADDER_HOME / "ladders_summary.json").write_text(json.dumps(
        {"pair": list(PAIR), "budget": args.budget,
         "seeds": list(S2_SEEDS), "board": board, "picks": picks,
         "total_wall_seconds": time.time() - t_all}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for core_tag, _s, _c in CORES:
        b = board[core_tag]
        axes[0].plot(LADDER_RS,
                     [b["uniform"][str(r)]["mean"] for r in LADDER_RS],
                     marker="o", label=core_tag)
        axes[1].plot(LADDER_RS,
                     [b["surf"][str(N)]["mean"] for N in LADDER_RS],
                     marker="o", label=core_tag)
    axes[0].set_xlabel("uniform grid r"); axes[1].set_xlabel("SURF N")
    for ax in axes:
        ax.set_ylabel("final worst GN (norm), mean of 3 seeds")
        ax.set_yscale("log"); ax.legend(fontsize=8)
    axes[0].set_title(f"S3 ladders, pair {PAIR[0]}v{PAIR[1]}, B=2500")
    fig.tight_layout()
    fig.savefig(LADDER_HOME / "ladders.png", dpi=150)
    print("[ladders] picks:", json.dumps(picks), flush=True)
    return picks


# ---------------------------------------------------------------- S4 ----

def _best_per_lambda(seg_lams, fvals, decimals=2):
    """v1-figure convention: group delivered points by rounded lambda_1,
    keep the lowest scalarized loss per group (the x0 row, lam = nan, is
    excluded)."""
    best = {}
    for lam, f in zip(seg_lams, fvals):
        w = lam[0]
        if not np.isfinite(w):
            continue
        key = round(float(w), decimals)
        sc = key * f[0] + (1.0 - key) * f[1]
        if key not in best or sc < best[key][0]:
            best[key] = (sc, f, w)
    return sorted(((w, f) for _sc, f, w in best.values()),
                  key=lambda t: t[0])


def _nondominated(points):
    pts = sorted(points, key=lambda t: (t[1][0], t[1][1]))
    front, best_y = [], np.inf
    for w, f in pts:
        if f[1] < best_y - 1e-15:
            front.append((w, f))
            best_y = f[1]
    return front


def _front_figure(runs, x0_f, out_path, title):
    """runs = {leg_name: summary-npz pairs of ONE representative seed}.
    Two panels: full windowed view + a zoom into the knee region (the
    fronts hug the axes and the interesting structure is below ~0.6)."""
    fig, (axF, axZ) = plt.subplots(1, 2, figsize=(13, 5.5))
    markers = {"surf": "^", "uniform": "s", "adaptive_ccp": "o"}
    lines = {"surf": "#e41a1c", "uniform": "#377eb8",
             "adaptive_ccp": "#ff7f00"}
    # uniform's frontier hugs the same axis band as adaptive's and
    # would vanish underneath it — dash it and draw it on top.
    styles = {"surf": dict(lw=1.6, zorder=3),
              "adaptive_ccp": dict(lw=2.4, zorder=4),
              "uniform": dict(lw=1.6, ls="--", zorder=5)}
    W, WZ = 3.0, 0.6
    sc = None
    for leg, (seg_lams, fvals) in runs.items():
        reps = _best_per_lambda(seg_lams, fvals)
        if not reps:
            continue
        ws = [w for w, _f in reps]
        xs = [f[0] for _w, f in reps]; ys = [f[1] for _w, f in reps]
        front = _nondominated(reps)
        fx = [f[0] for _w, f in front]; fy = [f[1] for _w, f in front]
        for ax in (axF, axZ):
            sc = ax.scatter(xs, ys, c=ws, cmap="viridis", vmin=0.0,
                            vmax=1.0, marker=markers[leg], s=42,
                            edgecolors="none",
                            label=f"{leg} best per lambda")
            ax.plot(fx, fy, color=lines[leg], label=f"{leg} frontier",
                    **styles[leg])
    axF.scatter([x0_f[0]], [x0_f[1]], marker="D", color="gray", s=60,
                label="Initial point")
    fig.colorbar(sc, ax=axZ, label="lambda_1")
    for ax, w, ttl in ((axF, W, title),
                       (axZ, WZ, "zoom: knee region")):
        n_out = sum(1 for _leg, (sl, fv) in runs.items()
                    for w_, f in _best_per_lambda(sl, fv)
                    if f[0] > w or f[1] > w)
        ax.set_xlim(-0.02 * w, w); ax.set_ylim(-0.02 * w, w)
        if n_out:
            ax.text(0.98, 0.02, f"{n_out} point(s) beyond view",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="gray")
        ax.set_xlabel("f1 (lower is better)")
        ax.set_ylabel("f2 (lower is better)")
        ax.set_title(ttl, fontsize=10)
    axF.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def stage_main(force=False):
    args = _main_args()
    picks = json.loads(
        (LADDER_HOME / "ladders_summary.json").read_text())["picks"]
    t_all = time.time()
    for core_tag, sname, scfg in MAIN_CORES:
        core_home = MAIN_HOME / core_tag
        r_star = picks[core_tag]["r_star"]
        n_star = picks[core_tag]["N_star"]
        grid = _sort_grid_for_warmstart(_uniform_simplex_grid(2, r_star))
        legs = {}
        for seed in MAIN_SEEDS:
            out = core_home / f"uniform_r{r_star}_seed{seed}"
            legs.setdefault("uniform", []).append(_load_or_run(
                out, lambda od: _run_leg_pair_stepper(
                    "baseline", _baseline_policy(grid), PAIR,
                    dict(S2_CFG), args, od,
                    {"r": r_star, "core": core_tag},
                    stepper_name=sname, stepper_cfg=scfg,
                    sampler_seed=seed, mu=CAMPAIGN_MU), force))
            out = core_home / f"surf_N{n_star}_seed{seed}"
            legs.setdefault("surf", []).append(_load_or_run(
                out, lambda od: run_surf_leg(
                    PAIR, dict(S2_CFG), args, od, {"core": core_tag},
                    N=n_star, stepper_name=sname, stepper_cfg=scfg,
                    sampler_seed=seed, mu=CAMPAIGN_MU,
                    w_min=SURF_W_MIN), force))

            def _ccp_run(od):
                stats: list = []
                sm = _run_leg_pair_stepper(
                    "adaptive_ccp", _ccp_policy(2, _ccp_cfg(), stats),
                    PAIR, dict(S2_CFG), args, od, {"core": core_tag},
                    stepper_name=sname, stepper_cfg=scfg,
                    sampler_seed=seed, mu=CAMPAIGN_MU)
                sm["ccp"] = _stats_block(stats)
                (od / "summary.json").write_text(
                    json.dumps(_json_ready(sm), indent=2),
                    encoding="utf-8")
                return sm
            out = core_home / f"adaptive_ccp_seed{seed}"
            legs.setdefault("adaptive_ccp", []).append(
                _load_or_run(out, _ccp_run, force))

        # ---- figures per core ----
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for leg, sms in legs.items():
            curves = [np.asarray(sm["audited_gn_norm_history"])
                      for sm in sms]
            Lmin = min(len(c) for c in curves)
            mean_curve = np.mean([c[:Lmin] for c in curves], axis=0)
            g = np.asarray(sms[0]["ck_grads"], dtype=float)[:Lmin]
            cpu = np.mean([np.asarray(sm["ck_cpu"], dtype=float)[:Lmin]
                           for sm in sms], axis=0)
            axes[0].plot(g, mean_curve, lw=1.4, label=leg)
            axes[1].plot(cpu, mean_curve, lw=1.4, label=leg)
        axes[0].set_xlabel("total gradient evaluations (grad_equiv)")
        axes[1].set_xlabel("CPU seconds")
        for ax in axes:
            ax.set_ylabel("best-so-far worst GN (norm)")
            ax.set_yscale("log"); ax.legend(fontsize=8)
        axes[0].set_title(f"main, core {core_tag}, "
                          f"pair {PAIR[0]}v{PAIR[1]}, B=20000, "
                          f"{len(MAIN_SEEDS)} seed(s)")
        fig.tight_layout()
        core_home.mkdir(parents=True, exist_ok=True)
        fig.savefig(core_home / "worst_gn_curves.png", dpi=150)
        plt.close(fig)

        runs = {}
        for leg, dirname in (("surf", f"surf_N{n_star}_seed41"),
                             ("uniform", f"uniform_r{r_star}_seed41"),
                             ("adaptive_ccp", "adaptive_ccp_seed41")):
            npz = np.load(core_home / dirname / "grams.npz")
            runs[leg] = (np.asarray(npz["seg_lams"], dtype=float),
                         np.asarray(npz["fvals"], dtype=float))
        x0_f = runs["uniform"][1][0]
        _front_figure(runs, x0_f, core_home / "pareto_front.png",
                      f"Pareto front (seed 41), core {core_tag}, "
                      f"pair {PAIR[0]}v{PAIR[1]}")
        print(f"[main] core {core_tag} figures -> {core_home}",
              flush=True)
    print(f"[main] ALL DONE in {time.time() - t_all:.0f}s", flush=True)


CC_SEEDS = (41, 141, 241)   # Sep-3 extension: does adam's late-budget
                            # edge persist? -> 3 seeds at B = 20,000


def stage_core_compare(force=False):
    """Head-to-head of the two S2 finalist cores on the adaptive-CCP leg
    alone, at the MAIN tier (ridge mu = CAMPAIGN_MU, B = 20,000).
    Seed-41 runs are shared with stage_main via resume-skip.  User
    requests: Sep 3 (single seed), extended to CC_SEEDS the same day
    ("does adam's advantage persist past B=2,500?")."""
    args = _main_args()
    curves = {}
    for core_tag, sname, scfg in CORES:
        for seed in CC_SEEDS:
            out = MAIN_HOME / core_tag / f"adaptive_ccp_seed{seed}"

            def _ccp_run(od, sn=sname, sc=scfg, tag=core_tag, sd=seed):
                stats: list = []
                sm = _run_leg_pair_stepper(
                    "adaptive_ccp", _ccp_policy(2, _ccp_cfg(), stats),
                    PAIR, dict(S2_CFG), args, od, {"core": tag},
                    stepper_name=sn, stepper_cfg=sc,
                    sampler_seed=sd, mu=CAMPAIGN_MU)
                sm["ccp"] = _stats_block(stats)
                (od / "summary.json").write_text(
                    json.dumps(_json_ready(sm), indent=2),
                    encoding="utf-8")
                return sm
            curves.setdefault(core_tag, []).append(
                _load_or_run(out, _ccp_run, force))

    colors = {"adagrad_x10": "#1f77b4", "adam_1e-3_b0.9": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for core_tag, sms in curves.items():
        cs = [np.asarray(sm["audited_gn_norm_history"], dtype=float)
              for sm in sms]
        L = min(len(c) for c in cs)
        g = np.asarray(sms[0]["ck_grads"], dtype=float)[:L]
        cpu = np.mean([np.asarray(sm["ck_cpu"], dtype=float)[:L]
                       for sm in sms], axis=0)
        mean_c = np.mean([c[:L] for c in cs], axis=0)
        for c in cs:
            axes[0].plot(g, c[:L], lw=0.6, alpha=0.35,
                         color=colors[core_tag])
        axes[0].plot(g, mean_c, lw=1.8, color=colors[core_tag],
                     label=f"{core_tag} (mean of {len(cs)})")
        axes[1].plot(cpu, mean_c, lw=1.8, color=colors[core_tag],
                     label=core_tag)
    axes[0].set_xlabel("total gradient evaluations (grad_equiv)")
    axes[1].set_xlabel("CPU seconds")
    for ax in axes:
        ax.set_ylabel("best-so-far worst GN (norm)")
        ax.set_yscale("log"); ax.legend(fontsize=9)
    axes[0].set_title(f"adaptive-CCP leg, core A vs core B, "
                      f"pair {PAIR[0]}v{PAIR[1]}, B=20000, "
                      f"seeds {list(CC_SEEDS)}")
    fig.tight_layout()
    V2_HOME.mkdir(parents=True, exist_ok=True)
    fig.savefig(V2_HOME / "core_compare_adaptive.png", dpi=150)
    plt.close(fig)
    for tag, sms in curves.items():
        finals = [sm["audited_gn_norm_history"][-1] for sm in sms]
        print(f"[core-compare] {tag:16s} finals="
              + " ".join(f"{v:.4e}" for v in finals)
              + f"  mean={np.mean(finals):.4e}", flush=True)
    print("[core-compare] figure ->",
          V2_HOME / "core_compare_adaptive.png", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",
                        choices=["ladders", "main", "core-compare"],
                        required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "ladders":
        stage_ladders(force=args.force)
    elif args.stage == "main":
        stage_main(force=args.force)
    else:
        stage_core_compare(force=args.force)


if __name__ == "__main__":
    main()
