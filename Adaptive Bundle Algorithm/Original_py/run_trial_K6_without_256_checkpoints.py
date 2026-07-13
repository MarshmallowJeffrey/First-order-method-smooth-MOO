"""Parameter TRIAL: K=6 at 96x96, budget mode, self-reported checkpoints.

What this is
------------
A one-off parameter-combination trial (July 11), NOT part of any existing
experiment family: the crossover configuration's problem scaled from K=2
to K=6 (hidden_sizes=[96,96] kept), run once in budget mode on the
without-256-checkpoints measurement track, producing the two standard
figures

    worst-case GN vs total gradient evaluations
    worst-case GN vs CPU time (log x-axis, equal-time marker)

with the same plotting machinery (`_plot_plateau_pair`) and the same
summary.json schema as `run_experiments_without_256_checkpoints.py`.
"worst-case GN" is each method's SELF-REPORTED value, as everywhere on
this track: the baseline reports the max over its grid nodes of its own
latest per-node gradient value; the adaptive method reports its own most
recent lambda-search value (its `lambda_max_starts`-start maximisation of
the max-min criterion).  No external 256-start solves anywhere.

Parameter derivation (user fixes: K=6, hidden_sizes=[96,96], and r >= 10;
max_grad_evals may be RAISED; everything else adjustable)
--------------------------------------------------------------------------
Reference: output/crossover/d11522_h96x96_tanh_n50000_B2000/summary.json
(K=2, p=20, n=50000, tanh, r=10, n_passes=1e5, steps_per_point_per_pass=5,
B=2000, eval_every=153, max_outer=1e6, max_inner=5, lambda_max_starts=8,
prune_inner=True, seeds 7/8).

A B=2000 calibration run at K=6 (July 11: r=2 stand-in grid, 64-start
search, max_inner=5; its folder was discarded because r=2 is below the
user's floor) measured the two costs that dimension everything here:

    joint-oracle call:  ~0.13 s   (43.3 s per 2,000 grads, K=6, n=50k)
    lambda search:      ~0.0137 s per start per bundle point, per round
                        (linear in bundle size m; m grows +1 per round)

Changes from the reference, and why:

- coarse_resolution r=10: user floor.  At K=6 the grid has C(15,5) =
  3003 nodes (vs 11 at K=2) and one baseline pass costs
  nodes*steps*K = 3003*5*6 = 90,090 grads.
- max_grad_evals 2,000 -> 180,180 = exactly TWO full baseline passes.
  Within pass 1 the baseline's self-reported worst case still contains
  never-visited nodes sitting at x0, so its curve cannot move before the
  pass completes; two passes show polish + re-polish.  Baseline gradient
  work: ~65 min at 0.13 s/call.
- max_inner 5 -> 25 (the plateau family's value for K>=3): a round costs
  K*max_inner = 150 grads; fewer, heavier rounds keep the bundle — and
  the per-round search cost, which grows linearly with it — in check.
- lambda_max_starts 8 -> 64 (this repo's K=6 precedent): at K=6 the
  search domain is the 5-D simplex and the search value IS the plotted
  metric on this track; 8 starts would under-search it and bias the
  curve optimistically.  Costs CPU only (charged to the adaptive time
  axis), no gradients.
- max_outer 1e6 -> 150: the adaptive method's BINDING fuse, 150 rounds
  = 22,500 grads = 12.5% of B.  Letting it run the full budget is
  arithmetically infeasible: 180,180/150 = 1,201 rounds at the measured
  search cost is 64*0.0137*1201^2/2 s ~ 176 h; 150 rounds cost ~2.8 h.
  DISCLOSED ASYMMETRY: the baseline receives the full 180,180-grad
  budget, the adaptive method stops at its round fuse — conservative
  AGAINST the adaptive method unless it has already plateaued (the
  figures and the plateau detector show whether it has).
- eval_every: baseline 4,500 (~40 checkpoints over B), adaptive 600
  (every 4 rounds, ~37 checkpoints).  The reference's 153 meant ~13
  checkpoints at B=2,000; the cadence is scaled to keep the checkpoint
  COUNT in the same regime, not the raw interval.  Self-reported
  checkpoints cost no extra solves (the baseline reads its stored node
  values, the adaptive method reads the search value it computes every
  round anyway), and checkpoint time is excluded from the CPU axes by
  protocol.
- d 11522 -> 11910: not a choice — the output head is 96*K+K, so d
  moves with K.
Everything else keeps the reference value: p=20, n=50000, tanh, seeds
7/8, steps_per_point_per_pass=5, prune_inner=True, n_passes fuse 1e5,
plateau detection (window 4, tol 0.05, consecutive 2).  Estimated wall
time ~4-5 h serial (baseline ~1.2 h, adaptive ~3 h).

Usage:
    python run_trial_K6_without_256_checkpoints.py [--smoke]

Outputs (originals untouched):
    output/trial_K6_d<d>_h96x96_tanh_n<n>_B<B>_without_256_checkpoints/
      README.md, summary.json,
      gn_vs_grad_evals_without_256_checkpoints.png,
      gn_vs_cpu_time_without_256_checkpoints.png
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

# experiments must be imported before the algorithm modules: it loads
# objectives_torch (and torch's OpenMP runtime) before any cyipopt import.
from experiments import (  # noqa: E402
    _make_mlp_problem,
    make_mlp_initial_point,
    detect_plateau,
    _plot_plateau_pair,
    _BL_KW,
    _A2_KW,
)
from bundle import prefer_fused_joint_oracle  # noqa: E402
from baseline_without_256_checkpoints import uniform_discretisation  # noqa: E402
from algorithm_without_256_checkpoints import (  # noqa: E402
    algorithm_adaptive,
    ipopt_available,
)
from run_experiments import (  # noqa: E402
    DATA_SEED,
    INIT_SEED,
    symmetric_time_to_target,
    _result_curves,
    _json_ready,
)

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent / "output"

TRIAL_CONFIG = dict(
    K=6, p=20, n=50_000, hidden_sizes=[96, 96], activation="tanh",
    coarse_resolution=10, n_passes=100_000, steps_per_point_per_pass=5,
    max_grad_evals=180_180,  # exactly 2 baseline passes at r=10, K=6
    baseline_eval_every_n_grads=4_500, adaptive_eval_every_n_grads=600,
    max_outer=150,  # adaptive round fuse; see module docstring
    max_inner=25, lambda_max_starts=64,
    prune_inner=True,
)
# Same post-processing as the reference crossover run.
DETECT = dict(window=4, relative_improvement_tol=0.05, consecutive_windows=2)

SMOKE_CONFIG = dict(
    K=6, p=6, n=200, hidden_sizes=[8], activation="tanh",
    coarse_resolution=2, n_passes=100_000, steps_per_point_per_pass=5,
    max_grad_evals=1_500,
    baseline_eval_every_n_grads=150, adaptive_eval_every_n_grads=150,
    max_outer=1_000_000, max_inner=5, lambda_max_starts=8,
    prune_inner=True,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny problem/budget, separate output folder")
    args = parser.parse_args()
    cfg = dict(SMOKE_CONFIG if args.smoke else TRIAL_CONFIG)

    objectives, gradients, L, joint = _make_mlp_problem(
        K=cfg["K"], p=cfg["p"], n=cfg["n"], hidden_sizes=cfg["hidden_sizes"],
        seed=DATA_SEED, activation=cfg["activation"], w_true_scale=1.0,
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(K=cfg["K"], p=cfg["p"],
                                hidden_sizes=cfg["hidden_sizes"],
                                seed=INIT_SEED)
    d = int(x0.size)
    if not ipopt_available():
        raise RuntimeError("IPOPT is required but unavailable.")

    hid = "x".join(str(h) for h in cfg["hidden_sizes"])
    tag = (f"trial_K{cfg['K']}_d{d}_h{hid}_{cfg['activation']}_n{cfg['n']}"
           f"_B{cfg['max_grad_evals']}_without_256_checkpoints")
    out_dir = OUTPUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"=== K={cfg['K']} trial: {tag} (self-reported checkpoints) ===",
          flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bl = uniform_discretisation(
            K=cfg["K"], objectives=objectives, grad_objectives=gradients,
            L=L, x0=x0, resolution=cfg["coarse_resolution"],
            n_passes=cfg["n_passes"],
            steps_per_point_per_pass=cfg["steps_per_point_per_pass"],
            eval_every_n_grads=cfg["baseline_eval_every_n_grads"],
            max_grad_evals=cfg["max_grad_evals"],
            evaluate_coverage=True, joint_oracle=joint_oracle, verbose=True,
        )
        a2 = algorithm_adaptive(
            K=cfg["K"], d=d, objectives=objectives,
            grad_objectives=gradients, L=L, x0=x0,
            max_outer=cfg["max_outer"], max_inner=cfg["max_inner"],
            eval_every_n_grads=cfg["adaptive_eval_every_n_grads"],
            target_cov=None, lambda_solver="ipopt", require_ipopt=True,
            lambda_max_starts=cfg["lambda_max_starts"],
            prune_inner=cfg["prune_inner"],
            max_grad_evals=cfg["max_grad_evals"],
            joint_oracle=joint_oracle, verbose=True,
        )
    warning_messages = sorted({str(w.message) for w in caught})

    plateaus = {
        "baseline": detect_plateau(
            bl["cov_history"], bl["grad_evals_history"], bl["cpu_times"],
            **DETECT),
        "ipopt_adaptive": detect_plateau(
            a2["cov_history"], a2["grad_evals_history"], a2["cpu_times"],
            **DETECT),
    }
    plateau_ratio = None
    if plateaus["baseline"]["found"] and plateaus["ipopt_adaptive"]["found"]:
        lvl = plateaus["ipopt_adaptive"]["plateau_level"]
        if lvl and lvl > 0:
            plateau_ratio = plateaus["baseline"]["plateau_level"] / lvl

    summary = {
        "metric": "self_reported (no 256-start checkpoint solve)",
        "trial_note": ("one-off K=6 parameter trial; reference config "
                       "output/crossover/d11522_h96x96_tanh_n50000_B2000 "
                       "with user-fixed r=10, B raised to 2 baseline "
                       "passes (180,180), max_inner 5->25, "
                       "lambda_max_starts 8->64, adaptive round fuse "
                       "max_outer=150 (see module docstring and "
                       "Note/Jul_11_note.md)"),
        "config": _json_ready(cfg),
        "data_seed": DATA_SEED,
        "init_seed": INIT_SEED,
        "d": d,
        "baseline": _result_curves(bl),
        "adaptive": _result_curves(a2),
        "plateaus": _json_ready(plateaus),
        "plateau_ratio_baseline_over_adaptive": _json_ready(plateau_ratio),
        "time_to_target": _json_ready(symmetric_time_to_target(bl, a2)),
        "runtime_warnings": warning_messages,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    title = (f"MLP K={cfg['K']} trial (p={cfg['p']}, n={cfg['n']}, "
             f"hidden_sizes={cfg['hidden_sizes']}, "
             f"{cfg['activation']}, r={cfg['coarse_resolution']}, "
             f"B={cfg['max_grad_evals']}, self-reported checkpoints)")
    _plot_plateau_pair(
        bl, plateaus["baseline"], "Baseline", _BL_KW,
        a2, plateaus["ipopt_adaptive"], "IPOPT Adaptive", _A2_KW,
        x_history_key="grad_evals_history",
        x_label="total gradient evaluations",
        title=title + ": Baseline vs IPOPT Adaptive",
        out_path=str(out_dir / "gn_vs_grad_evals_without_256_checkpoints.png"),
    )
    _plot_plateau_pair(
        bl, plateaus["baseline"], "Baseline", _BL_KW,
        a2, plateaus["ipopt_adaptive"], "IPOPT Adaptive", _A2_KW,
        x_history_key="cpu_times",
        x_label="CPU time (s, log scale)",
        title=title + ": Baseline vs IPOPT Adaptive (CPU time)",
        out_path=str(out_dir / "gn_vs_cpu_time_without_256_checkpoints.png"),
        x_log=True,
        mark_equal_time=True,
    )

    n_nodes = len(bl["coarse_grid"])
    pass_cost = (n_nodes * cfg["steps_per_point_per_pass"] * cfg["K"])
    readme = f"""# K={cfg['K']} parameter trial (96x96, budget mode, without-256 track)

One-off parameter-combination trial, July 11, 2026 — NOT part of the
plateau or crossover experiment families.  Everything here was produced
by `Original_py/run_trial_K6_without_256_checkpoints.py`; the module
docstring there derives every parameter and is the full specification.
Change record: `Note/Jul_11_note.md`.

- Problem: K={cfg['K']}, p={cfg['p']}, n={cfg['n']},
  hidden_sizes={cfg['hidden_sizes']} (d={d}), {cfg['activation']},
  seeds {DATA_SEED}/{INIT_SEED} — the crossover 96x96 instance with K=2
  replaced by K=6 (d moves 11522 -> {d} because the output head is
  96*K+K).
- Budget mode, B={cfg['max_grad_evals']} gradient evaluations;
  checkpoints record SELF-REPORTED worst-case GN (baseline: max over its
  own grid nodes at their own weights, every
  {cfg['baseline_eval_every_n_grads']} grads; adaptive: its own
  {cfg['lambda_max_starts']}-start lambda-search value, every
  {cfg['adaptive_eval_every_n_grads']} grads) — no 256-start external
  solves.
- Baseline grid: r={cfg['coarse_resolution']} (user-fixed floor) ->
  {n_nodes} nodes (C(K-1+r, K-1)); one pass costs {pass_cost} grads, so
  the budget allows {cfg['max_grad_evals'] / pass_cost:.1f} passes — B
  was raised from the reference 2,000 precisely so the baseline can
  polish every node twice (within one pass its self-reported max still
  contains never-visited nodes at x0 and cannot move).
- Adaptive: max_inner=25 (plateau-family K>=3 value),
  lambda_max_starts=64 (K=6 precedent; the search value IS the plotted
  metric), and a BINDING round fuse max_outer={cfg['max_outer']}
  (= {cfg['max_outer'] * cfg['K'] * cfg['max_inner']} grads,
  {100.0 * cfg['max_outer'] * cfg['K'] * cfg['max_inner'] / cfg['max_grad_evals']:.0f}%
  of B): a July 11 calibration run measured the lambda-search cost at
  ~0.0137 s/start/bundle-point per round, which makes full-budget rounds
  (1,201) infeasible (~176 h); the asymmetry is conservative AGAINST the
  adaptive method unless it has already plateaued — the curves and the
  plateau detector show whether it has.  All other parameters equal the
  reference run `output/crossover/d11522_h96x96_tanh_n50000_B2000/`.
- Figures: `gn_vs_grad_evals_without_256_checkpoints.png`,
  `gn_vs_cpu_time_without_256_checkpoints.png` (log time axis; the
  vertical line marks the smaller of the two final times — the
  equal-time comparison point).  `summary.json` has the curves, plateau
  detection (window {DETECT['window']}, tol
  {DETECT['relative_improvement_tol']}, consecutive
  {DETECT['consecutive_windows']}), and the symmetric time-to-target
  block.

Caveats, stated once: the two curves are DIFFERENT self-reported
quantities (the baseline never looks between its grid nodes; the
adaptive value is a heuristic lower bound of an NP-hard max) — they are
each method's own honest progress meter, not a shared yardstick; for
cross-method quality claims at a shared metric, a GN* (256-start) run
would be needed.  Single instance, seeds {DATA_SEED}/{INIT_SEED};
prune_inner=True voids the full-bundle proof condition (warning recorded
in summary.json) but not the trajectories.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"DONE in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
