"""A/B experiment runner: checkpoints WITHOUT the external 256-start metric.

What this runs
--------------
One plateau configuration (K=5) and one crossover configuration (96x96),
with the exact parameters of the corresponding runs in ``output/plateau/``
and ``output/crossover/`` — but using the measurement-variant modules

    baseline_without_256_checkpoints.py
    algorithm_without_256_checkpoints.py

whose checkpoints record each method's OWN worst-case GN for the current
round (the baseline: max over grid nodes of its latest own-weight step
gradient; the adaptive method: its own latest lambda-search value) instead
of the shared external 256-start ``pc_star`` yardstick.  Trajectories are
unaffected — the metric is pure observation — so the A/B comparison
isolates the measurement definition.

Outputs (originals untouched)
-----------------------------
    output/without_256_checkpoints/
      README.md
      plateau/K5_p6_n30_h4_tanh_r6_B30000/      (summary.json + 2 plots)
      crossover/d11522_h96x96_tanh_n50000_B2000/ (summary.json + 2 plots)
      ab_plateau_K5_gn_vs_grad_evals.png     \\  four A/B figures, each with
      ab_plateau_K5_gn_vs_cpu_time.png        \\ 4 curves: baseline/adaptive
      ab_crossover_h96x96_gn_vs_grad_evals.png / x with-256/self-reported
      ab_crossover_h96x96_gn_vs_cpu_time.png /

Usage
-----
    python run_experiments_without_256_checkpoints.py           # full runs
    python run_experiments_without_256_checkpoints.py --smoke   # tiny check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

# experiments must be imported before the algorithm modules: it loads
# objectives_torch (and torch's OpenMP runtime) before any cyipopt import.
import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from experiments import (  # noqa: E402
    _make_mlp_problem,
    make_mlp_initial_point,
    detect_plateau,
    _plot_plateau_pair,
    _BL_KW,
    _A2_KW,
)
import matplotlib.pyplot as plt  # noqa: E402
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
OUTPUT_ROOT_ORIG = HERE.parent.parent / "output"

# The two configurations, exactly as requested (and exactly matching the
# with-256 runs they are compared against).
PLATEAU_CONFIG = dict(
    K=5, p=6, n=30, hidden_sizes=[4], activation="tanh",
    coarse_resolution=6, n_passes=100_000, steps_per_point_per_pass=5,
    max_grad_evals=30_000,
    baseline_eval_every_n_grads=1_200, adaptive_eval_every_n_grads=1_200,
    max_outer=1_000_000, max_inner=25, lambda_max_starts=64,
    prune_inner=True,
)
PLATEAU_TAG = "K5_p6_n30_h4_tanh_r6_B30000"
PLATEAU_DETECT = dict(window=5, relative_improvement_tol=0.05,
                      consecutive_windows=2)

CROSSOVER_CONFIG = dict(
    K=2, p=20, n=50_000, hidden_sizes=[96, 96], activation="tanh",
    coarse_resolution=10, n_passes=100_000, steps_per_point_per_pass=5,
    max_grad_evals=2_000,
    baseline_eval_every_n_grads=153, adaptive_eval_every_n_grads=153,
    max_outer=1_000_000, max_inner=5, lambda_max_starts=8,
    prune_inner=True, plateau_window=4, plateau_consecutive_windows=2,
)
CROSSOVER_TAG = "d11522_h96x96_tanh_n50000_B2000"
CROSSOVER_DETECT = dict(window=4, relative_improvement_tol=0.05,
                        consecutive_windows=2)


def run_config(config: Dict, detect_opts: Dict, out_dir: Path,
               sweep_name: str) -> Dict:
    """Run one equal-budget comparison with self-reported checkpoints."""
    out_dir.mkdir(parents=True, exist_ok=True)
    objectives, gradients, L, joint = _make_mlp_problem(
        K=config["K"], p=config["p"], n=config["n"],
        hidden_sizes=config["hidden_sizes"], seed=DATA_SEED,
        activation=config["activation"], w_true_scale=1.0,
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(
        K=config["K"], p=config["p"], hidden_sizes=config["hidden_sizes"],
        seed=INIT_SEED,
    )
    d = int(x0.size)
    if not ipopt_available():
        raise RuntimeError("IPOPT is required but unavailable.")

    print(f"=== {sweep_name}: {out_dir.name} (self-reported checkpoints) ===",
          flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bl = uniform_discretisation(
            K=config["K"], objectives=objectives, grad_objectives=gradients,
            L=L, x0=x0, resolution=config["coarse_resolution"],
            n_passes=config["n_passes"],
            steps_per_point_per_pass=config["steps_per_point_per_pass"],
            eval_every_n_grads=config["baseline_eval_every_n_grads"],
            max_grad_evals=config["max_grad_evals"],
            evaluate_coverage=True, joint_oracle=joint_oracle, verbose=True,
        )
        a2 = algorithm_adaptive(
            K=config["K"], d=d, objectives=objectives,
            grad_objectives=gradients, L=L, x0=x0,
            max_outer=config["max_outer"], max_inner=config["max_inner"],
            eval_every_n_grads=config["adaptive_eval_every_n_grads"],
            target_cov=None, lambda_solver="ipopt", require_ipopt=True,
            lambda_max_starts=config["lambda_max_starts"],
            prune_inner=config["prune_inner"],
            max_grad_evals=config["max_grad_evals"],
            joint_oracle=joint_oracle, verbose=True,
        )
    warning_messages = sorted({str(w.message) for w in caught})

    plateaus = {
        "baseline": detect_plateau(
            bl["cov_history"], bl["grad_evals_history"], bl["cpu_times"],
            **detect_opts),
        "ipopt_adaptive": detect_plateau(
            a2["cov_history"], a2["grad_evals_history"], a2["cpu_times"],
            **detect_opts),
    }
    plateau_ratio = None
    if plateaus["baseline"]["found"] and plateaus["ipopt_adaptive"]["found"]:
        lvl = plateaus["ipopt_adaptive"]["plateau_level"]
        if lvl and lvl > 0:
            plateau_ratio = plateaus["baseline"]["plateau_level"] / lvl

    summary = {
        "metric": "self_reported (no 256-start checkpoint solve)",
        "config": _json_ready(config),
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

    title = (
        f"MLP {sweep_name} (K={config['K']}, p={config['p']}, "
        f"n={config['n']}, hidden_sizes={config['hidden_sizes']}, "
        f"activation={config['activation']}, self-reported checkpoints)"
    )
    _plot_plateau_pair(
        bl, plateaus["baseline"], "Baseline", _BL_KW,
        a2, plateaus["ipopt_adaptive"], "IPOPT Adaptive", _A2_KW,
        x_history_key="grad_evals_history",
        x_label="total gradient evaluations",
        title=title + ": Baseline vs IPOPT Adaptive",
        out_path=str(out_dir / "gn_vs_grad_evals.png"),
    )
    _plot_plateau_pair(
        bl, plateaus["baseline"], "Baseline", _BL_KW,
        a2, plateaus["ipopt_adaptive"], "IPOPT Adaptive", _A2_KW,
        x_history_key="cpu_times",
        x_label="CPU time (s, log scale)",
        title=title + ": Baseline vs IPOPT Adaptive (CPU time)",
        out_path=str(out_dir / "gn_vs_cpu_time.png"),
        x_log=True,
        mark_equal_time=True,
    )
    return summary


# =====================================================================
#  A/B comparison figures (4 curves each)
# =====================================================================
def _ab_curves(old_summary: Dict, new_summary: Dict, x_key: str):
    """Return the four (label, style, x, y) curves for one A/B figure."""
    curves = []
    for summary, suffix, ls, mfc in (
        (old_summary, "GN* yardstick (256-start)", "-", None),
        (new_summary, "self-reported", "--", "none"),
    ):
        for method, color, marker in (
            ("baseline", "#d62728", "s"),
            ("adaptive", "#9467bd", "^"),
        ):
            block = summary[method]
            style = dict(color=color, linestyle=ls, marker=marker,
                         markersize=4, linewidth=1.6)
            if mfc is not None:
                style["markerfacecolor"] = mfc
            curves.append((
                f"{method} — {suffix}",
                style,
                np.asarray(block[x_key], dtype=float),
                np.asarray(block["cov_history"], dtype=float),
            ))
    return curves


def ab_plot(old_summary: Dict, new_summary: Dict, *, x_key: str,
            x_label: str, title: str, out_path: Path,
            x_log: bool = False) -> None:
    curves = _ab_curves(old_summary, new_summary, x_key)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x0_pseudo = None
    if x_log:
        positive = [float(v) for _, _, x, _ in curves for v in x if v > 0]
        x0_pseudo = (min(positive) / 3.0) if positive else 1e-3
    for label, style, x, y in curves:
        y = np.maximum(y, np.finfo(float).tiny)
        if x_log:
            x = np.where(x > 0, x, x0_pseudo)
        ax.plot(x, y, label=label, **style)
    ax.set_xlabel(x_label)
    ax.set_ylabel("worst-case GN (per each curve's own definition)")
    ax.set_yscale("log")
    if x_log:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def config_mismatches(old_summary: Dict, new_config: Dict) -> List[str]:
    """Compare the shared config keys of the old run and the new one."""
    old_cfg = old_summary.get("config", {})
    out = []
    for key, new_val in new_config.items():
        if key in old_cfg and _json_ready(old_cfg[key]) != _json_ready(new_val):
            out.append(f"{key}: with-256 run had {old_cfg[key]!r}, "
                       f"this run has {new_val!r}")
    return out


README_TEMPLATE = """# A/B experiment: checkpoints without the 256-start metric

Generated by `Original_py/run_experiments_without_256_checkpoints.py`.
The original code and all directories outside this folder are untouched.

## What is different here (and only this)

At every checkpoint the ORIGINAL protocol scores the current point set
with an external yardstick: a fixed 256-start IPOPT maximisation of
GN(B) = max_lambda min_i ||grad F_lambda(x_i)||^2 (the `pc_star` solve),
identical for both methods.  In THIS variant each method instead records
the worst-case GN it computed ITSELF during the current round:

- **Baseline** (`baseline_without_256_checkpoints.py`): the worst (max)
  most-recently-computed ||grad F_{{lambda_i}}(x_i)||^2 across the grid
  nodes — each node scored at its OWN weight lambda_i, reusing the
  gradients its steps already computed (a node's value lags its position
  by one step, and by up to one pass for nodes not yet revisited; every
  node starts with its value at x0).  Weights BETWEEN grid nodes do not
  enter this number at all.
- **Adaptive method** (`algorithm_without_256_checkpoints.py`): its own
  most recent lambda-search value (the same max-min criterion, maximised
  by the method's own working search with `lambda_max_starts` starts —
  {plateau_starts} here for the plateau run, {crossover_starts} for the
  crossover run — read at the start of the current outer round, so it
  lags the bundle by that round's inner steps).

The algorithms themselves are byte-for-byte the algorithms of
`baseline.py` / `algorithm.py`: the metric is pure observation, so each
run's TRAJECTORY (points visited, budget spent) is identical to its
with-256 counterpart; only the recorded curve differs.  CPU-time axes
exclude checkpoint measurement cost in both variants, as always.

## Runs in this folder

- `plateau/{plateau_tag}/` — K=5 plateau configuration, budget 30k.
- `crossover/{crossover_tag}/` — 96x96 crossover configuration, budget 2k.

Both use the exact parameters of the corresponding with-256 runs in
`output/plateau/` and `output/crossover/` (seeds {data_seed}/{init_seed}).
Config check against those runs: {mismatch_note}

Each run directory holds the standard `summary.json`,
`gn_vs_grad_evals.png`, `gn_vs_cpu_time.png` (same plot conventions as the
main sweeps; the y-value is each method's SELF-REPORTED worst-case GN).

## The four A/B figures

`ab_plateau_K5_gn_vs_grad_evals.png`, `ab_plateau_K5_gn_vs_cpu_time.png`,
`ab_crossover_h96x96_gn_vs_grad_evals.png`,
`ab_crossover_h96x96_gn_vs_cpu_time.png` — each shows FOUR curves:

- solid red / solid purple: baseline / adaptive scored by the original
  256-start GN* yardstick (read from the existing with-256 summary.json);
- dashed red / dashed purple: the same two runs scored by their own
  self-reported worst-case GN (this folder's runs).

## How to read them honestly

The two dashed curves are DIFFERENT quantities, from each other and from
GN*: the baseline's self-report never looks between grid nodes, so it can
keep falling (gradient descent grinds each node's own gradient) while the
true worst case over ALL weights — which is what GN* estimates — stalls at
the discretisation floor.  The adaptive method's self-report is the same
max-min criterion as GN*, just maximised with its own search strength
instead of 256 starts, so its dashed curve should track its solid curve
closely.  The gap between the dashed and solid baseline curves is
therefore the honest picture of what the grid method cannot see about
itself; it is NOT a valid cross-method quality comparison.  For method
quality claims, use the main sweeps' GN* results.

CPU note: the with-256 curves come from the earlier serial runs on the
same machine; wall-clock positions are comparable but not identical
run-to-run (the gradient-axis figures are exactly comparable).
"""


def write_readme(root: Path, mismatches: List[str]) -> None:
    note = "no differences." if not mismatches else "; ".join(mismatches)
    root.joinpath("README.md").write_text(
        README_TEMPLATE.format(
            plateau_tag=PLATEAU_TAG,
            crossover_tag=CROSSOVER_TAG,
            plateau_starts=PLATEAU_CONFIG["lambda_max_starts"],
            crossover_starts=CROSSOVER_CONFIG["lambda_max_starts"],
            data_seed=DATA_SEED,
            init_seed=INIT_SEED,
            mismatch_note=note,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny budgets, separate _smoke output folder")
    args = parser.parse_args()

    plateau_cfg = dict(PLATEAU_CONFIG)
    crossover_cfg = dict(CROSSOVER_CONFIG)
    root_name = "without_256_checkpoints_comparison"
    if args.smoke:
        root_name += "_smoke"
        plateau_cfg.update(max_grad_evals=3_000,
                           baseline_eval_every_n_grads=600,
                           adaptive_eval_every_n_grads=600)
        crossover_cfg.update(max_grad_evals=400,
                             baseline_eval_every_n_grads=80,
                             adaptive_eval_every_n_grads=80)
    root = OUTPUT_ROOT_ORIG / root_name

    t0 = time.time()
    plateau_summary = run_config(
        plateau_cfg, PLATEAU_DETECT,
        root / "v1_IPOPT_vs_original_baseline_plateau_self_report" / PLATEAU_TAG,
        "plateau comparison")
    crossover_summary = run_config(
        crossover_cfg, CROSSOVER_DETECT,
        root / "v1_IPOPT_vs_original_baseline_crossover_self_report" / CROSSOVER_TAG,
        "plateau comparison")

    # A/B figures against the existing with-256 runs.
    mismatches: List[str] = []
    for old_path, new_summary, cfg, stem, nice in (
        (OUTPUT_ROOT_ORIG / "with_256_checkpoints"
         / "v1_IPOPT_vs_original_baseline_plateau" / PLATEAU_TAG / "summary.json",
         plateau_summary, plateau_cfg, "ab_plateau_K5",
         "plateau K=5 (30k budget)"),
        (OUTPUT_ROOT_ORIG / "with_256_checkpoints"
         / "v1_IPOPT_vs_original_baseline_crossover" / CROSSOVER_TAG / "summary.json",
         crossover_summary, crossover_cfg, "ab_crossover_h96x96",
         "crossover 96x96 (2k budget)"),
    ):
        old_summary = json.loads(old_path.read_text(encoding="utf-8"))
        mismatches += [f"[{stem}] {m}"
                       for m in config_mismatches(old_summary, cfg)]
        ab_plot(old_summary, new_summary,
                x_key="grad_evals_history",
                x_label="total gradient evaluations",
                title=f"A/B — {nice}: 256-start GN* vs self-reported",
                out_path=root / f"{stem}_gn_vs_grad_evals.png")
        ab_plot(old_summary, new_summary,
                x_key="cpu_times",
                x_label="CPU time (s, log scale)",
                title=f"A/B — {nice}: 256-start GN* vs self-reported "
                      "(CPU time)",
                out_path=root / f"{stem}_gn_vs_cpu_time.png",
                x_log=True)

    write_readme(root, mismatches)
    if mismatches:
        print("CONFIG MISMATCHES vs the with-256 runs:")
        for m in mismatches:
            print("  " + m)
    print(f"DONE in {time.time() - t0:.0f}s -> {root}", flush=True)


if __name__ == "__main__":
    main()
