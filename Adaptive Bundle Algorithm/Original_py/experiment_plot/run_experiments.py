"""Unified experiment runner: plateau sweep and CPU-time crossover sweep.

This script replaces the previous per-configuration notebooks
(run_plateau*.ipynb, mlp_crossover_h*.ipynb,
mlp_complexity_crossover_experiment.ipynb, Mlp_Compare.ipynb) with one
parameterised driver.  Both experiment families run the SAME protocol —
`experiments.experiment_mlp_plateau_comparison`, i.e. an equal-budget
head-to-head — and differ only in which problem parameter the sweep varies:

* plateau sweep   : vary K (number of objectives) at fixed budget and fixed
                    grid resolution.  The baseline's grid size C(r+K-1, K-1)
                    explodes with K while the adaptive method's bundle reuse
                    does not care, so the plateau-level ratio should grow
                    with K.
* crossover sweep : vary the MLP width (parameter count d) at fixed K=2 with
                    a large sample count, so each gradient-oracle call grows
                    expensive.  The adaptive method pays an oracle-free
                    per-round overhead (lambda search + T-map algebra); the
                    baseline pays almost none.  As d grows, oracle cost
                    dominates and the adaptive method's gradient savings
                    should overtake its overhead in wall-clock time.

Time-to-target extraction (symmetric, post-processed)
-----------------------------------------------------
The old crossover notebooks used an asymmetric design: the baseline ran its
full schedule and its final GN* became the adaptive method's stopping
target, so the baseline was always charged its entire budget even if it hit
that level early.  Here both methods run the same budget and the ratios are
extracted symmetrically from the recorded curves afterwards:

    target  = max(final best-so-far GN* of baseline,
                  final best-so-far GN* of adaptive)      # reached by BOTH
    t(m)    = CPU time / gradient count at the first checkpoint where
              method m's best-so-far GN* <= target
    ratio   = t(baseline) / t(adaptive)                   # >1: adaptive wins

Because the target is the WORSE of the two final levels, both curves are
guaranteed to reach it and the ratio is always defined.

Output layout (one directory per configuration)
-----------------------------------------------
    output/<experiment>/<config-tag>/
        gn_vs_grad_evals.png    GN* vs total gradient evaluations
        gn_vs_cpu_time.png      GN* vs CPU time
        summary.json            all curves, plateaus, ratios, parameters
        README.md               parameters, why they were chosen, analysis
    output/<experiment>/README.md          cross-configuration analysis
    output/<experiment>/<summary plot>.png trend plot across the sweep

Usage
-----
    python run_experiments.py plateau            # full plateau sweep
    python run_experiments.py crossover          # full crossover sweep
    python run_experiments.py plateau --smoke    # tiny sanity-check version
    python run_experiments.py crossover --smoke

Run serially on an otherwise idle machine: the CPU-time axis is wall-clock
time (checkpoint-metric cost excluded), so concurrent load would distort it.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from experiments import (
    experiment_mlp_plateau_comparison,
    mlp_parameter_count,
)

_ADAPTIVE_BUNDLE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = _ADAPTIVE_BUNDLE_ROOT / "output"

# Shared random seeds: seed=7 is the paper's data seed (Section 5.1.1);
# init_seed controls the He initialisation of the shared starting point.
DATA_SEED = 7
INIT_SEED = 8


# =====================================================================
#  Symmetric time-to-target extraction
# =====================================================================
def best_so_far(values: Sequence[float]) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def first_reach(curve_best: np.ndarray, axis: Sequence[float], target: float
                ) -> Optional[float]:
    """Axis value at the first checkpoint where best-so-far <= target."""
    hits = np.nonzero(curve_best <= target)[0]
    if hits.size == 0:
        return None
    return float(np.asarray(axis, dtype=float)[hits[0]])


def symmetric_time_to_target(bl: Dict, a2: Dict) -> Dict:
    """Both methods race to the same target: the WORSE of their final
    best-so-far GN* levels.  Ratios > 1 mean the adaptive method wins."""
    bl_best = best_so_far(bl["cov_history"])
    a2_best = best_so_far(a2["cov_history"])
    target = float(max(bl_best[-1], a2_best[-1]))

    out = {"target_gn": target}
    for label, res, curve in (("baseline", bl, bl_best),
                              ("adaptive", a2, a2_best)):
        out[f"{label}_cpu_to_target"] = first_reach(curve, res["cpu_times"], target)
        out[f"{label}_grads_to_target"] = first_reach(
            curve, res["grad_evals_history"], target)
        out[f"{label}_final_best_gn"] = float(curve[-1])

    def _ratio(num, den):
        if num is None or den is None or den <= 0.0:
            return None
        return float(num / den)

    out["cpu_ratio_baseline_over_adaptive"] = _ratio(
        out["baseline_cpu_to_target"], out["adaptive_cpu_to_target"])
    out["grad_ratio_baseline_over_adaptive"] = _ratio(
        out["baseline_grads_to_target"], out["adaptive_grads_to_target"])
    return out


# =====================================================================
#  Per-configuration run + persistence
# =====================================================================
def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _result_curves(res: Dict) -> Dict:
    keep = ("cov_history", "cpu_times", "grad_evals_history")
    out = {k: _json_ready(res[k]) for k in keep}
    out["best_so_far"] = _json_ready(best_so_far(res["cov_history"]))
    for extra in ("L_scale_final", "inner_cap_hits", "resolution"):
        if extra in res:
            out[extra] = _json_ready(res[extra])
    return out


def run_one_config(config: Dict, out_dir: Path, rationale: str) -> Dict:
    """Run one equal-budget comparison and persist everything into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = experiment_mlp_plateau_comparison(
            verbose=True,
            output_dir=str(out_dir),
            seed=DATA_SEED,
            init_seed=INIT_SEED,
            **config,
        )
    warning_messages = sorted({str(w.message) for w in caught})

    bl, a2 = result["baseline"], result["ipopt_adaptive"]
    ttt = symmetric_time_to_target(bl, a2)

    plateaus = result["plateaus"]
    plateau_ratio = None
    if plateaus["baseline"]["found"] and plateaus["ipopt_adaptive"]["found"]:
        lvl_a = plateaus["ipopt_adaptive"]["plateau_level"]
        if lvl_a and lvl_a > 0:
            plateau_ratio = plateaus["baseline"]["plateau_level"] / lvl_a

    summary = {
        "config": _json_ready(config),
        "data_seed": DATA_SEED,
        "init_seed": INIT_SEED,
        "d": result["d"],
        "baseline": _result_curves(bl),
        "adaptive": _result_curves(a2),
        "plateaus": _json_ready(plateaus),
        "plateau_ratio_baseline_over_adaptive": _json_ready(plateau_ratio),
        "time_to_target": _json_ready(ttt),
        "runtime_warnings": warning_messages,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # The driver names its plots baseline_vs_ipopt_*.png (with _001 suffixes
    # if re-run).  Normalise to stable names for this config directory.
    for src_stem, dst_name in (
        ("baseline_vs_ipopt_gradient_evaluations", "gn_vs_grad_evals.png"),
        ("baseline_vs_ipopt_cpu_time", "gn_vs_cpu_time.png"),
    ):
        candidates = sorted(out_dir.glob(f"{src_stem}*.png"),
                            key=lambda p: p.stat().st_mtime)
        if candidates:
            shutil.move(str(candidates[-1]), out_dir / dst_name)
            for stale in candidates[:-1]:
                stale.unlink()

    _write_config_readme(out_dir, config, summary, rationale)
    return summary


def _fmt(x, spec=".3e"):
    return "n/a" if x is None else format(x, spec)


def _write_config_readme(out_dir: Path, config: Dict, summary: Dict,
                         rationale: str) -> None:
    pl_b = summary["plateaus"]["baseline"]
    pl_a = summary["plateaus"]["ipopt_adaptive"]
    ttt = summary["time_to_target"]
    lines = [
        f"# {out_dir.name}",
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(summary["config"], indent=2),
        "```",
        "",
        f"Parameter dimension d = {summary['d']}; "
        f"data seed = {summary['data_seed']}, init seed = {summary['init_seed']}.",
        "",
        "## Why these parameters",
        "",
        rationale,
        "",
        "## Results",
        "",
        "| quantity | baseline | adaptive |",
        "|---|---|---|",
        f"| final best-so-far GN* | {_fmt(ttt['baseline_final_best_gn'])} "
        f"| {_fmt(ttt['adaptive_final_best_gn'])} |",
        f"| plateau found | {pl_b['found']} | {pl_a['found']} |",
        f"| plateau level | {_fmt(pl_b['plateau_level'])} "
        f"| {_fmt(pl_a['plateau_level'])} |",
        f"| plateau onset (grad evals) | {pl_b['onset_grad_evals']} "
        f"| {pl_a['onset_grad_evals']} |",
        f"| CPU time to common target {_fmt(ttt['target_gn'])} "
        f"| {_fmt(ttt['baseline_cpu_to_target'], '.2f')} s "
        f"| {_fmt(ttt['adaptive_cpu_to_target'], '.2f')} s |",
        f"| grad evals to common target | {_fmt(ttt['baseline_grads_to_target'], '.0f')} "
        f"| {_fmt(ttt['adaptive_grads_to_target'], '.0f')} |",
        "",
        f"Plateau ratio (baseline / adaptive): "
        f"**{_fmt(summary['plateau_ratio_baseline_over_adaptive'], '.2f')}**  ",
        f"CPU-time ratio to common target (baseline / adaptive): "
        f"**{_fmt(ttt['cpu_ratio_baseline_over_adaptive'], '.2f')}**  ",
        f"Gradient ratio to common target (baseline / adaptive): "
        f"**{_fmt(ttt['grad_ratio_baseline_over_adaptive'], '.2f')}**  ",
        "(ratios > 1 mean the adaptive method is better)",
        "",
        "## Health checks",
        "",
        f"- `L_scale_final` = {summary['adaptive'].get('L_scale_final')} "
        "(>1 means the probe-estimated smoothness constants were raised at "
        "runtime by the descent safeguard; expected on ReLU MLPs)",
        f"- `inner_cap_hits` = {summary['adaptive'].get('inner_cap_hits')} "
        "(budget mode: informational only)",
        "- Runtime warnings during the run:",
        "",
    ]
    if summary["runtime_warnings"]:
        lines += [f"  - {w}" for w in summary["runtime_warnings"]]
    else:
        lines += ["  - (none)"]
    lines += [
        "",
        "## Plots",
        "",
        "![GN* vs gradient evaluations](gn_vs_grad_evals.png)",
        "",
        "![GN* vs CPU time](gn_vs_cpu_time.png)",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
#  Plateau sweep (vary K)
# =====================================================================
PLATEAU_RATIONALE = """\
Equal-budget design: both methods consume the same gradient budget on the
same problem instance and initial point; neither stops early on a target.

- `K` is the swept variable. The baseline's grid has C(r+K-1, K-1) nodes,
  so at fixed resolution r the same budget is split over rapidly more nodes
  as K grows, while each scalarised step also costs K oracle calls.
- `coarse_resolution=6` is held fixed across K so the sweep isolates the
  effect of K. It is small enough that the baseline finishes at least two
  full passes inside the budget at every K (reaching its plateau), and
  large enough that its plateau is not trivially coarse at K=3.
- `max_grad_evals=30000` gives both methods enough budget to expose their
  plateaus at every K in the sweep (verified: plateau found for both).
- `p=6, n=30, hidden=[4]` keep the oracle cheap: this experiment is about
  gradient-efficiency, so wall-clock cost is dominated by measurement, not
  by the oracle, and run times stay in minutes.
- `activation="tanh"`: the paper's analysis assumes L-smooth objectives.
  ReLU violates that assumption (gradient jumps at activation kinks force
  the descent safeguard into extreme step-size reductions that penalise
  only the adaptive method — observed L_scale_final up to 2^25 on ReLU
  runs of this very sweep), while tanh objectives are C-infinity and
  satisfy it. The paper's Section 5.1 leaves the activation free.
- `prune_inner=True` follows the paper's Section 5 implementation note
  (only the best inner candidate joins the bundle) and keeps the bundle
  small, so the per-checkpoint metric solve stays fast.
- `steps_per_point_per_pass=5` (baseline) and `max_inner=25` (adaptive)
  are the defaults used throughout the project's earlier runs.
- Checkpoint cadence 1/25 of the budget: ~25 checkpoints, comfortably above
  the plateau detector's minimum of window*consecutive_windows = 10.
"""


def plateau_configs(smoke: bool = False) -> List[Dict]:
    ks = [3, 4] if smoke else [3, 4, 5, 6]
    budget = 4000 if smoke else 30000
    configs = []
    for K in ks:
        configs.append(dict(
            K=K, p=6, n=30, hidden_sizes=[4],
            activation="tanh",
            coarse_resolution=6,
            n_passes=100000,               # budget is the effective stop
            steps_per_point_per_pass=5,
            max_grad_evals=budget,
            baseline_eval_every_n_grads=budget // 25,
            adaptive_eval_every_n_grads=budget // 25,
            max_outer=10 ** 6,
            max_inner=25,
            lambda_max_starts=64,
            prune_inner=True,
        ))
    return configs


def plateau_tag(cfg: Dict) -> str:
    return (f"K{cfg['K']}_p{cfg['p']}_n{cfg['n']}"
            f"_h{'x'.join(map(str, cfg['hidden_sizes']))}"
            f"_{cfg.get('activation', 'relu')}"
            f"_r{cfg['coarse_resolution']}_B{cfg['max_grad_evals']}")


def plot_plateau_sweep(rows: List[Dict], out_dir: Path) -> Path:
    ks = [row["config"]["K"] for row in rows]
    # Equal-budget quality ratio: defined for every run.  The plateau-level
    # ratio is only defined when BOTH methods triggered plateau detection
    # inside the budget; overlay it where it exists.
    quality = [
        row["time_to_target"]["baseline_final_best_gn"]
        / row["time_to_target"]["adaptive_final_best_gn"]
        for row in rows
    ]
    plateau = [row["plateau_ratio_baseline_over_adaptive"] or np.nan
               for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(ks, quality, marker="o", color="#1f77b4",
            label="equal-budget quality ratio (final best-so-far)")
    if np.any(np.isfinite(plateau)):
        ax.plot(ks, plateau, marker="s", ls=":", color="#9467bd",
                label="plateau-level ratio (where both plateaued)")
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel("number of objectives K")
    ax.set_ylabel("baseline GN* / adaptive GN*")
    ax.set_yscale("log")
    ax.set_title("Equal-budget GN* ratio vs K  (>1: adaptive is better)")
    ax.set_xticks(ks)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)
    path = out_dir / "plateau_ratio_vs_K.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  Crossover sweep (vary width / parameter count d)
# =====================================================================
CROSSOVER_RATIONALE = """\
Equal-budget design at K=2 with an expensive oracle; the swept variable is
the MLP width (hence the parameter count d and the per-gradient cost).

- `K=2` keeps the lambda search cheap and reliable (the simplex is one-
  dimensional), so the sweep isolates oracle cost, not search difficulty.
- `activation="tanh"`: satisfies the paper's L-smoothness assumption
  (ReLU does not; see the plateau sweep rationale).
- `n=50000, p=20` make each oracle call a genuinely expensive forward +
  backward pass over 50k samples (the regime where gradient reuse should
  pay off in wall-clock time), matching the original crossover design.
- Width grows [16,16] -> [128,128]: d spans roughly 0.9k -> 20k, sweeping
  the per-oracle cost by more than an order of magnitude while the
  adaptive method's per-round algebraic overhead (O(m*K*d) lambda search /
  T-map work, small m due to pruning) grows only linearly in d.
- `max_grad_evals=2000`: at K=2 and r=10 (11 grid nodes) this lets the
  baseline finish ~18 full passes (5 steps per node per pass), deep into
  its plateau, and gives the adaptive method ~1000 scalarised iterations.
- `coarse_resolution=10` matches the original crossover notebooks.
- `prune_inner=True` per the paper's implementation note; also keeps the
  per-checkpoint metric cost flat as the run progresses.
- The probe-based L estimate (40 probes per objective) runs during problem
  setup, before the clock starts, so it does not touch either reported
  axis. The descent safeguard repairs any probe underestimate at runtime;
  `L_scale_final` is recorded in every summary.
- Checkpoint cadence ~1/13 of the budget: plateau detection is secondary
  here (window=4, consecutive=2 still allows it); the time-to-target
  ratios are the primary output.
"""


def crossover_configs(smoke: bool = False) -> List[Dict]:
    if smoke:
        widths = [[8, 8], [16, 16]]
        n = 2000
        budget = 400
    else:
        widths = [[16, 16], [32, 32], [64, 64], [96, 96], [128, 128]]
        n = 50000
        budget = 2000
    configs = []
    for hs in widths:
        configs.append(dict(
            K=2, p=20, n=n, hidden_sizes=hs,
            activation="tanh",
            coarse_resolution=10,
            n_passes=100000,
            steps_per_point_per_pass=5,
            max_grad_evals=budget,
            baseline_eval_every_n_grads=max(2, budget // 13),
            adaptive_eval_every_n_grads=max(2, budget // 13),
            max_outer=10 ** 6,
            max_inner=5,
            lambda_max_starts=8,
            prune_inner=True,
            plateau_window=4,
            plateau_consecutive_windows=2,
        ))
    return configs


def crossover_tag(cfg: Dict) -> str:
    d = mlp_parameter_count(cfg["K"], cfg["p"], cfg["hidden_sizes"])
    return (f"d{d}_h{'x'.join(map(str, cfg['hidden_sizes']))}"
            f"_{cfg.get('activation', 'relu')}"
            f"_n{cfg['n']}_B{cfg['max_grad_evals']}")


def _best_so_far_at(times: np.ndarray, best: np.ndarray, t: float) -> float:
    """Best-so-far GN* at wall-clock time ``t``, log-linearly interpolated
    between checkpoints.

    Why interpolate: checkpoints are sparse (~13 per run), and a step-
    function reading is unstable at the exact comparison time — e.g. if the
    comparison time falls 0.2 s BEFORE a method's first checkpoint, the
    step reading charges it its initial value even though it has already
    done essentially the full interval's work (observed: a 200x swing in
    the reported ratio from a 1.4% time difference). Interpolating
    log(GN*) linearly in time between the surrounding checkpoints is the
    standard smooth-curve reading and removes that boundary artefact; the
    checkpoint values themselves are also stored in every summary.json so
    the raw step reading can always be reconstructed."""
    if t <= times[0]:
        return float(best[0])
    if t >= times[-1]:
        return float(best[-1])
    logb = np.log(np.maximum(best, np.finfo(float).tiny))
    return float(np.exp(np.interp(t, times, logb)))


def equal_time_quality_ratio(row: Dict) -> float:
    """GN* ratio (baseline / adaptive best-so-far) at the earlier of the two
    methods' total CPU times — i.e. give both methods the same wall-clock
    time (in practice the baseline's, which is far shorter) and compare the
    quality reached (log-interpolated between checkpoints; see
    ``_best_so_far_at``).  Monotone in resources and well-defined even when
    the two methods' final qualities are orders of magnitude apart, where
    the time-to-common-target ratio degenerates."""
    bl, a2 = row["baseline"], row["adaptive"]
    bl_t = np.asarray(bl["cpu_times"], dtype=float)
    a2_t = np.asarray(a2["cpu_times"], dtype=float)
    bl_b = np.asarray(bl["best_so_far"], dtype=float)
    a2_b = np.asarray(a2["best_so_far"], dtype=float)
    t_common = min(bl_t[-1], a2_t[-1])
    bl_at = _best_so_far_at(bl_t, bl_b, t_common)
    a2_at = _best_so_far_at(a2_t, a2_b, t_common)
    return float(bl_at / a2_at)


def plot_crossover_sweep(rows: List[Dict], out_dir: Path) -> Path:
    ds = np.asarray([row["d"] for row in rows], dtype=float)
    eq_time = np.asarray([equal_time_quality_ratio(row) for row in rows])
    eq_grads = np.asarray([
        row["time_to_target"]["baseline_final_best_gn"]
        / row["time_to_target"]["adaptive_final_best_gn"]
        for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, y, ylabel, title, color in (
        (axes[0], eq_time, "baseline GN* / adaptive GN*  (equal wall-clock)",
         "Equal-TIME quality ratio (>1: adaptive wins)", "#9467bd"),
        (axes[1], eq_grads, "baseline GN* / adaptive GN*  (equal gradients)",
         "Equal-GRADIENT quality ratio (>1: adaptive wins)", "#1f77b4"),
    ):
        ax.plot(ds, y, marker="o", color=color)
        for row, xi, yi in zip(rows, ds, y):
            label = "x".join(map(str, row["config"]["hidden_sizes"]))
            ax.annotate(label, (xi, yi), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
        ax.axhline(1.0, color="black", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MLP parameter count d")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
    path = out_dir / "crossover_ratio_vs_d.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# =====================================================================
#  Sweep orchestration
# =====================================================================
def run_sweep(name: str, smoke: bool) -> None:
    if name == "plateau":
        configs, tag_fn, rationale = plateau_configs(smoke), plateau_tag, PLATEAU_RATIONALE
    elif name == "crossover":
        configs, tag_fn, rationale = crossover_configs(smoke), crossover_tag, CROSSOVER_RATIONALE
    else:
        raise ValueError(f"unknown sweep {name!r}")

    home = "with_256_checkpoints/v1_IPOPT_vs_original_baseline_" + name
    sweep_dir = OUTPUT_ROOT / (home + "_smoke" if smoke else home)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in configs:
        tag = tag_fn(cfg)
        print(f"\n##### [{name}] {tag} #####")
        rows.append(run_one_config(cfg, sweep_dir / tag, rationale))

    if name == "plateau":
        plot = plot_plateau_sweep(rows, sweep_dir)
    else:
        plot = plot_crossover_sweep(rows, sweep_dir)
    print(f"\nSweep summary plot: {plot}")

    index = {
        "experiment": name,
        "smoke": smoke,
        "configs": [
            {
                "tag": tag_fn(row["config"] | {}),
                "d": row["d"],
                "plateau_ratio_baseline_over_adaptive":
                    row["plateau_ratio_baseline_over_adaptive"],
                "cpu_ratio_baseline_over_adaptive":
                    row["time_to_target"]["cpu_ratio_baseline_over_adaptive"],
                "grad_ratio_baseline_over_adaptive":
                    row["time_to_target"]["grad_ratio_baseline_over_adaptive"],
                "L_scale_final": row["adaptive"].get("L_scale_final"),
            }
            for row in rows
        ],
    }
    (sweep_dir / "sweep_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", choices=["plateau", "crossover"])
    parser.add_argument("--smoke", action="store_true",
                        help="tiny fast version to validate the pipeline")
    args = parser.parse_args()
    run_sweep(args.sweep, args.smoke)


if __name__ == "__main__":
    main()
