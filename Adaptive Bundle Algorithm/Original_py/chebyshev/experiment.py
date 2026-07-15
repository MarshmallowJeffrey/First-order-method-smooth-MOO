"""Three-way experiment: uniform baseline vs adaptive bundle vs Chebyshev.

Run from ``Adaptive Bundle Algorithm/Original_py``:

    python -m chebyshev.experiment --smoke
    python -m chebyshev.experiment --cheb-mode paper
    python -m chebyshev.experiment --cheb-mode gn-aligned

The experiment intentionally reuses the existing MLP construction,
initialisation, baseline, adaptive bundle, and GN* metric code so the only new
variable is the Chebyshev update.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Support both ``python -m chebyshev.experiment`` and
# ``python chebyshev/experiment.py`` from Original_py.
_THIS_DIR = Path(__file__).resolve().parent
_ORIGINAL_PY = _THIS_DIR.parent
if str(_ORIGINAL_PY) not in sys.path:
    sys.path.insert(0, str(_ORIGINAL_PY))

# Import experiments first: it imports the PyTorch objective backend before
# algorithm.py attempts to import cyipopt/IPOPT, avoiding OpenMP runtime
# conflicts on macOS.
from experiments import (
    _make_mlp_problem,
    _resolve_hidden_sizes,
    make_mlp_initial_point,
    mlp_parameter_count,
)
from algorithm import algorithm_adaptive
from baseline import uniform_discretisation
from bundle import prefer_fused_joint_oracle
from chebyshev.weighted import chebyshev_adaptive


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "chebyshev"

_BL_KW = dict(color="#d62728", marker="s", ms=5, lw=1.6)
_A2_KW = dict(color="#1f77b4", marker="o", ms=4, lw=1.8)
_CH_KW = dict(color="#ff7f0e", marker="^", ms=5, lw=1.8)
_LOG_CPU_ZERO_FLOOR = 1e-2


def best_so_far(values: Sequence[float]) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _result_curves(result: Dict) -> Dict:
    out = {
        "cov_history": _json_ready(result["cov_history"]),
        "best_so_far": _json_ready(best_so_far(result["cov_history"])),
        "cpu_times": _json_ready(result["cpu_times"]),
        "total_iters_history": _json_ready(result["total_iters_history"]),
        "grad_evals_history": _json_ready(result["grad_evals_history"]),
    }
    for key in (
        "resolution",
        "L_scale_final",
        "inner_cap_hits",
        "termination_reason",
        "line_search",
        "normalization_powers",
        "append_non_improving",
        "return_last_if_no_gn_improvement",
        "fallback_update",
        "use_tmap_safeguard",
        "use_active_best_start",
        "tmap_safeguard_steps",
        "candidate_source_history",
        "candidate_score_history",
        "rejected_candidate_count_history",
        "fallback_steps_history",
        "inner_status_history",
    ):
        if key in result:
            out[key] = _json_ready(result[key])
    return out


def _first_reach(curve_best: np.ndarray, axis: Sequence[float], target: float) -> Optional[float]:
    hits = np.nonzero(curve_best <= target)[0]
    if hits.size == 0:
        return None
    return float(np.asarray(axis, dtype=float)[hits[0]])


def _three_way_time_to_target(
    baseline: Dict,
    adaptive: Dict,
    chebyshev: Dict,
) -> Dict:
    curves = {
        "baseline": best_so_far(baseline["cov_history"]),
        "adaptive": best_so_far(adaptive["cov_history"]),
        "chebyshev": best_so_far(chebyshev["cov_history"]),
    }
    target = float(max(curve[-1] for curve in curves.values()))
    results = {"target_gn": target}
    raw = {
        "baseline": baseline,
        "adaptive": adaptive,
        "chebyshev": chebyshev,
    }
    for name, curve in curves.items():
        result = raw[name]
        results[f"{name}_final_best_gn"] = float(curve[-1])
        results[f"{name}_cpu_to_target"] = _first_reach(
            curve, result["cpu_times"], target
        )
        results[f"{name}_grads_to_target"] = _first_reach(
            curve, result["grad_evals_history"], target
        )

    def ratio(num, den):
        if num is None or den is None or den <= 0.0:
            return None
        return float(num / den)

    results["cpu_ratio_baseline_over_chebyshev"] = ratio(
        results["baseline_cpu_to_target"], results["chebyshev_cpu_to_target"]
    )
    results["grad_ratio_baseline_over_chebyshev"] = ratio(
        results["baseline_grads_to_target"], results["chebyshev_grads_to_target"]
    )
    results["cpu_ratio_adaptive_over_chebyshev"] = ratio(
        results["adaptive_cpu_to_target"], results["chebyshev_cpu_to_target"]
    )
    results["grad_ratio_adaptive_over_chebyshev"] = ratio(
        results["adaptive_grads_to_target"], results["chebyshev_grads_to_target"]
    )
    return results


def _plot_three_way(
    baseline: Dict,
    adaptive: Dict,
    chebyshev: Dict,
    *,
    title: str,
    out_path: Path,
    plot_best: bool,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_t, ax_i, ax_g) = plt.subplots(1, 3, figsize=(16, 4.6))
    ylabel = r"$GN^*(\mathcal{B})$"

    def y(result: Dict) -> np.ndarray:
        values = np.asarray(result["cov_history"], dtype=float)
        return best_so_far(values) if plot_best else values

    def positive_cpu_times(result: Dict) -> np.ndarray:
        times = np.asarray(result["cpu_times"], dtype=float)
        return np.where(times <= 0.0, _LOG_CPU_ZERO_FLOOR, times)

    curves = [
        (baseline, f"uniform discretisation (r={baseline['resolution']})", _BL_KW),
        (adaptive, "adaptive bundle", _A2_KW),
        (chebyshev, "weighted Chebyshev", _CH_KW),
    ]
    for result, label, style in curves:
        iteration_axis = result.get("total_iters_history", result["grad_evals_history"])
        ax_t.plot(positive_cpu_times(result), y(result), label=label, **style)
        ax_i.plot(iteration_axis, y(result), label=label, **style)
        ax_g.plot(result["grad_evals_history"], y(result), label=label, **style)

    for ax in (ax_t, ax_i, ax_g):
        ax.set_yscale("log")
        ax.set_ylabel(ylabel if plot_best else r"raw $GN^*(\mathcal{B})$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    ax_t.set_xscale("log")
    ax_t.set_xlabel("CPU time (s, log scale)")
    ax_t.set_title("GN* vs log CPU time")
    ax_i.set_xlabel("iterations / oracle calls")
    ax_i.set_title("GN* vs iterations")
    ax_g.set_xlabel("total gradient evaluations")
    ax_g.set_title("GN* vs gradient evaluations")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _chebyshev_kwargs(mode: str) -> Dict:
    if mode == "paper":
        return {
            "candidate_anchor_count": 1,
            "normalization_powers": (1.0,),
            "line_search": "none",
            "append_non_improving": True,
        }
    if mode == "gn-aligned":
        return {
            "candidate_anchor_count": 1,
            "normalization_powers": (1.0,),
            "line_search": "gn_decrease",
            "append_non_improving": False,
            "return_last_if_no_gn_improvement": False,
            "fallback_update": "scalarized_gd",
            "use_active_best_start": True,
        }
    if mode == "hybrid":
        return {
            "candidate_anchor_count": 2,
            "normalization_powers": (1.0, 0.75, 0.5),
            "line_search": "gn_decrease",
            "append_non_improving": False,
            "return_last_if_no_gn_improvement": False,
            "fallback_update": "scalarized_gd",
            "use_tmap_safeguard": True,
            "use_active_best_start": True,
        }
    if mode == "high-k":
        return {
            "candidate_anchor_count": 1,
            "normalization_powers": (0.5, 1.0),
            "line_search": "gn_decrease",
            "append_non_improving": False,
            "return_last_if_no_gn_improvement": False,
            "fallback_update": "scalarized_gd",
            "use_tmap_safeguard": False,
            "use_active_best_start": True,
        }
    raise ValueError("mode must be 'paper', 'gn-aligned', 'hybrid', or 'high-k'.")


def _parse_float_sweep(text: str) -> Sequence[float]:
    values = [float(part) for part in text.replace("x", ",").split(",") if part]
    if not values:
        raise ValueError("normalization powers must contain at least one value.")
    if any(value < 0.0 for value in values):
        raise ValueError("normalization powers must be nonnegative.")
    return values


def _chebyshev_options_from_args(args) -> Dict:
    options = dict(_chebyshev_kwargs(args.cheb_mode))
    if args.candidate_anchors is not None:
        options["candidate_anchor_count"] = args.candidate_anchors
    if args.normalization_powers is not None:
        options["normalization_powers"] = tuple(_parse_float_sweep(args.normalization_powers))
    if args.line_search is not None:
        options["line_search"] = args.line_search
    if args.fallback_update is not None:
        options["fallback_update"] = args.fallback_update
    if args.append_non_improving is not None:
        options["append_non_improving"] = args.append_non_improving
    if args.use_tmap_safeguard is not None:
        options["use_tmap_safeguard"] = args.use_tmap_safeguard
    if args.use_active_best_start is not None:
        options["use_active_best_start"] = args.use_active_best_start
    return options


def _plot_single_result(result: Dict, *, title: str, out_path: Path, plot_best: bool) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_t, ax_i, ax_g) = plt.subplots(1, 3, figsize=(16, 4.6))
    values = np.asarray(result["cov_history"], dtype=float)
    y = best_so_far(values) if plot_best else values
    times = np.asarray(result["cpu_times"], dtype=float)
    times = np.where(times <= 0.0, _LOG_CPU_ZERO_FLOOR, times)
    iteration_axis = result.get("total_iters_history", result["grad_evals_history"])

    ax_t.plot(times, y, label="weighted Chebyshev", **_CH_KW)
    ax_i.plot(iteration_axis, y, label="weighted Chebyshev", **_CH_KW)
    ax_g.plot(result["grad_evals_history"], y, label="weighted Chebyshev", **_CH_KW)
    for ax in (ax_t, ax_i, ax_g):
        ax.set_yscale("log")
        ax.set_ylabel(r"$GN^*(\mathcal{B})$" if plot_best else r"raw $GN^*(\mathcal{B})$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    ax_t.set_xscale("log")
    ax_t.set_xlabel("CPU time (s, log scale)")
    ax_t.set_title("GN* vs log CPU time")
    ax_i.set_xlabel("iterations / oracle calls")
    ax_i.set_title("GN* vs iterations")
    ax_g.set_xlabel("total gradient evaluations")
    ax_g.set_title("GN* vs gradient evaluations")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def experiment_mlp_chebyshev_only(
    *,
    verbose: bool = True,
    K: int = 4,
    p: int = 20,
    n: int = 50000,
    h: Optional[int] = None,
    hidden_sizes: Optional[Sequence[int]] = None,
    activation: str = "tanh",
    seed: int = 7,
    init_seed: Optional[int] = 8,
    eval_every_n_grads: int = 300,
    max_grad_evals: int = 2000,
    max_outer: int = 1000000,
    max_inner: int = 5,
    lambda_max_starts: int = 16,
    lambda_solver: str = "ipopt",
    dual_solver: str = "ipopt",
    require_ipopt: bool = False,
    cheb_mode: str = "high-k",
    cheb_options: Optional[Dict] = None,
    cheb_delta: float = 1e-8,
    cheb_step_scale: float = 1.0,
    cheb_L_scale: float = 1.0,
    cheb_gn_tolerance: Optional[float] = None,
    plot_best: bool = True,
    output_dir: Optional[str] = None,
) -> Dict:
    resolved_hidden_sizes = _resolve_hidden_sizes(h, hidden_sizes)
    d = mlp_parameter_count(K, p, resolved_hidden_sizes)
    actual_init_seed = seed + 1 if init_seed is None else init_seed
    output_root = Path(output_dir) if output_dir is not None else (
        _OUTPUT_ROOT
        / "cheb_only"
        / f"K{K}_p{p}_n{n}_h{'x'.join(map(str, resolved_hidden_sizes))}"
        / f"{activation}_B{max_grad_evals}_{cheb_mode}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Chebyshev-only tuning run")
    print("=" * 72)
    print(
        f"  K={K}, p={p}, n={n}, hidden_sizes={resolved_hidden_sizes}, "
        f"activation={activation}, d={d}, budget={max_grad_evals}, "
        f"cheb_mode={cheb_mode}"
    )

    objectives, gradients, L, joint = _make_mlp_problem(
        K=K,
        p=p,
        n=n,
        hidden_sizes=resolved_hidden_sizes,
        seed=seed,
        activation=activation,
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(
        K=K,
        p=p,
        hidden_sizes=resolved_hidden_sizes,
        seed=actual_init_seed,
    )
    options = dict(_chebyshev_kwargs(cheb_mode) if cheb_options is None else cheb_options)
    print(f"  L={np.round(L, 4)} | init_seed={actual_init_seed}")
    print(f"  cheb_options={options}")

    chebyshev = chebyshev_adaptive(
        K=K,
        d=d,
        objectives=objectives,
        grad_objectives=gradients,
        L=L,
        x0=x0,
        max_outer=max_outer,
        max_inner=max_inner,
        eval_every_n_grads=eval_every_n_grads,
        target_cov=None,
        max_grad_evals=max_grad_evals,
        lambda_max_starts=lambda_max_starts,
        lambda_solver=lambda_solver,
        dual_solver=dual_solver,
        require_ipopt=require_ipopt,
        delta=cheb_delta,
        step_scale=cheb_step_scale,
        L_scale=cheb_L_scale,
        gn_tolerance=cheb_gn_tolerance,
        normalization_floor=0.0,
        joint_oracle=joint_oracle,
        verbose=verbose,
        **options,
    )

    title = (
        f"MLP Chebyshev-only tuning: K={K}, p={p}, n={n}, "
        f"hidden_sizes={resolved_hidden_sizes}, d={d}, {cheb_mode}"
    )
    plot_path = _plot_single_result(
        chebyshev,
        title=title,
        out_path=output_root / "chebyshev_gn_vs_budget.png",
        plot_best=plot_best,
    )
    raw_plot_path = _plot_single_result(
        chebyshev,
        title=title + " (raw)",
        out_path=output_root / "raw_chebyshev_gn_vs_budget.png",
        plot_best=False,
    )
    summary = {
        "config": {
            "K": K,
            "p": p,
            "n": n,
            "hidden_sizes": list(resolved_hidden_sizes),
            "activation": activation,
            "eval_every_n_grads": eval_every_n_grads,
            "max_grad_evals": max_grad_evals,
            "max_outer": max_outer,
            "max_inner": max_inner,
            "lambda_max_starts": lambda_max_starts,
            "lambda_solver": lambda_solver,
            "dual_solver": dual_solver,
            "cheb_mode": cheb_mode,
            "cheb_options": options,
        },
        "data_seed": seed,
        "init_seed": actual_init_seed,
        "d": d,
        "chebyshev": _result_curves(chebyshev),
        "plots": {
            "best_so_far": plot_path,
            "raw": raw_plot_path,
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2),
        encoding="utf-8",
    )
    curve = best_so_far(chebyshev["cov_history"])
    print(
        f"\nSummary\n  chebyshev final best GN*={curve[-1]:.4e} | "
        f"ge={chebyshev['grad_evals_history'][-1]} | "
        f"cpu={chebyshev['cpu_times'][-1]:.2f}s"
    )
    print(f"  plot: {plot_path}")
    print(f"  summary: {output_root / 'summary.json'}")
    return {
        "chebyshev": chebyshev,
        "summary": summary,
        "output_dir": str(output_root),
    }


def experiment_mlp_chebyshev_comparison(
    *,
    verbose: bool = True,
    K: int = 3,
    p: int = 6,
    n: int = 30,
    h: Optional[int] = None,
    hidden_sizes: Optional[Sequence[int]] = None,
    activation: str = "tanh",
    seed: int = 7,
    init_seed: Optional[int] = 8,
    coarse_resolution: int = 6,
    n_passes: int = 100000,
    steps_per_point_per_pass: int = 5,
    eval_every_n_grads: int = 1200,
    max_grad_evals: int = 30000,
    max_outer: int = 1000000,
    max_inner: int = 25,
    lambda_max_starts: int = 64,
    lambda_solver: str = "ipopt",
    dual_solver: str = "ipopt",
    require_ipopt: bool = False,
    cheb_mode: str = "paper",
    cheb_delta: float = 1e-8,
    cheb_step_scale: float = 1.0,
    cheb_L_scale: float = 1.0,
    cheb_gn_tolerance: Optional[float] = None,
    prune_inner: bool = True,
    plot_best: bool = True,
    output_dir: Optional[str] = None,
) -> Dict:
    resolved_hidden_sizes = _resolve_hidden_sizes(h, hidden_sizes)
    d = mlp_parameter_count(K, p, resolved_hidden_sizes)
    actual_init_seed = seed + 1 if init_seed is None else init_seed
    output_root = Path(output_dir) if output_dir is not None else (
        _OUTPUT_ROOT
        / f"K{K}_p{p}_n{n}_h{'x'.join(map(str, resolved_hidden_sizes))}"
        / f"{activation}_r{coarse_resolution}_B{max_grad_evals}_{cheb_mode}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Chebyshev comparison — baseline vs adaptive bundle vs Chebyshev")
    print("=" * 72)
    print(
        f"  K={K}, p={p}, n={n}, hidden_sizes={resolved_hidden_sizes}, "
        f"activation={activation}, d={d}, budget={max_grad_evals}, "
        f"cheb_mode={cheb_mode}"
    )

    objectives, gradients, L, joint = _make_mlp_problem(
        K=K,
        p=p,
        n=n,
        hidden_sizes=resolved_hidden_sizes,
        seed=seed,
        activation=activation,
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(
        K=K,
        p=p,
        hidden_sizes=resolved_hidden_sizes,
        seed=actual_init_seed,
    )
    print(f"  L={np.round(L, 4)} | init_seed={actual_init_seed}")

    print("\n  [uniform discretisation] ...")
    baseline = uniform_discretisation(
        K=K,
        objectives=objectives,
        grad_objectives=gradients,
        L=L,
        x0=x0,
        resolution=coarse_resolution,
        n_passes=n_passes,
        steps_per_point_per_pass=steps_per_point_per_pass,
        eval_every_n_grads=eval_every_n_grads,
        max_grad_evals=max_grad_evals,
        evaluate_coverage=True,
        joint_oracle=joint_oracle,
        verbose=verbose,
    )

    print("\n  [adaptive bundle] ...")
    adaptive = algorithm_adaptive(
        K=K,
        d=d,
        objectives=objectives,
        grad_objectives=gradients,
        L=L,
        x0=x0,
        max_outer=max_outer,
        max_inner=max_inner,
        eval_every_n_grads=eval_every_n_grads,
        target_cov=None,
        lambda_max_starts=lambda_max_starts,
        lambda_solver=lambda_solver,
        require_ipopt=require_ipopt,
        max_grad_evals=max_grad_evals,
        prune_inner=prune_inner,
        joint_oracle=joint_oracle,
        verbose=verbose,
    )

    print("\n  [weighted Chebyshev] ...")
    cheb_options = _chebyshev_kwargs(cheb_mode)
    chebyshev = chebyshev_adaptive(
        K=K,
        d=d,
        objectives=objectives,
        grad_objectives=gradients,
        L=L,
        x0=x0,
        max_outer=max_outer,
        max_inner=max_inner,
        eval_every_n_grads=eval_every_n_grads,
        target_cov=None,
        max_grad_evals=max_grad_evals,
        lambda_max_starts=lambda_max_starts,
        lambda_solver=lambda_solver,
        dual_solver=dual_solver,
        require_ipopt=require_ipopt,
        delta=cheb_delta,
        step_scale=cheb_step_scale,
        L_scale=cheb_L_scale,
        gn_tolerance=cheb_gn_tolerance,
        normalization_floor=0.0,
        joint_oracle=joint_oracle,
        verbose=verbose,
        **cheb_options,
    )

    title = (
        f"MLP Chebyshev comparison: K={K}, p={p}, n={n}, "
        f"hidden_sizes={resolved_hidden_sizes}, d={d}, {cheb_mode}"
    )
    plot_path = _plot_three_way(
        baseline,
        adaptive,
        chebyshev,
        title=title,
        out_path=output_root / "gn_vs_budget.png",
        plot_best=plot_best,
    )
    raw_plot_path = _plot_three_way(
        baseline,
        adaptive,
        chebyshev,
        title=title + " (raw)",
        out_path=output_root / "raw_gn_vs_budget.png",
        plot_best=False,
    )

    time_to_target = _three_way_time_to_target(baseline, adaptive, chebyshev)
    summary = {
        "config": {
            "K": K,
            "p": p,
            "n": n,
            "hidden_sizes": list(resolved_hidden_sizes),
            "activation": activation,
            "coarse_resolution": coarse_resolution,
            "n_passes": n_passes,
            "steps_per_point_per_pass": steps_per_point_per_pass,
            "eval_every_n_grads": eval_every_n_grads,
            "max_grad_evals": max_grad_evals,
            "max_outer": max_outer,
            "max_inner": max_inner,
            "lambda_max_starts": lambda_max_starts,
            "lambda_solver": lambda_solver,
            "dual_solver": dual_solver,
            "cheb_mode": cheb_mode,
            "cheb_options": cheb_options,
            "prune_inner": prune_inner,
        },
        "data_seed": seed,
        "init_seed": actual_init_seed,
        "d": d,
        "baseline": _result_curves(baseline),
        "adaptive": _result_curves(adaptive),
        "chebyshev": _result_curves(chebyshev),
        "time_to_target": _json_ready(time_to_target),
        "plots": {
            "best_so_far": plot_path,
            "raw": raw_plot_path,
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2),
        encoding="utf-8",
    )

    print("\nSummary")
    for name, result in (
        ("baseline", baseline),
        ("adaptive", adaptive),
        ("chebyshev", chebyshev),
    ):
        curve = best_so_far(result["cov_history"])
        print(
            f"  {name:<10} final best GN*={curve[-1]:.4e} | "
            f"ge={result['grad_evals_history'][-1]} | "
            f"cpu={result['cpu_times'][-1]:.2f}s"
        )
    print(f"  common target GN*={time_to_target['target_gn']:.4e}")
    print(
        "  baseline / chebyshev ratios: "
        f"cpu={time_to_target['cpu_ratio_baseline_over_chebyshev']}, "
        f"grad={time_to_target['grad_ratio_baseline_over_chebyshev']}"
    )
    print(f"  plot: {plot_path}")
    print(f"  summary: {output_root / 'summary.json'}")

    return {
        "baseline": baseline,
        "adaptive": adaptive,
        "chebyshev": chebyshev,
        "summary": summary,
        "output_dir": str(output_root),
    }


def _parse_hidden_sizes(text: Optional[str]) -> Optional[Sequence[int]]:
    if text is None or text == "":
        return None
    return [int(part) for part in text.replace(",", "x").split("x") if part]


def _parse_int_sweep(text: str) -> Sequence[int]:
    values = [int(part) for part in text.replace("x", ",").split(",") if part]
    if not values:
        raise ValueError("--k-sweep must contain at least one integer K value.")
    if any(value < 2 for value in values):
        raise ValueError("All K values in --k-sweep must be at least 2.")
    return values


def _hidden_sizes_tag(
    h: Optional[int],
    hidden_sizes: Optional[Sequence[int]],
    classmate_width: int,
) -> str:
    if hidden_sizes is not None:
        return "x".join(str(width) for width in hidden_sizes)
    if h is not None:
        return str(h)
    return f"{classmate_width}x{classmate_width}"


def _final_best(summary: Dict, method: str) -> float:
    return float(summary[method]["best_so_far"][-1])


def _plot_k_sweep_summary(runs: Sequence[Dict], out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ks = np.asarray([run["K"] for run in runs], dtype=int)
    series = [
        ("baseline", "uniform discretisation", _BL_KW),
        ("adaptive", "adaptive bundle", _A2_KW),
        ("chebyshev", "weighted Chebyshev", _CH_KW),
    ]
    for key, label, style in series:
        values = [_final_best(run["summary"], key) for run in runs]
        ax.plot(ks, values, label=label, **style)

    ax.set_yscale("log")
    ax.set_xlabel("number of objectives K")
    ax.set_ylabel(r"final best $GN^*(\mathcal{B})$")
    ax.set_title("Final stationarity coverage vs K")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def run_k_sweep(args, hidden_sizes: Optional[Sequence[int]]) -> None:
    k_values = _parse_int_sweep(args.k_sweep)
    hidden_tag = _hidden_sizes_tag(args.h, hidden_sizes, args.classmate_crossover_width)
    base_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (
            _OUTPUT_ROOT
            / "k_sweep"
            / (
                f"p{args.p}_n{args.n}_h{hidden_tag}_{args.activation}"
                f"_r{args.r}_B{args.budget}_{args.cheb_mode}"
            )
        )
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for K in k_values:
        result = experiment_mlp_chebyshev_comparison(
            verbose=True,
            K=K,
            p=args.p,
            n=args.n,
            h=args.h,
            hidden_sizes=hidden_sizes,
            activation=args.activation,
            coarse_resolution=args.r,
            eval_every_n_grads=args.eval_every,
            max_grad_evals=args.budget,
            max_inner=args.max_inner,
            lambda_max_starts=args.lambda_starts,
            lambda_solver=args.lambda_solver,
            dual_solver=args.dual_solver,
            require_ipopt=args.require_ipopt,
            cheb_mode=args.cheb_mode,
            plot_best=not args.raw_plot,
            output_dir=str(base_dir / f"K{K}"),
        )
        runs.append({
            "K": K,
            "output_dir": result["output_dir"],
            "summary_path": str(Path(result["output_dir"]) / "summary.json"),
            "summary": result["summary"],
        })

    sweep_plot = _plot_k_sweep_summary(runs, base_dir / "final_best_gn_vs_K.png")
    index = {
        "config": {
            "K_values": list(k_values),
            "p": args.p,
            "n": args.n,
            "hidden_sizes": list(hidden_sizes) if hidden_sizes is not None else None,
            "h": args.h,
            "activation": args.activation,
            "coarse_resolution": args.r,
            "max_grad_evals": args.budget,
            "eval_every_n_grads": args.eval_every,
            "max_inner": args.max_inner,
            "lambda_max_starts": args.lambda_starts,
            "lambda_solver": args.lambda_solver,
            "dual_solver": args.dual_solver,
            "cheb_mode": args.cheb_mode,
        },
        "runs": [
            {
                "K": run["K"],
                "output_dir": run["output_dir"],
                "summary_path": run["summary_path"],
                "baseline_final_best_gn": _final_best(run["summary"], "baseline"),
                "adaptive_final_best_gn": _final_best(run["summary"], "adaptive"),
                "chebyshev_final_best_gn": _final_best(run["summary"], "chebyshev"),
                "baseline_cpu_final": float(run["summary"]["baseline"]["cpu_times"][-1]),
                "adaptive_cpu_final": float(run["summary"]["adaptive"]["cpu_times"][-1]),
                "chebyshev_cpu_final": float(run["summary"]["chebyshev"]["cpu_times"][-1]),
            }
            for run in runs
        ],
        "plots": {
            "final_best_gn_vs_K": sweep_plot,
        },
    }
    index_path = base_dir / "sweep_index.json"
    index_path.write_text(json.dumps(_json_ready(index), indent=2), encoding="utf-8")
    print(f"\nK sweep complete: {index_path}")
    print(f"K sweep plot: {sweep_plot}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cheb-mode",
        choices=["paper", "gn-aligned", "hybrid", "high-k"],
        default="hybrid",
    )
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument(
        "--k-sweep",
        default=None,
        help="Comma/x-separated K values to run sequentially, e.g. 2,3,4.",
    )
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--h", type=int, default=None)
    parser.add_argument(
        "--hidden-sizes",
        default=None,
        help=(
            "Comma/x-separated hidden layer widths. If omitted, use the "
            "classmate crossover width [w,w]."
        ),
    )
    parser.add_argument(
        "--classmate-crossover-width",
        type=int,
        choices=[16, 32, 64, 96, 128],
        default=96,
        help="Width w for the classmate crossover MLP hidden_sizes=[w,w].",
    )
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=153)
    parser.add_argument("--max-inner", type=int, default=5)
    parser.add_argument("--lambda-starts", type=int, default=8)
    parser.add_argument("--lambda-solver", choices=["ipopt", "slsqp"], default="ipopt")
    parser.add_argument("--dual-solver", choices=["ipopt", "slsqp"], default="ipopt")
    parser.add_argument("--require-ipopt", action="store_true")
    parser.add_argument("--cheb-only", action="store_true")
    parser.add_argument("--candidate-anchors", type=int, default=None)
    parser.add_argument("--normalization-powers", default=None)
    parser.add_argument(
        "--line-search",
        choices=["none", "armijo", "gn_decrease", "gn_or_armijo"],
        default=None,
    )
    parser.add_argument(
        "--fallback-update",
        choices=["none", "scalarized_gd"],
        default=None,
    )
    parser.add_argument(
        "--append-non-improving",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-tmap-safeguard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-active-best-start",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--raw-plot", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        experiment_mlp_chebyshev_comparison(
            verbose=True,
            K=2,
            p=4,
            n=12,
            hidden_sizes=[3],
            activation="tanh",
            coarse_resolution=3,
            n_passes=1000,
            steps_per_point_per_pass=2,
            eval_every_n_grads=80,
            max_grad_evals=240,
            max_outer=40,
            max_inner=3,
            lambda_max_starts=4,
            lambda_solver="slsqp",
            dual_solver="slsqp",
            cheb_mode=args.cheb_mode,
            plot_best=not args.raw_plot,
            output_dir=args.output_dir,
        )
        return

    hidden_sizes = (
        _parse_hidden_sizes(args.hidden_sizes)
        if args.hidden_sizes is not None
        else (
            None if args.h is not None
            else [args.classmate_crossover_width, args.classmate_crossover_width]
        )
    )

    if args.k_sweep is not None:
        run_k_sweep(args, hidden_sizes)
        return

    if args.cheb_only:
        experiment_mlp_chebyshev_only(
            verbose=True,
            K=args.K,
            p=args.p,
            n=args.n,
            h=args.h,
            hidden_sizes=hidden_sizes,
            activation=args.activation,
            eval_every_n_grads=args.eval_every,
            max_grad_evals=args.budget,
            max_inner=args.max_inner,
            lambda_max_starts=args.lambda_starts,
            lambda_solver=args.lambda_solver,
            dual_solver=args.dual_solver,
            require_ipopt=args.require_ipopt,
            cheb_mode=args.cheb_mode,
            cheb_options=_chebyshev_options_from_args(args),
            plot_best=not args.raw_plot,
            output_dir=args.output_dir,
        )
        return

    experiment_mlp_chebyshev_comparison(
        verbose=True,
        K=args.K,
        p=args.p,
        n=args.n,
        h=args.h,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        coarse_resolution=args.r,
        eval_every_n_grads=args.eval_every,
        max_grad_evals=args.budget,
        max_inner=args.max_inner,
        lambda_max_starts=args.lambda_starts,
        lambda_solver=args.lambda_solver,
        dual_solver=args.dual_solver,
        require_ipopt=args.require_ipopt,
        cheb_mode=args.cheb_mode,
        plot_best=not args.raw_plot,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
