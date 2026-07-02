r"""
The uniform-discretisation baseline performance plots
    non-convex   (MLP)    :  GN*(B)  = sup_{lambda in Delta_K} min_{x_i\in B_m} ||grad F_lambda(x_i)||^2
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from objectives_torch import make_mlp_nonconvex
from baseline import uniform_discretisation
from bundle import prefer_fused_joint_oracle

# plot conventions (consistent with the rest of the project)
_BL_KW = dict(color="#d62728", marker="s", ms=5, lw=1.6, label="uniform discretisation")
_A2_KW = dict(color="#1f77b4", marker="o", ms=4, lw=1.8, label="adaptive bundle")
_IPOPT_KW = dict(color="#9467bd", marker="^", ms=5, lw=1.8)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PLOT_DIR = _PROJECT_ROOT / "experiment_results"
_ADAPTIVE_BUNDLE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PLATEAU_DIR = _ADAPTIVE_BUNDLE_ROOT / "output" / "plateau result"


def _resolve_hidden_sizes(
    h: Optional[int],
    hidden_sizes: Optional[Sequence[int]],
) -> List[int]:
    """Resolve the backward-compatible ``h`` shorthand to layer widths."""
    if h is not None and hidden_sizes is not None:
        raise ValueError("Pass either `h` or `hidden_sizes`, not both.")

    raw_sizes = [8] if h is None and hidden_sizes is None else (
        [h] if hidden_sizes is None else list(hidden_sizes)
    )
    if not raw_sizes:
        raise ValueError("hidden_sizes must contain at least one hidden layer.")
    if any(
        not isinstance(width, (int, np.integer))
        or isinstance(width, (bool, np.bool_))
        or width <= 0
        for width in raw_sizes
    ):
        raise ValueError(
            "Every hidden layer width must be a positive integer; "
            f"got {raw_sizes!r}."
        )
    return [int(width) for width in raw_sizes]


def mlp_parameter_count(K: int, p: int, hidden_sizes: Sequence[int]) -> int:
    """Return the flat parameter dimension for ``p -> hidden_sizes -> K``."""
    if K <= 0 or p <= 0:
        raise ValueError("K and p must both be positive.")
    sizes = _resolve_hidden_sizes(None, hidden_sizes)
    widths = [p, *sizes, K]
    return int(sum(
        fan_in * fan_out + fan_out
        for fan_in, fan_out in zip(widths[:-1], widths[1:])
    ))


def _make_mlp_problem(
    *,
    K: int,
    p: int,
    n: int,
    hidden_sizes: Sequence[int],
    seed: int,
    n_probes: int = 40,
):
    """Construct the canonical PyTorch MLP objective."""
    return make_mlp_nonconvex(
        K=K,
        p=p,
        n=n,
        hidden_sizes=list(hidden_sizes),
        seed=seed,
        n_probes=n_probes,
    )


def unique_plot_path(out_path: str) -> Path:
    """Create the output directory and return a non-overwriting PNG path.

    Bare filenames are stored under ``experiment_results/``.  Explicit
    directories are respected.  Existing files are never overwritten;
    ``_001``, ``_002``, ... is appended to the requested stem instead.
    """
    output = Path(out_path)
    if not output.is_absolute() and output.parent == Path("."):
        output = _DEFAULT_PLOT_DIR / output
    elif not output.is_absolute():
        output = _PROJECT_ROOT / output
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    candidate = output
    counter = 1
    while candidate.exists():
        candidate = output.with_name(
            f"{output.stem}_{counter:03d}{output.suffix}"
        )
        counter += 1
    return candidate


def make_mlp_initial_point(
    K: int,
    p: int,
    h: Optional[int] = None,
    seed: int = 0,
    *,
    hidden_sizes: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Create a reproducible flat He initialisation for any MLP depth.

    Parameter order is ``weight, bias`` for every linear layer, matching both
    the original one-layer NumPy layout and PyTorch's ``nn.Sequential``
    parameter iteration order. Hidden biases are small and positive so ReLU
    units are not all inactive; the output bias is zero.
    """
    if K <= 0 or p <= 0:
        raise ValueError("K and p must both be positive.")
    sizes = _resolve_hidden_sizes(h, hidden_sizes)

    rng = np.random.RandomState(seed)
    widths = [p, *sizes, K]
    blocks = []
    for layer_index, (fan_in, fan_out) in enumerate(
        zip(widths[:-1], widths[1:])
    ):
        weight = rng.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)
        is_output_layer = layer_index == len(widths) - 2
        bias = (
            np.zeros(fan_out)
            if is_output_layer
            else np.full(fan_out, 1e-2)
        )
        blocks.extend([weight.ravel(), bias])

    theta = np.concatenate(blocks)
    expected_d = mlp_parameter_count(K, p, sizes)
    if theta.shape != (expected_d,):
        raise RuntimeError(
            f"Initial point has shape {theta.shape}; expected ({expected_d},)."
        )
    return theta


def detect_plateau(
    cov_history: Sequence[float],
    grad_evals_history: Sequence[int],
    cpu_times: Sequence[float],
    *,
    window: int = 5,
    relative_improvement_tol: float = 0.05,
    consecutive_windows: int = 2,
) -> Dict:
    """Detect the onset and level of a sustained GN* plateau.

    Detection uses the best-so-far GN* curve to suppress harmless metric
    noise.  Starting at each candidate index, it checks
    ``consecutive_windows`` adjacent, non-overlapping blocks of ``window``
    checkpoints.  Every block must improve by less than
    ``relative_improvement_tol``, and the total best-so-far improvement from
    the candidate onset to the end of the run must obey the same tolerance.

    Returns a dictionary containing ``found``, ``onset_index``,
    ``onset_grad_evals``, ``onset_cpu_time``, and ``plateau_level``.
    ``plateau_level`` is the median raw GN* value from the detected onset to
    the end of the run.
    """
    cov = np.asarray(cov_history, dtype=float)
    grad_evals = np.asarray(grad_evals_history, dtype=int)
    times = np.asarray(cpu_times, dtype=float)

    if not (cov.ndim == grad_evals.ndim == times.ndim == 1):
        raise ValueError("Plateau histories must be one-dimensional.")
    if not (len(cov) == len(grad_evals) == len(times)):
        raise ValueError("Plateau histories must have identical lengths.")
    if window < 2:
        raise ValueError("window must be at least 2.")
    if consecutive_windows < 1:
        raise ValueError("consecutive_windows must be at least 1.")
    if not 0.0 <= relative_improvement_tol < 1.0:
        raise ValueError("relative_improvement_tol must be in [0, 1).")
    if np.any(~np.isfinite(cov)) or np.any(cov < 0.0):
        raise ValueError("cov_history must contain finite, nonnegative values.")

    not_found = {
        "found": False,
        "onset_index": None,
        "onset_grad_evals": None,
        "onset_cpu_time": None,
        "plateau_level": None,
    }
    required_points = window * consecutive_windows
    if len(cov) < required_points:
        return not_found

    best_so_far = np.minimum.accumulate(cov)
    tiny = np.finfo(float).tiny

    for start in range(len(cov) - required_points + 1):
        stable = True
        for block in range(consecutive_windows):
            left = start + block * window
            right = left + window - 1
            denom = max(float(best_so_far[left]), tiny)
            improvement = (best_so_far[left] - best_so_far[right]) / denom
            if improvement >= relative_improvement_tol:
                stable = False
                break

        if not stable:
            continue

        onset_value = float(best_so_far[start])
        if onset_value <= tiny:
            tail_improvement = 0.0
        else:
            tail_improvement = (onset_value - best_so_far[-1]) / onset_value
        if tail_improvement >= relative_improvement_tol:
            continue

        return {
            "found": True,
            "onset_index": int(start),
            "onset_grad_evals": int(grad_evals[start]),
            "onset_cpu_time": float(times[start]),
            "plateau_level": float(np.median(cov[start:])),
        }

    return not_found

def _plot_coverage(bl: Optional[Dict], a2: Optional[Dict], title: str, out_path: str) -> str:
    """Two-panel plot: coverage metric vs CPU time, and vs gradient evals."""
    output = unique_plot_path(out_path)
    ylabel = r"$\sup_{\lambda\in\Delta_K} [\min_{x_i\in\mathcal{B}_m} \|\nabla F_\lambda(x_i)\|^2]$"
    fig, (ax_t, ax_g) = plt.subplots(1, 2, figsize=(12, 4.6))

    if bl is not None:
        # Per-call baseline style: append the coarse-grid resolution r to the
        # legend label so the discretisation density is visible on the plot.
        bl_kw = {**_BL_KW, "label": f"uniform discretisation (r={bl['resolution']})"}
        ax_t.plot(bl["cpu_times"], bl["cov_history"], **bl_kw)
        ax_g.plot(bl["grad_evals_history"], bl["cov_history"], **bl_kw)

        # Final worst-case error reached by the uniform baseline, drawn as a
        # green horizontal reference line on both panels.
        final_err = bl["cov_history"][-1]
        final_kw = dict(color="#2ca02c", ls="--", lw=1.4,
                        label=f"baseline final error = {final_err:.3e}")
        ax_t.axhline(final_err, **final_kw)
        ax_g.axhline(final_err, **final_kw)

    if a2 is not None:
        ax_t.plot(a2["cpu_times"], a2["cov_history"], **_A2_KW)
        ax_g.plot(a2["grad_evals_history"], a2["cov_history"], **_A2_KW)

    ax_t.set_xlabel("CPU time (s)")
    ax_t.set_ylabel(ylabel)
    ax_t.set_yscale("log")
    ax_t.set_title("worst-case squared gradient norm vs CPU time")
    ax_t.grid(True, which="both", alpha=0.25)
    ax_t.legend(frameon=False, fontsize=9)

    ax_g.set_xlabel("total gradient evaluations")
    ax_g.set_ylabel(ylabel)
    ax_g.set_yscale("log")
    ax_g.set_title("worst-case squared gradient norm vs gradient evals")
    ax_g.grid(True, which="both", alpha=0.25)
    ax_g.legend(frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def _plot_plateau_pair(
    first_result: Dict,
    first_plateau: Dict,
    first_label: str,
    first_style: Dict,
    second_result: Dict,
    second_plateau: Dict,
    second_label: str,
    second_style: Dict,
    *,
    x_history_key: str,
    x_label: str,
    title: str,
    out_path: str,
) -> str:
    """Plot one pair of methods against a shared history variable."""
    output = unique_plot_path(out_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for result, plateau, label, style in (
        (first_result, first_plateau, first_label, first_style),
        (second_result, second_plateau, second_label, second_style),
    ):
        x = np.asarray(result[x_history_key], dtype=float)
        y = np.asarray(result["cov_history"], dtype=float)
        y_plot = np.maximum(y, np.finfo(float).tiny)
        plot_style = {key: value for key, value in style.items() if key != "label"}
        ax.plot(x, y_plot, label=label, **plot_style)

        if plateau["found"]:
            onset = plateau["onset_index"]
            color = plot_style["color"]
            ax.scatter(
                [x[onset]], [y_plot[onset]],
                color=color, edgecolor="black", linewidth=0.5,
                s=65, zorder=5,
                label=f"{label} plateau onset",
            )
            ax.axhline(
                max(plateau["plateau_level"], np.finfo(float).tiny),
                color=color, linestyle="--", linewidth=1.2, alpha=0.8,
                label=f"{label} plateau = {plateau['plateau_level']:.3e}",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$GN^*(\mathcal{B})$")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def _plateau_for_result(
    result: Dict,
    *,
    window: int,
    relative_improvement_tol: float,
    consecutive_windows: int,
) -> Dict:
    """Apply ``detect_plateau`` to an algorithm result dictionary."""
    return detect_plateau(
        result["cov_history"],
        result["grad_evals_history"],
        result["cpu_times"],
        window=window,
        relative_improvement_tol=relative_improvement_tol,
        consecutive_windows=consecutive_windows,
    )

def experiment_mlp_gn_coverage(
    verbose: bool = True,
    K: int = 5, p: int = 10, n: int = 20,
    h: Optional[int] = None, seed: int = 10,
    init_seed: Optional[int] = None,
    coarse_resolution: int = 9,
    n_passes: int = 15, steps_per_point_per_pass: int = 50,
    eval_every_n_grads: int = 5000,
    max_outer: int = 1000, max_inner: Optional[int] = None,
    lambda_max_starts: int = 256,
    prune_inner: bool = False,
    run_baseline: bool = True, run_adaptive: bool = True,
    out_path: Optional[str] = None,
    hidden_sizes: Optional[Sequence[int]] = None,
    max_grad_evals: Optional[int] = None,
    baseline_max_grad_evals: Optional[int] = None,
    adaptive_max_grad_evals: Optional[int] = None,
    lambda_solver: str = "ipopt",
    require_ipopt: bool = False,
    l_n_probes: int = 40,
    oracle_benchmark_repeats: int = 0,
) -> Dict:
    """Non-convex MLP: GN* coverage, adaptive vs uniform.

    ``h=...`` is retained as a one-layer shorthand for
    ``hidden_sizes=[h]``. All architectures use the PyTorch backend.

    ``max_grad_evals`` remains the backward-compatible shared budget.
    ``baseline_max_grad_evals`` and ``adaptive_max_grad_evals`` may override
    it independently.  This permits a fair time-to-target experiment: the
    baseline first defines its final GN* target, while adaptive may use a
    larger safety budget and stops as soon as it reaches that same target.
    """
    print("=" * 68)
    print("Coverage experiment — MLP (non-convex), metric = GN*")
    print("=" * 68)
    resolved_hidden_sizes = _resolve_hidden_sizes(h, hidden_sizes)
    d = mlp_parameter_count(K, p, resolved_hidden_sizes)
    baseline_budget = (
        max_grad_evals
        if baseline_max_grad_evals is None
        else baseline_max_grad_evals
    )
    adaptive_budget = (
        max_grad_evals
        if adaptive_max_grad_evals is None
        else adaptive_max_grad_evals
    )
    if max_inner is None:
        max_inner = 50 if K == 4 else 25
    setup_start = time.perf_counter()
    objs, grads, L, joint = _make_mlp_problem(
        K=K,
        p=p,
        n=n,
        hidden_sizes=resolved_hidden_sizes,
        seed=seed,
        n_probes=l_n_probes,
    )
    problem_setup_time = time.perf_counter() - setup_start
    joint_oracle = prefer_fused_joint_oracle(joint)
    actual_init_seed = seed + 1 if init_seed is None else init_seed
    x0 = make_mlp_initial_point(
        K=K,
        p=p,
        hidden_sizes=resolved_hidden_sizes,
        seed=actual_init_seed,
    )
    if (
        not isinstance(oracle_benchmark_repeats, (int, np.integer))
        or isinstance(oracle_benchmark_repeats, (bool, np.bool_))
        or oracle_benchmark_repeats < 0
    ):
        raise ValueError(
            "oracle_benchmark_repeats must be a nonnegative integer; "
            f"got {oracle_benchmark_repeats!r}."
        )
    oracle_ms = None
    if oracle_benchmark_repeats > 0:
        joint_oracle(x0)  # warm-up; excluded from the measurement
        benchmark_start = time.perf_counter()
        for _ in range(int(oracle_benchmark_repeats)):
            joint_oracle(x0)
        oracle_ms = (
            (time.perf_counter() - benchmark_start)
            * 1000.0
            / int(oracle_benchmark_repeats)
        )
    print(
        f"  K={K}, p={p}, n={n}, hidden_sizes={resolved_hidden_sizes}, "
        f"backend=torch, d={d}  |  "
        f"L={np.round(L,3)} | init_seed={actual_init_seed} | "
        f"setup={problem_setup_time:.2f}s"
        + (f" | oracle={oracle_ms:.2f}ms" if oracle_ms is not None else "")
    )

    bl = None
    if run_baseline:
        if verbose:
            print("\n  [uniform discretisation] ...")
        bl = uniform_discretisation(
            K=K, objectives=objs, grad_objectives=grads, L=L, x0=x0,
            resolution=coarse_resolution, n_passes=n_passes,
            steps_per_point_per_pass=steps_per_point_per_pass,
            eval_every_n_grads=eval_every_n_grads,
            max_grad_evals=baseline_budget,
            evaluate_coverage=True, joint_oracle=joint_oracle, verbose=verbose)

    a2 = None
    if run_adaptive:
        # Import after the selected objective backend has been loaded. On
        # macOS, importing cyipopt before Torch can initialise a conflicting
        # OpenMP runtime and abort the process.
        from algorithm import algorithm_adaptive

        if verbose:
            print("\n  [adaptive bundle] ...")
        target_cov = bl["cov_history"][-1] if bl is not None else None
        a2 = algorithm_adaptive(
            K=K, d=d, objectives=objs, grad_objectives=grads, L=L, x0=x0,
            max_outer=max_outer, max_inner=max_inner,
            eval_every_n_grads=eval_every_n_grads,
            target_cov=target_cov,
            lambda_max_starts=lambda_max_starts,
            lambda_solver=lambda_solver,
            require_ipopt=require_ipopt,
            max_grad_evals=adaptive_budget,
            prune_inner=prune_inner,
            joint_oracle=joint_oracle, verbose=verbose)

    architecture_tag = "x".join(str(width) for width in resolved_hidden_sizes)
    requested_out_path = out_path or (
        f"mlp_torch_K{K}_p{p}_n{n}_h{architecture_tag}"
        f"_seed{seed}_init{actual_init_seed}"
        f"_r{coarse_resolution}.png"
    )
    path = _plot_coverage(
        bl, a2,
        title=(
            "MLP with parameters: K={}, p={}, n={}, hidden_sizes={}, "
            "backend=torch, d={}"
        ).format(
            K, p, n, resolved_hidden_sizes, d,
        ),
        out_path=requested_out_path)
    if bl is not None:
        print(f"\n  BL  final GN* = {bl['cov_history'][-1]:.4e}  "
              f"(ge={bl['grad_evals_history'][-1]}, cpu={bl['cpu_times'][-1]:.2f}s)")
    if a2 is not None:
        print(f"  A2  final GN* = {a2['cov_history'][-1]:.4e}  "
              f"(ge={a2['grad_evals_history'][-1]}, cpu={a2['cpu_times'][-1]:.2f}s, "
              f"bundle={a2['bundle'].m})")
    return {
        "baseline": bl,
        "algorithm2": a2,
        "plot": path,
        "init_seed": actual_init_seed,
        "x0": x0.copy(),
        "hidden_sizes": resolved_hidden_sizes,
        "mlp_backend": "torch",
        "d": d,
        "problem_setup_time": problem_setup_time,
        "oracle_ms": oracle_ms,
        "l_n_probes": l_n_probes,
        "max_grad_evals": max_grad_evals,
        "baseline_max_grad_evals": baseline_budget,
        "adaptive_max_grad_evals": adaptive_budget,
        "lambda_solver": lambda_solver,
        "require_ipopt": require_ipopt,
    }


def experiment_mlp_plateau_comparison(
    verbose: bool = True,
    K: int = 3,
    p: int = 4,
    n: int = 60,
    h: Optional[int] = None,
    seed: int = 10,
    init_seed: Optional[int] = None,
    coarse_resolution: int = 10,
    n_passes: int = 1000,
    steps_per_point_per_pass: int = 10,
    baseline_eval_every_n_grads: Optional[int] = None,
    adaptive_eval_every_n_grads: int = 2000,
    max_grad_evals: int = 30000,
    max_outer: int = 10000,
    max_inner: int = 25,
    lambda_max_starts: int = 256,
    prune_inner: bool = False,
    plateau_window: int = 5,
    plateau_relative_improvement_tol: float = 0.05,
    plateau_consecutive_windows: int = 2,
    output_dir: str = str(_DEFAULT_PLATEAU_DIR),
    hidden_sizes: Optional[Sequence[int]] = None,
) -> Dict:
    """Compare baseline and strict-IPOPT adaptive plateaus.

    Both methods share one generated MLP problem, initial point, fused
    oracle, and gradient-evaluation budget.  Adaptive target-coverage stopping
    is disabled so each method can continue below the baseline's final level
    and expose its own plateau.
    """
    resolved_hidden_sizes = _resolve_hidden_sizes(h, hidden_sizes)
    # objectives_torch is imported at module load time, before algorithm.py
    # attempts to import cyipopt. This ordering avoids conflicting OpenMP
    # runtime initialisation on macOS.
    from algorithm import algorithm_adaptive, ipopt_available

    if not ipopt_available():
        raise RuntimeError(
            "The plateau comparison requires IPOPT, but cyipopt/IPOPT is unavailable."
        )

    print("=" * 72)
    print("Plateau comparison — baseline vs IPOPT adaptive")
    print("=" * 72)

    d = mlp_parameter_count(K, p, resolved_hidden_sizes)
    objectives, gradients, L, joint = _make_mlp_problem(
        K=K,
        p=p,
        n=n,
        hidden_sizes=resolved_hidden_sizes,
        seed=seed,
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    actual_init_seed = seed + 1 if init_seed is None else init_seed
    x0 = make_mlp_initial_point(
        K=K,
        p=p,
        hidden_sizes=resolved_hidden_sizes,
        seed=actual_init_seed,
    )

    if verbose:
        print(
            f"  Shared problem: K={K}, p={p}, n={n}, "
            f"hidden_sizes={resolved_hidden_sizes}, backend=torch, "
            f"d={d}, "
            f"max_grad_evals={max_grad_evals}, init_seed={actual_init_seed}"
        )

    baseline_result = uniform_discretisation(
        K=K,
        objectives=objectives,
        grad_objectives=gradients,
        L=L,
        x0=x0,
        resolution=coarse_resolution,
        n_passes=n_passes,
        steps_per_point_per_pass=steps_per_point_per_pass,
        eval_every_n_grads=baseline_eval_every_n_grads,
        max_grad_evals=max_grad_evals,
        evaluate_coverage=True,
        joint_oracle=joint_oracle,
        verbose=verbose,
    )

    ipopt_result = algorithm_adaptive(
        K=K,
        d=d,
        objectives=objectives,
        grad_objectives=gradients,
        L=L,
        x0=x0,
        max_outer=max_outer,
        max_inner=max_inner,
        eval_every_n_grads=adaptive_eval_every_n_grads,
        target_cov=None,
        lambda_solver="ipopt",
        require_ipopt=True,
        lambda_max_starts=lambda_max_starts,
        prune_inner=prune_inner,
        max_grad_evals=max_grad_evals,
        joint_oracle=joint_oracle,
        verbose=verbose,
    )

    detector_options = {
        "window": plateau_window,
        "relative_improvement_tol": plateau_relative_improvement_tol,
        "consecutive_windows": plateau_consecutive_windows,
    }
    plateaus = {
        "baseline": _plateau_for_result(baseline_result, **detector_options),
        "ipopt_adaptive": _plateau_for_result(ipopt_result, **detector_options),
    }

    output_root = Path(output_dir)
    problem_title = (
        "MLP plateau comparison "
        f"(K={K}, p={p}, n={n}, hidden_sizes={resolved_hidden_sizes}, "
        "backend=torch)"
    )
    gradient_plot = _plot_plateau_pair(
        baseline_result,
        plateaus["baseline"],
        "Baseline",
        _BL_KW,
        ipopt_result,
        plateaus["ipopt_adaptive"],
        "IPOPT Adaptive",
        _IPOPT_KW,
        x_history_key="grad_evals_history",
        x_label="total gradient evaluations",
        title=f"{problem_title}: Baseline vs IPOPT Adaptive",
        out_path=str(
            output_root / "baseline_vs_ipopt_gradient_evaluations.png"
        ),
    )
    cpu_plot = _plot_plateau_pair(
        baseline_result,
        plateaus["baseline"],
        "Baseline",
        _BL_KW,
        ipopt_result,
        plateaus["ipopt_adaptive"],
        "IPOPT Adaptive",
        _IPOPT_KW,
        x_history_key="cpu_times",
        x_label="CPU time (s)",
        title=f"{problem_title}: Baseline vs IPOPT Adaptive (CPU time)",
        out_path=str(output_root / "baseline_vs_ipopt_cpu_time.png"),
    )
    plots = {
        # Preserve the old result key for code that already reads it.
        "baseline_vs_ipopt": gradient_plot,
        "baseline_vs_ipopt_gradient_evaluations": gradient_plot,
        "baseline_vs_ipopt_cpu_time": cpu_plot,
    }

    def _ratio(numerator: Dict, denominator: Dict) -> Optional[float]:
        if not numerator["found"] or not denominator["found"]:
            return None
        denominator_level = denominator["plateau_level"]
        if denominator_level <= 0.0:
            return None
        return float(numerator["plateau_level"] / denominator_level)

    plateau_ratios = {
        "baseline_to_ipopt": _ratio(
            plateaus["baseline"], plateaus["ipopt_adaptive"]
        ),
    }

    print("\nPlateau summary")
    print("  Method            Found   Onset gradients   Onset CPU   Plateau GN*")
    for key, label in (
        ("baseline", "Baseline"),
        ("ipopt_adaptive", "IPOPT Adaptive"),
    ):
        plateau = plateaus[key]
        if plateau["found"]:
            print(
                f"  {label:<17} yes     "
                f"{plateau['onset_grad_evals']:>15,d}   "
                f"{plateau['onset_cpu_time']:>8.2f}s   "
                f"{plateau['plateau_level']:.4e}"
            )
        else:
            print(f"  {label:<17} no      {'-':>15}   {'-':>9}   -")

    return {
        "baseline": baseline_result,
        "ipopt_adaptive": ipopt_result,
        "plateaus": plateaus,
        "plateau_ratios": plateau_ratios,
        "plots": plots,
        "detector_options": detector_options,
        "max_grad_evals": max_grad_evals,
        "init_seed": actual_init_seed,
        "x0": x0.copy(),
        "hidden_sizes": resolved_hidden_sizes,
        "mlp_backend": "torch",
        "d": d,
    }


if __name__ == "__main__":
    experiment_mlp_gn_coverage()
