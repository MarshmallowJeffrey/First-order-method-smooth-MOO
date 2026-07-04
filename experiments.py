r"""
The uniform-discretisation baseline performance plots
    non-convex   (MLP)    :  GN*(B)  = sup_{lambda in Delta_K} min_{x_i\in B_m} ||grad F_lambda(x_i)||^2
"""

import time
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from objectives import make_mlp_nonconvex
from baseline import uniform_discretisation
from algorithm import algorithm_adaptive
from chebyshev import algorithm_chebyshev_adaptive

# plot conventions (consistent with the rest of the project)
_BL_KW = dict(color="#d62728", marker="s", ms=5, lw=1.6, label="uniform discretisation")
_A2_KW = dict(color="#1f77b4", marker="o", ms=4, lw=1.8, label="adaptive bundle")
_CHEB_KW = dict(color="#ff7f0e", marker="^", ms=5, lw=1.8)


def _mlp_initial_point(
    K: int,
    p: int,
    h: int,
    seed: int,
    initialization: str,
) -> np.ndarray:
    """Create one reproducible initial point shared by all methods."""
    d = h * p + h + K * h + K
    if initialization == "zero":
        return np.zeros(d)
    if initialization != "xavier":
        raise ValueError("initialization must be 'zero' or 'xavier'.")

    rng = np.random.default_rng(seed)
    w1_bound = np.sqrt(6.0 / (p + h))
    w2_bound = np.sqrt(6.0 / (h + K))
    W1 = rng.uniform(-w1_bound, w1_bound, size=h * p)
    b1 = np.zeros(h)
    W2 = rng.uniform(-w2_bound, w2_bound, size=K * h)
    b2 = np.zeros(K)
    return np.concatenate([W1, b1, W2, b2])


def _plot_coverage(
    bl: Optional[Dict],
    a2: Optional[Dict],
    mode: str,
    title: str,
    out_path: str,
    cheb: Optional[Dict] = None,
    reference_error: Optional[float] = None,
) -> str:
    """Two-panel plot: coverage metric vs CPU time, and vs gradient evals."""
    ylabel = (r"$\sup_{\lambda\in\Delta_K} [\min_{x_i\in\mathcal{B}_m} \|\nabla F_\lambda(x_i)\|^2]$"
              if mode == "gn" else
              r"$\sup_{\lambda\in\Delta_K}$ GAP$(\lambda; \mathcal{B}_m)$")
    fig, (ax_t, ax_g) = plt.subplots(1, 2, figsize=(13, 5.4))

    def marker_stride(values) -> int:
        return max(1, len(values) // 12)

    if bl is not None:
        # Per-call baseline style: append the coarse-grid resolution r to the
        # legend label so the discretisation density is visible on the plot.
        bl_kw = {**_BL_KW, "label": f"uniform discretisation (r={bl['resolution']})"}
        stride = marker_stride(bl["cov_history"])
        ax_t.plot(
            bl["cpu_times"],
            bl["cov_history"],
            markevery=stride,
            **bl_kw,
        )
        ax_g.plot(
            bl["grad_evals_history"],
            bl["cov_history"],
            markevery=stride,
            **bl_kw,
        )

        # Final worst-case error reached by the uniform baseline, drawn as a
        # green horizontal reference line on both panels.
        final_err = (
            bl["cov_history"][-1]
            if reference_error is None
            else reference_error
        )
        final_kw = dict(color="#2ca02c", ls="--", lw=1.4,
                        label=f"target error = {final_err:.3e}")
        ax_t.axhline(final_err, **final_kw)
        ax_g.axhline(final_err, **final_kw)

    if a2 is not None:
        adaptive_kw = {
            **_A2_KW,
            "label": f"adaptive bundle (inner={a2['max_inner']})",
        }
        stride = marker_stride(a2["cov_history"])
        ax_t.plot(
            a2["cpu_times"],
            a2["cov_history"],
            markevery=stride,
            **adaptive_kw,
        )
        ax_g.plot(
            a2["grad_evals_history"],
            a2["cov_history"],
            markevery=stride,
            **adaptive_kw,
        )

    if cheb is not None:
        policy_label = (
            "exact"
            if cheb["direction_policy"] == "chebyshev_only"
            else "GN-aligned hybrid"
        )
        cheb_kw = {
            **_CHEB_KW,
            "label": (
                f"weighted Chebyshev ({policy_label}, "
                f"{cheb['optimizer']}, step={cheb['step_scale']:g})"
            ),
        }
        stride = marker_stride(cheb["cov_history"])
        ax_t.plot(
            cheb["cpu_times"],
            cheb["cov_history"],
            markevery=stride,
            **cheb_kw,
        )
        ax_g.plot(
            cheb["grad_evals_history"],
            cheb["cov_history"],
            markevery=stride,
            **cheb_kw,
        )

    ax_t.set_xlabel("CPU time (s)")
    ax_t.set_ylabel(ylabel)
    ax_t.set_xscale("symlog", linthresh=1.0)
    ax_t.set_xticks([0, 1, 10])
    ax_t.set_xticklabels(["0", "1", "10"])
    ax_t.set_yscale("log")
    ax_t.set_title("worst-case squared gradient norm vs CPU time" if mode == "gn" else "worst-case function suboptimality vs CPU time")
    ax_t.grid(True, which="both", alpha=0.18)

    ax_g.set_xlabel("total gradient evaluations")
    ax_g.set_ylabel(ylabel)
    ax_g.set_xscale("symlog", linthresh=1_000)
    ax_g.set_xticks([0, 1_000, 10_000, 100_000, 1_000_000])
    ax_g.set_xticklabels(["0", "1e3", "1e4", "1e5", "1e6"])
    ax_g.set_yscale("log")
    ax_g.set_title("worst-case squared gradient norm vs gradient evals" if mode == "gn" else "worst-case function suboptimality vs gradient evals")
    ax_g.grid(True, which="both", alpha=0.18)

    fig.suptitle(title, fontsize=12)
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def experiment_mlp_gn_coverage(
    verbose: bool = True,
    K: int = 3, p: int = 10, n: int = 20, h: int = 8, seed: int = 10,
    mlp_activation: str = "relu",
    smoothness_probes: int = 40,
    smoothness_safety_factor: float = 2.0,
    initialization: str = "zero",
    initialization_seed: Optional[int] = None,
    coarse_resolution: int = 26,
    n_passes: int = 15, steps_per_point_per_pass: int = 50,
    eval_every_n_grads: int = 5000, checkpoint_every: int = 3,
    max_outer: int = 1000, max_inner: Optional[int] = None,
    lambda_max_starts: int = 256,
    lambda_selector: str = "optimize",
    lambda_random_starts: int = 512,
    target_cov: Optional[float] = None,
    run_baseline: bool = True, run_adaptive: bool = True,
    run_chebyshev: bool = False,
    chebyshev_max_outer: Optional[int] = None,
    chebyshev_max_inner: Optional[int] = None,
    chebyshev_delta: float = 1e-8,
    chebyshev_direction_policy: str = "chebyshev_only",
    chebyshev_gn_tolerance: Optional[float] = None,
    chebyshev_alignment_threshold: float = 0.1,
    chebyshev_step_scale: float = 1.0,
    chebyshev_optimizer: str = "gd",
    out_path: str = "mlp.png",
) -> Dict:
    """Non-convex MLP: compare uniform, adaptive bundle, and Chebyshev."""
    print("=" * 68)
    print("Coverage experiment — MLP (non-convex), metric = GN*")
    print("=" * 68)
    d = h * p + h + K * h + K
    if max_inner is None:
        max_inner = 50 if K == 4 else 25
    if chebyshev_max_outer is None:
        chebyshev_max_outer = max_outer
    if chebyshev_max_inner is None:
        chebyshev_max_inner = max_inner
    objs, grads, L, joint = make_mlp_nonconvex(
        K=K,
        p=p,
        n=n,
        h=h,
        seed=seed,
        activation=mlp_activation,
        smoothness_probes=smoothness_probes,
        smoothness_safety_factor=smoothness_safety_factor,
    )
    init_seed = seed if initialization_seed is None else initialization_seed
    x0 = _mlp_initial_point(K, p, h, init_seed, initialization)
    print(
        f"  K={K}, p={p}, n={n}, h={h}, d={d}  |  "
        f"L={np.round(L,3)} | init={initialization} | "
        f"activation={mlp_activation}"
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
            coverage_mode="gn",
            target_cov=target_cov,
            joint_oracle=joint,
            verbose=verbose)

    a2 = None
    if run_adaptive:
        if verbose:
            print("\n  [adaptive bundle] ...")
        adaptive_target = (
            target_cov
            if target_cov is not None
            else bl["cov_history"][-1] if bl is not None else None
        )
        a2 = algorithm_adaptive(
            K=K, d=d, objectives=objs, grad_objectives=grads, L=L, x0=x0,
            mode="gn", max_outer=max_outer, max_inner=max_inner,
            eval_every_n_grads=eval_every_n_grads,
            target_cov=adaptive_target,
            lambda_max_starts=lambda_max_starts,
            lambda_selector=lambda_selector,
            lambda_random_starts=lambda_random_starts,
            joint_oracle=joint, verbose=verbose)

    cheb = None
    if run_chebyshev:
        if verbose:
            print("\n  [weighted Chebyshev] ...")
        chebyshev_target = (
            target_cov
            if target_cov is not None
            else bl["cov_history"][-1] if bl is not None else None
        )
        cheb = algorithm_chebyshev_adaptive(
            K=K,
            d=d,
            objectives=objs,
            grad_objectives=grads,
            L=L,
            x0=x0,
            max_outer=chebyshev_max_outer,
            max_inner=chebyshev_max_inner,
            delta=chebyshev_delta,
            direction_policy=chebyshev_direction_policy,
            gn_tolerance=chebyshev_gn_tolerance,
            alignment_threshold=chebyshev_alignment_threshold,
            eval_every_n_grads=eval_every_n_grads,
            target_cov=chebyshev_target,
            lambda_max_starts=lambda_max_starts,
            lambda_selector=lambda_selector,
            lambda_random_starts=lambda_random_starts,
            step_scale=chebyshev_step_scale,
            optimizer=chebyshev_optimizer,
            joint_oracle=joint,
            verbose=verbose,
        )

    path = _plot_coverage(
        bl, a2, mode="gn",
        title=(
            "MLP: K={}, p={}, n={}, h={}, d={} ({}, {})"
            .format(K, p, n, h, d, initialization, mlp_activation)
        ),
        out_path=out_path,
        cheb=cheb,
        reference_error=target_cov)
    if bl is not None:
        print(f"\n  BL  final GN* = {bl['cov_history'][-1]:.4e}  "
              f"(ge={bl['grad_evals_history'][-1]}, cpu={bl['cpu_times'][-1]:.2f}s)")
    if a2 is not None:
        print(f"  A2  final GN* = {a2['cov_history'][-1]:.4e}  "
              f"(ge={a2['grad_evals_history'][-1]}, cpu={a2['cpu_times'][-1]:.2f}s, "
              f"bundle={a2['bundle'].m})")
    if cheb is not None:
        print(f"  CH  final GN* = {cheb['cov_history'][-1]:.4e}  "
              f"(ge={cheb['grad_evals_history'][-1]}, cpu={cheb['cpu_times'][-1]:.2f}s, "
              f"bundle={cheb['bundle'].m}, "
              f"status={cheb['termination_reason']}, "
              f"cheb_steps={cheb['chebyshev_steps']}, "
              f"fallback_steps={cheb['fallback_steps']})")
    return {
        "baseline": bl,
        "algorithm2": a2,
        "chebyshev": cheb,
        "plot": path,
    }


if __name__ == "__main__":
    experiment_mlp_gn_coverage()
