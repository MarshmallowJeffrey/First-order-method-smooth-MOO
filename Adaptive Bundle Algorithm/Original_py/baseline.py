"""
baseline.py  –  Uniform-discretisation baseline for smooth MOO
==============================================================

Evaluation protocol
-------------------
- The baseline uses a coarser resolution r and runs warm-started gradient descent across the coarse grid
  repeatedly.  The inner loop does NOT stop based on any per-point tolerance —
  instead, after every M total gradient-descent iterations
  (a "checkpoint"), we pause, evaluate the current worst-case error of
  the rounded solution map, record (CPU time, err), and resume.


- The final comparison plot is CPU time (x-axis) vs worst-case
  function-value suboptimality (y-axis).
"""

from __future__ import annotations
import time
from typing import Callable, Dict, List, Optional
import numpy as np

from bundle import (
    Bundle,
    prefer_fused_joint_oracle,
    validate_oracle_output,
    validate_problem_inputs,
)

# =====================================================================
#  Grid utilities
# =====================================================================
def _uniform_simplex_grid(K: int, resolution: int) -> np.ndarray:
    """Tile Δ_K at grid spacing 1/resolution.

    Returns an (N, K) array of grid points, where N = C(resolution + K − 1, K − 1).
    """
    if K == 1:
        return np.array([[1.0]])
    points: List[List[int]] = []

    def _recurse(remaining: int, depth: int, current: List[int]) -> None:
        if depth == K - 1:
            current.append(remaining)
            points.append(current[:])
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            _recurse(remaining - v, depth + 1, current)
            current.pop()

    _recurse(resolution, 0, [])
    return np.asarray(points, dtype=float) / resolution


def _sort_grid_for_warmstart(grid: np.ndarray) -> np.ndarray:
    """Lex sort: consecutive points are ℓ₁-close (≤ 2/resolution apart)."""
    order = np.lexsort(grid[:, ::-1].T)
    return grid[order]


def _nearest_coarse_index(lam: np.ndarray, coarse_grid: np.ndarray) -> int:
    """Return the index of the coarse-grid point nearest to ``lam`` in ℓ₁.

    Currently unused by repository code; retained as a utility for mapping an
    arbitrary weight to its nearest coarse-grid point.
    """
    dists = np.sum(np.abs(coarse_grid - lam[None, :]), axis=1)
    return int(np.argmin(dists))


def bundle_from_points(points: np.ndarray, K: int, d: int,
                       L: np.ndarray,
                       objectives: List[Callable],
                       grad_objectives: List[Callable],
                       joint_oracle: Optional[Callable] = None) -> Bundle:
    """Construct a temporary evaluation Bundle from baseline solution points.

    Evaluates all K objectives and gradients at every point, using the fused
    ``joint_oracle`` when provided.  This work is checkpoint measurement
    overhead rather than baseline optimisation work.
    """
    joint_oracle = prefer_fused_joint_oracle(joint_oracle)
    bundle = Bundle(K=K, d=d, L=np.asarray(L, dtype=float))
    points_arr = np.atleast_2d(np.asarray(points, dtype=float))
    for point in points_arr:
        bundle.add_point(
            point,
            objectives,
            grad_objectives,
            joint_oracle=joint_oracle,
        )
    return bundle


# =====================================================================
#  Progressive baseline:  GD along the coarse grid with checkpoints
# =====================================================================
def uniform_discretisation(
    K: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    resolution: int,
    n_passes: int = 1,
    steps_per_point_per_pass: int = 20,
    eval_every_n_grads: Optional[int] = None,
    max_grad_evals: Optional[int] = None,
    evaluate_coverage: bool = False,
    joint_oracle: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """Run the baseline in "progressive" mode, with periodic checkpoints.

    We construct a coarse grid G_r of Δ_K at resolution ``resolution``
    and walk through it in warm-start order (lex sort).  Each pass
    does ``steps_per_point_per_pass`` solver steps at every grid point.

    Checkpoint cadence
    ------------------
    By default (``eval_every_n_grads=None``) we checkpoint after every
    pass, matching the previous behaviour.  Setting
    ``eval_every_n_grads = M`` instead causes a checkpoint at the next
    pass-boundary after every M cumulative gradient-oracle evaluations
    (where one scalarised solver step costs K gradient oracle calls).

    One "pass" = one full sweep across all grid points with M_pp steps
    per point = |G_r| · M_pp scalarised iterations = |G_r| · M_pp · K
    gradient-oracle evaluations.

    Parameters
    ----------
    resolution                : coarse grid resolution  r.
    n_passes                  : total number of passes to run.
    steps_per_point_per_pass  : solver steps taken at each grid point per pass.
    eval_every_n_grads        : if set, checkpoint at the next pass boundary after every M gradient evals.
    max_grad_evals            : optional hard budget on cumulative gradient evaluations.

    Returns
    -------
    dict with keys:
        "coarse_grid"             : (N, K) array of grid points.
        "final_solutions"         : (N, d) array of final solutions.
        "cpu_times"               : list of CPU times at each checkpoint (s).
        "total_iters_history"     : cumulative scalarised iters per ckpt.
        "grad_evals_history"      : cumulative gradient-oracle evals per ckpt (= total_iters * K).
        "resolution"              : grid resolution used.
    """
    x0_arr = np.asarray(x0, dtype=float)
    if x0_arr.ndim != 1 or x0_arr.size == 0:
        raise ValueError(f"x0 must be a nonempty one-dimensional array; got {x0_arr.shape}.")
    d = int(x0_arr.size)
    L_arr, x0_arr = validate_problem_inputs(
        K, d, L, x0_arr, objectives, grad_objectives
    )
    if max_grad_evals is not None and max_grad_evals < K:
        raise ValueError(
            f"max_grad_evals must be at least K={K}; got {max_grad_evals}."
        )

    joint_oracle = prefer_fused_joint_oracle(joint_oracle)
    coarse_grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, resolution))
    N = coarse_grid.shape[0]

    # Initialise all grid-point solutions to x0.
    solutions = np.tile(x0_arr, (N, 1))


    cpu_times: List[float] = []
    cov_history: List[float] = []
    total_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    total_iters = 0
    grad_evals_at_last_ckpt = 0
    metric_prev_lam: Optional[np.ndarray] = None
    t_start = time.time()

    # Accumulator for time spent computing the maximum of PC over the unit simplex
    # across all prior checkpoints — subtracted from the next checkpoint's
    # recorded wall time so that previous checkpoints' evaluation costs
    # don't leak into the iterative-work measurement.  See the matching
    # comment in algorithm.py:algorithm_adaptive.
    checkpoint_overhead = 0.0
    def _checkpoint(label: str) -> None:
        nonlocal checkpoint_overhead, metric_prev_lam
        cpu_times.append(time.time() - t_start - checkpoint_overhead)
        ck_t0 = time.time()
        # Reference-map-free bundle-coverage metric (the note's GN*):
        # the uniform method's "bundle" is the set of current last-iterate
        # points (one per grid node).  Build it and score it with the same
        # pc_star maximiser the adaptive method uses.  Assembling the bundle
        # and the max-over-simplex solve are measurement overhead, excluded
        # from the recorded cpu / grad-eval axes.
        if evaluate_coverage:
            from algorithm import pc_star
            cov_bundle = bundle_from_points(
                solutions, K, solutions.shape[1], L_arr,
                objectives, grad_objectives, joint_oracle=joint_oracle)
            cov, cov_lam = pc_star(cov_bundle, prev_lam=metric_prev_lam)
            metric_prev_lam = cov_lam
            cov_history.append(cov)
        checkpoint_overhead += time.time() - ck_t0
        total_iters_history.append(total_iters)
        grad_evals_history.append(total_iters * K)
        if verbose:
            cov_str = f" | worst-case pc={cov_history[-1]:.4e}" if evaluate_coverage else ""
            print(f"  Baseline {label} | t={cpu_times[-1]:.2f}s "
                  f"| iters={total_iters} | grad_evals={total_iters * K}"
                  f"{cov_str}")

    # Checkpoint 0:  all solutions = x0.
    _checkpoint(f"pass 0/{n_passes}")
    # Grid construction, array allocation, and the initial coverage metric are
    # preprocessing rather than iterative algorithm work.  Set checkpoint 0 to
    # time zero and restart the clock before the first pass.
    cpu_times[0] = 0.0
    t_start = time.time()
    checkpoint_overhead = 0.0

    for pass_idx in range(1, n_passes + 1):
        budget_exhausted = False
        # One pass cycles through the grid.  Pass 1 chain-warm-starts each
        # grid point from its predecessor; later passes resume each grid
        # point from its own stored solution.
        x_prev = solutions[0].copy()
        for g_idx in range(N):
            lam = coarse_grid[g_idx]
            Ll = float(lam @ L_arr)

            if pass_idx == 1:
                x = x_prev.copy()
            else:
                x = solutions[g_idx].copy()

            # Vanilla GD.
            for _ in range(steps_per_point_per_pass):
                if (
                    max_grad_evals is not None
                    and (total_iters + 1) * K > max_grad_evals
                ):
                    budget_exhausted = True
                    break
                if joint_oracle is not None:
                    _, all_grads = validate_oracle_output(
                        *joint_oracle(x), K, d
                    )
                    g_lam = lam @ all_grads
                else:
                    grad_rows = []
                    for k, grad_objective in enumerate(grad_objectives):
                        grad_k = np.asarray(grad_objective(x), dtype=float)
                        if grad_k.shape != (d,):
                            raise ValueError(
                                f"grad_objectives[{k}] must return shape ({d},); "
                                f"got {grad_k.shape}."
                            )
                        if np.any(~np.isfinite(grad_k)):
                            raise ValueError(
                                f"grad_objectives[{k}] returned non-finite values."
                            )
                        grad_rows.append(grad_k)
                    g_lam = lam @ np.vstack(grad_rows)
                x = x - (1.0 / Ll) * g_lam
                total_iters += 1
                if (
                    max_grad_evals is not None
                    and total_iters * K >= max_grad_evals
                ):
                    budget_exhausted = True
                    break

            solutions[g_idx] = x
            x_prev = x
            if budget_exhausted:
                break

        # Decide whether to checkpoint at this pass boundary.
        cur_grad_evals = total_iters * K
        do_ckpt = (
            eval_every_n_grads is None
            or (cur_grad_evals - grad_evals_at_last_ckpt) >= eval_every_n_grads
            or pass_idx == n_passes
            or budget_exhausted
        )
        if do_ckpt:
            _checkpoint(f"pass {pass_idx}/{n_passes}")
            grad_evals_at_last_ckpt = cur_grad_evals
        if budget_exhausted:
            break

    return {
        "coarse_grid": coarse_grid,
        "final_solutions": solutions,
        "cpu_times": cpu_times,
        "total_iters_history": total_iters_history,
        "grad_evals_history": grad_evals_history,
        "cov_history": cov_history,
        "resolution": resolution,
        "max_grad_evals": max_grad_evals,
    }
