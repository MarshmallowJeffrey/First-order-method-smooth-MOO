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
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

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


# =====================================================================
#  Worst-case suboptimality evaluation
# =====================================================================
def _nearest_coarse_index(lam: np.ndarray, coarse_grid: np.ndarray) -> int:
    """Return g* = argmin_g ‖lam − coarse_grid[g]‖_1."""
    dists = np.sum(np.abs(coarse_grid - lam[None, :]), axis=1)
    return int(np.argmin(dists))


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
    coverage_mode: Optional[str] = None,
    joint_oracle: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """Run the baseline in "progressive" mode, with periodic checkpoints.

    We construct a coarse grid G_r of Δ_K at resolution ``resolution``
    and walk through it in warm-start order (lex sort).  Each pass
    does ``steps_per_point_per_pass`` solver steps at every grid point.

    Solver
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

    Returns
    -------
    dict with keys:
        "coarse_grid"             : (N, K) array of grid points.
        "final_solutions"         : (N, d) array of final solutions.
        "cpu_times"               : list of CPU times at each checkpoint (s).
        "worst_errs"              : list of worst-case suboptimality values.
        "total_iters_history"     : cumulative scalarised iters per ckpt.
        "grad_evals_history"      : cumulative gradient-oracle evals per ckpt (= total_iters * K).
        "resolution"              : grid resolution used.
    """
    coarse_grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, resolution))
    N = coarse_grid.shape[0]

    # Initialise all grid-point solutions to x0.
    solutions = np.tile(x0, (N, 1)).astype(float)


    cpu_times: List[float] = []
    worst_errs: List[float] = []
    cov_history: List[float] = []
    total_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    total_iters = 0
    grad_evals_at_last_ckpt = 0
    t_start = time.time()

    # Accumulator for time spent computing the maximum of PC over the unit simplex
    # across all prior checkpoints — subtracted from the next checkpoint's
    # recorded wall time so that previous checkpoints' evaluation costs
    # don't leak into the iterative-work measurement.  See the matching
    # comment in algorithm.py:algorithm_adaptive.
    checkpoint_overhead = 0.0
    def _checkpoint(label: str) -> None:
        nonlocal checkpoint_overhead
        cpu_times.append(time.time() - t_start - checkpoint_overhead)
        ck_t0 = time.time()
        err = float("nan")
        # Reference-map-free bundle-coverage metric (the note's GN*):
        # the uniform method's "bundle" is the set of current last-iterate
        # points (one per grid node).  Build it and score it with the same
        # pc_star maximiser the adaptive method uses.  Assembling the bundle
        # and the max-over-simplex solve are measurement overhead, excluded
        # from the recorded cpu / grad-eval axes.
        if coverage_mode is not None:
            from algorithm import bundle_from_points, pc_star
            cov_bundle = bundle_from_points(
                solutions, K, solutions.shape[1], L,
                objectives, grad_objectives, joint_oracle=joint_oracle)
            cov, _ = pc_star(cov_bundle, coverage_mode)
            cov_history.append(cov)
        checkpoint_overhead += time.time() - ck_t0
        worst_errs.append(err)
        total_iters_history.append(total_iters)
        grad_evals_history.append(total_iters * K)
        if verbose:
            cov_str = f" | worst-case pc={cov_history[-1]:.4e}" if coverage_mode is not None else ""
            print(f"  Baseline {label} | t={cpu_times[-1]:.2f}s "
                  f"| iters={total_iters} | grad_evals={total_iters * K}"
                  f"{cov_str}")

    # Checkpoint 0:  all solutions = x0.
    _checkpoint(f"pass 0/{n_passes}")
    # Setup work above (grid enumeration, array allocation, computing
    # mu_lams, the worst_case_suboptimality_baseline evaluation inside
    # _checkpoint itself) is preprocessing — not iterative algorithm
    # work.  Force checkpoint 0 to read cpu = 0.0 and reset the clock so
    # subsequent checkpoints measure only iterative work.  This makes
    # the baseline's checkpoint 0 align with A2's checkpoint 0 on the
    # CPU axis (both say "no work done yet").
    cpu_times[0] = 0.0
    t_start = time.time()
    checkpoint_overhead = 0.0

    for pass_idx in range(1, n_passes + 1):
        # One pass:  cycle through the grid.  On pass 1, chain warm-starts
        # from one grid point to the next *for the iterate*, but cold-start
        # NAG's momentum state at the new grid point.
        x_prev = solutions[0].copy()
        for g_idx in range(N):
            lam = coarse_grid[g_idx]
            Ll = float(lam @ L)

            if pass_idx == 1:
                x = x_prev.copy()
            else:
                x = solutions[g_idx].copy()

            # Vanilla GD.
            for _ in range(steps_per_point_per_pass):
                g_lam = sum(lam[k] * grad_objectives[k](x) for k in range(K))
                x = x - (1.0 / Ll) * g_lam
                total_iters += 1

            solutions[g_idx] = x
            x_prev = x

        # Decide whether to checkpoint at this pass boundary.
        cur_grad_evals = total_iters * K
        do_ckpt = (
            eval_every_n_grads is None
            or (cur_grad_evals - grad_evals_at_last_ckpt) >= eval_every_n_grads
            or pass_idx == n_passes
        )
        if do_ckpt:
            _checkpoint(f"pass {pass_idx}/{n_passes}")
            grad_evals_at_last_ckpt = cur_grad_evals

    return {
        "coarse_grid": coarse_grid,
        "final_solutions": solutions,
        "cpu_times": cpu_times,
        "worst_errs": worst_errs,
        "total_iters_history": total_iters_history,
        "grad_evals_history": grad_evals_history,
        "cov_history": cov_history,
        "resolution": resolution,
    }