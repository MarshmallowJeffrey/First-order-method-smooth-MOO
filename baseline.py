"""
baseline.py  –  Uniform-discretisation baseline for smooth MOO
==============================================================

Evaluation protocol
-------------------
We compare Algorithm 2 against the uniform-discretisation baseline on
a **CPU-time-vs-worst-case-accuracy** axis.

- A very fine uniform grid G_fine on Δ_K is treated as our benchmark
  for evaluating worst-case metric accuracy.  The reference optimal
  values F*_λ at each fine-grid point are computed offline once (by
  running gradient descent to very high accuracy) and stored.

- The baseline uses a coarser resolution r (r << resolution of G_fine)
  and runs warm-started gradient descent across the coarse grid
  repeatedly.  The inner loop does NOT stop based on any per-point
  tolerance — instead, after every M total gradient-descent iterations
  (a "checkpoint"), we pause, evaluate the current worst-case error of
  the rounded solution map, record (CPU time, err), and resume.

- Algorithm 2 is similarly checkpointed: after every outer iteration
  we evaluate the worst-case error of  T(·; B_t)  on G_fine and record
  (CPU time, err).

- The final comparison plot is CPU time (x-axis) vs worst-case
  function-value suboptimality (y-axis).
  Under PL-condition, the
  worst-case suboptimality  sup_{λ ∈ G_fine} [F_λ(x̂(λ)) − F*_λ]  is
  the right "apples-to-apples" metric across methods.
  Under generic non-convexity,
  worst-case suboptimality  sup_{λ ∈ G_fine} ||∇F_λ(x̂(λ))||^2  is
  the right "apples-to-apples" metric across methods.
"""

from __future__ import annotations
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from bundle import Bundle, T_map


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
#  Reference map:  compute F*_λ on a fine grid by high-accuracy GD
# =====================================================================
def compute_reference_map(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    fine_resolution: int = 40,
    n_iters: int = 20_000,
    grad_tol: float = 1e-10,
    verbose: bool = False,
) -> Dict:
    """Precompute the reference optimal-value map on a fine grid.

    For each λ in G_fine, run gradient descent on F_λ(·) to very high
    accuracy (either ``n_iters`` iterations or until the gradient norm
    falls below ``grad_tol``).  Warm-starting from the previous fine-
    grid point's solution dramatically accelerates convergence for
    strongly-convex problems.

    Returns a dict with keys "fine_grid", "F_star", "x_star".  This
    reference map is the "ground truth" used to evaluate worst-case
    suboptimality of any approximate solution map.  Compute it once per
    problem, re-use across methods and checkpoints.
    """
    grid = _uniform_simplex_grid(K, fine_resolution)
    grid = _sort_grid_for_warmstart(grid)
    N_fine = grid.shape[0]

    F_star = np.zeros(N_fine)
    x_star = np.zeros((N_fine, d))
    x_prev = x0.copy()

    for g, lam in enumerate(grid):
        x = x_prev.copy()
        Ll = float(lam @ L)
        for _ in range(n_iters):
            g_lam = sum(lam[k] * grad_objectives[k](x) for k in range(K))
            if np.linalg.norm(g_lam) < grad_tol:
                break
            x = x - (1.0 / Ll) * g_lam

        F_star[g] = sum(lam[k] * objectives[k](x) for k in range(K))
        x_star[g] = x
        x_prev = x

        if verbose and (g % max(1, N_fine // 10) == 0 or g == N_fine - 1):
            print(f"    reference: {g + 1:4d}/{N_fine}  |  F*_λ = {F_star[g]:.6e}")

    return {"fine_grid": grid, "F_star": F_star, "x_star": x_star}


# =====================================================================
#  Worst-case suboptimality evaluation
# =====================================================================
def _nearest_coarse_index(lam: np.ndarray, coarse_grid: np.ndarray) -> int:
    """Return g* = argmin_g ‖lam − coarse_grid[g]‖_1."""
    dists = np.sum(np.abs(coarse_grid - lam[None, :]), axis=1)
    return int(np.argmin(dists))


def worst_case_suboptimality_baseline(
    coarse_grid: np.ndarray,
    coarse_solutions: np.ndarray,
    reference_map: Dict,
    objectives: List[Callable],
    K: int,
) -> float:
    """Worst-case function-value suboptimality of the rounded baseline map.

        err := sup_{λ ∈ G_fine}  [F_λ(x̂_baseline(λ)) − F*_λ]

    where  x̂_baseline(λ) = coarse_solutions[argmin_g ‖λ − λ^(g)‖_1].
    """
    fine_grid = reference_map["fine_grid"]
    F_star = reference_map["F_star"]
    worst = -np.inf
    for i, lam in enumerate(fine_grid):
        g_star = _nearest_coarse_index(lam, coarse_grid)
        x_hat = coarse_solutions[g_star]
        F_lam = sum(lam[k] * objectives[k](x_hat) for k in range(K))
        err = F_lam - F_star[i]
        if err > worst:
            worst = err
    return float(worst)


def worst_case_suboptimality_algorithm2(
    bundle: Bundle,
    reference_map: Dict,
    objectives: List[Callable],
    K: int,
) -> float:
    """Worst-case function-value suboptimality of Algorithm 2's T(·; B) map.

    Uses  T(λ; B_t)  as the approximate solution at each fine-grid λ.
    """
    fine_grid = reference_map["fine_grid"]
    F_star = reference_map["F_star"]
    worst = -np.inf
    for i, lam in enumerate(fine_grid):
        x_hat = T_map(bundle, lam)
        F_lam = sum(lam[k] * objectives[k](x_hat) for k in range(K))
        err = F_lam - F_star[i]
        if err > worst:
            worst = err
    return float(worst)


# =====================================================================
#  Progressive baseline:  GD along the coarse grid with checkpoints
# =====================================================================
def uniform_discretisation_progressive(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    resolution: int,
    reference_map: Dict,
    n_passes: int = 20,
    steps_per_point_per_pass: int = 1,
    verbose: bool = False,
) -> Dict:
    """Run the baseline in "progressive" mode, with per-pass checkpoints.

    We construct a coarse grid G_r of Δ_K at resolution ``resolution``
    and walk through it in warm-start order (lex sort).  Each pass
    does ``steps_per_point_per_pass`` gradient-descent steps at every
    grid point.  After each pass, we evaluate the current worst-case
    suboptimality of the rounded solution map against ``reference_map``
    and record (CPU time, err).

    One "pass" = one full sweep across all grid points with M_pp steps
    per point = |G_r| · M_pp total gradient-descent iterations.

    Parameters
    ----------
    resolution                : coarse grid resolution  r.
    reference_map             : output of ``compute_reference_map(..)``.
    n_passes                  : total number of passes to run.
    steps_per_point_per_pass  : GD steps taken at each grid point per pass.

    Returns
    -------
    dict with keys:
        "coarse_grid"             : (N, K) array of grid points.
        "final_solutions"         : (N, d) array of final solutions.
        "cpu_times"               : list of CPU times at each checkpoint (s).
        "worst_errs"              : list of worst-case suboptimality values.
        "total_iters_history"     : list of cumulative GD iters per checkpoint.
        "resolution"              : grid resolution used.
    """
    coarse_grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, resolution))
    N = coarse_grid.shape[0]

    # Initialise all grid-point solutions to x0.
    solutions = np.tile(x0, (N, 1)).astype(float)

    cpu_times: List[float] = []
    worst_errs: List[float] = []
    total_iters_history: List[int] = []
    total_iters = 0
    t_start = time.time()

    # Checkpoint 0:  all solutions = x0.
    err0 = worst_case_suboptimality_baseline(
        coarse_grid, solutions, reference_map, objectives, K,
    )
    cpu_times.append(time.time() - t_start)
    worst_errs.append(err0)
    total_iters_history.append(0)
    if verbose:
        print(f"  Baseline pass 0/{n_passes} | t={cpu_times[-1]:.2f}s "
              f"| iters={total_iters} | err={err0:.4e}")

    for pass_idx in range(1, n_passes + 1):
        # One pass:  cycle through the grid.  On pass 1, chain warm-starts
        # from one grid point to the next.  On later passes, each grid
        # point continues from its own current solution.
        x_prev = solutions[0].copy()
        for g in range(N):
            lam = coarse_grid[g]
            Ll = float(lam @ L)
            if pass_idx == 1:
                x = x_prev.copy()  # warm start from previous grid's solution
            else:
                x = solutions[g].copy()  # continue from own current solution

            for _ in range(steps_per_point_per_pass):
                g_lam = sum(lam[k] * grad_objectives[k](x) for k in range(K))
                x = x - (1.0 / Ll) * g_lam
                total_iters += 1

            solutions[g] = x
            x_prev = x

        err_now = worst_case_suboptimality_baseline(
            coarse_grid, solutions, reference_map, objectives, K,
        )
        cpu_times.append(time.time() - t_start)
        worst_errs.append(err_now)
        total_iters_history.append(total_iters)
        if verbose:
            print(f"  Baseline pass {pass_idx}/{n_passes} | "
                  f"t={cpu_times[-1]:.2f}s | iters={total_iters} "
                  f"| err={err_now:.4e}")

    return {
        "coarse_grid": coarse_grid,
        "final_solutions": solutions,
        "cpu_times": cpu_times,
        "worst_errs": worst_errs,
        "total_iters_history": total_iters_history,
        "resolution": resolution,
    }


# =====================================================================
#  Instrumented Algorithm 2:  checkpoint after each outer iteration
# =====================================================================
def algorithm2_progressive(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    reference_map: Dict,
    mu: Optional[np.ndarray] = None,
    mode: str = "gap",
    max_outer: int = 50,
    max_inner: int = 400,
    verbose: bool = False,
) -> Dict:
    """Run Algorithm 2 with a checkpoint after every outer iteration.

    Thin wrapper around the algorithm.py primitives that interleaves
    worst-case-error evaluations with the main outer loop.

    Returns
    -------
    dict with keys:
        "bundle"                  : final Bundle.
        "cpu_times"               : list of CPU times at each checkpoint.
        "worst_errs"              : list of worst-case suboptimality values.
        "outer_iters_history"     : list of outer-iter counts per checkpoint.
        "pc_history"              : PC*_t values across outer iterations.
    """
    from algorithm import (
        _maximise_UB, _maximise_GAP, _maximise_GN,
        _bundle_update_adaptive,
    )
    from bundle import UB, GAP, GN

    if mode == "gap":
        # Use LB_2 (single-index minorant) inside both the λ-maximisation
        # and the inner-loop PC check.  LB_2 is ~100× faster than LB_1 and
        # doesn't hit the Gurobi size-limited license when the bundle grows
        # past ~100 points.  It gives a looser lower bound (hence a larger
        # GAP, i.e. a more conservative progress criterion), but that's
        # acceptable here because we're benchmarking raw "accuracy per CPU
        # second" rather than proving a per-iteration decrease rate.
        pc_fn = lambda bundle, lam: GAP(bundle, lam, variant="lb2")
        maximise_pc = lambda bundle: _maximise_GAP(bundle, variant="lb2")
        if mu is None:
            raise ValueError("mode='gap' requires mu (strong convexity).")
    elif mode == "ub":
        pc_fn, maximise_pc = UB, _maximise_UB
    elif mode == "gn":
        pc_fn, maximise_pc = GN, _maximise_GN
        if mu is None:
            raise ValueError("mode='gn' requires mu.")
    else:
        raise ValueError(f"Unknown mode: {mode!r}.")

    bundle = Bundle(K=K, d=d, L=L, mu=mu)
    bundle.add_point(x0.copy(), objectives, grad_objectives)

    cpu_times: List[float] = []
    worst_errs: List[float] = []
    outer_iters_history: List[int] = []
    pc_history: List[float] = []

    t_start = time.time()

    # Checkpoint 0:  bundle has one point (the initial iterate).
    err0 = worst_case_suboptimality_algorithm2(bundle, reference_map, objectives, K)
    cpu_times.append(time.time() - t_start)
    worst_errs.append(err0)
    outer_iters_history.append(0)
    if verbose:
        print(f"  A2 outer 0/{max_outer} | t={cpu_times[-1]:.2f}s | err={err0:.4e}")

    # The inner loop uses eps/3 as its stopping rule.  Since this wrapper
    # doesn't take an outer ε, set it very small so inner loops run to
    # the max_inner cap rather than stopping early.
    eps_dummy = 1e-12

    for t in range(max_outer):
        pc_star, best_lam = maximise_pc(bundle)
        pc_history.append(pc_star)

        steps = _bundle_update_adaptive(
            bundle, best_lam, pc_fn, eps_dummy,
            objectives, grad_objectives, max_inner,
        )

        err_now = worst_case_suboptimality_algorithm2(
            bundle, reference_map, objectives, K,
        )
        cpu_times.append(time.time() - t_start)
        worst_errs.append(err_now)
        outer_iters_history.append(t + 1)

        if verbose:
            print(f"  A2 outer {t + 1}/{max_outer} | t={cpu_times[-1]:.2f}s "
                  f"| PC*={pc_star:.4e} | inner={steps:3d} "
                  f"| bundle={bundle.m} | err={err_now:.4e}")

    return {
        "bundle": bundle,
        "cpu_times": cpu_times,
        "worst_errs": worst_errs,
        "outer_iters_history": outer_iters_history,
        "pc_history": pc_history,
    }
