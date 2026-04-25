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
    eval_every_n_grads: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Run the baseline in "progressive" mode, with periodic checkpoints.

    We construct a coarse grid G_r of Δ_K at resolution ``resolution``
    and walk through it in warm-start order (lex sort).  Each pass
    does ``steps_per_point_per_pass`` gradient-descent steps at every
    grid point.

    Checkpoint cadence
    ------------------
    By default (``eval_every_n_grads=None``) we checkpoint after every
    pass, matching the previous behaviour.  Setting
    ``eval_every_n_grads = M`` instead causes a checkpoint at the next
    pass-boundary after every M cumulative gradient-oracle evaluations
    (where one scalarised GD step costs K gradient oracle calls).

    One "pass" = one full sweep across all grid points with M_pp steps
    per point = |G_r| · M_pp scalarised GD iterations = |G_r| · M_pp · K
    gradient-oracle evaluations.

    Parameters
    ----------
    resolution                : coarse grid resolution  r.
    reference_map             : output of ``compute_reference_map(..)``.
    n_passes                  : total number of passes to run.
    steps_per_point_per_pass  : GD steps taken at each grid point per pass.
    eval_every_n_grads        : if set, checkpoint at the next pass
                                boundary after every M gradient evals.

    Returns
    -------
    dict with keys:
        "coarse_grid"             : (N, K) array of grid points.
        "final_solutions"         : (N, d) array of final solutions.
        "cpu_times"               : list of CPU times at each checkpoint (s).
        "worst_errs"              : list of worst-case suboptimality values.
        "total_iters_history"     : cumulative scalarised-GD iters per ckpt.
        "grad_evals_history"      : cumulative gradient-oracle evals per ckpt
                                    (= total_iters * K).
        "resolution"              : grid resolution used.
    """
    coarse_grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, resolution))
    N = coarse_grid.shape[0]

    # Initialise all grid-point solutions to x0.
    solutions = np.tile(x0, (N, 1)).astype(float)

    cpu_times: List[float] = []
    worst_errs: List[float] = []
    total_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    total_iters = 0
    grad_evals_at_last_ckpt = 0
    t_start = time.time()

    def _checkpoint(label: str) -> None:
        err = worst_case_suboptimality_baseline(
            coarse_grid, solutions, reference_map, objectives, K,
        )
        cpu_times.append(time.time() - t_start)
        worst_errs.append(err)
        total_iters_history.append(total_iters)
        grad_evals_history.append(total_iters * K)
        if verbose:
            print(f"  Baseline {label} | t={cpu_times[-1]:.2f}s "
                  f"| iters={total_iters} | grad_evals={total_iters * K} "
                  f"| err={err:.4e}")

    # Checkpoint 0:  all solutions = x0.
    _checkpoint(f"pass 0/{n_passes}")

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
    checkpoint_every: int = 1,
    eval_every_n_grads: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Run Algorithm 2 with periodic worst-case-error checkpoints.

    Thin wrapper around the algorithm.py primitives that interleaves
    worst-case-error evaluations with the main outer loop.

    Checkpoint cadence
    ------------------
    Two complementary controls:
      - ``checkpoint_every``    : checkpoint every k outer iterations.
      - ``eval_every_n_grads``  : if set, additionally checkpoint at
                                  the next outer-iteration boundary
                                  after every M cumulative gradient
                                  evaluations.

    Setting ``eval_every_n_grads = M`` makes A2 directly comparable
    to the baseline's gradient-vs-error curve at matched M.

    Parameters
    ----------
    checkpoint_every     : evaluate worst-case error every k outer iters.
    eval_every_n_grads   : checkpoint after each M cumulative gradient evals.
    """
    from algorithm import (
        _maximise_UB, _maximise_GAP, _maximise_GN,
        _bundle_update_adaptive,
    )
    from bundle import UB, GAP, GN

    if mode == "gap":
        # Use LB_2 (single-index minorant) inside both the λ-maximisation
        # and the inner-loop PC check.  LB_2 is ~100× faster than LB_1 and
        # avoids hitting the Gurobi size-limited license once the bundle
        # grows past ~100 points.
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
    # The initial bundle point cost K gradient evals at x0.
    grad_evals = K

    cpu_times: List[float] = []
    worst_errs: List[float] = []
    outer_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    pc_history: List[float] = []
    grad_evals_at_last_ckpt = 0

    t_start = time.time()

    def _checkpoint(label: str, pc_star=None, steps=None) -> None:
        err = worst_case_suboptimality_algorithm2(bundle, reference_map, objectives, K)
        cpu_times.append(time.time() - t_start)
        worst_errs.append(err)
        outer_iters_history.append(label_to_outer(label))
        grad_evals_history.append(grad_evals)
        if verbose:
            extra = ""
            if pc_star is not None:
                extra = f" | PC*={pc_star:.4e} | inner={steps:3d} | bundle={bundle.m}"
            print(f"  A2 {label} | t={cpu_times[-1]:.2f}s | grad_evals={grad_evals}"
                  f"{extra} | err={err:.4e}")

    def label_to_outer(label: str) -> int:
        # Helper:  parse label like "outer 5/20" -> 5,  "outer 0/20" -> 0
        try:
            return int(label.split()[1].split("/")[0])
        except Exception:
            return -1

    # Checkpoint 0:  bundle has one point (the initial iterate).
    _checkpoint(f"outer 0/{max_outer}")

    # Inner loop uses eps/3 as its stopping rule; here we want it to run
    # to the max_inner cap (or to the new pruning rule), so set ε tiny.
    eps_dummy = 1e-12

    for t in range(max_outer):
        pc_star, best_lam = maximise_pc(bundle)
        pc_history.append(pc_star)

        bundle_m_before = bundle.m
        steps = _bundle_update_adaptive(
            bundle, best_lam, pc_fn, eps_dummy,
            objectives, grad_objectives, max_inner,
        )
        # Each inner step corresponds to a retained bundle point (the new
        # _bundle_update_adaptive prunes BEFORE evaluating gradients, so
        # every committed step costs K gradient evals and is kept).
        # The bundle-size delta should equal steps, but we use the delta
        # directly to be robust to any future changes to the inner loop.
        retained_steps = bundle.m - bundle_m_before
        grad_evals += retained_steps * K

        do_checkpoint = (
            ((t + 1) % checkpoint_every == 0)
            or (t + 1 == max_outer)
            or (
                eval_every_n_grads is not None
                and (grad_evals - grad_evals_at_last_ckpt) >= eval_every_n_grads
            )
        )
        if do_checkpoint:
            _checkpoint(f"outer {t + 1}/{max_outer}", pc_star=pc_star, steps=steps)
            grad_evals_at_last_ckpt = grad_evals
            if verbose and retained_steps < steps:
                print(f"        (attempted {steps}, retained {retained_steps} "
                      f"after PC-drop pruning)")
        elif verbose:
            print(f"  A2 outer {t + 1}/{max_outer} | grad_evals={grad_evals} "
                  f"| PC*={pc_star:.4e} | inner={steps:3d} (retained {retained_steps}) "
                  f"| bundle={bundle.m} | (no checkpoint)")

    return {
        "bundle": bundle,
        "cpu_times": cpu_times,
        "worst_errs": worst_errs,
        "outer_iters_history": outer_iters_history,
        "grad_evals_history": grad_evals_history,
        "pc_history": pc_history,
    }
