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
  Under PL-condition, the worst-case suboptimality  sup_{λ ∈ G_fine} [F_λ(x̂(λ)) − F*_λ]
  is the right "apples-to-apples" metric across methods.
  Under generic non-convexity, worst-case suboptimality  sup_{λ ∈ G_fine} ||∇F_λ(x̂(λ))||^2
  is the right "apples-to-apples" metric across methods.
"""

from __future__ import annotations
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from bundle import Bundle, T_map


# =====================================================================
#  Nesterov's Accelerated Gradient (NAG) inner loop
# =====================================================================
def _nag_inner_loop_sc(
    x: np.ndarray,
    y: np.ndarray,
    n_iters: int,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    L_lam: float,
    mu_lam: float,
    grad_tol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    r"""Strongly-convex Nesterov accelerated gradient on a fixed
    scalarisation ``F_λ(·)``.

    Update rule (Nesterov, 1983; *Introductory Lectures on Convex
    Optimization*, Thm 2.2.2):

        x_{k+1}  =  y_k  −  (1/L_λ)  ∇F_λ(y_k)
        y_{k+1}  =  x_{k+1}  +  β · (x_{k+1} − x_k)

    with the constant momentum coefficient

        β  =  (√L_λ − √µ_λ) / (√L_λ + √µ_λ)

    Convergence rate
    ----------------
    For L-smooth and µ-strongly-convex F_λ,

        F_λ(x_k) − F*_λ  ≤  (1 − √(µ_λ/L_λ))^k  ·  (F_λ(x_0) + (µ_λ/2)‖x_0 − x*‖² − F*_λ),

    i.e. a **√κ_λ-iteration speedup** over vanilla gradient descent's
    (1 − µ_λ/L_λ)^k contraction.  For κ_λ = 100 (a typical conditioning
    for regularised logistic regression), NAG reaches a given target
    accuracy in ~10× fewer iterations than GD.

    State
    -----
    NAG carries two iterates ``(x, y)``:
      * ``x`` is the *primary* iterate, returned as the solution and used
        when reading off F_λ(x).
      * ``y`` is the *look-ahead* iterate, where the gradient is evaluated.

    Pass ``y = x`` to cold-start.  Each call performs at most ``n_iters``
    iterations and terminates early if ``‖∇F_λ(y_k)‖ < grad_tol``.

    Parameters
    ----------
    x, y         : current NAG state.  Cold-start with ``y = x``.
    n_iters      : maximum number of NAG steps.
    grad_fn      : callable mapping a point to ∇F_λ(point).
    L_lam        : smoothness constant L_λ = λ^T L.
    mu_lam       : strong-convexity constant µ_λ = λ^T µ  (must be > 0).
    grad_tol     : optional gradient-norm tolerance for early stopping.

    Returns
    -------
    (x_new, y_new, iters_used)
    """
    if mu_lam <= 0.0:
        raise ValueError(f"_nag_inner_loop_sc requires mu_lam > 0, got {mu_lam}.")
    sqrt_L = np.sqrt(L_lam)
    sqrt_mu = np.sqrt(mu_lam)
    beta = (sqrt_L - sqrt_mu) / (sqrt_L + sqrt_mu)
    inv_L = 1.0 / L_lam
    iters_used = 0

    for k in range(n_iters):
        g = grad_fn(y)
        if grad_tol is not None and np.linalg.norm(g) < grad_tol:
            break
        x_new = y - inv_L * g
        y = x_new + beta * (x_new - x)
        x = x_new
        iters_used = k + 1

    return x, y, iters_used


def compute_reference_map(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    fine_resolution: int = 80,
    n_iters: int = 20_000,
    grad_tol: float = 1e-10,
    mu: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict:
    """Precompute the reference optimal-value map on a fine grid.

    For each λ in G_fine, run an inner solver on F_λ(·) to very high
    accuracy (either ``n_iters`` iterations or until the gradient norm
    falls below ``grad_tol``).  Warm-starting from the previous fine-grid
    point's solution dramatically accelerates convergence for
    strongly-convex problems.

    Solver
    ------
    * **`mu` is given** (strongly-convex problems, e.g. regularised
      logistic regression):  Nesterov's accelerated gradient with the
      strongly-convex constant-momentum coefficient
          β = (√L_λ − √µ_λ) / (√L_λ + √µ_λ).
      This achieves a √κ_λ-iteration speedup over vanilla GD —
      ~10× fewer iterations for a typical κ_λ = 100.

    * **`mu` is None** (the generic / non-convex case): vanilla
      gradient descent, the original reference solver.  NAG without a
      provable strong-convexity constant can amplify the
      over-stepping that already plagues GD when the estimated
      Lipschitz constant under-approximates the local curvature
      (e.g. the ReLU MLP), so we stay with the conservative choice.

    Returns
    -------
    dict with keys "fine_grid", "F_star", "x_star".
    """
    grid = _uniform_simplex_grid(K, fine_resolution)
    grid = _sort_grid_for_warmstart(grid)
    N_fine = grid.shape[0]

    F_star = np.zeros(N_fine)
    x_star = np.zeros((N_fine, d))
    x_prev = x0.copy()

    for g_idx, lam in enumerate(grid):
        Ll = float(lam @ L)
        # Build a closure that returns the scalarised gradient at any z.
        # Hoist lam out of the closure as a default arg to avoid a
        # Python-level free-variable capture cost in the hot inner loop.
        lam_local = lam
        def grad_fn(z, lam_=lam_local):
            return sum(lam_[k] * grad_objectives[k](z) for k in range(K))

        if mu is not None:
            mul = float(lam @ mu)
            # Cold-start NAG at this fine-grid point: y = x.
            x_init = x_prev.copy()
            x_solved, _, _ = _nag_inner_loop_sc(
                x=x_init, y=x_init.copy(), n_iters=n_iters,
                grad_fn=grad_fn, L_lam=Ll, mu_lam=mul, grad_tol=grad_tol)
        else:
            x_solved = x_prev.copy()
            for _ in range(n_iters):
                g_lam = grad_fn(x_solved)
                if np.linalg.norm(g_lam) < grad_tol:
                    break
                x_solved = x_solved - (1.0 / Ll) * g_lam

        F_star[g_idx] = sum(lam[k] * objectives[k](x_solved) for k in range(K))
        x_star[g_idx] = x_solved
        x_prev = x_solved

        if verbose and (g_idx % max(1, N_fine // 10) == 0 or g_idx == N_fine - 1):
            print(f"    reference: {g_idx + 1:4d}/{N_fine}  |  F*_λ = {F_star[g_idx]:.6e}")

    return {"fine_grid": grid, "F_star": F_star, "x_star": x_star}


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
    n_passes: int = 1,
    steps_per_point_per_pass: int = 20,
    eval_every_n_grads: Optional[int] = None,
    mu: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict:
    """Run the baseline in "progressive" mode, with periodic checkpoints.

    We construct a coarse grid G_r of Δ_K at resolution ``resolution``
    and walk through it in warm-start order (lex sort).  Each pass
    does ``steps_per_point_per_pass`` solver steps at every grid point.

    Solver
    ------
    * **`mu` is given** (strongly-convex problems, e.g. regularised
      logistic regression):  Nesterov's accelerated gradient with the
      strongly-convex constant-momentum coefficient
          β = (√L_λ − √µ_λ) / (√L_λ + √µ_λ).
      Convergence rate is (1 − √(µ_λ/L_λ))^k vs vanilla GD's
      (1 − µ_λ/L_λ)^k — a √κ_λ-iteration speedup for free.

      The look-ahead iterate ``y`` is stored **per grid point** and
      carried across passes.  This is the key acceleration choice:
      each grid point's iterate effectively runs continuous NAG across
      passes, accumulating the √κ_λ speedup instead of cold-restarting
      NAG every pass.  Pass boundaries become invisible to the solver —
      only the gradient evaluation counter advances.

      In pass 1, where each grid point's iterate is chain-warm-started
      from the *previous* grid point's solution (a different λ), NAG's
      momentum state is cold-started at the new grid point (``y = x``)
      because the momentum direction at a different λ is generally
      meaningless at the current λ.  From pass 2 onward each grid
      point owns its momentum state.

    * **`mu` is None** (the generic / non-convex case): vanilla
      gradient descent, the original solver.

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
    reference_map             : output of ``compute_reference_map(..)``.
    n_passes                  : total number of passes to run.
    steps_per_point_per_pass  : solver steps taken at each grid point per pass.
    eval_every_n_grads        : if set, checkpoint at the next pass boundary after every M gradient evals.
    mu                        : optional µ-strong-convexity constants (shape (K,)). When given, the strongly-convex
                                NAG update rule is used in place of vanilla GD.

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

    # Per-grid-point NAG look-ahead iterate ``y_states[g]``.  Cold-start at
    # the beginning of pass 1 (y = x) gives NAG the same initial condition
    # as vanilla GD; from pass 2 onward we carry it across passes so each
    # grid point's iterate effectively runs continuous NAG.
    if mu is not None:
        y_states = solutions.copy()
        mu_lams = coarse_grid @ mu       # shape (N,)
    else:
        y_states = None
        mu_lams = None

    cpu_times: List[float] = []
    worst_errs: List[float] = []
    total_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    total_iters = 0
    grad_evals_at_last_ckpt = 0
    t_start = time.time()

    # Accumulator for time spent inside worst_case_suboptimality_baseline
    # across all prior checkpoints — subtracted from the next checkpoint's
    # recorded wall time so that previous checkpoints' evaluation costs
    # don't leak into the iterative-work measurement.  See the matching
    # comment in algorithm.py:algorithm2_progressive.
    checkpoint_overhead = 0.0

    def _checkpoint(label: str) -> None:
        nonlocal checkpoint_overhead
        cpu_times.append(time.time() - t_start - checkpoint_overhead)
        ck_t0 = time.time()
        err = worst_case_suboptimality_baseline(coarse_grid, solutions, reference_map, objectives, K,)
        checkpoint_overhead += time.time() - ck_t0
        worst_errs.append(err)
        total_iters_history.append(total_iters)
        grad_evals_history.append(total_iters * K)
        if verbose:
            print(f"  Baseline {label} | t={cpu_times[-1]:.2f}s "
                  f"| iters={total_iters} | grad_evals={total_iters * K} "
                  f"| err={err:.3e}")

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
                y = x.copy() if mu is not None else None
            else:
                x = solutions[g_idx].copy()
                y = y_states[g_idx].copy() if mu is not None else None

            if mu is not None:
                # Strongly-convex NAG.
                # Hoist lam into the closure as a default arg to avoid
                # Python-level free-variable capture cost in the hot loop.
                lam_local = lam
                def grad_fn(z, lam_=lam_local):
                    return sum(lam_[k] * grad_objectives[k](z) for k in range(K))
                x_new, y_new, iters_used = _nag_inner_loop_sc(
                    x=x, y=y,
                    n_iters=steps_per_point_per_pass,
                    grad_fn=grad_fn, L_lam=Ll,
                    mu_lam=float(mu_lams[g_idx]),
                    grad_tol=1e-8,                   # always run full burst
                )
                total_iters += iters_used
                # Persist NAG state for next pass.
                y_states[g_idx] = y_new
                x = x_new
            else:
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
        "resolution": resolution,
    }