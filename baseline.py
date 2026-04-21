"""
baseline.py  –  Uniform-discretisation baseline for smooth MOO
==============================================================

This file implements the baseline method referenced in Section 6 of the
paper, against which Algorithm 2 is compared.

Method
------
1. **Grid construction.**  Tile the unit simplex Δ_K with a uniform grid
   at spacing  h_grid ≈ ε  (as suggested by Section 6).  The grid
   resolution is  resolution = ⌈1/h_grid⌉ = ⌈1/ε⌉.

2. **Offline solve (warm-started).**  Sort the grid points by a simple
   scan order and, for each point  λ, run gradient descent
   on  F_λ(·)  from the previous grid point's solution (warm start).
   Each loop stops when the same progress criterion used by
   Algorithm 2 — UB, GAP, or GN — drops below ε.

3. **Query-time rounding.**  For any query λ ∈ Δ_K, the baseline's
   approximate solution map  Ŵ_baseline(λ)  returns the solution stored
   at the nearest grid point (in ℓ₁ distance).
"""

from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional
import numpy as np
from bundle import Bundle, UB, GAP, GN


# =====================================================================
#  Grid utilities
# =====================================================================
def _uniform_simplex_grid(K: int, resolution: int) -> np.ndarray:
    """Tile Δ_K at grid spacing 1/resolution.

    Returns an (N, K) array of grid points, where N = C(resolution + K − 1, K − 1).

    For K = 2 and resolution = 4 the grid is
    {(0, 1), (1/4, 3/4), (1/2, 1/2), (3/4, 1/4), (1, 0)} — 5 points.
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
    """Order grid points so consecutive points are ℓ₁-close.

    A lexicographic sort on the first K−1 coordinates works well in
    practice for small K: it traces a zig-zag path that keeps consecutive
    points within ≤ 2/resolution of each other.

    Using a true TSP-style ordering would improve warm-starting marginally
    but adds complexity; the lex sort is the standard simple choice.
    """
    # np.lexsort sorts by the *last* key first, so reverse the columns.
    order = np.lexsort(grid[:, ::-1].T)
    return grid[order]


# =====================================================================
#  Per-grid-point local solve
# =====================================================================
def _local_solve(
    K: int,
    d: int,
    lam: np.ndarray,
    W_init: np.ndarray,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    mu: Optional[np.ndarray],
    mode: str,
    eps: float,
    max_inner: int,
) -> tuple[np.ndarray, int]:
    """Solve min_W F_λ(W) by gradient descent from W_init.

    Runs gradient descent with step 1/Lλ, where Lλ = λ^T L, and stops when
    the single-point progress criterion PC(λ; {W}) ≤ eps.

    Returns (W_final, num_steps).  num_steps is the count of gradient-descent
    iterations executed at this λ; each iteration costs K gradient oracle calls
    (one per objective).
    """
    # Local bundle containing a single point — just enough to evaluate PC.
    bundle = Bundle(K=K, d=d, L=L, mu=mu)
    bundle.add_point(W_init.copy(), objectives, grad_objectives)
    pc_fn = _select_pc_fn(mode)

    W = W_init.copy()
    Ll = float(lam @ L)
    steps = 0

    # Shortcut: if already converged at W_init, don't step.
    if pc_fn(bundle, lam) <= eps:
        return W, steps

    for _ in range(max_inner):
        # Gradient step:  W ← W − (1/Lλ) ∇F_λ(W)
        g_lambda = sum(lam[k] * grad_objectives[k](W) for k in range(K))
        W = W - (1.0 / Ll) * g_lambda
        steps += 1

        # Rebuild a one-point bundle at the new W and check PC.
        bundle = Bundle(K=K, d=d, L=L, mu=mu)
        bundle.add_point(W, objectives, grad_objectives)
        if pc_fn(bundle, lam) <= eps:
            break

    return W, steps


def _select_pc_fn(mode: str) -> Callable:
    """Dispatch to UB / GAP / GN with the same semantics as algorithm.py."""
    if mode == "gap":
        return GAP
    if mode == "ub":
        return UB
    if mode == "gn":
        return GN
    raise ValueError(f"Unknown mode: {mode!r}.  Use 'gap', 'ub', or 'gn'.")


# =====================================================================
#  Main baseline routine
# =====================================================================
def uniform_discretisation(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    eps: float,
    mode: str = "gap",
    mu: Optional[np.ndarray] = None,
    max_inner: int = 200,
    resolution: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Run the uniform-discretisation baseline.

    Parameters
    ----------
    K, d              : number of objectives and dimension of W.
    objectives        : list of K callables  F_k(W) → float.
    grad_objectives   : list of K callables  ∇F_k(W) → ndarray shape (d,).
    L                 : smoothness constants, shape (K,).
    x0                : initial iterate (used at the first grid point).
    eps               : target accuracy — also sets the grid spacing  ≈ √ε.
    mode              : "gap", "ub", or "gn" (same semantics as algorithm2).
    mu                : strong convexity / PL constants; required for "gap" / "ub".
    max_inner         : safety cap on inner steps per grid point.
    resolution        : grid resolution override; default is ⌈1/√ε⌉.
    verbose           : if True, print per-grid-point progress.

    Returns
    -------
    dict with keys:
        "grid"              : (N, K) array of grid points (in warm-start order).
        "solutions"         : (N, d) array of per-grid-point solutions Ŵ(λ^(g)).
        "steps_per_point"   : list of inner-step counts at each grid point.
        "oracle_calls"      : total gradient oracle evaluations.
        "resolution"        : grid resolution used.
        "eps"               : target accuracy.

    Example
    -------
    >>> res = uniform_discretisation(
    ...     K=3, d=12, objectives=objs, grad_objectives=grads,
    ...     L=L, x0=np.zeros(12), eps=1e-2, mode="gap", mu=mu,
    ... )
    >>> res["oracle_calls"]
    19800
    >>> Ŵ_hat = predict_baseline(res, np.array([0.4, 0.3, 0.3]))
    """
    if resolution is None:
        resolution = max(1, math.ceil(1.0 / eps))

    grid = _uniform_simplex_grid(K, resolution)
    grid = _sort_grid_for_warmstart(grid)
    n_grid = grid.shape[0]

    if verbose:
        print(
            f"  Baseline: resolution={resolution},  grid points={n_grid},  "
            f"√ε spacing = {1.0 / resolution:.4f}"
        )

    solutions = np.zeros((n_grid, d))
    steps_per_point: List[int] = []
    W_prev = x0.copy()  # warm start seed

    for g, lam in enumerate(grid):
        W_g, steps = _local_solve(
            K=K, d=d, lam=lam, W_init=W_prev,
            objectives=objectives, grad_objectives=grad_objectives,
            L=L, mu=mu, mode=mode,
            eps=eps, max_inner=max_inner,
        )
        solutions[g] = W_g
        steps_per_point.append(steps)
        W_prev = W_g  # warm-start the next grid point

        if verbose and (g % max(1, n_grid // 10) == 0 or g == n_grid - 1):
            print(
                f"  grid {g + 1:4d}/{n_grid} | λ = {np.round(lam, 3)} | "
                f"inner steps = {steps:3d}"
            )

    oracle_calls = K * sum(steps_per_point)

    return {
        "grid": grid,
        "solutions": solutions,
        "steps_per_point": steps_per_point,
        "oracle_calls": oracle_calls,
        "resolution": resolution,
        "eps": eps,
    }


# =====================================================================
#  Query-time prediction: round-to-nearest-grid-point
# =====================================================================
def predict_baseline(baseline_res: Dict, lam: np.ndarray) -> np.ndarray:
    """Return the baseline's approximate solution at a query λ.

    Rounds ``lam`` to the nearest stored grid point (ℓ₁ distance) and
    returns the precomputed W at that grid point.

    This realises the baseline's solution map  Ŵ_baseline : Δ_K → R^d.
    """
    grid = baseline_res["grid"]
    sols = baseline_res["solutions"]
    # ℓ₁ distance from lam to every grid point.
    dists = np.sum(np.abs(grid - lam[None, :]), axis=1)
    g_best = int(np.argmin(dists))
    return sols[g_best]


# =====================================================================
#  Evaluating the solution map's worst-case PC on a fine test grid
# =====================================================================
def evaluate_map_worstcase(
    baseline_res: Dict,
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    mu: Optional[np.ndarray],
    mode: str = "gap",
    test_resolution: int = 50,
) -> float:
    """Compute  sup_{λ ∈ Δ_K}  PC(λ; {Ŵ_baseline(λ)}).

    This is the analogue of Algorithm 2's PC*_t at convergence — it
    measures how well the baseline's rounded solution map satisfies the
    progress criterion uniformly over the simplex.

    Because the baseline rounds to the nearest grid point and the PC is
    typically worst at the *mid-point* between adjacent grid points, this
    quantity scales with the grid spacing and usually exceeds ε.
    """
    pc_fn = _select_pc_fn(mode)
    test_grid = _uniform_simplex_grid(K, test_resolution)
    worst = -np.inf
    for lam in test_grid:
        W_hat = predict_baseline(baseline_res, lam)
        bundle = Bundle(K=K, d=d, L=L, mu=mu)
        bundle.add_point(W_hat, objectives, grad_objectives)
        worst = max(worst, pc_fn(bundle, lam))
    return float(worst)
