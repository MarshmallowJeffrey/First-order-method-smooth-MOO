"""
baseline_without_256_checkpoints.py  –  Uniform-discretisation baseline,
A/B measurement variant
========================================================================

Identical to ``baseline.py`` in every algorithmic respect (same steps, same
snake/warm-start order, same trajectory for the same seeds) EXCEPT what
checkpoints record: instead of building a snapshot bundle and running the
external fixed 256-start ``pc_star`` yardstick, ``cov_history`` records the
method's OWN knowledge — the worst (max) most-recently-computed
‖∇F_{λ_i}(x_i)‖² across the grid nodes, each node scored at its own weight
λ_i, using the gradients the method already computed for its steps (values
lag a node's position by one step, and by up to one pass for nodes not yet
revisited; every node is initialised with its value at x0).  Note this
self-reported criterion has no term for weights BETWEEN grid nodes — that
is exactly the difference the A/B experiment is designed to expose.  The
original file and its outputs are untouched; see
output/without_256_checkpoints/README.md.

Evaluation protocol
-------------------
- The baseline uses a coarser resolution r and runs warm-started gradient
  descent across the coarse grid repeatedly.  There are two stopping modes:

  * Budget mode (default, ``node_tol=None``): the inner loop does NOT stop
    based on any per-point tolerance — every node is polished for a fixed
    number of steps per pass until the pass schedule or the gradient budget
    ``max_grad_evals`` runs out.  This is the paper's Algorithm 1
    experimental protocol and the mode all benchmark sweeps use.

  * Certification mode (``node_tol`` set): at each node visit, before any
    step is taken, the gradient the first step would use anyway is checked;
    a node with ‖∇F_{λ_i}(x)‖² ≤ node_tol is marked "served" and frozen
    (later passes skip it, so the mark stays valid).  The run stops as soon
    as every node is served — the cumulative gradient count at that moment
    is the certification cost — or when ``max_grad_evals``, kept as a fuse,
    is hit first, in which case the failure is reported honestly in the
    result dictionary.  The check reuses the step gradient and never adds
    oracle calls; the evaluation that triggers a mark is counted like any
    other.

- In both modes, after every M total gradient evaluations (a "checkpoint"),
  we pause, evaluate the current worst-case error of the solution map,
  record (CPU time, err), and resume.

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


def _snake_compositions(s: int, m: int, forward: bool = True) -> List[List[int]]:
    """Enumerate compositions of ``s`` into ``m`` parts so consecutive ones
    differ by exactly one +1/−1 coordinate pair (ℓ₁ distance 2).

    Construction: iterate the first coordinate ``a`` upward, recursing on the
    remaining ``m−1`` parts and reversing direction on alternate blocks
    (boustrophedon).  By induction the forward enumeration starts at
    (0, …, 0, s) and ends at (s, 0, …, 0), which makes adjacent blocks meet at
    tails differing by exactly 1 in the first tail coordinate.
    """
    if m == 1:
        return [[s]]
    out: List[List[int]] = []
    for a in range(s + 1):
        block = _snake_compositions(s - a, m - 1, forward=(a % 2 == 0))
        out.extend([a] + tail for tail in block)
    return out if forward else out[::-1]


def _sort_grid_for_warmstart(grid: np.ndarray) -> np.ndarray:
    """Snake (boustrophedon) order: consecutive points are ℓ₁-close
    (≤ 2/resolution apart), matching the enumeration guarantee the paper's
    Algorithm 1 warm-start relies on.

    The previous plain lexicographic sort satisfied this only for K = 2: for
    K ≥ 3 every lexicographic "carry" boundary jumps up to the full simplex
    diameter (ℓ₁ = 2), seeding the chained warm start from a nearly opposite
    trade-off weight at a sizable fraction of grid transitions.
    """
    K = grid.shape[1]
    if K == 1:
        return grid
    # Recover the integer resolution r from the grid spacing: the grid is
    # exactly {c / r : c a composition of r into K parts}.
    r = int(round(1.0 / np.min(grid[grid > 0]))) if np.any(grid > 0) else 1
    order_keys = {
        tuple(comp): i
        for i, comp in enumerate(_snake_compositions(r, K))
    }
    counts = np.rint(grid * r).astype(int)
    order = np.argsort([order_keys[tuple(row)] for row in counts])
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
    node_tol: Optional[float] = None,
    evaluate_coverage: bool = False,
    joint_oracle: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """Run the baseline in "progressive" mode, with periodic checkpoints.

    We construct a coarse grid G_r of Δ_K at resolution ``resolution``
    and walk through it in snake (boustrophedon) warm-start order, so
    consecutive grid points are ℓ₁-adjacent (≤ 2/r apart) as the paper's
    Algorithm 1 enumeration requires.  Each pass does
    ``steps_per_point_per_pass`` solver steps at every grid point.

    Certification mode (``node_tol`` set)
    -------------------------------------
    ``node_tol`` is the per-node acceptance level: a node whose own
    weighting λ_i satisfies ‖∇F_{λ_i}(x)‖² ≤ node_tol counts as "served".
    At every node visit the gradient the first step would consume anyway
    is checked BEFORE stepping; a passing node is marked served, is not
    moved, and is skipped by all later passes (so the mark stays valid).
    The run ends when all nodes are served (certification success) or when
    ``max_grad_evals`` — retained as a fuse — or the pass schedule runs out
    first (certification failure, reported in the result dictionary).
    Accounting follows the paper's protocol: the check reuses the step
    gradient (never an extra oracle call) and the evaluation that triggers
    a mark is counted like any other; on success a final checkpoint is
    recorded as usual, so ``cov_history[-1]`` is the measured GN* of the
    delivered point set at certification time (metric cost excluded from
    both axes, as always).  With the default ``node_tol=None`` the
    behaviour is exactly the pre-existing budget mode.

    Checkpoint cadence
    ------------------
    By default (``eval_every_n_grads=None``) we checkpoint after every
    pass, matching the previous behaviour.  Setting
    ``eval_every_n_grads = M`` instead causes a checkpoint as soon as at
    least M additional gradient-oracle evaluations have been used, including
    in the middle of a pass (one scalarised solver step costs K gradient
    oracle calls).

    One "pass" = one full sweep across all grid points with M_pp steps
    per point = |G_r| · M_pp scalarised iterations = |G_r| · M_pp · K
    gradient-oracle evaluations.

    Parameters
    ----------
    resolution                : coarse grid resolution  r.
    n_passes                  : total number of passes to run.
    steps_per_point_per_pass  : solver steps taken at each grid point per pass.
    eval_every_n_grads        : if set, checkpoint every M gradient evals, including mid-pass.
    max_grad_evals            : optional hard budget on cumulative gradient evaluations.
                                In certification mode this is the fuse.
    node_tol                  : optional per-node acceptance level on ‖∇F_{λ_i}‖²
                                (certification mode, see above).  None = budget mode.

    Returns
    -------
    dict with keys:
        "coarse_grid"             : (N, K) array of grid points.
        "final_solutions"         : (N, d) array of final solutions.
        "cpu_times"               : list of CPU times at each checkpoint (s).
        "cov_history"             : measured GN* per checkpoint (only when
                                    ``evaluate_coverage=True``).
        "total_iters_history"     : cumulative scalarised iters per ckpt.
        "grad_evals_history"      : cumulative gradient-oracle evals per ckpt (= total_iters * K).
        "resolution"              : grid resolution used.
        "max_grad_evals"          : the budget/fuse that was in force.
        "node_tol"                : the acceptance level used (None = budget mode).
    and, describing certification (all None when ``node_tol`` is None):
        "certified"               : True iff every node was served before the
                                    fuse/pass schedule ran out.
        "certified_grad_evals"    : cumulative gradient evaluations at the
                                    moment the last node certified (the
                                    certification cost); None on failure.
        "node_served"             : list of N booleans, service status per node.
        "node_grad_sq"            : list of N floats, last measured ‖∇F_{λ_i}‖²
                                    per node.  For a served node this is the
                                    certifying value at its frozen point (exact
                                    from then on); for an unserved node it is
                                    the gradient of its most recent step, i.e.
                                    it lags that node's final position by one
                                    step; NaN if the node was never visited.
        "unserved_nodes"          : list of node indices not yet served
                                    (empty on certification success).
    """
    x0_arr = np.asarray(x0, dtype=float)
    if x0_arr.ndim != 1 or x0_arr.size == 0:
        raise ValueError(f"x0 must be a nonempty one-dimensional array; got {x0_arr.shape}.")
    d = int(x0_arr.size)
    L_arr, x0_arr = validate_problem_inputs(
        K, d, L, x0_arr, objectives, grad_objectives
    )
    if (
        eval_every_n_grads is not None
        and (
            not isinstance(eval_every_n_grads, (int, np.integer))
            or isinstance(eval_every_n_grads, (bool, np.bool_))
            or eval_every_n_grads < 1
        )
    ):
        raise ValueError(
            "eval_every_n_grads must be a positive integer or None; "
            f"got {eval_every_n_grads!r}."
        )
    if max_grad_evals is not None and max_grad_evals < K:
        raise ValueError(
            f"max_grad_evals must be at least K={K}; got {max_grad_evals}."
        )
    if node_tol is not None and (
        not isinstance(node_tol, (int, float, np.integer, np.floating))
        or isinstance(node_tol, (bool, np.bool_))
        or not np.isfinite(node_tol)
        or node_tol <= 0.0
    ):
        raise ValueError(
            f"node_tol must be a finite positive number or None; got {node_tol!r}."
        )

    joint_oracle = prefer_fused_joint_oracle(joint_oracle)
    coarse_grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, resolution))
    N = coarse_grid.shape[0]

    # Initialise all grid-point solutions to x0.
    solutions = np.tile(x0_arr, (N, 1))

    # Certification-mode state, plus the A/B variant's self-report store:
    # ``node_grad_sq`` holds the most recently computed ‖∇F_{λ_i}‖² per node.
    # In this variant it is written on EVERY step (a dot product on the step
    # gradient, never an oracle call) and doubles as the checkpoint metric.
    # Every node starts at x0, so initialise every node's value there; this
    # initial evaluation is preprocessing, excluded from both reported axes
    # exactly like the rest of the checkpoint-0 work.
    certifying = node_tol is not None
    node_served = np.zeros(N, dtype=bool)
    if joint_oracle is not None:
        _, _init_grads = validate_oracle_output(*joint_oracle(x0_arr), K, d)
    else:
        _init_grads = np.vstack([
            np.asarray(g(x0_arr), dtype=float) for g in grad_objectives
        ])
    _init_glam = coarse_grid @ _init_grads
    node_grad_sq = np.einsum("nd,nd->n", _init_glam, _init_glam)
    certified_grad_evals: Optional[int] = None
    all_served = False

    cpu_times: List[float] = []
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
        # A/B variant: no snapshot bundle, no external 256-start pc_star
        # solve.  Record the method's own knowledge: the worst (max)
        # most-recently-computed ‖∇F_{λ_i}‖² across the grid nodes, each
        # node scored at its own weight with a gradient the method already
        # computed for its steps.  Weights BETWEEN grid nodes do not enter
        # this number at all.
        if evaluate_coverage:
            cov_history.append(float(np.max(node_grad_sq)))
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
            if certifying and node_served[g_idx]:
                # A served node is frozen: no oracle calls, no movement, so
                # the mark stays valid.  Marks are only set during a visit,
                # so this can never skip a node on the pass-1 warm-start
                # chain (x_prev is not read on later passes).
                continue
            lam = coarse_grid[g_idx]
            Ll = float(lam @ L_arr)

            if pass_idx == 1:
                x = x_prev.copy()
            else:
                x = solutions[g_idx].copy()

            # Vanilla GD, with the certification-mode acceptance check on
            # each visit's first step gradient.
            for step_idx in range(steps_per_point_per_pass):
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
                certified_now = False
                # A/B variant: track every node's latest step-gradient norm
                # unconditionally — it is this variant's checkpoint metric.
                node_grad_sq[g_idx] = float(g_lam @ g_lam)
                if certifying:
                    if step_idx == 0 and node_grad_sq[g_idx] <= node_tol:
                        # Acceptance check, before stepping: the gradient
                        # the first step would have consumed certifies this
                        # node.  Mark it served and leave the point unmoved;
                        # the evaluation was real oracle work and is counted
                        # like any other.
                        node_served[g_idx] = True
                        certified_now = True
                if not certified_now:
                    x = x - (1.0 / Ll) * g_lam
                total_iters += 1
                if certified_now and bool(node_served.all()):
                    certified_grad_evals = total_iters * K
                    all_served = True

                # When gradient-based checkpointing is requested, record the
                # current state immediately rather than waiting for a full
                # simplex-grid pass.  Store the in-progress grid solution so
                # the checkpoint includes the step that triggered it.
                cur_grad_evals = total_iters * K
                if (
                    eval_every_n_grads is not None
                    and (
                        cur_grad_evals - grad_evals_at_last_ckpt
                    ) >= eval_every_n_grads
                ):
                    solutions[g_idx] = x
                    _checkpoint(
                        f"pass {pass_idx}/{n_passes}, "
                        f"point {g_idx + 1}/{N}"
                    )
                    grad_evals_at_last_ckpt = cur_grad_evals

                if all_served:
                    break
                if (
                    max_grad_evals is not None
                    and cur_grad_evals >= max_grad_evals
                ):
                    budget_exhausted = True
                    break
                if certified_now:
                    # The visit is over: no steps are taken at a node that
                    # certified on entry.
                    break

            solutions[g_idx] = x
            x_prev = x
            if budget_exhausted or all_served:
                break

        # With the default cadence, every pass boundary is a checkpoint.  For
        # gradient-based cadence, mid-pass checkpoints were already recorded;
        # only add a boundary checkpoint for the final state when needed.
        cur_grad_evals = total_iters * K
        final_state = pass_idx == n_passes or budget_exhausted or all_served
        do_ckpt = eval_every_n_grads is None or (
            final_state and cur_grad_evals != grad_evals_at_last_ckpt
        )
        if do_ckpt:
            _checkpoint(f"pass {pass_idx}/{n_passes}")
            grad_evals_at_last_ckpt = cur_grad_evals
        if budget_exhausted or all_served:
            break

    if verbose and certifying:
        if bool(node_served.all()):
            print(
                f"  Baseline certification: all {N} nodes served at "
                f"grad_evals={certified_grad_evals}."
            )
        else:
            print(
                "  Baseline certification FAILED within the budget/schedule: "
                f"{int(node_served.sum())}/{N} nodes served."
            )

    return {
        "coarse_grid": coarse_grid,
        "final_solutions": solutions,
        "cpu_times": cpu_times,
        "total_iters_history": total_iters_history,
        "grad_evals_history": grad_evals_history,
        "cov_history": cov_history,
        "resolution": resolution,
        "max_grad_evals": max_grad_evals,
        "node_tol": node_tol,
        "certified": bool(node_served.all()) if certifying else None,
        "certified_grad_evals": certified_grad_evals,
        "node_served": node_served.tolist() if certifying else None,
        "node_grad_sq": node_grad_sq.tolist() if certifying else None,
        "unserved_nodes": (
            [int(i) for i in np.flatnonzero(~node_served)]
            if certifying
            else None
        ),
    }
