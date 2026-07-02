"""
algorithm.py  –  Adaptive Algorithm (Algorithm 6)
"""

from __future__ import annotations
import numpy as np
import time
from typing import Callable, Dict, List, Optional, Tuple
from scipy.optimize import minimize as sp_minimize

# Optional IPOPT backend for the GN lambda-maximisation (used as K grows).
# cyipopt needs an IPOPT install; the import is guarded so the rest of the
# module still loads if it is absent, in which case `_maximise_GN` falls
# back to SLSQP with a one-time warning.

try:
    from cyipopt import minimize_ipopt as _ipopt_minimize
    _HAS_IPOPT = True
    _IPOPT_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    _ipopt_minimize = None
    _HAS_IPOPT = False
    _IPOPT_IMPORT_ERROR = exc
import warnings

from bundle import Bundle, prefer_fused_joint_oracle, validate_problem_inputs


def ipopt_available() -> bool:
    """Return whether the optional cyipopt/IPOPT backend is available."""
    return _HAS_IPOPT


# =====================================================================
#  Vectorised bundle helpers (CPU-optimisation, no semantic change)
# =====================================================================
# These helpers replace per-bundle-point Python loops with batched NumPy
# operations without changing the T-map calculation.
# ---------------------------------------------------------------------
def _bundle_arrays(bundle: Bundle) -> Tuple[np.ndarray, np.ndarray]:
    """Stack ``bundle.fvals`` / ``bundle.grads`` as contiguous arrays.

    Returns
    -------
    Fmat : (m, K) array, Fmat[i, k] = F_k(x_i)
    Jmat : (m, K, d) array, Jmat[i, k] = ∇F_k(x_i)
    """
    Fmat = np.asarray(bundle.fvals)
    Jmat = np.asarray(bundle.grads)
    return Fmat, Jmat


def _gn_value_batched(Jmat: np.ndarray, lam: np.ndarray) -> float:
    """Evaluate GN(λ; B) without computing its λ-gradient."""
    weighted_grads = np.einsum('ikd,k->id', Jmat, lam)       # (m, d)
    gnorms_sq = np.einsum('id,id->i', weighted_grads, weighted_grads)
    return float(np.min(gnorms_sq))


def _gn_value_and_jac_batched(
    Jmat: np.ndarray, lam: np.ndarray
) -> Tuple[float, np.ndarray, int]:
    """Batched evaluation and analytical λ-gradient of  GN(λ; B)  (Eq. 17).

    GN(λ) = min_i ‖J_i^T λ‖².

    Implementation
    --------------
    * Stack the bundle Jacobians and contract with λ in one ``einsum`` to
      obtain  ``G[i] = J_i^T λ``  for all bundle points simultaneously,
      replacing a Python loop over bundle points.
    * The argmin index ``i*`` is unique generically.  By Danskin's theorem
      the gradient of GN at smooth points is
          ∇_λ GN(λ) = 2 J_{i*} g_{i*},
      where ``g_{i*} = J_{i*}^T λ``.  ``J_{i*}`` has shape (K, d), so
      ``J_{i*} g_{i*}`` is a (K,) vector — the same shape as λ.
      If several bundle points tie for the minimum, GN is generally
      non-differentiable; ``np.argmin`` selects the first active branch.

    Returns
    -------
    gn_value : float
    gn_jac   : (K,) array,  ∇_λ GN(λ)
    i_star   : int          argmin index, useful for caller diagnostics
    """
    G = np.einsum('ikd,k->id', Jmat, lam)              # (m, d)
    gnorms_sq = np.einsum('id,id->i', G, G)            # (m,)
    i_star = int(np.argmin(gnorms_sq))
    g_istar = G[i_star]                                # (d,)
    gnorm_sq_istar = float(gnorms_sq[i_star])

    # ∇_λ ‖J_i*^T λ‖²  =  2 J_i* (J_i*^T λ)  =  2 J_i* g_i*    ∈  R^K
    grad_min_norm = 2.0 * (Jmat[i_star] @ g_istar)     # (K,)

    gn_value = gnorm_sq_istar
    gn_jac = grad_min_norm
    return gn_value, gn_jac, i_star


def _T_map_batched(Fmat: np.ndarray, Jmat: np.ndarray, points_arr: np.ndarray,
                   L: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Vectorised T-map evaluation for one weight vector (Eq. 13).

    Computes
        i*  = argmin_i { F_λ(x_i) − 1/(2 Lλ) ‖∇F_λ(x_i)‖² }
        T   = x_{i*} − (1/Lλ) ∇F_λ(x_{i*})
    in a single batched pass.

    Parameters
    ----------
    Fmat, Jmat   : as in ``_bundle_arrays``.
    points_arr   : (m, d) array, points_arr[i] = x_i (= np.asarray(bundle.points))
    L, lam       : K-vectors.
    """
    Ll = float(lam @ L)
    F_lam = Fmat @ lam
    grad_lam = np.einsum('ikd,k->id', Jmat, lam)         # (m, d)
    gnorm_sq = np.einsum('id,id->i', grad_lam, grad_lam) # (m,)
    u_vals = F_lam - 0.5 * gnorm_sq / Ll
    i_star = int(np.argmin(u_vals))
    return points_arr[i_star] - (1.0 / Ll) * grad_lam[i_star]



# =====================================================================
#  PC-specific λ maximisation
# =====================================================================
# GN:  non-concave  →  multi-start local search
# ---------------------------------------------------------------------------
def _gn_multistart_set(K: int, prev_lam, max_starts: int,
                       seed: int = 0):
    """Deterministic multi-start set for the non-concave GN maximisation.

    Priority order, truncated to ``max_starts``:
        centroid > vertices > near-corner points > prev_lam > edge midpoints

    The first blocks are O(K); the edge-midpoint block is O(K^2) and is the one
    that must be bounded as K grows.  When the full structured set fits under
    ``max_starts`` (small / moderate K) every start is emitted and the set is
    identical to the original SLSQP implementation's.  When K is large the edge
    block is *lazily* sampled (never materialised in full), so the routine stays
    both time- and memory-bounded -- this is what keeps the per-checkpoint
    maximiser tractable as K is pushed higher.

    The near-corner points (lambda_k = 0.8, remainder spread uniformly) and the
    edge midpoints recover GN basins on the non-convex problem that the vertices
    and centroid alone miss; see the original implementation note.
    """
    if not isinstance(K, (int, np.integer)) or isinstance(K, bool) or K < 1:
        raise ValueError(f"K must be a positive integer; got {K!r}.")
    if (
        not isinstance(max_starts, (int, np.integer))
        or isinstance(max_starts, bool)
        or max_starts < 1
    ):
        raise ValueError(
            f"max_starts must be a positive integer; got {max_starts!r}."
        )
    if K == 1:
        return [np.ones(1)]

    rng = np.random.RandomState(seed)
    EPS, NEAR = 1e-8, 0.8
    starts = []

    def room():
        return max_starts - len(starts)

    # centroid
    if room() > 0:
        starts.append(np.full(K, 1.0 / K))

    # vertices (O(K)); subsample only if they would overflow the budget
    if room() > 0:
        idx = np.arange(K)
        if K > room():
            idx = rng.choice(K, size=room(), replace=False)
        for k in idx:
            e = np.full(K, EPS)
            e[k] = 1.0 - (K - 1) * EPS
            starts.append(e)

    # near-corner starts (O(K)); subsample only if needed
    if room() > 0:
        idx = np.arange(K)
        if K > room():
            idx = rng.choice(K, size=room(), replace=False)
        for k in idx:
            e = np.full(K, (1.0 - NEAR) / (K - 1))
            e[k] = NEAR
            starts.append(e)

    # warm start from the previous outer iteration's optimum (additional only)
    if room() > 0 and prev_lam is not None:
        starts.append(np.clip(prev_lam, EPS, 1.0))

    # edge midpoints (O(K^2)): enumerate when small, else lazily sample
    total_edges = K * (K - 1) // 2
    if room() > 0 and total_edges > 0:
        def edge_mid(a, b):
            e = np.full(K, EPS)
            e[a] = 0.5 - (K - 2) * 0.5 * EPS
            e[b] = 0.5 - (K - 2) * 0.5 * EPS
            return e
        if total_edges <= room():
            for a in range(K):
                for b in range(a + 1, K):
                    starts.append(edge_mid(a, b))
        else:
            seen = set()
            while room() > 0 and len(seen) < total_edges:
                a = int(rng.randint(0, K - 1))
                b = int(rng.randint(a + 1, K))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                starts.append(edge_mid(a, b))
    return starts


def _maximise_GN(bundle: Bundle, prev_lam: Optional[np.ndarray] = None,
                 solver: str = "ipopt", max_starts: int = 256
                 ) -> Tuple[float, np.ndarray]:
    """Find  lambda* = argmax_{lambda in Delta_K}  GN(lambda; B_m).

        GN(lambda; Bm) = min_i ||J_F(x_i)^T lambda||^2   (or its s.c. scaling)

    Solver backends
    ---------------
    The decision variable is lambda in Delta_K, so the dimensionality of *this*
    problem is K (the number of objectives), independent of d / m / n.

    * ``solver="ipopt"`` (default): each local solve is handed to IPOPT via
      cyipopt's ``minimize_ipopt``.  IPOPT is an interior-point NLP solver whose
      advantage grows with K (the regime this code is scaling toward).  GN is
      only piecewise-smooth (a pointwise min of quadratics, non-differentiable
      on the index-switching manifold), so we use IPOPT's limited-memory
      (L-BFGS) Hessian approximation rather than an exact Hessian, which would
      be discontinuous across the min's pieces.  The analytical Danskin gradient
      (``_gn_value_and_jac_batched``) is still supplied, giving IPOPT exact
      first-order information.
    * ``solver="slsqp"``: the original scipy multi-start SLSQP path, kept as a
      fallback and for benchmarking.  It is also used automatically (with a
      one-time warning) when cyipopt is not importable.

    Multi-start / scaling in K
    --------------------------
    GN is neither convex nor concave in lambda, so a single local solve only
    finds a local max; ``_gn_multistart_set`` supplies the global coverage.
    ``max_starts`` bounds the number of local solves so the O(K^2) edge-midpoint
    block does not blow up the per-checkpoint maximisation at large K; for small
    K the full structured set fits and behaviour matches the original.

    Each local solve is wrapped in try/except and the start point itself is
    scored first, so a failed or early-terminated solve never loses ground and
    the returned value is monotone in the starts actually evaluated.

    This routine is invoked by ``pc_star`` for metric evaluation (excluded from
    the plotted cost axes) and by the outer loop to choose lambda.
    """
    if solver not in {"ipopt", "slsqp"}:
        raise ValueError("solver must be 'ipopt' or 'slsqp'.")
    if (
        not isinstance(max_starts, (int, np.integer))
        or isinstance(max_starts, bool)
        or max_starts < 1
    ):
        raise ValueError(
            f"max_starts must be a positive integer; got {max_starts!r}."
        )
    if bundle.m == 0:
        raise ValueError("Cannot maximise GN for an empty bundle.")

    K = bundle.K
    Jmat = np.asarray(bundle.grads)
    if K == 1:
        lam = np.ones(1)
        return _gn_value_batched(Jmat, lam), lam

    def neg_gn(lam):
        v, _, _ = _gn_value_and_jac_batched(Jmat, lam)
        return -v

    def neg_gn_jac(lam):
        _, j, _ = _gn_value_and_jac_batched(Jmat, lam)
        return -j

    con_eq = {"type": "eq",
              "fun": lambda l: float(np.sum(l) - 1.0),
              "jac": lambda l: np.ones(K)}
    constraints = [con_eq]
    bounds = [(1e-8, 1.0)] * K

    starts = _gn_multistart_set(K, prev_lam, max_starts)

    use_ipopt = (solver == "ipopt") and _HAS_IPOPT
    if solver == "ipopt" and not _HAS_IPOPT:
        warnings.warn(
            "cyipopt/IPOPT is unavailable; _maximise_GN is falling back to SLSQP. "
            "Install IPOPT + cyipopt (e.g. `conda install -c conda-forge "
            f"cyipopt`) to enable the IPOPT backend. Import error: "
            f"{_IPOPT_IMPORT_ERROR}",
            RuntimeWarning, stacklevel=2,
        )

    best_val = np.inf            # minimum of neg_gn == -(max GN)
    best_lam = starts[0]
    for lam0 in starts:
        # Score the start point itself first so a failed / early-terminated
        # solve never loses ground (keeps the result monotone in the starts).
        v0 = neg_gn(lam0)
        if v0 < best_val:
            best_val, best_lam = float(v0), np.asarray(lam0, dtype=float)

        try:
            if use_ipopt:
                res = _ipopt_minimize(
                    neg_gn, lam0, jac=neg_gn_jac,
                    bounds=bounds, constraints=constraints,
                    options={
                        "print_level": 0,
                        "sb": "yes",                    # suppress IPOPT banner
                        "tol": 1e-8,
                        "max_iter": 100,
                        "hessian_approximation": "limited-memory",
                    },
                )
            else:
                res = sp_minimize(
                    neg_gn, lam0, jac=neg_gn_jac, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"ftol": 1e-6, "maxiter": 60},
                )
        except Exception as exc:
            # A single failed local solve must not abort the maximisation.
            warnings.warn(
                f"GN local solve failed from start {lam0}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = float(res.fun)
            best_lam = np.asarray(res.x, dtype=float).copy()

    best_lam = np.maximum(best_lam, 0.0)
    s = best_lam.sum()
    best_lam = best_lam / s if s > 0 else np.full(K, 1.0 / K)
    return float(-best_val), best_lam


# =====================================================================
#  Adaptive inner loop (BundleUpdate with max_steps)
# =====================================================================
def _bundle_update_adaptive(
    bundle: Bundle,
    lam: np.ndarray,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    max_steps: int,
    eps_inner: Optional[float] = None,
    prune: bool = False,
    joint_oracle: Optional[Callable] = None,
) -> int:
    """Inner-loop BundleUpdate at fixed λ.

    Two stopping modes
    ------------------
    * ``eps_inner=None`` (default):  run exactly ``max_steps`` T_map
      iterations, then (if ``prune``) commit only the candidate with
      the smallest ∥∇F_λ∥ to the bundle.  Backward-compatible with the
      original fixed-budget inner loop.

    * ``eps_inner=ε/3`` (Algorithm 2 from the paper):  the convergence
      proof (Appendix B.1) requires the bundle update to drive
      ``GN(λ_t; B_{t+1}) ≤ ε/3`` at the active λ_t before the outer
      loop advances.  In this mode we add T_map iterates one at a time
      and recompute the active-λ GN after each addition; the inner loop
      terminates as soon as GN drops below ``eps_inner`` or after
      ``max_steps`` candidates (safety cap).

    Pruning heuristic
    -----------------
    The paper's §7 implementation note adds *only* the candidate with
    the smallest gradient norm to the bundle, motivated by the
    observation that the other candidates contribute negligibly once
    the active index is well-served.  We expose this as ``prune``:

    * ``prune=True`` — the runtime-efficient §7 heuristic.
    * ``prune=False`` (default) — keep every committed candidate; faithful to
      the proof in Appendix B.1 which assumes BundleUpdate appends
      every T_map iterate.

    Implementation note (CPU optimisation, no semantic change)
    ----------------------------------------------------------
    The original implementation built a ``copy.deepcopy(bundle)`` work
    bundle to hold the candidate chain, then committed only the winner
    to the real bundle.  We instead append candidates in-place and pop
    the losers at the end, avoiding O(m·K·d) bundle copying per outer.
    The T_map call uses the vectorised ``_T_map_batched`` helper.

    Returns
    -------
    Number of T_map iterations actually executed.  Equals ``max_steps``
    when no stopping rule fires within the safety cap; smaller otherwise.
    """
    base_m = bundle.m
    steps_taken = 0
    K = bundle.K
    d = bundle.d
    L_arr = bundle.L

    # ------------------------------------------------------------------
    # CPU optimisation:  maintain Fbuf/Jbuf/Pbuf as pre-allocated buffers
    # covering the base bundle plus up to ``max_steps`` new candidates.
    # This avoids the O((m + s)·K·d) rebuild that
    # ``_bundle_arrays(bundle)`` + ``np.asarray(bundle.points)`` would
    # otherwise pay on EVERY T-map step.  After ``bundle.add_point``,
    # we assign the new row into the next slot of the buffer (an
    # O(K·d) write) and slice the buffer to the active region for
    # ``_T_map_batched``.
    #
    # The optional inner stopping check reuses the cached gradients and calls
    # the scalar-only GN evaluator, avoiding both a bundle rebuild and the
    # unnecessary analytical λ-gradient used by the outer optimiser.
    cap = base_m + max_steps
    Fbuf = np.empty((cap, K), dtype=np.float64)
    Jbuf = np.empty((cap, K, d), dtype=np.float64)
    Pbuf = np.empty((cap, d), dtype=np.float64)
    if base_m > 0:
        Fbuf[:base_m] = np.asarray(bundle.fvals)
        Jbuf[:base_m] = np.asarray(bundle.grads)
        Pbuf[:base_m] = np.asarray(bundle.points)

    # ------------------------------------------------------------------
    # Generate the candidate chain on the real bundle.  Each T_map call
    # sees all previously-added in-round candidates, matching the proof's
    # BundleUpdate semantics.
    # ------------------------------------------------------------------
    for s in range(max_steps):
        active = base_m + s
        # Slice views (no copy) into the live portion of the buffers.
        Fmat = Fbuf[:active]
        Jmat = Jbuf[:active]
        points_arr = Pbuf[:active]

        x_new = _T_map_batched(Fmat, Jmat, points_arr, L_arr, lam)
        bundle.add_point(x_new, objectives, grad_objectives, joint_oracle=joint_oracle)
        steps_taken += 1

        # Append the just-evaluated row into the buffer — O(K·d) write.
        Fbuf[active] = bundle.fvals[-1]
        Jbuf[active] = bundle.grads[-1]
        Pbuf[active] = bundle.points[-1]

        if eps_inner is not None:
            # Include the newly-added row at index ``active``.
            pc_val = _gn_value_batched(Jbuf[:active + 1], lam)
            if pc_val <= eps_inner:
                break

    # ------------------------------------------------------------------
    # Optional pruning to the argmin-gnorm winner (paper §7 heuristic).
    # ------------------------------------------------------------------
    if prune and steps_taken > 1:
        # Pick argmin ∥∇F_λ(x^i)∥ from the cached gradients of the
        # candidates.  Vectorised via einsum.
        cand_Js = np.asarray(bundle.grads[base_m:base_m + steps_taken])  # (S, K, d)
        cand_grads_lam = np.einsum('skd,k->sd', cand_Js, lam)            # (S, d)
        cand_gnorms = np.einsum('sd,sd->s', cand_grads_lam, cand_grads_lam)
        best_local = int(np.argmin(cand_gnorms))
        best_idx = base_m + best_local

        # Save the winner, pop *all* candidates, push only the winner.
        win_x = bundle.points[best_idx]
        win_fv = bundle.fvals[best_idx]
        win_gv = bundle.grads[best_idx]
        for _ in range(steps_taken):
            bundle.pop_point()
        bundle.points.append(win_x)
        bundle.fvals.append(win_fv)
        bundle.grads.append(win_gv)

    return steps_taken


# =====================================================================
#  Instrumented Adaptive Algorithm:  checkpoint after each outer iteration
# =====================================================================
def algorithm_adaptive(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    *,
    max_outer: int = 120,
    max_inner: int = 25,
    epsilon: Optional[float] = None,
    eval_every_n_grads: Optional[int] = None,
    target_cov: Optional[float] = None,
    lambda_max_starts: int = 256,
    lambda_solver: str = "ipopt",
    require_ipopt: bool = False,
    max_grad_evals: Optional[int] = None,
    prune_inner: bool = False,
    joint_oracle: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """Run the adaptive bundle method with reference-map-free checkpoints.

    The current coverage experiment uses the GN progress criterion:

        GN*(B) = max_lambda min_i ||grad F_lambda(x_i)||^2.

    CPU time spent on checkpoint metric evaluation is excluded from the
    reported CPU axis, matching the baseline accounting.

    ``prune_inner=False`` is the default in every mode, so every inner-loop
    candidate is retained in the bundle as required by the full BundleUpdate
    used in Algorithm 2.  Explicit ``prune_inner=True`` remains available as
    a runtime heuristic, but forfeits the full-bundle proof condition in
    epsilon mode.
    """
    if (
        not isinstance(max_inner, (int, np.integer))
        or isinstance(max_inner, bool)
        or max_inner < 1
    ):
        raise ValueError(f"max_inner must be a positive integer; got {max_inner!r}.")
    if (
        not isinstance(lambda_max_starts, (int, np.integer))
        or isinstance(lambda_max_starts, bool)
        or lambda_max_starts < 1
    ):
        raise ValueError(
            "lambda_max_starts must be a positive integer; "
            f"got {lambda_max_starts!r}."
        )
    L_arr, x0_arr = validate_problem_inputs(
        K, d, L, x0, objectives, grad_objectives
    )
    if epsilon is not None and (
        not np.isscalar(epsilon)
        or not np.isfinite(epsilon)
        or epsilon <= 0.0
    ):
        raise ValueError(f"epsilon must be finite and strictly positive; got {epsilon!r}.")
    if isinstance(prune_inner, (bool, np.bool_)):
        effective_prune_inner = bool(prune_inner)
    else:
        raise ValueError(
            f"prune_inner must be True or False; got {prune_inner!r}."
        )
    if epsilon is not None and effective_prune_inner:
        warnings.warn(
            "prune_inner=True in epsilon mode discards inner candidates and "
            "therefore does not satisfy the full-bundle condition used by the "
            "epsilon convergence proof.",
            RuntimeWarning,
            stacklevel=2,
        )
    if lambda_solver not in {"ipopt", "slsqp"}:
        raise ValueError("lambda_solver must be 'ipopt' or 'slsqp'.")
    if require_ipopt and lambda_solver != "ipopt":
        raise ValueError("require_ipopt=True requires lambda_solver='ipopt'.")
    if require_ipopt and not _HAS_IPOPT:
        raise RuntimeError(
            "IPOPT was required for this run, but cyipopt/IPOPT is unavailable. "
            f"Import error: {_IPOPT_IMPORT_ERROR}"
        )
    if max_grad_evals is not None and max_grad_evals < K:
        raise ValueError(
            f"max_grad_evals must be at least K={K}; got {max_grad_evals}."
        )

    if lambda_solver == "ipopt" and _HAS_IPOPT:
        actual_lambda_solver = "ipopt"
    else:
        actual_lambda_solver = "slsqp"

    joint_oracle = prefer_fused_joint_oracle(joint_oracle)
    bundle = Bundle(K=K, d=d, L=L_arr)

    # Checkpoint 0 starts from the initial bundle but does not count its
    # oracle evaluation as iterative work, matching the baseline's setup.
    bundle.add_point(
        x0_arr,
        objectives,
        grad_objectives,
        joint_oracle=joint_oracle,
    )

    cpu_times: List[float] = []
    cov_history: List[float] = []
    pc_history: List[float] = []
    total_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    inner_steps_history: List[int] = []
    lambda_history: List[np.ndarray] = []

    total_iters = 0
    grad_evals_at_last_ckpt = 0
    # Keep measurement state separate so checkpoint frequency cannot change
    # the outer algorithm's multi-start set or selected lambda.
    outer_prev_lam: Optional[np.ndarray] = None
    metric_prev_lam: Optional[np.ndarray] = None
    checkpoint_overhead = 0.0
    t_start = time.time()

    def _checkpoint(label: str) -> None:
        nonlocal checkpoint_overhead, metric_prev_lam
        cpu_times.append(time.time() - t_start - checkpoint_overhead)
        ck_t0 = time.time()
        cov, cov_lam = pc_star(bundle, prev_lam=metric_prev_lam)
        metric_prev_lam = cov_lam
        cov_history.append(cov)
        checkpoint_overhead += time.time() - ck_t0
        total_iters_history.append(total_iters)
        grad_evals_history.append(total_iters * K)
        if verbose:
            print(
                f"  Adaptive {label} | t={cpu_times[-1]:.2f}s "
                f"| bundle={bundle.m} | iters={total_iters} "
                f"| grad_evals={total_iters * K} | worst-case pc={cov:.4e}"
            )

    _checkpoint(f"outer 0/{max_outer}")
    cpu_times[0] = 0.0
    t_start = time.time()
    checkpoint_overhead = 0.0

    eps_inner = None if epsilon is None else epsilon / 3.0

    for outer in range(1, max_outer + 1):
        inner_step_cap = max_inner
        if max_grad_evals is not None:
            remaining_steps = (max_grad_evals - total_iters * K) // K
            if remaining_steps <= 0:
                break
            inner_step_cap = min(inner_step_cap, int(remaining_steps))

        pc_val, lam = _maximise_GN(
            bundle,
            prev_lam=outer_prev_lam,
            solver=lambda_solver,
            max_starts=lambda_max_starts,
        )
        outer_prev_lam = lam
        pc_history.append(pc_val)
        lambda_history.append(lam.copy())

        if epsilon is not None and pc_val <= 2 * epsilon / 3:
            _checkpoint(f"outer {outer}/{max_outer}")
            break

        steps = _bundle_update_adaptive(
            bundle=bundle,
            lam=lam,
            objectives=objectives,
            grad_objectives=grad_objectives,
            max_steps=inner_step_cap,
            eps_inner=eps_inner,
            prune=effective_prune_inner,
            joint_oracle=joint_oracle,
        )
        inner_steps_history.append(steps)
        total_iters += steps

        cur_grad_evals = total_iters * K
        budget_reached = (
            max_grad_evals is not None
            and cur_grad_evals + K > max_grad_evals
        )
        do_ckpt = (
            eval_every_n_grads is None
            or (cur_grad_evals - grad_evals_at_last_ckpt) >= eval_every_n_grads
            or outer == max_outer
            or budget_reached
        )
        if do_ckpt:
            _checkpoint(f"outer {outer}/{max_outer}")
            grad_evals_at_last_ckpt = cur_grad_evals
            if target_cov is not None and cov_history[-1] <= target_cov:
                break
        if budget_reached:
            break

    return {
        "bundle": bundle,
        "cpu_times": cpu_times,
        "cov_history": cov_history,
        "pc_history": pc_history,
        "total_iters_history": total_iters_history,
        "grad_evals_history": grad_evals_history,
        "inner_steps_history": inner_steps_history,
        "lambda_history": lambda_history,
        "lambda_solver": actual_lambda_solver,
        "prune_inner": effective_prune_inner,
        "epsilon": epsilon,
        "max_grad_evals": max_grad_evals,
        "max_outer": max_outer,
        "max_inner": max_inner,
    }








# =====================================================================
#  Bundle-coverage stationarity metric (reference-map-free)
#
#  Used for the comparison described in the research note: instead of a
#  precomputed fine-resolution reference map, each method is scored by the
#  worst-case (over the simplex) progress criterion of its own bundle:
#
#    GN*(B)  = max_{lambda in Delta_K}  min_i ||grad F_lambda(x_i)||^2   (non-convex)
#
#  This is computable directly from the gradients at the bundle points, with
#  no reference map.
# =====================================================================
def pc_star(bundle: Bundle, prev_lam: Optional[np.ndarray] = None
            ) -> Tuple[float, np.ndarray]:
    """Evaluate the worst-case-over-the-simplex progress criterion.

    GN*(B)  = max_{lambda} min_i ||grad F_lambda(x_i)||^2   (non-convex)

    Metric evaluation uses the same gradient-based multi-start maximisation as
    the adaptive outer loop.
    """
    return _maximise_GN(bundle, prev_lam=prev_lam)
