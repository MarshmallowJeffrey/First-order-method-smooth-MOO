"""
algorithm.py  –  Algorithm 2 (Simple Adaptive Algorithm v2) from Section 3.1
==============================================================================

Two key improvements over the generic discretisation-based version:

1. **PC-specific λ maximisation.**  Instead of evaluating every PC on a
   simplex grid, we exploit each criterion's structure:

   - UB(λ; Bm) is concave in λ  (minimum of concave quadratics, see
     Proposition 6 proof).  We maximise it with scipy's SLSQP.
   - GAP₁ = UB − LB₁ is a difference-of-concave (DC) function (Prop. 6).
     We use a multi-start local search, since each local problem is small.
   - GN(λ; Bm) is non-concave but piecewise-rational.  We use multi-start
     local search seeded at simplex vertices and the previous λ_t.

2. **Adaptive inner-loop stopping.**  The convergence proof of Algorithm 2
   (Theorem 1, Appendix B.1) requires  PC(λ_t; B_{t+1}) ≤ ε/3  after the
   inner update.  Instead of precomputing a theoretical upper-bound M_t on
   the number of inner iterations, we run the inner loop and stop as soon
   as the actual PC at λ_t drops below ε/3.  This is both simpler and
   tighter — the algorithm does exactly as much work as needed.

Illustrative example
--------------------
Consider K = 3 objectives on R^12 (multi-class logistic regression).

    F_k(x) = per-class cross-entropy + (reg/2) ‖x‖²

With PC = GAP and ε = 0.01, the algorithm:
  1. Initialises the bundle at x_0 = 0.
  2. Finds λ_t = argmax GAP(λ; B_t) via multi-start SLSQP on Δ_3.
  3. Runs inner gradient-descent steps at λ_t, checking GAP(λ_t; B)
     after each step, and stops as soon as GAP(λ_t; B) ≤ ε/3.
  4. Repeats until max_{λ} GAP(λ; B_t) ≤ ε.
"""

from __future__ import annotations

import numpy as np
import math
from typing import Callable, Dict, List, Optional, Tuple
from scipy.optimize import minimize as sp_minimize

from bundle import Bundle, UB, GAP, GN, LB, T_map


# =====================================================================
#  PC-specific λ maximisation
# =====================================================================

# ---------------------------------------------------------------------------
# UB:  concave in λ  →  single convex optimisation
# ---------------------------------------------------------------------------
def _maximise_UB(bundle: Bundle) -> Tuple[float, np.ndarray]:
    """Find  λ* = argmax_{λ ∈ Δ_K}  UB(λ; B_m).

    Structure
    ---------
    From Eq. (12)/(24) and the proof of Proposition 6:

        UB(λ; Bm) = min_{i ∈ [m]} u_i(λ)

    where  u_i(λ) = λ^T F(x_i) − (1/(2Lλ)) λ^T J_F(x_i) J_F(x_i)^T λ.

    Each u_i is concave in λ  (since −(1/(2Lλ)) ‖J^T λ‖² is concave when
    Lλ = λ^T L is linear in λ — verified in the proof of Prop. 6).
    UB is the pointwise minimum of concave functions, hence concave.

    Maximising a concave function over the simplex is a convex problem.
    We use scipy SLSQP with a few random restarts (the landscape is
    concave but may have kinks from the min operation).

    Illustrative example
    --------------------
    With m = 5 bundle points and K = 3, we maximise UB over the
    2D simplex Δ_3.  SLSQP with 3–5 starting points reliably finds
    the global max since UB is concave.
    """
    K = bundle.K
    m = bundle.m

    def neg_ub(lam):
        return -UB(bundle, lam)

    def neg_ub_grad(lam):
        """Subgradient of −UB(λ) via Danskin's theorem.

        Since UB(λ) = min_i u_i(λ), by Danskin's theorem a subgradient
        of UB at λ is ∇u_{i*}(λ) where i* achieves the minimum.

        ∇u_i(λ) = F(x_i) − J_F(x_i) J_F(x_i)^T λ / Lλ
                   + (λ^T J_F(x_i) J_F(x_i)^T λ) / (2 Lλ²) · L

        (from Eq. in proof of Prop. 6, page 19).
        """
        Ll = bundle.L_lam(lam)
        best_val = np.inf
        best_i = 0
        for i in range(m):
            fi = bundle.F_lam(i, lam)
            gi = bundle.grad_F_lam(i, lam)
            val = fi - 0.5 / Ll * np.dot(gi, gi)
            if val < best_val:
                best_val = val
                best_i = i

        # Gradient of u_{i*}(λ)
        Ji = bundle.grads[best_i]          # (K, d)
        JJT = Ji @ Ji.T                    # (K, K)
        JJTlam = JJT @ lam                 # (K,)
        quad = lam @ JJTlam                # scalar
        grad = bundle.fvals[best_i] - JJTlam / Ll + (quad / (2.0 * Ll**2)) * bundle.L
        return -grad  # negate for minimisation

    # Constraints and bounds for Δ_K
    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0, "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K  # small lb to keep Lλ > 0

    # Multi-start: vertices + uniform + previous best
    starts = []
    for k in range(K):
        e = np.full(K, 1e-8)
        e[k] = 1.0 - (K - 1) * 1e-8
        starts.append(e)
    starts.append(np.ones(K) / K)

    best_val = np.inf
    best_lam = starts[0]
    for lam0 in starts:
        res = sp_minimize(neg_ub, lam0, jac=neg_ub_grad, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 200})
        if res.fun < best_val:
            best_val = res.fun
            best_lam = res.x.copy()

    # Project onto simplex (enforce numerics)
    best_lam = np.maximum(best_lam, 0.0)
    best_lam /= best_lam.sum()
    return float(-best_val), best_lam


# ---------------------------------------------------------------------------
# GAP₁ = UB − LB₁:  difference-of-concave  →  multi-start local search
# ---------------------------------------------------------------------------
def _maximise_GAP(bundle: Bundle, variant: str = "lb2") -> Tuple[float, np.ndarray]:
    """Find  λ* = argmax_{λ ∈ Δ_K}  GAP(λ; B_m).

    Structure
    ---------
    GAP₁(λ) = UB(λ) − LB₁(λ) where both UB and LB₁ are concave in λ
    (Proposition 6).  So GAP₁ is a difference-of-concave (DC) function.

    DC maximisation is NP-hard in general, but here λ ∈ Δ_K has only
    K−1 degrees of freedom with K typically small (2–10).  We use
    multi-start SLSQP: each local solve finds a local maximum, and
    we take the best.

    Starting points:  K vertices + uniform + midpoints of each edge.
    For K ≤ 5 this is ≤ 16 starts, each very cheap.

    Illustrative example
    --------------------
    With K = 3, GAP is a DC function on the 2D triangle Δ_3.
    We launch ~7 local searches (3 vertices + 1 centre + 3 edge
    midpoints) and return the best.
    """
    K = bundle.K

    def neg_gap(lam):
        return -GAP(bundle, lam, variant=variant)

    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0, "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K

    # Build starting points: vertices + uniform + edge midpoints
    starts = []
    for k in range(K):
        e = np.full(K, 1e-8)
        e[k] = 1.0 - (K - 1) * 1e-8
        starts.append(e)
    starts.append(np.ones(K) / K)
    # Edge midpoints (pairs of vertices)
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            e = np.full(K, 1e-8)
            e[k1] = 0.5 - (K - 2) * 0.5e-8
            e[k2] = 0.5 - (K - 2) * 0.5e-8
            starts.append(e)

    best_val = np.inf
    best_lam = starts[0]
    for lam0 in starts:
        res = sp_minimize(neg_gap, lam0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 200})
        if res.fun < best_val:
            best_val = res.fun
            best_lam = res.x.copy()

    best_lam = np.maximum(best_lam, 0.0)
    best_lam /= best_lam.sum()
    return float(-best_val), best_lam


# ---------------------------------------------------------------------------
# GN:  non-concave  →  multi-start local search
# ---------------------------------------------------------------------------
def _maximise_GN(bundle: Bundle) -> Tuple[float, np.ndarray]:
    """Find  λ* = argmax_{λ ∈ Δ_K}  GN(λ; B_m).

    Structure
    ---------
    From Eq. (17):

        GN(λ; Bm) = (1/2)(1/µλ − 1/Lλ) · min_i ‖J_F(x_i)^T λ‖²

    where µλ = λ^T µ, Lλ = λ^T L.  The scale factor (1/(2µλ) − 1/(2Lλ))
    is convex in λ (sum of convex reciprocals of linear functions), and
    min_i ‖J^T λ‖² is concave.  Their product is neither convex nor
    concave.

    When mu is not available (generic non-convex), GN falls back to
    min_i ‖J_F(x_i)^T λ‖² which *is* concave.

    We use multi-start SLSQP as for GAP.

    Illustrative example
    --------------------
    With K = 2, the simplex is a line segment [0,1].  GN(λ) is a
    1D piecewise-rational function — a few local searches from the
    endpoints and midpoint reliably find the global max.
    """
    K = bundle.K

    def neg_gn(lam):
        return -GN(bundle, lam)

    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0,
                   "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K

    starts = []
    for k in range(K):
        e = np.full(K, 1e-8)
        e[k] = 1.0 - (K - 1) * 1e-8
        starts.append(e)
    starts.append(np.ones(K) / K)
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            e = np.full(K, 1e-8)
            e[k1] = 0.5 - (K - 2) * 0.5e-8
            e[k2] = 0.5 - (K - 2) * 0.5e-8
            starts.append(e)

    best_val = np.inf
    best_lam = starts[0]
    for lam0 in starts:
        res = sp_minimize(neg_gn, lam0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 200})
        if res.fun < best_val:
            best_val = res.fun
            best_lam = res.x.copy()

    best_lam = np.maximum(best_lam, 0.0)
    best_lam /= best_lam.sum()
    return float(-best_val), best_lam


# =====================================================================
#  Adaptive inner loop (BundleUpdate with ε/3 stopping)
# =====================================================================
def _bundle_update_adaptive(
    bundle: Bundle,
    lam: np.ndarray,
    pc_fn: Callable,
    eps_third: float,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    max_steps: int,
) -> int:
    """Run inner gradient-descent steps at fixed λ, stopping adaptively.

    From the proof of Theorem 1 (Appendix B.1):

        "At any iteration t of Algorithm 2, [...] applying Corollary 4.2
         [...] yields  PC(λ_t; B_{t+1}) ≤ ε/3."

    The theoretical IIC is a *sufficient* upper bound on the number of
    steps needed to achieve this.  Instead, we run steps one at a time
    and check the actual PC value, stopping as soon as:

        PC(λ_t; B_current) ≤ ε/3

    This is tighter: if the problem is well-conditioned at λ_t, we stop
    early; if the theoretical bound is loose, we don't over-iterate.

    A safety cap ``max_steps`` prevents infinite loops in degenerate cases.

    Parameters
    ----------
    bundle          : current bundle (modified in place).
    lam             : weight vector λ_t.
    pc_fn           : progress criterion function PC(bundle, lam) → float.
    eps_third       : the threshold ε/3.
    objectives      : list of K objective callables.
    grad_objectives : list of K gradient callables.
    max_steps       : safety cap on inner iterations.

    Returns
    -------
    Number of inner steps actually taken.

    Illustrative example
    --------------------
    Suppose PC(λ_t; B_t) = 0.5 and ε = 0.03, so ε/3 = 0.01.
    The inner loop adds gradient-descent points at λ_t:
      step 1: PC = 0.30  (still > 0.01)
      step 2: PC = 0.08  (still > 0.01)
      step 3: PC = 0.005 (≤ 0.01 → stop)
    Total: 3 oracle calls instead of the theoretical upper bound.
    """
    steps = 0
    for _ in range(max_steps):
        x_new = T_map(bundle, lam)
        bundle.add_point(x_new, objectives, grad_objectives)
        steps += 1

        # Check if PC at λ_t has dropped below ε/3
        pc_val = pc_fn(bundle, lam)
        if pc_val <= eps_third:
            break

    return steps


# =====================================================================
#  Algorithm 2  –  the main routine
# =====================================================================
def algorithm2(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    eps: float = 1e-3,
    mode: str = "gap",                # "ub", "gap", or "gn"
    mu: Optional[np.ndarray] = None,  # needed for "gap" and "ub" (PL)
    max_outer: int = 200,
    max_inner: int = 500,             # safety cap on inner steps
    verbose: bool = False,
) -> Dict:
    """Run Algorithm 2 from the paper.

    Parameters
    ----------
    K, d             : number of objectives, dimension.
    objectives       : list of K callables  f_k(x) -> float.
    grad_objectives  : list of K callables  g_k(x) -> np.ndarray shape (d,).
    L                : smoothness constants, shape (K,).
    x0               : initial point, shape (d,).
    eps              : target accuracy.
    mode             : "gap"  – strongly convex (GAP = UB − LB),
                       "ub"   – interpolation + PL (upper bound),
                       "gn"   – generic non-convex (gradient norm).
    mu               : strong convexity / PL constants, shape (K,).
    max_outer        : max outer iterations.
    max_inner        : safety cap on inner steps per outer iteration.
    verbose          : print progress.

    Returns
    -------
    dict with keys:
        "bundle"       : final Bundle object,
        "pc_history"   : list of PC*_t at each outer iteration,
        "lam_history"  : list of λ_t chosen at each iteration,
        "oracle_calls" : total number of oracle (gradient) evaluations,
        "outer_iters"  : number of outer iterations executed.
    """
    # ---- choose PC function and maximiser ----
    if mode == "gap":
        assert mu is not None
        pc_fn = GAP
        maximise_pc = _maximise_GAP
    elif mode == "ub":
        assert mu is not None
        pc_fn = UB
        maximise_pc = _maximise_UB
    elif mode == "gn":
        pc_fn = GN
        maximise_pc = _maximise_GN
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ---- initialise bundle ----
    bundle = Bundle(K=K, d=d, L=L, mu=mu)
    bundle.add_point(x0, objectives, grad_objectives)

    eps_third = eps / 3.0

    pc_history = []
    lam_history = []
    inner_steps_history = []
    oracle_calls = K  # initial point

    for t in range(max_outer):
        # Step 1: find  λ_t = argmax_{λ ∈ Δ_K}  PC(λ; B_t)
        pc_star, best_lam = maximise_pc(bundle)

        pc_history.append(pc_star)
        lam_history.append(best_lam.copy())

        if verbose:
            print(f"  outer iter {t:3d} | PC* = {pc_star:.6e} | λ = {best_lam}")

        if pc_star <= eps:
            if verbose:
                print(f"  Converged at outer iteration {t}.")
            break

        # Step 2: inner loop — add points at λ_t until PC(λ_t; B) ≤ ε/3
        steps = _bundle_update_adaptive(
            bundle, best_lam, pc_fn, eps_third,
            objectives, grad_objectives, max_inner,
        )
        inner_steps_history.append(steps)
        oracle_calls += steps * K

        if verbose:
            pc_after = pc_fn(bundle, best_lam)
            print(f"           inner steps = {steps:3d} | "
                  f"PC(λ_t; B) = {pc_after:.6e} after update")

    return {
        "bundle": bundle,
        "pc_history": pc_history,
        "lam_history": lam_history,
        "inner_steps_history": inner_steps_history,
        "oracle_calls": oracle_calls,
        "outer_iters": len(pc_history),
    }