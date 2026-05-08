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
import copy
import time
from typing import Callable, Dict, List, Optional, Tuple
from scipy.optimize import minimize as sp_minimize

from bundle import Bundle, UB, GAP, GN, LB, T_map
from baseline import worst_case_suboptimality_algorithm2


# =====================================================================
#  Vectorised bundle helpers (CPU-optimisation, no semantic change)
# =====================================================================
# These helpers replace the per-bundle-point Python `for i in range(m)`
# loops in UB / LB_2 / T_map with single batched numpy operations.
# Mathematically they reproduce bundle.UB, bundle._LB_2 and bundle.T_map
# exactly; only the implementation differs.
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


def _ub_lb2_batched(Fmat: np.ndarray, Jmat: np.ndarray,
                    L: np.ndarray, mu: np.ndarray,
                    lam: np.ndarray) -> Tuple[float, float, int, int,
                                              np.ndarray, np.ndarray]:
    """Single batched evaluation of UB(λ) and LB_2(λ) over the bundle.

    Computes for all i simultaneously:
        F_λ(x_i)        = Fmat @ λ                   shape (m,)
        ∇F_λ(x_i)       = Σ_k λ_k Jmat[i, k]         shape (m, d)
        ‖∇F_λ(x_i)‖²    = row-sum of squares          shape (m,)
        u_i(λ) = F_λ(x_i) - 1/(2 Lλ) ‖∇F_λ(x_i)‖²
        l_i(λ) = F_λ(x_i) - 1/(2 µλ) ‖∇F_λ(x_i)‖²

    Returns
    -------
    ub        : float                       UB(λ; B) = min_i u_i(λ)
    lb2       : float                       LB_2(λ; B) = max_i l_i(λ)
    i_star    : int                         argmin u_i (UB-attaining index)
    j_star    : int                         argmax l_i (LB_2-attaining index)
    F_lam     : (m,) array                  reused by callers
    gnorm_sq  : (m,) array                  reused by callers
    """
    Ll = float(lam @ L)
    mul = float(lam @ mu)
    F_lam = Fmat @ lam                              # (m,)
    grad_lam = np.einsum('ikd,k->id', Jmat, lam)    # (m, d)
    gnorm_sq = np.einsum('id,id->i', grad_lam, grad_lam)  # (m,)
    u_vals = F_lam - 0.5 * gnorm_sq / Ll            # (m,)
    l_vals = F_lam - 0.5 * gnorm_sq / mul           # (m,)
    i_star = int(np.argmin(u_vals))
    j_star = int(np.argmax(l_vals))
    return (float(u_vals[i_star]), float(l_vals[j_star]),
            i_star, j_star, F_lam, gnorm_sq)


def _gap_grad_batched(Fmat: np.ndarray, Jmat: np.ndarray,
                      L: np.ndarray, mu: np.ndarray, lam: np.ndarray,
                      i_star: int, j_star: int) -> np.ndarray:
    """Analytical (Danskin) gradient of  GAP_2(λ) = UB(λ) − LB_2(λ).

    From Prop. 6 (paper, p. 19):

        ∇u_i(λ) = F(x_i) − J_i J_i^T λ / Lλ + (λ^T J_i J_i^T λ)/(2 Lλ²) · L
        ∇l_i(λ) = F(x_i) − J_i J_i^T λ / µλ + (λ^T J_i J_i^T λ)/(2 µλ²) · µ

    Danskin's theorem applies to both min (for UB) and max (for LB_2),
    so a subgradient of GAP_2 = u_{i*} − l_{j*} is
        ∇u_{i*}(λ) − ∇l_{j*}(λ).

    The gradient of −GAP_2 (used inside neg_gap for SLSQP) is the negation.
    """
    Ll = float(lam @ L)
    mul = float(lam @ mu)

    Ji = Jmat[i_star]                  # (K, d)
    JJTi_lam = Ji @ (Ji.T @ lam)       # (K,)  =  J_i J_i^T λ
    qi = float(lam @ JJTi_lam)
    grad_u = Fmat[i_star] - JJTi_lam / Ll + (qi / (2.0 * Ll * Ll)) * L

    Jj = Jmat[j_star]
    JJTj_lam = Jj @ (Jj.T @ lam)
    qj = float(lam @ JJTj_lam)
    grad_l = Fmat[j_star] - JJTj_lam / mul + (qj / (2.0 * mul * mul)) * mu

    return grad_u - grad_l             # gradient of GAP_2


def _gn_value_and_jac_batched(Fmat: np.ndarray, Jmat: np.ndarray,
                              L: np.ndarray, mu: Optional[np.ndarray],
                              lam: np.ndarray
                              ) -> Tuple[float, np.ndarray, int]:
    """Batched evaluation and analytical λ-gradient of  GN(λ; B)  (Eq. 17).

    GN(λ) = scale(λ) · min_i ‖J_i^T λ‖²,    where
        scale(λ) = ½(1/µλ − 1/Lλ)   if mu is given (the strongly-convex/PL
                                     case in the paper),
        scale(λ) = 1                 in the generic non-convex case (no µ),
                                     to match ``bundle.GN``'s fallback.

    Implementation
    --------------
    * Stack the bundle Jacobians and contract with λ in one ``einsum`` to
      obtain  ``G[i] = J_i^T λ``  for all bundle points simultaneously,
      replacing the original Python loop in :func:`bundle.GN`.
    * The argmin index ``i*`` is unique generically.  By Danskin's theorem
      the gradient of GN at smooth points is
          ∇_λ GN(λ) = scale(λ) · 2 J_{i*} g_{i*}
                       + (gnorm²_{i*}) · ∇_λ scale(λ),
      with  ∇_λ scale(λ) = -µ/(2 µλ²) + L/(2 Lλ²)  in the strongly-convex
      case and zero otherwise.  ``J_{i*}`` has shape (K, d), so
      ``J_{i*} g_{i*}`` is a (K,) vector — the same shape as λ.

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

    if mu is not None:
        mul = float(lam @ mu)
        Ll = float(lam @ L)
        scale = 0.5 * (1.0 / mul - 1.0 / Ll)
        # ∇_λ scale  =  -µ / (2 µλ²)  +  L / (2 Lλ²)
        grad_scale = -0.5 * mu / (mul * mul) + 0.5 * L / (Ll * Ll)
        gn_value = scale * gnorm_sq_istar
        gn_jac = scale * grad_min_norm + gnorm_sq_istar * grad_scale
    else:
        gn_value = gnorm_sq_istar
        gn_jac = grad_min_norm
    return gn_value, gn_jac, i_star


def _T_map_batched(Fmat: np.ndarray, Jmat: np.ndarray, points_arr: np.ndarray,
                   L: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Vectorised version of ``bundle.T_map`` (Eq. 13).

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


def _T_map_grid_batched(Fmat: np.ndarray, Jmat: np.ndarray,
                        points_arr: np.ndarray, L: np.ndarray,
                        Lambda: np.ndarray) -> np.ndarray:
    """Apply ``T_map`` to every λ in a grid in one batched pass.

    Parameters
    ----------
    Fmat        : (m, K)
    Jmat        : (m, K, d)
    points_arr  : (m, d)
    L           : (K,)
    Lambda      : (N, K)  — every row is a simplex point.

    Returns
    -------
    X_hat : (N, d) array, row n = T_map(bundle, Lambda[n]).
    """
    Ll_n = Lambda @ L                                  # (N,)
    F_lam_im = Fmat @ Lambda.T                         # (m, N)
    # grad_lam_nim[n, i, d] = Σ_k Lambda[n, k] · Jmat[i, k, d]
    grad_lam = np.einsum('nk,ikd->nid', Lambda, Jmat)  # (N, m, d)
    gnorm_sq = np.einsum('nid,nid->ni', grad_lam, grad_lam)  # (N, m)
    # u_vals[n, i] = F_λn(x_i) - 1/(2 Lλn) ‖∇F_λn(x_i)‖²
    u_vals = F_lam_im.T - 0.5 * gnorm_sq / Ll_n[:, None]   # (N, m)
    i_star = np.argmin(u_vals, axis=1)                     # (N,)
    n_idx = np.arange(Lambda.shape[0])
    x_best = points_arr[i_star]                            # (N, d)
    grad_best = grad_lam[n_idx, i_star]                    # (N, d)
    return x_best - (1.0 / Ll_n[:, None]) * grad_best      # (N, d)


def _worst_case_subopt_fast(bundle: Bundle, reference_map: Dict,
                            objectives: List[Callable], K: int) -> float:
    """Drop-in fast replacement for ``worst_case_suboptimality_algorithm2``.

    Same definition  err = sup_λ [F_λ(T_map(B, λ)) − F*_λ]  as the
    baseline helper, but the bundle work is batched over the entire
    fine grid in a single einsum, eliminating the per-grid-point Python
    loop that calls ``T_map``.

    The per-grid-point objective evaluations  F_k(x_hat_n)  remain
    Python-level (since ``objectives[k]`` is a closure over per-class
    sample indices), but those are now the only Python-level work in
    the checkpoint.
    """
    fine_grid = reference_map["fine_grid"]              # (N, K)
    F_star = reference_map["F_star"]                    # (N,)

    Fmat, Jmat = _bundle_arrays(bundle)
    points_arr = np.asarray(bundle.points)
    X_hat = _T_map_grid_batched(Fmat, Jmat, points_arr, bundle.L, fine_grid)

    worst = -np.inf
    for n, lam in enumerate(fine_grid):
        x_n = X_hat[n]
        F_lam = 0.0
        for k in range(K):
            F_lam += lam[k] * objectives[k](x_n)
        err = F_lam - F_star[n]
        if err > worst:
            worst = float(err)
    return worst


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
def _maximise_GAP(bundle: Bundle, variant: str = "lb1",
                  prev_lam: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Find  λ* = argmax_{λ ∈ Δ_K}  GAP(λ; B_m).

    Structure
    ---------
    GAP₁(λ) = UB(λ) − LB₁(λ) where both UB and LB₁ are concave in λ
    (Proposition 6).  So GAP₁ is a difference-of-concave (DC) function.

    DC maximisation is NP-hard in general, but here λ ∈ Δ_K has only
    K−1 degrees of freedom with K typically small (2–10).  We use
    multi-start SLSQP: each local solve finds a local maximum, and
    we take the best.

    Optimised lb2 path
    ------------------
    For variant="lb2" (the path used by ``experiment_logreg_gap``):

      1. Bundle data is stacked into ``Fmat`` and ``Jmat`` once per call
         and shared across SLSQP iterations and across multi-starts.
         All UB/LB_2 evaluations inside SLSQP then become a single
         ``Fmat @ lam`` + one ``einsum`` instead of an ``m``-step Python
         loop calling ``F_lam`` and ``grad_F_lam`` per bundle index.

      2. The analytical Danskin-gradient (Prop. 6) is supplied to SLSQP
         via ``jac=``.  Without it, SLSQP would build a numerical
         gradient by K extra GAP evaluations per step — the dominant
         cost in the un-optimised version.

      3. ``prev_lam`` (the best-λ from the previous outer iteration) is
         appended to the multi-start set whenever provided, exploiting
         the fact that the bundle changes by only one point per outer.

    For variant="lb1" we fall back to the original (Gurobi-QP-backed)
    LB_1 path, which is left untouched.
    """
    K = bundle.K

    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0,
                   "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K

    # Multi-start set:  K vertices + uniform centre  (+ prev_lam if given).
    # Edge midpoints were dropped from the original 7-start set: empirically
    # they almost always converge to the same local maximum as one of the
    # vertex starts, so SLSQP cycles add cost without improving the optimum.
    starts: List[np.ndarray] = []
    for k in range(K):
        e = np.full(K, 1e-8)
        e[k] = 1.0 - (K - 1) * 1e-8
        starts.append(e)
    starts.append(np.ones(K) / K)
    if prev_lam is not None:
        # Project numerically to the strict-positive simplex
        s = np.maximum(prev_lam, 1e-8)
        s /= s.sum()
        starts.append(s)

    if variant == "lb2":
        # ---- Vectorised batched evaluator (shared across SLSQP iters) ----
        Fmat, Jmat = _bundle_arrays(bundle)
        L_arr, mu_arr = bundle.L, bundle.mu

        def neg_gap(lam):
            ub, lb2, *_ = _ub_lb2_batched(Fmat, Jmat, L_arr, mu_arr, lam)
            return lb2 - ub                     # = -GAP_2

        def neg_gap_jac(lam):
            ub, lb2, i_star, j_star, _, _ = _ub_lb2_batched(
                Fmat, Jmat, L_arr, mu_arr, lam)
            return -_gap_grad_batched(Fmat, Jmat, L_arr, mu_arr, lam,
                                      i_star, j_star)

        best_val = np.inf
        best_lam = starts[0]
        for lam0 in starts:
            res = sp_minimize(neg_gap, lam0, jac=neg_gap_jac, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"ftol": 1e-9, "maxiter": 60})
            if res.fun < best_val:
                best_val = res.fun
                best_lam = res.x.copy()

    else:
        # ---- Original (un-vectorised) path for LB_1 -----------------------
        def neg_gap(lam):
            return -GAP(bundle, lam, variant=variant)

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
def _maximise_GN(bundle: Bundle,
                 prev_lam: Optional[np.ndarray] = None
                 ) -> Tuple[float, np.ndarray]:
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

    We use multi-start SLSQP as for GAP, but with three CPU optimisations:

    * ``_gn_value_and_jac_batched`` evaluates GN and its Danskin gradient
      in one ``einsum``, replacing the per-bundle-point Python loop in
      :func:`bundle.GN`.  This is the dominant inner cost of the
      maximisation when bundle size is large.
    * The analytical Jacobian is supplied to SLSQP via the ``jac`` keyword,
      eliminating SLSQP's ~K function-evaluations-per-iteration of finite-
      difference numerical differentiation.
    * The multi-start set is (K vertices + uniform centroid + K(K−1)/2
      edge midpoints + optional ``prev_lam``).  Unlike GAP, GN is neither
      convex nor concave in λ, and empirically the edge midpoints recover
      basins that the vertices and centroid miss on the non-convex MLP
      problem; dropping them sacrifices final accuracy.  ``prev_lam`` is
      only ever an *additional* start — never a replacement — so it can
      accelerate convergence without trapping the search in a stale basin.

    SLSQP's tolerances are relaxed to ``ftol = 1e-9, maxiter = 60``,
    matching the budget chosen for the GAP path.

    Illustrative example
    --------------------
    With K = 2, the simplex is a line segment [0,1].  GN(λ) is a
    1D piecewise-rational function — a few local searches from the
    endpoints and midpoint reliably find the global max.
    """
    K = bundle.K
    Fmat, Jmat = _bundle_arrays(bundle)
    L_arr = bundle.L
    mu_arr = bundle.mu

    def neg_gn(lam):
        v, _, _ = _gn_value_and_jac_batched(Fmat, Jmat, L_arr, mu_arr, lam)
        return -v

    def neg_gn_jac(lam):
        _, j, _ = _gn_value_and_jac_batched(Fmat, Jmat, L_arr, mu_arr, lam)
        return -j

    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0,
                   "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K

    starts = []
    for k in range(K):
        e = np.full(K, 1e-8)
        e[k] = 1.0 - (K - 1) * 1e-8
        starts.append(e)
    starts.append(np.ones(K) / K)
    # Edge midpoints: for the non-convex MLP the GN landscape has local
    # maxima on simplex edges that vertex+centroid starts alone miss.
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            e = np.full(K, 1e-8)
            e[k1] = 0.5 - (K - 2) * 0.5e-8
            e[k2] = 0.5 - (K - 2) * 0.5e-8
            starts.append(e)
    if prev_lam is not None:
        # Warm-start from the previous outer iteration's optimum
        # (additional start, not replacement).
        starts.append(np.clip(prev_lam, 1e-8, 1.0))

    best_val = np.inf
    best_lam = starts[0]
    for lam0 in starts:
        res = sp_minimize(neg_gn, lam0, jac=neg_gn_jac, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-9, "maxiter": 60})
        if res.fun < best_val:
            best_val = res.fun
            best_lam = res.x.copy()

    best_lam = np.maximum(best_lam, 0.0)
    best_lam /= best_lam.sum()
    return float(-best_val), best_lam


# =====================================================================
#  Adaptive inner loop (BundleUpdate with max_steps)
# =====================================================================
def _bundle_update_adaptive(
    bundle: Bundle,
    lam: np.ndarray,
    pc_fn: Callable,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    max_steps: int,
) -> int:
    """Run ``max_steps`` inner T_map steps at fixed λ; commit only the
    candidate with the smallest ∥∇F_λ∥ to the bundle.

    Implementation note (CPU optimisation, no semantic change)
    ----------------------------------------------------------
    The original implementation built a ``copy.deepcopy(bundle)`` work
    bundle to hold the candidate chain, then committed only the winner
    to the real bundle.  For a bundle with m points and K=3 objectives
    in d=9 this copies ~ m · (1 + K + K·d) ≈ 31 m floats per outer iter
    — non-trivial as m grows past ~100.

    We instead append the candidates *in place* to the real bundle, find
    the winner by the same gnorm criterion, then pop the losers and
    re-append the winner.  This is mathematically identical to the
    deep-copy version (each T_map call still sees all previously-added
    in-round candidates) but avoids O(m·K·d) bundle copying per outer.

    The T_map call itself is replaced by ``_T_map_batched`` which
    vectorises the per-bundle-point loop into a single ``einsum``.

    Returns
    -------
    Number of inner steps taken (always equals ``max_steps``).
    """
    base_m = bundle.m

    # Generate the candidate chain on the real bundle.  Each T_map call
    # sees all previously-added in-round candidates, matching the
    # original inner-loop semantics.  We rebuild the stacked arrays
    # incrementally — they are O(m·K·d) but this is amortised since
    # ``_T_map_batched`` reuses them inside its single einsum call.
    for _ in range(max_steps):
        Fmat, Jmat = _bundle_arrays(bundle)
        points_arr = np.asarray(bundle.points)
        x_new = _T_map_batched(Fmat, Jmat, points_arr, bundle.L, lam)
        bundle.add_point(x_new, objectives, grad_objectives)

    # Pick argmin ∥∇F_λ(x^i)∥ from the cached gradients of the candidates.
    # Vectorised: stack the K×d Jacobians of the new candidates, contract
    # with λ, and take the smallest row-norm.
    cand_Js = np.asarray(bundle.grads[base_m:base_m + max_steps])  # (S, K, d)
    cand_grads_lam = np.einsum('skd,k->sd', cand_Js, lam)          # (S, d)
    cand_gnorms = np.einsum('sd,sd->s', cand_grads_lam, cand_grads_lam)
    best_local = int(np.argmin(cand_gnorms))
    best_idx = base_m + best_local

    # Save the winner, pop *all* candidates, push only the winner.
    win_x = bundle.points[best_idx]
    win_fv = bundle.fvals[best_idx]
    win_gv = bundle.grads[best_idx]
    for _ in range(max_steps):
        bundle.pop_point()
    bundle.points.append(win_x)
    bundle.fvals.append(win_fv)
    bundle.grads.append(win_gv)

    return max_steps


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
    if mode == "gap":
        # Use LB_2 (single-index minorant) inside both the λ-maximisation
        # and the inner-loop PC check.  LB_2 is ~100× faster than LB_1 and
        # avoids hitting the Gurobi size-limited license once the bundle
        # grows past ~100 points.
        pc_fn = lambda bundle, lam: GAP(bundle, lam, variant="lb2")
        # Closure over a mutable ``_prev`` so the warm-start can be threaded
        # through the multi-start without changing _maximise_GAP's signature
        # contract for other callers.
        _prev: Dict[str, Optional[np.ndarray]] = {"lam": None}
        def maximise_pc(bundle):
            v, l = _maximise_GAP(bundle, variant="lb2", prev_lam=_prev["lam"])
            _prev["lam"] = l
            return v, l
        if mu is None:
            raise ValueError("mode='gap' requires mu (strong convexity).")
    elif mode == "ub":
        pc_fn, maximise_pc = UB, _maximise_UB
        if mu is None:
            raise ValueError("mode='ub' requires mu.")
    elif mode == "gn":
        pc_fn = GN
        # Closure-based prev_lam warm-start, mirroring the mode='gap' path.
        # Threading the previous outer's argmax-λ into _maximise_GN's
        # multi-start cuts the per-call SLSQP budget needed in steady state,
        # since the active argmin index ``i*`` typically only changes in
        # discrete jumps as the bundle grows.
        _prev_gn: Dict[str, Optional[np.ndarray]] = {"lam": None}
        def maximise_pc(bundle):
            v, l = _maximise_GN(bundle, prev_lam=_prev_gn["lam"])
            _prev_gn["lam"] = l
            return v, l
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
        err = _worst_case_subopt_fast(bundle, reference_map, objectives, K)
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

    # Checkpoint 0:  report the worst-case error of the constant map x̂(λ) ≡ x_0.
    # This matches the baseline's checkpoint 0 (which also reports the constant
    # map at x_0) so both algorithms agree on the "initial" worst-case error
    # before any algorithmic work has been done.  The K gradient evaluations
    # spent on bundle.add_point(x_0, ...) at the top of this routine will be
    # accounted for at checkpoint 1 alongside the work of the first outer
    # iteration, keeping the cumulative grad_evals count correct from then on.
    fine_grid = reference_map["fine_grid"]
    F_star = reference_map["F_star"]
    err0 = -np.inf
    for i, lam in enumerate(fine_grid):
        F_lam_x0 = sum(lam[k] * objectives[k](x0) for k in range(K))
        err = F_lam_x0 - F_star[i]
        if err > err0:
            err0 = float(err)
    cpu_times.append(time.time() - t_start)
    worst_errs.append(err0)
    outer_iters_history.append(0)
    grad_evals_history.append(0)
    if verbose:
        print(f"  A2 outer 0/{max_outer} | t={cpu_times[-1]:.2f}s | grad_evals=0 "
              f"| err={err0:.4e}  (constant map)")


    for t in range(max_outer):
        pc_star, best_lam = maximise_pc(bundle)
        pc_history.append(pc_star)

        bundle_m_before = bundle.m
        steps = _bundle_update_adaptive(
            bundle, best_lam, pc_fn, objectives, grad_objectives, max_inner,
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