"""Chebyshev-center direction selection (Yoon, Bae, Ho-Nguyen & Lee, 2026).

Implements the dual solvers and update-direction machinery of
"Chebyshev Center-Based Direction Selection for Multi-Objective
Optimization and Training PINNs" (arXiv:2605.09975).  These are used as
the inner-loop subroutine of ``algorithm4_progressive`` in algorithm.py,
which substitutes Algorithm 4 for the BundleUpdate inner loop of
``algorithm2_progressive`` while keeping the same outer PC-maximisation
and worst-case-error checkpointing.

We restrict to the Euclidean case  p = q = 2  throughout (the setting of
the paper's experiments).  For m = 2 and m = 3 the dual (D)

        min_{alpha in simplex}  || sum_i alpha_i ghat_i ||_2

is solved in closed form (Algorithms 2 and 3 of the paper); for m >= 4 we
solve it with a projected-gradient Frank-Wolfe routine (the paper uses
Frank-Wolfe for the generic case, as in MGDA).

Symbol map between the paper and this codebase
----------------------------------------------
* their "m loss terms"              == our K objectives
* their gradient g_i                == our  nabla F_k(theta)  (one per objective)
* their normalized ghat_i           == g_i / ||g_i||_2
* their dual variable alpha in simplex_m  == weights over the K objectives
* their primal direction v          == ascent-normalized combination (recovered)
* their final update direction d_t  == (sum_i g_i^T v) v   (adaptive scalar)

NOTE: alpha here is the *Chebyshev dual weight* and is internal to the
direction solver.  It is NOT the outer-loop weighting lambda of
algorithm2_progressive.  In algorithm4_progressive the inner loop solves a
FIXED-lambda problem  min_theta sum_k lambda_k F_k(theta);  the gradients
fed to this module are the lambda-weighted per-objective gradients (see
algorithm.py for how they are formed).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
#  Exact dual solver, m = 2  (Algorithm 2)
# ---------------------------------------------------------------------------
def dual_solver_m2(ghat: np.ndarray) -> np.ndarray:
    """Exact dual weights for two ell2-normalized gradients.

    Algorithm 2: for ell2-normalized ghat_1, ghat_2 the minimizer of
    ||alpha ghat_1 + (1-alpha) ghat_2||_2 over alpha in [0,1] is alpha = 1/2
    whenever ghat_1 != ghat_2; if ghat_1 == ghat_2 every alpha gives the
    same direction, so 1/2 remains valid.  Hence alpha* = (1/2, 1/2)
    unconditionally.

    Parameters
    ----------
    ghat : (2, d) array of ell2-normalized gradients.

    Returns
    -------
    alpha : (2,) simplex weights, always (0.5, 0.5).
    """
    return np.array([0.5, 0.5], dtype=np.float64)


# ---------------------------------------------------------------------------
#  Exact dual solver, m = 3  (Algorithm 3)
# ---------------------------------------------------------------------------
def dual_solver_m3(ghat: np.ndarray, tau: float = 1e-9, eps: float = 1e-12) -> np.ndarray:
    """Exact dual weights for three ell2-normalized gradients.

    Implements Algorithm 3 of the paper.  The minimizer of
    ||sum_i alpha_i ghat_i||_2 over the 3-simplex is one of:
      (1) a zero-norm convex combination   (0 in conv{ghat_1,ghat_2,ghat_3}),
      (2) the interior KKT point  alpha = G^+ 1 / (1^T G^+ 1)  if feasible,
      (3) an edge midpoint  (1/2, 1/2)  on the pair with smallest Gram entry.

    Parameters
    ----------
    ghat : (3, d) array of ell2-normalized gradients.
    tau  : feasibility tolerance for the interior-point nonnegativity test.
    eps  : numerical tolerance for the  1^T G^+ 1 > 0  test.

    Returns
    -------
    alpha : (3,) simplex weights.
    """
    G = ghat @ ghat.T                       # (3, 3) Gram matrix, G_ij = ghat_i . ghat_j
    G = 0.5 * (G + G.T)

    # --- Case 1: is 0 in conv{ghat_i}?  Solve G alpha = 0 on the simplex. ---
    alpha0 = _zero_in_hull(G, eps=eps)
    if alpha0 is not None:
        return alpha0

    # --- Case 2: interior KKT candidate  alpha_tilde = G^+ 1 / (1^T G^+ 1) ---
    ones = np.ones(3)
    Gpinv = np.linalg.pinv(G)
    z = Gpinv @ ones
    c = float(ones @ z)
    if c > eps:
        alpha_tilde = z / c
        if np.all(alpha_tilde >= -tau):
            alpha_tilde = np.clip(alpha_tilde, 0.0, None)
            s = alpha_tilde.sum()
            if s > eps:
                return alpha_tilde / s

    # --- Case 3: edge midpoint on the pair with smallest Gram entry. ---
    # Smallest G_ij over i<j (most "opposed" pair) -> midpoint is the
    # min-norm point on that edge for unit-normalized gradients.
    best_pair = (0, 1)
    best_val = np.inf
    for i in range(3):
        for j in range(i + 1, 3):
            if G[i, j] < best_val:
                best_val = G[i, j]
                best_pair = (i, j)
    alpha = np.zeros(3, dtype=np.float64)
    alpha[best_pair[0]] = 0.5
    alpha[best_pair[1]] = 0.5
    return alpha


def _zero_in_hull(G: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    """Return alpha in simplex_3 with G alpha = 0 if one exists, else None.

    For unit-normalized gradients, 0 lies in conv{ghat_i} iff the min-norm
    convex combination has zero norm, i.e. min_{alpha in simplex} alpha^T G alpha = 0.
    We detect this by checking whether the unconstrained min-norm solution
    on the simplex achieves (near) zero objective.  We test the interior
    stationary point and the edges; if any feasible simplex point yields
    alpha^T G alpha ~ 0, we return it.
    """
    # Quick reject: if all Gram entries are positive and bounded away from
    # zero, the gradients are within a halfspace and 0 is not in the hull.
    # We still do the full check for safety.
    ones = np.ones(3)
    # Interior stationary point of alpha^T G alpha on the affine hull
    # {sum alpha = 1}:  alpha = G^+ 1 / (1^T G^+ 1).
    Gpinv = np.linalg.pinv(G)
    z = Gpinv @ ones
    c = float(ones @ z)
    candidates = []
    if abs(c) > eps:
        a = z / c
        candidates.append(a)
    # Edge minimizers and vertices.
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        candidates.append(e)
    for i in range(3):
        for j in range(i + 1, 3):
            # min of quadratic on edge between i and j
            gii, gjj, gij = G[i, i], G[j, j], G[i, j]
            denom = gii + gjj - 2.0 * gij
            if abs(denom) > eps:
                t = (gjj - gij) / denom
                t = min(max(t, 0.0), 1.0)
            else:
                t = 0.5
            a = np.zeros(3)
            a[i] = t
            a[j] = 1.0 - t
            candidates.append(a)
    for a in candidates:
        if np.all(a >= -1e-12):
            a = np.clip(a, 0.0, None)
            s = a.sum()
            if s <= eps:
                continue
            a = a / s
            val = float(a @ G @ a)
            if val <= 1e-10:
                return a
    return None


# ---------------------------------------------------------------------------
#  Generic Frank-Wolfe dual solver, m >= 4
# ---------------------------------------------------------------------------
def dual_solver_fw(ghat: np.ndarray, max_iter: int = 50, tol: float = 1e-9) -> np.ndarray:
    """Frank-Wolfe minimizer of ||sum_i alpha_i ghat_i||_2 over the simplex.

    Used for m >= 4, where no closed form is available (the paper also
    falls back to Frank-Wolfe in this regime, as for MGDA).  Minimizes the
    smooth convex objective  phi(alpha) = alpha^T G alpha  with G the Gram
    matrix of the normalized gradients; the linear-minimization oracle over
    the simplex is a vertex (argmin of the gradient), giving the standard
    MGDA-style FW iteration with the 2/(t+2) step size.

    Parameters
    ----------
    ghat     : (m, d) array of ell2-normalized gradients.
    max_iter : Frank-Wolfe iteration cap.
    tol      : stop when the FW duality gap is below this.

    Returns
    -------
    alpha : (m,) simplex weights.
    """
    m = ghat.shape[0]
    G = ghat @ ghat.T                       # (m, m)
    G = 0.5 * (G + G.T)
    alpha = np.full(m, 1.0 / m, dtype=np.float64)
    for t in range(max_iter):
        grad = 2.0 * (G @ alpha)            # gradient of alpha^T G alpha
        s_idx = int(np.argmin(grad))        # LMO over simplex -> a vertex
        d_dir = -alpha.copy()
        d_dir[s_idx] += 1.0                 # move toward vertex s
        gap = float(-grad @ d_dir)          # FW gap = -<grad, s - alpha>
        if gap <= tol:
            break
        gamma = 2.0 / (t + 2.0)
        alpha = alpha + gamma * d_dir
    return alpha


# ---------------------------------------------------------------------------
#  Dispatch + direction recovery + final scaled direction
# ---------------------------------------------------------------------------
def solve_dual(ghat: np.ndarray) -> np.ndarray:
    """Solve the ell2 Chebyshev dual (D) for m normalized gradients.

    Dispatches to the exact solver for m in {2, 3} and to Frank-Wolfe
    otherwise.  ``ghat`` must already be ell2-normalized, shape (m, d).
    """
    m = ghat.shape[0]
    if m == 1:
        return np.array([1.0], dtype=np.float64)
    if m == 2:
        return dual_solver_m2(ghat)
    if m == 3:
        return dual_solver_m3(ghat)
    return dual_solver_fw(ghat)


def chebyshev_direction(grads: np.ndarray, eps_norm: float = 1e-12
                        ) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute the Algorithm-4 update direction from raw per-objective gradients.

    Steps (Euclidean p = q = 2 case of Algorithm 1 / Algorithm 4):
      1. ell2-normalize each gradient:  ghat_i = g_i / ||g_i||_2.
      2. Solve the dual (D) for alpha in the simplex (closed form for m<=3).
      3. Form  w = sum_i alpha_i ghat_i.
      4. Recover primal direction  v = w / ||w||_2   (p = 2 case of Prop. 3.3).
      5. Adaptive scalar:  d = (sum_i g_i^T v) * v.

    Parameters
    ----------
    grads    : (m, d) array of raw per-objective gradients g_i.
    eps_norm : gradients with ell2-norm below this are treated as zero and
               dropped from the normalization (their normalized direction
               is undefined); if *all* gradients are near-zero the point is
               already stationary and we return a zero direction.

    Returns
    -------
    d        : (d,) final update direction (descent direction:  theta <- theta - eta d).
    w_norm   : ||w||_2, the Pareto-stationarity surrogate from the paper.
    alpha    : (m,) dual weights over the (kept) objectives.
    """
    m, d = grads.shape
    norms = np.linalg.norm(grads, axis=1)            # (m,)
    keep = norms > eps_norm
    if not np.any(keep):
        # All gradients ~0: stationary point, no movement.
        return np.zeros(d, dtype=np.float64), 0.0, np.full(m, 1.0 / m)

    ghat_kept = grads[keep] / norms[keep][:, None]   # (m_kept, d)
    alpha_kept = solve_dual(ghat_kept)               # (m_kept,)
    w = alpha_kept @ ghat_kept                        # (d,)
    w_norm = float(np.linalg.norm(w))

    if w_norm <= eps_norm:
        # w ~ 0  =>   epsilon-Pareto-stationary; no usable direction.
        alpha_full = np.zeros(m, dtype=np.float64)
        alpha_full[keep] = alpha_kept
        return np.zeros(d, dtype=np.float64), w_norm, alpha_full

    v = w / w_norm                                    # primal direction, ||v||_2 = 1
    # Adaptive scalar: sum over ALL objectives of g_i^T v (uses raw g_i).
    scalar = float(np.sum(grads @ v))                 # sum_i g_i^T v
    d_dir = scalar * v

    alpha_full = np.zeros(m, dtype=np.float64)
    alpha_full[keep] = alpha_kept
    return d_dir, w_norm, alpha_full
