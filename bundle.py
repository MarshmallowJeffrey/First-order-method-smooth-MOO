"""
bundle.py  –  Core bundle data structure
==========================================================================

This module implements the *bundle* B_m from Section 3 of the paper:

    B_m = { (x_i, F_1(x_i), ..., F_K(x_i), ∇F_1(x_i), ..., ∇F_K(x_i)) }_{i=1}^m

All quantities use the *λ-dependent* smoothness constants  Lλ = Σ_k λ_k L_k.

Illustrative example
--------------------
Suppose we have K = 2 objectives on R^2:

    F_1(x) = 0.5 * x^T A_1 x,   F_2(x) = 0.5 * x^T A_2 x

with A_1 = diag(2, 10) (so L_1=10) and A_2 = diag(4, 6) (L_2=6).

For λ = (0.5, 0.5) we get  Lλ = 8.
Starting from x_1 = (1, 1), the bundle stores F_k(x_1) and ∇F_k(x_1),
and we can immediately evaluate GN(λ; B_1).

After adding x_2 = T(λ; B_1)  (one gradient descent step picking the best
bundle point), each progress criterion is guaranteed to decrease or stay the
same (Assumption 3.1, global monotonicity).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable



# ---------------------------------------------------------------------------
# Bundle data structure
# ---------------------------------------------------------------------------
@dataclass
class Bundle:
    """Stores zeroth- and first-order oracle information at visited points.

    Attributes
    ----------
    K : int
        Number of objective functions.
    d : int
        Dimension of the decision variable x ∈ R^d.
    points : list[np.ndarray]
        List of iterates  x_1, …, x_m  (each shape (d,)).
    fvals : list[np.ndarray]
        fvals[i] = (F_1(x_i), …, F_K(x_i))  shape (K,).
    grads : list[np.ndarray]
        grads[i] = J_F(x_i)  the K×d Jacobian at x_i.
    L : np.ndarray
        Smoothness constants (L_1, …, L_K), shape (K,).
    """

    K: int
    d: int
    L: np.ndarray                       # shape (K,)
    points: List[np.ndarray] = field(default_factory=list)
    fvals: List[np.ndarray] = field(default_factory=list)
    grads: List[np.ndarray] = field(default_factory=list)

    # ---- helpers ----
    @property
    def m(self) -> int:
        """Current bundle size."""
        return len(self.points)

    def L_lam(self, lam: np.ndarray) -> float:
        """Lλ = Σ_k λ_k L_k."""
        return float(lam @ self.L)

    def F_lam(self, idx: int, lam: np.ndarray) -> float:
        """Fλ(x_i) = λ^T F(x_i)."""
        return float(lam @ self.fvals[idx])

    def grad_F_lam(self, idx: int, lam: np.ndarray) -> np.ndarray:
        """∇Fλ(x_i) = J_F(x_i)^T λ,  shape (d,)."""
        return self.grads[idx].T @ lam   # (d, K) @ (K,) = (d,)

    def add_point(self, x: np.ndarray, objectives: List[Callable], grad_objectives: List[Callable],
                  joint_oracle: Optional[Callable] = None):
        """Evaluate all objectives and gradients at x and append to bundle.

        Parameters
        ----------
        x                 : iterate at which to evaluate.
        objectives        : list of K F_i closures (used when ``joint_oracle`` is None).
        grad_objectives   : list of K ∇F_i closures (used when ``joint_oracle`` is None).
        joint_oracle      : optional fused oracle ``θ → (fv, gv)`` returning
                            ``(K,)`` and ``(K, d)`` arrays in a single pass.  When
                            provided, eliminates the redundant forward-pass work that
                            otherwise occurs when ``F_i`` and ``∇F_i`` are called
                            sequentially.  See ``make_mlp_nonconvex`` /
                            ``make_logreg_strongly_convex`` for fused oracles.
        """
        if joint_oracle is not None:
            fv, gv = joint_oracle(x)
        else:
            fv = np.array([f(x) for f in objectives])
            gv = np.vstack([g(x) for g in grad_objectives])   # (K, d)
        self.points.append(x.copy())
        self.fvals.append(fv)
        self.grads.append(gv)

    def pop_point(self):
        """Pop the last element out of the oracle."""
        self.points.pop()
        self.fvals.pop()
        self.grads.pop()


# ---------------------------------------------------------------------------
# Progress criteria  (Section 5.2)
# ---------------------------------------------------------------------------
def GN(bundle: Bundle, lam: np.ndarray) -> float:
    """Gradient-norm progress criterion  (Eq. 17).
        GN(λ; B_m) = min_i  ‖∇Fλ(x_i)‖².
    """
    min_gnorm_sq = np.inf
    for i in range(bundle.m):
        gi = bundle.grad_F_lam(i, lam)
        gnorm_sq = float(np.dot(gi, gi))
        if gnorm_sq < min_gnorm_sq:
            min_gnorm_sq = gnorm_sq
    return min_gnorm_sq


# ---------------------------------------------------------------------------
# Mapping  T(λ; B_m)  –  the new point to add  (Eq. 13)
# ---------------------------------------------------------------------------
def T_map(bundle: Bundle, lam: np.ndarray) -> np.ndarray:
    """Compute T(λ; B_m) = x_{i*} − (1/Lλ) ∇Fλ(x_{i*})

    where  i* = argmin_{i∈[m]}{ Fλ(x_i) − 1/(2Lλ) ‖∇Fλ(x_i)‖² }.
    (Eq. 13 – one step of gradient descent from the best bundle point.)
    """
    Ll = bundle.L_lam(lam)
    best_val = np.inf
    best_i = 0
    for i in range(bundle.m):
        fi = bundle.F_lam(i, lam)
        gi = bundle.grad_F_lam(i, lam)
        val = fi - 0.5 / Ll * np.dot(gi, gi)
        if val < best_val:
            best_val = val
            best_i = i
    xi = bundle.points[best_i]
    gi = bundle.grad_F_lam(best_i, lam)
    return xi - (1.0 / Ll) * gi