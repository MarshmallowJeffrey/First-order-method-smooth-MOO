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
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Callable


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def validate_problem_inputs(
    K: int,
    d: int,
    L: np.ndarray,
    x0: np.ndarray,
    objectives: List[Callable],
    grad_objectives: List[Callable],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shared smooth-MOO inputs and return normalised arrays."""
    K = _positive_int("K", K)
    d = _positive_int("d", d)
    if len(objectives) != K:
        raise ValueError(f"objectives must contain K={K} callables; got {len(objectives)}.")
    if len(grad_objectives) != K:
        raise ValueError(
            f"grad_objectives must contain K={K} callables; got {len(grad_objectives)}."
        )

    L_arr = np.asarray(L, dtype=float)
    if L_arr.shape != (K,):
        raise ValueError(f"L must have shape ({K},); got {L_arr.shape}.")
    if np.any(~np.isfinite(L_arr)) or np.any(L_arr <= 0.0):
        raise ValueError("L must contain finite, strictly positive values.")

    x_arr = np.asarray(x0, dtype=float)
    if x_arr.shape != (d,):
        raise ValueError(f"x0 must have shape ({d},); got {x_arr.shape}.")
    if np.any(~np.isfinite(x_arr)):
        raise ValueError("x0 must contain only finite values.")
    return L_arr.copy(), x_arr.copy()


def validate_oracle_output(
    fvals: np.ndarray,
    grads: np.ndarray,
    K: int,
    d: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one all-objective oracle result."""
    fv = np.asarray(fvals, dtype=float)
    gv = np.asarray(grads, dtype=float)
    if fv.shape != (K,):
        raise ValueError(f"Oracle function values must have shape ({K},); got {fv.shape}.")
    if gv.shape != (K, d):
        raise ValueError(f"Oracle gradients must have shape ({K}, {d}); got {gv.shape}.")
    if np.any(~np.isfinite(fv)):
        raise ValueError("Oracle function values must be finite.")
    if np.any(~np.isfinite(gv)):
        raise ValueError("Oracle gradients must be finite.")
    return fv, gv


def prefer_fused_joint_oracle(
    joint_oracle: Optional[Callable],
) -> Optional[Callable]:
    """Prefer ``joint_oracle.fused`` with a persistent standard fallback.

    If no fused implementation is exposed, return the supplied oracle as-is.
    After a fused implementation raises ``MemoryError`` or
    ``NotImplementedError``, all subsequent calls use the standard oracle.
    """
    if joint_oracle is None:
        return None

    fused_oracle = getattr(joint_oracle, "fused", None)
    if fused_oracle is None:
        return joint_oracle

    use_fused = True

    def preferred_oracle(x):
        nonlocal use_fused
        if use_fused:
            try:
                return fused_oracle(x)
            except (MemoryError, NotImplementedError) as exc:
                use_fused = False
                warnings.warn(
                    f"Fused joint oracle unavailable ({exc}); "
                    "falling back to the standard joint oracle.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return joint_oracle(x)

    return preferred_oracle



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

    def __post_init__(self) -> None:
        self.K = _positive_int("K", self.K)
        self.d = _positive_int("d", self.d)
        L_arr = np.asarray(self.L, dtype=float)
        if L_arr.shape != (self.K,):
            raise ValueError(f"L must have shape ({self.K},); got {L_arr.shape}.")
        if np.any(~np.isfinite(L_arr)) or np.any(L_arr <= 0.0):
            raise ValueError("L must contain finite, strictly positive values.")
        if not (len(self.points) == len(self.fvals) == len(self.grads)):
            raise ValueError("points, fvals, and grads must have identical lengths.")
        self.L = L_arr.copy()

        for index in range(len(self.points)):
            point = np.asarray(self.points[index], dtype=float)
            if point.shape != (self.d,) or np.any(~np.isfinite(point)):
                raise ValueError(
                    f"points[{index}] must be a finite array with shape ({self.d},)."
                )
            fv, gv = validate_oracle_output(
                self.fvals[index], self.grads[index], self.K, self.d
            )
            self.points[index] = point.copy()
            self.fvals[index] = fv.copy()
            self.grads[index] = gv.copy()

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
                            sequentially.
        """
        if len(objectives) != self.K or len(grad_objectives) != self.K:
            raise ValueError(
                f"Expected K={self.K} objective and gradient callables; got "
                f"{len(objectives)} and {len(grad_objectives)}."
            )
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape != (self.d,):
            raise ValueError(f"x must have shape ({self.d},); got {x_arr.shape}.")
        if np.any(~np.isfinite(x_arr)):
            raise ValueError("x must contain only finite values.")

        if joint_oracle is not None:
            fv, gv = joint_oracle(x_arr)
        else:
            fv = np.array([f(x_arr) for f in objectives])
            grad_rows = []
            for index, grad_objective in enumerate(grad_objectives):
                grad = np.asarray(grad_objective(x_arr), dtype=float)
                if grad.shape != (self.d,):
                    raise ValueError(
                        f"grad_objectives[{index}] must return shape ({self.d},); "
                        f"got {grad.shape}."
                    )
                if np.any(~np.isfinite(grad)):
                    raise ValueError(
                        f"grad_objectives[{index}] returned non-finite values."
                    )
                grad_rows.append(grad)
            gv = np.vstack(grad_rows)   # (K, d)
        fv, gv = validate_oracle_output(fv, gv, self.K, self.d)
        self.points.append(x_arr.copy())
        self.fvals.append(fv.copy())
        self.grads.append(gv.copy())

    def pop_point(self):
        """Pop the last element out of the oracle."""
        self.points.pop()
        self.fvals.pop()
        self.grads.pop()
