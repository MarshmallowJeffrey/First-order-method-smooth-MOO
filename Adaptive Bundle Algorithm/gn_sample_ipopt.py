"""GN maximisation with random sampling and IPOPT.

This module collects the Sample and IPOPT-related logic from ``algorithm.py``
into one reusable file.  It solves

    max_{lambda in Delta_K} min_i ||J_i.T @ lambda||^2,

where ``Delta_K`` is the K-dimensional probability simplex and ``J_i`` is the
objective Jacobian at bundle point ``i``.

The public functions accept either:

1. a Bundle-like object with a ``grads`` attribute; or
2. a NumPy array with shape ``(m, K, d)``.

IPOPT support is optional.  Install it with, for example:

    conda install -c conda-forge cyipopt

If ``cyipopt`` is unavailable, the IPOPT function automatically falls back to
SciPy's SLSQP solver.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as scipy_minimize

try:
    from cyipopt import minimize_ipopt as ipopt_minimize

    HAS_IPOPT = True
except (ModuleNotFoundError, ImportError):
    ipopt_minimize = None
    HAS_IPOPT = False


EPS = 1e-8


def _as_jacobian_array(bundle_or_jacobians: Any) -> np.ndarray:
    """Return and validate a Jacobian stack with shape ``(m, K, d)``."""

    source = (
        bundle_or_jacobians.grads
        if hasattr(bundle_or_jacobians, "grads")
        else bundle_or_jacobians
    )
    jacobians = np.asarray(source, dtype=float)

    if jacobians.ndim != 3:
        raise ValueError("Expected Jacobians with shape (m, K, d).")
    if any(size == 0 for size in jacobians.shape):
        raise ValueError("The Jacobian array must not contain an empty axis.")
    if not np.all(np.isfinite(jacobians)):
        raise ValueError("The Jacobian array contains NaN or infinite values.")

    return jacobians


def _normalise_simplex_point(lam: np.ndarray, K: int) -> np.ndarray:
    """Clip and normalise a vector so it is a valid simplex point."""

    point = np.asarray(lam, dtype=float)
    if point.shape != (K,):
        raise ValueError(f"Expected lambda with shape ({K},), got {point.shape}.")
    point = np.maximum(point, EPS)
    return point / point.sum()


def _gn_value_and_gradient(
    jacobians: np.ndarray, lam: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Evaluate GN and its analytical gradient with respect to ``lambda``.

    At a smooth point, Danskin's theorem gives the gradient using the bundle
    point that attains the minimum squared gradient norm.
    """

    weighted_gradients = np.einsum("mkd,k->md", jacobians, lam)
    squared_norms = np.einsum(
        "md,md->m", weighted_gradients, weighted_gradients
    )
    active_index = int(np.argmin(squared_norms))
    active_gradient = weighted_gradients[active_index]
    gn_gradient = 2.0 * (jacobians[active_index] @ active_gradient)
    return float(squared_norms[active_index]), gn_gradient


def _structured_starts(
    K: int,
    previous_lambda: Optional[np.ndarray],
    max_starts: int,
    seed: int = 0,
) -> list[np.ndarray]:
    """Build bounded multi-start points for the non-concave optimisation.

    The priority is centroid, vertices, near-corner points, previous solution,
    and edge midpoints.  Edge midpoints are sampled lazily when K is large.
    """

    if max_starts < 1:
        raise ValueError("max_starts must be at least 1.")

    rng = np.random.RandomState(seed)
    starts: list[np.ndarray] = []

    def remaining_capacity() -> int:
        return max_starts - len(starts)

    # Simplex centroid.
    starts.append(np.full(K, 1.0 / K))

    # Simplex vertices, shifted slightly inward to respect IPOPT bounds.
    if remaining_capacity() > 0:
        indices = np.arange(K)
        if K > remaining_capacity():
            indices = rng.choice(K, size=remaining_capacity(), replace=False)
        for k in indices:
            point = np.full(K, EPS)
            point[k] = 1.0 - (K - 1) * EPS
            starts.append(point)

    # Near-corner points improve coverage of non-concave GN basins.
    if K > 1 and remaining_capacity() > 0:
        indices = np.arange(K)
        if K > remaining_capacity():
            indices = rng.choice(K, size=remaining_capacity(), replace=False)
        for k in indices:
            point = np.full(K, 0.2 / (K - 1))
            point[k] = 0.8
            starts.append(point)

    # Warm-start from the previous outer iteration.
    if remaining_capacity() > 0 and previous_lambda is not None:
        starts.append(_normalise_simplex_point(previous_lambda, K))

    # Enumerate or lazily sample edge midpoints.
    total_edges = K * (K - 1) // 2
    if remaining_capacity() > 0 and total_edges > 0:

        def edge_midpoint(a: int, b: int) -> np.ndarray:
            point = np.full(K, EPS)
            point[a] = 0.5 - (K - 2) * 0.5 * EPS
            point[b] = 0.5 - (K - 2) * 0.5 * EPS
            return point

        if total_edges <= remaining_capacity():
            for a in range(K):
                for b in range(a + 1, K):
                    starts.append(edge_midpoint(a, b))
        else:
            selected_edges: set[tuple[int, int]] = set()
            while remaining_capacity() > 0 and len(selected_edges) < total_edges:
                a = int(rng.randint(0, K - 1))
                b = int(rng.randint(a + 1, K))
                if (a, b) not in selected_edges:
                    selected_edges.add((a, b))
                    starts.append(edge_midpoint(a, b))

    return starts


def gn_over_samples(jacobians: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Evaluate GN for many simplex samples in one vectorised operation.

    Parameters
    ----------
    jacobians:
        Array with shape ``(m, K, d)``.
    lambdas:
        Array with shape ``(S, K)``, where S is the number of samples.

    Returns
    -------
    np.ndarray
        One GN value for each of the S simplex samples.
    """

    jacobians = _as_jacobian_array(jacobians)
    lambdas = np.asarray(lambdas, dtype=float)
    K = jacobians.shape[1]
    if lambdas.ndim != 2 or lambdas.shape[1] != K:
        raise ValueError(f"Expected samples with shape (S, {K}).")

    gram_matrices = np.einsum("mkd,mld->mkl", jacobians, jacobians)
    quadratic_values = np.einsum(
        "sk,mkl,sl->sm", lambdas, gram_matrices, lambdas
    )
    return quadratic_values.min(axis=1)


def maximise_gn_sample(
    bundle_or_jacobians: Any,
    previous_lambda: Optional[np.ndarray] = None,
    *,
    n_random: int = 2048,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Approximately maximise GN using structured and Dirichlet samples."""

    if n_random < 0:
        raise ValueError("n_random must be non-negative.")

    jacobians = _as_jacobian_array(bundle_or_jacobians)
    K = jacobians.shape[1]
    starts = _structured_starts(K, previous_lambda, max_starts=2 * K + 2)
    sample_blocks = [np.asarray(starts, dtype=float)]

    if n_random > 0:
        rng = np.random.RandomState(seed)
        sample_blocks.append(rng.dirichlet(np.ones(K), size=n_random))

    lambdas = np.vstack(sample_blocks)
    values = gn_over_samples(jacobians, lambdas)
    best_index = int(np.argmax(values))
    return float(values[best_index]), lambdas[best_index].copy()


def maximise_gn_ipopt(
    bundle_or_jacobians: Any,
    previous_lambda: Optional[np.ndarray] = None,
    *,
    max_starts: int = 256,
) -> Tuple[float, np.ndarray]:
    """Maximise GN with multi-start IPOPT, or SLSQP when IPOPT is absent."""

    jacobians = _as_jacobian_array(bundle_or_jacobians)
    K = jacobians.shape[1]
    starts = _structured_starts(K, previous_lambda, max_starts)

    # Cache the objective and gradient because optimisers often request both at
    # the same lambda value.
    cache: list[Any] = [None, None, None]

    def negative_gn(lam: np.ndarray) -> float:
        point = np.asarray(lam, dtype=float)
        key = point.tobytes()
        if cache[0] != key:
            value, gradient = _gn_value_and_gradient(jacobians, point)
            cache[0], cache[1], cache[2] = key, -value, -gradient
        return float(cache[1])

    def negative_gn_gradient(lam: np.ndarray) -> np.ndarray:
        negative_gn(lam)
        return np.asarray(cache[2])

    equality_constraint = {
        "type": "eq",
        "fun": lambda lam: float(np.sum(lam) - 1.0),
        "jac": lambda lam: np.ones(K),
    }
    constraints = [equality_constraint]
    bounds = [(EPS, 1.0)] * K

    if not HAS_IPOPT:
        warnings.warn(
            "cyipopt is unavailable; falling back to SciPy SLSQP.",
            RuntimeWarning,
            stacklevel=2,
        )

    best_negative_value = np.inf
    best_lambda = starts[0]

    for initial_lambda in starts:
        # Score the initial point so a failed local solve cannot lose it.
        initial_value = negative_gn(initial_lambda)
        if initial_value < best_negative_value:
            best_negative_value = initial_value
            best_lambda = initial_lambda.copy()

        try:
            if HAS_IPOPT:
                result = ipopt_minimize(
                    negative_gn,
                    initial_lambda,
                    jac=negative_gn_gradient,
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        "print_level": 0,
                        "sb": "yes",
                        "tol": 1e-6,
                        "max_iter": 200,
                        "hessian_approximation": "limited-memory",
                    },
                )
            else:
                result = scipy_minimize(
                    negative_gn,
                    initial_lambda,
                    jac=negative_gn_gradient,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"ftol": 1e-6, "maxiter": 60},
                )
        except Exception:
            # One failed local solve should not terminate the entire search.
            continue

        if np.isfinite(result.fun) and result.fun < best_negative_value:
            best_negative_value = float(result.fun)
            best_lambda = np.asarray(result.x, dtype=float).copy()

    best_lambda = _normalise_simplex_point(best_lambda, K)
    return float(-best_negative_value), best_lambda


def maximise_gn(
    bundle_or_jacobians: Any,
    *,
    method: str = "ipopt",
    previous_lambda: Optional[np.ndarray] = None,
    max_starts: int = 256,
    n_random: int = 2048,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Unified entry point for the Sample and IPOPT implementations.

    Set ``method="sample"`` for the faster approximate search, or
    ``method="ipopt"`` for multi-start local optimisation.
    """

    if method == "sample":
        return maximise_gn_sample(
            bundle_or_jacobians,
            previous_lambda,
            n_random=n_random,
            seed=seed,
        )
    if method == "ipopt":
        return maximise_gn_ipopt(
            bundle_or_jacobians,
            previous_lambda,
            max_starts=max_starts,
        )
    raise ValueError("method must be either 'sample' or 'ipopt'.")


__all__ = [
    "HAS_IPOPT",
    "gn_over_samples",
    "maximise_gn",
    "maximise_gn_ipopt",
    "maximise_gn_sample",
]
