"""Preference-weighted Chebyshev method for the MLP coverage experiment.

This module is intentionally independent from the adaptive bundle update.
It reuses only the existing worst-preference and GN* measurement routines so
all compared methods are scored with the same coverage metric.
"""

from __future__ import annotations

import itertools
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as sp_minimize

from algorithm import _maximise_GN, _maximise_GN_sampled, _pc_star_metric
from bundle import Bundle


def _dual_objective(beta: np.ndarray, gram: np.ndarray) -> float:
    return 0.5 * float(beta @ gram @ beta)


def _solve_dual_slsqp(
    normalized_grads: np.ndarray,
    lam: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Solve D_lambda with a general constrained optimizer."""
    K = lam.size
    gram = normalized_grads @ normalized_grads.T
    beta0 = np.ones(K, dtype=float)
    beta0 /= float(lam @ beta0)

    result = sp_minimize(
        fun=lambda beta: _dual_objective(beta, gram),
        x0=beta0,
        jac=lambda beta: gram @ beta,
        method="SLSQP",
        bounds=[(0.0, None)] * K,
        constraints=[{
            "type": "eq",
            "fun": lambda beta: float(lam @ beta - 1.0),
            "jac": lambda beta: lam,
        }],
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Weighted Chebyshev dual solve failed: {result.message}"
        )

    beta = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    beta /= float(lam @ beta)
    residual = float(np.linalg.norm(beta @ normalized_grads))
    return beta, residual


def solve_weighted_chebyshev_dual(
    normalized_grads: np.ndarray,
    lam: np.ndarray,
    *,
    active_set_limit: int = 10,
    feasibility_tol: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """Solve the weighted dual by enumerating its active supports.

    The experiment uses small K, so solving the equality-constrained quadratic
    on every nonempty support is faster and more reproducible than invoking
    IPOPT at every neural-network update. SLSQP remains the fallback for large
    K or a numerically singular active-set enumeration.
    """
    normalized_grads = np.asarray(normalized_grads, dtype=float)
    lam = np.asarray(lam, dtype=float)
    K = lam.size

    if normalized_grads.ndim != 2 or normalized_grads.shape[0] != K:
        raise ValueError("normalized_grads must have shape (K, d).")
    if np.any(lam < 0.0) or not np.isclose(lam.sum(), 1.0):
        raise ValueError("lambda must lie in the simplex.")
    if K > active_set_limit:
        return _solve_dual_slsqp(normalized_grads, lam)

    gram = normalized_grads @ normalized_grads.T
    best_beta: Optional[np.ndarray] = None
    best_value = np.inf

    for support_size in range(1, K + 1):
        for support_tuple in itertools.combinations(range(K), support_size):
            support = np.asarray(support_tuple, dtype=int)
            gram_s = gram[np.ix_(support, support)]
            lam_s = lam[support]
            kkt = np.block([
                [gram_s, lam_s[:, None]],
                [lam_s[None, :], np.zeros((1, 1))],
            ])
            rhs = np.concatenate([np.zeros(support_size), np.ones(1)])
            solution, _, _, _ = np.linalg.lstsq(kkt, rhs, rcond=None)
            beta_s = solution[:support_size]

            if abs(float(lam_s @ beta_s) - 1.0) > feasibility_tol:
                continue
            if np.min(beta_s) < -feasibility_tol:
                continue

            beta = np.zeros(K, dtype=float)
            beta[support] = np.maximum(beta_s, 0.0)
            denom = float(lam @ beta)
            if denom <= 0.0:
                continue
            beta /= denom
            value = _dual_objective(beta, gram)
            if value < best_value:
                best_value = value
                best_beta = beta

    if best_beta is None:
        return _solve_dual_slsqp(normalized_grads, lam)
    residual = float(np.linalg.norm(best_beta @ normalized_grads))
    return best_beta, residual


def weighted_chebyshev_inner(
    lam: np.ndarray,
    x0: np.ndarray,
    initial_fvals: np.ndarray,
    initial_grads: np.ndarray,
    L: np.ndarray,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    *,
    joint_oracle: Optional[Callable] = None,
    max_steps: int = 25,
    delta: float = 1e-8,
    direction_policy: str = "chebyshev_only",
    gn_tolerance: Optional[float] = None,
    alignment_threshold: float = 0.1,
    step_scale: float = 1.0,
    optimizer: str = "gd",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    gradient_tol: float = 1e-14,
) -> Dict:
    """Run the displayed weighted-Chebyshev inner loop at fixed lambda.

    ``chebyshev_only`` follows the displayed algorithm: it stops when the
    weighted Chebyshev residual is at most ``delta`` and otherwise uses only
    the recovered Chebyshev direction. ``hybrid`` is an explicit GN-aligned
    variant that may use the scalarized gradient when the Chebyshev direction
    is undefined or poorly aligned. The two policies are labelled separately
    in plots and results.
    """
    if direction_policy not in {"chebyshev_only", "hybrid"}:
        raise ValueError(
            "direction_policy must be 'chebyshev_only' or 'hybrid'."
        )
    if optimizer not in {"gd", "adam"}:
        raise ValueError("optimizer must be 'gd' or 'adam'.")
    if delta < 0.0:
        raise ValueError("delta must be nonnegative.")
    if gn_tolerance is not None and gn_tolerance < 0.0:
        raise ValueError("gn_tolerance must be nonnegative.")
    if not 0.0 <= alignment_threshold <= 1.0:
        raise ValueError("alignment_threshold must lie in [0, 1].")
    if not 0.0 < step_scale <= 1.0:
        raise ValueError("step_scale must lie in (0, 1].")

    lam = np.asarray(lam, dtype=float)
    if np.any(lam < 0.0) or not np.all(np.isfinite(lam)):
        raise ValueError("lambda must be finite and nonnegative.")
    lam_sum = float(lam.sum())
    if lam_sum <= 0.0:
        raise ValueError("lambda must have positive sum.")
    lam = lam / lam_sum

    x = np.asarray(x0, dtype=float).copy()
    fvals = np.asarray(initial_fvals, dtype=float).copy()
    grads = np.asarray(initial_grads, dtype=float).copy()
    eta = step_scale / float(lam @ np.asarray(L, dtype=float))

    first_moment = np.zeros_like(x)
    second_moment = np.zeros_like(x)
    residual_history: List[float] = []
    scalarized_grad_history: List[float] = []
    alignment_history: List[float] = []
    direction_history: List[str] = []
    status = "max_steps"
    steps = 0

    def evaluate(point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if joint_oracle is not None:
            fv, gv = joint_oracle(point)
            return np.asarray(fv, dtype=float), np.asarray(gv, dtype=float)
        fv = np.asarray([objective(point) for objective in objectives])
        gv = np.vstack([gradient(point) for gradient in grad_objectives])
        return fv, gv

    for iteration in range(max_steps):
        scalarized_grad = grads.T @ lam
        scalarized_grad_norm = float(np.linalg.norm(scalarized_grad))
        scalarized_grad_history.append(scalarized_grad_norm)
        if (
            direction_policy == "hybrid"
            and gn_tolerance is not None
            and scalarized_grad_norm * scalarized_grad_norm <= gn_tolerance
        ):
            status = "scalarized_stationary"
            break

        grad_norms = np.linalg.norm(grads, axis=1)
        chebyshev_defined = bool(np.all(grad_norms > gradient_tol))
        residual = np.nan
        alignment = np.nan
        direction_kind = "chebyshev"

        if chebyshev_defined:
            normalized_grads = grads / grad_norms[:, None]
            beta, residual = solve_weighted_chebyshev_dual(
                normalized_grads, lam
            )
        else:
            beta = np.empty(lam.size)

        if not chebyshev_defined or residual <= delta:
            if direction_policy == "chebyshev_only":
                status = (
                    "chebyshev_undefined"
                    if not chebyshev_defined
                    else "chebyshev_stationary"
                )
                residual_history.append(float(residual))
                alignment_history.append(float(alignment))
                break
            direction = scalarized_grad
            direction_kind = "scalarized_fallback"
        else:
            w = beta @ normalized_grads
            v = w / residual
            adaptive_scalar = float(scalarized_grad @ v)
            alignment = (
                adaptive_scalar / scalarized_grad_norm
                if scalarized_grad_norm > 0.0
                else 0.0
            )
            use_chebyshev = (
                adaptive_scalar > 0.0
                and np.isfinite(alignment)
                and (
                    direction_policy == "chebyshev_only"
                    or alignment >= alignment_threshold
                )
            )
            if use_chebyshev:
                direction = adaptive_scalar * v
            elif direction_policy == "hybrid":
                direction = scalarized_grad
                direction_kind = "scalarized_fallback"
            else:
                status = "non_descent_chebyshev_direction"
                residual_history.append(float(residual))
                alignment_history.append(float(alignment))
                break

        residual_history.append(float(residual))
        alignment_history.append(float(alignment))
        direction_history.append(direction_kind)

        if optimizer == "adam":
            first_moment = (
                adam_beta1 * first_moment + (1.0 - adam_beta1) * direction
            )
            second_moment = (
                adam_beta2 * second_moment
                + (1.0 - adam_beta2) * direction * direction
            )
            t = iteration + 1
            m_hat = first_moment / (1.0 - adam_beta1 ** t)
            v_hat = second_moment / (1.0 - adam_beta2 ** t)
            update = m_hat / (np.sqrt(v_hat) + adam_epsilon)
        else:
            update = direction

        x = x - eta * update
        fvals, grads = evaluate(x)
        steps += 1

    return {
        "x": x,
        "fvals": fvals,
        "grads": grads,
        "steps": steps,
        "status": status,
        "dual_residual_history": residual_history,
        "scalarized_grad_history": scalarized_grad_history,
        "alignment_history": alignment_history,
        "direction_history": direction_history,
    }


def _select_outer_preference(
    bundle: Bundle,
    *,
    selector: str,
    prev_lam: Optional[np.ndarray],
    max_starts: int,
    n_random: int,
    seed: int,
) -> Tuple[float, np.ndarray]:
    if selector == "sample":
        return _maximise_GN_sampled(
            bundle,
            prev_lam=prev_lam,
            n_random=n_random,
            seed=seed,
        )
    if selector == "optimize":
        return _maximise_GN(
            bundle,
            prev_lam=prev_lam,
            max_starts=max_starts,
        )
    raise ValueError("lambda_selector must be 'optimize' or 'sample'.")


def algorithm_chebyshev_adaptive(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    *,
    max_outer: int = 120,
    max_inner: int = 25,
    delta: float = 1e-8,
    direction_policy: str = "chebyshev_only",
    gn_tolerance: Optional[float] = None,
    alignment_threshold: float = 0.1,
    epsilon: Optional[float] = None,
    eval_every_n_grads: Optional[int] = None,
    target_cov: Optional[float] = None,
    lambda_max_starts: int = 256,
    lambda_selector: str = "optimize",
    lambda_random_starts: int = 2048,
    step_scale: float = 1.0,
    optimizer: str = "gd",
    joint_oracle: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """Run the anchor-based outer algorithm with a Chebyshev inner solver."""
    L_arr = np.asarray(L, dtype=float)
    bundle = Bundle(K=K, d=d, L=L_arr)
    bundle.add_point(
        np.asarray(x0, dtype=float),
        objectives,
        grad_objectives,
        joint_oracle=joint_oracle,
    )

    anchor_lambdas: List[np.ndarray] = [np.full(K, 1.0 / K)]
    anchor_indices: List[int] = [0]
    cpu_times: List[float] = []
    cov_history: List[float] = []
    total_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    lambda_history: List[np.ndarray] = []
    inner_status_history: List[str] = []
    chebyshev_steps = 0
    fallback_steps = 0

    total_iters = 0
    grad_evals_at_last_ckpt = 0
    outer_prev_lam: Optional[np.ndarray] = None
    metric_prev_lam: Optional[np.ndarray] = None
    checkpoint_overhead = 0.0
    start_time = time.time()
    termination_reason = "max_outer"

    def checkpoint(label: str) -> Tuple[float, np.ndarray]:
        nonlocal checkpoint_overhead, metric_prev_lam
        cpu_times.append(time.time() - start_time - checkpoint_overhead)
        checkpoint_start = time.time()
        cov, cov_lam = _pc_star_metric(
            bundle,
            "gn",
            prev_lam=metric_prev_lam,
        )
        metric_prev_lam = cov_lam
        cov_history.append(cov)
        checkpoint_overhead += time.time() - checkpoint_start
        total_iters_history.append(total_iters)
        grad_evals_history.append(total_iters * K)
        if verbose:
            print(
                f"  Chebyshev {label} | t={cpu_times[-1]:.2f}s "
                f"| bundle={bundle.m} | iters={total_iters} "
                f"| grad_evals={total_iters * K} | worst-case pc={cov:.4e}"
            )
        return cov, cov_lam

    checkpoint(f"outer 0/{max_outer}")
    cpu_times[0] = 0.0
    start_time = time.time()
    checkpoint_overhead = 0.0

    for outer in range(1, max_outer + 1):
        pc_value, lam = _select_outer_preference(
            bundle,
            selector=lambda_selector,
            prev_lam=outer_prev_lam,
            max_starts=lambda_max_starts,
            n_random=lambda_random_starts,
            seed=outer,
        )
        outer_prev_lam = lam
        lambda_history.append(lam.copy())

        if epsilon is not None and pc_value <= 2.0 * epsilon / 3.0:
            termination_reason = "epsilon_reached"
            checkpoint(f"outer {outer}/{max_outer}")
            break

        anchor_array = np.asarray(anchor_lambdas)
        distances = np.linalg.norm(anchor_array - lam, axis=1)
        min_distance = float(distances.min())
        nearest_candidates = np.flatnonzero(
            np.isclose(distances, min_distance, rtol=0.0, atol=1e-12)
        )
        # Repeated worst preferences should continue from their newest
        # solution instead of replaying the same inner loop from an old anchor.
        nearest_anchor = int(nearest_candidates[-1])
        bundle_index = anchor_indices[nearest_anchor]
        inner = weighted_chebyshev_inner(
            lam=lam,
            x0=bundle.points[bundle_index],
            initial_fvals=bundle.fvals[bundle_index],
            initial_grads=bundle.grads[bundle_index],
            L=L_arr,
            objectives=objectives,
            grad_objectives=grad_objectives,
            joint_oracle=joint_oracle,
            max_steps=max_inner,
            delta=delta,
            direction_policy=direction_policy,
            gn_tolerance=gn_tolerance,
            alignment_threshold=alignment_threshold,
            step_scale=step_scale,
            optimizer=optimizer,
        )
        steps = int(inner["steps"])
        total_iters += steps
        inner_status_history.append(str(inner["status"]))
        directions = inner["direction_history"]
        chebyshev_steps += directions.count("chebyshev")
        fallback_steps += directions.count("scalarized_fallback")

        if steps == 0:
            cov, _ = checkpoint(
                f"outer {outer}/{max_outer} ({inner['status']})"
            )
            if target_cov is not None and cov <= target_cov:
                termination_reason = "target_reached"
            else:
                termination_reason = str(inner["status"])
            break

        bundle.points.append(np.asarray(inner["x"], dtype=float).copy())
        bundle.fvals.append(np.asarray(inner["fvals"], dtype=float).copy())
        bundle.grads.append(np.asarray(inner["grads"], dtype=float).copy())
        anchor_lambdas.append(np.asarray(lam, dtype=float).copy())
        anchor_indices.append(bundle.m - 1)

        current_grad_evals = total_iters * K
        do_checkpoint = (
            eval_every_n_grads is None
            or current_grad_evals - grad_evals_at_last_ckpt
            >= eval_every_n_grads
            or outer == max_outer
        )
        if do_checkpoint:
            cov, _ = checkpoint(f"outer {outer}/{max_outer}")
            grad_evals_at_last_ckpt = current_grad_evals
            if target_cov is not None and cov <= target_cov:
                termination_reason = "target_reached"
                break

    return {
        "bundle": bundle,
        "anchors": list(zip(anchor_lambdas, anchor_indices)),
        "cpu_times": cpu_times,
        "cov_history": cov_history,
        "total_iters_history": total_iters_history,
        "grad_evals_history": grad_evals_history,
        "lambda_history": lambda_history,
        "inner_status_history": inner_status_history,
        "chebyshev_steps": chebyshev_steps,
        "fallback_steps": fallback_steps,
        "direction_policy": direction_policy,
        "optimizer": optimizer,
        "step_scale": step_scale,
        "termination_reason": termination_reason,
        "max_outer": max_outer,
        "max_inner": max_inner,
    }
