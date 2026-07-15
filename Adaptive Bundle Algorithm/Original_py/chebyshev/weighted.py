"""Weighted-Chebyshev adaptive algorithm.

This module is intentionally separate from ``algorithm.py``.  It reuses the
existing bundle data structure, GN* metric, fused oracle handling, and IPOPT
lambda maximiser, while replacing the adaptive bundle inner ``T_map`` update
with the paper-style weighted-Chebyshev exact-GD update.

The three layers are:

1. ``solve_weighted_chebyshev_dual`` solves the Chebyshev beta subproblem.
2. ``weighted_chebyshev_exact_gd`` solves one fixed preference lambda.
3. ``chebyshev_adaptive`` wraps the inner loop in the same worst-lambda
   outer loop used by the adaptive bundle method.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize as sp_minimize

from algorithm import (
    _HAS_IPOPT,
    _IPOPT_IMPORT_ERROR,
    _T_map_batched,
    _ipopt_minimize,
    _maximise_GN,
    pc_star,
)
from bundle import (
    Bundle,
    prefer_fused_joint_oracle,
    validate_oracle_output,
    validate_problem_inputs,
)

Array = np.ndarray


def _simplex(lam: Array) -> Array:
    """Project a nonnegative vector to the probability simplex by clipping."""
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if lam.size == 0 or np.any(~np.isfinite(lam)):
        raise ValueError("lambda must be a finite nonempty vector.")
    lam = np.maximum(lam, 0.0)
    mass = float(lam.sum())
    if mass <= 0.0:
        return np.full(lam.size, 1.0 / lam.size)
    return lam / mass


def scalarized_gradient(lam: Array, grads: Array) -> Array:
    """Return ∇F_lambda(x) from all objective gradients at one point."""
    return np.asarray(grads, dtype=float).T @ _simplex(lam)


def scalarized_gn(lam: Array, grads: Array) -> float:
    """Return ||∇F_lambda(x)||^2 from all objective gradients at one point."""
    g = scalarized_gradient(lam, grads)
    return float(g @ g)


def scalarized_value(lam: Array, fvals: Array) -> float:
    """Return F_lambda(x) from all objective values at one point."""
    return float(_simplex(lam) @ np.asarray(fvals, dtype=float))


def solve_weighted_chebyshev_dual(
    grads: Array,
    lam: Array,
    *,
    normalization_power: float = 1.0,
    normalization_floor: float = 0.0,
    lambda_floor: float = 1e-10,
    grad_floor: float = 1e-14,
    solver: str = "ipopt",
    require_ipopt: bool = False,
) -> Tuple[Array, float, Array, Array]:
    """Solve the weighted-Chebyshev dual subproblem.

    The paper's weighted direction solves

        min_beta 0.5 ||sum_k beta_k ghat_k||^2
        s.t. beta >= 0,  sum_k lambda_k beta_k = 1,

    where

        ghat_k = lambda_k grad F_k / ||lambda_k grad F_k||.

    ``normalization_power=1`` gives the exact paper normalization.  Smaller
    powers are optional experimental variants; the outer algorithm defaults to
    the paper setting.
    """
    if solver not in {"ipopt", "slsqp"}:
        raise ValueError("solver must be 'ipopt' or 'slsqp'.")
    if normalization_power < 0.0:
        raise ValueError("normalization_power must be nonnegative.")
    if normalization_floor < 0.0:
        raise ValueError("normalization_floor must be nonnegative.")
    if require_ipopt and not _HAS_IPOPT:
        raise RuntimeError(
            "IPOPT was required for the Chebyshev dual, but cyipopt/IPOPT "
            f"is unavailable. Import error: {_IPOPT_IMPORT_ERROR}"
        )

    grads = np.asarray(grads, dtype=float)
    lam = _simplex(lam)
    if grads.ndim != 2 or grads.shape[0] != lam.size:
        raise ValueError(
            "grads must have shape (K, d), with K equal to len(lambda)."
        )

    weighted = lam[:, None] * grads
    norms = np.linalg.norm(weighted, axis=1)
    active = (lam > lambda_floor) & (norms > grad_floor)
    normalized_full = np.zeros_like(weighted)

    if not np.any(active):
        return np.zeros_like(lam), 0.0, active, normalized_full

    lam_active = lam[active]
    denom = norms[active] ** normalization_power + normalization_floor
    if np.any(denom <= 0.0):
        raise RuntimeError("Chebyshev normalization denominator is nonpositive.")
    normalized = weighted[active] / denom[:, None]
    normalized_full[active] = normalized
    gram = normalized @ normalized.T

    beta0 = np.ones(lam_active.size, dtype=float)
    beta0 /= float(lam_active @ beta0)

    def objective(beta: Array) -> float:
        beta = np.asarray(beta, dtype=float)
        return 0.5 * float(beta @ gram @ beta)

    def jacobian(beta: Array) -> Array:
        return gram @ np.asarray(beta, dtype=float)

    constraints = [{
        "type": "eq",
        "fun": lambda beta: float(lam_active @ beta - 1.0),
        "jac": lambda beta: lam_active,
    }]
    bounds = [(0.0, None)] * lam_active.size

    use_ipopt = solver == "ipopt" and _HAS_IPOPT
    if solver == "ipopt" and not _HAS_IPOPT:
        warnings.warn(
            "cyipopt/IPOPT is unavailable; Chebyshev dual is falling back "
            f"to SLSQP. Import error: {_IPOPT_IMPORT_ERROR}",
            RuntimeWarning,
            stacklevel=2,
        )

    if use_ipopt:
        result = _ipopt_minimize(
            objective,
            beta0,
            jac=jacobian,
            bounds=bounds,
            constraints=constraints,
            options={
                "print_level": 0,
                "sb": "yes",
                "tol": 1e-9,
                "max_iter": 200,
                "hessian_approximation": "limited-memory",
            },
        )
    else:
        result = sp_minimize(
            objective,
            beta0,
            jac=jacobian,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 200, "disp": False},
        )

    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise RuntimeError("Chebyshev dual solve returned non-finite values.")

    beta_active = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    feasibility = float(lam_active @ beta_active)
    if feasibility <= 0.0:
        raise RuntimeError("Chebyshev dual solve returned infeasible beta.")
    beta_active /= feasibility

    beta = np.zeros_like(lam)
    beta[active] = beta_active
    residual = float(np.linalg.norm(beta_active @ normalized))
    return beta, residual, active, normalized_full


def weighted_chebyshev_exact_gd(
    lam: Array,
    x0: Array,
    initial_fvals: Array,
    initial_grads: Array,
    L: Array,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    *,
    joint_oracle: Optional[Callable] = None,
    max_steps: int = 25,
    delta: float = 1e-8,
    step_scale: float = 1.0,
    L_scale: float = 1.0,
    gn_tolerance: Optional[float] = None,
    normalization_power: float = 1.0,
    normalization_floor: float = 0.0,
    lambda_floor: float = 1e-10,
    grad_floor: float = 1e-14,
    descent_floor: float = 1e-14,
    line_search: str = "none",
    armijo_c1: float = 1e-4,
    gn_decrease_tol: float = 1e-12,
    line_search_shrink: float = 0.5,
    line_search_max_steps: int = 12,
    return_last_if_no_gn_improvement: bool = False,
    fallback_update: str = "none",
    max_oracle_calls: Optional[int] = None,
    dual_solver: str = "ipopt",
    require_ipopt: bool = False,
) -> Dict:
    """Run Weighted-Chebyshev-Exact-GD for one fixed preference.

    ``line_search='none'`` is the paper-style fixed step.  ``'armijo'`` checks
    scalarized objective decrease.  ``'gn_decrease'`` accepts only steps
    decreasing ||∇F_lambda||^2.  ``'gn_or_armijo'`` first tries the GN check,
    then accepts scalarized Armijo descent along the same Chebyshev direction
    so the inner loop does not stall immediately when GN decrease is locally
    too strict.
    """
    if (
        not isinstance(max_steps, (int, np.integer))
        or isinstance(max_steps, (bool, np.bool_))
        or max_steps < 0
    ):
        raise ValueError(f"max_steps must be a nonnegative integer; got {max_steps!r}.")
    if step_scale <= 0.0 or L_scale <= 0.0:
        raise ValueError("step_scale and L_scale must be positive.")
    if line_search not in {"none", "armijo", "gn_decrease", "gn_or_armijo"}:
        raise ValueError(
            "line_search must be 'none', 'armijo', 'gn_decrease', "
            "or 'gn_or_armijo'."
        )
    if not 0.0 < line_search_shrink < 1.0:
        raise ValueError("line_search_shrink must lie in (0, 1).")
    if fallback_update not in {"none", "scalarized_gd"}:
        raise ValueError("fallback_update must be 'none' or 'scalarized_gd'.")

    lam = _simplex(lam)
    L_arr = np.asarray(L, dtype=float)
    if L_arr.shape != (lam.size,) or np.any(~np.isfinite(L_arr)) or np.any(L_arr <= 0.0):
        raise ValueError("L must be a finite positive vector with shape (K,).")
    eta0 = step_scale / max(float(lam @ L_arr) * L_scale, 1e-12)

    x = np.asarray(x0, dtype=float).copy()
    fvals = np.asarray(initial_fvals, dtype=float).copy()
    grads = np.asarray(initial_grads, dtype=float).copy()
    joint_oracle = prefer_fused_joint_oracle(joint_oracle)

    oracle_calls = 0
    oracle_budget_exhausted = False

    def evaluate(point: Array) -> Tuple[Array, Array]:
        nonlocal oracle_calls, oracle_budget_exhausted
        if max_oracle_calls is not None and oracle_calls >= max_oracle_calls:
            oracle_budget_exhausted = True
            raise RuntimeError("weighted_chebyshev_exact_gd oracle budget exhausted")
        oracle_calls += 1
        if joint_oracle is not None:
            fv, gv = joint_oracle(point)
            return validate_oracle_output(fv, gv, lam.size, point.size)
        fv = np.asarray([f(point) for f in objectives], dtype=float)
        gv = np.vstack([g(point) for g in grad_objectives])
        return validate_oracle_output(fv, gv, lam.size, point.size)

    best_x = x.copy()
    best_fvals = fvals.copy()
    best_grads = grads.copy()
    best_gn = scalarized_gn(lam, grads)
    initial_gn = float(best_gn)

    residual_history: List[float] = []
    scalarized_gn_history: List[float] = []
    scalarized_value_history: List[float] = []
    accepted_step_sizes: List[float] = []
    line_search_trials_history: List[int] = []
    directional_derivative_history: List[float] = []
    accepted_update_history: List[str] = []
    fallback_steps = 0
    steps = 0
    status = "max_steps"

    for _ in range(max_steps):
        sg = scalarized_gradient(lam, grads)
        current_gn = float(sg @ sg)
        current_value = scalarized_value(lam, fvals)
        scalarized_gn_history.append(current_gn)
        scalarized_value_history.append(current_value)

        if current_gn < best_gn:
            best_gn = current_gn
            best_x = x.copy()
            best_fvals = fvals.copy()
            best_grads = grads.copy()
        if gn_tolerance is not None and best_gn <= gn_tolerance:
            status = "scalarized_stationary"
            break

        beta, residual, active, normalized = solve_weighted_chebyshev_dual(
            grads,
            lam,
            normalization_power=normalization_power,
            normalization_floor=normalization_floor,
            lambda_floor=lambda_floor,
            grad_floor=grad_floor,
            solver=dual_solver,
            require_ipopt=require_ipopt,
        )
        residual_history.append(residual)
        if residual <= delta:
            status = "chebyshev_stationary"
            break
        if not np.any(active):
            status = "chebyshev_undefined"
            break

        w = beta @ normalized
        w_norm = float(np.linalg.norm(w))
        if not np.isfinite(w_norm):
            status = "nonfinite_chebyshev_direction"
            break
        if w_norm <= grad_floor:
            status = "zero_chebyshev_direction"
            break

        v = w / w_norm
        derivative = float(sg @ v)
        directional_derivative_history.append(derivative)
        if not np.isfinite(derivative):
            status = "nonfinite_directional_derivative"
            break
        if abs(derivative) <= descent_floor:
            status = "zero_projected_descent"
            break

        required = max(
            gn_decrease_tol,
            gn_decrease_tol * max(1.0, current_gn),
        )

        def try_step(
            direction: Array,
            predicted: float,
            *,
            allow_armijo: bool,
        ) -> Tuple[bool, int, float, Optional[Array], Optional[Array], Optional[Array]]:
            nonlocal oracle_budget_exhausted
            eta = eta0
            trials = 0
            if not np.all(np.isfinite(direction)) or float(direction @ direction) <= 0.0:
                return False, trials, eta, None, None, None

            for _ls in range(1 if line_search == "none" else line_search_max_steps):
                if max_oracle_calls is not None and oracle_calls >= max_oracle_calls:
                    oracle_budget_exhausted = True
                    break
                trials += 1
                candidate_x = x - eta * direction
                try:
                    candidate_f, candidate_g = evaluate(candidate_x)
                except RuntimeError as exc:
                    if str(exc) == "weighted_chebyshev_exact_gd oracle budget exhausted":
                        break
                    raise
                accepted = False
                if (
                    np.all(np.isfinite(candidate_x))
                    and np.all(np.isfinite(candidate_f))
                    and np.all(np.isfinite(candidate_g))
                ):
                    if line_search == "none":
                        accepted = True
                    elif line_search == "armijo" and allow_armijo:
                        accepted = (
                            scalarized_value(lam, candidate_f)
                            <= current_value - armijo_c1 * eta * predicted
                        )
                    else:
                        trial_gn = scalarized_gn(lam, candidate_g)
                        accepted = trial_gn <= current_gn - required
                        if (
                            not accepted
                            and allow_armijo
                            and line_search == "gn_or_armijo"
                        ):
                            accepted = (
                                scalarized_value(lam, candidate_f)
                                <= current_value - armijo_c1 * eta * predicted
                            )

                if accepted:
                    return (
                        True,
                        trials,
                        float(eta),
                        np.asarray(candidate_x, dtype=float),
                        np.asarray(candidate_f, dtype=float),
                        np.asarray(candidate_g, dtype=float),
                    )
                eta *= line_search_shrink
            if max_oracle_calls is not None and oracle_calls >= max_oracle_calls:
                oracle_budget_exhausted = True
            return False, trials, float(eta), None, None, None

        direction = derivative * v
        accepted, trials, eta, x_trial, f_trial, g_trial = try_step(
            direction,
            derivative ** 2,
            allow_armijo=line_search in {"armijo", "gn_or_armijo"},
        )
        update_source = "chebyshev"

        if not accepted and fallback_update == "scalarized_gd" and line_search != "none":
            fallback_accepted, fallback_trials, fallback_eta, x_fb, f_fb, g_fb = try_step(
                sg,
                current_gn,
                allow_armijo=False,
            )
            trials += fallback_trials
            if fallback_accepted:
                accepted = True
                eta = fallback_eta
                x_trial, f_trial, g_trial = x_fb, f_fb, g_fb
                update_source = "scalarized_gd_fallback"
                fallback_steps += 1

        line_search_trials_history.append(trials)
        if not accepted:
            status = "max_oracle_calls" if oracle_budget_exhausted else "line_search_failed"
            break

        x = np.asarray(x_trial, dtype=float)
        fvals = np.asarray(f_trial, dtype=float)
        grads = np.asarray(g_trial, dtype=float)
        accepted_step_sizes.append(float(eta))
        accepted_update_history.append(update_source)
        steps += 1

        new_gn = scalarized_gn(lam, grads)
        if new_gn < best_gn:
            best_gn = new_gn
            best_x = x.copy()
            best_fvals = fvals.copy()
            best_grads = grads.copy()

    use_last = (
        return_last_if_no_gn_improvement
        and steps > 0
        and best_gn >= initial_gn - max(gn_decrease_tol, gn_decrease_tol * max(1.0, initial_gn))
    )
    returned_x = x.copy() if use_last else best_x
    returned_fvals = fvals.copy() if use_last else best_fvals
    returned_grads = grads.copy() if use_last else best_grads
    returned_score = scalarized_gn(lam, returned_grads)

    return {
        "x": returned_x,
        "fvals": returned_fvals,
        "grads": returned_grads,
        "last_x": x,
        "last_fvals": fvals,
        "last_grads": grads,
        "best_scalarized_gn": float(best_gn),
        "returned_scalarized_gn": float(returned_score),
        "returned_last_iterate": bool(use_last),
        "steps": steps,
        "oracle_calls": oracle_calls,
        "status": status,
        "dual_residual_history": residual_history,
        "scalarized_gn_history": scalarized_gn_history,
        "scalarized_value_history": scalarized_value_history,
        "accepted_step_sizes": accepted_step_sizes,
        "line_search_trials_history": line_search_trials_history,
        "directional_derivative_history": directional_derivative_history,
        "accepted_update_history": accepted_update_history,
        "fallback_steps": fallback_steps,
        "normalization_power": float(normalization_power),
        "line_search": line_search,
        "fallback_update": fallback_update,
    }


def chebyshev_adaptive(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: Array,
    x0: Array,
    *,
    max_outer: int = 120,
    max_inner: int = 25,
    epsilon: Optional[float] = None,
    target_cov: Optional[float] = None,
    eval_every_n_grads: Optional[int] = None,
    max_grad_evals: Optional[int] = None,
    lambda_max_starts: int = 256,
    lambda_solver: str = "ipopt",
    dual_solver: str = "ipopt",
    require_ipopt: bool = False,
    delta: float = 1e-8,
    step_scale: float = 1.0,
    L_scale: float = 1.0,
    gn_tolerance: Optional[float] = None,
    candidate_anchor_count: int = 1,
    normalization_powers: Sequence[float] = (1.0,),
    normalization_floor: float = 0.0,
    line_search: str = "none",
    append_non_improving: bool = True,
    improvement_tol: float = 1e-12,
    return_last_if_no_gn_improvement: bool = False,
    fallback_update: str = "none",
    use_tmap_safeguard: bool = False,
    use_active_best_start: bool = False,
    joint_oracle: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """Run the Chebyshev analogue of the adaptive-bundle outer loop.

    Outer loop:
        choose lambda_t = argmax_lambda GN(lambda; B_t),
        warm-start from the nearest anchor,
        run weighted-Chebyshev exact-GD,
        append the returned point to the bundle.

    The defaults are paper-aligned: one nearest anchor, alpha=1, fixed step,
    and appending even if the active-lambda GN score does not improve.

    ``use_tmap_safeguard=True`` is the stronger experimental hybrid used in
    the comparison runs: Chebyshev candidates still propose directions, but
    the adaptive bundle T-map chain is also evaluated at the same active
    lambda, and the candidate with the smallest active-lambda GN is appended.
    """
    if (
        not isinstance(max_inner, (int, np.integer))
        or isinstance(max_inner, (bool, np.bool_))
        or max_inner < 1
    ):
        raise ValueError(f"max_inner must be a positive integer; got {max_inner!r}.")
    if candidate_anchor_count < 1:
        raise ValueError("candidate_anchor_count must be at least 1.")
    if not normalization_powers:
        raise ValueError("normalization_powers must not be empty.")
    if any(power < 0.0 for power in normalization_powers):
        raise ValueError("normalization_powers must be nonnegative.")
    if lambda_solver not in {"ipopt", "slsqp"}:
        raise ValueError("lambda_solver must be 'ipopt' or 'slsqp'.")
    if dual_solver not in {"ipopt", "slsqp"}:
        raise ValueError("dual_solver must be 'ipopt' or 'slsqp'.")
    if fallback_update not in {"none", "scalarized_gd"}:
        raise ValueError("fallback_update must be 'none' or 'scalarized_gd'.")
    if require_ipopt and (not _HAS_IPOPT or lambda_solver != "ipopt" or dual_solver != "ipopt"):
        raise RuntimeError(
            "IPOPT was required, but lambda_solver/dual_solver are not both "
            f"IPOPT with cyipopt available. Import error: {_IPOPT_IMPORT_ERROR}"
        )

    L_arr, x0_arr = validate_problem_inputs(
        K, d, L, np.asarray(x0, dtype=float), objectives, grad_objectives
    )
    if max_grad_evals is not None and max_grad_evals < K:
        raise ValueError(
            f"max_grad_evals must be at least K={K}; got {max_grad_evals}."
        )
    joint_oracle = prefer_fused_joint_oracle(joint_oracle)

    bundle = Bundle(K=K, d=d, L=L_arr)
    bundle.add_point(
        x0_arr,
        objectives,
        grad_objectives,
        joint_oracle=joint_oracle,
    )

    anchor_lambdas: List[Array] = [np.full(K, 1.0 / K)]
    anchor_indices: List[int] = [0]

    cpu_times: List[float] = []
    cov_history: List[float] = []
    pc_history: List[float] = []
    total_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    inner_steps_history: List[int] = []
    lambda_history: List[Array] = []
    inner_status_history: List[str] = []
    candidate_source_history: List[str] = []
    candidate_score_history: List[float] = []
    rejected_candidate_count_history: List[int] = []
    fallback_steps_history: List[int] = []
    tmap_safeguard_steps = 0

    total_oracle_calls = 0
    grad_evals_at_last_ckpt = 0
    outer_prev_lam: Optional[Array] = None
    metric_prev_lam: Optional[Array] = None
    checkpoint_overhead = 0.0
    t_start = time.time()
    termination_reason = "max_outer"

    def _checkpoint(label: str) -> None:
        nonlocal checkpoint_overhead, metric_prev_lam
        cpu_times.append(time.time() - t_start - checkpoint_overhead)
        ck_t0 = time.time()
        cov, cov_lam = pc_star(bundle, prev_lam=metric_prev_lam)
        metric_prev_lam = cov_lam
        cov_history.append(cov)
        checkpoint_overhead += time.time() - ck_t0
        total_iters_history.append(total_oracle_calls)
        grad_evals_history.append(total_oracle_calls * K)
        if verbose:
            print(
                f"  Chebyshev {label} | t={cpu_times[-1]:.2f}s "
                f"| bundle={bundle.m} | oracle_calls={total_oracle_calls} "
                f"| grad_evals={total_oracle_calls * K} "
                f"| worst-case pc={cov:.4e}"
            )

    _checkpoint(f"outer 0/{max_outer}")
    cpu_times[0] = 0.0
    t_start = time.time()
    checkpoint_overhead = 0.0

    stop_threshold = (
        2.0 * float(epsilon) / 3.0
        if epsilon is not None
        else target_cov
    )

    def evaluate_point(point: Array) -> Tuple[Array, Array]:
        if joint_oracle is not None:
            fv, gv = joint_oracle(point)
            return validate_oracle_output(fv, gv, K, d)
        fv = np.asarray([f(point) for f in objectives], dtype=float)
        gv = np.vstack([g(point) for g in grad_objectives])
        return validate_oracle_output(fv, gv, K, d)

    def tmap_candidate_chain(
        lam_cur: Array,
        step_cap: int,
    ) -> Tuple[Optional[Dict], int]:
        """Generate a temporary T-map chain and keep its best active-lambda point."""
        if step_cap <= 0:
            return None, 0

        tmp = Bundle(K=K, d=d, L=L_arr)
        tmp.points = [point.copy() for point in bundle.points]
        tmp.fvals = [fval.copy() for fval in bundle.fvals]
        tmp.grads = [grad.copy() for grad in bundle.grads]

        best: Optional[Dict] = None
        calls = 0
        for step in range(step_cap):
            x_new, _u_star = _T_map_batched(
                np.asarray(tmp.fvals),
                np.asarray(tmp.grads),
                np.asarray(tmp.points),
                L_arr * L_scale,
                lam_cur,
            )
            f_new, g_new = evaluate_point(x_new)
            calls += 1
            if not (
                np.all(np.isfinite(x_new))
                and np.all(np.isfinite(f_new))
                and np.all(np.isfinite(g_new))
            ):
                break

            tmp.points.append(np.asarray(x_new, dtype=float).copy())
            tmp.fvals.append(np.asarray(f_new, dtype=float).copy())
            tmp.grads.append(np.asarray(g_new, dtype=float).copy())

            score = scalarized_gn(lam_cur, g_new)
            if best is None or score < best["score"]:
                best = {
                    "x": np.asarray(x_new, dtype=float),
                    "fvals": np.asarray(f_new, dtype=float),
                    "grads": np.asarray(g_new, dtype=float),
                    "score": float(score),
                    "steps": step + 1,
                    "oracle_calls": calls,
                    "status": "tmap_chain",
                    "source": "tmap_chain",
                    "fallback_steps": 0,
                    "anchor_position": -1,
                    "distance_rank": -1,
                }

        return best, calls

    for outer in range(1, max_outer + 1):
        if max_grad_evals is not None:
            remaining_calls = (max_grad_evals - total_oracle_calls * K) // K
            if remaining_calls <= 0:
                termination_reason = "max_grad_evals"
                break
        else:
            remaining_calls = max_inner * max(1, candidate_anchor_count) * len(normalization_powers)

        pc_value, lam = _maximise_GN(
            bundle,
            prev_lam=outer_prev_lam,
            solver=lambda_solver,
            max_starts=lambda_max_starts,
        )
        outer_prev_lam = lam
        pc_history.append(pc_value)
        lambda_history.append(lam.copy())

        if stop_threshold is not None and pc_value <= stop_threshold:
            termination_reason = (
                "paper_gn_threshold"
                if epsilon is not None
                else "target_reached"
            )
            _checkpoint(f"outer {outer}/{max_outer} ({termination_reason})")
            break

        anchor_array = np.vstack(anchor_lambdas)
        order = np.argsort(np.linalg.norm(anchor_array - lam[None, :], axis=1))
        order = order[: min(candidate_anchor_count, len(order))]

        start_specs: List[Tuple[str, int, int, int]] = []
        seen_bundle_indices = set()
        if use_active_best_start:
            active_scores = [scalarized_gn(lam, grad) for grad in bundle.grads]
            active_best_index = int(np.argmin(active_scores))
            start_specs.append(("active_best", -1, active_best_index, -1))
            seen_bundle_indices.add(active_best_index)
        for distance_rank, anchor_position in enumerate(order):
            bundle_index = anchor_indices[int(anchor_position)]
            if bundle_index in seen_bundle_indices:
                continue
            start_specs.append((
                "nearest_anchor",
                int(anchor_position),
                int(bundle_index),
                int(distance_rank),
            ))
            seen_bundle_indices.add(bundle_index)

        candidates: List[Dict] = []
        outer_calls = 0
        outer_steps = 0

        for start_kind, anchor_position, bundle_index, distance_rank in start_specs:
            if max_grad_evals is not None and outer_calls >= remaining_calls:
                break
            for power in normalization_powers:
                if max_grad_evals is not None and outer_calls >= remaining_calls:
                    break
                step_cap = max_inner
                if max_grad_evals is not None:
                    step_cap = min(step_cap, int(remaining_calls - outer_calls))
                if step_cap <= 0:
                    break

                inner = weighted_chebyshev_exact_gd(
                    lam,
                    bundle.points[bundle_index],
                    bundle.fvals[bundle_index],
                    bundle.grads[bundle_index],
                    L_arr,
                    objectives,
                    grad_objectives,
                    joint_oracle=joint_oracle,
                    max_steps=step_cap,
                    delta=delta,
                    step_scale=step_scale,
                    L_scale=L_scale,
                    gn_tolerance=gn_tolerance,
                    normalization_power=float(power),
                    normalization_floor=normalization_floor,
                    line_search=line_search,
                    return_last_if_no_gn_improvement=return_last_if_no_gn_improvement,
                    fallback_update=fallback_update,
                    max_oracle_calls=None if max_grad_evals is None else int(remaining_calls - outer_calls),
                    dual_solver=dual_solver,
                    require_ipopt=require_ipopt,
                )
                score = scalarized_gn(lam, inner["grads"])
                outer_calls += int(inner["oracle_calls"])
                outer_steps += int(inner["steps"])
                if np.isfinite(score):
                    source = f"chebyshev_alpha={float(power):g}"
                    if int(inner.get("fallback_steps", 0)) > 0:
                        source += "+gd_fallback"
                    candidates.append({
                        "x": inner["x"],
                        "fvals": inner["fvals"],
                        "grads": inner["grads"],
                        "score": float(score),
                        "steps": int(inner["steps"]),
                        "oracle_calls": int(inner["oracle_calls"]),
                        "status": str(inner["status"]),
                        "source": source,
                        "fallback_steps": int(inner.get("fallback_steps", 0)),
                        "start_kind": start_kind,
                        "anchor_position": int(anchor_position),
                        "distance_rank": int(distance_rank),
                    })

        if use_tmap_safeguard and (max_grad_evals is None or outer_calls < remaining_calls):
            tmap_cap = max_inner
            if max_grad_evals is not None:
                tmap_cap = min(tmap_cap, int(remaining_calls - outer_calls))
            tmap_candidate, tmap_calls = tmap_candidate_chain(lam, tmap_cap)
            outer_calls += tmap_calls
            outer_steps += tmap_calls
            tmap_safeguard_steps += tmap_calls
            if tmap_candidate is not None and np.isfinite(tmap_candidate["score"]):
                candidates.append(tmap_candidate)

        total_oracle_calls += outer_calls
        inner_steps_history.append(outer_steps)

        if not candidates:
            termination_reason = "no_finite_candidate"
            _checkpoint(f"outer {outer}/{max_outer} ({termination_reason})")
            break

        active_score = min(scalarized_gn(lam, grad) for grad in bundle.grads)
        required_improvement = max(
            improvement_tol,
            improvement_tol * max(1.0, active_score),
        )
        improving_candidates = [
            candidate for candidate in candidates
            if candidate["score"] < active_score - required_improvement
        ]
        rejected_candidate_count_history.append(
            len(candidates) - len(improving_candidates)
        )

        candidate_pool = improving_candidates if improving_candidates else candidates
        best = min(candidate_pool, key=lambda candidate: candidate["score"])
        if not append_non_improving and not improving_candidates:
            termination_reason = "no_candidate_improvement"
            _checkpoint(f"outer {outer}/{max_outer} ({termination_reason})")
            break

        bundle.points.append(np.asarray(best["x"], dtype=float).copy())
        bundle.fvals.append(np.asarray(best["fvals"], dtype=float).copy())
        bundle.grads.append(np.asarray(best["grads"], dtype=float).copy())
        anchor_lambdas.append(lam.copy())
        anchor_indices.append(bundle.m - 1)

        inner_status_history.append(str(best["status"]))
        candidate_source_history.append(str(best["source"]))
        candidate_score_history.append(float(best["score"]))
        fallback_steps_history.append(int(best.get("fallback_steps", 0)))

        cur_grad_evals = total_oracle_calls * K
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
            if stop_threshold is not None and cov_history[-1] <= stop_threshold:
                termination_reason = (
                    "paper_gn_threshold"
                    if epsilon is not None
                    else "target_reached"
                )
                break
        if budget_reached:
            termination_reason = "max_grad_evals"
            break

    return {
        "bundle": bundle,
        "anchors": list(zip(anchor_lambdas, anchor_indices)),
        "anchor_lambdas": anchor_lambdas,
        "anchor_points": [bundle.points[index] for index in anchor_indices],
        "cpu_times": cpu_times,
        "cov_history": cov_history,
        "pc_history": pc_history,
        "total_iters_history": total_iters_history,
        "grad_evals_history": grad_evals_history,
        "inner_steps_history": inner_steps_history,
        "lambda_history": lambda_history,
        "inner_status_history": inner_status_history,
        "candidate_source_history": candidate_source_history,
        "candidate_score_history": candidate_score_history,
        "rejected_candidate_count_history": rejected_candidate_count_history,
        "fallback_steps_history": fallback_steps_history,
        "tmap_safeguard_steps": tmap_safeguard_steps,
        "termination_reason": termination_reason,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "lambda_solver": lambda_solver if _HAS_IPOPT or lambda_solver == "slsqp" else "slsqp",
        "dual_solver": dual_solver if _HAS_IPOPT or dual_solver == "slsqp" else "slsqp",
        "normalization_powers": tuple(float(power) for power in normalization_powers),
        "line_search": line_search,
        "append_non_improving": append_non_improving,
        "return_last_if_no_gn_improvement": return_last_if_no_gn_improvement,
        "fallback_update": fallback_update,
        "use_tmap_safeguard": use_tmap_safeguard,
        "use_active_best_start": use_active_best_start,
        "total_oracle_calls": total_oracle_calls,
        "max_grad_evals": max_grad_evals,
    }
