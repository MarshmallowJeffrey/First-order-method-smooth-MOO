"""algorithm_ccp_without_256_checkpoints.py — Accelerated Algorithm 2
with the multistart-CCP λ-search (drop-in replacement for the IPOPT
tiers), A/B measurement variant (self-reported checkpoints).

NEW FILE (Aug 9, 2026).  Design record: Note/Aug_8_note.md.  The
comparison's IPOPT arm is ``algorithm_adaptive_fast`` with
``lambda_tier_mode="strict"``; this module is its CCP twin.  Everything
except the λ-search is IMPORTED from
``algorithm_fast_without_256_checkpoints`` (momentum-SVRG inner loop,
Gram cache, gradient-equivalent accounting, checkpoint semantics) so
the two arms share one implementation of every common part; no
existing file is modified.

Differences to ``algorithm_adaptive_fast``:

* λ-search: one ``CCPLambdaSolver`` instance per run (cross-round pool,
  warm-started HiGHS game LP; see ccp_lambda_solver.py).  Single tier —
  the strict/cheap machinery and its knobs do not exist here.  IPOPT is
  never required.
* per-round solver telemetry is returned as ``ccp_stats_history``
  (rho, N_new, pool occupancy/cap/dropped counts, LP iterations,
  sandwich closures, wall time — the Aug-8 note's log fields).
* the returned dict carries ``lambda_solver="ccp"`` instead of
  ``lambda_tier_mode`` / ``tier_history``.

Checkpoint semantics are inherited from the without-256 track:
``cov_history`` records the method's OWN most recent λ-search value,
checkpoint 0 is backfilled with the round-1 value, and checkpoint
bookkeeping time is excluded from the CPU axis.  The outer stopping
rule is unchanged: stop when the λ-search value pc_val <= 2ε/3 (the
CCP value has exactly the same heuristic lower-bound status as the
IPOPT multistart value, so the two arms remain comparable).
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from bundle import prefer_fused_joint_oracle, validate_problem_inputs
from bundle_fast import BundleFast, prune_inactive
from algorithm_fast_without_256_checkpoints import _bundle_update_msvrg
from ccp_lambda_solver import CCPConfig, CCPLambdaSolver


def algorithm_adaptive_ccp(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    *,
    stoch_oracle,
    epsilon: Optional[float] = 1e-3,
    max_outer: int = 150,
    eval_every_n_grads: Optional[float] = None,
    max_grad_evals: Optional[float] = None,
    # --- CCP λ-search (replaces the IPOPT tier knobs) ---
    ccp_config: Optional[CCPConfig] = None,
    # --- Momentum-SVRG inner loop (identical to the fast module) ---
    msvrg_step_const: float = 0.1,
    msvrg_momentum: float = 0.9,
    msvrg_epoch_len: Optional[int] = None,
    msvrg_max_segments: int = 10,
    msvrg_trigger_rho: float = 0.7,
    msvrg_trigger_patience: int = 2,
    msvrg_rel_target: Optional[float] = None,
    # --- delivery-time pruning ---
    prune_grid_r: int = 10,
    return_pre_prune: bool = False,
    record_segment_checkpoints: bool = False,
    joint_oracle: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """Accelerated Algorithm 2, λ-search by multistart CCP.

    Same contract as ``algorithm_adaptive_fast`` (histories on the
    gradient-equivalent axis, self-reported ``cov_history``, delivery
    pruning), with the CCP solver stats appended per round.
    """
    L_arr, x0_arr = validate_problem_inputs(
        K, d, L, x0, objectives, grad_objectives
    )
    if epsilon is not None and (
        not np.isscalar(epsilon) or not np.isfinite(epsilon) or epsilon <= 0.0
    ):
        raise ValueError(f"epsilon must be finite and positive; got {epsilon!r}.")
    if msvrg_rel_target is not None and not (0.0 < msvrg_rel_target < 1.0):
        raise ValueError(
            f"msvrg_rel_target must lie in (0, 1) or be None; "
            f"got {msvrg_rel_target!r}."
        )

    lambda_solver = CCPLambdaSolver(
        K, ccp_config if ccp_config is not None else CCPConfig())

    joint_oracle = prefer_fused_joint_oracle(joint_oracle)
    bundle = BundleFast(K=K, d=d, L=L_arr)
    bundle.add_point(x0_arr, objectives, grad_objectives,
                     joint_oracle=joint_oracle)   # checkpoint-0 setup, uncharged

    n = stoch_oracle.n
    ifo_start = stoch_oracle.ifo_count

    cpu_times: List[float] = []
    cov_history: List[float] = []
    m_history: List[int] = []
    pc_history: List[float] = []
    ccp_stats_history: List[Dict] = []
    inner_target_history: List[Optional[float]] = []
    lambda_history: List[np.ndarray] = []
    grad_evals_history: List[float] = []
    segments_history: List[int] = []
    inner_cap_hits = 0
    inner_cap_warned = False
    safeguard_warned = False

    joint_calls = 0          # charged joint-oracle calls (excludes x0 setup)
    grad_equiv = 0.0         # joint_calls*K + minibatch IFO * K/n
    grad_equiv_at_ckpt = 0.0
    lambda_search_seconds = 0.0
    L_scale = 1.0
    stop_reason = "round_fuse"

    checkpoint_overhead = 0.0
    t_start = time.time()

    def _checkpoint(label: str) -> None:
        nonlocal checkpoint_overhead
        cpu_times.append(time.time() - t_start - checkpoint_overhead)
        ck_t0 = time.time()
        cov = pc_history[-1] if pc_history else float("nan")
        cov_history.append(cov)
        m_history.append(int(bundle.m))
        checkpoint_overhead += time.time() - ck_t0
        grad_evals_history.append(float(grad_equiv))
        if verbose:
            print(
                f"  CCP {label} | t={cpu_times[-1]:.2f}s | bundle={bundle.m} "
                f"| grad_equiv={grad_equiv:.1f} | self-reported pc={cov:.4e}",
                flush=True,
            )

    _checkpoint(f"outer 0/{max_outer}")
    cpu_times[0] = 0.0
    t_start = time.time()
    checkpoint_overhead = 0.0

    eps_inner = None if epsilon is None else epsilon / 3.0
    stop_line = None if epsilon is None else 2.0 * epsilon / 3.0

    def _seg_ckpt(fe_so_far: int) -> None:
        # Recording only (Jul 27 semantics): refresh the running grad
        # counter and record a checkpoint when the cadence is due.
        nonlocal grad_equiv, grad_equiv_at_ckpt
        grad_equiv = ((joint_calls + fe_so_far) * K
                      + (stoch_oracle.ifo_count - ifo_start) * K / float(n))
        if (eval_every_n_grads is None
                or (grad_equiv - grad_equiv_at_ckpt) >= eval_every_n_grads):
            _checkpoint(f"outer {outer} seg {fe_so_far}")
            grad_equiv_at_ckpt = grad_equiv

    for outer in range(1, max_outer + 1):
        if max_grad_evals is not None and grad_equiv + K > max_grad_evals:
            stop_reason = "budget"
            break

        ls_t0 = time.time()
        pc_val, lam = lambda_solver.solve(bundle.gram_stack(),
                                          epsilon=epsilon)
        lambda_search_seconds += time.time() - ls_t0
        pc_history.append(pc_val)
        ccp_stats_history.append(lambda_solver.stats_last)
        lambda_history.append(lam.copy())

        if stop_line is not None and pc_val <= stop_line:
            stop_reason = "epsilon_certified"
            _checkpoint(f"outer {outer}/{max_outer}")
            break

        # Relative inner target: same variant switch as the fast module
        # (absolute eps/3 floor keeps the endgame the original algorithm).
        if eps_inner is None:
            inner_target = None
        elif msvrg_rel_target is not None:
            inner_target = max(eps_inner, msvrg_rel_target * pc_val)
        else:
            inner_target = eps_inner
        inner_target_history.append(inner_target)

        inner = _bundle_update_msvrg(
            bundle, lam, objectives, grad_objectives, joint_oracle,
            stoch_oracle,
            eps_inner=inner_target, L_scale=L_scale,
            step_const=msvrg_step_const, momentum=msvrg_momentum,
            epoch_len=msvrg_epoch_len, max_segments=msvrg_max_segments,
            trigger_rho=msvrg_trigger_rho,
            trigger_patience=msvrg_trigger_patience,
            on_full_eval=(_seg_ckpt if record_segment_checkpoints else None),
        )
        if inner["L_scale"] > L_scale:
            if not safeguard_warned:
                warnings.warn(
                    "Descent safeguard fired: supplied L underestimates the "
                    "curvature along the iterates; step sizes reduced "
                    "adaptively (L_scale doubled).",
                    RuntimeWarning, stacklevel=2,
                )
                safeguard_warned = True
            L_scale = inner["L_scale"]
        if inner["target_met"] is False:
            inner_cap_hits += 1
            if not inner_cap_warned:
                warnings.warn(
                    "epsilon mode: an inner loop exhausted its segment cap "
                    "before reaching the round's inner target at the active "
                    "lambda; the Algorithm 2 termination argument does not "
                    "apply to such rounds.",
                    RuntimeWarning, stacklevel=2,
                )
                inner_cap_warned = True

        joint_calls += inner["full_evals"]
        grad_equiv = (joint_calls * K
                      + (stoch_oracle.ifo_count - ifo_start) * K / float(n))
        segments_history.append(inner["segments"])

        do_ckpt = (
            eval_every_n_grads is None
            or (grad_equiv - grad_equiv_at_ckpt) >= eval_every_n_grads
            or outer == max_outer
        )
        if do_ckpt:
            _checkpoint(f"outer {outer}/{max_outer}")
            grad_equiv_at_ckpt = grad_equiv

    # Backfill checkpoint 0 with the round-1 search value (same {x0} bundle).
    if cov_history and np.isnan(cov_history[0]) and pc_history:
        cov_history[0] = float(pc_history[0])

    # ---- delivery-time pruning (run-time policy stays inclusive) ----
    pre_prune: Optional[Dict] = None
    if return_pre_prune:
        pre_prune = {
            "points": np.asarray(bundle.points, dtype=float).copy(),
            "fvals": np.asarray(bundle.fvals, dtype=float).copy(),
            "gram_stack": bundle.gram_stack().copy(),
        }
    extra = [lambda_history[-1]] if lambda_history else None
    prune_report = prune_inactive(
        bundle, grid_resolution=prune_grid_r, extra_lams=extra,
    )

    return {
        "bundle": bundle,
        "cpu_times": cpu_times,
        "cov_history": cov_history,
        "m_history": m_history,
        "pre_prune": pre_prune,
        "pc_history": pc_history,
        "ccp_stats_history": ccp_stats_history,
        "inner_target_history": inner_target_history,
        "msvrg_rel_target": msvrg_rel_target,
        "lambda_history": lambda_history,
        "grad_evals_history": grad_evals_history,
        "segments_history": segments_history,
        "joint_calls": joint_calls,
        "ifo_minibatch_total": int(stoch_oracle.ifo_count - ifo_start),
        "grad_equiv_total": float(grad_equiv),
        "L_scale_final": L_scale,
        "inner_cap_hits": inner_cap_hits,
        "lambda_search_seconds": float(lambda_search_seconds),
        "epsilon": epsilon,
        "stop_reason": stop_reason,
        "prune_report": prune_report,
        "lambda_solver": "ccp",
        "max_outer": max_outer,
    }
