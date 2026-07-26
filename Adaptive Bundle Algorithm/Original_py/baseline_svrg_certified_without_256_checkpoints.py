"""baseline_svrg_certified_without_256_checkpoints.py

Uniform-discretisation baseline with the SAME Momentum-SVRG inner solver
as the fast adaptive method, certification-style, on the
without-256-checkpoints track.

NEW FILE (July 20, 2026).  Nothing in the original files is modified;
this module IMPORTS the untouched grid utilities from
``baseline_without_256_checkpoints`` and (lazily) the strict λ-search
from ``algorithm_fast_without_256_checkpoints``.  Design record:
``Note/Jul_20_note.md``.

EDITED July 25, 2026 (user-authorised in-place change, see
``Note/Jul_25_note.md``): the per-checkpoint trajectory metric is
restored — ``cov_history`` records max over ALL grid nodes of the
best-known value ``best_val`` at each checkpoint, exactly the July-8
baseline's lag semantics (every node initialised at its x0 score;
unvisited nodes keep their last known value; weights BETWEEN grid nodes
never enter).  Pure cached bookkeeping, no oracle calls, so both cost
axes are unaffected.

What it is
----------
Algorithm 1 (uniform grid over Δ_K, snake warm-start order), but each
grid node is solved by the v3 fast method's inner loop — segments of
stratified minibatch Momentum-SVRG steps with the descent safeguard —
instead of fixed-count full-batch GD passes.  Two-level tolerance:

* ``node_tol``      — the node's SERVICE certificate: node λ_i counts as
  served when some delivered point x has ‖∇F_{λ_i}(x)‖² = λ_iᵀ G(x) λ_i
  ≤ node_tol (G = J Jᵀ the cached Gram at x).  Exact, deterministic:
  randomness affects only how long solving takes, never the certificate
  (same Las-Vegas structure as the fast method: every acceptance is on
  FULL gradients).
* ``solve_target``  — the per-node SVRG target (default node_tol/4,
  mirroring v3's rel_target=0.25): each solved node is pushed a factor
  below the service level so its delivered point can also serve nearby
  nodes.

Share modes
-----------
* ``"gram"`` (default): every delivered point's K×K Gram is scored
  against all still-unserved nodes (a vectorised λᵀGλ sweep — cached
  linear algebra, zero oracle calls, the same cache discipline the fast
  method's whole λ-search runs on).  This is "Algorithm 1 + memoisation":
  the delivered SET certifies the grid, one point may serve many nodes.
* ``"none"``: only the chain warm-start point may serve a node at visit
  time (the July-8 certification baseline's check-before-stepping,
  with the gradient reused from the cache instead of re-paid).

Accounting (identical basis to the fast method)
-----------------------------------------------
grad-equivalents = joint_calls·K + minibatch_IFO·K/n.  Epoch-end full
joint calls are charged at K everywhere (legacy-consistent; they also
feed the share/scoring Grams).  Minibatch steps are charged by the rows
actually consumed: rows of zero-weight classes are dropped from the
batch BEFORE the oracle call (their gradient contribution is exactly
zero, so the estimator is unchanged while boundary nodes stop paying
for samples they cannot use).  The x0 Jacobian is checkpoint-0 setup,
uncharged, exactly like the fast method's initial ``add_point``.  CPU
axis: grid construction and the final scoring/verification are excluded
(preprocessing / metric); serve-sweeps are algorithm work and are
included.

Delivery-time metric (still without-256-checkpoints)
----------------------------------------------------
After the run, ONE strict-tier in-family λ-search
(``_maximise_GN_fast``, 64 starts — the same maximiser that signs the
fast method's certificates, NOT the external 256-start yardstick) is
run on the delivered Gram stack and reported as
``delivered_gn_strict``.  Its cost is excluded from both axes and timed
separately in ``metric_seconds``.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

from bundle import (
    prefer_fused_joint_oracle,
    validate_oracle_output,
    validate_problem_inputs,
)
from baseline_without_256_checkpoints import (
    _sort_grid_for_warmstart,
    _uniform_simplex_grid,
)


# =====================================================================
#  Minimal Gram-stack holder accepted by _maximise_GN_fast
# =====================================================================
class _GramSet:
    """Duck-typed stand-in for BundleFast: exposes m / K / gram_stack()."""

    def __init__(self, grams: List[np.ndarray], K: int) -> None:
        self._Ms = np.asarray(grams, dtype=float)
        self.K = int(K)
        self.m = int(self._Ms.shape[0]) if self._Ms.size else 0

    def gram_stack(self) -> np.ndarray:
        return self._Ms


def _support_batch(batch: List[np.ndarray], lam: np.ndarray) -> List[np.ndarray]:
    """Drop rows of zero-weight classes (their gradient term is exactly 0).

    The stratified estimator over the remaining classes is unchanged;
    the oracle then charges IFO only for rows actually consumed.
    """
    return [b if lam[i] > 0.0 else b[:0] for i, b in enumerate(batch)]


def _serve_sweep(grid: np.ndarray, best_val: np.ndarray, served: np.ndarray,
                 gram: np.ndarray, node_tol: float,
                 chunk: int = 262144, mark: bool = True) -> int:
    """Score one delivered Gram against all nodes; mark newly served ones.

    Pure cached linear algebra (λᵀGλ per node), no oracle calls.  Returns
    the number of nodes newly served by this point.  ``mark=False`` only
    updates ``best_val`` (metric bookkeeping) without granting service —
    used to initialise every node's value at x0 in share_mode="none",
    replicating the July-8 baseline's metric initialisation (Jul 25 edit).
    """
    n_new = 0
    N = grid.shape[0]
    for s in range(0, N, chunk):
        lam_c = grid[s:s + chunk]
        vals = np.einsum("nk,kl,nl->n", lam_c, gram, lam_c)
        blk_best = best_val[s:s + chunk]
        np.minimum(blk_best, vals, out=blk_best)
        if mark:
            blk_served = served[s:s + chunk]
            newly = (~blk_served) & (blk_best <= node_tol)
            n_new += int(newly.sum())
            blk_served |= newly
    return n_new


def baseline_svrg_certified(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    resolution: int,
    stoch_oracle,
    *,
    node_tol: float = 0.02,
    solve_target: Optional[float] = None,   # default: node_tol / 4
    share_mode: str = "gram",
    msvrg_step_const: float = 0.1,
    msvrg_momentum: float = 0.5,
    msvrg_epoch_len: Optional[int] = None,
    msvrg_max_segments: int = 10,
    msvrg_trigger_rho: float = 0.7,
    msvrg_trigger_patience: int = 2,
    max_segment_retries: int = 4,
    max_grad_evals: Optional[float] = None,
    max_wall_seconds: Optional[float] = None,
    eval_every_n_grads: Optional[float] = 4500.0,
    lambda_max_starts: int = 64,
    global_stop_gn: Optional[float] = None,
    joint_oracle: Optional[Callable] = None,
    verify_certificates: bool = True,
    return_points: bool = False,
    return_grams: bool = False,
    verbose: bool = False,
) -> Dict:
    """Certify a uniform Δ_K grid with the v3 Momentum-SVRG inner solver.

    Returns a dict of histories, counters and the delivery-time strict
    GN* of the delivered point set (see module docstring).
    """
    L_arr, x0_arr = validate_problem_inputs(
        K, d, L, x0, objectives, grad_objectives
    )
    if share_mode not in {"gram", "none"}:
        raise ValueError("share_mode must be 'gram' or 'none'.")
    if not (np.isfinite(node_tol) and node_tol > 0.0):
        raise ValueError(f"node_tol must be finite and positive; got {node_tol!r}.")
    if solve_target is None:
        solve_target = node_tol / 4.0
    if not (0.0 < solve_target <= node_tol):
        raise ValueError(
            f"solve_target must lie in (0, node_tol]; got {solve_target!r}."
        )

    joint_oracle = prefer_fused_joint_oracle(joint_oracle)
    n = stoch_oracle.n
    ifo_start = stoch_oracle.ifo_count
    epoch_len = (msvrg_epoch_len if msvrg_epoch_len is not None
                 else max(1, int(np.ceil(n / float(stoch_oracle.b_total)))))

    # ---- preprocessing (excluded from the CPU axis) ----
    grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, resolution))
    N = grid.shape[0]
    served = np.zeros(N, dtype=bool)
    best_val = np.full(N, np.inf)

    # x0 Jacobian: checkpoint-0 setup, uncharged (mirrors the fast
    # method's initial add_point).
    f0, J0 = validate_oracle_output(*joint_oracle(x0_arr), K, d)
    # Point COORDINATES are kept only on request (return_points): at scale
    # they dominate memory (segments × d floats) while certification,
    # sharing and scoring all run on the K×K Grams.  The delivered set is
    # re-derivable exactly (deterministic seeds); see Note/Jul_20_note.md.
    delivered_points: List[np.ndarray] = [x0_arr.copy()] if return_points else []
    n_delivered = 1
    delivered_grams: List[np.ndarray] = [J0 @ J0.T]
    delivered_serve_counts: List[int] = [0]
    if share_mode == "gram":
        delivered_serve_counts[0] = _serve_sweep(
            grid, best_val, served, delivered_grams[0], node_tol)
    else:
        # Jul 25 edit: metric initialisation only (old-baseline
        # semantics) — every node's best-known value starts at its x0
        # score; service in "none" mode is granted strictly at visit.
        _serve_sweep(grid, best_val, served, delivered_grams[0], node_tol,
                     mark=False)

    chain_x, chain_J, chain_f = x0_arr.copy(), J0, f0

    # ---- counters / histories ----
    joint_calls = 0
    grad_equiv = 0.0
    grad_at_ckpt = 0.0
    L_scale = 1.0
    solved_nodes = 0
    served_by_share = int(served.sum())
    served_by_chain = 0
    served_above_target = 0
    censored_nodes = 0
    safeguard_retries = 0
    segments_total = 0
    minibatch_steps_total = 0
    stop_reason = "completed"
    safeguard_warned = False

    cpu_times: List[float] = []
    grad_evals_history: List[float] = []
    served_history: List[int] = []
    delivered_history: List[int] = []
    cov_history: List[float] = []   # Jul 25 edit: old-baseline grid meter

    t_start = time.time()

    def _checkpoint(label: str) -> None:
        cpu_times.append(time.time() - t_start)
        grad_evals_history.append(float(grad_equiv))
        served_history.append(int(served.sum()))
        delivered_history.append(n_delivered)
        # Max over ALL grid nodes of the best value known so far for that
        # node (x0-initialised).  O(N) on a cached array — no oracle work,
        # so this metric is free on both cost axes.
        cov_history.append(float(best_val.max()))
        if verbose:
            print(
                f"  BL-SVRG {label} | t={cpu_times[-1]:.2f}s "
                f"| grad_equiv={grad_equiv:.1f} "
                f"| served={served_history[-1]}/{N} "
                f"| delivered={delivered_history[-1]} "
                f"| censored={censored_nodes} "
                f"| worst={cov_history[-1]:.4e}",
                flush=True,
            )

    _checkpoint("node 0")
    cpu_times[0] = 0.0
    t_start = time.time()

    def _grad_equiv_now() -> float:
        return (joint_calls * K
                + (stoch_oracle.ifo_count - ifo_start) * K / float(n))

    # Jul 26 (equal-level stop, user request): optional global stopping
    # criterion — stop as soon as the delivered set's strict full-simplex
    # GN* is <= global_stop_gn, using the SAME two-tier stop-verify
    # discipline as the fast method (cheap screen at checkpoint cadence,
    # only a strict search may sign the stop).  With this option the
    # searches DECIDE termination, so their time deliberately stays
    # INSIDE the CPU axis, mirroring the fast method's lambda-search
    # accounting.  Default None: original serve-all-nodes behaviour,
    # unchanged.
    _gs_maximise = None
    gs_prev_lam: Optional[np.ndarray] = None
    if global_stop_gn is not None:
        from algorithm_fast_without_256_checkpoints import (
            _maximise_GN_fast as _gs_maximise)

    # =================================================================
    #  Main sweep: snake order over the grid
    # =================================================================
    for idx in range(N):
        if served[idx]:
            continue
        if max_wall_seconds is not None and (time.time() - t_start) >= max_wall_seconds:
            stop_reason = "wall_fuse"
            break
        if max_grad_evals is not None and grad_equiv + K > max_grad_evals:
            stop_reason = "grad_fuse"
            break

        lam = grid[idx]

        # Chain service check (free: cached Jacobian reweighting).  In
        # gram mode this is redundant — the chain point is delivered and
        # already swept — so it only fires in share_mode="none".
        if share_mode == "none":
            g_chain = chain_J.T @ lam
            val_chain = float(g_chain @ g_chain)
            best_val[idx] = min(best_val[idx], val_chain)
            if val_chain <= node_tol:
                served[idx] = True
                served_by_chain += 1
                continue

        # ---- solve this node with Momentum-SVRG segments ----
        anchor_x, anchor_J, anchor_f = chain_x, chain_J, chain_f
        node_solved = False
        node_val = np.inf
        for _seg in range(msvrg_max_segments):
            if max_grad_evals is not None and grad_equiv + K > max_grad_evals:
                stop_reason = "grad_fuse"
                break
            g_a_full = anchor_J.T @ lam            # exact ∇F_λ(anchor), cached
            F_a = float(anchor_f @ lam)
            seg_retry = 0
            while True:
                Ll = float(lam @ L_arr) * L_scale
                eta = msvrg_step_const / Ll
                stoch_oracle.set_anchor(anchor_x)
                y = anchor_x.copy()
                u_vec = np.zeros(d)
                consec = 0
                for _t in range(epoch_len):
                    batch = _support_batch(stoch_oracle.sample_batch(), lam)
                    g_y_S, g_a_S = stoch_oracle.grad_pair(y, lam, batch)
                    v = g_y_S - g_a_S + g_a_full
                    u_vec = msvrg_momentum * u_vec + v
                    y = y - eta * u_vec
                    minibatch_steps_total += 1
                    if float(v @ v) <= msvrg_trigger_rho * solve_target:
                        consec += 1
                        if consec >= msvrg_trigger_patience:
                            break
                    else:
                        consec = 0

                # Segment end: full evaluation (charged), always delivered.
                f_y, J_y = validate_oracle_output(*joint_oracle(y), K, d)
                joint_calls += 1
                grad_equiv = _grad_equiv_now()
                gram_y = J_y @ J_y.T
                if return_points:
                    delivered_points.append(y.copy())
                n_delivered += 1
                delivered_grams.append(gram_y)
                if share_mode == "gram":
                    n_new = _serve_sweep(grid, best_val, served, gram_y, node_tol)
                    delivered_serve_counts.append(n_new)
                    served_by_share += n_new
                else:
                    val_y = float(lam @ gram_y @ lam)
                    best_val[idx] = min(best_val[idx], val_y)
                    if val_y <= node_tol:
                        served[idx] = True
                    delivered_serve_counts.append(int(served[idx]))
                node_val = float(lam @ gram_y @ lam)

                F_y = float(f_y @ lam)
                if F_y > F_a + 1e-10 * (1.0 + abs(F_a)):
                    # Descent safeguard (identical to the fast inner loop):
                    # L underestimates the curvature here — halve the step,
                    # reset momentum, re-run from the SAME anchor.  The
                    # violating point stays delivered (paid for; harmless
                    # under the min).
                    L_scale *= 2.0
                    safeguard_retries += 1
                    seg_retry += 1
                    if not safeguard_warned:
                        warnings.warn(
                            "Descent safeguard fired: supplied L "
                            "underestimates the curvature along the "
                            "iterates; step sizes reduced adaptively.",
                            RuntimeWarning, stacklevel=2,
                        )
                        safeguard_warned = True
                    if L_scale > 2.0 ** 60:
                        raise RuntimeError(
                            "Descent safeguard scaled L beyond 2^60; the "
                            "objectives do not appear L-smooth along the "
                            "iterates."
                        )
                    if seg_retry <= max_segment_retries:
                        continue
                break

            segments_total += 1
            # Anchor advances to the segment end-point (monotone in F_λ by
            # the safeguard) — the single-λ simplification of the fast
            # method's T-map argmin.
            anchor_x, anchor_J, anchor_f = y, J_y, f_y

            if node_val <= solve_target:
                node_solved = True
                break

        if stop_reason == "grad_fuse":
            break

        # Node post-mortem.  Consult best_val[idx] (maintained by the
        # sweeps / chain checks), not the LAST segment value: the SVRG
        # value is not monotone, and an earlier delivered point may
        # already certify this node.
        if node_solved:
            solved_nodes += 1
            served[idx] = True
        elif best_val[idx] <= node_tol:
            # Segment cap hit, but the service certificate still holds.
            solved_nodes += 1
            served_above_target += 1
            served[idx] = True
        else:
            censored_nodes += 1

        # The chain warm start advances to this node's last full-evaluated
        # point regardless of outcome (ℓ₁-adjacent warm start for the next
        # unserved node).
        chain_x, chain_J, chain_f = anchor_x, anchor_J, anchor_f

        due_ckpt = (eval_every_n_grads is None
                    or (grad_equiv - grad_at_ckpt) >= eval_every_n_grads)
        if due_ckpt:
            _checkpoint(f"node {idx + 1}")
            grad_at_ckpt = grad_equiv
            if _gs_maximise is not None:
                gset = _GramSet(delivered_grams, K)
                v_cheap, lam_c = _gs_maximise(gset, prev_lam=gs_prev_lam,
                                              tier="cheap")
                gs_prev_lam = lam_c
                if float(v_cheap) <= global_stop_gn:
                    v_strict, lam_s = _gs_maximise(
                        gset, prev_lam=lam_c, tier="strict",
                        max_starts=lambda_max_starts)
                    gs_prev_lam = lam_s
                    if float(v_strict) <= global_stop_gn:
                        stop_reason = "global_stop_gn"
                        break

    wall_seconds = time.time() - t_start
    _checkpoint("final")
    cpu_times[-1] = wall_seconds

    # =================================================================
    #  Delivery-time metric + certificate verification (excluded axes)
    # =================================================================
    t_metric = time.time()
    from algorithm_fast_without_256_checkpoints import _maximise_GN_fast
    gram_set = _GramSet(delivered_grams, K)

    # Jul 25 fix (see Note/Jul_25_note.md §6): comparable-meter
    # trajectory.  For every checkpoint, the strict full-simplex GN* of
    # the delivered-set PREFIX at that moment — the same meter family as
    # the fast curve's y-axis, so the two methods' lines can share axes
    # without cross-meter misreads.  Post-hoc on cached Grams, warm-
    # started along the prefix chain; cost lands in metric_seconds,
    # excluded from both cost axes like all measurement on this track.
    delivered_gn_strict_history: List[float] = []
    _prev_lam = None
    for m_j in delivered_history:
        sub = _GramSet(delivered_grams[:m_j], K)
        v_j, _prev_lam = _maximise_GN_fast(
            sub, prev_lam=_prev_lam, tier="strict",
            max_starts=lambda_max_starts,
        )
        delivered_gn_strict_history.append(float(v_j))

    delivered_gn_strict, lam_star = _maximise_GN_fast(
        gram_set, prev_lam=_prev_lam, tier="strict",
        max_starts=lambda_max_starts,
    )
    # Both searches lower-bound the true sup; the better estimate of a
    # maximum is the LARGER value (never understate the criterion).
    delivered_gn_strict = max(float(delivered_gn_strict),
                              delivered_gn_strict_history[-1])
    delivered_gn_strict_history[-1] = delivered_gn_strict

    verify_report: Optional[Dict] = None
    if verify_certificates:
        Ms = gram_set.gram_stack()
        served_idx = np.flatnonzero(served)
        sample_cap = 200_000
        if served_idx.size > sample_cap:
            rng = np.random.RandomState(0)
            check_idx = rng.choice(served_idx, size=sample_cap, replace=False)
            sampled = True
        else:
            check_idx, sampled = served_idx, False
        worst = 0.0
        for s in range(0, check_idx.size, 4096):
            lam_c = grid[check_idx[s:s + 4096]]
            vals = np.einsum("nk,mkl,nl->nm", lam_c, Ms, lam_c).min(axis=1)
            worst = max(worst, float(vals.max()) if vals.size else 0.0)
        ok = worst <= node_tol * (1.0 + 1e-9)
        verify_report = {
            "checked_nodes": int(check_idx.size),
            "sampled": sampled,
            "worst_served_value": worst,
            "certificates_hold": bool(ok),
        }
        if not ok:
            raise AssertionError(
                f"Certificate verification failed: a served node scores "
                f"{worst!r} > node_tol={node_tol!r}."
            )
    metric_seconds = time.time() - t_metric

    return {
        "resolution": resolution,
        "n_nodes": N,
        "share_mode": share_mode,
        "node_tol": node_tol,
        "solve_target": solve_target,
        "cpu_times": cpu_times,
        "grad_evals_history": grad_evals_history,
        "served_history": served_history,
        "delivered_history": delivered_history,
        "cov_history": cov_history,
        "delivered_gn_strict_history": delivered_gn_strict_history,
        "n_served": int(served.sum()),
        "n_unserved": int(N - served.sum()),
        "solved_nodes": solved_nodes,
        "served_by_share": served_by_share,
        "served_by_chain": served_by_chain,
        "served_above_target": served_above_target,
        "censored_nodes": censored_nodes,
        "n_delivered": n_delivered,
        "delivered_serve_counts": delivered_serve_counts,
        "segments_total": segments_total,
        "minibatch_steps_total": minibatch_steps_total,
        "safeguard_retries": safeguard_retries,
        "L_scale_final": L_scale,
        "joint_calls": joint_calls,
        "ifo_minibatch_total": int(stoch_oracle.ifo_count - ifo_start),
        "grad_equiv_total": float(grad_equiv),
        "wall_seconds": wall_seconds,
        "metric_seconds": metric_seconds,
        "delivered_gn_strict": float(delivered_gn_strict),
        "lambda_star": np.asarray(lam_star, dtype=float),
        "grid_worst_best_val": float(np.max(best_val[np.isfinite(best_val)]))
            if np.any(np.isfinite(best_val)) else float("inf"),
        "global_stop_gn": global_stop_gn,
        "stop_reason": stop_reason,
        "verify_report": verify_report,
        "epoch_len": epoch_len,
        "delivered_points": (np.asarray(delivered_points, dtype=float)
                             if return_points else None),
        # Jul 25 addition (between-node-gap figures, Note §7): the cached
        # delivery data needed for post-hoc gap analysis — the delivered
        # Gram stack, every node's final certified value, and the grid.
        "delivered_grams_stack": (np.asarray(delivered_grams)
                                  if return_grams else None),
        "node_best_val": (best_val.copy() if return_grams else None),
        "grid": (grid if return_grams else None),
    }
