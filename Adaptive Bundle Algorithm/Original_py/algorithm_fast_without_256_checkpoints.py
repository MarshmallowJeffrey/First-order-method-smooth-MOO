"""algorithm_fast_without_256_checkpoints.py  –  Accelerated Algorithm 2,
A/B measurement variant (self-reported checkpoints).

NEW FILE (July 15, 2026).  Part of the ``_fast`` implementation set; the
original ``algorithm_without_256_checkpoints.py`` is untouched and remains
the reference for the unaccelerated method.  This module implements the
acceleration plan (see the July 15 plan document and the run folder's
README):

1.  **Gram-path GN evaluation** (exact rewrite, not an approximation):
        GN(λ) = min_i ‖J_i^T λ‖² = min_i λ^T M_i λ,   ∇_λ = 2 M_{i*} λ,
    with M_i = J_i J_i^T cached by ``bundle_fast.BundleFast``.  Every GN
    evaluation drops from O(m·K·d) to O(m·K²), decoupling the λ-search
    from d entirely.  Equivalence to the einsum path is asserted to 1e-12
    by ``sanity_checks_fast.py``.

2.  **Two-tier λ-search with stop-verify**: a cheap tier (centroid +
    vertices + prev_lam ≈ K+2 starts, loose IPOPT settings) for ordinary
    rounds, and the strict tier (the original start set / tolerances) that
    alone may sign a stopping certificate.  A cheap value ≤ 2ε/3 triggers
    a strict re-solve; only a strict value ≤ 2ε/3 stops the run.  A failed
    verify hands the strict λ to the inner loop (a legitimate violator)
    and, with ``sticky_strict``, stays strict thereafter.  NOTE: after the
    Gram rewrite the strict tier is itself sub-second, so production runs
    may simply set ``lambda_tier_mode="strict"`` — the recording metric
    then matches the legacy curves exactly.  The two-tier machinery is
    kept for larger K / larger bundles.

3.  **Momentum-SVRG inner loop** (``_bundle_update_msvrg``): at fixed λ the
    inner task is to drive GN(λ; B) ≤ ε/3 — a single smooth nonconvex
    problem in F_λ.  Segments replace per-step full evaluations:

        anchor  a = argmin_i {F_λ(x_i) − ‖∇F_λ(x_i)‖²/(2 L_λ)}  (T-map rule)
        u ← 0, y ← a, g_a = J_a^T λ  (free from the bundle cache)
        repeat ≤ p_seg times, interruptible by the early trigger:
            v = ∇f_{λ,S}(y) − ∇f_{λ,S}(a) + g_a     (SVRG estimator)
            u = β u + v;   y ← y − η u,  η = step_const/(L_λ·L_scale)
            trigger: ‖v‖² ≤ ρ·(ε/3) for `patience` consecutive steps
        full joint evaluation of y → ALWAYS added to the bundle
        (inclusive policy) → exact Gram-path ε/3 check on full gradients.

    At the anchor v equals the full gradient, so β=0, p_seg=1, b=n, step_const=1
    reproduces the original T-map inner loop exactly (asserted by the
    sanity script).  Randomness can only delay the inner loop, never fake
    a certificate: every acceptance test runs on full gradients.

    Descent safeguard: if the segment end has F_λ(y) > F_λ(a) (with the
    original relative slack), L_scale doubles (η halves), momentum resets,
    and the segment re-runs from the same anchor; the violating point
    stays in the bundle (paid for, harmless under the min).  L_scale is
    monotone with the original 2^60 fuse.

4.  **Gradient-equivalent accounting**: the legacy axis assumed every
    inner step costs K full gradients.  Here a full joint call costs K
    grad-equivalents and a minibatch step costs 2b·K/n (classes partition
    the samples, so a joint call is n per-sample gradients).  Histories
    are recorded on the grad-equivalent axis (float); raw IFO and joint
    call counts are returned alongside.

Checkpoint semantics are inherited from the without-256 track:
``cov_history`` records the method's OWN most recent λ-search value (which
lags the bundle by the current round's inner work), checkpoint 0 is
backfilled with the round-1 value, and checkpoint bookkeeping time is
excluded from the CPU axis.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize as sp_minimize

try:
    from cyipopt import minimize_ipopt as _ipopt_minimize
    _HAS_IPOPT = True
    _IPOPT_IMPORT_ERROR = None
except (ImportError, OSError) as exc:  # pragma: no cover - environment guard
    _ipopt_minimize = None
    _HAS_IPOPT = False
    _IPOPT_IMPORT_ERROR = exc

from bundle import validate_problem_inputs, prefer_fused_joint_oracle
from bundle_fast import BundleFast, prune_inactive
from algorithm_without_256_checkpoints import _gn_multistart_set

__all__ = [
    "ipopt_available",
    "algorithm_adaptive_fast",
]


def ipopt_available() -> bool:
    return _HAS_IPOPT


# =====================================================================
#  Gram-path GN evaluation (exact identity; see module docstring)
# =====================================================================
def _gn_value_batched_gram(Ms: np.ndarray, lam: np.ndarray) -> float:
    """GN(λ; B) = min_i λ^T M_i λ  from the stacked Gram cache (m, K, K)."""
    vals = np.einsum('k,ikl,l->i', lam, Ms, lam)
    return float(np.min(vals))


def _gn_value_and_jac_batched_gram(
    Ms: np.ndarray, lam: np.ndarray
) -> Tuple[float, np.ndarray, int]:
    """Value, Danskin λ-gradient and argmin index on the Gram path.

    ∇_λ (λ^T M_i λ) = 2 M_i λ; at the (generically unique) argmin i* the
    piecewise-smooth min inherits 2 M_{i*} λ, exactly as the original
    einsum path inherits 2 J_{i*}(J_{i*}^T λ).
    """
    vals = np.einsum('k,ikl,l->i', lam, Ms, lam)
    i_star = int(np.argmin(vals))
    jac = 2.0 * (Ms[i_star] @ lam)
    return float(vals[i_star]), jac, i_star


# =====================================================================
#  λ multistart sets: strict tier reuses the original generator; the
#  cheap tier keeps only the O(K) blocks (centroid + vertices + prev).
# =====================================================================
def _gn_multistart_set_cheap(K: int, prev_lam: Optional[np.ndarray]):
    if K == 1:
        return [np.ones(1)]
    EPS = 1e-8
    starts = [np.full(K, 1.0 / K)]
    for k in range(K):
        e = np.full(K, EPS)
        e[k] = 1.0 - (K - 1) * EPS
        starts.append(e)
    if prev_lam is not None:
        starts.append(np.clip(prev_lam, EPS, 1.0))
    return starts


def _maximise_GN_fast(
    bundle: BundleFast,
    prev_lam: Optional[np.ndarray] = None,
    *,
    tier: str = "strict",
    max_starts: int = 64,
    strict_tol: float = 1e-8,
    strict_max_iter: int = 100,
    cheap_tol: float = 1e-4,
    cheap_max_iter: int = 30,
    solver: str = "ipopt",
) -> Tuple[float, np.ndarray]:
    """argmax_{λ ∈ Δ_K} GN(λ; B) on the Gram path, at the given tier.

    Identical multi-start scoring discipline to the original
    ``_maximise_GN`` (score every start at its simplex projection first;
    score solver answers at their projections; a failed local solve never
    loses ground), with the GN evaluator swapped for the O(m·K²) Gram
    path and per-tier start sets / IPOPT settings.
    """
    if tier not in {"strict", "cheap"}:
        raise ValueError("tier must be 'strict' or 'cheap'.")
    if bundle.m == 0:
        raise ValueError("Cannot maximise GN for an empty bundle.")

    K = bundle.K
    Ms = bundle.gram_stack()
    if K == 1:
        lam = np.ones(1)
        return _gn_value_batched_gram(Ms, lam), lam

    def neg_gn(lam):
        v = _gn_value_batched_gram(Ms, lam)
        return -v

    def neg_gn_jac(lam):
        _, j, _ = _gn_value_and_jac_batched_gram(Ms, lam)
        return -j

    con_eq = {"type": "eq",
              "fun": lambda l: float(np.sum(l) - 1.0),
              "jac": lambda l: np.ones(K)}
    constraints = [con_eq]
    bounds = [(1e-8, 1.0)] * K

    if tier == "strict":
        starts = _gn_multistart_set(K, prev_lam, max_starts)
        tol, max_iter = strict_tol, strict_max_iter
    else:
        starts = _gn_multistart_set_cheap(K, prev_lam)
        tol, max_iter = cheap_tol, cheap_max_iter

    use_ipopt = (solver == "ipopt") and _HAS_IPOPT
    if solver == "ipopt" and not _HAS_IPOPT:
        warnings.warn(
            "cyipopt/IPOPT unavailable; _maximise_GN_fast falls back to "
            f"SLSQP. Import error: {_IPOPT_IMPORT_ERROR}",
            RuntimeWarning, stacklevel=2,
        )

    def _project_simplex(lam_vec: np.ndarray) -> np.ndarray:
        lam_vec = np.maximum(np.asarray(lam_vec, dtype=float), 0.0)
        s = float(lam_vec.sum())
        if not np.isfinite(s) or s <= 0.0:
            return np.full(K, 1.0 / K)
        return lam_vec / s

    best_val = np.inf
    best_lam = _project_simplex(starts[0])
    for lam0 in starts:
        lam0 = _project_simplex(lam0)
        v0 = neg_gn(lam0)
        if v0 < best_val:
            best_val, best_lam = float(v0), lam0
        try:
            if use_ipopt:
                res = _ipopt_minimize(
                    neg_gn, lam0, jac=neg_gn_jac,
                    bounds=bounds, constraints=constraints,
                    options={
                        "print_level": 0,
                        "sb": "yes",
                        "tol": tol,
                        "max_iter": max_iter,
                        "hessian_approximation": "limited-memory",
                    },
                )
            else:
                res = sp_minimize(
                    neg_gn, lam0, jac=neg_gn_jac, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"ftol": max(tol, 1e-6), "maxiter": max_iter},
                )
        except Exception as exc:
            warnings.warn(
                f"GN local solve failed from start {lam0}: {exc}",
                RuntimeWarning, stacklevel=2,
            )
            continue
        lam_res = _project_simplex(res.x)
        v_res = neg_gn(lam_res)
        if np.isfinite(v_res) and v_res < best_val:
            best_val = float(v_res)
            best_lam = lam_res

    return float(-best_val), best_lam


# =====================================================================
#  Momentum-SVRG inner loop (segments; inclusive bundle policy)
# =====================================================================
def _bundle_update_msvrg(
    bundle: BundleFast,
    lam: np.ndarray,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    joint_oracle: Optional[Callable],
    stoch_oracle,
    *,
    eps_inner: Optional[float],
    L_scale: float,
    step_const: float = 0.1,
    momentum: float = 0.9,
    epoch_len: Optional[int] = None,
    max_segments: int = 10,
    trigger_rho: float = 0.7,
    trigger_patience: int = 2,
    max_segment_retries: int = 4,
) -> Dict:
    """One inner loop at fixed λ: segments of Momentum-SVRG steps.

    Returns a dict with segments/full_evals/minibatch_steps/L_scale/
    target_met (None when eps_inner is None) — see module docstring for
    the exact per-segment recipe.
    """
    K, d = bundle.K, bundle.d
    L_arr = bundle.L
    n = stoch_oracle.n
    b = stoch_oracle.b_total
    if epoch_len is None:
        epoch_len = max(1, int(np.ceil(n / float(b))))

    target_met: Optional[bool] = None if eps_inner is None else False
    segments = 0
    full_evals = 0
    minibatch_steps = 0
    retries = 0

    # ------------------------------------------------------------------
    # Per-round scalarised caches (July 16 fix for the stacked-copy
    # waste).  At fixed λ, F_λ(x_i), ∇F_λ(x_i) and ‖∇F_λ(x_i)‖² of the
    # EXISTING bundle points never change during this inner loop, so
    # they are built ONCE per call — in chunks, so no (m, K, d)
    # temporary is ever materialised — and new rows are appended
    # incrementally as segments add points.  The previous code
    # re-stacked the whole bundle every segment (three O(m·K·d)
    # np.asarray copies plus a full einsum), which came to dominate
    # long runs (~24 s/round at m ≈ 4000).  Only u_vals depends on the
    # current L_scale; it is recomputed per segment as an O(m) vector
    # operation.  Same math, different bookkeeping.
    # ------------------------------------------------------------------
    m0 = bundle.m
    cap = m0 + max_segments * (max_segment_retries + 1) + 2
    Fl = np.empty(cap)                  # F_λ(x_i)
    Gl = np.empty((cap, d))             # ∇F_λ(x_i)
    Gn = np.empty(cap)                  # ‖∇F_λ(x_i)‖²
    _CH = 128
    for s0 in range(0, m0, _CH):
        Jc = np.asarray(bundle.grads[s0:s0 + _CH])          # (≤CH, K, d)
        Fc = np.asarray(bundle.fvals[s0:s0 + _CH])          # (≤CH, K)
        Gc = np.einsum('ikd,k->id', Jc, lam)
        n_rows = Gc.shape[0]
        Gl[s0:s0 + n_rows] = Gc
        Gn[s0:s0 + n_rows] = np.einsum('id,id->i', Gc, Gc)
        Fl[s0:s0 + n_rows] = Fc @ lam
    mcur = m0

    def _append_cache_row() -> None:
        nonlocal mcur
        g_new = bundle.grads[-1].T @ lam
        Gl[mcur] = g_new
        Gn[mcur] = float(g_new @ g_new)
        Fl[mcur] = float(np.asarray(bundle.fvals[-1]) @ lam)
        mcur += 1

    for _seg in range(max_segments):
        # ---- anchor by the T-map selection rule, from the caches ----
        Ll = float(lam @ L_arr) * L_scale
        u_vals = Fl[:mcur] - 0.5 * Gn[:mcur] / Ll
        ai = int(np.argmin(u_vals))
        anchor = np.asarray(bundle.points[ai], dtype=float).copy()
        g_a = Gl[ai].copy()
        F_a = float(Fl[ai])

        # ---- run the segment (with safeguard retries) ----
        seg_retry = 0
        while True:
            Ll = float(lam @ L_arr) * L_scale              # retry ⇒ larger
            eta = step_const / Ll
            stoch_oracle.set_anchor(anchor)
            y = anchor.copy()
            u_vec = np.zeros(d)
            consec = 0
            steps_this = 0
            for _t in range(epoch_len):
                batch = stoch_oracle.sample_batch()
                g_y_S, g_a_S = stoch_oracle.grad_pair(y, lam, batch)
                v = g_y_S - g_a_S + g_a
                u_vec = momentum * u_vec + v
                y = y - eta * u_vec
                steps_this += 1
                if eps_inner is not None:
                    if float(v @ v) <= trigger_rho * eps_inner:
                        consec += 1
                        if consec >= trigger_patience:
                            break
                    else:
                        consec = 0
            minibatch_steps += steps_this

            # ---- segment end: full evaluation, ALWAYS into the bundle ----
            bundle.add_point(y, objectives, grad_objectives,
                             joint_oracle=joint_oracle)
            full_evals += 1
            _append_cache_row()
            F_y = float(Fl[mcur - 1])

            if F_y > F_a + 1e-10 * (1.0 + abs(F_a)):
                # Descent safeguard: the scaled L is still too small along
                # this trajectory.  Double it (η halves), reset momentum,
                # re-run from the SAME anchor.  The violating point stays
                # in the bundle (paid for; harmless under the min).
                L_scale *= 2.0
                retries += 1
                seg_retry += 1
                if L_scale > 2.0 ** 60:
                    raise RuntimeError(
                        "Descent safeguard scaled L beyond 2^60; the "
                        "objectives do not appear L-smooth along the "
                        "iterates."
                    )
                if seg_retry <= max_segment_retries:
                    continue
            break

        segments += 1

        # ---- exact ε/3 acceptance on full gradients (Gram path) ----
        if eps_inner is not None:
            gn_val = _gn_value_batched_gram(bundle.gram_stack(), lam)
            if gn_val <= eps_inner:
                target_met = True
                break

    return {
        "segments": segments,
        "full_evals": full_evals,
        "minibatch_steps": minibatch_steps,
        "L_scale": L_scale,
        "target_met": target_met,
        "safeguard_retries": retries,
    }


# =====================================================================
#  Outer loop: accelerated adaptive algorithm, self-reported checkpoints
# =====================================================================
def algorithm_adaptive_fast(
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
    # --- λ-search tiers ---
    lambda_max_starts: int = 64,
    lambda_tier_mode: str = "strict",       # "strict" | "two_tier"
    cheap_tol: float = 1e-4,
    cheap_max_iter: int = 30,
    strict_tol: float = 1e-8,
    strict_max_iter: int = 100,
    sticky_strict: bool = True,
    require_ipopt: bool = True,
    # --- Momentum-SVRG inner loop ---
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
    joint_oracle: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """Accelerated Algorithm 2 on the without-256-checkpoints track.

    See the module docstring for the acceleration inventory.  Histories
    are on the gradient-equivalent axis: one joint-oracle call = K, one
    minibatch step = 2b·K/n (float).  ``cov_history`` keeps the track's
    lag semantics (most recent λ-search value).  After stopping, the
    bundle is pruned by λ-activation (``prune_report``) — the run-time
    policy is inclusive, pruning exists only at delivery.
    """
    L_arr, x0_arr = validate_problem_inputs(
        K, d, L, x0, objectives, grad_objectives
    )
    if epsilon is not None and (
        not np.isscalar(epsilon) or not np.isfinite(epsilon) or epsilon <= 0.0
    ):
        raise ValueError(f"epsilon must be finite and positive; got {epsilon!r}.")
    if lambda_tier_mode not in {"strict", "two_tier"}:
        raise ValueError("lambda_tier_mode must be 'strict' or 'two_tier'.")
    if msvrg_rel_target is not None and not (0.0 < msvrg_rel_target < 1.0):
        raise ValueError(
            f"msvrg_rel_target must lie in (0, 1) or be None; "
            f"got {msvrg_rel_target!r}."
        )
    if require_ipopt and not _HAS_IPOPT:
        raise RuntimeError(
            "IPOPT was required but cyipopt/IPOPT is unavailable. "
            f"Import error: {_IPOPT_IMPORT_ERROR}"
        )

    joint_oracle = prefer_fused_joint_oracle(joint_oracle)
    bundle = BundleFast(K=K, d=d, L=L_arr)
    bundle.add_point(x0_arr, objectives, grad_objectives,
                     joint_oracle=joint_oracle)   # checkpoint-0 setup, uncharged

    n = stoch_oracle.n
    ifo_start = stoch_oracle.ifo_count

    cpu_times: List[float] = []
    cov_history: List[float] = []
    m_history: List[int] = []       # bundle size at each checkpoint (Jul 26)
    pc_history: List[float] = []
    tier_history: List[str] = []
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
    in_strict_mode = (lambda_tier_mode == "strict")
    prev_lam: Optional[np.ndarray] = None
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
                f"  Fast {label} | t={cpu_times[-1]:.2f}s | bundle={bundle.m} "
                f"| grad_equiv={grad_equiv:.1f} | self-reported pc={cov:.4e}",
                flush=True,
            )

    _checkpoint(f"outer 0/{max_outer}")
    cpu_times[0] = 0.0
    t_start = time.time()
    checkpoint_overhead = 0.0

    eps_inner = None if epsilon is None else epsilon / 3.0
    stop_line = None if epsilon is None else 2.0 * epsilon / 3.0

    for outer in range(1, max_outer + 1):
        if max_grad_evals is not None and grad_equiv + K > max_grad_evals:
            stop_reason = "budget"
            break

        tier = "strict" if in_strict_mode else "cheap"
        ls_t0 = time.time()
        pc_val, lam = _maximise_GN_fast(
            bundle, prev_lam=prev_lam, tier=tier,
            max_starts=lambda_max_starts,
            strict_tol=strict_tol, strict_max_iter=strict_max_iter,
            cheap_tol=cheap_tol, cheap_max_iter=cheap_max_iter,
        )
        if (tier == "cheap" and stop_line is not None
                and pc_val <= stop_line):
            # Stop-verify: only the strict tier signs certificates.
            pc_val, lam = _maximise_GN_fast(
                bundle, prev_lam=lam, tier="strict",
                max_starts=lambda_max_starts,
                strict_tol=strict_tol, strict_max_iter=strict_max_iter,
            )
            tier = "cheap+verify"
            if sticky_strict:
                in_strict_mode = True
        lambda_search_seconds += time.time() - ls_t0
        prev_lam = lam
        pc_history.append(pc_val)
        tier_history.append(tier)
        lambda_history.append(lam.copy())

        if stop_line is not None and pc_val <= stop_line:
            stop_reason = "epsilon_certified"
            _checkpoint(f"outer {outer}/{max_outer}")
            break

        # Relative inner target (Algorithm-2 VARIANT when enabled): the
        # round's task is "cut the selected worst direction to a γ
        # fraction", floored by the paper's absolute eps/3 so the endgame
        # is exactly the original algorithm.  The stopping certificate is
        # unaffected (the strict 2eps/3 stop line above never moves); the
        # early-trigger threshold inside the inner loop scales with this
        # target automatically (it is rho * eps_inner there).
        if eps_inner is None:
            inner_target = None
        elif msvrg_rel_target is not None:
            inner_target = max(eps_inner, msvrg_rel_target * pc_val)
        else:
            inner_target = eps_inner
        inner_target_history.append(inner_target)

        ifo_before = stoch_oracle.ifo_count
        inner = _bundle_update_msvrg(
            bundle, lam, objectives, grad_objectives, joint_oracle,
            stoch_oracle,
            eps_inner=inner_target, L_scale=L_scale,
            step_const=msvrg_step_const, momentum=msvrg_momentum,
            epoch_len=msvrg_epoch_len, max_segments=msvrg_max_segments,
            trigger_rho=msvrg_trigger_rho,
            trigger_patience=msvrg_trigger_patience,
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
        ifo_inner = stoch_oracle.ifo_count - ifo_before
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
    # Jul 26: optional pre-prune snapshot so post-hoc scorers can rebuild
    # the bundle state at every checkpoint (prefix of length m_history[j]).
    # Off by default: on large problems the points copy is memory-heavy.
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
        "tier_history": tier_history,
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
        "lambda_tier_mode": lambda_tier_mode,
        "max_outer": max_outer,
    }
