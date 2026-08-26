"""Pure fixed-budget protocol at K = 2 with an EXACT 1-D quality meter.

NEW FILE (July 30, 2026).  Change record: ``Note/Jul_30_note.md``.
Engine files untouched.  User-approved design (conversation of Jul 30):

Same protocol as ``run_pure_budget_K6_without_256_checkpoints.py``
(July 27, session 12) — shared segment unit, shared s, chain warm
start, stop = budget, ONLY the next-lambda policy differs, no tolerance
parameter anywhere — applied to the K = 2 instance of the SAME problem
family (planted linear-softmax data, [96, 96] tanh MLP, per-class
cross-entropy objectives; the July-8 Pareto-front family).

What is NEW relative to the K = 6 runner:

1. **Exact 1-D meter (no multistart search in any MEASUREMENT).**
   At K = 2,
   lambda = (w, 1-w) and each delivered point contributes
   q_i(w) = lam(w)^T M_i lam(w) — an explicit CONVEX quadratic in w
   (three stored Gram entries).  GN(w) = min_i q_i(w) is evaluated in
   closed form on a dense w-grid (default 200,001 points, vectorised,
   chunked) and the winning neighbourhood is polished exactly
   (active-pair crossings).  No 64-start search, hence no
   search-limited under-reporting (the eps1e-4 false-certificate
   lesson, Note/Jul_27_note.md section 8).  Consequence: exact prefix
   audits are monotone non-increasing BY MATHEMATICS (adding points can
   only lower the min), so the K6 monotone-envelope repair is not
   needed; an assert enforces it instead.
2. **Both legs get full audited trajectories** (exact meter at every
   checkpoint, off both cost axes) — no --backfill-audits pass needed.
3. **Per-segment recording for Pareto fronts**: every segment endpoint
   already receives a full joint evaluation; its objective vector
   (F1, F2) was ALREADY stored by the K6 runner (``fvals`` in
   grams.npz).  This runner additionally stores the cumulative
   grad-equivalent spend and the lambda of every segment
   (``seg_grads``, ``seg_lams``), so the delivered set — and therefore
   the discovered front — can be reconstructed at ANY prefix budget.
4. **Front artifacts** (--figure): per-method nondominated discovered
   fronts in the (F1, F2) plane + query-free metrics (IGD and max
   distance to the UNION front — no oracle front exists for the MLP
   family; the union of both methods' fronts is the mutual reference),
   alongside the GN-vs-budget figures.

Adaptive targeting (on-axis decision work): the strict multistart
IPOPT worst-lambda search, SAME mechanism as the K = 6 runner (user
decision, Jul 30; default 64 starts, --targeting-starts) — the method
under test keeps its continuous aim-anywhere selection.  The exact 1-D
meter serves the AUDITS only (--audit-grid).  A grid-targeting mode
(--decision-mode grid, --decision-grid N) is kept solely for a
possible ablation leg; OFF by default.  Targeting misses hurt only the
method's own efficiency — quality is always scored by the exact audit.

Usage (one leg per invocation; legs skip themselves if their summary
already exists, --force re-runs):
    python run_pure_budget_K2_without_256_checkpoints.py
        --run baseline --r 10 --s 1 [--budget 20000]
    python ... --run adaptive --s 1
    python ... --figure          # collect all legs, draw + README
    python ... --smoke           # tiny end-to-end check
"""
import argparse
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path  # noqa: E402

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from experiments import make_mlp_initial_point  # noqa: E402  (torch first)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bundle import (  # noqa: E402
    prefer_fused_joint_oracle,
    validate_oracle_output,
)
from objectives_torch_fast import make_mlp_nonconvex_fast  # noqa: E402
from algorithm_fast_without_256_checkpoints import (  # noqa: E402
    _maximise_GN_fast,
    ipopt_available,
)
from baseline_svrg_certified_without_256_checkpoints import (  # noqa: E402
    _GramSet,
    _support_batch,
)
from baseline_without_256_checkpoints import (  # noqa: E402
    _sort_grid_for_warmstart,
    _uniform_simplex_grid,
)
from run_pure_budget_K6_without_256_checkpoints import (  # noqa: E402
    MAX_SAFEGUARD_RETRIES,
    _adaptive_policy,
    _baseline_policy,
    _Budget,
)
from run_experiments import DATA_SEED, INIT_SEED, _json_ready  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent.parent / "output"
K2_HOME = OUTPUT_ROOT / "pure_budget_without_256_checkpoints_SVRG_IPOPT_Baseline/pure_budget_K2_without_256_checkpoints"

TRIAL_INSTANCE = dict(
    K=2, p=20, n=50_000, hidden_sizes=[96, 96], activation="tanh",
    msvrg_batch=4096, sampler_seed=41,
    msvrg_step_const=0.1, msvrg_momentum=0.5, msvrg_epoch_len=None,
)
SMOKE_INSTANCE = dict(
    K=2, p=6, n=300, hidden_sizes=[8], activation="tanh",
    msvrg_batch=60, sampler_seed=41,
    msvrg_step_const=0.1, msvrg_momentum=0.5, msvrg_epoch_len=5,
)

_ADAPT_KW = dict(ms=4, lw=1.8)
_AD_COLORS = {64: "#2ca02c", 24: "#17becf"}
_AD_MARKERS = {64: "^", 24: "v"}


def _ad_color(sm):
    return _AD_COLORS.get(sm.get("targeting_starts"), "#2ca02c")


def _ad_marker(sm):
    return _AD_MARKERS.get(sm.get("targeting_starts"), "^")


def _pseudo_zero(xs_list):
    """Common artificial log-axis position for the t=0 checkpoint: one
    third of the smallest positive x across ALL plotted curves of the
    figure (user decision Jul 31 — every leg's zero pinned to the SAME
    spot; the true start is identical for all legs: x0, value ~2.33,
    time 0)."""
    pos = [float(x[x > 0].min()) for x in xs_list if (x > 0).any()]
    return (min(pos) / 3.0) if pos else 1e-3


def _leg_dir_k2(home, policy, r, s, args=None):
    """K2 leg naming: adaptive legs carry their targeting spec, so
    several adaptive configurations (24- vs 64-start, grid ablation)
    can coexist under one budget home."""
    if policy != "adaptive":
        return home / f"baseline_r{r:02d}_s{s}"
    if getattr(args, "decision_mode", "search") == "search":
        return home / f"adaptive_s{s}_ts{args.targeting_starts}"
    return home / f"adaptive_s{s}_grid{args.decision_grid}"


# ---------------------------------------------------------------------------
# Exact 1-D meter: GN*(stack) = max_w min_i q_i(w), q_i convex quadratic.
# ---------------------------------------------------------------------------

def _quad_coeffs(Ms):
    """Per-point coefficients of q_i(w) = A_i w^2 + B_i w + C_i.

    lam(w) = (w, 1-w);  q(w) = w^2 M00 + 2 w (1-w) M01 + (1-w)^2 M11.
    A = M00 - 2 M01 + M11 >= 0 for PSD M, so every q_i is convex.
    """
    Ms = np.asarray(Ms, dtype=float)
    a, b, c = Ms[:, 0, 0], Ms[:, 0, 1], Ms[:, 1, 1]
    return a - 2.0 * b + c, 2.0 * (b - c), c


def _env_at(A, B, C, w):
    """Exact envelope values min_i q_i(w) at the given w array."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    Q = A[:, None] * w[None, :] ** 2 + B[:, None] * w[None, :] + C[:, None]
    return Q.min(axis=0)


def exact_gn_1d(Ms, grid_points=200_001, chunk=20_000, polish=True,
                certify=False):
    """Exact max over w in [0, 1] of min over points of lam(w)^T M lam(w).

    Dense-grid scan (every value is the true function value — plain
    arithmetic on stored Grams, no iterative search that could miss a
    pocket), then closed-form polish of the winning neighbourhood: the
    envelope of convex quadratics attains its max at an endpoint or at a
    crossing of two active quadratics, so the crossings of the locally
    lowest quadratics inside the winning grid cell are solved exactly
    and the FULL envelope is re-evaluated there.

    Returns (gn_star, w_star), or (gn_star, w_star, upper_bound) with
    ``certify=True``.  gn_star is a TRUE function value, hence a lower
    bound of the max; upper_bound is a PROVEN upper bound: inside any
    grid cell [w_j, w_j + h], env(w) <= q_i(w) for the quadratic i
    active at either cell end, and a quadratic with known coefficients
    moves by at most max|q_i'| * h across the cell, so
    env <= min(env(w_j) + S_left * h, env(w_j + h) + S_right * h)
    cell-wise.  The true max is therefore certified to lie in
    [gn_star, upper_bound]; the interval width is reported, not
    assumed.  (No such bound exists for multistart search.)
    """
    A, B, C = _quad_coeffs(Ms)
    grid = np.linspace(0.0, 1.0, int(grid_points))
    G = grid.size
    env = np.empty(G)
    act = np.empty(G, dtype=np.int64)
    for lo in range(0, G, int(chunk)):
        w = grid[lo:lo + int(chunk)]
        Q = (A[:, None] * w[None, :] ** 2 + B[:, None] * w[None, :]
             + C[:, None])
        env[lo:lo + w.size] = Q.min(axis=0)
        act[lo:lo + w.size] = Q.argmin(axis=0)
    jbest = int(env.argmax())
    best_v, best_w = float(env[jbest]), float(grid[jbest])

    def _finish(v, w):
        if not certify:
            return v, w
        h = 1.0 / (G - 1)
        Ai, Bi = A[act], B[act]
        s_here = 2.0 * Ai * grid + Bi
        s_l = np.maximum(np.abs(s_here[:-1]),
                         np.abs(2.0 * Ai[:-1] * grid[1:] + Bi[:-1]))
        s_r = np.maximum(np.abs(s_here[1:]),
                         np.abs(2.0 * Ai[1:] * grid[:-1] + Bi[1:]))
        u_cell = np.minimum(env[:-1] + s_l * h, env[1:] + s_r * h)
        ub = float(max(float(u_cell.max()), float(env[0]), float(env[-1]),
                       v))
        return v, w, ub

    if not polish:
        return _finish(best_v, best_w)

    h = 1.0 / (int(grid_points) - 1)
    wl, wr = max(0.0, best_w - h), min(1.0, best_w + h)
    # Locally lowest quadratics at the three sample w's (top-64 each).
    cand = set()
    for w0 in (wl, best_w, wr):
        q = A * w0 * w0 + B * w0 + C
        cand.update(np.argsort(q)[:64].tolist())
    cand = np.asarray(sorted(cand), dtype=int)
    # Pairwise crossings of the candidate quadratics inside [wl, wr].
    ws = [wl, wr, best_w]
    Ac, Bc, Cc = A[cand], B[cand], C[cand]
    for i in range(cand.size):
        dA = Ac[i] - Ac[i + 1:]
        dB = Bc[i] - Bc[i + 1:]
        dC = Cc[i] - Cc[i + 1:]
        # Quadratic crossings.
        with np.errstate(all="ignore"):
            disc = dB * dB - 4.0 * dA * dC
            ok = (np.abs(dA) > 1e-300) & (disc >= 0.0)
            sq = np.sqrt(np.where(ok, disc, 0.0))
            for sgn in (+1.0, -1.0):
                r = np.where(ok, (-dB + sgn * sq) / (2.0 * dA), np.nan)
                r = r[(r >= wl) & (r <= wr)]
                ws.extend(float(t) for t in r)
            # Linear crossings (dA ~ 0).
            lin = (~ok) & (np.abs(dB) > 1e-300)
            r = np.where(lin, -dC / np.where(lin, dB, 1.0), np.nan)
            r = r[(r >= wl) & (r <= wr)]
            ws.extend(float(t) for t in r)
    ws = np.asarray(ws, dtype=float)
    env_ws = _env_at(A, B, C, ws)
    j = int(np.argmax(env_ws))
    if float(env_ws[j]) > best_v:
        best_v, best_w = float(env_ws[j]), float(ws[j])
    return _finish(best_v, best_w)


def _adaptive_policy_exact(decision_grid):
    """ABLATION-ONLY targeting: exact worst w on a coarse decision grid.

    Not used by default (user decision, Jul 30: the method's own
    targeting is the strict multistart IPOPT search, as at K = 6).
    Kept behind --decision-mode grid for a possible menu-density
    ablation leg.  Decision work runs inside the leg's wall clock,
    i.e. ON the adaptive CPU axis.  No polish here — targeting only
    steers; the audits use the fine grid + polish.
    """
    def nxt(grams, fvals, prev_lam):
        Ms = np.asarray(grams, dtype=float)
        _v, w = exact_gn_1d(Ms, grid_points=decision_grid, polish=False)
        return np.array([w, 1.0 - w])
    return nxt


# ---------------------------------------------------------------------------
# Nondominated fronts in the (F1, F2) plane (both minimised).
# ---------------------------------------------------------------------------

def _nondominated(F):
    """Indices of the nondominated subset of an (m, 2) value array."""
    F = np.asarray(F, dtype=float)
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep, best_f2 = [], np.inf
    for i in order:
        if F[i, 1] < best_f2 - 1e-15:
            keep.append(int(i))
            best_f2 = F[i, 1]
    return np.asarray(keep, dtype=int)


CENTRAL_BOUND = 1.0  # both losses <= this = the genuine trade-off region


def _hv_2d(front, ref):
    """2-D hypervolume of a nondominated front w.r.t. reference point
    ``ref`` (staircase area; points beyond ref contribute nothing)."""
    P = front[(front[:, 0] <= ref[0]) & (front[:, 1] <= ref[1])]
    if P.size == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in P:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


def _spacing_metrics(front):
    """SURF-style uniformity metrics on consecutive front segments:
    CV = std/mean of segment lengths, Gap Ratio = max/min segment
    length (SURF paper, Table 1 / Eq. 16).  Needs >= 3 points."""
    if front.shape[0] < 3:
        return None, None
    P = front[np.argsort(front[:, 0])]
    seg = np.sqrt((np.diff(P, axis=0) ** 2).sum(axis=1))
    seg = seg[seg > 0]
    if seg.size < 2:
        return None, None
    return (float(seg.std() / seg.mean()),
            float(seg.max() / seg.min()))


def _front_metrics(F_method, F_union):
    """IGD and max distance from the union front to the method front."""
    if F_method.size == 0:
        return dict(igd_to_union=float("inf"), maxdist_to_union=float("inf"))
    d = np.sqrt(((F_union[:, None, :] - F_method[None, :, :]) ** 2)
                .sum(axis=2)).min(axis=1)
    return dict(igd_to_union=float(d.mean()),
                maxdist_to_union=float(d.max()))


# ---------------------------------------------------------------------------
# Shared executor (K6 protocol verbatim; K2 additions marked "K2:").
# ---------------------------------------------------------------------------

def _run_leg(policy_name, next_lam, cfg, args, out_dir, extra_cfg):
    t_build = time.time()
    objectives, gradients, L, joint, stoch = make_mlp_nonconvex_fast(
        K=cfg["K"], p=cfg["p"], n=cfg["n"],
        hidden_sizes=cfg["hidden_sizes"], seed=DATA_SEED,
        activation=cfg["activation"], w_true_scale=1.0,
        batch_size=cfg["msvrg_batch"], sampler_seed=cfg["sampler_seed"],
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(K=cfg["K"], p=cfg["p"],
                                hidden_sizes=cfg["hidden_sizes"],
                                seed=INIT_SEED)
    K, d, n = cfg["K"], int(x0.size), cfg["n"]
    L_arr = np.asarray(L, dtype=float)
    epoch_len = (cfg["msvrg_epoch_len"] if cfg["msvrg_epoch_len"]
                 else max(1, int(np.ceil(n / float(cfg["msvrg_batch"])))))
    print(f"[{policy_name}] instance in {time.time() - t_build:.1f}s "
          f"(epoch_len={epoch_len})", flush=True)

    f0, J0 = validate_oracle_output(*joint_oracle(x0), K, d)
    grams = [J0 @ J0.T]
    fvals = [np.asarray(f0, dtype=float)]
    chain_x, chain_J, chain_f = x0.copy(), J0, f0

    budget = _Budget(K, n, stoch, args.budget)
    L_scale = 1.0
    safeguard_retries = 0
    lam_history = []
    seg_grads, seg_lams = [0.0], [[np.nan] * K]   # K2: per-segment records
    ck_grads, ck_cpu, ck_m = [0.0], [0.0], [1]
    grad_at_ck = 0.0
    t0 = time.time()
    decision_seconds = 0.0

    prev_lam = None
    while budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
        t_dec = time.time()
        lam = np.asarray(next_lam(grams, fvals, prev_lam), dtype=float)
        decision_seconds += time.time() - t_dec
        prev_lam = lam
        lam_history.append(lam.copy())

        retries_here = 0
        for _k in range(args.s):
            if not budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
                break
            g_a_full = chain_J.T @ lam
            F_a = float(chain_f @ lam)
            eta = cfg["msvrg_step_const"] / (float(lam @ L_arr) * L_scale)
            stoch.set_anchor(chain_x)
            y = chain_x.copy()
            u_vec = np.zeros(d)
            for _t in range(epoch_len):
                batch = _support_batch(stoch.sample_batch(), lam)
                g_y_S, g_a_S = stoch.grad_pair(y, lam, batch)
                u_vec = cfg["msvrg_momentum"] * u_vec + (g_y_S - g_a_S
                                                         + g_a_full)
                y = y - eta * u_vec
            f_y, J_y = validate_oracle_output(*joint_oracle(y), K, d)
            budget.joint_calls += 1
            grams.append(J_y @ J_y.T)
            fvals.append(np.asarray(f_y, dtype=float))
            seg_grads.append(float(budget.spent()))       # K2
            seg_lams.append([float(t) for t in lam])      # K2
            if float(f_y @ lam) > F_a + 1e-10 * (1.0 + abs(F_a)):
                L_scale *= 2.0
                safeguard_retries += 1
                retries_here += 1
                if retries_here > MAX_SAFEGUARD_RETRIES:
                    chain_x, chain_J, chain_f = y, J_y, f_y
                    retries_here = 0
            else:
                chain_x, chain_J, chain_f = y, J_y, f_y

            if budget.spent() - grad_at_ck >= args.eval_every:
                grad_at_ck = budget.spent()
                ck_grads.append(budget.spent())
                ck_cpu.append(time.time() - t0)
                ck_m.append(len(grams))

    wall = time.time() - t0
    ck_grads.append(budget.spent())
    ck_cpu.append(wall)
    ck_m.append(len(grams))
    Ms = np.asarray(grams, dtype=float)
    print(f"[{policy_name}] budget spent: {budget.spent():.1f} of "
          f"{args.budget} | segments={len(grams) - 1} | wall={wall:.1f}s "
          f"| decision={decision_seconds:.1f}s "
          f"| L_scale={L_scale}", flush=True)

    # ---- post-hoc measurement (off both axes): EXACT prefix audits ----
    # K2: every checkpoint, both policies; exact values are monotone
    # non-increasing in the prefix by mathematics — asserted, not
    # envelope-repaired.
    t_a = time.time()
    audited, audit_ws, audit_ub = [], [], []
    for m_ck in ck_m:
        v, w, ub = exact_gn_1d(Ms[:m_ck], grid_points=args.audit_grid,
                               certify=True)
        assert ub >= v - 1e-15, f"certified bound below value: {ub} < {v}"
        audited.append(float(v))
        audit_ws.append(float(w))
        audit_ub.append(float(ub))
    mono_tol = 1e-12
    for i in range(1, len(audited)):
        if ck_m[i] >= ck_m[i - 1]:
            assert audited[i] <= audited[i - 1] + mono_tol, (
                f"exact prefix audit not monotone at ck {i}: "
                f"{audited[i - 1]} -> {audited[i]}")
    final_audit, w_star = audited[-1], audit_ws[-1]
    audit_seconds = time.time() - t_a

    lam_arr = np.asarray(lam_history, dtype=float)
    distinct = (np.unique(lam_arr.round(12), axis=0).shape[0]
                if lam_arr.size else 0)
    summary = {
        "protocol": ("pure fixed budget at K=2: shared segment unit, "
                     "shared s, chain warm start; ONLY the next-lambda "
                     "policy differs; no tolerance parameter anywhere; "
                     "stop = budget; EXACT 1-D meter (no multistart "
                     "search) for targeting and audits"),
        "policy": policy_name,
        "config_instance": _json_ready(cfg),
        "budget": args.budget,
        "s": args.s,
        "eval_every": args.eval_every,
        "meter": "exact-1d (audits)",
        "decision_mode": (getattr(args, "decision_mode", "search")
                          if policy_name == "adaptive" else None),
        "targeting_starts": (
            args.targeting_starts if policy_name == "adaptive"
            and getattr(args, "decision_mode", "search") == "search"
            else None),
        "decision_grid": (
            args.decision_grid if policy_name == "adaptive"
            and getattr(args, "decision_mode", "search") == "grid"
            else None),
        "audit_grid": args.audit_grid,
        "extra": _json_ready(extra_cfg),
        "grad_equiv_total": float(budget.spent()),
        "joint_calls": int(budget.joint_calls),
        "wall_seconds": wall,
        "decision_seconds": decision_seconds,
        "segments_total": int(len(grams) - 1),
        "m_final": int(Ms.shape[0]),
        "n_decisions": int(len(lam_history)),
        "n_distinct_lambdas": int(distinct),
        "L_scale_final": L_scale,
        "safeguard_retries": int(safeguard_retries),
        "ck_grads": ck_grads, "ck_cpu": ck_cpu, "ck_m": ck_m,
        "final_audit": float(final_audit),
        "final_audit_upper": float(audit_ub[-1]),
        "audited_gn_history": audited,
        "audited_gn_upper_history": audit_ub,
        "audit_certified_gap_max": float(max(u - v for u, v
                                             in zip(audit_ub, audited))),
        "audit_w_history": audit_ws,
        "w_star": float(w_star),
        "audit_seconds": audit_seconds,
        "data_seed": DATA_SEED, "init_seed": INIT_SEED,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "grams.npz", gram_stack=Ms,
                        fvals=np.asarray(fvals),
                        lam_history=lam_arr,
                        seg_grads=np.asarray(seg_grads, dtype=float),
                        seg_lams=np.asarray(seg_lams, dtype=float))
    print(f"[{policy_name}] final EXACT audit = {final_audit:.6e} "
          f"(w* = {w_star:.5f}; certified gap "
          f"{audit_ub[-1] - final_audit:.2e}) -> {out_dir}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Figures: GN vs budget/CPU (audited trajectories, all legs) + fronts.
# ---------------------------------------------------------------------------

def _load_legs(home):
    legs = []
    for p in sorted(home.glob("*/summary.json")):
        with open(p, "r", encoding="utf-8") as fh:
            sm = json.load(fh)
        sm["_dir"] = p.parent
        legs.append(sm)
    return legs


def _figure(home, args, cfg):
    legs = _load_legs(home)
    ad = [s for s in legs if s["policy"] == "adaptive"]
    bl = [s for s in legs if s["policy"] == "baseline"]
    if not ad:
        raise SystemExit("--figure: no adaptive leg found")
    # Jul 31 (user): the 24- and 64-start adaptive legs are bit-identical
    # twins (section 7a of Note/Jul_30_note.md) — DISPLAY only the
    # 24-start line; a 64-start leg is hidden whenever a same-s 24-start
    # leg exists.  Data, summaries and the README leg table keep both.
    ad_show = [sm for sm in ad
               if not (sm.get("targeting_starts") == 64
                       and any(x.get("targeting_starts") == 24
                               and x["s"] == sm["s"] for x in ad))]

    reds = plt.get_cmap("Reds")
    rs = sorted({s["extra"]["r"] for s in bl})

    def _bl_color(r):
        return reds(0.45 + 0.5 * rs.index(r) / max(1, len(rs) - 1))

    # ---- GN vs grads / CPU: K=6 pure-budget figure style (user, Jul 30
    # revision): adaptive = audited trajectory curve; baseline legs =
    # FINAL POINTS only, labels staggered into a connector-linked
    # column.  Baseline trajectories stay in the summaries, unplotted.
    for axis_key, x_label, fname in (
        ("ck_grads", "total gradient evaluations, grad-equivalents",
         "pure_budget_K2_gn_vs_grads.png"),
        ("ck_cpu", "CPU time, seconds", "pure_budget_K2_gn_vs_cpu.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        x0_pseudo = _pseudo_zero([np.asarray(sm[axis_key], dtype=float)
                                  for sm in ad_show])
        for sm in ad_show:
            hx = np.asarray(sm[axis_key], dtype=float)
            hy = np.maximum(np.asarray(sm["audited_gn_history"], dtype=float),
                            np.finfo(float).tiny)
            hx = np.where(hx > 0, hx, x0_pseudo)
            kw = dict(_ADAPT_KW)
            kw["color"] = _ad_color(sm)
            kw["marker"] = _ad_marker(sm)
            if sm["s"] != args.s:
                kw["ls"] = "--"
            if sm.get("decision_mode", "search") == "search":
                lbl = (f"adaptive policy (s={sm['s']}, "
                       f"{sm.get('targeting_starts')}-start) — exact "
                       f"prefix audit")
            else:
                lbl = (f"adaptive ABLATION (s={sm['s']}, "
                       f"grid {sm.get('decision_grid')}) — exact prefix "
                       f"audit")
            ax.plot(hx, hy, label=lbl, **kw)

        pts = []
        for sm in bl:
            r = sm["extra"]["r"]
            color = _bl_color(r)
            main = sm["s"] == args.s
            x_end = float(sm["grad_equiv_total"] if axis_key == "ck_grads"
                          else sm["wall_seconds"])
            y_end = sm["final_audit"]
            if main:
                ax.scatter([x_end], [y_end], marker="x", s=52, zorder=6,
                           color=color, linewidth=1.8)
            else:
                ax.scatter([x_end], [y_end], marker="o", s=40, zorder=6,
                           facecolors="none", edgecolor=color,
                           linewidth=1.4)
            pts.append((y_end, x_end, f"r={r}: {y_end:.3g}"))
        pts.sort(key=lambda t: -t[0])
        prev_y, dy = None, 6
        for y_end, x_end, text in pts:
            close = (prev_y is not None
                     and abs(np.log10(max(y_end, 1e-300))
                             - np.log10(max(prev_y, 1e-300))) < 0.35)
            dy = (dy - 15) if close else 6
            prev_y = y_end
            ax.annotate(text, xy=(x_end, y_end), xytext=(-12, dy),
                        textcoords="offset points", ha="right",
                        fontsize=7.0, color="#8b1a1a",
                        arrowprops=dict(arrowstyle="-", color="#8b1a1a",
                                        lw=0.5, alpha=0.5))
        ax.scatter([], [], marker="x", s=52, color=reds(0.7), linewidth=1.8,
                   label=f"baseline s={args.s} — final exact audit")
        if any(sm["s"] != args.s for sm in bl):
            ax.scatter([], [], marker="o", s=40, facecolors="none",
                       edgecolor=reds(0.7), linewidth=1.4,
                       label="baseline sensitivity — final exact audit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("full-simplex GN* — EXACT 1-D meter")
        ax.set_title(
            f"MLP K=2, p={cfg['p']}, n={cfg['n']}, h={cfg['hidden_sizes']}, "
            f"{cfg['activation']} — PURE fixed budget, no tolerance inputs\n"
            f"B={args.budget:,.0f} grad-equivalents per leg; shared segment "
            f"unit; only the next-lambda policy differs",
            fontsize=9.6,
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.2, loc="lower left")
        fig.tight_layout()
        fig.savefig(home / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Companion figure (user, Jul 31): CPU axis WITH full baseline
    # trajectory lines.  The endpoint-style figure above remains the
    # figure of record; this one makes the baseline plateaus visible.
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x0_pseudo = _pseudo_zero([np.asarray(sm["ck_cpu"], dtype=float)
                              for sm in bl + ad_show])
    for sm in bl:
        r = sm["extra"]["r"]
        hx = np.asarray(sm["ck_cpu"], dtype=float)
        hy = np.maximum(np.asarray(sm["audited_gn_history"], dtype=float),
                        np.finfo(float).tiny)
        hx = np.where(hx > 0, hx, x0_pseudo)
        style = "-" if sm["s"] == args.s else "--"
        ax.plot(hx, hy, style, color=_bl_color(r), lw=1.4, marker="x",
                ms=3.5, label=f"baseline r={r} (s={sm['s']}) — exact audit")
    for sm in ad_show:
        hx = np.asarray(sm["ck_cpu"], dtype=float)
        hy = np.maximum(np.asarray(sm["audited_gn_history"], dtype=float),
                        np.finfo(float).tiny)
        hx = np.where(hx > 0, hx, x0_pseudo)
        kw = dict(_ADAPT_KW)
        kw["color"] = _ad_color(sm)
        kw["marker"] = _ad_marker(sm)
        if sm["s"] != args.s:
            kw["ls"] = "--"
        ax.plot(hx, hy,
                label=f"adaptive policy (s={sm['s']}, "
                      f"{sm.get('targeting_starts')}-start) — exact "
                      f"prefix audit", **kw)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU time, seconds")
    ax.set_ylabel("full-simplex GN* — EXACT 1-D meter")
    ax.set_title(
        f"MLP K=2 — PURE fixed budget, B={args.budget:,.0f} "
        f"grad-equivalents per leg\nfull audited trajectories on the "
        f"CPU axis (companion to the endpoint-style figure)",
        fontsize=9.6)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.2, loc="lower left")
    fig.tight_layout()
    fig.savefig(home / "pure_budget_K2_gn_vs_cpu_trajectories.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Discovered Pareto fronts (final budget) + metrics ----
    fronts = {}
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    main_legs = ([sm for sm in bl if sm["s"] == args.s]
                 + [sm for sm in ad_show if sm["s"] == args.s])
    for sm in main_legs:
        npz = np.load(sm["_dir"] / "grams.npz")
        F = np.asarray(npz["fvals"], dtype=float)
        idx = _nondominated(F)
        if sm["policy"] == "baseline":
            key = f"baseline_r{sm['extra']['r']}"
            color, marker, z = _bl_color(sm["extra"]["r"]), "x", 4
        elif sm.get("decision_mode", "search") == "search":
            key = f"adaptive_ts{sm.get('targeting_starts')}"
            color, marker, z = _ad_color(sm), _ad_marker(sm), 5
        else:
            key = f"adaptive_grid{sm.get('decision_grid')}"
            color, marker, z = "#9467bd", "s", 5
        fronts[key] = dict(F=F, front=F[idx], n_points=int(F.shape[0]),
                           n_front=int(idx.size),
                           certified_eps=float(
                               sm.get("final_audit_upper", float("nan"))),
                           color=color, marker=marker, z=z)
    F_union_all = np.vstack([v["front"] for v in fronts.values()])
    union_front = F_union_all[_nondominated(F_union_all)]
    order = np.argsort(union_front[:, 0])
    union_front = union_front[order]

    cb = CENTRAL_BOUND
    union_central = union_front[(union_front[:, 0] <= cb)
                                & (union_front[:, 1] <= cb)]
    # SURF Fig-6 style (user, Jul 30 revision): nondominated fronts
    # ONLY — one marker-line per method, no delivered clouds, no union
    # overlay, no auxiliary lines.  Metrics are still computed on the
    # union/central references below; they are data, not display.
    for key, v in fronts.items():
        fr = v["front"][np.argsort(v["front"][:, 0])]
        ax.plot(fr[:, 0], fr[:, 1], marker=v["marker"], ms=5, lw=1.4,
                color=v["color"], alpha=0.85, zorder=v["z"],
                label=f"{key} ({v['n_front']} pts)")
        v["metrics"] = _front_metrics(v["front"], union_front)
        if union_central.size:
            cen = _front_metrics(v["front"], union_central)
            v["metrics"]["igd_central"] = cen["igd_to_union"]
            v["metrics"]["maxdist_central"] = cen["maxdist_to_union"]
        else:
            v["metrics"]["igd_central"] = None
            v["metrics"]["maxdist_central"] = None
        # SURF-paper metrics (Table 1 there: HV / IGD / CV / Gap Ratio),
        # computed on the CENTRAL part of the method's front with the
        # central corner (cb, cb) as the HV reference point (convention
        # disclosed in the README).
        fc = v["front"][(v["front"][:, 0] <= cb) & (v["front"][:, 1] <= cb)]
        v["metrics"]["hv_central"] = _hv_2d(fc, (cb, cb))
        cv, gap = _spacing_metrics(fc)
        v["metrics"]["cv_central"] = cv
        v["metrics"]["gap_ratio_central"] = gap
        v["metrics"]["n_front_central"] = int(fc.shape[0])
    # Log-log axes: the losses span ~1e-5 .. ~30 (cross-entropy is
    # strictly positive) and the trade-off knee lives below 1e-2 on
    # this instance — a linear panel cannot show the fronts.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("F1 (class-1 cross-entropy, full batch; log)")
    ax.set_ylabel("F2 (class-2 cross-entropy, full batch; log)")
    ax.set_title(
        f"Discovered nondominated fronts at equal budget "
        f"B={args.budget:,.0f} (s={args.s})", fontsize=10.5)
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(fontsize=8.0, loc="upper right")
    fig.tight_layout()
    fig.savefig(home / "pure_budget_K2_fronts.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- union-front composition (user request, Jul 31): which leg
    # contributed each point of the union front, raw and central. ----
    def _owners(U):
        out = []
        for row in U:
            hit = [k for k, v in fronts.items()
                   if (np.abs(v["front"] - row) < 1e-12).all(axis=1).any()]
            out.append(hit[0] if hit else "?")
        return out

    owners_raw = _owners(union_front)
    owners_cen = _owners(union_central)
    comp = {"raw": {k: int(sum(o == k for o in owners_raw))
                    for k in fronts},
            "central": {k: int(sum(o == k for o in owners_cen))
                        for k in fronts},
            "n_raw": int(union_front.shape[0]),
            "n_central": int(union_central.shape[0])}
    print("[union composition] raw:", comp["raw"],
          "| central:", comp["central"], flush=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    Uc = union_central[np.argsort(union_central[:, 0])]
    ax.plot(Uc[:, 0], Uc[:, 1], lw=0.9, color="0.6", zorder=1,
            label=f"central union front ({Uc.shape[0]} pts)")
    for key, v in fronts.items():
        mask = np.asarray([o == key for o in owners_cen], dtype=bool)
        if not mask.any():
            continue
        pts = union_central[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=46, color=v["color"],
                   marker=v["marker"], zorder=3,
                   label=f"{key}: {int(mask.sum())} pts")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("F1 (class-1 cross-entropy, full batch; log)")
    ax.set_ylabel("F2 (class-2 cross-entropy, full batch; log)")
    ax.set_title(
        f"Central union front (both losses <= {cb:g}) at equal budget "
        f"B={args.budget:,.0f} —\nwhich method contributed each point",
        fontsize=10.0)
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(fontsize=8.0, loc="upper right")
    fig.tight_layout()
    fig.savefig(home / "pure_budget_K2_union_front_composition.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    front_metrics = {k: dict(n_points=v["n_points"], n_front=v["n_front"],
                             certified_eps=v["certified_eps"],
                             central_bound=cb,
                             **v["metrics"]) for k, v in fronts.items()}
    front_metrics["union_front_composition"] = comp
    (home / "front_metrics.json").write_text(
        json.dumps(_json_ready(front_metrics), indent=2), encoding="utf-8")

    # ---- README ----
    def _leg_label(sm):
        if sm["policy"] == "baseline":
            return f"baseline r={sm['extra']['r']}"
        if sm.get("decision_mode", "search") == "search":
            return f"adaptive {sm.get('targeting_starts')}-start"
        return f"adaptive grid{sm.get('decision_grid')} (ablation)"

    rows = "\n".join(
        f"| {_leg_label(sm)} | {sm['s']} "
        f"| {sm['n_distinct_lambdas']:,} "
        f"| {sm['grad_equiv_total']:.0f} | {sm['wall_seconds']:.0f} "
        f"| {sm['final_audit']:.4e} |"
        for sm in bl + ad)
    def _fmt(x):
        return "—" if x is None else f"{x:.4e}"

    leg_metrics = {k: v for k, v in front_metrics.items()
                   if k != "union_front_composition"}
    mrows = "\n".join(
        f"| {k} | {v['n_points']:,} | {v['n_front']:,} "
        f"| {v['certified_eps']:.4e} "
        f"| {v['igd_to_union']:.4e} | {v['maxdist_to_union']:.4e} "
        f"| {_fmt(v['igd_central'])} | {_fmt(v['maxdist_central'])} |"
        for k, v in leg_metrics.items())
    srows = "\n".join(
        f"| {k} | {v['n_front_central']} | {_fmt(v['hv_central'])} "
        f"| {_fmt(v['cv_central'])} | {_fmt(v['gap_ratio_central'])} |"
        for k, v in leg_metrics.items())
    epoch_len = (cfg["msvrg_epoch_len"] if cfg["msvrg_epoch_len"]
                 else max(1, int(np.ceil(cfg["n"]
                                         / float(cfg["msvrg_batch"])))))
    starts_list = sorted({sm.get("targeting_starts") for sm in ad
                          if sm.get("decision_mode", "search") == "search"
                          and sm.get("targeting_starts") is not None})
    if starts_list:
        ad_policy_desc = (
            f"strict worst-lambda search (adaptive, "
            f"{'/'.join(str(t) for t in starts_list)}-start IPOPT legs, "
            f"warm-started — same mechanism as the K = 6 runner; decision "
            f"time inside its CPU axis)")
    else:
        ad0 = ad[0]
        ad_policy_desc = (f"exact worst-w targeting (adaptive, ABLATION "
                          f"mode, {ad0.get('decision_grid')}-point decision "
                          f"grid; decision time inside its CPU axis)")
    readme = f"""# Pure fixed-budget comparison at K = 2 (exact 1-D meter; without-256 track)

Produced by `Original_py/run_pure_budget_K2_without_256_checkpoints.py`
(July 30, 2026; change record `Note/Jul_30_note.md`).  Protocol as the
K = 6 July-27 experiment: every leg spends the SAME budget
B = {args.budget:,.0f} grad-equivalents in identical work units (one
segment = {epoch_len} minibatch Momentum-SVRG steps, no trigger, + 1
full joint evaluation), s = {args.s} consecutive segments per
allocation decision, chain warm start for BOTH methods, stop = budget.
The ONLY difference is the next-lambda policy: snake grid order
(baseline, per r; K = 2 grid has r + 1 nodes) vs {ad_policy_desc}.

Quality meter: EXACT at K = 2 — GN(w) = min over delivered points of a
convex quadratic in w; evaluated on a {args.audit_grid:,}-point w-grid
with closed-form polish of the winning neighbourhood, and CERTIFIED by
a per-cell slope bound: every audit stores a true-value lower bound
AND a proven upper bound (summary keys `audited_gn_history` /
`audited_gn_upper_history`; `audit_certified_gap_max` is the widest
interval of the leg), so "exact" is a checkable statement per run, not
a label.  No multistart search anywhere in the MEASUREMENT — the
adaptive's own targeting search is method-internal navigation whose
misses can only hurt its own efficiency, never the reported quality —
hence no search-limited under-reporting (the eps1e-4 lesson).  Exact
prefix audits are monotone by mathematics (asserted).  Audits are off
both cost axes; adaptive targeting is on-axis method work.

| leg | s | distinct lambdas visited | grad-equiv | wall s | exact audit |
|-----|---|--------------------------|------------|--------|-------------|
{rows}

(baseline "distinct lambdas" = grid nodes visited; at K = 2 every leg
covers its full grid many times over.)

## Discovered fronts (final budget, s = {args.s} legs)

Delivered set = every segment endpoint's full-batch (F1, F2); front =
its nondominated subset; reference = union of all plotted fronts (no
oracle front exists for this family).  IGD/max-dist are distances from
the union front to each method front (raw value-space Euclidean).

Metrics come in TWO variants because the raw union front is dominated
by degenerate SPECIALIST TAILS: grid legs whose vertex nodes (w = 0 or
w = 1) camp on a single-class objective for many passes drive that
class's loss toward 0 while the other explodes (F1 up to ~27 here) —
nondominated by construction, but not a genuine trade-off.  A
GN-steered method stops visiting a region once it is near-stationary
and never manufactures such tails.  So next to the raw metrics, the
"central" variant restricts the REFERENCE front to the region where
both losses are <= {cb:g} (method fronts are never clipped).  The
figure (SURF Fig-6 style, user revision Jul 30) shows the per-method
nondominated fronts ONLY, on log-log axes — the knee lives below 1e-2
on this instance, so linear axes cannot show the fronts.

"Certified eps (U)" is each leg's final certified UPPER bound on
GN* = max over w of [min over its delivered points of the
lam(w)-weighted gradient-norm^2]: for EVERY trade-off weight w in
[0, 1], the leg's delivered set PROVABLY contains a point whose
weighted stationarity measure is <= U.  This is the epsilon of
"epsilon-Pareto front" in the stationarity sense — a positive claim
only an upper bound can sign (a search value is a lower bound and
cannot; the eps1e-4 false-certificate lesson).

| leg | delivered pts | front pts | certified eps (U) | IGD raw | max-dist raw | IGD central | max-dist central |
|-----|---------------|-----------|-------------------|---------|--------------|-------------|------------------|
{mrows}

SURF-paper-style metrics (their Table 1: HV / IGD / CV / Gap Ratio),
computed on each method's front restricted to the central region, with
the central corner ({cb:g}, {cb:g}) as the hypervolume reference point
(conventions disclosed here; SURF's originals are computed on bounded
RL fronts, ours collapse toward the origin, hence the central
restriction).  HV higher = better; CV and Gap Ratio lower = more
uniform spacing (SURF's headline axis).  Single realization — no
mean/std yet (SURF reports 8 seeds).

| leg | central front pts | HV (ref ({cb:g},{cb:g})) | CV | Gap Ratio |
|-----|-------------------|--------------------------|----|-----------|
{srows}

`grams.npz` per leg additionally stores `seg_grads` / `seg_lams`
(cumulative grad-equivalent spend and lambda per segment), so the
delivered set and its front can be reconstructed at ANY prefix budget.

CPU-axis caveat: adaptive decision time is method work, on-axis;
audits are off-axis for everyone.  MLP torch runs are not
bit-reproducible in this environment (session-12 finding); one
realization each.
"""
    (home / "README.md").write_text(readme, encoding="utf-8")
    print(f"FIGURE + README -> {home}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, choices=["baseline", "adaptive"])
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--budget", type=float, default=20_000.0)
    parser.add_argument("--eval-every", type=float, default=250.0)
    parser.add_argument("--decision-mode", type=str,
                        choices=["search", "grid"], default="search",
                        help="adaptive targeting: 'search' = strict "
                             "multistart IPOPT worst-lambda search (the "
                             "method's mechanism, as at K=6; default); "
                             "'grid' = ablation-only coarse-grid argmax")
    parser.add_argument("--targeting-starts", type=int, default=64)
    parser.add_argument("--decision-grid", type=int, default=2001)
    parser.add_argument("--audit-grid", type=int, default=200_001)
    parser.add_argument("--figure", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = dict(SMOKE_INSTANCE if args.smoke else TRIAL_INSTANCE)
    home = K2_HOME / (f"B{args.budget:.0f}" + ("_SMOKE" if args.smoke else ""))
    home.mkdir(parents=True, exist_ok=True)

    def _adaptive_pol(a):
        if a.decision_mode == "search":
            if not ipopt_available():
                raise RuntimeError(
                    "IPOPT is required for --decision-mode search.")
            return _adaptive_policy(cfg["K"], a.targeting_starts)
        return _adaptive_policy_exact(a.decision_grid)

    if args.smoke:
        args.budget = 400.0
        args.eval_every = 25.0
        args.targeting_starts = 8
        args.decision_grid = 501
        args.audit_grid = 20_001
        home = K2_HOME / "B400_SMOKE"
        home.mkdir(parents=True, exist_ok=True)
        for r, s in ((4, 2), (4, 1)):
            grid = _sort_grid_for_warmstart(
                _uniform_simplex_grid(cfg["K"], r))
            a2 = argparse.Namespace(**{**vars(args), "s": s})
            _run_leg("baseline", _baseline_policy(grid), cfg, a2,
                     _leg_dir_k2(home, "baseline", r, s), {"r": r})
        a2 = argparse.Namespace(**{**vars(args), "s": 2})
        _run_leg("adaptive", _adaptive_pol(a2),
                 cfg, a2, _leg_dir_k2(home, "adaptive", None, 2, a2), {})
        a2 = argparse.Namespace(**{**vars(args), "s": 2})
        _figure(home, a2, cfg)
        # Cross-checks.
        for p in home.glob("*/summary.json"):
            with open(p, "r", encoding="utf-8") as fh:
                sm = json.load(fh)
            assert sm["grad_equiv_total"] <= args.budget + 1e-6
            assert np.isfinite(sm["final_audit"])
            hist = sm["audited_gn_history"]
            assert all(hist[i + 1] <= hist[i] + 1e-12
                       for i in range(len(hist) - 1)), "not monotone"
            npz = np.load(p.parent / "grams.npz")
            assert npz["fvals"].shape[1] == 2
            assert npz["seg_grads"].shape[0] == npz["fvals"].shape[0]
            ub_hist = sm["audited_gn_upper_history"]
            assert all(u >= v - 1e-15 for u, v in zip(ub_hist, hist))
            # Exact meter vs the K6-style multistart search: the search
            # is a lower bound of the true max, so it must land inside
            # the certified interval's reach: search <= upper bound,
            # and the exact value must not sit below the search.
            if ipopt_available():
                Ms = np.asarray(npz["gram_stack"], dtype=float)
                gs = _GramSet(list(Ms), cfg["K"])
                v_search, _lam = _maximise_GN_fast(
                    gs, prev_lam=None, tier="strict", max_starts=16)
                v_exact, _w, v_ub = exact_gn_1d(
                    Ms, grid_points=args.audit_grid, certify=True)
                assert v_exact >= float(v_search) - 1e-9, (
                    f"exact {v_exact} < search {v_search}")
                assert float(v_search) <= v_ub + 1e-9, (
                    f"search {v_search} above certified upper {v_ub}")
                print(f"[smoke] {p.parent.name}: search "
                      f"{float(v_search):.6e} <= exact {v_exact:.6e} <= "
                      f"upper {v_ub:.6e} (gap {v_ub - v_exact:.2e})  OK",
                      flush=True)
        assert (home / "pure_budget_K2_fronts.png").exists()
        assert (home / "front_metrics.json").exists()
        print("SMOKE OK", flush=True)
        return

    if args.figure:
        _figure(home, args, cfg)
        return
    if not args.run:
        raise SystemExit("pass --run baseline|adaptive, --figure or --smoke")

    out_dir = _leg_dir_k2(home, args.run, args.r, args.s, args)
    if (out_dir / "summary.json").exists() and not args.force:
        print(f"[skip] {out_dir} already has a summary (use --force)",
              flush=True)
        return
    if args.run == "baseline":
        grid = _sort_grid_for_warmstart(
            _uniform_simplex_grid(cfg["K"], args.r))
        _run_leg("baseline", _baseline_policy(grid), cfg, args, out_dir,
                 {"r": args.r})
    else:
        _run_leg("adaptive", _adaptive_pol(args), cfg, args, out_dir, {})


if __name__ == "__main__":
    main()
