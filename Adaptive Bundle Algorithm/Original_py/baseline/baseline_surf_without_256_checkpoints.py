"""baseline_surf_without_256_checkpoints.py — the SURF leg ("baseline1")
of the K = 2 pair campaign v2.  Design spec: BASELINE_SURF.md (+ Zh twin),
Sep 2, 2026; campaign plan: Note/Sep_2_note.md.

NEW FILE (Sep 2, 2026).  SURF (Jiang-Huang-Chen, arXiv:2605.20619)
Algorithm 1 / Rule 2 kept verbatim as the lambda-allocation strategy —
quantile weights through the arc-length CDF, chord estimate (its eq. 12),
PCHIP monotone interpolation, damped update (its eq. 13) — wrapped around
this project's own machinery: MSVRG segments with the walk rule injected
from stepper_core, the descent safeguard, the v1 budget meter and delivery
conventions.  Fixed budget, NO tolerance-based stopping anywhere.

Loop (single phase, runs to budget exhaustion):

  Phi_0 = identity on a 1001-point w-grid; N+1 slots all start at x0.
  round t:  w_n = Phi_t^{-1}(n/N), lambda_n = [w_n, 1-w_n]  (endpoints in)
            per slot: ONE MSVRG segment from the slot's own anchor
            (vertical warm start), charged full eval at the segment end
            (descent check + front measurement + Gram delivery in one),
            v1 accept/reject semantics per slot (ascent: L_scale *= 2,
            anchor stays; force-accept after MAX_SAFEGUARD_RETRIES).
            chord arc  s(w_{n+1}) = s(w_n) + max(||f_{n+1} - f_n||, eps_arc)
            over the slots' current accepted f-vectors, PCHIP -> Phi~,
            Phi_{t+1} = alpha*Phi~ + (1-alpha)*Phi_t   (alpha = 0.3).
  Partial final rounds deliver points but do not update Phi.

Stepper semantics (user-approved Sep 3, after the MODPO branch's
same-slot design): ONE stepper instance PER SLOT, state persisted
across rounds — in this leg the slot IS the node, so the design rule
"keep state within a node, clear on node switch" maps to per-slot
persistence (the slot's weight drifts a little each round; that is not
a node switch).  A slot's stepper is initialised at its first visit and
cleared only by its own safeguard ascents.  This matters most for the
adam core (moments need warm-up; a per-visit reset would keep it in
permanent warm-up) and is a few MB of extra state.

x0 is evaluated once, uncharged — the v1 house convention (every leg
identical, so the shared t = 0 anchor is fair); BASELINE_SURF.md's
"charged" wording is corrected by this file.

History (Sep 2): on the UNREGULARIZED pair the vertex solutions diverge
and verbatim SURF collapses onto the vertices (evidence kept under
v2_campaign/SMOKE/ and mu_smoke/).  A windowed-chord variant existed
briefly the same evening; the user's decision is to regularize the
problem instead (mu > 0 via objectives_mnist_pair_ridge), so the
windowing code was removed and this leg runs SURF's measurement
verbatim.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

import _layout  # noqa: F401
from bundle import validate_oracle_output  # noqa: E402
from baseline_svrg_certified_without_256_checkpoints import (  # noqa: E402
    _support_batch,
)
from run_pure_budget_K6_without_256_checkpoints import (  # noqa: E402
    MAX_SAFEGUARD_RETRIES,
    _Budget,
)
from run_pure_budget_K2_without_256_checkpoints import exact_gn_1d  # noqa: E402
from run_pure_budget_K2_mnist_pair_without_256_checkpoints import (  # noqa: E402
    INIT_SEED,
    PROBE_SEED,
    SAMPLER_SEED,
    _test_eval_stack,
)
from run_experiments import _json_ready  # noqa: E402
from objectives_mnist_pair import (  # noqa: E402
    load_mnist_pair,
    make_mnist_pair,
    make_pair_initial_point,
)
from stepper_core import make_stepper  # noqa: E402

PHI_GRID_POINTS = 1001
EPS_ARC = 1e-12
SURF_ALPHA = 0.3


def run_surf_leg(pair, cfg, args, out_dir, extra_cfg, N,
                 stepper_name="const", stepper_cfg=None,
                 sampler_seed=SAMPLER_SEED, surf_alpha=SURF_ALPHA,
                 mu=0.0, w_min=0.0):
    # w_min > 0 trims the weight dial to [w_min, 1 - w_min] (an affine
    # map after Phi^{-1}).  Exact vertices carry ZERO weight on one
    # objective, which is what lets that objective diverge and the
    # measured arc explode; at any w > 0 the scalarized minimizer keeps
    # both objectives finite, so the trimmed front is bounded.  Same
    # remedy as the MODPO branch's ADAPTIVE_LAMBDA_MIN endpoint-collapse
    # fix.  w_min = 0 is the verbatim dial.
    """SURF leg executor.  args needs budget / eval_every / audit_grid /
    smoke (same _Args contract as the stepper executor); N = number of
    segments on the weight dial (N+1 slots, endpoints included)."""
    a, b = pair
    t_build = time.time()
    if mu != 0.0:
        from objectives_mnist_pair_ridge import make_mnist_pair_ridge
        (_obj, _grad, L, joint_oracle, stoch, meta) = make_mnist_pair_ridge(
            a, b, mu, per_class=cfg["per_class"],
            batch_size=cfg["msvrg_batch"], sampler_seed=sampler_seed,
            init_seed=INIT_SEED, n_probes=cfg["n_probes"],
            probe_seed=PROBE_SEED)
    else:
        (_obj, _grad, L, joint_oracle, stoch, meta) = make_mnist_pair(
            a, b, per_class=cfg["per_class"], batch_size=cfg["msvrg_batch"],
            sampler_seed=sampler_seed, init_seed=INIT_SEED,
            n_probes=cfg["n_probes"], probe_seed=PROBE_SEED)
    K, n, d = meta["K"], meta["n"], meta["d"]
    x0 = make_pair_initial_point(INIT_SEED)
    L_arr = np.asarray(L, dtype=float)
    epoch_len = max(1, int(np.ceil(n / float(cfg["msvrg_batch"]))))
    scfg = dict(cfg)
    scfg.update(stepper_cfg or {})
    steppers = [make_stepper(stepper_name, d, scfg) for _ in range(N + 1)]
    stepper_inited = [False] * (N + 1)
    print(f"[surf_N{N}|{a}v{b}|{stepper_name}] instance in "
          f"{time.time() - t_build:.1f}s (n={n} d={d} "
          f"per_class={meta['per_class']} epoch_len={epoch_len} "
          f"L=[{L_arr[0]:.3f},{L_arr[1]:.3f}] seed={sampler_seed})",
          flush=True)

    f0, J0 = validate_oracle_output(*joint_oracle(x0), K, d)
    grams = [J0 @ J0.T]
    fvals = [np.asarray(f0, dtype=float)]
    thetas = [x0.copy()]

    # per-slot state: anchor point + cached full (f, J) + retry counter
    slot_x = [x0.copy() for _ in range(N + 1)]
    slot_f = [np.asarray(f0, dtype=float) for _ in range(N + 1)]
    slot_J = [J0 for _ in range(N + 1)]
    slot_retries = [0] * (N + 1)

    wgrid = np.linspace(0.0, 1.0, PHI_GRID_POINTS)
    phi = wgrid.copy()                       # Phi_0 = identity
    quantiles = np.arange(N + 1) / float(N)

    budget = _Budget(K, n, stoch, args.budget)
    L_scale = 1.0
    safeguard_retries = 0
    seg_grads, seg_lams = [0.0], [[np.nan] * K]
    ck_grads, ck_cpu, ck_m = [0.0], [0.0], [1]
    grad_at_ck = 0.0
    w_rounds, arc_lengths = [], []
    rounds_complete = 0
    t0 = time.time()
    overhead_seconds = 0.0                   # Phi inversion + update

    stopped = False
    while not stopped:
        t_ov = time.time()
        w_nodes = np.interp(quantiles, phi, wgrid)
        w_nodes[0], w_nodes[-1] = 0.0, 1.0
        if w_min > 0.0:
            w_nodes = w_min + (1.0 - 2.0 * w_min) * w_nodes
        overhead_seconds += time.time() - t_ov
        round_full = True
        for slot in range(N + 1):
            if not budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
                stopped, round_full = True, False
                break
            w = float(w_nodes[slot])
            lam = np.array([w, 1.0 - w])
            L_lam = float(lam @ L_arr)
            st = steppers[slot]
            if not stepper_inited[slot]:
                st.on_lambda_change(lam, L_lam, L_scale)
                stepper_inited[slot] = True
            g_a_full = slot_J[slot].T @ lam
            F_a = float(slot_f[slot] @ lam)
            st.start_segment(slot_x[slot], g_a_full, L_lam, L_scale,
                             epoch_len)
            stoch.set_anchor(slot_x[slot])
            y = slot_x[slot].copy()
            for _t in range(epoch_len):
                batch = _support_batch(stoch.sample_batch(), lam)
                g_y_S, g_a_S = stoch.grad_pair(y, lam, batch)
                y = st.step(y, (g_y_S - g_a_S + g_a_full))
            f_y, J_y = validate_oracle_output(*joint_oracle(y), K, d)
            budget.joint_calls += 1
            grams.append(J_y @ J_y.T)
            fvals.append(np.asarray(f_y, dtype=float))
            thetas.append(y.copy())
            seg_grads.append(float(budget.spent()))
            seg_lams.append([float(v) for v in lam])
            accepted = not (float(f_y @ lam)
                            > F_a + 1e-10 * (1.0 + abs(F_a)))
            if not accepted:
                L_scale *= 2.0
                safeguard_retries += 1
                slot_retries[slot] += 1
                if slot_retries[slot] > MAX_SAFEGUARD_RETRIES:
                    slot_x[slot], slot_f[slot], slot_J[slot] = y, f_y, J_y
                    slot_retries[slot] = 0
            else:
                slot_x[slot], slot_f[slot], slot_J[slot] = y, f_y, J_y
                slot_retries[slot] = 0
            st.on_segment_result(accepted, L_lam, L_scale)

            if budget.spent() - grad_at_ck >= args.eval_every:
                grad_at_ck = budget.spent()
                ck_grads.append(budget.spent())
                ck_cpu.append(time.time() - t0)
                ck_m.append(len(grams))

        if round_full:
            rounds_complete += 1
            w_rounds.append([float(v) for v in w_nodes])
            t_ov = time.time()
            F = np.asarray(slot_f, dtype=float)          # (N+1, 2), w-order
            chords = np.linalg.norm(np.diff(F, axis=0), axis=1)
            s_vals = np.concatenate(
                [[0.0], np.cumsum(np.maximum(chords, EPS_ARC))])
            arc_lengths.append(float(s_vals[-1]))
            s_interp = PchipInterpolator(w_nodes, s_vals)(
                np.clip(wgrid, w_nodes[0], w_nodes[-1]))
            phi_tilde = s_interp / s_interp[-1]
            phi = surf_alpha * phi_tilde + (1.0 - surf_alpha) * phi
            phi[0], phi[-1] = 0.0, 1.0
            overhead_seconds += time.time() - t_ov

    wall = time.time() - t0
    ck_grads.append(budget.spent())
    ck_cpu.append(wall)
    ck_m.append(len(grams))
    Ms = np.asarray(grams, dtype=float)
    print(f"[surf_N{N}|{a}v{b}|{stepper_name}] budget spent: "
          f"{budget.spent():.1f} of {args.budget} | segments={len(grams)-1} "
          f"| rounds={rounds_complete} | wall={wall:.1f}s "
          f"| overhead={overhead_seconds:.2f}s | L_scale={L_scale}",
          flush=True)

    # ---- post-hoc, off both axes: EXACT prefix audits (v1 meter) ----
    t_a = time.time()
    audited, audit_ws, audit_ub = [], [], []
    for m_ck in ck_m:
        v, w, ub = exact_gn_1d(Ms[:m_ck], grid_points=args.audit_grid,
                               certify=True)
        assert ub >= v - 1e-15
        audited.append(float(v))
        audit_ws.append(float(w))
        audit_ub.append(float(ub))
    mono_viol = sum(int(audited[i] > audited[i - 1] + 1e-12)
                    for i in range(1, len(audited)) if ck_m[i] >= ck_m[i - 1])
    if mono_viol:                       # 1e-12-level meter jitter: record
        import warnings
        warnings.warn(f"exact prefix audit: {mono_viol} non-monotone "
                      f"step(s) (recorded, not clipped)")
    audit_seconds = time.time() - t_a

    t_t = time.time()
    X_test, y_test = load_mnist_pair(a, b, train=False)
    test_ce, test_err = _test_eval_stack(thetas, X_test, y_test)
    test_seconds = time.time() - t_t

    if args.smoke:
        idx = len(thetas) // 2
        f_re, _J_re = joint_oracle(thetas[idx])
        assert np.allclose(f_re, fvals[idx], atol=1e-9), (
            f"theta round-trip failed at {idx}")
        assert np.all(np.diff(phi) >= -1e-15), "Phi not monotone"
        print(f"[smoke] theta round-trip + Phi monotone OK", flush=True)

    summary = {
        "protocol": ("pure fixed budget at K=2, SURF leg (v2): quantile "
                     "weights through the arc-length CDF (SURF Alg. 1 "
                     "verbatim: chord estimate, PCHIP, damped update "
                     f"alpha={surf_alpha}), one MSVRG segment per slot "
                     "per round, vertical warm start, walk rule from "
                     "stepper_core, v1 accept/reject + budget meter; no "
                     "tolerance anywhere; stop = budget"),
        "policy": "surf",
        "pair": [int(a), int(b)],
        "config_instance": _json_ready(cfg),
        "budget": args.budget, "eval_every": args.eval_every,
        "audit_grid": args.audit_grid,
        "mu": float(mu),
        "surf": {"N": int(N), "alpha": float(surf_alpha),
                 "phi_grid_points": PHI_GRID_POINTS,
                 "eps_arc": EPS_ARC,
                 "w_min": float(w_min),
                 "rounds_complete": int(rounds_complete),
                 "arc_length_history": arc_lengths,
                 "w_first_round": w_rounds[0] if w_rounds else None,
                 "w_last_round": w_rounds[-1] if w_rounds else None,
                 "phi_final_coarse": [float(v) for v in
                                     np.interp(np.linspace(0, 1, 21),
                                               wgrid, phi)]},
        "stepper": {"name": stepper_name,
                    "cfg": _json_ready(stepper_cfg or {}),
                    "per_slot": True,
                    "final_diag": _json_ready([st.diag()
                                               for st in steppers])},
        "sampler_seed": int(sampler_seed),
        "extra": _json_ready(extra_cfg),
        "K": K, "d": d, "n": n, "per_class": meta["per_class"],
        "L_calibrated": [float(v) for v in L_arr],
        "grad_equiv_total": float(budget.spent()),
        "joint_calls": int(budget.joint_calls),
        "wall_seconds": wall,
        "decision_seconds": overhead_seconds,   # Phi machinery = decisions
        "segments_total": int(len(grams) - 1),
        "m_final": int(Ms.shape[0]),
        "L_scale_final": L_scale,
        "safeguard_retries": int(safeguard_retries),
        "ck_grads": ck_grads, "ck_cpu": ck_cpu, "ck_m": ck_m,
        "final_audit": float(audited[-1]),
        "final_audit_upper": float(audit_ub[-1]),
        "audited_gn_history": audited,
        "audited_gn_upper_history": audit_ub,
        "audited_gn_norm_history": [float(np.sqrt(max(v, 0.0)))
                                    for v in audited],
        "audit_w_history": audit_ws,
        "w_star": float(audit_ws[-1]),
        "audit_seconds": audit_seconds,
        "test_seconds": test_seconds,
        "final_test_ce": [float(v) for v in test_ce[-1]],
        "final_test_err": [float(v) for v in test_err[-1]],
        "init_seed": INIT_SEED, "probe_seed": PROBE_SEED,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "grams.npz", gram_stack=Ms,
                        fvals=np.asarray(fvals),
                        lam_history=np.asarray(
                            [w for rnd in w_rounds for w in rnd],
                            dtype=float),
                        seg_grads=np.asarray(seg_grads, dtype=float),
                        seg_lams=np.asarray(seg_lams, dtype=float),
                        test_ce=test_ce, test_err=test_err,
                        phi_final=phi)
    np.savez_compressed(out_dir / "thetas.npz",
                        theta_stack=np.asarray(thetas))
    print(f"[surf_N{N}|{a}v{b}|{stepper_name}] final EXACT audit = "
          f"{audited[-1]:.6e} (norm {np.sqrt(max(audited[-1], 0)):.6e}) "
          f"-> {out_dir}", flush=True)
    return summary
