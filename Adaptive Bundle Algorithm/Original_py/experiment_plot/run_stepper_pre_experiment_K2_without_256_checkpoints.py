"""run_stepper_pre_experiment_K2_without_256_checkpoints.py — campaign-v2
stages S1 (gates) and S2 (optimization-core pre-experiment) on the locked
pair 4v9.  Design: ADAPTIVE_STEPPERS.md (Sep-2 revision) + Note/Sep_2_note.md.

NEW FILE (Sep 2, 2026).  No existing file is modified.  The v1 executor
``_run_leg_pair`` is imported UNTOUCHED as the Gate-0 reference; this file
adds ``_run_leg_pair_stepper`` — the same executor with the inner walk
delegated to ``stepper_core`` (Core Engine/stepper_core.py).  With
stepper="const" the float-op sequence is identical to v1 and no extra
randomness is consumed, so Gate 0 demands BIT-EXACT equality.

Stages:

* ``--stage gate0``  v1 adaptive-CCP smoke run vs stepper="const" smoke
  run on 4v9 (B=800, per_class=300): gram_stack / fvals / lam_history /
  theta_stack / seg_grads / checkpoint arrays / audit histories must all
  be exactly equal.
* ``--stage gate1``  bb, adagrad(alpha_mult=3), adam(defaults) on the
  same smoke instance: everything finite, L_scale bounded, audits
  monotone (executor asserts), final audit <= initial.  NOTE: run on the
  locked pair's smoke instance rather than the bandit toy — same intent
  (cheap NaN/divergence catch), strictly more relevant machinery.
* ``--stage s2``     the pre-experiment: 11 configs x 3 sampler seeds
  {41, 141, 241}, B = 2,500, eval_every = 50, s = 5, adaptive-CCP leg
  only; judge = best-so-far worst GN (NORM scale) vs grad_equiv, CPU
  secondary, ties to the simpler stepper.  Run only after the S1
  sign-off.

Usage:
    python run_stepper_pre_experiment_K2_without_256_checkpoints.py --stage gate0
    python run_stepper_pre_experiment_K2_without_256_checkpoints.py --stage gate1
    python run_stepper_pre_experiment_K2_without_256_checkpoints.py --stage s2
"""

from __future__ import annotations

import argparse
import json
import time

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

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
from run_pure_budget_K2_ccp_without_256_checkpoints import (  # noqa: E402
    _ccp_policy,
    _stats_block,
)
from run_pure_budget_K2_mnist_pair_without_256_checkpoints import (  # noqa: E402
    CAMPAIGN_ROOT,
    INIT_SEED,
    PROBE_SEED,
    SAMPLER_SEED,
    _run_leg_pair,
    _test_eval_stack,
)
from run_experiments import _json_ready  # noqa: E402
from ccp_lambda_solver import CCPConfig  # noqa: E402
from objectives_mnist_pair import (  # noqa: E402
    load_mnist_pair,
    make_mnist_pair,
    make_pair_initial_point,
)
from stepper_core import make_stepper  # noqa: E402

PAIR = (4, 9)                       # locked by the user, Sep 2 (S0)
S2_HOME = CAMPAIGN_ROOT / "stepper_pre_experiment"
GATE_HOME = S2_HOME / "gates"
S2_SEEDS = (41, 141, 241)

SMOKE_CFG = dict(per_class=300, msvrg_batch=256, msvrg_step_const=0.1,
                 msvrg_momentum=0.5, n_probes=5)
S2_CFG = dict(per_class=None, msvrg_batch=1024, msvrg_step_const=0.1,
              msvrg_momentum=0.5, n_probes=40)

S2_CONFIGS = [
    ("const", {}),
    ("bb", {}),
    ("adagrad", {"adagrad_alpha_mult": 1.0}),
    ("adagrad", {"adagrad_alpha_mult": 3.0}),
    ("adagrad", {"adagrad_alpha_mult": 10.0}),
    ("adam", {"adam_alpha": 1e-4, "adam_beta2": 0.9}),
    ("adam", {"adam_alpha": 1e-4, "adam_beta2": 0.99}),
    ("adam", {"adam_alpha": 3e-4, "adam_beta2": 0.9}),
    ("adam", {"adam_alpha": 3e-4, "adam_beta2": 0.99}),
    ("adam", {"adam_alpha": 1e-3, "adam_beta2": 0.9}),
    ("adam", {"adam_alpha": 1e-3, "adam_beta2": 0.99}),
]


def _cfg_tag(name, scfg):
    if not scfg:
        return name
    bits = []
    for k in sorted(scfg):
        v = scfg[k]
        bits.append(f"{k.split('_')[-1]}{v:g}")
    return name + "_" + "_".join(bits)


class _Args:
    def __init__(self, budget, eval_every, audit_grid, s, smoke):
        self.budget = budget
        self.eval_every = eval_every
        self.audit_grid = audit_grid
        self.s = s
        self.smoke = smoke


def _run_leg_pair_stepper(policy_name, next_lam, pair, cfg, args, out_dir,
                          extra_cfg, stepper_name="const", stepper_cfg=None,
                          sampler_seed=SAMPLER_SEED, mu=0.0):
    """The v1 executor with the walk rule delegated to stepper_core.
    Identical to ``_run_leg_pair`` except: (a) the four marked stepper
    hook sites, (b) additive summary fields (stepper block + norm-scale
    audit history), (c) the sampler seed is a parameter (S2 seeds)."""
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
    X_np, y_np = meta.pop("_X"), meta.pop("_y")
    K, n, d = meta["K"], meta["n"], meta["d"]
    x0 = make_pair_initial_point(INIT_SEED)
    L_arr = np.asarray(L, dtype=float)
    epoch_len = max(1, int(np.ceil(n / float(cfg["msvrg_batch"]))))
    scfg = dict(cfg)
    scfg.update(stepper_cfg or {})
    stepper = make_stepper(stepper_name, d, scfg)        # stepper hook 0
    print(f"[{policy_name}|{a}v{b}|{stepper_name}] instance in "
          f"{time.time() - t_build:.1f}s (n={n} d={d} "
          f"per_class={meta['per_class']} epoch_len={epoch_len} "
          f"L=[{L_arr[0]:.3f},{L_arr[1]:.3f}] seed={sampler_seed})",
          flush=True)

    f0, J0 = validate_oracle_output(*joint_oracle(x0), K, d)
    grams = [J0 @ J0.T]
    fvals = [np.asarray(f0, dtype=float)]
    thetas = [x0.copy()]
    chain_x, chain_J, chain_f = x0.copy(), J0, f0

    budget = _Budget(K, n, stoch, args.budget)
    L_scale = 1.0
    safeguard_retries = 0
    lam_history = []
    seg_grads, seg_lams = [0.0], [[np.nan] * K]
    seg_diag = []
    ck_grads, ck_cpu, ck_m = [0.0], [0.0], [1]
    grad_at_ck = 0.0
    t0 = time.time()
    decision_seconds = 0.0

    prev_lam = None
    while budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
        t_dec = time.time()
        lam = np.asarray(next_lam(grams, fvals, prev_lam), dtype=float)
        decision_seconds += time.time() - t_dec
        L_lam = float(lam @ L_arr)
        if prev_lam is None or not np.array_equal(lam, prev_lam):
            stepper.on_lambda_change(lam, L_lam, L_scale)  # stepper hook 1
        prev_lam = lam
        lam_history.append(lam.copy())

        retries_here = 0
        for _k in range(args.s):
            if not budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
                break
            g_a_full = chain_J.T @ lam
            F_a = float(chain_f @ lam)
            stepper.start_segment(chain_x, g_a_full, L_lam, L_scale,
                                  epoch_len)                # stepper hook 2
            stoch.set_anchor(chain_x)
            y = chain_x.copy()
            for _t in range(epoch_len):
                batch = _support_batch(stoch.sample_batch(), lam)
                g_y_S, g_a_S = stoch.grad_pair(y, lam, batch)
                y = stepper.step(y, (g_y_S - g_a_S + g_a_full))  # hook 3
            f_y, J_y = validate_oracle_output(*joint_oracle(y), K, d)
            budget.joint_calls += 1
            grams.append(J_y @ J_y.T)
            fvals.append(np.asarray(f_y, dtype=float))
            thetas.append(y.copy())
            seg_grads.append(float(budget.spent()))
            seg_lams.append([float(t) for t in lam])
            accepted = not (float(f_y @ lam) > F_a + 1e-10 * (1.0 + abs(F_a)))
            if not accepted:
                L_scale *= 2.0
                safeguard_retries += 1
                retries_here += 1
                if retries_here > MAX_SAFEGUARD_RETRIES:
                    chain_x, chain_J, chain_f = y, J_y, f_y
                    retries_here = 0
            else:
                chain_x, chain_J, chain_f = y, J_y, f_y
                retries_here = 0
            stepper.on_segment_result(accepted, L_lam, L_scale)  # hook 4
            seg_diag.append(stepper.diag())

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
    print(f"[{policy_name}|{a}v{b}|{stepper_name}] budget spent: "
          f"{budget.spent():.1f} of {args.budget} | segments={len(grams) - 1} "
          f"| wall={wall:.1f}s | decision={decision_seconds:.1f}s "
          f"| L_scale={L_scale}", flush=True)

    t_a = time.time()
    audited, audit_ws, audit_ub = [], [], []
    for m_ck in ck_m:
        v, w, ub = exact_gn_1d(Ms[:m_ck], grid_points=args.audit_grid,
                               certify=True)
        assert ub >= v - 1e-15
        audited.append(float(v))
        audit_ws.append(float(w))
        audit_ub.append(float(ub))
    # Prefix audits are monotone by mathematics; the exact meter's
    # closed-form polish can still jitter at the 1e-12 level on long
    # stacks (seen at B=10,000, Sep 3).  Record and warn instead of
    # aborting the leg (the K3 executor's convention).
    mono_viol = sum(int(audited[i] > audited[i - 1] + 1e-12)
                    for i in range(1, len(audited)) if ck_m[i] >= ck_m[i - 1])
    if mono_viol:
        import warnings
        warnings.warn(f"exact prefix audit: {mono_viol} non-monotone "
                      f"step(s) at the 1e-12 level (recorded, not clipped)")
    audit_seconds = time.time() - t_a

    t_t = time.time()
    X_test, y_test = load_mnist_pair(a, b, train=False)
    test_ce, test_err = _test_eval_stack(thetas, X_test, y_test)
    test_seconds = time.time() - t_t

    if args.smoke:
        idx = len(thetas) // 2
        f_re, _J_re = joint_oracle(thetas[idx])
        assert np.allclose(f_re, fvals[idx], atol=1e-9), (
            f"theta round-trip failed at {idx}: {f_re} vs {fvals[idx]}")
        print(f"[smoke] theta round-trip OK at index {idx}", flush=True)

    lam_arr = np.asarray(lam_history, dtype=float)
    distinct = (np.unique(lam_arr.round(12), axis=0).shape[0]
                if lam_arr.size else 0)
    n_test = [int((y_test == k).sum()) for k in (0, 1)]
    summary = {
        "audit_mono_violations": int(mono_viol),
        "protocol": ("pure fixed budget at K=2 (v2 stepper executor): "
                     "shared segment unit, shared s, chain warm start; "
                     "walk rule delegated to stepper_core; ONLY the "
                     "next-lambda policy and the stepper differ between "
                     "runs; no tolerance anywhere; stop = budget; EXACT "
                     "1-D meter for audits (squared scale, norm-scale "
                     "history added)"),
        "policy": policy_name,
        "pair": [int(a), int(b)],
        "config_instance": _json_ready(cfg),
        "budget": args.budget, "s": args.s, "eval_every": args.eval_every,
        "audit_grid": args.audit_grid,
        "meter": "exact-1d (audits); test split = ALL official t10k rows",
        "extra": _json_ready(extra_cfg),
        "stepper": {"name": stepper_name,
                    "cfg": _json_ready(stepper_cfg or {}),
                    "final_diag": _json_ready(stepper.diag())},
        "sampler_seed": int(sampler_seed),
        "mu": float(mu),
        "K": K, "d": d, "n": n, "per_class": meta["per_class"],
        "n_test": n_test,
        "L_calibrated": [float(v) for v in L_arr],
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
        "stepper_seg_diag": _json_ready(seg_diag),
        "init_seed": INIT_SEED, "probe_seed": PROBE_SEED,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "grams.npz", gram_stack=Ms,
                        fvals=np.asarray(fvals),
                        lam_history=lam_arr,
                        seg_grads=np.asarray(seg_grads, dtype=float),
                        seg_lams=np.asarray(seg_lams, dtype=float),
                        test_ce=test_ce, test_err=test_err)
    np.savez_compressed(out_dir / "thetas.npz",
                        theta_stack=np.asarray(thetas))
    print(f"[{policy_name}|{a}v{b}|{stepper_name}] final EXACT audit = "
          f"{audited[-1]:.6e} (norm {np.sqrt(max(audited[-1], 0)):.6e}) "
          f"-> {out_dir}", flush=True)
    return summary


def _smoke_args():
    return _Args(budget=800.0, eval_every=100.0, audit_grid=20_001, s=2,
                 smoke=True)


def _ccp_cfg(seed=0):
    return CCPConfig(N0=2000, r=10, seed=seed, seed_sampler="exp",
                     adaptive_seed_schedule=False)


def gate0():
    """v1 executor vs stepper='const': bit-exact equality on 4v9 smoke."""
    args = _smoke_args()
    ref_dir = GATE_HOME / "gate0_ref"
    new_dir = GATE_HOME / "gate0_const"
    stats_ref: list = []
    sm_ref = _run_leg_pair("adaptive_ccp", _ccp_policy(2, _ccp_cfg(),
                                                       stats_ref),
                           PAIR, dict(SMOKE_CFG), args, ref_dir,
                           {"role": "gate0 reference (v1 executor)"})
    stats_new: list = []
    sm_new = _run_leg_pair_stepper(
        "adaptive_ccp", _ccp_policy(2, _ccp_cfg(), stats_new), PAIR,
        dict(SMOKE_CFG), args, new_dir,
        {"role": "gate0 candidate (stepper executor)"},
        stepper_name="const")

    npz_ref = np.load(ref_dir / "grams.npz")
    npz_new = np.load(new_dir / "grams.npz")
    th_ref = np.load(ref_dir / "thetas.npz")["theta_stack"]
    th_new = np.load(new_dir / "thetas.npz")["theta_stack"]
    checks = {
        "gram_stack": np.array_equal(npz_ref["gram_stack"],
                                     npz_new["gram_stack"]),
        "fvals": np.array_equal(npz_ref["fvals"], npz_new["fvals"]),
        "lam_history": np.array_equal(npz_ref["lam_history"],
                                      npz_new["lam_history"]),
        "seg_grads": np.array_equal(npz_ref["seg_grads"],
                                    npz_new["seg_grads"]),
        "theta_stack": np.array_equal(th_ref, th_new),
        "test_ce": np.array_equal(npz_ref["test_ce"], npz_new["test_ce"]),
        "grad_equiv_total": sm_ref["grad_equiv_total"]
        == sm_new["grad_equiv_total"],
        "L_scale_final": sm_ref["L_scale_final"] == sm_new["L_scale_final"],
        "safeguard_retries": sm_ref["safeguard_retries"]
        == sm_new["safeguard_retries"],
        "ck_m": sm_ref["ck_m"] == sm_new["ck_m"],
        "ck_grads": sm_ref["ck_grads"] == sm_new["ck_grads"],
        "audited_gn_history": sm_ref["audited_gn_history"]
        == sm_new["audited_gn_history"],
    }
    report = {"gate": 0, "pair": list(PAIR), "checks": checks,
              "pass": all(checks.values())}
    GATE_HOME.mkdir(parents=True, exist_ok=True)
    (GATE_HOME / "gate0_report.json").write_text(
        json.dumps(report, indent=2))
    for k, v in checks.items():
        print(f"[gate0] {k:20s} {'OK' if v else 'MISMATCH'}", flush=True)
    print(f"[gate0] {'PASS' if report['pass'] else 'FAIL'}", flush=True)
    return report


def gate1():
    """Safety: the three adaptive steppers on the 4v9 smoke instance."""
    args = _smoke_args()
    arms = [("bb", {}),
            ("adagrad", {"adagrad_alpha_mult": 3.0}),
            ("adam", {"adam_alpha": 3e-4, "adam_beta2": 0.99})]
    rows = []
    for name, scfg in arms:
        tag = _cfg_tag(name, scfg)
        out_dir = GATE_HOME / f"gate1_{tag}"
        stats: list = []
        sm = _run_leg_pair_stepper(
            "adaptive_ccp", _ccp_policy(2, _ccp_cfg(), stats), PAIR,
            dict(SMOKE_CFG), args, out_dir, {"role": f"gate1 {tag}"},
            stepper_name=name, stepper_cfg=scfg)
        npz = np.load(out_dir / "grams.npz")
        finite = (np.isfinite(npz["gram_stack"]).all()
                  and np.isfinite(npz["fvals"]).all()
                  and np.isfinite(np.load(out_dir / "thetas.npz")
                                  ["theta_stack"]).all())
        hist = sm["audited_gn_history"]
        rows.append({
            "stepper": tag,
            "finite": bool(finite),
            "L_scale_final": sm["L_scale_final"],
            "L_scale_bounded": bool(sm["L_scale_final"] <= 2.0 ** 20),
            "safeguard_retries": sm["safeguard_retries"],
            "audit_monotone": True,      # executor asserts, reaching here = ok
            "progress": bool(hist[-1] <= hist[0]),
            "final_audit_norm": float(np.sqrt(max(hist[-1], 0.0))),
        })
    ok = all(r["finite"] and r["L_scale_bounded"] and r["progress"]
             for r in rows)
    report = {"gate": 1, "pair": list(PAIR), "arms": rows, "pass": ok}
    (GATE_HOME / "gate1_report.json").write_text(
        json.dumps(report, indent=2))
    for r in rows:
        print(f"[gate1] {r['stepper']:22s} finite={r['finite']} "
              f"L_scale={r['L_scale_final']:g} "
              f"retries={r['safeguard_retries']} "
              f"final_GN={r['final_audit_norm']:.4e} "
              f"progress={r['progress']}", flush=True)
    print(f"[gate1] {'PASS' if ok else 'FAIL'}", flush=True)
    return report


def stage_s2(force=False):
    """The pre-experiment: 11 configs x 3 seeds, adaptive-CCP leg,
    B = 2,500.  Judge assembled into s2_summary.json + figures."""
    import matplotlib.pyplot as plt
    args = _Args(budget=2_500.0, eval_every=50.0, audit_grid=20_001, s=5,
                 smoke=False)
    results = {}
    t_all = time.time()
    for name, scfg in S2_CONFIGS:
        tag = _cfg_tag(name, scfg)
        for seed in S2_SEEDS:
            out_dir = S2_HOME / f"{tag}_seed{seed}"
            if (out_dir / "summary.json").exists() and not force:
                sm = json.loads((out_dir / "summary.json").read_text())
                print(f"[s2] skip {tag} seed{seed} (resume)", flush=True)
            else:
                stats: list = []
                sm = _run_leg_pair_stepper(
                    "adaptive_ccp", _ccp_policy(2, _ccp_cfg(), stats),
                    PAIR, dict(S2_CFG), args, out_dir,
                    {"role": f"S2 pre-experiment {tag} seed{seed}"},
                    stepper_name=name, stepper_cfg=scfg,
                    sampler_seed=seed)
                sm["ccp"] = _stats_block(stats)
                (out_dir / "summary.json").write_text(
                    json.dumps(_json_ready(sm), indent=2), encoding="utf-8")
            results.setdefault(tag, []).append(sm)

    # ---- judge: per config, mean final norm-scale worst GN + curves ----
    board = []
    for tag, sms in results.items():
        finals = [sm["audited_gn_norm_history"][-1] for sm in sms]
        walls = [sm["wall_seconds"] for sm in sms]
        board.append({"config": tag,
                      "final_worst_gn_norm_mean": float(np.mean(finals)),
                      "final_worst_gn_norm_per_seed": [float(v)
                                                       for v in finals],
                      "wall_seconds_mean": float(np.mean(walls)),
                      "safeguard_retries": [sm["safeguard_retries"]
                                            for sm in sms]})
    board.sort(key=lambda r: r["final_worst_gn_norm_mean"])
    (S2_HOME / "s2_summary.json").write_text(
        json.dumps({"pair": list(PAIR), "budget": args.budget,
                    "seeds": list(S2_SEEDS), "board": board,
                    "total_wall_seconds": time.time() - t_all}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for tag, sms in sorted(results.items()):
        g = np.asarray(sms[0]["ck_grads"], dtype=float)
        curves = [np.sqrt(np.maximum(np.asarray(sm["audited_gn_history"]),
                                     0.0)) for sm in sms]
        L = min(len(c) for c in curves)
        mean_curve = np.mean([c[:L] for c in curves], axis=0)
        axes[0].plot(g[:L], mean_curve, lw=1.2, label=tag)
        cpu = np.mean([np.asarray(sm["ck_cpu"], dtype=float)[:L]
                       for sm in sms], axis=0)
        axes[1].plot(cpu, mean_curve, lw=1.2, label=tag)
    axes[0].set_xlabel("total gradient evaluations (grad_equiv)")
    axes[1].set_xlabel("CPU seconds")
    for ax in axes:
        ax.set_ylabel("best-so-far worst GN (norm)")
        ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[0].set_title(f"S2 pre-experiment, pair {PAIR[0]}v{PAIR[1]}, "
                      f"B={args.budget:.0f}, mean of {len(S2_SEEDS)} seeds")
    fig.tight_layout()
    fig.savefig(S2_HOME / "s2_worst_gn_curves.png", dpi=150)
    print("[s2] board (best first):", flush=True)
    for r in board:
        print(f"    {r['config']:24s} GN={r['final_worst_gn_norm_mean']:.5e} "
              f"wall={r['wall_seconds_mean']:.0f}s", flush=True)
    return board


S2_EXT_BUDGET = 10_000.0
S2_EXT_MU = 1e-3            # == CAMPAIGN_MU of run_surf_compare_K2 (the
                            # campaign problem; kept literal here to avoid
                            # a circular import)
S2_EXT_CONFIGS = S2_CONFIGS  # user decision Sep 3: ALL 11 configs — this
                             # figure replaces the S2 figure in the report


def stage_s2_extend(force=False):
    """The AUTHORITATIVE optimizer-core selection (user decisions Sep 3):
    all 11 S2 configs, 3 seeds, B = 10,000 (4x the S2 tier), on the
    CAMPAIGN problem (ridge mu = 1e-3) — so the core is chosen on the
    same instance every leg runs on.  The mu = 0 S2 run is demoted to a
    preliminary screen.  Question answered on the way: does adam's late
    edge persist past B = 2,500?"""
    import matplotlib.pyplot as plt
    args = _Args(budget=S2_EXT_BUDGET, eval_every=200.0,
                 audit_grid=20_001, s=5, smoke=False)
    home = S2_HOME / f"extended_B{int(S2_EXT_BUDGET)}_mu{S2_EXT_MU:g}"
    results = {}
    t_all = time.time()
    for name, scfg in S2_EXT_CONFIGS:
        tag = _cfg_tag(name, scfg)
        for seed in S2_SEEDS:
            out_dir = home / f"{tag}_seed{seed}"
            if (out_dir / "summary.json").exists() and not force:
                sm = json.loads((out_dir / "summary.json").read_text())
                print(f"[s2-ext] skip {tag} seed{seed}", flush=True)
            else:
                stats: list = []
                sm = _run_leg_pair_stepper(
                    "adaptive_ccp", _ccp_policy(2, _ccp_cfg(), stats),
                    PAIR, dict(S2_CFG), args, out_dir,
                    {"role": f"S2 extension {tag} seed{seed}",
                     "mu": S2_EXT_MU},
                    stepper_name=name, stepper_cfg=scfg,
                    sampler_seed=seed, mu=S2_EXT_MU)
                sm["ccp"] = _stats_block(stats)
                (out_dir / "summary.json").write_text(
                    json.dumps(_json_ready(sm), indent=2), encoding="utf-8")
            results.setdefault(tag, []).append(sm)

    board = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for tag, sms in sorted(results.items()):
        cs = [np.asarray(sm["audited_gn_norm_history"], dtype=float)
              for sm in sms]
        L = min(len(c) for c in cs)
        g = np.asarray(sms[0]["ck_grads"], dtype=float)[:L]
        cpu = np.mean([np.asarray(sm["ck_cpu"], dtype=float)[:L]
                       for sm in sms], axis=0)
        mean_c = np.mean([c[:L] for c in cs], axis=0)
        axes[0].plot(g, mean_c, lw=1.4, label=tag)
        axes[1].plot(cpu, mean_c, lw=1.4, label=tag)
        # value at the S2 budget and at the end, for the board
        i25 = int(np.searchsorted(g, 2500.0))
        board.append({"config": tag,
                      "gn_at_2500": float(mean_c[min(i25, L - 1)]),
                      "gn_final": float(mean_c[-1]),
                      "per_seed_final": [float(c[-1]) for c in cs]})
    for ax in axes:
        ax.axvline(2500.0, color="gray", lw=0.8, ls=":") if ax is axes[0] \
            else None
        ax.set_ylabel("best-so-far worst GN (norm)")
        ax.set_yscale("log"); ax.legend(fontsize=8)
    axes[0].set_xlabel("total gradient evaluations (grad_equiv)")
    axes[1].set_xlabel("CPU seconds")
    axes[0].set_title(f"core selection, ridge mu={S2_EXT_MU:g}, "
                      f"B={int(S2_EXT_BUDGET)}, pair {PAIR[0]}v{PAIR[1]}, "
                      f"mean of {len(S2_SEEDS)} seeds", fontsize=10)
    axes[1].set_title("dotted line (left) = S2 screen budget 2,500",
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(home / "s2_extended_curves.png", dpi=150)
    board.sort(key=lambda r: r["gn_final"])
    (home / "s2_extended_summary.json").write_text(json.dumps(
        {"budget": S2_EXT_BUDGET, "seeds": list(S2_SEEDS), "board": board,
         "total_wall_seconds": time.time() - t_all}, indent=2))
    print("[s2-ext] board (best final first):", flush=True)
    for r in board:
        print(f"    {r['config']:24s} @2500={r['gn_at_2500']:.4e} "
              f"final={r['gn_final']:.4e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",
                        choices=["gate0", "gate1", "s2", "s2-extend"],
                        required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "gate0":
        gate0()
    elif args.stage == "gate1":
        gate1()
    elif args.stage == "s2":
        stage_s2(force=args.force)
    else:
        stage_s2_extend(force=args.force)


if __name__ == "__main__":
    main()
