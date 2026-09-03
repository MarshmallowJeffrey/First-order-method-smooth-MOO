"""run_k3_stepper_campaign_without_256_checkpoints.py — K = 3 migration of
the adaptive optimization cores (campaign v2 design, Sep 3, 2026).

NEW FILE.  Nothing existing is modified.  The K = 3 ridge executor
``_run_leg_triple_ridge`` (Aug 26) is imported UNTOUCHED as the Gate-0
reference; this file adds ``_run_leg_triple_stepper`` — the same
executor with the inner walk delegated to ``stepper_core`` (the K-agnostic
module built for K = 2) and a ``sampler_seed`` parameter.

User decisions (Sep 3):
* problem: ridge, mu = 1e-4 (the K3 house value; the existing const-core
  campaign on the same triple/mu is the historical comparison set);
* triple {4,7,9} (TAG Phase 1 top-1) — S0 done, nothing to rerun;
* optimization cores INHERITED from K = 2: core A = adagrad x10,
  core B = adam(1e-3, 0.9).  No core scan at K = 3 (future work);
* S3 = uniform ladder r in {10, 20, 30} rescanned under the new cores;
  no SURF leg at K = 3 (needs a surface-density generalisation);
* S4 = per core {uniform@r*, adaptive-CCP}, seed 41, B = 40,000, versus
  the const-core Exp-6 runs (triple_4v7v9_B40000_mu0.0001).

Stages:
    --stage gate0     ridge executor vs stepper="const": bit-exact (smoke)
    --stage gate1     adagrad x10 / adam on the smoke instance: finite,
                      bounded L_scale, progress
    --stage ladders   2 cores x r in {10,20,30} x seeds {41,141,241},
                      B = 5,000 (1/8 of Exp 5/6) -> r* per core
    --stage main      2 cores x {uniform@r*, adaptive} x seed 41 x
                      B = 40,000 + figures (norm-scale worst GN, both axes,
                      const-core Exp-6 curves overlaid)
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout  # noqa: F401
from bundle import validate_oracle_output  # noqa: E402
from baseline_svrg_certified_without_256_checkpoints import (  # noqa: E402
    _support_batch,
)
from baseline_without_256_checkpoints import (  # noqa: E402
    _sort_grid_for_warmstart,
    _uniform_simplex_grid,
)
from run_pure_budget_K6_without_256_checkpoints import (  # noqa: E402
    MAX_SAFEGUARD_RETRIES,
    _baseline_policy,
    _Budget,
)
from run_pure_budget_K2_ccp_without_256_checkpoints import (  # noqa: E402
    _ccp_policy,
    _stats_block,
)
from run_pure_budget_K3_mnist_triple_without_256_checkpoints import (  # noqa: E402
    CAMPAIGN_ROOT,
    INIT_SEED,
    PROBE_SEED,
    SAMPLER_SEED,
    _audit_instruments,
    _grid_maxmin,
    _test_eval_stack,
)
from run_pure_budget_K3_mnist_triple_ridge_without_256_checkpoints import (  # noqa: E402
    _run_leg_triple_ridge,
)
from run_experiments import _json_ready  # noqa: E402
from ccp_lambda_solver import CCPConfig  # noqa: E402
from objectives_mnist_triple import (  # noqa: E402
    load_mnist_triple,
    make_mnist_triple,
    make_triple_initial_point,
)
from objectives_mnist_triple_ridge import make_mnist_triple_ridge  # noqa: E402
from stepper_core import make_stepper  # noqa: E402

K = 3
TRIPLE = (4, 7, 9)
CAMPAIGN_MU = 1e-4
K3_HOME = CAMPAIGN_ROOT / "v2_stepper_mu0.0001"
GATE_HOME = K3_HOME / "gates"
LADDER_HOME = K3_HOME / "ladders"
MAIN_HOME = K3_HOME / "main"
EXP6_HOME = CAMPAIGN_ROOT / "triple_4v7v9_B40000_mu0.0001"   # const-core ref

CORES = [
    ("adagrad_x10", "adagrad", {"adagrad_alpha_mult": 10.0}),
    ("adam_1e-3_b0.9", "adam", {"adam_alpha": 1e-3, "adam_beta2": 0.9}),
]
LADDER_RS = [10, 20, 30]
LADDER_SEEDS = (41, 141, 241)
MAIN_SEEDS = (41,)

SMOKE_CFG = dict(per_class=300, msvrg_batch=256, msvrg_step_const=0.1,
                 msvrg_momentum=0.5, n_probes=5, mu=CAMPAIGN_MU)
FULL_CFG = dict(per_class=None, msvrg_batch=1024, msvrg_step_const=0.1,
                msvrg_momentum=0.5, n_probes=40, mu=CAMPAIGN_MU)


class _Args:
    def __init__(self, budget, eval_every, s, smoke, grid_check_res=500,
                 audit_ipopt_starts=64, audit_ccp_n0=8192, audit_ccp_r=20):
        self.budget = budget
        self.eval_every = eval_every
        self.s = s
        self.smoke = smoke
        self.grid_check_res = grid_check_res
        self.audit_ipopt_starts = audit_ipopt_starts
        self.audit_ccp_n0 = audit_ccp_n0
        self.audit_ccp_r = audit_ccp_r


def _smoke_args():
    return _Args(budget=800.0, eval_every=100.0, s=2, smoke=True,
                 grid_check_res=200)


def _ccp_cfg(seed=0):
    return CCPConfig(N0=2000, r=10, seed=seed, seed_sampler="exp",
                     adaptive_seed_schedule=False)


def _run_leg_triple_stepper(policy_name, next_lam, triple, cfg, args, out_dir,
                            extra_cfg, stepper_name="const",
                            stepper_cfg=None, sampler_seed=SAMPLER_SEED):
    """The K3 ridge executor with the walk rule delegated to stepper_core
    (four hook sites), a sampler-seed parameter and additive summary
    fields.  Everything else — instance factory, budget meter, delivery,
    safeguard, two-instrument audits, dense-grid cross-check, test
    evaluation — is the Aug-26 code path verbatim."""
    a, b, c_dg = triple
    tag = f"{a}v{b}v{c_dg}"
    mu = float(cfg.get("mu", 0.0))
    t_build = time.time()
    if mu != 0.0:
        (_obj, _grad, L, joint_oracle, stoch, meta) = make_mnist_triple_ridge(
            triple, mu, per_class=cfg["per_class"],
            batch_size=cfg["msvrg_batch"], sampler_seed=sampler_seed,
            init_seed=INIT_SEED, n_probes=cfg["n_probes"],
            probe_seed=PROBE_SEED)
    else:
        (_obj, _grad, L, joint_oracle, stoch, meta) = make_mnist_triple(
            triple, per_class=cfg["per_class"], batch_size=cfg["msvrg_batch"],
            sampler_seed=sampler_seed, init_seed=INIT_SEED,
            n_probes=cfg["n_probes"], probe_seed=PROBE_SEED)
    X_np, y_np = meta.pop("_X"), meta.pop("_y")
    n, d = meta["n"], meta["d"]
    x0 = make_triple_initial_point(INIT_SEED)
    L_arr = np.asarray(L, dtype=float)
    epoch_len = max(1, int(np.ceil(n / float(cfg["msvrg_batch"]))))
    scfg = dict(cfg)
    scfg.update(stepper_cfg or {})
    stepper = make_stepper(stepper_name, d, scfg)          # hook 0
    print(f"[{policy_name}|{tag}|{stepper_name}] instance in "
          f"{time.time() - t_build:.1f}s (n={n} d={d} "
          f"per_class={meta['per_class']} mu={mu:g} epoch_len={epoch_len} "
          f"L=[{L_arr[0]:.3f},{L_arr[1]:.3f},{L_arr[2]:.3f}] "
          f"seed={sampler_seed})", flush=True)

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
            stepper.on_lambda_change(lam, L_lam, L_scale)  # hook 1
        prev_lam = lam
        lam_history.append(lam.copy())

        retries_here = 0
        for _k in range(args.s):
            if not budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
                break
            g_a_full = chain_J.T @ lam
            F_a = float(chain_f @ lam)
            stepper.start_segment(chain_x, g_a_full, L_lam, L_scale,
                                  epoch_len)                # hook 2
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
    print(f"[{policy_name}|{tag}|{stepper_name}] budget spent: "
          f"{budget.spent():.1f} of {args.budget} | segments={len(grams) - 1} "
          f"| wall={wall:.1f}s | decision={decision_seconds:.1f}s "
          f"| L_scale={L_scale}", flush=True)

    t_a = time.time()
    audited, audit_i, audit_c, audit_lams = [], [], [], []
    lam_prev = None
    for m_ck in ck_m:
        v_i, v_c, v, lam_bst = _audit_instruments(Ms[:m_ck], args, lam_prev)
        audited.append(v); audit_i.append(v_i); audit_c.append(v_c)
        audit_lams.append([float(t) for t in lam_bst])
        lam_prev = lam_bst
    mono_viol = sum(int(audited[i + 1] > audited[i] + 1e-12)
                    for i in range(len(audited) - 1)
                    if ck_m[i + 1] >= ck_m[i])
    audit_seconds = time.time() - t_a

    t_g = time.time()
    v_grid, lam_grid = _grid_maxmin(Ms, args.grid_check_res)
    grid_seconds = time.time() - t_g
    grid_beat = bool(v_grid > audited[-1] + 1e-12)
    final_audit = float(max(audited[-1], v_grid))

    t_t = time.time()
    X_test, y_test = load_mnist_triple(triple, train=False)
    test_ce, test_err = _test_eval_stack(thetas, X_test, y_test)
    test_seconds = time.time() - t_t

    if args.smoke:
        idx = len(thetas) // 2
        f_re, _J_re = joint_oracle(thetas[idx])
        assert np.allclose(f_re, fvals[idx], atol=1e-9), "theta round-trip"
        print(f"[smoke] theta round-trip OK at index {idx}", flush=True)

    lam_arr = np.asarray(lam_history, dtype=float)
    summary = {
        "protocol": ("pure fixed budget at K=3 WITH ridge (v2 stepper "
                     "executor): walk rule delegated to stepper_core; "
                     "otherwise the Aug-26 ridge executor verbatim; "
                     "audit_v2 two-instrument meter + final dense-grid "
                     "cross-check (squared scale; norm-scale history "
                     "added)"),
        "policy": policy_name, "triple": [int(a), int(b), int(c_dg)],
        "mu": mu, "config_instance": _json_ready(cfg),
        "budget": args.budget, "s": args.s, "eval_every": args.eval_every,
        "extra": _json_ready(extra_cfg),
        "stepper": {"name": stepper_name,
                    "cfg": _json_ready(stepper_cfg or {}),
                    "final_diag": _json_ready(stepper.diag())},
        "sampler_seed": int(sampler_seed),
        "K": K, "d": d, "n": n, "per_class": meta["per_class"],
        "L_calibrated": [float(v) for v in L_arr],
        "grad_equiv_total": float(budget.spent()),
        "joint_calls": int(budget.joint_calls),
        "wall_seconds": wall, "decision_seconds": decision_seconds,
        "segments_total": int(len(grams) - 1), "m_final": int(Ms.shape[0]),
        "n_decisions": int(len(lam_history)),
        "L_scale_final": L_scale,
        "safeguard_retries": int(safeguard_retries),
        "ck_grads": ck_grads, "ck_cpu": ck_cpu, "ck_m": ck_m,
        "final_audit": final_audit,
        "audited_gn_history": audited,
        "audited_gn_norm_history": [float(np.sqrt(max(v, 0.0)))
                                    for v in audited],
        "audit_ipopt_history": audit_i, "audit_ccp_history": audit_c,
        "audit_lam_history": audit_lams,
        "audit_mono_violations": int(mono_viol),
        "final_grid_value": float(v_grid),
        "final_grid_lam": [float(t) for t in lam_grid],
        "grid_beat_instruments": grid_beat,
        "audit_seconds": audit_seconds, "grid_seconds": grid_seconds,
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
                        fvals=np.asarray(fvals), lam_history=lam_arr,
                        seg_grads=np.asarray(seg_grads, dtype=float),
                        seg_lams=np.asarray(seg_lams, dtype=float),
                        test_ce=test_ce, test_err=test_err)
    np.savez_compressed(out_dir / "thetas.npz",
                        theta_stack=np.asarray(thetas))
    print(f"[{policy_name}|{tag}|{stepper_name}] final audit = "
          f"{final_audit:.6e} (norm {np.sqrt(final_audit):.6e}, grid "
          f"{v_grid:.6e}) -> {out_dir}", flush=True)
    return summary


def _load_or_run(out_dir, fn, force=False):
    if (out_dir / "summary.json").exists() and not force:
        print(f"[skip] {out_dir.name} (resume)", flush=True)
        return json.loads((out_dir / "summary.json").read_text())
    return fn(out_dir)


# ------------------------------------------------------------ gates ----

def gate0():
    args = _smoke_args()
    ref_dir, new_dir = GATE_HOME / "gate0_ref", GATE_HOME / "gate0_const"
    sm_ref = _run_leg_triple_ridge(
        "adaptive_ccp", _ccp_policy(K, _ccp_cfg(), []), TRIPLE,
        dict(SMOKE_CFG), args, ref_dir, {"role": "gate0 reference"})
    sm_new = _run_leg_triple_stepper(
        "adaptive_ccp", _ccp_policy(K, _ccp_cfg(), []), TRIPLE,
        dict(SMOKE_CFG), args, new_dir, {"role": "gate0 candidate"},
        stepper_name="const")
    r, n = np.load(ref_dir / "grams.npz"), np.load(new_dir / "grams.npz")
    checks = {
        "gram_stack": np.array_equal(r["gram_stack"], n["gram_stack"]),
        "fvals": np.array_equal(r["fvals"], n["fvals"]),
        "lam_history": np.array_equal(r["lam_history"], n["lam_history"]),
        "seg_grads": np.array_equal(r["seg_grads"], n["seg_grads"]),
        "theta_stack": np.array_equal(
            np.load(ref_dir / "thetas.npz")["theta_stack"],
            np.load(new_dir / "thetas.npz")["theta_stack"]),
        "grad_equiv_total": sm_ref["grad_equiv_total"] == sm_new["grad_equiv_total"],
        "L_scale_final": sm_ref["L_scale_final"] == sm_new["L_scale_final"],
        "ck_m": sm_ref["ck_m"] == sm_new["ck_m"],
        "audited_gn_history": sm_ref["audited_gn_history"]
        == sm_new["audited_gn_history"],
    }
    ok = all(checks.values())
    GATE_HOME.mkdir(parents=True, exist_ok=True)
    (GATE_HOME / "gate0_report.json").write_text(json.dumps(
        {"gate": 0, "triple": list(TRIPLE), "mu": CAMPAIGN_MU,
         "checks": checks, "pass": ok}, indent=2))
    for k, v in checks.items():
        print(f"[gate0] {k:20s} {'OK' if v else 'MISMATCH'}", flush=True)
    print(f"[gate0] {'PASS' if ok else 'FAIL'}", flush=True)


def gate1():
    args = _smoke_args()
    rows = []
    for core_tag, sname, scfg in CORES:
        out = GATE_HOME / f"gate1_{core_tag}"
        sm = _run_leg_triple_stepper(
            "adaptive_ccp", _ccp_policy(K, _ccp_cfg(), []), TRIPLE,
            dict(SMOKE_CFG), args, out, {"role": f"gate1 {core_tag}"},
            stepper_name=sname, stepper_cfg=scfg)
        npz = np.load(out / "grams.npz")
        finite = bool(np.isfinite(npz["gram_stack"]).all()
                      and np.isfinite(npz["fvals"]).all())
        h = sm["audited_gn_history"]
        rows.append({"core": core_tag, "finite": finite,
                     "L_scale_final": sm["L_scale_final"],
                     "retries": sm["safeguard_retries"],
                     "progress": bool(h[-1] <= h[0]),
                     "final_norm": float(np.sqrt(max(h[-1], 0.0)))})
    ok = all(r["finite"] and r["L_scale_final"] <= 2.0 ** 20 and r["progress"]
             for r in rows)
    (GATE_HOME / "gate1_report.json").write_text(json.dumps(
        {"gate": 1, "arms": rows, "pass": ok}, indent=2))
    for r in rows:
        print(f"[gate1] {r['core']:16s} finite={r['finite']} "
              f"L_scale={r['L_scale_final']:g} retries={r['retries']} "
              f"final_GN={r['final_norm']:.4e} progress={r['progress']}",
              flush=True)
    print(f"[gate1] {'PASS' if ok else 'FAIL'}", flush=True)


# ---------------------------------------------------------- ladders ----

def stage_ladders(force=False):
    args = _Args(budget=5_000.0, eval_every=250.0, s=5, smoke=False)
    board, picks = {}, {}
    for core_tag, sname, scfg in CORES:
        board[core_tag] = {}
        for r in LADDER_RS:
            grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, r))
            finals = []
            for seed in LADDER_SEEDS:
                out = LADDER_HOME / core_tag / f"uniform_r{r}_seed{seed}"
                sm = _load_or_run(out, lambda od, g=grid: _run_leg_triple_stepper(
                    "baseline", _baseline_policy(g), TRIPLE, dict(FULL_CFG),
                    args, od, {"r": r, "n_nodes": int(g.shape[0]),
                               "core": core_tag},
                    stepper_name=sname, stepper_cfg=scfg,
                    sampler_seed=seed), force)
                finals.append(float(np.sqrt(max(sm["final_audit"], 0.0))))
            board[core_tag][str(r)] = {"mean": float(np.mean(finals)),
                                       "per_seed": finals}
        picks[core_tag] = int(min(board[core_tag],
                                  key=lambda k: board[core_tag][k]["mean"]))
    LADDER_HOME.mkdir(parents=True, exist_ok=True)
    (LADDER_HOME / "ladders_summary.json").write_text(json.dumps(
        {"triple": list(TRIPLE), "mu": CAMPAIGN_MU, "budget": args.budget,
         "seeds": list(LADDER_SEEDS), "board": board, "r_star": picks},
        indent=2))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for core_tag in board:
        ax.plot(LADDER_RS, [board[core_tag][str(r)]["mean"] for r in LADDER_RS],
                marker="o", label=core_tag)
    ax.set_xlabel("uniform grid r"); ax.set_yscale("log")
    ax.set_ylabel("final worst GN (norm), mean of 3 seeds")
    ax.set_title("K3 ladders, {4,7,9}, ridge mu=1e-4, B=5000"); ax.legend()
    fig.tight_layout(); fig.savefig(LADDER_HOME / "ladders.png", dpi=150)
    print("[ladders] r*:", json.dumps(picks), flush=True)
    return picks


# ------------------------------------------------------------- main ----

MAIN_EXTRA_RS = [20, 30]   # Sep 4 (user): the B=5,000 ladder is too
                           # small at K=3 (66/231/496 nodes) and picked
                           # r*=10, which plateaus at B=40,000.  The
                           # uniform resolution is therefore selected AT
                           # the main budget: r in {10,20,30}, B=40,000,
                           # seed 41 (full-budget ladder).
MAIN_CORES = [c for c in CORES if c[0] == "adam_1e-3_b0.9"]
                           # Sep 4 (user): the K3 main line (full-budget
                           # ladder + S4) uses the adam core only; the
                           # adagrad runs already on disk are kept but
                           # not part of the report.


def stage_main(force=False):
    args = _Args(budget=40_000.0, eval_every=500.0, s=5, smoke=False)
    picks = json.loads((LADDER_HOME / "ladders_summary.json").read_text())["r_star"]
    legs = {}
    for core_tag, sname, scfg in MAIN_CORES:
        r_star = picks[core_tag]
        for r in sorted({r_star, *MAIN_EXTRA_RS}):
            grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, r))
            for seed in MAIN_SEEDS:
                out = MAIN_HOME / core_tag / f"uniform_r{r}_seed{seed}"
                legs[(core_tag, f"uniform_r{r}")] = _load_or_run(
                    out, lambda od, g=grid, rr=r: _run_leg_triple_stepper(
                        "baseline", _baseline_policy(g), TRIPLE,
                        dict(FULL_CFG), args, od,
                        {"r": rr, "core": core_tag,
                         "ladder_pick": rr == r_star},
                        stepper_name=sname, stepper_cfg=scfg,
                        sampler_seed=seed), force)
        for seed in MAIN_SEEDS:

            def _ccp(od, sn=sname, sc=scfg, tag=core_tag, sd=seed):
                stats: list = []
                sm = _run_leg_triple_stepper(
                    "adaptive_ccp", _ccp_policy(K, _ccp_cfg(), stats), TRIPLE,
                    dict(FULL_CFG), args, od, {"core": tag},
                    stepper_name=sn, stepper_cfg=sc, sampler_seed=sd)
                sm["ccp"] = _stats_block(stats)
                (od / "summary.json").write_text(
                    json.dumps(_json_ready(sm), indent=2), encoding="utf-8")
                return sm
            out = MAIN_HOME / core_tag / f"adaptive_ccp_seed{seed}"
            legs[(core_tag, "adaptive")] = _load_or_run(out, _ccp, force)

    # const-core reference (Exp 6 on the same triple / mu / budget)
    ref = {}
    for name in ("adaptive_s5_ccp", "baseline_r20_s5"):
        p = EXP6_HOME / name / "summary.json"
        if p.exists():
            ref[name] = json.loads(p.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"adagrad_x10": "#1f77b4", "adam_1e-3_b0.9": "#ff7f0e"}
    for (core_tag, leg), sm in legs.items():
        g = np.asarray(sm["ck_grads"], dtype=float)
        v = np.sqrt(np.maximum(np.asarray(sm["audited_gn_history"]), 0.0))
        cpu = np.asarray(sm["ck_cpu"], dtype=float)
        ls = {"adaptive": "-", "uniform_r10": "--", "uniform_r20": ":",
              "uniform_r30": "-."}.get(leg, ":")
        axes[0].plot(g, v, ls, color=colors[core_tag], lw=1.5,
                     label=f"{leg} / {core_tag}")
        axes[1].plot(cpu, v, ls, color=colors[core_tag], lw=1.5,
                     label=f"{leg} / {core_tag}")
    for name, sm in ref.items():
        g = np.asarray(sm["ck_grads"], dtype=float)
        v = np.sqrt(np.maximum(np.asarray(sm["audited_gn_history"]), 0.0))
        cpu = np.asarray(sm["ck_cpu"], dtype=float)
        ls = "-" if "adaptive" in name else "--"
        axes[0].plot(g, v, ls, color="gray", lw=1.2,
                     label=f"{name} / const (Exp 6)")
        axes[1].plot(cpu, v, ls, color="gray", lw=1.2,
                     label=f"{name} / const (Exp 6)")
    axes[0].set_xlabel("total gradient evaluations (grad_equiv)")
    axes[1].set_xlabel("CPU seconds")
    for ax in axes:
        ax.set_ylabel("best-so-far worst GN (norm)")
        ax.set_yscale("log"); ax.legend(fontsize=7)
    axes[0].set_title("K3 main: {4,7,9}, ridge mu=1e-4, B=40000, seed 41")
    fig.tight_layout()
    MAIN_HOME.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAIN_HOME / "worst_gn_curves.png", dpi=150)
    board = {f"{c}/{l}": float(np.sqrt(max(sm["final_audit"], 0.0)))
             for (c, l), sm in legs.items()}
    board.update({f"const/{n}": float(np.sqrt(max(sm["final_audit"], 0.0)))
                  for n, sm in ref.items()})
    (MAIN_HOME / "main_summary.json").write_text(json.dumps(
        {"triple": list(TRIPLE), "mu": CAMPAIGN_MU, "r_star": picks,
         "final_worst_gn_norm": board}, indent=2))
    for k, v in sorted(board.items(), key=lambda t: t[1]):
        print(f"[main] {k:32s} {v:.4e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",
                        choices=["gate0", "gate1", "ladders", "main"],
                        required=True)
    parser.add_argument("--force", action="store_true")
    a = parser.parse_args()
    {"gate0": gate0, "gate1": gate1,
     "ladders": lambda: stage_ladders(force=a.force),
     "main": lambda: stage_main(force=a.force)}[a.stage]()


if __name__ == "__main__":
    main()
