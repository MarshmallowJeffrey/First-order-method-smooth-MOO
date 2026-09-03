"""run_pure_budget_K3_mnist_triple_ridge_without_256_checkpoints.py —
the K = 3 MNIST digit-triple campaign WITH the ridge penalty
(mu/2)*||theta||^2 on every objective: pure fixed budget,
adaptive(CCP) vs simplex-grid baselines, theta snapshots and
OFFICIAL-TEST evaluation.

NEW FILE (Aug 26, 2026).  User-approved (Aug-26 Q&A): add a penalty
term to the formal K = 3 experiment; mu = 1e-4 fixed by the user's
direct call ("拍脑袋" ruling — scale analysis: ~1.6/n, penalty at
init ~0.016 = 1.5% of the ln 3 window, taming gradient mu*||theta||
reaches 1e-3 scale within the budget).  Per the user's explicit
instruction NO existing file is modified: this runner is a replica of
``run_pure_budget_K3_mnist_triple_without_256_checkpoints`` whose
audit/test/cost-check helpers are IMPORTED from that module; only the
executor and main are restated here, with exactly these deltas:

* problem family = ``make_mnist_triple_ridge(triple, mu, ...)``
  (see ``objectives_mnist_triple_ridge`` for the form, motivation and
  the MSVRG/ifo invariances of the penalty);
* ``--mu`` argument (default 1e-4, the approved campaign value);
  mu = 0.0 is allowed ONLY as the replica-fidelity gate — it must
  reproduce the base runner's stored smoke numbers bit-identically;
* every output home gains the suffix ``_mu<mu>`` (also for mu = 0,
  as ``_mu0``) so no base-campaign record can ever be touched;
* summary/manifest record ``mu``.

Everything else — protocol, meter (audit_v2 two-instrument +
final dense-grid cross-check), legs r in {10, 20, 30} + adaptive CCP,
B = 40,000, s = 5, eval_every = 500, seeds, snapshots, official-test
evaluation — is the Aug-26 campaign verbatim.  REPORTING AXES: train
fvals / Gram stacks / GN* audits are in PENALISED coordinates (the
problem being solved); the test evaluation stays RAW CE; raw train
values are recoverable at plot time from thetas.npz.

Usage:
    python run_pure_budget_K3_mnist_triple_ridge_without_256_checkpoints.py                 # mu=1e-4 campaign
    python run_pure_budget_K3_mnist_triple_ridge_without_256_checkpoints.py --smoke         # Smoke B at mu=1e-4
    python run_pure_budget_K3_mnist_triple_ridge_without_256_checkpoints.py --smoke --mu 0  # replica-fidelity gate
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402, F401  (import order: keep torch's libomp first)

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
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
from run_experiments import _json_ready  # noqa: E402
from algorithm_fast_without_256_checkpoints import ipopt_available  # noqa: E402
from ccp_lambda_solver import CCPConfig  # noqa: E402
from run_pure_budget_K3_mnist_triple_without_256_checkpoints import (  # noqa: E402
    BASELINE_RS,
    CAMPAIGN_ROOT,
    INIT_SEED,
    K,
    PROBE_SEED,
    SAMPLER_SEED,
    TRIPLE_DEFAULT,
    _audit_instruments,
    _grid_maxmin,
    _seg_cost_check,
    _test_eval_stack,
)
from objectives_mnist_triple import (  # noqa: E402
    load_mnist_triple,
    make_triple_initial_point,
)
from objectives_mnist_triple_ridge import make_mnist_triple_ridge  # noqa: E402


def _run_leg_triple_ridge(policy_name, next_lam, triple, cfg, args, out_dir,
                          extra_cfg):
    """Shared pure-budget executor — verbatim replica of the base
    runner's ``_run_leg_triple`` except: penalised problem family,
    mu in the instance print and in summary.  ONLY the next-lambda
    policy differs between legs."""
    a, b, c_dg = triple
    tag = f"{a}v{b}v{c_dg}"
    t_build = time.time()
    (_obj, _grad, L, joint_oracle, stoch, meta) = make_mnist_triple_ridge(
        triple, cfg["mu"], per_class=cfg["per_class"],
        batch_size=cfg["msvrg_batch"], sampler_seed=SAMPLER_SEED,
        init_seed=INIT_SEED, n_probes=cfg["n_probes"],
        probe_seed=PROBE_SEED)
    X_np, y_np = meta.pop("_X"), meta.pop("_y")
    n, d = meta["n"], meta["d"]
    x0 = make_triple_initial_point(INIT_SEED)
    L_arr = np.asarray(L, dtype=float)
    epoch_len = max(1, int(np.ceil(n / float(cfg["msvrg_batch"]))))
    print(f"[{policy_name}|{tag}] instance in {time.time() - t_build:.1f}s "
          f"(n={n} d={d} per_class={meta['per_class']} mu={cfg['mu']:g} "
          f"epoch_len={epoch_len} "
          f"L=[{L_arr[0]:.3f},{L_arr[1]:.3f},{L_arr[2]:.3f}])", flush=True)

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
            thetas.append(y.copy())
            seg_grads.append(float(budget.spent()))
            seg_lams.append([float(t) for t in lam])
            if float(f_y @ lam) > F_a + 1e-10 * (1.0 + abs(F_a)):
                L_scale *= 2.0
                safeguard_retries += 1
                retries_here += 1
                if retries_here > MAX_SAFEGUARD_RETRIES:
                    chain_x, chain_J, chain_f = y, J_y, f_y
                    retries_here = 0
            else:
                chain_x, chain_J, chain_f = y, J_y, f_y
                retries_here = 0

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
    print(f"[{policy_name}|{tag}] budget spent: {budget.spent():.1f} of "
          f"{args.budget} | segments={len(grams) - 1} | wall={wall:.1f}s "
          f"| decision={decision_seconds:.1f}s | L_scale={L_scale}",
          flush=True)

    # ---- post-hoc, off both axes: two-instrument prefix audits ----
    t_a = time.time()
    audited, audit_i, audit_c, audit_lams = [], [], [], []
    lam_prev = None
    for m_ck in ck_m:
        v_i, v_c, v, lam_bst = _audit_instruments(Ms[:m_ck], args, lam_prev)
        audited.append(v)
        audit_i.append(v_i)
        audit_c.append(v_c)
        audit_lams.append([float(t) for t in lam_bst])
        lam_prev = lam_bst
    mono_viol = sum(int(audited[i + 1] > audited[i] + 1e-12)
                    for i in range(len(audited) - 1)
                    if ck_m[i + 1] >= ck_m[i])
    if mono_viol:
        warnings.warn(f"audit history has {mono_viol} non-monotone "
                      f"steps (instrument-miss diagnostic; recorded, "
                      f"not clipped).")
    audit_seconds = time.time() - t_a

    # ---- final-stack dense-grid cross-check (independent lower bound)
    t_g = time.time()
    v_grid, lam_grid = _grid_maxmin(Ms, args.grid_check_res)
    grid_seconds = time.time() - t_g
    grid_beat_instruments = bool(v_grid > audited[-1] + 1e-12)
    final_audit = float(max(audited[-1], v_grid))
    lam_star = (list(map(float, lam_grid)) if grid_beat_instruments
                else audit_lams[-1])
    if grid_beat_instruments:
        warnings.warn(f"dense grid (res {args.grid_check_res}) beat the "
                      f"instruments at the final stack: {v_grid:.6e} > "
                      f"{audited[-1]:.6e} — raise audit multistarts.")

    # ---- post-hoc, off both axes: official-test evaluation (RAW CE) ----
    t_t = time.time()
    X_test, y_test = load_mnist_triple(triple, train=False)
    test_ce, test_err = _test_eval_stack(thetas, X_test, y_test)
    test_seconds = time.time() - t_t

    if args.smoke:
        # Smoke B round-trip: a stored theta must reproduce its
        # recorded (penalised) full-batch objective vector.
        idx = len(thetas) // 2
        f_re, _J_re = joint_oracle(thetas[idx])
        assert np.allclose(f_re, fvals[idx], atol=1e-9), (
            f"theta round-trip failed at {idx}: {f_re} vs {fvals[idx]}")
        print(f"[smoke] theta round-trip OK at index {idx}", flush=True)

    lam_arr = np.asarray(lam_history, dtype=float)
    distinct = (np.unique(lam_arr.round(12), axis=0).shape[0]
                if lam_arr.size else 0)
    n_test = [int((y_test == k).sum()) for k in range(K)]
    summary = {
        "protocol": ("pure fixed budget at K=3 (MNIST digit triple, patch-"
                     "softplus) WITH ridge penalty (mu/2)*||theta||^2 on "
                     "every objective (train/audit axes penalised, test "
                     "axis raw CE): shared segment unit, shared s, chain "
                     "warm start; ONLY the next-lambda policy differs; no "
                     "tolerance anywhere; stop = budget; audit_v2 two-"
                     "instrument meter per checkpoint + final dense-grid "
                     "cross-check; theta snapshots + official-test "
                     "evaluation, all off-axis"),
        "policy": policy_name,
        "triple": [int(a), int(b), int(c_dg)],
        "mu": float(cfg["mu"]),
        "config_instance": _json_ready(cfg),
        "budget": args.budget, "s": args.s, "eval_every": args.eval_every,
        "meter": ("audit_v2 two-instrument ("
                  f"{'IPOPT' if ipopt_available() else 'SLSQP-fallback'}"
                  f" strict-{args.audit_ipopt_starts} + CCP N0="
                  f"{args.audit_ccp_n0}/r={args.audit_ccp_r}) per "
                  f"checkpoint; final grid res {args.grid_check_res}; "
                  "test split = ALL official t10k rows"),
        "audit_nlp_backend": ("ipopt" if ipopt_available()
                              else "slsqp-fallback"),
        "extra": _json_ready(extra_cfg),
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
        "final_audit": final_audit,
        "audited_gn_history": audited,
        "audit_ipopt_history": audit_i,
        "audit_ccp_history": audit_c,
        "audit_lam_history": audit_lams,
        "audit_mono_violations": int(mono_viol),
        "final_grid_value": float(v_grid),
        "final_grid_lam": [float(t) for t in lam_grid],
        "grid_beat_instruments": grid_beat_instruments,
        "lam_star": lam_star,
        "audit_seconds": audit_seconds,
        "grid_seconds": grid_seconds,
        "test_seconds": test_seconds,
        "final_test_ce": [float(v) for v in test_ce[-1]],
        "final_test_err": [float(v) for v in test_err[-1]],
        "init_seed": INIT_SEED, "sampler_seed": SAMPLER_SEED,
        "probe_seed": PROBE_SEED,
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
    print(f"[{policy_name}|{tag}] final audit = {final_audit:.6e} "
          f"(grid {v_grid:.6e}, mono_viol={mono_viol}) | final test CE = "
          f"({test_ce[-1][0]:.4f}, {test_ce[-1][1]:.4f}, "
          f"{test_ce[-1][2]:.4f}) -> {out_dir}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=40_000.0)
    parser.add_argument("--eval-every", type=float, default=500.0)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--triple", type=int, nargs=3, default=None)
    parser.add_argument("--ccp-seed", type=int, default=0)
    parser.add_argument("--audit-ipopt-starts", type=int, default=64)
    parser.add_argument("--audit-ccp-n0", type=int, default=8192)
    parser.add_argument("--audit-ccp-r", type=int, default=20)
    parser.add_argument("--grid-check-res", type=int, default=500)
    parser.add_argument("--mu", type=float, default=1e-4,
                        help="ridge coefficient (campaign value 1e-4, "
                             "user-fixed Aug 26); 0.0 only as the "
                             "replica-fidelity gate vs the base runner")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.budget, args.eval_every = 800.0, 100.0
        args.s = 2
        args.grid_check_res = 200
        cfg = dict(per_class=300, msvrg_batch=256, msvrg_step_const=0.1,
                   msvrg_momentum=0.5, n_probes=5, mu=args.mu)
        triples = [TRIPLE_DEFAULT]
        rs = [4]
        home_root = CAMPAIGN_ROOT / "SMOKE"
    else:
        cfg = dict(per_class=None, msvrg_batch=1024, msvrg_step_const=0.1,
                   msvrg_momentum=0.5, n_probes=40, mu=args.mu)
        triples = ([tuple(args.triple)] if args.triple
                   else [TRIPLE_DEFAULT])
        rs = list(BASELINE_RS)
        home_root = CAMPAIGN_ROOT
    mu_tag = f"_mu{args.mu:g}"     # ALWAYS suffixed (also _mu0): the base
    #                                campaign homes can never be touched.

    ccp_cfg = CCPConfig(N0=2000, r=10, seed=args.ccp_seed,
                        seed_sampler="exp", adaptive_seed_schedule=False)
    t_all = time.time()
    for triple in triples:
        a, b, c_dg = triple
        tag = f"{a}v{b}v{c_dg}"
        home = home_root / f"triple_{tag}_B{args.budget:.0f}{mu_tag}"
        home.mkdir(parents=True, exist_ok=True)
        manifest_path = home / "campaign_manifest.json"
        manifest = {"campaign": f"K3 MNIST triple {tag} pure budget, ridge "
                                f"mu={args.mu:g} "
                                f"(baseline r{rs} + adaptive CCP)",
                    "mu": args.mu,
                    "machine": platform.platform(), "smoke": args.smoke,
                    "legs": []}

        def _do(name, fn):
            out_dir = home / name
            if (out_dir / "summary.json").exists() and not args.force:
                print(f"[campaign] skip {name} (summary exists)", flush=True)
                return
            print(f"[campaign] === {tag} / {name} ===", flush=True)
            t0 = time.time()
            sm = fn(out_dir)
            manifest["legs"].append(
                {"leg": name, "wall_seconds": time.time() - t0,
                 "decision_seconds": sm.get("decision_seconds"),
                 "final_audit": sm.get("final_audit"),
                 "final_test_err": sm.get("final_test_err")})
            manifest_path.write_text(
                json.dumps(_json_ready(manifest), indent=2),
                encoding="utf-8")

        for r in rs:
            grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, r))
            _do(f"baseline_r{r:02d}_s{args.s}",
                lambda od, g=grid, rr=r: _run_leg_triple_ridge(
                    "baseline", _baseline_policy(g), triple, cfg, args, od,
                    {"r": rr, "n_nodes": int(g.shape[0])}))

        def _ccp_leg(od):
            stats: list = []
            sm = _run_leg_triple_ridge("adaptive_ccp",
                                       _ccp_policy(K, ccp_cfg, stats),
                                       triple, cfg, args, od,
                                       {"ccp_config": vars(ccp_cfg)})
            sm["ccp"] = _stats_block(stats)
            (od / "summary.json").write_text(
                json.dumps(_json_ready(sm), indent=2), encoding="utf-8")
            return sm

        _do(f"adaptive_s{args.s}_ccp", _ccp_leg)
        manifest["total_wall_seconds"] = time.time() - t_all
        manifest_path.write_text(
            json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")

    print(f"[campaign] ALL DONE in {time.time() - t_all:.0f}s", flush=True)

    if args.smoke:
        a, b, c_dg = TRIPLE_DEFAULT
        home = home_root / f"triple_{a}v{b}v{c_dg}_B{args.budget:.0f}{mu_tag}"
        n_smoke = 300 * K
        epoch_len = int(np.ceil(n_smoke / cfg["msvrg_batch"]))
        for p in home.glob("*/summary.json"):
            sm = json.loads(p.read_text())
            assert sm["grad_equiv_total"] <= args.budget + 1e-6
            hist = sm["audited_gn_history"]
            assert all(np.isfinite(hist)) and all(v >= 0 for v in hist)
            assert sm["final_grid_value"] <= sm["final_audit"] + 1e-9
            assert all(abs(max(i_v, c_v) - v) < 1e-12
                       for i_v, c_v, v in zip(sm["audit_ipopt_history"],
                                              sm["audit_ccp_history"],
                                              hist))
            npz = np.load(p.parent / "grams.npz")
            m = npz["fvals"].shape[0]
            assert npz["test_ce"].shape == (m, K)
            assert np.isfinite(npz["test_ce"]).all()
            assert np.isfinite(npz["test_err"]).all()
            assert (npz["test_err"] >= 0).all() and (npz["test_err"] <= 1).all()
            th = np.load(p.parent / "thetas.npz")["theta_stack"]
            assert th.shape == (m, sm["d"])
            lams = np.asarray(npz["lam_history"], dtype=float)
            assert np.all(lams >= -1e-12)
            assert np.allclose(lams.sum(axis=1), 1.0, atol=1e-9)
            _seg_cost_check(p.parent, n_smoke, epoch_len,
                            cfg["msvrg_batch"])
        print("[smoke] all leg checks OK; verifying figures render...",
              flush=True)
        import plot_K3_mnist_triple_without_256_checkpoints as plotter
        plotter.make_figures(home)
        for fname in ("gn_vs_grads.png", "gn_vs_cpu.png",
                      "front_train.png", "front_test.png",
                      "front_err_test.png",
                      "test_ce_vs_budget.png", "front_metrics.json"):
            assert (home / fname).exists(), f"missing {fname}"
        print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
