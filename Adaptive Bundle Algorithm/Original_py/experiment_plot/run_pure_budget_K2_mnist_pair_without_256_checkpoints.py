"""run_pure_budget_K2_mnist_pair_without_256_checkpoints.py — the K = 2
MNIST digit-pair campaign: pure fixed budget, adaptive(CCP) vs multi-r
grid baselines, with theta snapshots and OFFICIAL-TEST evaluation.

NEW FILE (Aug 13, 2026).  Design record: ``Note/Aug_13_note.md``
(user-approved Aug-13 plan; conflict smoke chose the pairs).  No
existing file is modified — executor building blocks are imported:

* protocol pieces (``_Budget``, ``MAX_SAFEGUARD_RETRIES``,
  ``_baseline_policy``) from the K6 pure-budget runner;
* the EXACT 1-D meter ``exact_gn_1d`` (+ front helpers) from the K2
  planted runner — audits at every checkpoint, monotone by
  mathematics, with certified upper bounds;
* the CCP policy (``_ccp_policy``, ``_stats_block``) from the K2 CCP
  runner;
* the problem family from ``objectives_mnist_pair`` (per-class mean
  CE on a 2-logit patch-softplus net, d = 8,098; per_class = balanced
  maximum; batch 1024; NO regularisation).

What is NEW relative to every earlier runner (both Aug-13 additions):

1. **Theta snapshots** — the parameter vector of EVERY delivered
   point (x0 + each segment endpoint, aligned index-for-index with
   ``fvals``) is kept and written to ``thetas.npz``.  Pure recording:
   no extra oracle work, nothing on either cost axis (in-memory
   append during the run; file written after the wall clock stops).
2. **Test-side evaluation** — after the leg (off both axes), every
   delivered theta is re-scored on ALL official t10k rows of the two
   digits: per-class mean CE and per-class error rate (1 - recall),
   stored as ``test_ce`` / ``test_err`` in ``grams.npz``.  Front
   figures and test-vs-budget curves are drawn by the companion
   ``plot_K2_mnist_pair_without_256_checkpoints.py``.

Legs per pair (user go, Aug 13): baseline r in {10, 20, 40} at s = 5
+ ``adaptive_s5_ccp`` (production CCP: N0=2000, r=10, exp sampler,
pool on, adaptive schedule off).  B = 20,000 grad-equivalents each,
eval_every = 250, audit grid 200,001.  NO IPOPT leg.

Usage:
    python run_pure_budget_K2_mnist_pair_without_256_checkpoints.py            # both pairs, all legs
    python run_pure_budget_K2_mnist_pair_without_256_checkpoints.py --pair 3 5 # one pair
    python run_pure_budget_K2_mnist_pair_without_256_checkpoints.py --smoke    # Smoke B
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

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
from run_pure_budget_K2_without_256_checkpoints import (  # noqa: E402
    exact_gn_1d,
)
from run_pure_budget_K2_ccp_without_256_checkpoints import (  # noqa: E402
    _ccp_policy,
    _stats_block,
)
from run_experiments import _json_ready  # noqa: E402
from ccp_lambda_solver import CCPConfig  # noqa: E402
from objectives_torch import _load_theta_into_net  # noqa: E402
from objectives_mnist_pair import (  # noqa: E402
    PairPatchMLP,
    load_mnist_pair,
    make_mnist_pair,
    make_pair_initial_point,
)

HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = (HERE.parent.parent / "output"
                 / "CCP/K2_mnist_pair_without_256_checkpoints")
PAIRS_DEFAULT = [(3, 5), (7, 9)]
BASELINE_RS = [10, 20, 40]
INIT_SEED, SAMPLER_SEED, PROBE_SEED = 8, 41, 7


def _test_eval_stack(thetas, X_test, y_test):
    """Per-class mean CE and error rate for every theta (no gradients).

    One net, reloaded per theta — the whole stack is a pure forward
    sweep over the official test rows of the two digits."""
    import torch.nn.functional as F
    net = PairPatchMLP()
    X = torch.from_numpy(np.ascontiguousarray(X_test))
    rows = [torch.from_numpy(np.nonzero(y_test == k)[0]).long()
            for k in (0, 1)]
    targets = [torch.full((len(r),), k, dtype=torch.long)
               for k, r in enumerate(rows)]
    ce = np.empty((len(thetas), 2))
    err = np.empty((len(thetas), 2))
    with torch.no_grad():
        for i, th in enumerate(thetas):
            _load_theta_into_net(net, np.asarray(th, dtype=float))
            Z = net(X)
            for k in (0, 1):
                Zk = Z[rows[k]]
                ce[i, k] = float(F.cross_entropy(Zk, targets[k],
                                                 reduction="mean"))
                err[i, k] = float((Zk.argmax(dim=1) != targets[k])
                                  .to(torch.float64).mean())
    return ce, err


def _run_leg_pair(policy_name, next_lam, pair, cfg, args, out_dir,
                  extra_cfg):
    """Shared pure-budget executor (K2 replica) + theta snapshots +
    official-test evaluation.  ONLY the next-lambda policy differs
    between legs."""
    a, b = pair
    t_build = time.time()
    (_obj, _grad, L, joint_oracle, stoch, meta) = make_mnist_pair(
        a, b, per_class=cfg["per_class"], batch_size=cfg["msvrg_batch"],
        sampler_seed=SAMPLER_SEED, init_seed=INIT_SEED,
        n_probes=cfg["n_probes"], probe_seed=PROBE_SEED)
    X_np, y_np = meta.pop("_X"), meta.pop("_y")
    K, n, d = meta["K"], meta["n"], meta["d"]
    x0 = make_pair_initial_point(INIT_SEED)
    L_arr = np.asarray(L, dtype=float)
    epoch_len = max(1, int(np.ceil(n / float(cfg["msvrg_batch"]))))
    print(f"[{policy_name}|{a}v{b}] instance in {time.time() - t_build:.1f}s "
          f"(n={n} d={d} per_class={meta['per_class']} "
          f"epoch_len={epoch_len} L=[{L_arr[0]:.3f},{L_arr[1]:.3f}])",
          flush=True)

    f0, J0 = validate_oracle_output(*joint_oracle(x0), K, d)
    grams = [J0 @ J0.T]
    fvals = [np.asarray(f0, dtype=float)]
    thetas = [x0.copy()]                      # Aug-13: snapshot stack
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
            thetas.append(y.copy())           # Aug-13: snapshot
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
    print(f"[{policy_name}|{a}v{b}] budget spent: {budget.spent():.1f} of "
          f"{args.budget} | segments={len(grams) - 1} | wall={wall:.1f}s "
          f"| decision={decision_seconds:.1f}s | L_scale={L_scale}",
          flush=True)

    # ---- post-hoc, off both axes: EXACT prefix audits (K2 meter) ----
    t_a = time.time()
    audited, audit_ws, audit_ub = [], [], []
    for m_ck in ck_m:
        v, w, ub = exact_gn_1d(Ms[:m_ck], grid_points=args.audit_grid,
                               certify=True)
        assert ub >= v - 1e-15
        audited.append(float(v))
        audit_ws.append(float(w))
        audit_ub.append(float(ub))
    for i in range(1, len(audited)):
        if ck_m[i] >= ck_m[i - 1]:
            assert audited[i] <= audited[i - 1] + 1e-12, (
                f"exact prefix audit not monotone at ck {i}")
    audit_seconds = time.time() - t_a

    # ---- post-hoc, off both axes: official-test evaluation ----
    t_t = time.time()
    X_test, y_test = load_mnist_pair(a, b, train=False)
    test_ce, test_err = _test_eval_stack(thetas, X_test, y_test)
    test_seconds = time.time() - t_t

    if args.smoke:
        # Smoke B round-trip: a stored theta must reproduce its
        # recorded full-batch objective vector.
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
        "protocol": ("pure fixed budget at K=2 (MNIST digit pair, patch-"
                     "softplus): shared segment unit, shared s, chain warm "
                     "start; ONLY the next-lambda policy differs; no "
                     "tolerance anywhere; stop = budget; EXACT 1-D meter "
                     "for audits; theta snapshots + official-test "
                     "evaluation are Aug-13 additions, both off-axis"),
        "policy": policy_name,
        "pair": [int(a), int(b)],
        "config_instance": _json_ready(cfg),
        "budget": args.budget, "s": args.s, "eval_every": args.eval_every,
        "audit_grid": args.audit_grid,
        "meter": "exact-1d (audits); test split = ALL official t10k rows",
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
        "final_audit": float(audited[-1]),
        "final_audit_upper": float(audit_ub[-1]),
        "audited_gn_history": audited,
        "audited_gn_upper_history": audit_ub,
        "audit_w_history": audit_ws,
        "w_star": float(audit_ws[-1]),
        "audit_seconds": audit_seconds,
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
    print(f"[{policy_name}|{a}v{b}] final EXACT audit = {audited[-1]:.6e} "
          f"| final test CE = ({test_ce[-1][0]:.4f}, {test_ce[-1][1]:.4f}) "
          f"err = ({test_err[-1][0]:.4f}, {test_err[-1][1]:.4f}) "
          f"-> {out_dir}", flush=True)
    return summary


def _seg_cost_check(out_dir, n, epoch_len, batch):
    """Smoke B: every segment's grad-equivalent cost must equal the
    exact formula for one of the three support patterns (full, only
    class A, only class B)."""
    npz = np.load(out_dir / "grams.npz")
    seg = np.diff(np.asarray(npz["seg_grads"], dtype=float))
    b_k = batch // 2
    allowed = {round(epoch_len * 2.0 * rows * 2.0 / n + 2.0, 12)
               for rows in (batch, b_k)}
    bad = [c for c in seg if all(abs(c - v) > 1e-9 for v in allowed)]
    assert not bad, f"segment costs off-formula: {bad[:3]} vs {allowed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=20_000.0)
    parser.add_argument("--eval-every", type=float, default=250.0)
    parser.add_argument("--audit-grid", type=int, default=200_001)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--pair", type=int, nargs=2, default=None)
    parser.add_argument("--ccp-seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.budget, args.eval_every = 800.0, 100.0
        args.audit_grid, args.s = 20_001, 2
        cfg = dict(per_class=300, msvrg_batch=256, msvrg_step_const=0.1,
                   msvrg_momentum=0.5, n_probes=5)
        pairs = [(3, 5)]
        rs = [4]
        home_root = CAMPAIGN_ROOT / "SMOKE"
    else:
        cfg = dict(per_class=None, msvrg_batch=1024, msvrg_step_const=0.1,
                   msvrg_momentum=0.5, n_probes=40)
        pairs = [tuple(args.pair)] if args.pair else list(PAIRS_DEFAULT)
        rs = list(BASELINE_RS)
        home_root = CAMPAIGN_ROOT

    ccp_cfg = CCPConfig(N0=2000, r=10, seed=args.ccp_seed,
                        seed_sampler="exp", adaptive_seed_schedule=False)
    t_all = time.time()
    for pair in pairs:
        a, b = pair
        home = home_root / f"pair_{a}v{b}_B{args.budget:.0f}"
        home.mkdir(parents=True, exist_ok=True)
        manifest_path = home / "campaign_manifest.json"
        manifest = {"campaign": f"K2 MNIST pair {a}v{b} pure budget "
                                f"(baseline r{rs} + adaptive CCP)",
                    "machine": platform.platform(), "smoke": args.smoke,
                    "legs": []}

        def _do(name, fn):
            out_dir = home / name
            if (out_dir / "summary.json").exists() and not args.force:
                print(f"[campaign] skip {name} (summary exists)", flush=True)
                return
            print(f"[campaign] === {a}v{b} / {name} ===", flush=True)
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
            grid = _sort_grid_for_warmstart(_uniform_simplex_grid(2, r))
            _do(f"baseline_r{r:02d}_s{args.s}",
                lambda od, g=grid, rr=r: _run_leg_pair(
                    "baseline", _baseline_policy(g), pair, cfg, args, od,
                    {"r": rr}))

        def _ccp_leg(od):
            stats: list = []
            sm = _run_leg_pair("adaptive_ccp",
                               _ccp_policy(2, ccp_cfg, stats), pair, cfg,
                               args, od, {"ccp_config": vars(ccp_cfg)})
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
        home = home_root / f"pair_3v5_B{args.budget:.0f}"
        n_smoke = 600
        epoch_len = int(np.ceil(n_smoke / cfg["msvrg_batch"]))
        for p in home.glob("*/summary.json"):
            sm = json.loads(p.read_text())
            assert sm["grad_equiv_total"] <= args.budget + 1e-6
            hist = sm["audited_gn_history"]
            assert all(hist[i + 1] <= hist[i] + 1e-12
                       for i in range(len(hist) - 1)), "audit not monotone"
            assert all(u >= v - 1e-15 for u, v in
                       zip(sm["audited_gn_upper_history"], hist))
            npz = np.load(p.parent / "grams.npz")
            m = npz["fvals"].shape[0]
            assert npz["test_ce"].shape == (m, 2)
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
        # resume check: a second _do must skip every leg
        print("[smoke] all leg checks OK; verifying figures render...",
              flush=True)
        import plot_K2_mnist_pair_without_256_checkpoints as plotter
        plotter.make_figures(home)
        for fname in ("gn_vs_grads.png", "gn_vs_cpu.png",
                      "front_train.png", "front_test.png",
                      "front_err_test.png",
                      "test_ce_vs_budget.png", "front_metrics.json"):
            assert (home / fname).exists(), f"missing {fname}"
        print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
