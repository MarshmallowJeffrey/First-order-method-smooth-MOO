"""run_dots_K10_without_256_checkpoints.py — K = 10 MNIST all-digits
fixed-budget campaign: adaptive λ-bundle (CCP) versus uniform simplex grids
of resolution r, one run each, seed 41 (design of Sep 10, 2026; handbook
``K10_EXPERIMENT_HANDBOOK.md``).

NEW FILE.  Nothing existing is modified.  The executor is the K = 3
stepper executor (``run_k3_stepper_campaign_without_256_checkpoints``)
carried to K = 10 with these declared differences:

* problem factory ``objectives_mnist_patch_k10.make_mnist_patch_k10``
  (all ten digits, 5,421 rows each, ridge mu = 1e-4, per-class-forward
  joint oracle, torch device selectable — training may run on a GPU);
* the CCP decisions use ``CCPLambdaSolverBulk`` (block LP transport;
  same algorithm, gate ``sanity_checks_ccp_bulk.py``);
* the worst-GN audit is a SEPARATE STAGE run after training (its CPU
  load must not land on the wall-clock axis of a later run) and uses
  the CCP instrument only: every checkpoint N0 = 8,192 / r = 20; the
  end point N0 = 32,768 / r = 20 with two sampler seeds plus an exact
  evaluation of 100,000 uniform random lambdas (no local search); the
  reported final value is the maximum of the three.  No IPOPT, no dense
  simplex grid (impossible at K = 10);
* no test-set evaluation (theta stacks are saved for a later one);
* tables: Table 1 (final worst GN + ratio, nodes, passes) and Table 2
  (Monte-Carlo hypervolume of the training front, reference (ln 10)^10,
  plus the fraction of delivered points inside the reference box).

Stages (``--stage``):
    train     the legs, serial, resume-skip on an existing summary.json
    audit     worst-GN instruments on every leg's grams.npz -> audit.json
              (merged into summary.json so the existing plotters work)
    tables    table1_K10.{md,json}, table2_K10.{md,json}
    figures   the two paper figures via the existing plot scripts
    all       train -> audit -> tables -> figures   (default)

Home: output/CCP/K10_mnist_without_256_checkpoints/mu<mu>/dots_B<B>/adam_1e-3_b0.9/
      legs adaptive_ccp_seed<seed>/, uniform_r<r>_seed<seed>/
      (``--smoke``: output/CCP/K10_mnist_without_256_checkpoints/SMOKE/)

Usage:
    python run_dots_K10_without_256_checkpoints.py --smoke                  # wiring test, CPU, ~1 min
    python run_dots_K10_without_256_checkpoints.py --stage train --device cuda
    python run_dots_K10_without_256_checkpoints.py --stage audit
    python run_dots_K10_without_256_checkpoints.py --stage tables
    python run_dots_K10_without_256_checkpoints.py --stage figures
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _layout  # noqa: F401
from bundle import validate_oracle_output  # noqa: E402
from baseline_svrg_certified_without_256_checkpoints import _support_batch  # noqa: E402
from baseline_without_256_checkpoints import (  # noqa: E402
    _sort_grid_for_warmstart,
    _uniform_simplex_grid,
)
from run_pure_budget_K6_without_256_checkpoints import (  # noqa: E402
    MAX_SAFEGUARD_RETRIES,
    _baseline_policy,
    _Budget,
)
from run_pure_budget_K2_ccp_without_256_checkpoints import _stats_block  # noqa: E402
from run_experiments import _json_ready  # noqa: E402
from ccp_lambda_solver import CCPConfig, phi_batch, sample_simplex_exp  # noqa: E402
from ccp_lambda_solver_bulk import CCPLambdaSolverBulk  # noqa: E402
from objectives_mnist_patch_k10 import (  # noqa: E402
    BALANCED_MAX_PER_CLASS,
    device_description,
    make_mnist_patch_k10,
    make_patch_initial_point,
    resolve_device,
)
from stepper_core import make_stepper  # noqa: E402

K = 10
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
K10_ROOT = HERE.parent.parent / "output" / "CCP" / "K10_mnist_without_256_checkpoints"
CORE_TAG = "adam_1e-3_b0.9"
STEPPER_NAME, STEPPER_CFG = "adam", {"adam_alpha": 1e-3, "adam_beta2": 0.9}
INIT_SEED, PROBE_SEED, DEFAULT_SAMPLER_SEED = 8, 7, 41
DEFAULT_RS = [3, 4, 5, 6, 7]
FULL_CFG = dict(per_class=BALANCED_MAX_PER_CLASS, msvrg_batch=1024,
                msvrg_step_const=0.1, msvrg_momentum=0.5, n_probes=40, mu=1e-4)
SMOKE_CFG = dict(per_class=300, msvrg_batch=256, msvrg_step_const=0.1,
                 msvrg_momentum=0.5, n_probes=5, mu=1e-4)
LN_K = float(np.log(K))


# ---------------------------------------------------------------- misc --
def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def _machine(dev):
    return {"platform": platform.platform(), "python": sys.version.split()[0],
            "torch": torch.__version__, "device": device_description(dev),
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "git_commit": _git_commit(), "hostname": platform.node()}


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _ccp_cfg(seed=0):
    return CCPConfig(N0=2000, r=10, seed=seed, seed_sampler="exp",
                     adaptive_seed_schedule=False)


def _ccp_policy_bulk(config, stats_out):
    """next_lam closure on the block-transport solver (same algorithm)."""
    solver = CCPLambdaSolverBulk(K, config)

    def nxt(grams, fvals, prev_lam):
        _val, lam = solver.solve(np.asarray(grams, dtype=float))
        stats_out.append(solver.stats_last)
        return lam
    return nxt


def _grid_cost_table(grid, epoch_len, b_k, n):
    """Budget units per segment for every grid node (support effect)."""
    rows = np.array([b_k[l > 0].sum() for l in grid], dtype=float)
    return K + epoch_len * 2.0 * rows * K / float(n)


def _home(a):
    if a.smoke:
        return K10_ROOT / "SMOKE"
    return K10_ROOT / f"mu{a.mu:g}" / f"dots_B{int(a.budget)}" / CORE_TAG


# ------------------------------------------------------------ executor --
def _run_leg_k10(policy_name, next_lam, cfg, args, out_dir, extra_cfg, dev,
                 sampler_seed):
    """Pure fixed-budget executor (K = 3 stepper executor carried to K = 10).

    Shared segment unit, shared s, chain warm start; ONLY ``next_lam``
    differs between legs; decision time is inside the wall clock (it is
    part of the method); no audit here (see ``_audit_leg``).
    """
    t_build = time.time()
    (_obj, _grad, L, joint_oracle, stoch, meta) = make_mnist_patch_k10(
        per_class=cfg["per_class"], mu=cfg["mu"], batch_size=cfg["msvrg_batch"],
        sampler_seed=sampler_seed, init_seed=INIT_SEED, n_probes=cfg["n_probes"],
        probe_seed=PROBE_SEED, device=str(dev))
    n, d = meta["n"], meta["d"]
    x0 = make_patch_initial_point(INIT_SEED)
    L_arr = np.asarray(L, dtype=float)
    epoch_len = max(1, int(np.ceil(n / float(cfg["msvrg_batch"]))))
    scfg = dict(cfg)
    scfg.update(STEPPER_CFG)
    stepper = make_stepper(STEPPER_NAME, d, scfg)                # hook 0
    print(f"[{policy_name}|K10|{STEPPER_NAME}] instance in {time.time() - t_build:.1f}s "
          f"(n={n} d={d} per_class={meta['per_class']} mu={cfg['mu']:g} "
          f"epoch_len={epoch_len} L in [{L_arr.min():.3f},{L_arr.max():.3f}] "
          f"seed={sampler_seed} device={meta['device_description']})", flush=True)

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
    _sync(dev)
    t0 = time.time()
    decision_seconds = 0.0
    prev_lam = None
    next_report = 0.1
    while budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
        t_dec = time.time()
        lam = np.asarray(next_lam(grams, fvals, prev_lam), dtype=float)
        decision_seconds += time.time() - t_dec
        L_lam = float(lam @ L_arr)
        if prev_lam is None or not np.array_equal(lam, prev_lam):
            stepper.on_lambda_change(lam, L_lam, L_scale)          # hook 1
        prev_lam = lam
        lam_history.append(lam.copy())

        retries_here = 0
        for _k in range(args.s):
            if not budget.allows_segment(epoch_len, cfg["msvrg_batch"]):
                break
            g_a_full = chain_J.T @ lam
            F_a = float(chain_f @ lam)
            stepper.start_segment(chain_x, g_a_full, L_lam, L_scale, epoch_len)   # hook 2
            stoch.set_anchor(chain_x)
            y = chain_x.copy()
            for _t in range(epoch_len):
                batch = _support_batch(stoch.sample_batch(), lam)
                g_y_S, g_a_S = stoch.grad_pair(y, lam, batch)
                y = stepper.step(y, (g_y_S - g_a_S + g_a_full))        # hook 3
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
            stepper.on_segment_result(accepted, L_lam, L_scale)        # hook 4
            seg_diag.append(stepper.diag())

            if budget.spent() - grad_at_ck >= args.eval_every:
                grad_at_ck = budget.spent()
                _sync(dev)
                ck_grads.append(budget.spent())
                ck_cpu.append(time.time() - t0)
                ck_m.append(len(grams))
            if budget.spent() >= next_report * args.budget:
                print(f"[{policy_name}|K10] {100 * next_report:.0f}% of budget: "
                      f"segments={len(grams) - 1} wall={time.time() - t0:.0f}s "
                      f"decision={decision_seconds:.0f}s", flush=True)
                next_report += 0.1

    _sync(dev)
    wall = time.time() - t0
    ck_grads.append(budget.spent())
    ck_cpu.append(wall)
    ck_m.append(len(grams))
    Ms = np.asarray(grams, dtype=float)
    lam_arr = np.asarray(lam_history, dtype=float)
    seg_lam_arr = np.asarray(seg_lams, dtype=float)
    supp = (seg_lam_arr[1:] > 0).sum(axis=1)
    support_hist = {int(s): int((supp == s).sum()) for s in np.unique(supp)}
    print(f"[{policy_name}|K10|{STEPPER_NAME}] budget spent: {budget.spent():.1f} of "
          f"{args.budget:g} | segments={len(grams) - 1} | wall={wall:.1f}s | "
          f"decision={decision_seconds:.1f}s ({100 * decision_seconds / max(wall, 1e-9):.1f}%) "
          f"| L_scale={L_scale} | support histogram {support_hist}", flush=True)

    summary = {
        "protocol": ("pure fixed budget at K=10 (MNIST all digits, patch-softplus, ridge): "
                     "shared segment unit, shared s, chain warm start; ONLY the next-lambda "
                     "policy differs; no tolerance parameter; stop = budget; worst GN "
                     "audited post hoc by the audit stage (CCP instruments + random check)"),
        "policy": policy_name, "K": K, "mu": cfg["mu"],
        "config_instance": _json_ready(cfg), "extra": _json_ready(extra_cfg),
        "budget": args.budget, "s": args.s, "eval_every": args.eval_every,
        "stepper": {"name": STEPPER_NAME, "cfg": _json_ready(STEPPER_CFG),
                    "final_diag": _json_ready(stepper.diag())},
        "sampler_seed": int(sampler_seed), "init_seed": INIT_SEED, "probe_seed": PROBE_SEED,
        "d": d, "n": n, "per_class": meta["per_class"], "epoch_len": epoch_len,
        "batch_per_class": [int(v) for v in stoch.b_k],
        "L_calibrated": [float(v) for v in L_arr],
        "grad_equiv_total": float(budget.spent()), "joint_calls": int(budget.joint_calls),
        "wall_seconds": wall, "decision_seconds": decision_seconds,
        "decision_share": float(decision_seconds / max(wall, 1e-9)),
        "segments_total": int(len(grams) - 1), "m_final": int(Ms.shape[0]),
        "n_decisions": int(len(lam_history)),
        "support_histogram": support_hist,
        "L_scale_final": L_scale, "safeguard_retries": int(safeguard_retries),
        "ck_grads": ck_grads, "ck_cpu": ck_cpu, "ck_m": ck_m,
        "audit": None,                      # filled by the audit stage
        "stepper_seg_diag": _json_ready(seg_diag),
        "machine": _machine(dev), "device": meta["device"],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2),
                                          encoding="utf-8")
    np.savez_compressed(out_dir / "grams.npz", gram_stack=Ms, fvals=np.asarray(fvals),
                        lam_history=lam_arr, seg_grads=np.asarray(seg_grads, dtype=float),
                        seg_lams=seg_lam_arr)
    theta_stack = np.asarray(thetas, dtype=np.float32 if args.thetas_float32 else np.float64)
    np.savez_compressed(out_dir / "thetas.npz", theta_stack=theta_stack)
    print(f"[{policy_name}|K10] saved summary.json, grams.npz, thetas.npz "
          f"({theta_stack.nbytes / 1e6:.0f} MB raw) -> {out_dir}", flush=True)
    return summary


def _load_or_run(out_dir, fn, force=False):
    if (out_dir / "summary.json").exists() and not force:
        print(f"[skip] {out_dir.name} (summary.json exists; resume)", flush=True)
        return json.loads((out_dir / "summary.json").read_text())
    return fn(out_dir)


# --------------------------------------------------------------- train --
def stage_train(a, home, cfg, dev):
    rs = a.rs
    args = argparse.Namespace(budget=float(a.budget), eval_every=float(a.budget) / a.n_checkpoints,
                              s=a.s, thetas_float32=a.thetas_float32)
    home.mkdir(parents=True, exist_ok=True)
    manifest_p = home / "campaign_manifest.json"
    manifest = (json.loads(manifest_p.read_text()) if manifest_p.exists()
                else {"campaign": "K10 MNIST all-digits dots campaign", "legs": []})
    manifest.update({"budget": a.budget, "s": a.s, "n_checkpoints": a.n_checkpoints,
                     "eval_every": args.eval_every, "rs": rs, "seed": a.seed, "mu": cfg["mu"],
                     "per_class": cfg["per_class"], "core": CORE_TAG,
                     "machine": _machine(dev), "home": str(home)})

    def _done(name, sm, t_leg):
        manifest["legs"] = [l for l in manifest["legs"] if l["leg"] != name]
        manifest["legs"].append({"leg": name, "leg_wall_seconds": t_leg,
                                 "wall_seconds": sm["wall_seconds"],
                                 "decision_seconds": sm["decision_seconds"],
                                 "segments": sm["segments_total"], "m_final": sm["m_final"],
                                 "finished_at": sm.get("finished_at")})
        manifest_p.write_text(json.dumps(_json_ready(manifest), indent=2))

    t_all = time.time()
    if a.only in (None, "adaptive"):
        name = f"adaptive_ccp_seed{a.seed}"

        def _ccp(od):
            stats = []
            sm = _run_leg_k10("adaptive_ccp", _ccp_policy_bulk(_ccp_cfg(), stats), cfg,
                              args, od, {"core": CORE_TAG, "ccp_config": vars(_ccp_cfg())},
                              dev, a.seed)
            sm["ccp"] = _stats_block(stats)
            (od / "summary.json").write_text(json.dumps(_json_ready(sm), indent=2),
                                             encoding="utf-8")
            return sm
        t0 = time.time()
        sm = _load_or_run(home / name, _ccp, a.force)
        _done(name, sm, time.time() - t0)
        print(f"[dots-K10] adaptive: leg {time.time() - t0:.0f}s, total {time.time() - t_all:.0f}s",
              flush=True)

    if a.only in (None, "uniform"):
        for r in rs:
            grid = _sort_grid_for_warmstart(_uniform_simplex_grid(K, r))
            name = f"uniform_r{r}_seed{a.seed}"
            t0 = time.time()
            sm = _load_or_run(home / name, lambda od, g=grid, rr=r: _run_leg_k10(
                "baseline", _baseline_policy(g), cfg, args, od,
                {"r": rr, "core": CORE_TAG, "nodes": int(g.shape[0])}, dev, a.seed), a.force)
            _done(name, sm, time.time() - t0)
            print(f"[dots-K10] uniform r={r} ({grid.shape[0]} nodes): leg {time.time() - t0:.0f}s, "
                  f"total {time.time() - t_all:.0f}s", flush=True)
    print(f"[dots-K10] TRAIN DONE in {time.time() - t_all:.0f}s -> {home}", flush=True)


# --------------------------------------------------------------- audit --
def _random_check(Q, n_samples, seed):
    """Exact envelope on n uniform random lambdas — no local search."""
    rng = np.random.default_rng(seed)
    best, best_lam = -np.inf, None
    chunk = 2000
    for i in range(0, n_samples, chunk):
        lams = sample_simplex_exp(min(chunk, n_samples - i), K, rng)
        vals, _ = phi_batch(Q, lams)
        j = int(np.argmax(vals))
        if float(vals[j]) > best:
            best, best_lam = float(vals[j]), lams[j].copy()
    return best, best_lam


def _audit_leg(leg_dir, a):
    sm = json.loads((leg_dir / "summary.json").read_text())
    if sm.get("audit") and not a.force:
        print(f"[audit] skip {leg_dir.name} (already audited)", flush=True)
        return sm
    Q = np.load(leg_dir / "grams.npz")["gram_stack"]
    ck_m = [int(v) for v in sm["ck_m"]]
    t_a = time.time()
    hist, lams, walls = [], [], []
    for m_ck in ck_m:
        t = time.time()
        solver = CCPLambdaSolverBulk(K, CCPConfig(N0=a.audit_n0, r=a.audit_r, seed=1,
                                                  seed_sampler="exp",
                                                  adaptive_seed_schedule=False))
        v, lam = solver.solve(Q[:m_ck])
        hist.append(float(v)); lams.append([float(x) for x in lam]); walls.append(time.time() - t)
    # end point: heavier CCP with two sampler seeds + random check
    end = {}
    for sd in (1, 2):
        t = time.time()
        solver = CCPLambdaSolverBulk(K, CCPConfig(N0=a.final_n0, r=a.audit_r, seed=sd,
                                                  seed_sampler="exp",
                                                  adaptive_seed_schedule=False))
        v, lam = solver.solve(Q)
        end[f"ccp_N{a.final_n0}_seed{sd}"] = {"value": float(v), "lam": [float(x) for x in lam],
                                              "seconds": time.time() - t}
    t = time.time()
    v_r, lam_r = _random_check(Q, a.random_n, seed=3)
    end[f"random_{a.random_n}"] = {"value": float(v_r), "lam": [float(x) for x in lam_r],
                                   "seconds": time.time() - t}
    end["checkpoint_instrument_final"] = {"value": hist[-1], "lam": lams[-1]}
    winner = max(end, key=lambda k: end[k]["value"])
    final = float(end[winner]["value"])
    hist_final = list(hist)
    hist_final[-1] = max(hist_final[-1], final)      # the last stack IS the final stack
    mono_viol = sum(int(hist_final[i + 1] > hist_final[i] + 1e-12) for i in range(len(hist_final) - 1))
    audit = {
        "instrument_checkpoint": f"CCP N0={a.audit_n0} r={a.audit_r} seed=1 (fresh per stack)",
        "instrument_endpoint": (f"max of CCP N0={a.final_n0} r={a.audit_r} seeds 1,2 and "
                                f"{a.random_n} uniform random lambdas (exact envelope)"),
        "scale": "squared (min ||J^T lam||^2); *_norm fields are square roots",
        "ck_m": ck_m, "audited_gn_history": hist_final,
        "audited_gn_norm_history": [float(np.sqrt(max(v, 0.0))) for v in hist_final],
        "audit_lam_history": lams, "checkpoint_seconds": walls,
        "endpoint": end, "endpoint_winner": winner,
        "final_audit": final, "final_audit_norm": float(np.sqrt(max(final, 0.0))),
        "audit_mono_violations": int(mono_viol), "audit_seconds": time.time() - t_a,
        "audited_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (leg_dir / "audit.json").write_text(json.dumps(_json_ready(audit), indent=2))
    # merge into summary.json (fields the plot scripts read)
    sm["audit"] = {k: audit[k] for k in ("instrument_checkpoint", "instrument_endpoint",
                                         "endpoint_winner", "audit_seconds", "audited_at")}
    for k in ("audited_gn_history", "audited_gn_norm_history", "audit_lam_history",
              "final_audit", "final_audit_norm", "audit_mono_violations"):
        sm[k] = audit[k]
    (leg_dir / "summary.json").write_text(json.dumps(_json_ready(sm), indent=2), encoding="utf-8")
    print(f"[audit] {leg_dir.name}: m={Q.shape[0]} checkpoints={len(ck_m)} "
          f"final worst GN (norm) = {audit['final_audit_norm']:.4e} (winner {winner}, "
          f"mono violations {mono_viol}) in {audit['audit_seconds']:.0f}s", flush=True)
    return sm


def _leg_dirs(home, a):
    legs = []
    if a.only in (None, "adaptive"):
        legs.append(home / f"adaptive_ccp_seed{a.seed}")
    if a.only in (None, "uniform"):
        legs += [home / f"uniform_r{r}_seed{a.seed}" for r in a.rs]
    return [d for d in legs if (d / "summary.json").exists()]


def stage_audit(a, home):
    t0 = time.time()
    for d in _leg_dirs(home, a):
        _audit_leg(d, a)
    print(f"[dots-K10] AUDIT DONE in {time.time() - t0:.0f}s", flush=True)


# -------------------------------------------------------------- tables --
def _mc_hypervolume(F, ref, n_samples, seed=2026):
    """Monte-Carlo hypervolume of the set F (m, K) w.r.t. the reference
    box [0, ref]^K: fraction of uniform samples dominated by some point,
    times ref^K.  Points outside the box contribute nothing."""
    F = np.asarray(F, dtype=float)
    inside = np.all(F < ref, axis=1)
    frac_inside = float(inside.mean()) if len(F) else 0.0
    P = F[inside]
    if P.shape[0] == 0:
        return 0.0, 0.0, frac_inside, 0
    # non-dominated filter (chunked pairwise)
    keep = np.ones(P.shape[0], dtype=bool)
    for i in range(0, P.shape[0], 1000):
        blk = P[i:i + 1000]
        # dominated if some other point <= blk in all coords and < in one
        dom = np.zeros(blk.shape[0], dtype=bool)
        for j in range(0, P.shape[0], 3000):
            other = P[j:j + 3000]
            le = np.all(other[None, :, :] <= blk[:, None, :], axis=2)
            lt = np.any(other[None, :, :] < blk[:, None, :], axis=2)
            dom |= np.any(le & lt, axis=1)
        keep[i:i + 1000] = ~dom
    P = P[keep]
    rng = np.random.default_rng(seed)
    hits = 0
    chunk = 1000
    for i in range(0, n_samples, chunk):
        U = rng.random((min(chunk, n_samples - i), K)) * ref
        dominated = np.zeros(U.shape[0], dtype=bool)
        for j in range(0, P.shape[0], 3000):
            blk = P[j:j + 3000]
            dominated |= np.any(np.all(blk[None, :, :] <= U[:, None, :], axis=2), axis=1)
            if dominated.all():
                break
        hits += int(dominated.sum())
    p = hits / float(n_samples)
    vol = ref ** K
    return p * vol, float(np.sqrt(p * (1 - p) / n_samples)) * vol, frac_inside, int(P.shape[0])


def stage_tables(a, home, cfg):
    legs = _leg_dirs(home, a)
    sms = {d.name: json.loads((d / "summary.json").read_text()) for d in legs}
    ad_name = f"adaptive_ccp_seed{a.seed}"
    if ad_name not in sms or "final_audit_norm" not in sms[ad_name]:
        raise SystemExit("tables need the audited adaptive leg (run --stage audit first)")
    ad = sms[ad_name]
    n, epoch_len, b_k = ad["n"], ad["epoch_len"], np.asarray(ad["batch_per_class"])
    rows1 = [{"method": "adaptive λ-bundle (CCP)", "r": None, "nodes": None, "passes": None,
              "final_worst_gn": ad["final_audit_norm"], "ratio": 1.0,
              "segments": ad["segments_total"], "decision_share": ad["decision_share"],
              "wall_seconds": ad["wall_seconds"]}]
    for r in a.rs:
        name = f"uniform_r{r}_seed{a.seed}"
        if name not in sms or "final_audit_norm" not in sms[name]:
            continue
        sm = sms[name]
        grid = _uniform_simplex_grid(K, r)
        per_pass = a.s * float(_grid_cost_table(grid, epoch_len, b_k, n).sum())
        rows1.append({"method": "uniform grid", "r": r, "nodes": int(grid.shape[0]),
                      "passes": float(sm["grad_equiv_total"]) / per_pass, "per_pass_budget": per_pass,
                      "final_worst_gn": sm["final_audit_norm"],
                      "ratio": sm["final_audit_norm"] / ad["final_audit_norm"],
                      "segments": sm["segments_total"], "wall_seconds": sm["wall_seconds"]})
    md = ["| method | r | nodes | passes | final worst GN (norm) | method / CCP |",
          "|---|---|---|---|---|---|"]
    for row in rows1:
        md.append(f"| {row['method']} | {row['r'] if row['r'] else '—'} | "
                  f"{row['nodes'] if row['nodes'] else '—'} | "
                  f"{('%.2f' % row['passes']) if row['passes'] else '—'} | "
                  f"{row['final_worst_gn']:.3e} | {row['ratio']:.2f}x |")
    md.append(f"\nB = {a.budget:,.0f}, seed {a.seed}, s = {a.s}; passes = budget spent / per-pass "
              f"budget of the grid (support-aware cost per segment); CCP decision share "
              f"{100 * ad['decision_share']:.1f}% of its wall clock.")
    (home / "table1_K10.md").write_text("\n".join(md), encoding="utf-8")
    (home / "table1_K10.json").write_text(json.dumps(_json_ready(rows1), indent=2))
    print("\n".join(md), flush=True)

    # Table 2: Monte-Carlo hypervolume of the training front
    rows2 = []
    for d in legs:
        F = np.load(d / "grams.npz")["fvals"]
        t = time.time()
        hv, se, frac, n_front = _mc_hypervolume(F, LN_K, a.hv_samples)
        best_balanced = float(np.min(F.max(axis=1)))
        rows2.append({"leg": d.name, "hv": hv, "hv_se": se, "frac_inside_box": frac,
                      "n_nondominated_inside": n_front, "n_points": int(F.shape[0]),
                      "best_max_class_ce": best_balanced, "seconds": time.time() - t})
        print(f"[table2] {d.name}: HV {hv:.4f} ± {se:.4f}, inside box {100 * frac:.1f}%, "
              f"front {n_front}, best max-class CE {best_balanced:.3f} ({time.time() - t:.0f}s)",
              flush=True)
    hv_ad = next(r["hv"] for r in rows2 if r["leg"] == ad_name)
    md2 = [f"| run | HV (train, MC, ref (ln 10)^10 = {LN_K ** K:.1f}) | gap vs CCP | "
           "delivered points inside the reference box | non-dominated inside | best max-class CE |",
           "|---|---|---|---|---|---|"]
    for r in rows2:
        gap = 100.0 * (hv_ad - r["hv"]) / hv_ad if hv_ad > 0 else float("nan")
        md2.append(f"| {r['leg']} | {r['hv']:.4f} ± {r['hv_se']:.4f} | {gap:+.2f}% | "
                   f"{100 * r['frac_inside_box']:.1f}% | {r['n_nondominated_inside']} | "
                   f"{r['best_max_class_ce']:.3f} |")
    md2.append(f"\nMonte Carlo with {a.hv_samples:,} uniform samples in [0, ln 10]^10 (seed 2026); "
               "gap = (HV_CCP − HV_run) / HV_CCP. A delivered point with any class CE ≥ ln 10 "
               "lies outside the reference box and contributes nothing.")
    (home / "table2_K10.md").write_text("\n".join(md2), encoding="utf-8")
    (home / "table2_K10.json").write_text(json.dumps(_json_ready(rows2), indent=2))
    print("\n".join(md2), flush=True)


# ------------------------------------------------------------- figures --
def stage_figures(a, home):
    py = sys.executable
    cmds = [
        [py, str(HERE / "plot_curves_family_without_256_checkpoints.py"), "--home", str(home),
         "--family", "uniform", "--mark", "laststep", "--seed", str(a.seed),
         "--name", "worst_gn_curves_uniform_all_laststep"],
        [py, str(HERE / "plot_dots_figure_without_256_checkpoints.py"), "--home", str(home),
         "--dot", "laststep", "--connect", "--clean", "--legend-style", "ccp",
         "--seed", str(a.seed), "--name", "worst_gn_dots_paper",
         "--title", f"K = 10 MNIST, all digits, B = {a.budget:,.0f}: adaptive CCP vs uniform grids"],
    ]
    for cmd in cmds:
        print("[figures] " + " ".join(cmd[1:]), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        tail = (res.stdout + res.stderr).strip().splitlines()[-3:]
        print("\n".join("    " + l for l in tail), flush=True)
        if res.returncode != 0:
            print(f"[figures] FAILED (exit {res.returncode}) — see the message above", flush=True)
    print(f"[dots-K10] figures -> {home}", flush=True)


# ---------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "audit", "tables", "figures", "all"], default="all")
    ap.add_argument("--budget", type=float, default=500_000.0)
    ap.add_argument("--n-checkpoints", type=int, default=40)
    ap.add_argument("--s", type=int, default=5)
    ap.add_argument("--rs", default=None, help="comma-separated uniform resolutions (default 3,4,5,6,7)")
    ap.add_argument("--only", choices=["uniform", "adaptive"], default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SAMPLER_SEED)
    ap.add_argument("--mu", type=float, default=FULL_CFG["mu"])
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's)")
    ap.add_argument("--thetas-float32", action="store_true", help="store theta stacks as float32")
    ap.add_argument("--skip-calibration", action="store_true",
                    help="train on cuda without a passed calibration_K10.json")
    ap.add_argument("--audit-n0", type=int, default=8192)
    ap.add_argument("--audit-r", type=int, default=20)
    ap.add_argument("--final-n0", type=int, default=32768)
    ap.add_argument("--random-n", type=int, default=100_000)
    ap.add_argument("--hv-samples", type=int, default=1_000_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny wiring test on CPU (~1 min)")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = dict(SMOKE_CFG if a.smoke else FULL_CFG)
    cfg["mu"] = float(a.mu)
    if a.per_class:
        cfg["per_class"] = int(a.per_class)
    if a.smoke:
        a.budget, a.n_checkpoints, a.s = 800.0, 8, 2
        a.rs = a.rs or "2"
        a.audit_n0, a.final_n0, a.random_n, a.hv_samples = 512, 2048, 5000, 20000
        a.device = "cpu" if a.device == "auto" else a.device
        a.skip_calibration = True
    a.rs = [int(v) for v in a.rs.split(",")] if a.rs else list(DEFAULT_RS)
    home = _home(a)
    dev = resolve_device(a.device) if a.stage in ("train", "all") else torch.device("cpu")
    print(f"[dots-K10] stage={a.stage} home={home} B={a.budget:g} rs={a.rs} s={a.s} "
          f"checkpoints={a.n_checkpoints} seed={a.seed} mu={cfg['mu']:g} "
          f"per_class={cfg['per_class']} device={a.device} smoke={a.smoke}", flush=True)

    if a.stage in ("train", "all"):
        if dev.type == "cuda" and not a.skip_calibration:
            cal = K10_ROOT / "calibration_K10.json"
            ok = cal.exists() and json.loads(cal.read_text()).get("pass") is True
            if not ok:
                raise SystemExit(
                    f"GPU training requires a passed calibration: run\n"
                    f"  python calibrate_gpu_K10_without_256_checkpoints.py --device {a.device}\n"
                    f"first (expected {cal} with \"pass\": true), or pass --skip-calibration.")
        stage_train(a, home, cfg, dev)
    if a.stage in ("audit", "all"):
        stage_audit(a, home)
    if a.stage in ("tables", "all"):
        stage_tables(a, home, cfg)
    if a.stage in ("figures", "all"):
        stage_figures(a, home)

    if a.smoke:
        for d in _leg_dirs(home, a):
            sm = json.loads((d / "summary.json").read_text())
            assert sm["grad_equiv_total"] <= a.budget + 1e-6, "budget overrun"
            assert np.isfinite(sm["final_audit"]), "non-finite audit"
            assert (d / "grams.npz").exists() and (d / "thetas.npz").exists()
        for f in ("table1_K10.md", "table2_K10.md", "worst_gn_dots_paper.png",
                  "worst_gn_curves_uniform_all_laststep.png"):
            assert (home / f).exists(), f"missing {f}"
        print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
