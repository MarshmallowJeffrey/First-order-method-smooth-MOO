"""run_pure_budget_K3_mnist_triple_without_256_checkpoints.py — the K = 3
MNIST digit-triple campaign: pure fixed budget, adaptive(CCP) vs
simplex-grid baselines, with theta snapshots and OFFICIAL-TEST
evaluation.

NEW FILE (Aug 26, 2026).  Design: Aug-25/26 Q&A (user-approved;
Smoke A chose the triple, top-1 decision: {3,5,8} only; B = 40,000
signed off Aug 26).  No existing file is modified — executor building
blocks are imported:

* protocol pieces (``_Budget``, ``MAX_SAFEGUARD_RETRIES``,
  ``_baseline_policy``) from the K6 pure-budget runner;
* the simplex grid + snake (boustrophedon) warm-start order from
  ``baseline_without_256_checkpoints`` (K-generic since Aug 25);
* the CCP policy (``_ccp_policy``, ``_stats_block``) from the K2 CCP
  runner (K-generic);
* the problem family from ``objectives_mnist_triple`` (per-class mean
  CE on a 3-logit patch-softplus net, d = 8,195; per_class = balanced
  maximum; batch 1024; NO regularisation).

QUALITY METER (the one convention change vs the K = 2 pair campaign):
the exact certified 1-D meter does not exist at K = 3 (the audit
quantity is the coverage max-min GN*(stack) = max_lam min_i
lam^T M_i lam over the 2-simplex — a nonconvex maximin with no
affordable certified evaluation; a closed-form support-enumeration QP
only solves the SINGLE-point min_lam, a different quantity).  We
therefore use the K6/K10 audit_v2 convention, in-runner and off both
cost axes:

    audit(stack) = max( IPOPT strict multistart value,
                        heavy CCP value (N0=8192, r=20, fresh solver) )

at EVERY checkpoint prefix of EVERY leg (both instruments are lower
bounds of the true GN*; the previous checkpoint's argmax lambda is
threaded into the IPOPT start set).  The estimated history is not
forced monotone: violations of the true non-increasing property are
counted and reported (instrument-miss diagnostic), never clipped.
Additionally, at the FINAL stack a dense simplex-grid lower bound
(resolution 500 -> 125,751 nodes, exact chunked BLAS evaluation) is
computed; if it exceeds the instrument value the miss is recorded and
the grid value (a valid lower bound) is taken as final_audit.

Theta snapshots + official-test evaluation are carried verbatim from
the K = 2 runner (both off-axis; test = ALL official t10k rows of the
three digits).

Legs (user go, Aug 26; r-set revised to {10, 20, 30} by the user the
same day — r=5 too coarse, r=30 (496 nodes, ~1.9 cycles at B=40,000)
deliberately probes the grid-coverage breakdown end): baseline r in
{10, 20, 30} at s = 5 + ``adaptive_s5_ccp`` (production CCP: N0=2000,
r=10, exp sampler, pool on, adaptive schedule off).  B = 40,000
grad-equivalents each, eval_every = 500.  NO IPOPT leg.

Usage:
    python run_pure_budget_K3_mnist_triple_without_256_checkpoints.py            # top-1 triple, all legs
    python run_pure_budget_K3_mnist_triple_without_256_checkpoints.py --triple 3 5 8
    python run_pure_budget_K3_mnist_triple_without_256_checkpoints.py --smoke    # Smoke B
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

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from bundle import validate_oracle_output  # noqa: E402
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
    _baseline_policy,
    _Budget,
)
from run_pure_budget_K2_ccp_without_256_checkpoints import (  # noqa: E402
    _ccp_policy,
    _stats_block,
)
from run_experiments import _json_ready  # noqa: E402
from algorithm_fast_without_256_checkpoints import (  # noqa: E402
    _maximise_GN_fast,
    ipopt_available,
)
from ccp_lambda_solver import CCPConfig, CCPLambdaSolver  # noqa: E402
from objectives_torch import _load_theta_into_net  # noqa: E402
from objectives_mnist_triple import (  # noqa: E402
    TriplePatchMLP,
    load_mnist_triple,
    make_mnist_triple,
    make_triple_initial_point,
)

HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = (HERE.parent.parent / "output"
                 / "CCP/K3_mnist_triple_without_256_checkpoints")
TRIPLE_DEFAULT = (3, 5, 8)      # Smoke A top-1 (S_int 0.8317, Aug 26)
BASELINE_RS = [10, 20, 30]
K = 3
INIT_SEED, SAMPLER_SEED, PROBE_SEED = 8, 41, 7


# ---------------------------------------------------------------------------
# K = 3 audit meter (audit_v2 convention, in-runner)
# ---------------------------------------------------------------------------

def _audit_instruments(Ms_prefix, args, prev_lam, ccp_seed=1):
    """Two-instrument lower bound of GN*(stack): max(IPOPT strict
    multistart, heavy CCP).  Fresh CCP solver per stack (audit_v2
    convention); prev checkpoint's argmax lambda threads into the
    IPOPT start set.  Returns (v_ipopt, v_ccp, v_best, lam_best)."""
    gs = _GramSet(list(Ms_prefix), K)
    v_i, lam_i = _maximise_GN_fast(gs, prev_lam=prev_lam, tier="strict",
                                   max_starts=args.audit_ipopt_starts)
    solver = CCPLambdaSolver(K, CCPConfig(
        N0=args.audit_ccp_n0, r=args.audit_ccp_r, seed=ccp_seed,
        seed_sampler="exp", adaptive_seed_schedule=False))
    v_c, lam_c = solver.solve(np.asarray(Ms_prefix, dtype=float))
    if v_c >= v_i:
        return float(v_i), float(v_c), float(v_c), np.asarray(lam_c, float)
    return float(v_i), float(v_c), float(v_i), np.asarray(lam_i, float)


def _grid_maxmin(Ms, resolution, chunk=4096):
    """EXACT max over the resolution-r simplex grid of min_i lam^T M_i
    lam — an independent lower bound of GN*(stack) used to cross-check
    the instruments.  Quadratic forms via the 6-monomial BLAS matmul
    (M symmetric), node-chunked to bound memory."""
    Ms = np.asarray(Ms, dtype=float)
    lams = _uniform_simplex_grid(K, resolution)
    Q = np.stack([Ms[:, 0, 0], Ms[:, 1, 1], Ms[:, 2, 2],
                  2.0 * Ms[:, 0, 1], 2.0 * Ms[:, 0, 2],
                  2.0 * Ms[:, 1, 2]], axis=1)          # (m, 6)
    best, best_lam = -np.inf, None
    for i in range(0, lams.shape[0], chunk):
        B = lams[i:i + chunk]
        P = np.stack([B[:, 0] ** 2, B[:, 1] ** 2, B[:, 2] ** 2,
                      B[:, 0] * B[:, 1], B[:, 0] * B[:, 2],
                      B[:, 1] * B[:, 2]], axis=1)      # (b, 6)
        mins = (P @ Q.T).min(axis=1)
        j = int(np.argmax(mins))
        if float(mins[j]) > best:
            best, best_lam = float(mins[j]), B[j].copy()
    return best, best_lam


def _test_eval_stack(thetas, X_test, y_test):
    """Per-class mean CE and error rate for every theta (no gradients).

    One net, reloaded per theta — the whole stack is a pure forward
    sweep over the official test rows of the three digits."""
    import torch.nn.functional as F
    net = TriplePatchMLP()
    X = torch.from_numpy(np.ascontiguousarray(X_test))
    rows = [torch.from_numpy(np.nonzero(y_test == k)[0]).long()
            for k in range(K)]
    targets = [torch.full((len(r),), k, dtype=torch.long)
               for k, r in enumerate(rows)]
    ce = np.empty((len(thetas), K))
    err = np.empty((len(thetas), K))
    with torch.no_grad():
        for i, th in enumerate(thetas):
            _load_theta_into_net(net, np.asarray(th, dtype=float))
            Z = net(X)
            for k in range(K):
                Zk = Z[rows[k]]
                ce[i, k] = float(F.cross_entropy(Zk, targets[k],
                                                 reduction="mean"))
                err[i, k] = float((Zk.argmax(dim=1) != targets[k])
                                  .to(torch.float64).mean())
    return ce, err


def _run_leg_triple(policy_name, next_lam, triple, cfg, args, out_dir,
                    extra_cfg):
    """Shared pure-budget executor (K2 replica) + theta snapshots +
    official-test evaluation + audit_v2-style checkpoint audits.
    ONLY the next-lambda policy differs between legs."""
    a, b, c_dg = triple
    tag = f"{a}v{b}v{c_dg}"
    t_build = time.time()
    (_obj, _grad, L, joint_oracle, stoch, meta) = make_mnist_triple(
        triple, per_class=cfg["per_class"], batch_size=cfg["msvrg_batch"],
        sampler_seed=SAMPLER_SEED, init_seed=INIT_SEED,
        n_probes=cfg["n_probes"], probe_seed=PROBE_SEED)
    X_np, y_np = meta.pop("_X"), meta.pop("_y")
    n, d = meta["n"], meta["d"]
    x0 = make_triple_initial_point(INIT_SEED)
    L_arr = np.asarray(L, dtype=float)
    epoch_len = max(1, int(np.ceil(n / float(cfg["msvrg_batch"]))))
    print(f"[{policy_name}|{tag}] instance in {time.time() - t_build:.1f}s "
          f"(n={n} d={d} per_class={meta['per_class']} "
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

    # ---- post-hoc, off both axes: official-test evaluation ----
    t_t = time.time()
    X_test, y_test = load_mnist_triple(triple, train=False)
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
    n_test = [int((y_test == k).sum()) for k in range(K)]
    summary = {
        "protocol": ("pure fixed budget at K=3 (MNIST digit triple, patch-"
                     "softplus): shared segment unit, shared s, chain warm "
                     "start; ONLY the next-lambda policy differs; no "
                     "tolerance anywhere; stop = budget; audit_v2 two-"
                     "instrument meter per checkpoint + final dense-grid "
                     "cross-check; theta snapshots + official-test "
                     "evaluation, all off-axis"),
        "policy": policy_name,
        "triple": [int(a), int(b), int(c_dg)],
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


def _stratified_bk(batch, n_k):
    """Replicate TripleStochLamOracle's stratified allocation."""
    n_k = np.asarray(n_k, dtype=float)
    raw = batch * n_k / n_k.sum()
    b_k = np.maximum(1, np.floor(raw).astype(int))
    short = batch - int(b_k.sum())
    if short > 0:
        for j in np.argsort(-(raw - np.floor(raw)))[:short]:
            b_k[j] += 1
    return np.minimum(b_k, n_k.astype(int))


def _seg_cost_check(out_dir, n, epoch_len, batch):
    """Smoke B: every segment's grad-equivalent cost must equal the
    exact formula for one of the 7 support patterns (any nonempty
    subset of the three classes)."""
    npz = np.load(out_dir / "grams.npz")
    seg = np.diff(np.asarray(npz["seg_grads"], dtype=float))
    b_k = _stratified_bk(batch, [n // K] * K)
    rows_options = {int(sum(b_k[i] for i in sup))
                    for m in range(1, 1 << K)
                    for sup in [[i for i in range(K) if m >> i & 1]]}
    allowed = {round(epoch_len * 2.0 * rows * K / n + K, 12)
               for rows in rows_options}
    bad = [c for c in seg if all(abs(c - v) > 1e-9 for v in allowed)]
    assert not bad, f"segment costs off-formula: {bad[:3]} vs {allowed}"


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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.budget, args.eval_every = 800.0, 100.0
        args.s = 2
        args.grid_check_res = 200
        cfg = dict(per_class=300, msvrg_batch=256, msvrg_step_const=0.1,
                   msvrg_momentum=0.5, n_probes=5)
        triples = [TRIPLE_DEFAULT]
        rs = [4]
        home_root = CAMPAIGN_ROOT / "SMOKE"
    else:
        cfg = dict(per_class=None, msvrg_batch=1024, msvrg_step_const=0.1,
                   msvrg_momentum=0.5, n_probes=40)
        triples = ([tuple(args.triple)] if args.triple
                   else [TRIPLE_DEFAULT])
        rs = list(BASELINE_RS)
        home_root = CAMPAIGN_ROOT

    ccp_cfg = CCPConfig(N0=2000, r=10, seed=args.ccp_seed,
                        seed_sampler="exp", adaptive_seed_schedule=False)
    t_all = time.time()
    for triple in triples:
        a, b, c_dg = triple
        tag = f"{a}v{b}v{c_dg}"
        home = home_root / f"triple_{tag}_B{args.budget:.0f}"
        home.mkdir(parents=True, exist_ok=True)
        manifest_path = home / "campaign_manifest.json"
        manifest = {"campaign": f"K3 MNIST triple {tag} pure budget "
                                f"(baseline r{rs} + adaptive CCP)",
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
                lambda od, g=grid, rr=r: _run_leg_triple(
                    "baseline", _baseline_policy(g), triple, cfg, args, od,
                    {"r": rr, "n_nodes": int(g.shape[0])}))

        def _ccp_leg(od):
            stats: list = []
            sm = _run_leg_triple("adaptive_ccp",
                                 _ccp_policy(K, ccp_cfg, stats), triple,
                                 cfg, args, od,
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
        home = home_root / f"triple_{a}v{b}v{c_dg}_B{args.budget:.0f}"
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
