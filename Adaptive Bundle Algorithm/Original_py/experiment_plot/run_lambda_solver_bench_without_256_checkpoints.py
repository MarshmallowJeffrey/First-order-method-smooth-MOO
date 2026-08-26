"""run_lambda_solver_bench_without_256_checkpoints.py — experiment 2:
controlled λ-solver benchmark, CCP polish vs single-start IPOPT, on
frozen Gram snapshots.

NEW FILE (Aug 9, 2026).  Design: Note/Aug_9_note.md §6 (user-confirmed
rules: paired seed batches, strict fixed-time cutoffs, one 60 s
trajectory read at every prefix).  No existing file is modified.

Snapshots: early / mid / late Gram-stack prefixes of the K2 and K6
adaptive-IPOPT legs from the comparison campaign (6 stacks, saved as
Q_i only).

2a (paired, same starts): per snapshot, ``n_seed_batches`` batches of
    [vertices + lambda_A + 2048 Exp(1) seeds] -> shared screening
    (top r = 10, l1-separated) -> the SAME starts go to both polishers:
    * ccp:   CCPLambdaSolver._polish (warm HiGHS game LP)
    * ipopt: one cyipopt local solve from the start (the fast module's
      per-start recipe: Danskin gradient, L-BFGS Hessian, tol 1e-8)
    Per restart: phi, wall time, iterations, unified residual
    delta(lam*) = val(M^(lam*)) - phi(lam*), active-set size, success.
2b (fixed time, shared stream): ONE 60 s trajectory per (snapshot,
    method) over one shared ordered unscreened seed stream; in-flight
    restarts finish and are recorded; best@T uses only restarts with
    t_complete <= T (no timeout bonus).  best@10s falls out as a
    prefix.

Outputs -> output/ccp_compare_without_256_checkpoints/lambda_solver_bench/:
    snapshots/*.npz, bench_2a.csv, bench_2b.csv, bench_summary.json,
    summary.md, fig_2a_time_box.png, fig_2b_best_vs_time.png,
    fig_2b_best_vs_restarts.png

Usage:
    python run_lambda_solver_bench_without_256_checkpoints.py            # full
    python run_lambda_solver_bench_without_256_checkpoints.py --quick    # tiny
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from algorithm_fast_without_256_checkpoints import (  # noqa: E402
    _HAS_IPOPT,
    _gn_value_and_jac_batched_gram,
    _ipopt_minimize,
)
from ccp_lambda_solver import (  # noqa: E402
    CCPConfig,
    CCPLambdaSolver,
    _GameLP,
    _active_set,
    _phi_terms,
    _project_simplex,
    phi_batch,
    sample_simplex_exp,
)

HERE = Path(__file__).resolve().parent
COMPARE_ROOT = (HERE.parent.parent / "output"
                / "CCP/ccp_compare_without_256_checkpoints")
BENCH_HOME = COMPARE_ROOT / "lambda_solver_bench"
BASE_SEED = 20260809
R_RETAIN = 10
N_BATCH_SEEDS = 2048
SEP_L1 = 0.05
TAU_REL = 1e-8
DEDUP = dict(l1=1e-3, phi_rel=1e-9, active_tol=1e-7)


# =====================================================================
#  Snapshots
# =====================================================================
def collect_snapshots(quick: bool):
    """early/mid/late prefixes of the campaign's adaptive-IPOPT legs."""
    out = []
    specs = (("K2", COMPARE_ROOT / "K2_B20000" / "adaptive_s5_ts24"),
             ("K6", COMPARE_ROOT / "K6_B80912" / "adaptive_s5_ts24"))
    for tag, leg in specs:
        if not (leg / "summary.json").exists():
            print(f"[snapshots] {leg} missing — skipped", flush=True)
            continue
        sm = json.loads((leg / "summary.json").read_text())
        Ms = np.asarray(np.load(leg / "grams.npz")["gram_stack"], float)
        ck_m = [m for m in sm["ck_m"] if m >= 2]
        picks = sorted({ck_m[0], ck_m[len(ck_m) // 2], Ms.shape[0]})
        for stage, m in zip(("early", "mid", "late"), picks):
            out.append({"name": f"{tag}_{stage}_m{m}", "K": Ms.shape[1],
                        "Q": Ms[:m].copy(), "m": int(m), "src": str(leg)})
    if quick:
        out = out[:2]
    (BENCH_HOME / "snapshots").mkdir(parents=True, exist_ok=True)
    for s in out:
        np.savez_compressed(BENCH_HOME / "snapshots" / f"{s['name']}.npz",
                            Q=s["Q"], meta=json.dumps(
                                {k: s[k] for k in ("name", "K", "m", "src")}))
    return out


# =====================================================================
#  Polishers
# =====================================================================
def make_seed_pack(Q, K, rng, n_random):
    """vertices + lambda_A + Exp(1) random block (production seed mix)."""
    A = np.diagonal(Q, axis1=1, axis2=2)
    lp = _GameLP(K)
    _vA, lam_A = lp.resolve(A)
    seeds = np.vstack([np.eye(K), _project_simplex(lam_A, K)[None, :],
                       sample_simplex_exp(n_random, K, rng)])
    return seeds


def screen_top_r(Q, seeds, r=R_RETAIN, sep=SEP_L1):
    phis, _ = phi_batch(Q, seeds)
    order = np.argsort(-phis)
    kept = []
    for pos in order:
        if all(float(np.abs(seeds[pos] - seeds[q]).sum()) > sep
               for q in kept):
            kept.append(int(pos))
            if len(kept) >= r:
                break
    for pos in order:
        if len(kept) >= r:
            break
        if int(pos) not in kept:
            kept.append(int(pos))
    return seeds[kept]


class Bench:
    """Both polishers + the unified residual on one snapshot."""

    def __init__(self, Q, K):
        self.Q, self.K = np.ascontiguousarray(Q), K
        self.solver = CCPLambdaSolver(
            K, CCPConfig(seed_sampler="exp", adaptive_seed_schedule=False))
        self.res_lp = _GameLP(K)          # residual ruler (shared)

    def delta_residual(self, lam):
        G, phis = _phi_terms(self.Q, lam)
        t_star, _ = self.res_lp.resolve(2.0 * G - phis[:, None])
        return float(t_star - np.min(phis))

    def _common(self, lam):
        lam = _project_simplex(lam, self.K)
        phis = _phi_terms(self.Q, lam)[1]
        phi = float(np.min(phis))
        act = len(_active_set(phis, DEDUP["active_tol"]))
        return lam, phi, act

    def run_ccp(self, lam0):
        t0 = time.perf_counter()
        lam, phi, _phis, iters, delta = self.solver._polish(
            self.Q, lam0, epsilon=None)
        dt = time.perf_counter() - t0
        lam, phi, act = self._common(lam)
        tau = TAU_REL * max(1.0, abs(phi))
        return {"phi": phi, "time_s": dt, "iters": iters,
                "delta": self.delta_residual(lam), "active": act,
                "converged": int(delta <= tau), "solver_success": 1,
                "lam": lam}

    def run_ipopt(self, lam0):
        Q, K = self.Q, self.K
        def neg(lam):
            v, _, _ = _gn_value_and_jac_batched_gram(Q, lam)
            return -v
        def jac(lam):
            _, j, _ = _gn_value_and_jac_batched_gram(Q, lam)
            return -j
        cons = [{"type": "eq", "fun": lambda l: float(np.sum(l) - 1.0),
                 "jac": lambda l: np.ones(K)}]
        t0 = time.perf_counter()
        res = _ipopt_minimize(
            neg, _project_simplex(lam0, K), jac=jac,
            bounds=[(1e-8, 1.0)] * K, constraints=cons,
            options={"print_level": 0, "sb": "yes", "tol": 1e-8,
                     "max_iter": 100,
                     "hessian_approximation": "limited-memory"})
        dt = time.perf_counter() - t0
        lam, phi, act = self._common(res.x)
        # never lose ground to the raw start (fast-module discipline)
        lam0p, phi0, act0 = self._common(lam0)
        if phi0 > phi:
            lam, phi, act = lam0p, phi0, act0
        delta = self.delta_residual(lam)
        tau = TAU_REL * max(1.0, abs(phi))
        return {"phi": phi, "time_s": dt,
                "iters": int(getattr(res, "nit", -1)),
                "delta": delta, "active": act,
                "converged": int(delta <= tau),
                "solver_success": int(bool(getattr(res, "success", False))),
                "lam": lam}


def dedup_count(cands):
    """Distinct local maxima under the solver's dedup keys."""
    kept = []
    for c in sorted(cands, key=lambda c: -c["phi"]):
        dup = False
        for k in kept:
            if float(np.abs(c["lam"] - k["lam"]).sum()) <= DEDUP["l1"]:
                dup = True
                break
            if (c["act_set"] == k["act_set"] and abs(c["phi"] - k["phi"])
                    <= DEDUP["phi_rel"] * max(1.0, abs(k["phi"]))):
                dup = True
                break
        if not dup:
            kept.append(c)
    return len(kept)


# =====================================================================
#  2a / 2b
# =====================================================================
def run_2a(snap, n_batches, rows):
    Q, K, name = snap["Q"], snap["K"], snap["name"]
    bench = Bench(Q, K)
    for b in range(n_batches):
        rng = np.random.default_rng(
            [BASE_SEED, 21, zlib.crc32(name.encode()), b])
        starts = screen_top_r(Q, make_seed_pack(Q, K, rng, N_BATCH_SEEDS))
        for j, lam0 in enumerate(starts):
            for method in ("ccp", "ipopt"):
                out = (bench.run_ccp if method == "ccp"
                       else bench.run_ipopt)(lam0)
                lam = out.pop("lam")
                rows.append({"snapshot": name, "K": K, "m": snap["m"],
                             "batch": b, "start": j, "method": method,
                             **out})
    print(f"[2a] {name}: {n_batches} batches done", flush=True)


def run_2b(snap, T_max, rows, curves):
    Q, K, name = snap["Q"], snap["K"], snap["name"]
    for method in ("ccp", "ipopt"):
        bench = Bench(Q, K)
        rng = np.random.default_rng(
            [BASE_SEED, 22, zlib.crc32(name.encode())])
        stream = make_seed_pack(Q, K, rng, 4)   # deterministic head
        runner = bench.run_ccp if method == "ccp" else bench.run_ipopt
        runner(np.full(K, 1.0 / K))             # untimed warm-up
        recs, i = [], 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < T_max and i < 300_000:
            if i >= len(stream):                # extend the shared stream
                stream = np.vstack(
                    [stream, sample_simplex_exp(4096, K, rng)])
            out = runner(stream[i])
            out["t_complete"] = time.perf_counter() - t0
            recs.append(out)
            i += 1
        for j, rec in enumerate(recs):
            phis = _phi_terms(Q, rec["lam"])[1]
            rec["act_set"] = _active_set(phis, DEDUP["active_tol"])
            rows.append({"snapshot": name, "K": K, "m": snap["m"],
                         "method": method, "restart": j,
                         "phi": rec["phi"], "time_s": rec["time_s"],
                         "t_complete": rec["t_complete"],
                         "iters": rec["iters"], "delta": rec["delta"],
                         "converged": rec["converged"]})
        curves[(name, method)] = recs
        n10 = sum(1 for r in recs if r["t_complete"] <= 10.0)
        b10 = max((r["phi"] for r in recs if r["t_complete"] <= 10.0),
                  default=float("nan"))
        b60 = max((r["phi"] for r in recs if r["t_complete"] <= T_max),
                  default=float("nan"))
        print(f"[2b] {name} {method:5s}: {len(recs)} restarts "
              f"({n10} within 10s) best@10s={b10:.6e} "
              f"best@{T_max:.0f}s={b60:.6e} "
              f"distinct={dedup_count(recs)}", flush=True)


# =====================================================================
#  Figures + summary
# =====================================================================
def figures(rows2a, curves, T_max):
    snaps = sorted({r["snapshot"] for r in rows2a})
    # 2a per-restart time boxplot (raw samples, log axis)
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(snaps), 4.6))
    data, labels, colors = [], [], []
    for sn in snaps:
        for method, col in (("ccp", "#1f77b4"), ("ipopt", "#d62728")):
            data.append([r["time_s"] for r in rows2a
                         if r["snapshot"] == sn and r["method"] == method])
            labels.append(f"{sn}\n{method}")
            colors.append(col)
    bp = ax.boxplot(data, showfliers=True, whis=(5, 95), patch_artist=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("single-restart wall time, s (log)")
    ax.set_title("2a paired restarts: per-restart time "
                 "(all raw samples; identical starts)")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(BENCH_HOME / "fig_2a_time_box.png", dpi=160)
    plt.close(fig)

    # 2b best-so-far vs time and vs restart index (strict completion)
    for xkey, xlabel, fname, logx in (
        ("t_complete", "elapsed wall time, s",
         "fig_2b_best_vs_time.png", True),
        ("idx", "restarts completed", "fig_2b_best_vs_restarts.png", True),
    ):
        n = len({sn for sn, _ in curves})
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.9),
                                 squeeze=False)
        for ax, sn in zip(axes[0], sorted({sn for sn, _ in curves})):
            for method, col in (("ccp", "#1f77b4"), ("ipopt", "#d62728")):
                recs = curves.get((sn, method), [])
                if not recs:
                    continue
                xs = ([r["t_complete"] for r in recs] if xkey == "t_complete"
                      else list(range(1, len(recs) + 1)))
                best = np.maximum.accumulate([r["phi"] for r in recs])
                ax.step(xs, best, where="post", color=col, lw=1.7,
                        label=f"{method} ({len(recs)} restarts)")
            if xkey == "t_complete":
                ax.axvline(10.0, color="gray", ls=":", lw=1)
            if logx:
                ax.set_xscale("log")
            ax.set_title(sn, fontsize=9)
            ax.set_xlabel(xlabel)
            ax.legend(fontsize=7)
        axes[0][0].set_ylabel("best phi found (GNS lower bound)")
        fig.suptitle(f"2b fixed wall-clock ({T_max:.0f}s, shared seed "
                     "stream, strict t_complete cutoffs)", fontsize=10)
        fig.tight_layout()
        fig.savefig(BENCH_HOME / fname, dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()
    if not _HAS_IPOPT:
        raise SystemExit("cyipopt required for the bench")
    BENCH_HOME.mkdir(parents=True, exist_ok=True)
    T_max = 3.0 if args.quick else args.T
    n_batches = 2 if args.quick else args.batches

    snaps = collect_snapshots(args.quick)
    if not snaps:
        raise SystemExit("no snapshots (run the campaigns first)")
    rows2a, rows2b, curves = [], [], {}
    for snap in snaps:
        run_2a(snap, n_batches, rows2a)
    for snap in snaps:
        run_2b(snap, T_max, rows2b, curves)

    for fname, rows in (("bench_2a.csv", rows2a), ("bench_2b.csv", rows2b)):
        with open(BENCH_HOME / fname, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    # summary tables
    summary = {"T_max": T_max, "n_batches": n_batches, "snapshots": {}}
    for sn in sorted({r["snapshot"] for r in rows2a}):
        block = {}
        for method in ("ccp", "ipopt"):
            a = [r for r in rows2a if r["snapshot"] == sn
                 and r["method"] == method]
            b = [r for r in rows2b if r["snapshot"] == sn
                 and r["method"] == method]
            phis = np.array([r["phi"] for r in a])
            times = np.array([r["time_s"] for r in a])
            block[method] = {
                "2a_n": len(a),
                "2a_phi_best": float(phis.max()),
                "2a_phi_mean": float(phis.mean()),
                "2a_phi_median": float(np.median(phis)),
                "2a_time_median_s": float(np.median(times)),
                "2a_time_p95_s": float(np.percentile(times, 95)),
                "2a_converged_frac": float(np.mean(
                    [r["converged"] for r in a])),
                "2b_restarts": len(b),
                "2b_restarts_within_10s": int(sum(
                    1 for r in b if r["t_complete"] <= 10.0)),
                "2b_best_at_10s": float(max(
                    (r["phi"] for r in b if r["t_complete"] <= 10.0),
                    default=float("nan"))),
                "2b_best_at_T": float(max(
                    (r["phi"] for r in b if r["t_complete"] <= T_max),
                    default=float("nan"))),
                "2b_distinct_maxima": dedup_count(curves[(sn, method)]),
            }
        pair = [(x["phi"], y["phi"]) for x, y in zip(
            sorted((r for r in rows2a if r["snapshot"] == sn
                    and r["method"] == "ccp"),
                   key=lambda r: (r["batch"], r["start"])),
            sorted((r for r in rows2a if r["snapshot"] == sn
                    and r["method"] == "ipopt"),
                   key=lambda r: (r["batch"], r["start"])))]
        wins = sum(1 for c, i in pair if c > i + 1e-12)
        ties = sum(1 for c, i in pair if abs(c - i) <= 1e-12)
        block["2a_paired_ccp_wins"] = wins
        block["2a_paired_ties"] = ties
        block["2a_paired_total"] = len(pair)
        summary["snapshots"][sn] = block
    (BENCH_HOME / "bench_summary.json").write_text(
        json.dumps(summary, indent=2))

    figures(rows2a, curves, T_max)

    lines = ["# lambda-solver bench (experiment 2)", "",
             f"T = {T_max:.0f}s, batches = {n_batches}; rules: "
             "Note/Aug_9_note.md §6.", ""]
    for sn, block in summary["snapshots"].items():
        lines.append(f"## {sn}")
        lines.append("")
        lines.append("| metric | ccp | ipopt |")
        lines.append("|---|---|---|")
        keys = sorted(block["ccp"].keys())
        for k in keys:
            lines.append(f"| {k} | {block['ccp'][k]:.6g} "
                         f"| {block['ipopt'][k]:.6g} |")
        lines.append(f"| paired ccp wins / ties / total | "
                     f"{block['2a_paired_ccp_wins']} / "
                     f"{block['2a_paired_ties']} / "
                     f"{block['2a_paired_total']} | |")
        lines.append("")
    (BENCH_HOME / "summary.md").write_text("\n".join(lines))
    print(f"[bench] outputs -> {BENCH_HOME}", flush=True)


if __name__ == "__main__":
    main()
