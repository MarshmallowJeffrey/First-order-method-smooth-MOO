"""run_conflict_smoke_K2_mnist_pairs_without_256_checkpoints.py —
Smoke A for the K = 2 MNIST pair campaign: rank 5 candidate digit
pairs by objective-conflict strength and calibrate the per-segment
grad-equivalent cost c.

NEW FILE (Aug 13, 2026).  User-approved design (Aug-13 Q&A):

* pairs: 4-9, 3-5, 7-9, 3-8, 5-8 (classic MNIST confusions);
* per pair: 5 FIXED lambdas (1,0) (.75,.25) (.5,.5) (.25,.75) (0,1),
  each an INDEPENDENT chain from the same He x0 (no snake hand-off —
  the probe asks "where does this lambda alone go"), each chain gets
  a fresh sampler with the same seed so every lambda sees identical
  minibatch streams; N_SEG = 15 segments per chain;
* segment loop is the verbatim pure-budget executor unit (anchor full
  gradient + MSVRG epoch + safeguard); metering via _Budget with an
  unreachable limit — c is read off consecutive spent() diffs;
* conflict scores, both normalised by ln 2 (the guess-level CE of a
  balanced 2-class task):
  - vertex score  = [F_A((0,1)) - F_A((1,0))] + [F_B((1,0)) - F_B((0,1))]
    over chain ends.  Aug-13 first-run finding: with no regularisation
    the fully-ignored class DIVERGES (CE 14-31 after 15 segments), so
    this score is dominated by divergence speed and ranks all pairs in
    a narrow band — kept as a diagnostic only;
  - interior score = [F_A((.25,.75)) - F_A((.75,.25))]
                   + [F_B((.75,.25)) - F_B((.25,.75))] — both classes
    keep weight, nothing diverges; this is the pair-discriminating
    ruler and decides top2;
* data: per_class=None -> balanced maximum (Aug-13 decision: take all
  MNIST train rows available, 5,421-5,949/class depending on pair);
  batch stays 1024, epoch_len follows ceil(n/b);
* L probes at the smoke tier (10 pairs; production uses 40).

Outputs (new campaign home, Aug-13 convention):
    output/K2_mnist_pair_without_256_checkpoints/conflict_smoke/
        pair_<a>v<b>.json      per-lambda trajectories, c, checks
        scatter_pair_<a>v<b>.png
        conflict_overview.png  five panels, shared axes
        conflict_ranking.json  scores, c table, chosen top-2
        RANKING.md             human-readable table

Usage:
    python run_conflict_smoke_K2_mnist_pairs_without_256_checkpoints.py
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from bundle import validate_oracle_output
from baseline_svrg_certified_without_256_checkpoints import _support_batch
from run_pure_budget_K6_without_256_checkpoints import (
    MAX_SAFEGUARD_RETRIES,
    _Budget,
)
from run_experiments import _json_ready
from objectives_mnist_pair import (
    PairStochLamOracle,
    make_mnist_pair,
    make_pair_initial_point,
)

HERE = Path(__file__).resolve().parent
HOME = (HERE.parent.parent / "output" / "CCP/K2_mnist_pair_without_256_checkpoints"
        / "conflict_smoke")

PAIRS = [(4, 9), (3, 5), (7, 9), (3, 8), (5, 8)]
LAMBDAS = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]
N_SEG = 15
BATCH = 1024
N_PROBES = 10          # smoke tier; production = 40
STEP_CONST = 0.1
MOMENTUM = 0.5
INIT_SEED, SAMPLER_SEED = 8, 41
LN2 = float(np.log(2.0))


def _wiring_check(x0, joint_oracle, X_np, y_np):
    """Full-batch stochastic gradient must equal the scalarized joint
    gradient (the Aug-9 verification, K = 2 edition)."""
    lam = np.array([0.5, 0.5])
    stoch = PairStochLamOracle(X_np, y_np, batch_size=BATCH,
                               seed=SAMPLER_SEED)
    stoch.set_anchor(x0)
    g_y, g_a = stoch.grad_pair(x0, lam, stoch.full_batch())
    _, J0 = joint_oracle(x0)
    ref = J0.T @ lam
    rel = float(np.linalg.norm(g_y - ref) / max(np.linalg.norm(ref), 1e-30))
    rel_aa = float(np.linalg.norm(g_y - g_a)
                   / max(np.linalg.norm(g_y), 1e-30))
    assert rel < 1e-10 and rel_aa < 1e-12, (rel, rel_aa)
    return rel


def _run_chain(lam, x0, L_arr, joint_oracle, X_np, y_np, n, d):
    """One fixed-lambda chain from x0: N_SEG verbatim executor segments."""
    lam = np.asarray(lam, dtype=float)
    stoch = PairStochLamOracle(X_np, y_np, batch_size=BATCH,
                               seed=SAMPLER_SEED)
    budget = _Budget(2, n, stoch, 1e18)
    epoch_len = max(1, int(np.ceil(n / float(BATCH))))
    f0, J0 = validate_oracle_output(*joint_oracle(x0), 2, d)
    budget.joint_calls += 1
    chain_x, chain_J, chain_f = x0.copy(), J0, f0
    spent_after = [budget.spent()]
    fvals = [chain_f.copy()]
    L_scale, retries_here, safeguard_retries = 1.0, 0, 0
    t0 = time.time()
    for _seg in range(N_SEG):
        g_a_full = chain_J.T @ lam
        F_a = float(chain_f @ lam)
        eta = STEP_CONST / (float(lam @ L_arr) * L_scale)
        stoch.set_anchor(chain_x)
        y_vec = chain_x.copy()
        u_vec = np.zeros(d)
        for _t in range(epoch_len):
            batch = _support_batch(stoch.sample_batch(), lam)
            g_y_S, g_a_S = stoch.grad_pair(y_vec, lam, batch)
            u_vec = MOMENTUM * u_vec + (g_y_S - g_a_S + g_a_full)
            y_vec = y_vec - eta * u_vec
        f_y, J_y = validate_oracle_output(*joint_oracle(y_vec), 2, d)
        budget.joint_calls += 1
        spent_after.append(budget.spent())
        fvals.append(np.asarray(f_y, dtype=float))
        if float(f_y @ lam) > F_a + 1e-10 * (1.0 + abs(F_a)):
            L_scale *= 2.0
            safeguard_retries += 1
            retries_here += 1
            if retries_here > MAX_SAFEGUARD_RETRIES:
                chain_x, chain_J, chain_f = y_vec, J_y, f_y
                retries_here = 0
        else:
            chain_x, chain_J, chain_f = y_vec, J_y, f_y
            retries_here = 0
    seg_costs = np.diff(np.asarray(spent_after))
    return {
        "lam": [float(v) for v in lam],
        "final_f": [float(v) for v in chain_f],
        "fvals": [[float(v) for v in f] for f in fvals],
        "seg_costs": [float(v) for v in seg_costs],
        "c_mean": float(seg_costs.mean()),
        "epoch_len": epoch_len,
        "L_scale_final": L_scale,
        "safeguard_retries": safeguard_retries,
        "wall_seconds": time.time() - t0,
    }


def _cost_formula(n, epoch_len, rows_per_step):
    return epoch_len * 2.0 * rows_per_step * 2.0 / float(n) + 2.0


def _pair_scatter(ax, pair, chains, score):
    ends = np.array([c["final_f"] for c in chains])
    ax.plot(ends[:, 0], ends[:, 1], "-o", color="#1f6fb4", zorder=3)
    for c, (fa, fb) in zip(chains, ends):
        ax.annotate(f"{c['lam'][0]:g}", (fa, fb), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(LN2, ls="--", lw=0.6, color="gray")
    ax.axvline(LN2, ls="--", lw=0.6, color="gray")
    ax.set_xlabel(f"F_A (digit {pair[0]} mean CE)")
    ax.set_ylabel(f"F_B (digit {pair[1]} mean CE)")
    ax.set_title(f"{pair[0]} vs {pair[1]}  score={score:.3f}", fontsize=9)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)


def main() -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    results = []
    for pair in PAIRS:
        a, b = pair
        t_build = time.time()
        (_obj, _grad, L, joint_oracle, _stoch, meta) = make_mnist_pair(
            a, b, per_class=None, batch_size=BATCH,
            sampler_seed=SAMPLER_SEED, init_seed=INIT_SEED,
            n_probes=N_PROBES)
        X_np, y_np = meta.pop("_X"), meta.pop("_y")
        n, d = meta["n"], meta["d"]
        L_arr = np.asarray(L, dtype=float)
        x0 = make_pair_initial_point(INIT_SEED)
        rel = _wiring_check(x0, joint_oracle, X_np, y_np)
        print(f"[{a}v{b}] built in {time.time() - t_build:.1f}s "
              f"(n={n} d={d} per_class={meta['per_class']} "
              f"L=[{L_arr[0]:.3f},{L_arr[1]:.3f}] wiring rel={rel:.1e})",
              flush=True)

        chains = [_run_chain(lam, x0, L_arr, joint_oracle, X_np, y_np, n, d)
                  for lam in LAMBDAS]
        end = {tuple(c["lam"]): np.asarray(c["final_f"]) for c in chains}
        sac_A = float(end[(0.0, 1.0)][0] - end[(1.0, 0.0)][0])
        sac_B = float(end[(1.0, 0.0)][1] - end[(0.0, 1.0)][1])
        score = (sac_A + sac_B) / LN2
        sac_A_int = float(end[(0.25, 0.75)][0] - end[(0.75, 0.25)][0])
        sac_B_int = float(end[(0.75, 0.25)][1] - end[(0.25, 0.75)][1])
        score_int = (sac_A_int + sac_B_int) / LN2

        # monotone trade-off check along lam_A descending
        FA = [end[tuple(l)][0] for l in LAMBDAS]
        FB = [end[tuple(l)][1] for l in LAMBDAS]
        tol = 1e-3
        mono_viol = (sum(FA[i + 1] < FA[i] - tol for i in range(4))
                     + sum(FB[i + 1] > FB[i] + tol for i in range(4)))

        # c calibration vs the exact formula (vertex lams drop one class)
        epoch_len = chains[0]["epoch_len"]
        b_k = int(BATCH // 2)
        c_interior_formula = _cost_formula(n, epoch_len, BATCH)
        c_vertex_formula = _cost_formula(n, epoch_len, b_k)
        c_interior = float(np.mean([c["c_mean"] for c in chains[1:4]]))
        c_vertex = float(np.mean([chains[0]["c_mean"],
                                  chains[4]["c_mean"]]))
        c_ok = (abs(c_interior - c_interior_formula) < 1e-9
                and abs(c_vertex - c_vertex_formula) < 1e-9)

        rec = {"pair": [a, b], "n": n, "d": d,
               "per_class": meta["per_class"],
               "L": [float(v) for v in L_arr],
               "wiring_rel_err": rel,
               "score": score, "sac_A": sac_A, "sac_B": sac_B,
               "score_A_norm": sac_A / LN2, "score_B_norm": sac_B / LN2,
               "score_interior": score_int,
               "sac_A_interior": sac_A_int, "sac_B_interior": sac_B_int,
               "mono_violations": int(mono_viol),
               "epoch_len": epoch_len,
               "c_interior": c_interior, "c_vertex": c_vertex,
               "c_interior_formula": c_interior_formula,
               "c_vertex_formula": c_vertex_formula,
               "c_matches_formula": bool(c_ok),
               "min_end_F": float(min(min(FA), min(FB))),
               "chains": chains}
        results.append(rec)
        (HOME / f"pair_{a}v{b}.json").write_text(
            json.dumps(_json_ready(rec), indent=2), encoding="utf-8")

        fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
        _pair_scatter(ax, pair, chains, score_int)
        fig.tight_layout()
        fig.savefig(HOME / f"scatter_pair_{a}v{b}.png")
        plt.close(fig)
        print(f"[{a}v{b}] S_int={score_int:.4f} "
              f"(A {sac_A_int / LN2:.4f} + B {sac_B_int / LN2:.4f}) "
              f"| vertex diag={score:.1f} mono_viol={mono_viol} "
              f"c_int={c_interior:.3f} c_vtx={c_vertex:.3f} "
              f"formula_ok={c_ok}", flush=True)

    order = sorted(results, key=lambda r: -r["score_interior"])
    top2 = [r["pair"] for r in order[:2]]

    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6), dpi=150)
    for ax, rec in zip(axes, results):
        _pair_scatter(ax, tuple(rec["pair"]), rec["chains"], rec["score"])
    fig.tight_layout()
    fig.savefig(HOME / "conflict_overview.png")
    plt.close(fig)

    ranking = {"campaign": "K2 MNIST pair conflict smoke (Smoke A)",
               "machine": platform.platform(),
               "n_seg": N_SEG, "batch": BATCH, "n_probes": N_PROBES,
               "lambdas": LAMBDAS,
               "score_def": ("PRIMARY score_interior = [F_A((.25,.75)) - "
                             "F_A((.75,.25))] + [F_B((.75,.25)) - "
                             "F_B((.25,.75))] / ln2 (no divergence); "
                             "vertex score kept as diagnostic (dominated "
                             "by unregularised divergence of the ignored "
                             "class)"),
               "ranking": [{k: r[k] for k in
                            ("pair", "score_interior", "sac_A_interior",
                             "sac_B_interior", "score", "sac_A", "sac_B",
                             "mono_violations", "n", "per_class",
                             "c_interior", "c_vertex",
                             "c_matches_formula", "min_end_F", "L")}
                           for r in order],
               "top2": top2,
               "total_wall_seconds": time.time() - t_all}
    (HOME / "conflict_ranking.json").write_text(
        json.dumps(_json_ready(ranking), indent=2), encoding="utf-8")

    lines = ["# K2 pair conflict smoke — ranking (Aug 13, 2026)", "",
             "Ranked by S_int (interior score, the pair-discriminating "
             "ruler); the vertex score is a divergence diagnostic.", "",
             "| pair | S_int | A part | B part | vertex diag | "
             "mono viol | c int | c vtx | formula ok |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in order:
        lines.append(
            f"| {r['pair'][0]} vs {r['pair'][1]} "
            f"| {r['score_interior']:.4f} "
            f"| {r['sac_A_interior'] / LN2:.4f} "
            f"| {r['sac_B_interior'] / LN2:.4f} "
            f"| {r['score']:.1f} "
            f"| {r['mono_violations']} | {r['c_interior']:.3f} "
            f"| {r['c_vertex']:.3f} | {r['c_matches_formula']} |")
    lines += ["", f"top2 = {top2}",
              f"total wall = {ranking['total_wall_seconds']:.0f}s", ""]
    (HOME / "RANKING.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[smoke] DONE in {ranking['total_wall_seconds']:.0f}s "
          f"-> {HOME}", flush=True)
    print(f"[smoke] top2 = {top2}", flush=True)


if __name__ == "__main__":
    main()
