"""run_conflict_smoke_K3_mnist_triples_without_256_checkpoints.py —
Smoke A for the K = 3 MNIST triple campaign: rank 5 candidate digit
triples by objective-conflict strength and calibrate the per-segment
grad-equivalent cost c at all three support sizes.

NEW FILE (Aug 26, 2026).  User-approved design (Aug-25/26 Q&A), the
K = 2 Smoke A (Aug 13) carried to triples:

* triples: 3-5-8, 4-7-9, 3-8-9, 5-6-8, 2-3-8 (built from the pairs
  the K = 2 smoke measured as most conflicting: 3-5 .90, 7-9 .84,
  3-8 .81, 4-9 .73, 5-8 .72 — plus classic MNIST confusions);
* per triple: 10 FIXED lambdas, each an INDEPENDENT chain from the
  same He x0 (no snake hand-off — the probe asks "where does this
  lambda alone go"), each chain gets a fresh sampler with the same
  seed so every lambda sees identical minibatch streams; N_SEG = 15
  segments per chain.  The 10 lambdas, by group:
  - 3 VERTICES (1,0,0)-perm: divergence diagnostic (TWO ignored
    classes diverge — the K = 2 lesson says never score these) and
    the c(h=1) cost measurement;
  - 3 EDGE MIDPOINTS (.5,.5,0)-perm: K = 3-specific — c(h=2) cost
    measurement (simplex-grid baselines put many nodes on edges) and
    the single-ignored-class divergence diagnostic;
  - 3 FAVOR points (.6,.2,.2)-perm: the SCORING chains — all weights
    strictly positive (nothing diverges), favoured:starved = 3:1
    exactly as the K = 2 interior points (.75,.25);
  - 1 CENTROID (1/3,1/3,1/3): shape-check anchor, not scored;
* segment loop is the verbatim pure-budget executor unit (anchor full
  gradient + MSVRG epoch + safeguard); metering via _Budget with an
  unreachable limit — c is read off consecutive spent() diffs and
  asserted per chain against c = K + epoch_len*2*rows_support*K/n;
* interior conflict score, normalised by ln 3 (the guess-level CE of
  a balanced 3-class task):
      part_k = mean_{j != k} F_k(end of favor-j) - F_k(end of favor-k)
      S_int  = (part_0 + part_1 + part_2) / ln 3
  i.e. per axis: how much class k's CE rises when down-weighted (0.2)
  vs favoured (0.6) — the interior front's extent along that axis.
  The vertex analogue is kept as a divergence diagnostic only;
* shape check (the K = 2 monotonicity check generalised): for every
  axis k require F_k(favor-k) <= F_k(centroid) <= F_k(favor-j), j != k
  (tol 1e-3); violations counted, 9 comparisons per triple;
* data: per_class=None -> balanced maximum; batch stays 1024
  (stratified ~342/341/341), epoch_len follows ceil(n/b);
* L probes at the smoke tier (10 pairs; production uses 40).

Outputs (new campaign home):
    output/CCP/K3_mnist_triple_without_256_checkpoints/conflict_smoke/
        triple_<a>v<b>v<c>.json     per-lambda trajectories, c, checks
        scatter_triple_<a>v<b>v<c>.png   2x3: full + zoomed projections
        conflict_overview.png       5 triples x 3 zoomed projections
        conflict_ranking.json       scores, c table, chosen top-1
        RANKING.md                  human-readable table

Usage:
    python run_conflict_smoke_K3_mnist_triples_without_256_checkpoints.py
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
from objectives_mnist_triple import (
    TripleStochLamOracle,
    make_mnist_triple,
    make_triple_initial_point,
)

HERE = Path(__file__).resolve().parent
HOME = (HERE.parent.parent / "output"
        / "CCP/K3_mnist_triple_without_256_checkpoints" / "conflict_smoke")

TRIPLES = [(3, 5, 8), (4, 7, 9), (3, 8, 9), (5, 6, 8), (2, 3, 8)]
THIRD = 1.0 / 3.0
CHAIN_SPECS = [  # (label, lambda, group) — order fixed, indices relied on
    ("V0", (1.0, 0.0, 0.0), "vertex"),
    ("V1", (0.0, 1.0, 0.0), "vertex"),
    ("V2", (0.0, 0.0, 1.0), "vertex"),
    ("E01", (0.5, 0.5, 0.0), "edge"),
    ("E02", (0.5, 0.0, 0.5), "edge"),
    ("E12", (0.0, 0.5, 0.5), "edge"),
    ("F0", (0.6, 0.2, 0.2), "favor"),
    ("F1", (0.2, 0.6, 0.2), "favor"),
    ("F2", (0.2, 0.2, 0.6), "favor"),
    ("C", (THIRD, THIRD, THIRD), "centroid"),
]
K = 3
N_SEG = 15
BATCH = 1024
N_PROBES = 10          # smoke tier; production = 40
STEP_CONST = 0.1
MOMENTUM = 0.5
INIT_SEED, SAMPLER_SEED = 8, 41
LN3 = float(np.log(3.0))
GROUP_STYLE = {"vertex": dict(color="#d62728", marker="x"),
               "edge": dict(color="#ff9f1c", marker="s"),
               "favor": dict(color="#1f6fb4", marker="o"),
               "centroid": dict(color="#2ca02c", marker="^")}


def _wiring_check(x0, joint_oracle, X_np, y_np):
    """Full-batch stochastic gradient must equal the scalarized joint
    gradient (the Aug-9 verification, K = 3 edition, centroid lambda)."""
    lam = np.full(K, THIRD)
    stoch = TripleStochLamOracle(X_np, y_np, batch_size=BATCH,
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


def _run_chain(label, lam, x0, L_arr, joint_oracle, X_np, y_np, n, d):
    """One fixed-lambda chain from x0: N_SEG verbatim executor segments."""
    lam = np.asarray(lam, dtype=float)
    stoch = TripleStochLamOracle(X_np, y_np, batch_size=BATCH,
                                 seed=SAMPLER_SEED)
    budget = _Budget(K, n, stoch, 1e18)
    epoch_len = max(1, int(np.ceil(n / float(BATCH))))
    rows_support = int(sum(int(stoch.b_k[i]) for i in range(K)
                           if lam[i] > 0.0))
    f0, J0 = validate_oracle_output(*joint_oracle(x0), K, d)
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
        f_y, J_y = validate_oracle_output(*joint_oracle(y_vec), K, d)
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
        "label": label,
        "lam": [float(v) for v in lam],
        "support": int(np.count_nonzero(lam > 0.0)),
        "rows_support": rows_support,
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
    return epoch_len * 2.0 * rows_per_step * K / float(n) + K


PROJ = [(0, 1), (1, 2), (0, 2)]


def _triple_panels(axes_row, triple, chains, score, zoom):
    ends = np.array([c["final_f"] for c in chains])
    for ax, (i, j) in zip(axes_row, PROJ):
        for c, end in zip(chains, ends):
            group = next(g for lbl, _l, g in CHAIN_SPECS
                         if lbl == c["label"])
            ax.plot([end[i]], [end[j]], ls="", ms=5, zorder=3,
                    **GROUP_STYLE[group])
            ax.annotate(c["label"], (end[i], end[j]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(LN3, ls="--", lw=0.6, color="gray")
        ax.axvline(LN3, ls="--", lw=0.6, color="gray")
        ax.set_xlabel(f"F_{i} (digit {triple[i]} mean CE)", fontsize=8)
        ax.set_ylabel(f"F_{j} (digit {triple[j]} mean CE)", fontsize=8)
        if zoom:
            ax.set_xlim(0.0, 1.15 * LN3)
            ax.set_ylim(0.0, 1.15 * LN3)
        else:
            ax.set_xlim(left=0.0)
            ax.set_ylim(bottom=0.0)
        ax.set_title(f"{'zoom' if zoom else 'full'}  "
                     f"S_int={score:.3f}", fontsize=8)


def main() -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    results = []
    for triple in TRIPLES:
        a, b, c_dg = triple
        tag = f"{a}v{b}v{c_dg}"
        t_build = time.time()
        (_obj, _grad, L, joint_oracle, _stoch, meta) = make_mnist_triple(
            triple, per_class=None, batch_size=BATCH,
            sampler_seed=SAMPLER_SEED, init_seed=INIT_SEED,
            n_probes=N_PROBES)
        X_np, y_np = meta.pop("_X"), meta.pop("_y")
        n, d = meta["n"], meta["d"]
        L_arr = np.asarray(L, dtype=float)
        x0 = make_triple_initial_point(INIT_SEED)
        rel = _wiring_check(x0, joint_oracle, X_np, y_np)
        print(f"[{tag}] built in {time.time() - t_build:.1f}s "
              f"(n={n} d={d} per_class={meta['per_class']} "
              f"L=[{L_arr[0]:.3f},{L_arr[1]:.3f},{L_arr[2]:.3f}] "
              f"wiring rel={rel:.1e})", flush=True)

        chains = [_run_chain(lbl, lam, x0, L_arr, joint_oracle,
                             X_np, y_np, n, d)
                  for lbl, lam, _g in CHAIN_SPECS]
        by_label = {c["label"]: np.asarray(c["final_f"]) for c in chains}

        # interior score (favor chains only; all-positive lambdas)
        fav = [by_label[f"F{k}"] for k in range(K)]
        cen = by_label["C"]
        parts = [float(np.mean([fav[j][k] for j in range(K) if j != k])
                       - fav[k][k]) for k in range(K)]
        score_int = float(sum(parts) / LN3)

        # vertex analogue — divergence diagnostic only
        vtx = [by_label[f"V{k}"] for k in range(K)]
        vparts = [float(np.mean([vtx[j][k] for j in range(K) if j != k])
                        - vtx[k][k]) for k in range(K)]
        score_vertex = float(sum(vparts) / LN3)

        # shape check: F_k(favor-k) <= F_k(centroid) <= F_k(favor-j)
        tol = 1e-3
        shape_viol = 0
        for k in range(K):
            shape_viol += int(cen[k] < fav[k][k] - tol)
            shape_viol += sum(int(fav[j][k] < cen[k] - tol)
                              for j in range(K) if j != k)

        # divergence diagnostics (ignored-class CE at zero-weight ends)
        vertex_diverge_max = float(max(vtx[k][j] for k in range(K)
                                       for j in range(K) if j != k))
        edge_diverge_max = float(max(
            by_label[c["label"]][c["lam"].index(0.0)]
            for c in chains if c["label"].startswith("E")))

        # c calibration vs the exact formula, per chain (support-aware)
        epoch_len = chains[0]["epoch_len"]
        c_ok = True
        for c in chains:
            expected = _cost_formula(n, epoch_len, c["rows_support"])
            c["c_formula"] = expected
            c_ok = c_ok and abs(c["c_mean"] - expected) < 1e-9
        c_vertex = float(np.mean([c["c_mean"] for c in chains[0:3]]))
        c_edge = float(np.mean([c["c_mean"] for c in chains[3:6]]))
        c_interior = float(np.mean([c["c_mean"] for c in chains[6:10]]))

        rec = {"triple": list(triple), "n": n, "d": d,
               "per_class": meta["per_class"],
               "L": [float(v) for v in L_arr],
               "b_k": [int(v) for v in
                       TripleStochLamOracle(X_np, y_np, batch_size=BATCH,
                                            seed=SAMPLER_SEED).b_k],
               "wiring_rel_err": rel,
               "score_interior": score_int, "parts_interior": parts,
               "score_vertex": score_vertex, "parts_vertex": vparts,
               "shape_violations": int(shape_viol),
               "vertex_diverge_max": vertex_diverge_max,
               "edge_diverge_max": edge_diverge_max,
               "epoch_len": epoch_len,
               "c_vertex": c_vertex, "c_edge": c_edge,
               "c_interior": c_interior,
               "c_matches_formula": bool(c_ok),
               "chains": chains}
        results.append(rec)
        (HOME / f"triple_{tag}.json").write_text(
            json.dumps(_json_ready(rec), indent=2), encoding="utf-8")

        fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), dpi=150)
        _triple_panels(axes[0], triple, chains, score_int, zoom=False)
        _triple_panels(axes[1], triple, chains, score_int, zoom=True)
        fig.suptitle(f"{a} vs {b} vs {c_dg} — S_int={score_int:.4f}",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(HOME / f"scatter_triple_{tag}.png")
        plt.close(fig)
        print(f"[{tag}] S_int={score_int:.4f} "
              f"(parts {parts[0]:.3f}/{parts[1]:.3f}/{parts[2]:.3f}) "
              f"| vertex diag={score_vertex:.1f} shape_viol={shape_viol} "
              f"| c int={c_interior:.3f} edge={c_edge:.3f} "
              f"vtx={c_vertex:.3f} formula_ok={c_ok} "
              f"| diverge vtx={vertex_diverge_max:.1f} "
              f"edge={edge_diverge_max:.1f}", flush=True)

    order = sorted(results, key=lambda r: -r["score_interior"])
    top1 = order[0]["triple"]

    fig, axes = plt.subplots(len(TRIPLES), 3, figsize=(12.6, 3.5 * 5),
                             dpi=150)
    for row, rec in zip(axes, results):
        _triple_panels(row, tuple(rec["triple"]), rec["chains"],
                       rec["score_interior"], zoom=True)
        row[0].set_ylabel(f"{rec['triple']}\n" + row[0].get_ylabel(),
                          fontsize=8)
    fig.suptitle("K3 triple conflict smoke — zoomed projections",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(HOME / "conflict_overview.png")
    plt.close(fig)

    ranking = {"campaign": "K3 MNIST triple conflict smoke (Smoke A)",
               "machine": platform.platform(),
               "n_seg": N_SEG, "batch": BATCH, "n_probes": N_PROBES,
               "chain_specs": [{"label": lbl, "lam": list(lam),
                                "group": g} for lbl, lam, g in CHAIN_SPECS],
               "score_def": ("PRIMARY score_interior: part_k = "
                             "mean_{j!=k} F_k(favor-j end) - "
                             "F_k(favor-k end), S_int = sum(parts)/ln3; "
                             "favor lambdas (.6,.2,.2)-perm keep every "
                             "class alive (3:1 as K=2's (.75,.25)). "
                             "Vertex analogue kept as diagnostic only "
                             "(unregularised ignored classes diverge). "
                             "Shape check: F_k(favor-k) <= F_k(centroid) "
                             "<= F_k(favor-j), 9 comparisons, tol 1e-3."),
               "ranking": [{key: r[key] for key in
                            ("triple", "score_interior", "parts_interior",
                             "score_vertex", "shape_violations",
                             "vertex_diverge_max", "edge_diverge_max",
                             "n", "per_class", "epoch_len", "b_k",
                             "c_interior", "c_edge", "c_vertex",
                             "c_matches_formula", "L")}
                           for r in order],
               "top1": top1,
               "total_wall_seconds": time.time() - t_all}
    (HOME / "conflict_ranking.json").write_text(
        json.dumps(_json_ready(ranking), indent=2), encoding="utf-8")

    lines = ["# K3 triple conflict smoke — ranking (Aug 26, 2026)", "",
             "Ranked by S_int (interior score over the favor chains, "
             "ln-3 normalised); the vertex score is a divergence "
             "diagnostic.  Campaign decision: TOP-1 only (user call, "
             "Aug 26).", "",
             "| triple | S_int | part_0 | part_1 | part_2 | vertex diag "
             "| shape viol | c int | c edge | c vtx | formula ok |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in order:
        t = r["triple"]
        p = r["parts_interior"]
        lines.append(
            f"| {t[0]}-{t[1]}-{t[2]} | {r['score_interior']:.4f} "
            f"| {p[0] / LN3:.4f} | {p[1] / LN3:.4f} | {p[2] / LN3:.4f} "
            f"| {r['score_vertex']:.1f} | {r['shape_violations']} "
            f"| {r['c_interior']:.3f} | {r['c_edge']:.3f} "
            f"| {r['c_vertex']:.3f} | {r['c_matches_formula']} |")
    lines += ["", f"top1 = {top1}",
              f"total wall = {ranking['total_wall_seconds']:.0f}s", ""]
    (HOME / "RANKING.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[smoke] DONE in {ranking['total_wall_seconds']:.0f}s "
          f"-> {HOME}", flush=True)
    print(f"[smoke] top1 = {top1}", flush=True)


if __name__ == "__main__":
    main()
