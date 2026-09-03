"""run_tag_affinity_K3_mnist_triples_without_256_checkpoints.py —
Phase 1 (A0 pilot + A1 full scan) of the TAG-style long-run affinity
screening plan (~/Desktop/K3_MNIST_TAG_Affinity_实验方案.docx, Aug 26,
2026).  Phase 2 (mini Smoke A on the Top 5) is deliberately NOT
implemented here — user instruction Aug 26: "只做一阶段".

What this does, per the plan:

* One neutral centroid chain per triple: lambda = (1/3,1/3,1/3), the
  VERBATIM Smoke-A executor unit (anchor full gradient + MSVRG epoch,
  momentum 0.5, eta = 0.1/(lam@L * L_scale), safeguard with
  MAX_RETRIES = 4) — the same loop that was replicated and validated
  bit-for-bit against the conflict_smoke archive on Aug 26 (reg_demo).
* At affinity checkpoints t (pilot: every segment 0..15; scan:
  {0,3,6,9,12,15}) take the chain's real (f, J) from joint_oracle and,
  for each objective i, build a THROWAWAY probe
      theta_probe = theta.copy() - (alpha / L_i) * J[i]
  evaluate the other objectives full-batch (no_grad), record
      Z[t,i,j] = 1 - F_j(theta_probe) / max(F_j(theta), eps),
  and discard the probe.  The real theta, momentum, sampler RNG and
  metering are never touched (no budget meter is used at all).
* Aggregate per direction: Zbar (plain mean), C = mean(max(0,-Z))
  (conflict strength), P = mean(1[Z<0]) (conflict persistence); per
  triple: c_j = 0.5*sum_{i!=j} C[i,j], Cmean = mean(c_j), Cbalanced =
  min(c_j).  Ranking: Cbalanced first, Cmean as tie-breaker (§2.4).

Fairness (§3.2): per_class FIXED at 5421 for every triple (all ten
digits hold >= 5421 train rows, so n = 16,263 and epoch_len = 16 —
identical grad-equivalent budget everywhere); same init seed 8, same
sampler seed 41, same minibatch stream; L_i estimated per triple with
the factory recipe at the smoke tier (10 probe pairs, seed 7);
affinity losses evaluated on the FULL training batch; official test
files are never opened.

Scopes (§5.2): --scope all (MOO-faithful, default, the main result)
updates all 8,195 parameters; --scope shared (TAG-faithful diagnostic,
for the later A2 pass) zeroes the probe step on the 3-logit head
(last 291 entries).

Usage:
    python run_tag_affinity_K3_mnist_triples_without_256_checkpoints.py \
        --stage pilot            # A0: 5 candidates, alphas .025/.05/.10
    python run_tag_affinity_K3_mnist_triples_without_256_checkpoints.py \
        --stage scan             # A1: all C(10,3)=120, alpha .05, resume-skip

Outputs under output/CCP/K3_mnist_triple_without_256_checkpoints/
tag_affinity/{pilot,scan}/: per-triple JSON, ranking.csv/json, and the
plan's §6 figures that belong to phase 1 (pilot: tag_vs_smokeA.png,
affinity_over_time + conflict matrix for 3-5-8; scan: top-1 heatmaps +
top-15 bar).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import _layout  # noqa: F401
from objectives_mnist_triple import (
    K_TRIPLE,
    TriplePatchMLP,
    TripleStochLamOracle,
    make_triple_initial_point,
)
from objectives_mnist_patch import _fetch, _read_idx
from objectives_torch import _flatten_grads, _load_theta_into_net

HERE = Path(__file__).resolve().parent
HOME = (HERE.parent.parent / "output" / "CCP"
        / "K3_mnist_triple_without_256_checkpoints" / "tag_affinity")

K = K_TRIPLE
PER_CLASS = 5421            # fixed for ALL triples (§3.2), min digit count
N_SEG = 15
BATCH = 1024
STEP_CONST = 0.1
MOMENTUM = 0.5
MAX_RETRIES = 4             # == MAX_SAFEGUARD_RETRIES (run_pure_budget_K6:86)
INIT_SEED, SAMPLER_SEED, PROBE_SEED = 8, 41, 7
N_PROBES = 10               # smoke tier
HEAD_PARAMS = 3 * 96 + 3    # trailing head block for --scope shared
EPS = 1e-12
LAM = np.full(K, 1.0 / 3.0)

PILOT_TRIPLES = [(3, 5, 8), (4, 7, 9), (3, 8, 9), (5, 6, 8), (2, 3, 8)]
S_INT_SMOKE = {(3, 5, 8): 0.8317, (4, 7, 9): 0.7855, (3, 8, 9): 0.5837,
               (5, 6, 8): 0.5268, (2, 3, 8): 0.6294}
# smoke's centroid-chain endpoint for 3-5-8 (identical data: min count 5421)
SMOKE_C_ENDPOINT_358 = np.array([0.3312, 0.2959, 0.2983])

_RAW = {}


def _train_raw():
    if not _RAW:
        _RAW["images"] = _read_idx(_fetch("train-images-idx3-ubyte.gz"))
        _RAW["labels"] = _read_idx(_fetch("train-labels-idx1-ubyte.gz")).astype(np.int64)
    return _RAW["images"], _RAW["labels"]


def build_data(triple):
    """First PER_CLASS rows of each digit in dataset order, labels 0/1/2
    (mirrors load_mnist_triple, with the plan's fixed per_class)."""
    images, labels = _train_raw()
    idx = [np.nonzero(labels == dg)[0][:PER_CLASS] for dg in triple]
    if any(i.size < PER_CLASS for i in idx):
        raise ValueError(f"triple {triple}: fewer than {PER_CLASS} rows.")
    X = np.concatenate([images[i].reshape(PER_CLASS, -1) for i in idx])
    X = np.ascontiguousarray(X.astype(np.float64) / 255.0)
    y = np.concatenate([np.full(PER_CLASS, k, dtype=np.int64)
                        for k in range(K)])
    return X, y


class Problem:
    """Full-batch evaluators on one shared net (values / joint / scalarized)."""

    def __init__(self, X_np, y_np):
        self.net = TriplePatchMLP()
        self.X = torch.from_numpy(np.ascontiguousarray(X_np))
        self.rows = [torch.from_numpy(np.nonzero(y_np == k)[0]).long()
                     for k in range(K)]
        self.d = int(sum(p.numel() for p in self.net.parameters()))

    def _losses(self, theta):
        _load_theta_into_net(self.net, np.asarray(theta, dtype=float))
        Z = self.net(self.X)
        out = []
        for k in range(K):
            Zk = Z[self.rows[k]]
            tgt = torch.full((len(self.rows[k]),), k, dtype=torch.long)
            out.append(F.cross_entropy(Zk, tgt, reduction="mean"))
        return out

    def values(self, theta):
        with torch.no_grad():
            return np.array([float(l) for l in self._losses(theta)])

    def joint(self, theta):
        losses = self._losses(theta)
        fv = np.array([float(l.detach()) for l in losses])
        Jm = np.empty((K, self.d))
        for k in range(K):
            grads = torch.autograd.grad(losses[k], list(self.net.parameters()),
                                        retain_graph=(k < K - 1))
            Jm[k] = _flatten_grads(self.net, grads)
        return fv, Jm

    def scal_grad(self, theta, lam):
        losses = self._losses(theta)
        scal = sum(float(l_) * loss for l_, loss in zip(lam, losses))
        grads = torch.autograd.grad(scal, list(self.net.parameters()))
        return _flatten_grads(self.net, grads)


def estimate_L(problem):
    """Factory recipe at the smoke tier: 10 random parameter pairs,
    L_i = max ||dJ_i|| / ||dtheta||."""
    rng = np.random.RandomState(PROBE_SEED)
    L = np.zeros(K)
    for _ in range(N_PROBES):
        t1 = make_triple_initial_point(rng.randint(1 << 30)) \
            + 0.5 * rng.randn(problem.d) * 0.1
        t2 = t1 + 0.5 * rng.randn(problem.d)
        _, J1 = problem.joint(t1)
        _, J2 = problem.joint(t2)
        denom = float(np.linalg.norm(t2 - t1))
        L = np.maximum(L, np.linalg.norm(J2 - J1, axis=1) / denom)
    return L


def probe_Z(problem, theta, fv, Jm, L, alphas, scope):
    """The plan's §2.2 lookahead at one checkpoint.  Probe arrays are
    local copies, dropped on return — theta is never modified."""
    out = {}
    for alpha in alphas:
        Zm = np.zeros((K, K))
        for i in range(K):
            step = Jm[i].copy()
            if scope == "shared":
                step[-HEAD_PARAMS:] = 0.0
            theta_probe = theta - (alpha / L[i]) * step
            f_probe = problem.values(theta_probe)
            for j in range(K):
                if i != j:
                    Zm[i, j] = 1.0 - f_probe[j] / max(fv[j], EPS)
        out[alpha] = Zm
    return out


def centroid_chain(problem, X_np, y_np, L, alphas, ckpts, scope):
    """Verbatim Smoke-A executor unit for lambda=(1/3,1/3,1/3), with
    affinity measurements at ckpts.  Returns (Z-series, chain endpoint)."""
    stoch = TripleStochLamOracle(X_np, y_np, batch_size=BATCH,
                                 seed=SAMPLER_SEED)
    n = X_np.shape[0]
    epoch_len = max(1, int(np.ceil(n / float(BATCH))))
    x0 = make_triple_initial_point(INIT_SEED)
    fv, Jm = problem.joint(x0)
    chain_x, chain_f, chain_J = x0.copy(), fv, Jm
    series = {a: [] for a in alphas}
    if 0 in ckpts:
        for a, Zm in probe_Z(problem, chain_x, chain_f, chain_J, L,
                             alphas, scope).items():
            series[a].append((0, Zm))
    L_scale, retries_here = 1.0, 0
    for seg in range(1, N_SEG + 1):
        g_a_full = chain_J.T @ LAM if chain_J is not None \
            else problem.scal_grad(chain_x, LAM)
        F_a = float(chain_f @ LAM)
        eta = STEP_CONST / (float(LAM @ L) * L_scale)
        stoch.set_anchor(chain_x)
        y_vec = chain_x.copy()
        u_vec = np.zeros(problem.d)
        for _t in range(epoch_len):
            batch = stoch.sample_batch()      # centroid: full support
            g_y_S, g_a_S = stoch.grad_pair(y_vec, LAM, batch)
            u_vec = MOMENTUM * u_vec + (g_y_S - g_a_S + g_a_full)
            y_vec = y_vec - eta * u_vec
        need_joint = seg in ckpts or seg == N_SEG
        if need_joint:
            f_y, J_y = problem.joint(y_vec)
        else:
            f_y, J_y = problem.values(y_vec), None
        if float(f_y @ LAM) > F_a + 1e-10 * (1.0 + abs(F_a)):
            L_scale *= 2.0
            retries_here += 1
            if retries_here > MAX_RETRIES:
                chain_x, chain_f, chain_J = y_vec, f_y, J_y
                retries_here = 0
        else:
            chain_x, chain_f, chain_J = y_vec, f_y, J_y
            retries_here = 0
        if seg in ckpts and chain_x is y_vec:
            for a, Zm in probe_Z(problem, chain_x, chain_f, chain_J, L,
                                 alphas, scope).items():
                series[a].append((seg, Zm))
        if chain_J is None and seg < N_SEG:
            chain_J = None  # scal_grad path next segment
    return series, chain_f


def aggregate(series_a):
    """Zbar / C / P matrices and the §2.4 ranking scores for one alpha."""
    stack = np.stack([Zm for _t, Zm in series_a])          # (T, 3, 3)
    Zbar = stack.mean(axis=0)
    Cm = np.maximum(0.0, -stack).mean(axis=0)
    Pm = (stack < 0).mean(axis=0)
    for M in (Zbar, Cm, Pm):
        np.fill_diagonal(M, 0.0)
    c_j = 0.5 * Cm.sum(axis=0)                             # incoming conflict
    return {"Zbar": Zbar.tolist(), "C": Cm.tolist(), "P": Pm.tolist(),
            "c_j": c_j.tolist(), "Cmean": float(c_j.mean()),
            "Cbalanced": float(c_j.min()),
            "checkpoints": [int(t) for t, _ in series_a]}


def run_triple(triple, alphas, ckpts, scope, out_dir):
    t0 = time.time()
    X_np, y_np = build_data(triple)
    problem = Problem(X_np, y_np)
    L = estimate_L(problem)
    series, endpoint = centroid_chain(problem, X_np, y_np, L, alphas,
                                      ckpts, scope)
    rec = {"triple": list(triple), "per_class": PER_CLASS,
           "n": int(X_np.shape[0]), "scope": scope,
           "L": [float(v) for v in L],
           "centroid_endpoint_f": [float(v) for v in endpoint],
           "alphas": {}}
    for a in alphas:
        rec["alphas"][f"{a:g}"] = aggregate(series[a])
        rec["alphas"][f"{a:g}"]["Z_series"] = [
            {"t": int(t), "Z": Zm.tolist()} for t, Zm in series[a]]
    rec["wall_seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"triple_{triple[0]}v{triple[1]}v{triple[2]}.json"
    path.write_text(json.dumps(rec, indent=1))
    return rec


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra @ ra) * (rb @ rb)))
    return float(ra @ rb / den) if den else 0.0


def heatmap(ax, M, title, cmap, labels):
    M = np.array(M, dtype=float)
    Mp = M.copy(); np.fill_diagonal(Mp, np.nan)
    im = ax.imshow(Mp, cmap=cmap)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"→{l}" for l in labels], fontsize=8)
    ax.set_yticklabels([f"{l}→" for l in labels], fontsize=8)
    for i in range(K):
        for j in range(K):
            if i != j:
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                        fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


def write_ranking(records, alpha_key, out_dir, csv_name="ranking"):
    rows = []
    for rec in records:
        agg = rec["alphas"][alpha_key]
        rows.append({"triple": "-".join(map(str, rec["triple"])),
                     "Cbalanced": agg["Cbalanced"], "Cmean": agg["Cmean"],
                     "c_j": agg["c_j"],
                     "wall_seconds": rec["wall_seconds"]})
    rows.sort(key=lambda r: (-r["Cbalanced"], -r["Cmean"]))
    (out_dir / f"{csv_name}.json").write_text(json.dumps(rows, indent=1))
    with open(out_dir / f"{csv_name}.csv", "w") as fh:
        fh.write("rank,triple,Cbalanced,Cmean,c_0,c_1,c_2,wall_seconds\n")
        for r_i, r in enumerate(rows, 1):
            fh.write(f"{r_i},{r['triple']},{r['Cbalanced']:.6f},"
                     f"{r['Cmean']:.6f},"
                     + ",".join(f"{c:.6f}" for c in r["c_j"])
                     + f",{r['wall_seconds']}\n")
    return rows


def stage_pilot(args):
    out = HOME / "pilot"
    alphas = [0.025, 0.05, 0.10]
    ckpts = set(range(0, N_SEG + 1))
    records = []
    for triple in PILOT_TRIPLES:
        rec = run_triple(tuple(triple), alphas, ckpts, args.scope, out)
        agg = rec["alphas"]["0.05"]
        print(f"[pilot] {triple}: Cbalanced={agg['Cbalanced']:.4f} "
              f"Cmean={agg['Cmean']:.4f} c_j={np.round(agg['c_j'], 4)} "
              f"L={np.round(rec['L'], 3)} ({rec['wall_seconds']}s)",
              flush=True)
        if tuple(triple) == (3, 5, 8):
            diff = np.abs(np.array(rec["centroid_endpoint_f"])
                          - SMOKE_C_ENDPOINT_358).max()
            print(f"[pilot] 3-5-8 centroid endpoint "
                  f"{np.round(rec['centroid_endpoint_f'], 4)} vs smoke "
                  f"{SMOKE_C_ENDPOINT_358} (max diff {diff:.2e}) "
                  f"{'OK' if diff < 5e-4 else 'MISMATCH'}", flush=True)
    # ---- step-size linearity (§11.2): |Z| should ~double with alpha ----
    ratios = {}
    for lo, hi in ((0.025, 0.05), (0.05, 0.10)):
        num, den = [], []
        for rec in records or []:
            pass
    # (recompute from files to keep it simple)
    recs = [json.loads((out / f"triple_{a}v{b}v{c}.json").read_text())
            for a, b, c in PILOT_TRIPLES]
    records = recs
    for lo, hi in ((0.025, 0.05), (0.05, 0.10)):
        r_all = []
        for rec in recs:
            Zlo = np.array([np.abs(np.array(s["Z"]))
                            for s in rec["alphas"][f"{lo:g}"]["Z_series"]])
            Zhi = np.array([np.abs(np.array(s["Z"]))
                            for s in rec["alphas"][f"{hi:g}"]["Z_series"]])
            mask = Zlo > 1e-6
            r_all.append(float(np.median(Zhi[mask] / Zlo[mask])))
        ratios[f"{hi:g}/{lo:g}"] = r_all
    print(f"[pilot] |Z| ratio medians (expect ~2 if lookahead is local): "
          f"{ {k: list(np.round(v, 2)) for k, v in ratios.items()} }",
          flush=True)
    # ---- ranking + G5 direction vs S_int, per alpha ----
    summary = {"ratios": ratios, "alphas": {}}
    for a in alphas:
        rows = write_ranking(recs, f"{a:g}", out, csv_name=f"ranking_a{a:g}")
        tag = [r["Cbalanced"] for rec in recs
               for r in [rec["alphas"][f"{a:g}"]]]
        s_int = [S_INT_SMOKE[tuple(rec["triple"])] for rec in recs]
        rho = spearman(np.array(tag), np.array(s_int))
        summary["alphas"][f"{a:g}"] = {
            "order": [r["triple"] for r in rows], "spearman_vs_S_int": rho}
        print(f"[pilot] alpha={a:g}: order={[r['triple'] for r in rows]} "
              f"Spearman(TAG, S_int)={rho:+.3f}", flush=True)
    # cross-alpha rank stability (G3 ingredient)
    cb = {a: np.array([rec["alphas"][f"{a:g}"]["Cbalanced"] for rec in recs])
          for a in alphas}
    summary["rank_stability_rho"] = {
        "0.05_vs_0.025": spearman(cb[0.05], cb[0.025]),
        "0.10_vs_0.05": spearman(cb[0.10], cb[0.05])}
    print(f"[pilot] rank stability rho: {summary['rank_stability_rho']}",
          flush=True)
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=1))
    # ---- figures: tag_vs_smokeA + over-time/conflict matrix for 3-5-8 ----
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=150)
    for rec in recs:
        t = tuple(rec["triple"])
        x, y = S_INT_SMOKE[t], rec["alphas"]["0.05"]["Cbalanced"]
        ax.scatter(x, y, s=28, color="#1f6fb4")
        ax.annotate("-".join(map(str, t)), (x, y), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Smoke A S_int"); ax.set_ylabel("TAG Cbalanced (a=0.05)")
    rho = summary["alphas"]["0.05"]["spearman_vs_S_int"]
    ax.set_title(f"TAG vs Smoke A on the 5 candidates (rho={rho:+.2f})",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out / "tag_vs_smokeA.png"); plt.close(fig)

    rec358 = next(r for r in recs if tuple(r["triple"]) == (3, 5, 8))
    labels = [str(d) for d in rec358["triple"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=150)
    ts = [s["t"] for s in rec358["alphas"]["0.05"]["Z_series"]]
    for i in range(K):
        for j in range(K):
            if i != j:
                zs = [s["Z"][i][j]
                      for s in rec358["alphas"]["0.05"]["Z_series"]]
                axes[0].plot(ts, zs, marker=".", ms=3, lw=1,
                             label=f"{labels[i]}→{labels[j]}")
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_xlabel("segment"); axes[0].set_ylabel("Z (a=0.05)")
    axes[0].set_title("3-5-8: affinity over time", fontsize=9)
    axes[0].legend(fontsize=6, ncol=2)
    im = heatmap(axes[1], rec358["alphas"]["0.05"]["C"],
                 "3-5-8: conflict C (a=0.05)", "Reds", labels)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "pilot_358_affinity.png"); plt.close(fig)
    print(f"[pilot] outputs in {out}", flush=True)


def stage_scan(args):
    out = HOME / "scan"
    out.mkdir(parents=True, exist_ok=True)
    alphas = [0.05]
    ckpts = {0, 3, 6, 9, 12, 15}
    triples = list(itertools.combinations(range(10), 3))
    t0 = time.time()
    recs = []
    for n_i, triple in enumerate(triples, 1):
        path = out / f"triple_{triple[0]}v{triple[1]}v{triple[2]}.json"
        if path.exists():
            recs.append(json.loads(path.read_text()))
            continue
        rec = run_triple(triple, alphas, ckpts, args.scope, out)
        recs.append(rec)
        agg = rec["alphas"]["0.05"]
        print(f"[scan {n_i:3d}/120] {triple} Cbalanced={agg['Cbalanced']:.4f} "
              f"Cmean={agg['Cmean']:.4f} ({rec['wall_seconds']}s, "
              f"elapsed {int(time.time()-t0)}s)", flush=True)
    rows = write_ranking(recs, "0.05", out)
    print("[scan] top 10 by Cbalanced:", flush=True)
    for r in rows[:10]:
        print(f"   {r['triple']}: Cbalanced={r['Cbalanced']:.4f} "
              f"Cmean={r['Cmean']:.4f} c_j={np.round(r['c_j'], 4)}",
              flush=True)
    # figures: top-1 heatmaps + top-15 bar
    top = rows[0]["triple"]
    rec = next(r for r in recs
               if "-".join(map(str, r["triple"])) == top)
    labels = [str(d) for d in rec["triple"]]
    agg = rec["alphas"]["0.05"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=150)
    im0 = heatmap(axes[0], agg["Zbar"], f"{top}: mean affinity Z̄",
                  "RdBu_r", labels)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = heatmap(axes[1], agg["C"], f"{top}: conflict C", "Reds", labels)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "affinity_matrix_top1.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=150)
    names = [r["triple"] for r in rows[:15]]
    vals = [r["Cbalanced"] for r in rows[:15]]
    colors = ["#c23b3b" if n == "3-5-8" else "#1f6fb4" for n in names]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, fontsize=7)
    ax.set_ylabel("Cbalanced (a=0.05)")
    ax.set_title("TAG scan: top 15 of 120 triples", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "ranking_top15.png"); plt.close(fig)
    print(f"[scan] outputs in {out} "
          f"(total {int(time.time()-t0)}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "scan"], required=True)
    ap.add_argument("--scope", choices=["all", "shared"], default="all")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    (stage_pilot if args.stage == "pilot" else stage_scan)(args)


if __name__ == "__main__":
    main()
