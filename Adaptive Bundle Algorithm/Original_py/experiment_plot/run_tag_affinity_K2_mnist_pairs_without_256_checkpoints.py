"""run_tag_affinity_K2_mnist_pairs_without_256_checkpoints.py — S0 of the
K = 2 pair campaign (Sep 2, 2026): TAG-style lookahead affinity scan of
ALL C(10,2) = 45 MNIST digit pairs, the K = 2 port of the Aug-26 K = 3
Phase-1 scan (run_tag_affinity_K3_mnist_triples_without_256_checkpoints).

Protocol (verbatim K3 A1 unless noted):

* One neutral centroid chain per pair: lambda = (1/2, 1/2), the Smoke-A
  executor unit (anchor full gradient + MSVRG epoch, momentum 0.5,
  eta = 0.1/(lam@L * L_scale), safeguard with MAX_RETRIES = 4).
* Lookahead probes are taken at EVERY segment checkpoint t in 0..15
  (K3 pilot cadence).  For each objective i the THROWAWAY probe is
      theta_probe = theta.copy() - (alpha / L_i) * J[i],  alpha = 0.05,
  the other objective is evaluated full-batch (no_grad), and
      Z[t,i,j] = 1 - F_j(theta_probe) / max(F_j(theta), eps)
  is recorded; the probe is discarded (chain state never touched).
* Aggregation (K3 §2.4, K = 2): C = mean_t max(0, -Z),
  c_j = 0.5 * sum_{i!=j} C[i,j], Cbalanced = min(c_j), Cmean = mean(c_j).
  Ranking: Cbalanced first, Cmean tie-break, MOST conflicting on top.
* TWO checkpoint grids from the same run (grid-stability check baked in,
  after the Phase-1 lesson that triple order flipped with the grid):
  - scan grid {0,3,6,9,12,15} — the K3 A1 cadence, the HEADLINE ranking;
  - full grid 0..15 — the K3 pilot cadence, the stability variant.
* Fairness: per_class FIXED at 5421 for every pair (n = 10,842,
  epoch_len = 11); same init seed 8, sampler seed 41; L_i per pair with
  the factory recipe at the smoke tier (10 probe pairs, seed 7);
  affinity losses on the FULL training batch; test files never opened.
* Cross-check: the Aug-13 conflict-smoke ranking (5 candidate pairs,
  output/CCP/K2_mnist_pair_without_256_checkpoints/conflict_smoke/) is
  reported next to the TAG ranks of the same pairs.

Scope is "all" only (MOO-faithful; the K3 shared-scope diagnostic was
never part of Phase 1's main result).  Resume-skip: existing per-pair
JSONs are not recomputed.

Usage:
    python run_tag_affinity_K2_mnist_pairs_without_256_checkpoints.py

Outputs under output/CCP/K2_mnist_pair_without_256_checkpoints/
tag_affinity/scan/: per-pair JSON, ranking_scan.{csv,json} (headline),
ranking_full.{csv,json}, stability_crosscheck.json, RANKING.md,
top15_cbalanced.png, grid_stability_scatter.png.
"""

from __future__ import annotations

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
from objectives_mnist_pair import (
    PairPatchMLP,
    PairStochLamOracle,
    make_pair_initial_point,
)
from objectives_mnist_patch import _fetch, _read_idx
from objectives_torch import _flatten_grads, _load_theta_into_net

HERE = Path(__file__).resolve().parent
HOME = (HERE.parent.parent / "output" / "CCP"
        / "K2_mnist_pair_without_256_checkpoints" / "tag_affinity")
CONFLICT_SMOKE_DIR = (HERE.parent.parent / "output" / "CCP"
                      / "K2_mnist_pair_without_256_checkpoints"
                      / "conflict_smoke")

K = 2
PER_CLASS = 5421            # fixed for ALL pairs, min digit count
N_SEG = 15
BATCH = 1024
STEP_CONST = 0.1
MOMENTUM = 0.5
MAX_RETRIES = 4
INIT_SEED, SAMPLER_SEED, PROBE_SEED = 8, 41, 7
N_PROBES = 10               # smoke tier
EPS = 1e-12
LAM = np.full(K, 1.0 / K)
ALPHA = 0.05                # the validated A1 scan value
CKPTS_FULL = list(range(0, N_SEG + 1))       # probes taken at all of these
SCAN_GRID = [0, 3, 6, 9, 12, 15]             # headline ranking grid

ALL_PAIRS = list(itertools.combinations(range(10), 2))

_RAW = {}


def _train_raw():
    if not _RAW:
        _RAW["images"] = _read_idx(_fetch("train-images-idx3-ubyte.gz"))
        _RAW["labels"] = _read_idx(
            _fetch("train-labels-idx1-ubyte.gz")).astype(np.int64)
    return _RAW["images"], _RAW["labels"]


def build_data(pair):
    """First PER_CLASS rows of each digit in dataset order, labels 0/1
    (mirrors load_mnist_pair, with the plan's fixed per_class)."""
    images, labels = _train_raw()
    idx = [np.nonzero(labels == dg)[0][:PER_CLASS] for dg in pair]
    if any(i.size < PER_CLASS for i in idx):
        raise ValueError(f"pair {pair}: fewer than {PER_CLASS} rows.")
    X = np.concatenate([images[i].reshape(PER_CLASS, -1) for i in idx])
    X = np.ascontiguousarray(X.astype(np.float64) / 255.0)
    y = np.concatenate([np.full(PER_CLASS, k, dtype=np.int64)
                        for k in range(K)])
    return X, y


class Problem:
    """Full-batch evaluators on one shared pair net."""

    def __init__(self, X_np, y_np):
        self.net = PairPatchMLP()
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
            grads = torch.autograd.grad(losses[k],
                                        list(self.net.parameters()),
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
        t1 = make_pair_initial_point(rng.randint(1 << 30)) \
            + 0.5 * rng.randn(problem.d) * 0.1
        t2 = t1 + 0.5 * rng.randn(problem.d)
        _, J1 = problem.joint(t1)
        _, J2 = problem.joint(t2)
        denom = float(np.linalg.norm(t2 - t1))
        L = np.maximum(L, np.linalg.norm(J2 - J1, axis=1) / denom)
    return L


def probe_Z(problem, theta, fv, Jm, L):
    """§2.2 lookahead at one checkpoint, single alpha.  Probe arrays are
    local copies, dropped on return — theta is never modified."""
    Zm = np.zeros((K, K))
    for i in range(K):
        theta_probe = theta - (ALPHA / L[i]) * Jm[i]
        f_probe = problem.values(theta_probe)
        for j in range(K):
            if i != j:
                Zm[i, j] = 1.0 - f_probe[j] / max(fv[j], EPS)
    return Zm


def centroid_chain(problem, X_np, y_np, L):
    """Verbatim Smoke-A executor unit for lambda=(1/2,1/2), with
    affinity measurements at every checkpoint in CKPTS_FULL."""
    stoch = PairStochLamOracle(X_np, y_np, batch_size=BATCH,
                               seed=SAMPLER_SEED)
    n = X_np.shape[0]
    epoch_len = max(1, int(np.ceil(n / float(BATCH))))
    x0 = make_pair_initial_point(INIT_SEED)
    fv, Jm = problem.joint(x0)
    chain_x, chain_f, chain_J = x0.copy(), fv, Jm
    series = []
    if 0 in CKPTS_FULL:
        series.append((0, probe_Z(problem, chain_x, chain_f, chain_J, L)))
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
            batch = stoch.sample_batch()
            g_y_S, g_a_S = stoch.grad_pair(y_vec, LAM, batch)
            u_vec = MOMENTUM * u_vec + (g_y_S - g_a_S + g_a_full)
            y_vec = y_vec - eta * u_vec
        f_y, J_y = problem.joint(y_vec)
        if float(f_y @ LAM) > F_a + 1e-10 * (1.0 + abs(F_a)):
            L_scale *= 2.0
            retries_here += 1
            if retries_here > MAX_RETRIES:
                chain_x, chain_f, chain_J = y_vec, f_y, J_y
                retries_here = 0
        else:
            chain_x, chain_f, chain_J = y_vec, f_y, J_y
            retries_here = 0
        if seg in CKPTS_FULL and chain_x is y_vec:
            series.append((seg, probe_Z(problem, chain_x, chain_f,
                                        chain_J, L)))
    return series, chain_f


def aggregate(series, grid):
    """Zbar / C / P matrices and the §2.4 scores over one checkpoint
    grid (only checkpoints present in the series AND the grid count)."""
    sel = [Zm for t, Zm in series if t in grid]
    stack = np.stack(sel)                                   # (T, 2, 2)
    Zbar = stack.mean(axis=0)
    Cm = np.maximum(0.0, -stack).mean(axis=0)
    Pm = (stack < 0).mean(axis=0)
    for M in (Zbar, Cm, Pm):
        np.fill_diagonal(M, 0.0)
    c_j = 0.5 * Cm.sum(axis=0)                              # incoming conflict
    return {"Zbar": Zbar.tolist(), "C": Cm.tolist(), "P": Pm.tolist(),
            "c_j": c_j.tolist(), "Cmean": float(c_j.mean()),
            "Cbalanced": float(c_j.min()),
            "checkpoints": [int(t) for t, _ in series if t in grid]}


def run_pair(pair, out_dir):
    t0 = time.time()
    X_np, y_np = build_data(pair)
    problem = Problem(X_np, y_np)
    L = estimate_L(problem)
    series, endpoint = centroid_chain(problem, X_np, y_np, L)
    rec = {"pair": list(pair), "per_class": PER_CLASS,
           "n": int(X_np.shape[0]), "alpha": ALPHA,
           "L": [float(v) for v in L],
           "centroid_endpoint_f": [float(v) for v in endpoint],
           "grids": {"scan": aggregate(series, SCAN_GRID),
                     "full": aggregate(series, CKPTS_FULL)},
           "Z_series": [{"t": int(t), "Z": Zm.tolist()}
                        for t, Zm in series]}
    rec["wall_seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"pair_{pair[0]}v{pair[1]}.json").write_text(
        json.dumps(rec, indent=1))
    return rec


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra @ ra) * (rb @ rb)))
    return float(ra @ rb / den) if den else 0.0


def write_ranking(records, grid_key, out_dir):
    rows = []
    for rec in records:
        agg = rec["grids"][grid_key]
        a, b = rec["pair"]
        rows.append({"pair": f"{a}v{b}",
                     "Cbalanced": agg["Cbalanced"], "Cmean": agg["Cmean"],
                     "c_j": agg["c_j"],
                     "C_ab": agg["C"][0][1], "C_ba": agg["C"][1][0],
                     "P_ab": agg["P"][0][1], "P_ba": agg["P"][1][0],
                     "wall_seconds": rec["wall_seconds"]})
    rows.sort(key=lambda r: (-r["Cbalanced"], -r["Cmean"]))
    (out_dir / f"ranking_{grid_key}.json").write_text(
        json.dumps(rows, indent=1))
    with open(out_dir / f"ranking_{grid_key}.csv", "w") as fh:
        fh.write("rank,pair,Cbalanced,Cmean,c_0,c_1,C_ab,C_ba,"
                 "P_ab,P_ba,wall_seconds\n")
        for r_i, r in enumerate(rows, 1):
            fh.write(f"{r_i},{r['pair']},{r['Cbalanced']:.6f},"
                     f"{r['Cmean']:.6f},"
                     + ",".join(f"{c:.6f}" for c in r["c_j"])
                     + f",{r['C_ab']:.6f},{r['C_ba']:.6f}"
                     f",{r['P_ab']:.3f},{r['P_ba']:.3f}"
                     f",{r['wall_seconds']}\n")
    return rows


def crosscheck(rows_scan, rows_full, out_dir):
    """Grid stability + Aug-13 conflict-smoke comparison."""
    rank_scan = {r["pair"]: i for i, r in enumerate(rows_scan, 1)}
    rank_full = {r["pair"]: i for i, r in enumerate(rows_full, 1)}
    pairs = [r["pair"] for r in rows_scan]
    rho = spearman(np.array([rank_scan[p] for p in pairs], dtype=float),
                   np.array([rank_full[p] for p in pairs], dtype=float))
    out = {"grid_spearman_scan_vs_full": rho,
           "top1_scan": rows_scan[0]["pair"],
           "top1_full": rows_full[0]["pair"],
           "top5_scan": [r["pair"] for r in rows_scan[:5]],
           "top5_full": [r["pair"] for r in rows_full[:5]]}
    smoke_path = CONFLICT_SMOKE_DIR / "conflict_ranking.json"
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text())
        smoke_rows = smoke if isinstance(smoke, list) else \
            smoke.get("ranking", [])
        cmp_rows = []
        for s in smoke_rows:
            name = s.get("pair") if isinstance(s, dict) else str(s)
            if name is None:
                continue
            name = str(name).replace("-", "v").replace("_", "v")
            cmp_rows.append({"pair": name,
                             "smoke_record": s,
                             "tag_rank_scan": rank_scan.get(name),
                             "tag_rank_full": rank_full.get(name)})
        out["aug13_conflict_smoke"] = cmp_rows
    (out_dir / "stability_crosscheck.json").write_text(
        json.dumps(out, indent=1))
    return out


def figures(rows_scan, rows_full, out_dir):
    top = rows_scan[:15]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(top)), [r["Cbalanced"] for r in top],
           color="#d95f02")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([r["pair"] for r in top], rotation=45, fontsize=8)
    ax.set_ylabel("Cbalanced (scan grid)")
    ax.set_title("TAG affinity K2: top-15 most conflicting pairs")
    fig.tight_layout(); fig.savefig(out_dir / "top15_cbalanced.png")
    plt.close(fig)

    cb_scan = {r["pair"]: r["Cbalanced"] for r in rows_scan}
    cb_full = {r["pair"]: r["Cbalanced"] for r in rows_full}
    pairs = list(cb_scan)
    fig, ax = plt.subplots(figsize=(5, 5))
    xs = [cb_scan[p] for p in pairs]; ys = [cb_full[p] for p in pairs]
    ax.scatter(xs, ys, s=14, color="#1b9e77")
    for p in [r["pair"] for r in rows_scan[:5]]:
        ax.annotate(p, (cb_scan[p], cb_full[p]), fontsize=7)
    lim = [0, max(max(xs), max(ys)) * 1.05]
    ax.plot(lim, lim, lw=0.6, color="gray")
    ax.set_xlabel("Cbalanced, scan grid {0,3,6,9,12,15}")
    ax.set_ylabel("Cbalanced, full grid 0..15")
    ax.set_title("Checkpoint-grid stability")
    fig.tight_layout(); fig.savefig(out_dir / "grid_stability_scatter.png")
    plt.close(fig)


def write_md(rows_scan, cc, out_dir):
    lines = ["# TAG affinity K2 pair ranking (S0, Sep 2, 2026)", "",
             "Headline grid = K3 A1 scan cadence {0,3,6,9,12,15}; "
             "alpha = 0.05; most conflicting first.", "",
             "| rank | pair | Cbalanced | Cmean |", "|---|---|---|---|"]
    for i, r in enumerate(rows_scan, 1):
        lines.append(f"| {i} | {r['pair']} | {r['Cbalanced']:.4f} "
                     f"| {r['Cmean']:.4f} |")
    lines += ["", f"Grid stability: Spearman(scan, full) = "
              f"{cc['grid_spearman_scan_vs_full']:.3f}; "
              f"top-1 scan = {cc['top1_scan']}, "
              f"top-1 full = {cc['top1_full']}."]
    (out_dir / "RANKING.md").write_text("\n".join(lines) + "\n")


def main():
    out = HOME / "scan"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for pair in ALL_PAIRS:
        path = out / f"pair_{pair[0]}v{pair[1]}.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
            print(f"[skip] {pair} (resume)", flush=True)
            continue
        rec = run_pair(pair, out)
        records.append(rec)
        print(f"[done] {pair}  Cbal_scan="
              f"{rec['grids']['scan']['Cbalanced']:.4f}  "
              f"{rec['wall_seconds']}s", flush=True)
    rows_scan = write_ranking(records, "scan", out)
    rows_full = write_ranking(records, "full", out)
    cc = crosscheck(rows_scan, rows_full, out)
    figures(rows_scan, rows_full, out)
    write_md(rows_scan, cc, out)
    print("[scan] outputs in", out, flush=True)


if __name__ == "__main__":
    main()
