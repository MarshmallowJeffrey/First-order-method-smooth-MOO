"""plot_ccp_compare_K10_mnist_without_256_checkpoints.py — figures for
the K = 10 MNIST patch-softplus trial (two adaptive legs, audit_v2).

NEW FILE (Aug 9, 2026).  Reads the trial home + audit_v2.json; writes:
    K10_gn_vs_grads.png, K10_gn_vs_cpu.png, K10_gap_vs_decision.png,
    K10_per_class_losses.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HOME = (Path(__file__).resolve().parent.parent.parent / "output"
        / "CCP/ccp_compare_without_256_checkpoints" / "K10_mnist10k_B55000")
STYLE = {"adaptive": dict(color="#2ca02c", marker="^",
                          label="adaptive (IPOPT ts24)"),
         "adaptive_ccp": dict(color="#1f77b4", marker="o",
                              label="adaptive (CCP)")}


def legs():
    out = []
    for p in sorted(HOME.glob("*/summary.json")):
        sm = json.loads(p.read_text())
        sm["_dir"] = p.parent
        sm["_av"] = json.loads((p.parent / "audit_v2.json").read_text())
        out.append(sm)
    return out


def gn_figures(ls):
    for xi, xlabel, fname in (
        (0, "total gradient evaluations, grad-equivalents",
         "K10_gn_vs_grads.png"),
        (1, "CPU time, seconds", "K10_gn_vs_cpu.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        # shared pseudo-zero: both legs' first audited stack is the
        # identical {x0} bundle, so the curves share their origin
        series = []
        for sm in ls:
            by_m = {}
            for m, g, c in zip(sm["ck_m"], sm["ck_grads"], sm["ck_cpu"]):
                by_m.setdefault(int(m), (float(g), float(c)))
            xs, ys = [], []
            for m, v in zip(sm["_av"]["stacks_m"], sm["_av"]["v2"]):
                gc = by_m.get(int(m), (sm["ck_grads"][-1], sm["ck_cpu"][-1]))
                xs.append(gc[xi])
                ys.append(v)
            series.append((np.asarray(xs), np.asarray(ys), sm["policy"]))
        floors = [x[x > 0].min() for x, _, _ in series if np.any(x > 0)]
        shared_floor = min(floors) if floors else 1e-2
        for xs, ys, pol in series:
            ax.plot(np.maximum(xs, shared_floor), ys, lw=1.8, ms=4,
                    **STYLE[pol])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GN* of delivered set "
                      "(audit_v2 = max(strict-64, CCP N=8192))")
        ax.set_title("K10 MNIST patch-softplus trial — pure fixed budget "
                     "(B=55000, n=10k)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(HOME / fname, dpi=160)
        plt.close(fig)
        print(f"[plot] {fname}", flush=True)


def gap_figure(ls):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for sm in ls:
        g = sm["_av"]["gap"]
        st = STYLE[sm["policy"]]
        ax.semilogy(g["decision"], np.maximum(g["gap"], 1e-18), lw=1.4,
                    marker=".", ms=4, color=st["color"], label=st["label"])
    ax.set_xlabel("decision index (checkpoint-aligned)")
    ax.set_ylabel("targeting gap  audit_v2 − phi(lambda chosen)")
    ax.set_title("K10 targeting gap (two-instrument lower-bound reference)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(HOME / "K10_gap_vs_decision.png", dpi=160)
    plt.close(fig)
    print("[plot] K10_gap_vs_decision.png", flush=True)


def per_class_figure(ls):
    fig, axes = plt.subplots(1, len(ls), figsize=(6.2 * len(ls), 4.4),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, sm in zip(axes[0], ls):
        npz = np.load(sm["_dir"] / "grams.npz")
        F = np.asarray(npz["fvals"], float)          # (m, 10)
        xg = np.asarray(npz["seg_grads"], float)
        for k in range(10):
            ax.plot(xg, F[:, k], lw=0.9, color=cmap(k), alpha=0.85,
                    label=f"digit {k}")
        ax.set_xlabel("grad-equivalents spent")
        ax.set_title(STYLE[sm["policy"]]["label"], fontsize=10)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("per-class cross-entropy at delivered points")
    axes[0][-1].legend(fontsize=6, ncol=2)
    fig.suptitle("K10 MNIST — per-class objective values along each leg",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(HOME / "K10_per_class_losses.png", dpi=160)
    plt.close(fig)
    print("[plot] K10_per_class_losses.png", flush=True)


def central_front_metrics(ls):
    """Aug-10 revision (user): score the 10-D fronts against the
    central reference front (slide method, as in the K6 revision):
    R_central = union-front points with every objective <= c (c = 1
    when nonempty, else the median of per-point max_k), Central IGD /
    max-distance (lower better) and Central HV (share of
    [z_ideal, c]^10 dominated, Sobol Monte-Carlo, higher better)."""
    from scipy.stats import qmc
    from run_pure_budget_K2_without_256_checkpoints import _nondominated
    fronts = {}
    for sm in ls:
        F = np.asarray(np.load(sm["_dir"] / "grams.npz")["fvals"], float)
        fronts[sm["policy"]] = F[_nondominated(F)]
    union = np.vstack(list(fronts.values()))
    union = union[_nondominated(union)]
    c_box = 1.0
    central = union[np.all(union <= c_box, axis=1)]
    if central.shape[0] == 0:
        c_box = float(np.median(union.max(axis=1)))
        central = union[np.all(union <= c_box, axis=1)]
    z_ideal = central.min(axis=0)
    sob = qmc.Sobol(d=union.shape[1], scramble=True, seed=0)
    Z = z_ideal + sob.random(2 ** 19) * (c_box - z_ideal)
    metrics = {"reference": {"union_size": int(union.shape[0]),
                             "central_size": int(central.shape[0]),
                             "c_box": c_box,
                             "z_ideal": [float(v) for v in z_ideal]}}
    for name, F in fronts.items():
        d = np.linalg.norm(central[:, None, :] - F[None, :, :],
                           axis=2).min(axis=1)
        Fc = F[np.all(F <= c_box, axis=1)]
        dom = np.zeros(Z.shape[0], dtype=bool)
        for chunk in range(0, Fc.shape[0], 64):
            P = Fc[chunk:chunk + 64]
            dom |= np.any(
                np.all(Z[:, None, :] >= P[None, :, :], axis=2), axis=1)
        metrics[name] = {"n_front_points": int(F.shape[0]),
                         "n_central_points": int(Fc.shape[0]),
                         "central_igd": float(d.mean()),
                         "central_max_distance": float(d.max()),
                         "central_hv": float(dom.mean())}
    (HOME / "K10_front_metrics_ccp_compare.json").write_text(
        json.dumps(metrics, indent=2))

    names = list(fronts)
    panels = (("central_igd", "Central IGD  (lower better)"),
              ("central_max_distance", "Central max-distance  (lower "
               "better)"),
              ("central_hv", "Central HV  (higher better)"))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    for ax, (key, title) in zip(axes, panels):
        vals = [metrics[n][key] for n in names]
        bars = ax.bar(range(len(names)), vals,
                      color=[STYLE[n]["color"] for n in names], alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4g}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([STYLE[n]["label"] for n in names], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("K10 MNIST fronts vs the central reference front "
                 f"(|R_central| = {central.shape[0]}, "
                 f"box [z_ideal, {c_box:.3g}]^10)", fontsize=10)
    fig.tight_layout()
    fig.savefig(HOME / "K10_front_central_metrics.png", dpi=160)
    plt.close(fig)
    print("[plot] K10_front_central_metrics.png + metrics json", flush=True)


if __name__ == "__main__":
    ls = legs()
    if not ls:
        raise SystemExit(f"no legs under {HOME}")
    gn_figures(ls)
    gap_figure(ls)
    per_class_figure(ls)
    central_front_metrics(ls)
