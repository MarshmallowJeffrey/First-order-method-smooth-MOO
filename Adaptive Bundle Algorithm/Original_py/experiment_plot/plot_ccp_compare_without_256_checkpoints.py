"""plot_ccp_compare_without_256_checkpoints.py — figures for the
CCP-vs-IPOPT comparison campaign (experiment 1).

NEW FILE (Aug 9, 2026).  Reads the campaign homes written by the
orchestrators (and audit_v2.json at K6); writes figures back into the
same homes.  The original runners' --figure code is untouched — it
filters on policy == "adaptive"/"baseline" and never sees the CCP legs.

Figures per K:
    <K>_gn_vs_grads.png   audited GN trajectories (adaptive legs) +
    <K>_gn_vs_cpu.png     baseline final points; K2 audits are exact,
                          K6 audits are audit_v2 (two-instrument max)
    <K>_gap_vs_decision.png   targeting optimality gap per decision:
                          K2 exact meter at EVERY decision (20001-grid
                          + crossing polish); K6 checkpoint-aligned
                          decisions vs audit_v2 (Aug-9 Q&A granularity)
    <K>_fronts.png        discovered nondominated fronts (F1, F2;
                          K6 = projection) + front metrics json

Usage:
    python plot_ccp_compare_without_256_checkpoints.py --which K2
    python plot_ccp_compare_without_256_checkpoints.py --which K6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from run_pure_budget_K2_without_256_checkpoints import (  # noqa: E402
    _nondominated,
    exact_gn_1d,
)
from ccp_lambda_solver import phi_batch  # noqa: E402

HERE = Path(__file__).resolve().parent
COMPARE_ROOT = (HERE.parent.parent / "output"
                / "CCP/ccp_compare_without_256_checkpoints")
STYLE = {"adaptive": dict(color="#2ca02c", marker="^",
                          label="adaptive (IPOPT ts24)"),
         "adaptive_ccp": dict(color="#1f77b4", marker="o",
                              label="adaptive (CCP)")}


def _legs(home):
    legs = []
    for p in sorted(home.glob("*/summary.json")):
        if "artifact" in p.parent.name:      # archived anomalous runs
            continue
        sm = json.loads(p.read_text())
        sm["_dir"] = p.parent
        legs.append(sm)
    ad = [s for s in legs if s["policy"] in ("adaptive", "adaptive_ccp")]
    bl = [s for s in legs if s["policy"] == "baseline"]
    return ad, bl


def _audit_axes(sm, which):
    """(x_grads, x_cpu, audited_values) for one adaptive leg."""
    if which == "K2":
        return (np.asarray(sm["ck_grads"], float),
                np.asarray(sm["ck_cpu"], float),
                np.asarray(sm["audited_gn_history"], float))
    av = json.loads((sm["_dir"] / "audit_v2.json").read_text())
    by_m = {}
    for m, g, c in zip(sm["ck_m"], sm["ck_grads"], sm["ck_cpu"]):
        by_m.setdefault(int(m), (float(g), float(c)))
    xs_g, xs_c, ys = [], [], []
    for m, v in zip(av["stacks_m"], av["v2"]):
        g, c = by_m.get(int(m), (sm["ck_grads"][-1], sm["ck_cpu"][-1]))
        xs_g.append(g)
        xs_c.append(c)
        ys.append(v)
    return np.asarray(xs_g), np.asarray(xs_c), np.asarray(ys)


def _baseline_final(sm, which):
    if which == "K2":
        v = sm["final_audit"]
    else:
        av = json.loads((sm["_dir"] / "audit_v2.json").read_text())
        v = av["v2"][-1]
    return (float(sm["ck_grads"][-1]), float(sm["ck_cpu"][-1]), float(v))


def gn_figures(home, which, ad, bl):
    reds = plt.get_cmap("Reds")
    rs = sorted({s["extra"]["r"] for s in bl})
    for xi, xlabel, fname in ((0, "total gradient evaluations, "
                               "grad-equivalents", f"{which}_gn_vs_grads.png"),
                              (1, "CPU time, seconds",
                               f"{which}_gn_vs_cpu.png")):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        # shared pseudo-zero: both adaptive legs start from the SAME
        # x origin (their first audited stack is the identical {x0}
        # bundle, so the curves share their starting point exactly)
        floors = []
        for sm in ad:
            x = _audit_axes(sm, which)[xi]
            if np.any(x > 0):
                floors.append(x[x > 0].min())
        shared_floor = min(floors) if floors else 1e-2
        for sm in ad:
            axes3 = _audit_axes(sm, which)
            st = STYLE[sm["policy"]]
            ax.plot(np.maximum(axes3[xi], shared_floor),
                    axes3[2], lw=1.8, ms=4, **st)
        for sm in bl:
            g, c, v = _baseline_final(sm, which)
            r = sm["extra"]["r"]
            col = reds(0.45 + 0.5 * rs.index(r) / max(1, len(rs) - 1))
            x = (g, c)[xi]
            ax.plot([x], [v], marker="s", color=col, ms=7, ls="none",
                    label=f"baseline r={r}, s={sm['s']} (final)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        meter = ("exact 1-D audit" if which == "K2"
                 else "audit_v2 = max(strict-64, CCP N=8192)")
        ax.set_ylabel(f"GN* of delivered set ({meter})")
        ax.set_title(f"{which} pure fixed budget — baselines vs adaptive "
                     "(IPOPT) vs adaptive (CCP)")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(home / fname, dpi=160)
        plt.close(fig)
        print(f"[plot] {fname}", flush=True)


def gap_figure_K2(home, ad):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for sm in ad:
        Ms = np.asarray(np.load(sm["_dir"] / "grams.npz")["gram_stack"],
                        float)
        lams = np.asarray(np.load(sm["_dir"] / "grams.npz")["lam_history"],
                          float)
        s = int(sm["s"])
        gaps, ds = [], []
        for d in range(len(lams)):
            m_d = 1 + s * d
            if m_d > Ms.shape[0]:
                break
            prefix = Ms[:m_d]
            phi_d, _ = phi_batch(prefix, lams[d][None, :])
            ref = exact_gn_1d(prefix, grid_points=20_001, polish=True)[0]
            gaps.append(max(float(ref - phi_d[0]), 1e-18))
            ds.append(d)
        st = STYLE[sm["policy"]]
        ax.semilogy(ds, gaps, lw=1.1, alpha=0.85, color=st["color"],
                    label=st["label"])
    ax.set_xlabel("decision index")
    ax.set_ylabel("targeting gap  phi_exact − phi(lambda chosen)")
    ax.set_title("K2 targeting optimality gap per decision (exact meter)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(home / "K2_gap_vs_decision.png", dpi=160)
    plt.close(fig)
    print("[plot] K2_gap_vs_decision.png", flush=True)


def gap_figure_K6(home, ad):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for sm in ad:
        av = json.loads((sm["_dir"] / "audit_v2.json").read_text())
        g = av["gap"]
        st = STYLE[sm["policy"]]
        ax.semilogy(g["decision"], np.maximum(g["gap"], 1e-18), lw=1.4,
                    marker=".", ms=4, color=st["color"], label=st["label"])
    ax.set_xlabel("decision index (checkpoint-aligned)")
    ax.set_ylabel("targeting gap  audit_v2 − phi(lambda chosen)")
    ax.set_title("K6 targeting gap vs audit_v2 (two-instrument lower-bound "
                 "reference — heuristic, not exact)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(home / "K6_gap_vs_decision.png", dpi=160)
    plt.close(fig)
    print("[plot] K6_gap_vs_decision.png", flush=True)


def _igd(F, ref):
    d = np.linalg.norm(ref[:, None, :] - F[None, :, :], axis=2).min(axis=1)
    return float(d.mean()), float(d.max())


def fronts_figure(home, which, ad, bl):
    legs = ad + [sm for sm in bl if sm["extra"]["r"] == 10
                 and sm["s"] == ad[0]["s"]]
    fronts = {}
    for sm in legs:
        F = np.asarray(np.load(sm["_dir"] / "grams.npz")["fvals"], float)
        fronts[sm["policy"] + ("" if sm["policy"] != "baseline"
                               else f"_r{sm['extra']['r']}")] = \
            F[_nondominated(F)]
    union = np.vstack(list(fronts.values()))
    union = union[_nondominated(union)]
    metrics = {}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    colors = {"adaptive": "#2ca02c", "adaptive_ccp": "#1f77b4",
              "baseline_r10": "#d62728"}
    for name, F in fronts.items():
        igd, dmax = _igd(F, union)
        metrics[name] = {"n_points": int(F.shape[0]), "igd_to_union": igd,
                         "max_dist_to_union": dmax}
        order = np.argsort(F[:, 0])
        ax.plot(np.maximum(F[order, 0], 1e-6),
                np.maximum(F[order, 1], 1e-6), "-o", ms=3, lw=1.4,
                color=colors.get(name, "gray"), alpha=0.85, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    proj = "" if which == "K2" else " (F1-F2 projection of the 6-D front)"
    ax.set_title(f"{which} discovered nondominated fronts{proj}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(home / f"{which}_fronts.png", dpi=160)
    plt.close(fig)
    (home / f"{which}_front_metrics_ccp_compare.json").write_text(
        json.dumps(metrics, indent=2))
    print(f"[plot] {which}_fronts.png + metrics", flush=True)


def _central_front_metrics_K6(home, ad, bl):
    """Aug-10 revision (user): score 6-D fronts against the CENTRAL
    reference front — R_central = {r in union front : r_k <= 1 for all
    k} — with Central IGD (lower better), Central max-distance (lower
    better) and Central HV (share of [z_ideal, 1]^6 dominated; higher
    better, Sobol Monte-Carlo).  Replaces the F1-F2 projection figure."""
    from scipy.stats import qmc
    legs = ad + bl
    fronts = {}
    for sm in legs:
        F = np.asarray(np.load(sm["_dir"] / "grams.npz")["fvals"], float)
        name = (sm["policy"] if sm["policy"] != "baseline"
                else f"baseline_r{sm['extra']['r']}_s{sm['s']}")
        fronts[name] = F[_nondominated(F)]
    union = np.vstack(list(fronts.values()))
    union = union[_nondominated(union)]
    # central box bound: the slide's c = 1 when feasible; on instances
    # whose loss scale makes that empty (here: every union point has
    # some class CE > 2), fall back to the same idea data-driven —
    # c = median over union points of max_k r_k, keeping the "jointly
    # moderate" half and dropping specialists/wanderers.
    c_box = 1.0
    central = union[np.all(union <= c_box, axis=1)]
    if central.shape[0] == 0:
        c_box = float(np.median(union.max(axis=1)))
        central = union[np.all(union <= c_box, axis=1)]
    z_ideal = central.min(axis=0)
    K = union.shape[1]
    sob = qmc.Sobol(d=K, scramble=True, seed=0)
    Z = z_ideal + sob.random(2 ** 19) * (c_box - z_ideal)  # box samples
    union_from = {}
    for name, F in fronts.items():
        union_from[name] = int(sum(
            any(np.allclose(u, p) for p in F) for u in union))
    metrics = {"reference": {"union_size": int(union.shape[0]),
                             "union_composition": union_from,
                             "central_size": int(central.shape[0]),
                             "c_box": c_box,
                             "z_ideal": [float(v) for v in z_ideal]}}
    for name, F in fronts.items():
        d = np.linalg.norm(central[:, None, :] - F[None, :, :],
                           axis=2).min(axis=1)
        Fc = F[np.all(F <= c_box, axis=1)]
        if Fc.shape[0]:
            dom = np.zeros(Z.shape[0], dtype=bool)
            for chunk in range(0, Fc.shape[0], 64):
                P = Fc[chunk:chunk + 64]
                dom |= np.any(
                    np.all(Z[:, None, :] >= P[None, :, :], axis=2), axis=1)
            hv = float(dom.mean())
        else:
            hv = 0.0
        metrics[name] = {"n_front_points": int(F.shape[0]),
                         "n_central_points": int(Fc.shape[0]),
                         "central_igd": float(d.mean()),
                         "central_max_distance": float(d.max()),
                         "central_hv": hv}
    (home / "K6_front_metrics_ccp_compare.json").write_text(
        json.dumps(metrics, indent=2))

    # Aug-10 (user, exp-3 report revision): the same scoring with the
    # baseline EXCLUDED — union/reference/c are re-derived from the two
    # adaptive fronts alone, so the numbers are NOT comparable to the
    # three-method table above and are stored separately.
    fronts2 = {sm["policy"]: fronts[sm["policy"]] for sm in ad}
    union2 = np.vstack(list(fronts2.values()))
    union2 = union2[_nondominated(union2)]
    c2 = 1.0
    central2 = union2[np.all(union2 <= c2, axis=1)]
    if central2.shape[0] == 0:
        c2 = float(np.median(union2.max(axis=1)))
        central2 = union2[np.all(union2 <= c2, axis=1)]
    z2 = central2.min(axis=0)
    sob2 = qmc.Sobol(d=K, scramble=True, seed=0)
    Z2 = z2 + sob2.random(2 ** 19) * (c2 - z2)
    m2 = {"reference": {"union_size": int(union2.shape[0]),
                        "central_size": int(central2.shape[0]),
                        "c_box": c2, "z_ideal": [float(v) for v in z2]}}
    for name, F in fronts2.items():
        d = np.linalg.norm(central2[:, None, :] - F[None, :, :],
                           axis=2).min(axis=1)
        Fc = F[np.all(F <= c2, axis=1)]
        dom = np.zeros(Z2.shape[0], dtype=bool)
        for chunk in range(0, Fc.shape[0], 64):
            P = Fc[chunk:chunk + 64]
            dom |= np.any(
                np.all(Z2[:, None, :] >= P[None, :, :], axis=2), axis=1)
        m2[name] = {"n_front_points": int(F.shape[0]),
                    "n_central_points": int(Fc.shape[0]),
                    "central_igd": float(d.mean()),
                    "central_max_distance": float(d.max()),
                    "central_hv": float(dom.mean())}
    (home / "K6_front_metrics_adaptive_only.json").write_text(
        json.dumps(m2, indent=2))
    print("[plot] K6_front_metrics_adaptive_only.json (no-baseline "
          "variant)", flush=True)

    names = [n for n in fronts]
    colors = {"adaptive": "#2ca02c", "adaptive_ccp": "#1f77b4"}
    panels = (("central_igd", "Central IGD  (lower = better coverage)"),
              ("central_max_distance", "Central max-distance  (lower = "
               "better worst miss)"),
              ("central_hv", "Central HV  (higher = more region "
               "controlled)"))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    for ax, (key, title) in zip(axes, panels):
        vals = [metrics[n][key] for n in names]
        bars = ax.bar(range(len(names)), vals,
                      color=[colors.get(n, "#d62728") for n in names],
                      alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4g}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("adaptive", "adapt.")
                            for n in names], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("K6 fronts scored against the central reference front "
                 f"(|R_central| = {central.shape[0]}, "
                 f"box [z_ideal, {c_box:.3g}]^6)", fontsize=10)
    fig.tight_layout()
    fig.savefig(home / "K6_front_central_metrics.png", dpi=160)
    plt.close(fig)
    print("[plot] K6_front_central_metrics.png + metrics json", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["K2", "K6"], required=True)
    args = ap.parse_args()
    home = COMPARE_ROOT / ("K2_B20000" if args.which == "K2"
                           else "K6_B80912")
    ad, bl = _legs(home)
    if not ad:
        raise SystemExit(f"no adaptive legs under {home}")
    if args.which == "K6":
        # Aug-10 revision (user): K6 figures keep only baseline_r10_s1
        bl = [sm for sm in bl
              if sm["extra"]["r"] == 10 and sm["s"] == 1]
    gn_figures(home, args.which, ad, bl)
    if args.which == "K2":
        gap_figure_K2(home, ad)
        fronts_figure(home, args.which, ad, bl)
    else:
        gap_figure_K6(home, ad)
        _central_front_metrics_K6(home, ad, bl)


if __name__ == "__main__":
    main()
