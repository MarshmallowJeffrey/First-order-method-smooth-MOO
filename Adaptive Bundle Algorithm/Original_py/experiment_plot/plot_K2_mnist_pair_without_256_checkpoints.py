"""plot_K2_mnist_pair_without_256_checkpoints.py — figures for the
K = 2 MNIST digit-pair campaign (Aug 13, 2026; NEW FILE, nothing
overwritten).  Reads the legs of ONE pair home and draws the five core
figures of the user-approved plan:

1. gn_vs_grads.png       exact audited GN vs grad-equivalents (train)
2. gn_vs_cpu.png         exact audited GN vs CPU seconds (train)
3. fronts_train_test.png train front (solid) + OFFICIAL-TEST front
                         (dashed hollow) per leg, per-class mean CE
                         axes, window capped at ~ln 2 (the divergence
                         arms of vertex grid nodes live outside)
4. front_err_test.png    test per-class error fronts (1 - recall each
                         class), linear axes
5. test_ce_vs_budget.png test per-class CE at checkpoints vs budget
                         and vs CPU (the paper's "test error vs
                         effective passes" analogue)

plus front_metrics.json (train/test front sizes, IGD / max-dist to the
union front raw + central <= ln 2 variant, central HV with reference
(ln 2, ln 2)) and a short README.md.

Test values were computed by the runner (off both axes) on ALL
official t10k rows of the two digits; this script only reads arrays.

Usage:
    python plot_K2_mnist_pair_without_256_checkpoints.py            # every pair home found
    python plot_K2_mnist_pair_without_256_checkpoints.py --home DIR # one home
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from run_pure_budget_K2_without_256_checkpoints import (  # noqa: E402
    _front_metrics,
    _hv_2d,
    _nondominated,
    _pseudo_zero,
)
from run_experiments import _json_ready  # noqa: E402

HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = (HERE.parent.parent / "output"
                 / "CCP/K2_mnist_pair_without_256_checkpoints")
LN2 = float(np.log(2.0))
_AD_COLOR = "#2ca02c"


def _load_legs(home: Path):
    legs = []
    for p in sorted(home.glob("*/summary.json")):
        sm = json.loads(p.read_text(encoding="utf-8"))
        npz = np.load(p.parent / "grams.npz")
        sm["_dir"] = p.parent
        sm["_fvals"] = np.asarray(npz["fvals"], dtype=float)
        sm["_test_ce"] = np.asarray(npz["test_ce"], dtype=float)
        sm["_test_err"] = np.asarray(npz["test_err"], dtype=float)
        sm["_seg_lams"] = np.asarray(npz["seg_lams"], dtype=float)
        legs.append(sm)
    if not legs:
        raise SystemExit(f"no legs under {home}")
    return legs


def _style(sm, rs):
    if sm["policy"] == "baseline":
        r = sm["extra"]["r"]
        reds = plt.get_cmap("Reds")
        color = reds(0.40 + 0.5 * rs.index(r) / max(1, len(rs) - 1))
        return f"baseline r={r}", color, "x"
    return "adaptive CCP", _AD_COLOR, "^"


def _digit_labels(sm):
    a, b = sm["pair"]
    return f"digit-{a} mean CE", f"digit-{b} mean CE"


def make_figures(home: Path) -> None:
    home = Path(home)
    legs = _load_legs(home)
    rs = sorted({sm["extra"]["r"] for sm in legs
                 if sm["policy"] == "baseline"})
    pair = legs[0]["pair"]
    a, b = pair
    budget = legs[0]["budget"]
    xlab_a, xlab_b = _digit_labels(legs[0])

    # ---- 1 + 2: audited GN trajectories --------------------------------
    for axis_key, x_label, fname in (
        ("ck_grads", "total gradient evaluations, grad-equivalents",
         "gn_vs_grads.png"),
        ("ck_cpu", "CPU time, seconds", "gn_vs_cpu.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        x0_pseudo = _pseudo_zero([np.asarray(sm[axis_key], dtype=float)
                                  for sm in legs])
        for sm in legs:
            label, color, marker = _style(sm, rs)
            hx = np.asarray(sm[axis_key], dtype=float)
            hy = np.maximum(np.asarray(sm["audited_gn_history"],
                                       dtype=float), np.finfo(float).tiny)
            hx = np.where(hx > 0, hx, x0_pseudo)
            lw = 1.8 if sm["policy"] != "baseline" else 1.3
            ax.plot(hx, hy, color=color, marker=marker, ms=3.5, lw=lw,
                    label=f"{label} — exact prefix audit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("full-simplex GN* — EXACT 1-D meter")
        ax.set_title(
            f"MNIST pair {a} vs {b} (patch-softplus, d={legs[0]['d']}) — "
            f"pure fixed budget B={budget:,.0f}\nshared segment unit; only "
            f"the next-lambda policy differs", fontsize=9.6)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower left")
        fig.tight_layout()
        fig.savefig(home / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- 3a + 3b: train front / test front, SEPARATE figures -----------
    # Aug-13 user revision: (a) too many lines — show only adaptive CCP
    # plus the SINGLE best baseline (lowest final exact audit, i.e. the
    # winner of the GN figures); (b) train and test fronts get one
    # figure each.  Points outside the central window (either loss
    # > ~ln 2: the unregularised divergence arms of vertex nodes) are
    # dropped BEFORE plotting — otherwise their off-window coordinates
    # still drive the log-axis autoscale and squash the actual fronts.
    win = LN2 * 1.05
    metrics = {}
    for sm in legs:
        label, _c, _m = _style(sm, rs)
        F_tr = sm["_fvals"]
        F_te = sm["_test_ce"]
        fr_tr = F_tr[_nondominated(F_tr)]
        fr_te = F_te[_nondominated(F_te)]
        metrics[label] = {"front_train": fr_tr[np.argsort(fr_tr[:, 0])],
                          "front_test": fr_te[np.argsort(fr_te[:, 0])],
                          "n_points": int(F_tr.shape[0])}
    best_bl = min((sm for sm in legs if sm["policy"] == "baseline"),
                  key=lambda sm: sm["final_audit"])
    show = [sm for sm in legs
            if sm is best_bl or sm["policy"] != "baseline"]
    for key, fname, sub in (
        ("front_train", "front_train.png",
         "TRAIN front (per-class mean CE on the training subset)"),
        ("front_test", "front_test.png",
         "TEST front (per-class mean CE on ALL official t10k rows)"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 5.4))
        for sm in show:
            label, color, marker = _style(sm, rs)
            fr = metrics[label][key]
            cw = fr[(fr[:, 0] <= win) & (fr[:, 1] <= win)]
            ax.plot(cw[:, 0], cw[:, 1], color=color, marker=marker,
                    ms=4.5, lw=1.3,
                    label=f"{label} ({len(cw)} central pts)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(right=LN2 * 1.08)
        ax.set_ylim(top=LN2 * 1.08)
        ax.axhline(LN2, ls=":", lw=0.8, color="gray")
        ax.axvline(LN2, ls=":", lw=0.8, color="gray")
        ax.set_xlabel(f"{xlab_a} (log; window <= ln 2)")
        ax.set_ylabel(f"{xlab_b} (log; window <= ln 2)")
        ax.set_title(
            f"MNIST pair {a} vs {b} — {sub}\nequal budget "
            f"B={budget:,.0f}; adaptive CCP vs best baseline "
            f"(r={best_bl['extra']['r']})", fontsize=9.8)
        ax.grid(True, alpha=0.25, ls=":")
        ax.legend(fontsize=8.0, loc="lower left")
        fig.tight_layout()
        fig.savefig(home / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    stale = home / "fronts_train_test.png"
    if stale.exists():
        stale.unlink()

    # ---- 3c: same two figures + "best point per grid lambda" overlay ---
    # (user request, Aug-13 evening.)  For each grid lambda of the best
    # baseline, the representative is the point with the SMALLEST train
    # lambda^T F among the points delivered UNDER that lambda — the
    # finite-budget stand-in for "one lambda, one converged point".
    # Selection is train-side only; the test panel re-scores the SAME
    # thetas.  Vertex representatives live on the divergence arms and
    # fall outside the ln-2 window (the in-window count says so).
    def _rep_indices(sm):
        """One representative per distinct lambda: smallest own
        lambda^T F on train; ties (favoured-class CE underflowing to 0
        at vertex lambdas) broken by the complementary weighted
        value."""
        sl = sm["_seg_lams"]
        valid = np.nonzero(~np.isnan(sl[:, 0]))[0]
        keys = np.round(sl[valid], 12)
        uniq = np.unique(keys, axis=0)
        F_tr = sm["_fvals"]
        rep_set = set()
        for u in uniq:
            ids = valid[np.all(keys == u, axis=1)]
            order = np.lexsort((F_tr[ids] @ (1.0 - u), F_tr[ids] @ u))
            rep_set.add(int(ids[order[0]]))
        return np.asarray(sorted(rep_set), dtype=int), uniq.shape[0]

    ad_leg = next(sm for sm in show if sm["policy"] != "baseline")
    for rep_sm, rep_name, tag, mkw in (
        (best_bl, f"grid lambda (r={best_bl['extra']['r']})", "replam",
         dict(s=64, facecolors="none", edgecolors="#1f4e9c",
              linewidths=1.5)),
        (ad_leg, "CCP lambda", "replam_ccp",
         dict(s=14, facecolors="none", edgecolors="#1f4e9c",
              linewidths=0.8, alpha=0.75)),
    ):
        rep_idx, n_uniq = _rep_indices(rep_sm)
        for key, src, fname, sub in (
            ("front_train", "_fvals", f"front_train_{tag}.png",
             "TRAIN front (per-class mean CE on the training subset)"),
            ("front_test", "_test_ce", f"front_test_{tag}.png",
             "TEST front (per-class mean CE on ALL official t10k rows)"),
        ):
            fig, ax = plt.subplots(figsize=(7.0, 5.4))
            for sm in show:
                label, color, marker = _style(sm, rs)
                fr = metrics[label][key]
                cw = fr[(fr[:, 0] <= win) & (fr[:, 1] <= win)]
                ax.plot(cw[:, 0], cw[:, 1], color=color, marker=marker,
                        ms=4.5, lw=1.3,
                        label=f"{label} ({len(cw)} central pts)")
            rep = np.asarray(rep_sm[src])[rep_idx]
            rw = rep[(rep[:, 0] <= win) & (rep[:, 1] <= win)]
            ax.scatter(rw[:, 0], rw[:, 1], zorder=6, **mkw,
                       label=(f"best point per {rep_name}, "
                              f"train-selected ({len(rw)}/{n_uniq} "
                              f"in window)"))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(right=LN2 * 1.08)
            ax.set_ylim(top=LN2 * 1.08)
            ax.axhline(LN2, ls=":", lw=0.8, color="gray")
            ax.axvline(LN2, ls=":", lw=0.8, color="gray")
            ax.set_xlabel(f"{xlab_a} (log; window <= ln 2)")
            ax.set_ylabel(f"{xlab_b} (log; window <= ln 2)")
            ax.set_title(
                f"MNIST pair {a} vs {b} — {sub}\nequal budget "
                f"B={budget:,.0f}; adaptive CCP vs best baseline "
                f"(r={best_bl['extra']['r']}) + per-lambda "
                f"representatives", fontsize=9.8)
            ax.grid(True, alpha=0.25, ls=":")
            ax.legend(fontsize=8.0, loc="lower left")
            fig.tight_layout()
            fig.savefig(home / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ---- 4: test error fronts ------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    err_max = 0.0
    for sm in legs:
        label, color, marker = _style(sm, rs)
        E = sm["_test_err"]
        fr = E[_nondominated(E)]
        fr = fr[np.argsort(fr[:, 0])]
        err_max = max(err_max, float(fr[:, 0].max()),
                      float(fr[:, 1].max()))
        ax.plot(fr[:, 0], fr[:, 1], color=color, marker=marker, ms=5,
                lw=1.2, ls="--" if sm["policy"] != "baseline" else "-",
                label=f"{label} ({len(fr)} pts)")
        metrics[label]["front_test_err"] = fr
    ax.set_xlabel(f"digit-{a} test error rate (1 - recall)")
    ax.set_ylabel(f"digit-{b} test error rate (1 - recall)")
    ax.set_xlim(0, min(1.0, err_max * 1.15 + 0.01))
    ax.set_ylim(0, min(1.0, err_max * 1.15 + 0.01))
    ax.set_title(
        f"MNIST pair {a} vs {b} — per-class TEST error fronts at equal "
        f"budget B={budget:,.0f}", fontsize=9.8)
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(home / "front_err_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 5: best balanced TEST CE discovered so far vs budget / CPU ----
    # The raw chain snapshot follows whatever lambda is active, so its
    # test CE oscillates wildly (vertex visits spike the ignored class)
    # — unreadable (Aug-13 finding).  The paper-style curve is the
    # QUALITY DISCOVERED SO FAR: prefix-minimum over delivered points of
    # the mean test CE of the two classes, sampled at checkpoints.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, axis_key, x_label in (
        (axes[0], "ck_grads", "grad-equivalents"),
        (axes[1], "ck_cpu", "CPU seconds"),
    ):
        x0_pseudo = _pseudo_zero([np.asarray(sm[axis_key], dtype=float)
                                  for sm in legs])
        for sm in legs:
            label, color, _marker = _style(sm, rs)
            idx = np.asarray(sm["ck_m"], dtype=int) - 1
            hx = np.asarray(sm[axis_key], dtype=float)
            hx = np.where(hx > 0, hx, x0_pseudo)
            best = np.minimum.accumulate(sm["_test_ce"].mean(axis=1))
            ax.plot(hx, best[idx], color=color, lw=1.5,
                    ls="--" if sm["policy"] != "baseline" else "-",
                    label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("best mean test CE of any delivered point so far")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(
        f"MNIST pair {a} vs {b} — balanced TEST quality discovered along "
        f"the run (prefix-best mean per-class test CE), B={budget:,.0f}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(home / "test_ce_vs_budget.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- metrics json + README ----------------------------------------
    union_tr = np.vstack([m["front_train"] for m in metrics.values()])
    union_tr = union_tr[_nondominated(union_tr)]
    central_tr = union_tr[(union_tr[:, 0] <= LN2)
                          & (union_tr[:, 1] <= LN2)]
    out = {}
    for label, m in metrics.items():
        d = _front_metrics(m["front_train"], union_tr)
        cen = (_front_metrics(m["front_train"], central_tr)
               if central_tr.size else
               dict(igd_to_union=None, maxdist_to_union=None))
        fc = m["front_train"][(m["front_train"][:, 0] <= LN2)
                              & (m["front_train"][:, 1] <= LN2)]
        te = m["front_test"][(m["front_test"][:, 0] <= LN2)
                             & (m["front_test"][:, 1] <= LN2)]
        out[label] = {
            "n_points": m["n_points"],
            "n_front_train": int(m["front_train"].shape[0]),
            "n_front_test": int(m["front_test"].shape[0]),
            "n_front_test_err": int(m["front_test_err"].shape[0]),
            "igd_to_union_train": d["igd_to_union"],
            "maxdist_to_union_train": d["maxdist_to_union"],
            "igd_central_train": cen["igd_to_union"],
            "maxdist_central_train": cen["maxdist_to_union"],
            "hv_central_train": _hv_2d(fc, (LN2, LN2)),
            "hv_central_test": _hv_2d(te, (LN2, LN2)),
        }
    out["_conventions"] = {
        "central_bound": LN2,
        "hv_reference": [LN2, LN2],
        "note": ("central = both losses <= ln 2 (guess-level CE of the "
                 "balanced pair); union/reference from TRAIN fronts of "
                 "all legs; test values on ALL official t10k rows")}
    (home / "front_metrics.json").write_text(
        json.dumps(_json_ready(out), indent=2), encoding="utf-8")

    rows = "\n".join(
        f"| {label} | {v['n_points']:,} | {v['n_front_train']} "
        f"| {v['n_front_test']} | {v['hv_central_train']:.4f} "
        f"| {v['hv_central_test']:.4f} |"
        for label, v in out.items() if not label.startswith("_"))
    (home / "README.md").write_text(f"""# K2 MNIST pair {a} vs {b} — pure fixed budget campaign (Aug 13, 2026)

Legs: baseline grids r in {rs} + adaptive CCP, all at s = {legs[0]['s']},
B = {budget:,.0f} grad-equivalents, batch {legs[0]['config_instance']['msvrg_batch']},
per_class = {legs[0]['per_class']} (balanced maximum), d = {legs[0]['d']}.
Quality: EXACT 1-D meter at every checkpoint (certified).  Test values:
ALL official t10k rows of digits {a} and {b}.  Figures: gn_vs_grads,
gn_vs_cpu, front_train + front_test (each showing adaptive CCP vs the
best baseline only, window <= ln 2 — Aug-13 user revision),
front_err_test, test_ce_vs_budget.  Divergence arms of vertex grid
nodes lie outside the ln-2 window by design (no regularisation — see
Note/Aug_13_note.md).

| leg | delivered pts | train front | test front | HV central train | HV central test |
|-----|---------------|-------------|------------|------------------|-----------------|
{rows}
""", encoding="utf-8")
    print(f"[figures] 5 figures + front_metrics.json + README -> {home}",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=str, default=None)
    args = parser.parse_args()
    homes = ([Path(args.home)] if args.home else
             sorted(CAMPAIGN_ROOT.glob("pair_*_B*")))
    if not homes:
        raise SystemExit("no pair homes found")
    for home in homes:
        if home.is_dir() and list(home.glob("*/summary.json")):
            make_figures(home)


if __name__ == "__main__":
    main()
