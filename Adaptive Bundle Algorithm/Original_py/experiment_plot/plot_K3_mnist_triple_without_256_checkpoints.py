"""plot_K3_mnist_triple_without_256_checkpoints.py — figures for the
K = 3 MNIST digit-triple campaign (Aug 26, 2026; NEW FILE, nothing
overwritten).  Reads the legs of ONE triple home and draws the five
core figures — the K = 2 pair set with the fronts lifted to 3-D:

1. gn_vs_grads.png       audited GN* vs grad-equivalents (train);
                         meter = audit_v2 two-instrument max
2. gn_vs_cpu.png         audited GN* vs CPU seconds (train)
3. front_train.png       TRAIN front, adaptive CCP vs the single best
   front_test.png        baseline (lowest final audit).  Aug-26
                         restyle (user request, modelled on the
                         breakable-bottles reference figure): ONE row
                         of three 3-D views at fixed angles
                         (22,-60)/(18,-140)/(34,115), LINEAR axes,
                         window capped at ~ln 3 (divergence arms of
                         vertex / edge grid nodes live outside);
                         pairwise log-log projections move to the
                         companion front_{train,test}_proj.png;
                         front_test is the same layout on
                         OFFICIAL-TEST per-class CE
4. front_err_test.png    test per-class error fronts (1 - recall each
                         class), all legs, same 3-view layout, linear
                         axes (+ front_err_test_proj.png companion)
5. test_ce_vs_budget.png prefix-best mean per-class TEST CE vs budget
                         and vs CPU (the paper's "test error vs
                         effective passes" analogue)

plus front_metrics.json (train/test front sizes, 3-D IGD / max-dist to
the union front raw + central <= ln 3 variant, central 3-D HV with
reference (ln 3, ln 3, ln 3)) and a short README.md.

Test values were computed by the runner (off both axes) on ALL
official t10k rows of the three digits; this script only reads arrays.

Usage:
    python plot_K3_mnist_triple_without_256_checkpoints.py            # every triple home found
    python plot_K3_mnist_triple_without_256_checkpoints.py --home DIR # one home
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
    _pseudo_zero,
)
from run_experiments import _json_ready  # noqa: E402

HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = (HERE.parent.parent / "output"
                 / "CCP/K3_mnist_triple_without_256_checkpoints")
LN3 = float(np.log(3.0))
_AD_COLOR = "#2ca02c"
PROJ = [(0, 1), (1, 2), (0, 2)]


# ---------------------------------------------------------------------------
# K-D front helpers (the K2 module's are 2-D staircase specific)
# ---------------------------------------------------------------------------

def _nondominated_kd(F, block=512):
    """Indices of the nondominated subset of an (m, K) value array
    (minimisation; pairwise check, block-chunked)."""
    F = np.asarray(F, dtype=float)
    m = F.shape[0]
    keep = np.ones(m, dtype=bool)
    for i in range(0, m, block):
        blk = F[i:i + block]
        le = (F[None, :, :] <= blk[:, None, :]).all(axis=2)
        lt = (F[None, :, :] < blk[:, None, :]).any(axis=2)
        keep[i:i + block] &= ~(le & lt).any(axis=1)
    return np.nonzero(keep)[0]


def _hv_3d(front, ref):
    """3-D hypervolume of a point set w.r.t. reference ``ref``
    (minimisation): sweep the third axis ascending, integrating the
    2-D staircase hypervolume of the accumulated (f1, f2) projections
    over each slab.  Points beyond ref contribute nothing."""
    P = np.asarray(front, dtype=float)
    P = P[(P[:, 0] <= ref[0]) & (P[:, 1] <= ref[1]) & (P[:, 2] <= ref[2])]
    if P.size == 0:
        return 0.0
    P = P[np.argsort(P[:, 2], kind="stable")]
    zs = np.append(P[:, 2], ref[2])
    hv = 0.0
    for k in range(P.shape[0]):
        dz = zs[k + 1] - zs[k]
        if dz > 0.0:
            hv += _hv_2d(P[:k + 1, :2], (ref[0], ref[1])) * dz
    return float(hv)


def _ymin_step(pts2d):
    """For a 2-D point set (minimisation), return (xs, ymin) such that
    the region dominated at horizontal coordinate u is {v >= ymin(u)}
    with ymin(u) = min{y_i : x_i <= u} (staircase prefix-min)."""
    order = np.argsort(pts2d[:, 0], kind="stable")
    xs = pts2d[order, 0]
    ymin = np.minimum.accumulate(pts2d[order, 1])
    return xs, ymin


def _dominated_mask(pts2d, U, V):
    """Boolean mask over the (V-rows, U-cols) grid: cell dominated by
    at least one of pts2d (both coordinates >=)."""
    if pts2d.size == 0:
        return np.zeros((V.size, U.size), dtype=bool)
    xs, ymin = _ymin_step(pts2d)
    idx = np.searchsorted(xs, U, side="right") - 1
    thr = np.where(idx >= 0, ymin[np.clip(idx, 0, None)], np.inf)
    return V[:, None] >= thr[None, :]


def _hv_slice_figure(home, fr_a, fr_b, label_a, label_b, digits, budget):
    """Supplementary figure (user request, Aug 26): make the 3-D
    hypervolume difference VISIBLE.  Four horizontal slices at fixed
    F_3: rasterised 2-D dominated regions (both / only A / only B),
    plus the per-height area difference and its running integral —
    which by construction ends at HV(A) - HV(B)."""
    ref2 = (LN3, LN3)
    zs = np.linspace(0.0, LN3, 400)
    area_a = np.array([_hv_2d(fr_a[fr_a[:, 2] <= z][:, :2], ref2)
                       for z in zs])
    area_b = np.array([_hv_2d(fr_b[fr_b[:, 2] <= z][:, :2], ref2)
                       for z in zs])
    dA = area_a - area_b
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (dA[1:] + dA[:-1]) * np.diff(zs))])
    # data-driven slice heights: where the running |contribution| passes
    # 15/45/75% of its final value, plus the argmax of dA
    tot = cum[-1] if cum[-1] != 0 else 1.0
    picks = []
    for q in (0.15, 0.45, 0.75):
        k = int(np.argmin(np.abs(cum - q * tot)))
        picks.append(zs[k])
    picks.append(zs[int(np.argmax(dA))])
    picks = sorted(set(round(float(z), 3) for z in picks))
    while len(picks) < 4:                     # degenerate fallback
        picks.append(round(float(picks[-1]) + 0.1, 3))
    picks = picks[:4]

    lo = max(1e-3, 0.8 * min(float(fr_a.min()), float(fr_b.min())))
    U = np.geomspace(lo, LN3, 500)
    V = np.geomspace(lo, LN3, 500)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff", "#e0e0e0", "#2ca02c", "#fb8a6a"])

    fig, axes = plt.subplots(1, 5, figsize=(17.0, 3.9))
    for ax, z in zip(axes[:4], picks):
        pa = fr_a[fr_a[:, 2] <= z][:, :2]
        pb = fr_b[fr_b[:, 2] <= z][:, :2]
        ma = _dominated_mask(pa, U, V)
        mb = _dominated_mask(pb, U, V)
        code = np.zeros(ma.shape, dtype=int)
        code[ma & mb] = 1
        code[ma & ~mb] = 2
        code[~ma & mb] = 3
        ax.pcolormesh(U, V, code, cmap=cmap, vmin=0, vmax=3,
                      shading="auto", rasterized=True)
        if pa.size:
            ax.plot(pa[:, 0], pa[:, 1], ls="", marker="^", ms=2.5,
                    color="#1a6e1a")
        if pb.size:
            ax.plot(pb[:, 0], pb[:, 1], ls="", marker="x", ms=2.5,
                    color="#c0442a")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, LN3)
        ax.set_ylim(lo, LN3)
        aa = _hv_2d(pa, ref2)
        ab = _hv_2d(pb, ref2)
        ax.set_title(f"slice F_{digits[2]}-CE <= {z:.3f}\n"
                     f"area {label_a.split()[0]}={aa:.3f}, "
                     f"{label_b.split()[0]}={ab:.3f}, "
                     f"diff={aa - ab:+.4f}", fontsize=7.5)
        ax.set_xlabel(f"digit-{digits[0]} CE", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(f"digit-{digits[1]} CE", fontsize=8)
        ax.tick_params(labelsize=7)
    ax = axes[4]
    ax.plot(zs, dA, color="#555555", lw=1.2, label="area diff at height z")
    ax.axhline(0.0, color="gray", lw=0.6, ls=":")
    ax2 = ax.twinx()
    ax2.plot(zs, cum, color="#2ca02c", lw=1.8,
             label="running integral (= HV gap)")
    ax2.annotate(f"total = {cum[-1]:+.4f}", (zs[-1], cum[-1]),
                 xytext=(-10, -28), textcoords="offset points",
                 ha="right", fontsize=8, color="#1a6e1a")
    ax.set_xlabel(f"slice height z (digit-{digits[2]} CE)", fontsize=8)
    ax.set_ylabel("2-D area difference", fontsize=8)
    ax2.set_ylabel("cumulative volume difference", fontsize=8)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    ax.set_title("area diff and its integral over z", fontsize=8)
    lines = [l for l in ax.get_lines() + ax2.get_lines()
             if not l.get_label().startswith("_")]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=6.5,
              loc="lower right")
    fig.suptitle(
        f"MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]} — WHERE "
        f"the TEST hypervolume gap lives ({label_a} vs {label_b}, "
        f"B={budget:,.0f}).  Green region: dominated only by {label_a}; "
        f"orange: only by {label_b}; gray: both.  Integrating the area "
        f"difference over the slice height reproduces the 3-D HV gap.",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(home / "hv_slices_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(cum[-1])


def _front_surface_figure(home, series, digits, budget, edge_max=0.45,
                          nbins=18):
    """Supplementary figure (user request, Aug 26 late): the MODPO-style
    rendering — faint scatter of the full central cloud + a shaded
    triangulated FRONTIER SHEET through the nondominated points.
    Triangulation is Delaunay in the (F_1, F_2) plane; triangles with
    any 3-D edge longer than ``edge_max`` are dropped so the sheet does
    not fabricate bridges across the empty region between the arms.
    ``series`` = [(label, color, cloud_pts, front_pts)] in draw order
    (baseline first, adaptive last)."""
    import matplotlib.tri as mtri
    from matplotlib.patches import Patch

    def _envelope(fr, nbins=nbins):
        """Lower envelope of the front on a log-spaced (F_1, F_2) grid:
        one representative per occupied cell, the min-F_3 point.  This
        is what turns the raw nondominated cloud (multi-valued in z
        along the F_3 arm) into a clean single-valued sheet."""
        lo = max(1e-3, 0.9 * float(fr[:, :2].min()))
        edges = np.geomspace(lo, LN3, nbins + 1)
        xi = np.clip(np.searchsorted(edges, fr[:, 0]) - 1, 0, nbins - 1)
        yi = np.clip(np.searchsorted(edges, fr[:, 1]) - 1, 0, nbins - 1)
        best = {}
        for k in range(fr.shape[0]):
            key = (int(xi[k]), int(yi[k]))
            if key not in best or fr[k, 2] < fr[best[key], 2]:
                best[key] = k
        return fr[np.array(sorted(best.values()))]

    env_series = [(label, color, _envelope(fr))
                  for label, color, _cloud, fr in series]
    fig = plt.figure(figsize=(13.4, 4.9))
    for p, (elev, azim) in enumerate(VIEWS3):
        ax = fig.add_subplot(1, 3, p + 1, projection="3d")
        for label, color, env in env_series:
            ax.scatter(env[:, 0], env[:, 1], env[:, 2], color=color,
                       s=7, alpha=0.9, depthshade=False)
            x, y, z = env[:, 0], env[:, 1], env[:, 2]
            key = np.round(x, 8) + 1j * np.round(y, 8)
            _, uniq = np.unique(key, return_index=True)
            xu, yu, zu = x[uniq], y[uniq], z[uniq]
            if xu.size < 4:
                continue
            tri = mtri.Triangulation(xu, yu)
            t = tri.triangles
            P = np.stack([xu, yu, zu], axis=1)
            a, b, c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
            elen = np.maximum.reduce([
                np.linalg.norm(a - b, axis=1),
                np.linalg.norm(b - c, axis=1),
                np.linalg.norm(a - c, axis=1)])
            keep = elen <= edge_max
            if keep.any():
                ax.plot_trisurf(xu, yu, zu, triangles=t[keep],
                                color=color, alpha=0.55, linewidth=0.15,
                                edgecolor=color, shade=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=8)
        ax.set_xlabel(f"digit-{digits[0]} mean CE", fontsize=7,
                      labelpad=-1)
        ax.set_ylabel(f"digit-{digits[1]} mean CE", fontsize=7,
                      labelpad=-1)
        ax.set_zlabel(f"digit-{digits[2]} mean CE", fontsize=7,
                      labelpad=-1)
        ax.tick_params(labelsize=6, pad=-1)
        if p == 0:
            ax.legend(handles=[
                Patch(facecolor=c, alpha=0.6,
                      label=f"{lbl} frontier sheet ({len(env)} envelope "
                            f"pts)")
                for lbl, c, env in env_series],
                fontsize=6.5, loc="upper left")
    fig.suptitle(
        f"MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]}: TEST "
        f"Pareto frontier as a shaded sheet (MODPO-style rendering)   "
        f"(matched budget B={budget:,.0f}; sheet + dots = lower envelope "
        f"of the nondominated set on a {nbins}x{nbins} log (F1,F2) grid; "
        f"triangles bridging gaps > {edge_max} CE removed)", fontsize=9.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.05,
                        wspace=0.10)
    fig.savefig(home / "front_test_surface.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _dominance_map_figure(home, fr_a, fr_b, name_a, name_b, digits,
                          budget, ngrid=240, tol=5e-3):
    """One-glance verdict figure (user request, Aug 26): for each pair
    of objectives at budget (u, v), which method reaches the LOWER
    third objective?  Colour = z_env(B) - z_env(A) where z_env(u,v) =
    min{F_k : point with the other two coords <= (u, v)} (the dominated
    -region envelope; prefix-min on a log grid).  Green = A deeper,
    orange = B deeper.  Three panels = the three orientations."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list(
        "adv", ["#d95f2b", "#ffffff", "#2ca02c"])
    lo = max(1e-3, 0.8 * min(float(fr_a.min()), float(fr_b.min())))
    G = np.geomspace(lo, LN3, ngrid)

    def _env(fr, i, j, k):
        Z = np.full((ngrid, ngrid), np.inf)
        xi = np.clip(np.searchsorted(G, fr[:, i]), 0, ngrid - 1)
        yj = np.clip(np.searchsorted(G, fr[:, j]), 0, ngrid - 1)
        for m in range(fr.shape[0]):
            if fr[m, k] < Z[yj[m], xi[m]]:
                Z[yj[m], xi[m]] = fr[m, k]
        Z = np.minimum.accumulate(np.minimum.accumulate(Z, axis=0),
                                  axis=1)
        return Z

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6))
    stats = []
    vmax = 0.15
    for ax, (i, j, k) in zip(axes, [(0, 1, 2), (1, 2, 0), (0, 2, 1)]):
        Za = _env(fr_a, i, j, k)
        Zb = _env(fr_b, i, j, k)
        both = np.isfinite(Za) & np.isfinite(Zb)
        D = np.where(both, Zb - Za, np.nan)
        only_a = np.isfinite(Za) & ~np.isfinite(Zb)
        only_b = np.isfinite(Zb) & ~np.isfinite(Za)
        D[only_a] = vmax          # region only A reaches at all
        D[only_b] = -vmax
        valid = ~np.isnan(D)
        g = float((D[valid] > tol).mean() * 100)
        o = float((D[valid] < -tol).mean() * 100)
        stats.append((g, o))
        ax.pcolormesh(G, G, np.clip(D, -vmax, vmax), cmap=cmap,
                      norm=TwoSlopeNorm(0.0, -vmax, vmax),
                      shading="auto", rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"digit-{digits[i]} CE budget", fontsize=10)
        ax.set_ylabel(f"digit-{digits[j]} CE budget", fontsize=10)
        ax.set_title(f"who reaches lower digit-{digits[k]} CE?\n"
                     f"GREEN {g:.0f}%  vs  orange {o:.0f}%",
                     fontsize=11)
        ax.tick_params(labelsize=8)
    g_all = np.mean([s[0] for s in stats])
    o_all = np.mean([s[1] for s in stats])
    fig.suptitle(
        f"MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]} — "
        f"TEST-front advantage map at equal budget B={budget:,.0f}:  "
        f"GREEN = {name_a} reaches deeper,  orange = {name_b} deeper.  "
        f"Across all three orientations: green {g_all:.0f}%, orange "
        f"{o_all:.0f}%.", fontsize=12, y=1.06)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=TwoSlopeNorm(0.0, -vmax, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label(f"depth advantage (CE); + = {name_a}", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(home / "dominance_map_test.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


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


VIEWS3 = [(22, -60), (18, -140), (34, 115)]   # reference style, Aug-26 restyle


def _row_3d(fig, series, digits, axis_label, lims=None):
    """Aug-26 restyle (user request, modelled on the breakable-bottles
    reference figure): ONE row of three 3-D views at fixed angles,
    LINEAR axes autoscaled to the plotted (windowed) points, dots only,
    per-panel angle caption.  ``series`` = [(label, color, marker,
    (n,3) points), ...]."""
    axes = []
    for p, (elev, azim) in enumerate(VIEWS3):
        ax = fig.add_subplot(1, 3, p + 1, projection="3d")
        for label, color, marker, pts in series:
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color,
                           marker=marker, s=14, depthshade=False,
                           alpha=0.85, label=label)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=8)
        ax.set_xlabel(f"digit-{digits[0]} {axis_label}", fontsize=7,
                      labelpad=-1)
        ax.set_ylabel(f"digit-{digits[1]} {axis_label}", fontsize=7,
                      labelpad=-1)
        ax.set_zlabel(f"digit-{digits[2]} {axis_label}", fontsize=7,
                      labelpad=-1)
        ax.tick_params(labelsize=6, pad=-1)
        if lims is not None:
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
            ax.set_zlim(*lims)
        axes.append(ax)
    axes[0].legend(fontsize=7, loc="upper left")
    return axes


def _row_proj(fig, series, digits, log_axes=True, lim=None):
    """Companion file: the three pairwise projections (the quantitative
    read the 3-D row cannot give)."""
    axes = fig.subplots(1, 3)
    for ax, (i, j) in zip(axes, PROJ):
        for label, color, marker, pts in series:
            if pts.size:
                ax.plot(pts[:, i], pts[:, j], ls="", color=color,
                        marker=marker, ms=3.5, label=label)
        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(right=LN3 * 1.08)
            ax.set_ylim(top=LN3 * 1.08)
            ax.axhline(LN3, ls=":", lw=0.8, color="gray")
            ax.axvline(LN3, ls=":", lw=0.8, color="gray")
        elif lim is not None:
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
        ax.set_xlabel(f"digit-{digits[i]}", fontsize=8)
        ax.set_ylabel(f"digit-{digits[j]}", fontsize=8)
        ax.grid(True, which="both", alpha=0.25, ls=":")
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="lower left")
    return axes


def make_figures(home: Path) -> None:
    home = Path(home)
    legs = _load_legs(home)
    rs = sorted({sm["extra"]["r"] for sm in legs
                 if sm["policy"] == "baseline"})
    digits = legs[0]["triple"]
    budget = legs[0]["budget"]

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
                    label=f"{label} — two-instrument audit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("full-simplex GN* — audit_v2 two-instrument meter")
        ax.set_title(
            f"MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]} "
            f"(patch-softplus, d={legs[0]['d']}) — pure fixed budget "
            f"B={budget:,.0f}\nshared segment unit; only the next-lambda "
            f"policy differs", fontsize=9.6)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower left")
        fig.tight_layout()
        fig.savefig(home / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- 3a + 3b: train front / test front (3-D + projections) ---------
    # K2 Aug-13 conventions carried over: adaptive CCP vs the SINGLE
    # best baseline (lowest final audit); off-window points (any loss
    # > ~ln 3: the divergence arms of vertex / edge grid nodes) are
    # dropped BEFORE plotting.
    metrics = {}
    for sm in legs:
        label, _c, _m = _style(sm, rs)
        F_tr = sm["_fvals"]
        F_te = sm["_test_ce"]
        fr_tr = F_tr[_nondominated_kd(F_tr)]
        fr_te = F_te[_nondominated_kd(F_te)]
        metrics[label] = {"front_train": fr_tr, "front_test": fr_te,
                          "n_points": int(F_tr.shape[0])}
    best_bl = min((sm for sm in legs if sm["policy"] == "baseline"),
                  key=lambda sm: sm["final_audit"])
    show = [sm for sm in legs
            if sm is best_bl or sm["policy"] != "baseline"]
    win = LN3 * 1.05
    for key, fname, sub in (
        ("front_train", "front_train.png",
         "TRAIN loss space (per-class mean CE, training subset)"),
        ("front_test", "front_test.png",
         "TEST loss space (per-class mean CE, ALL official t10k rows)"),
    ):
        series = []
        # baselines first, adaptive LAST: matplotlib 3-D does not
        # depth-sort across scatter artists, so the later series is
        # painted on top wherever clouds overlap (Aug-26 fix: the first
        # render painted orange over green in the dense central bowl,
        # visually inverting who owns the origin corner).
        for sm in sorted(show, key=lambda s: s["policy"] != "baseline"):
            label, color, marker = _style(sm, rs)
            fr = metrics[label][key]
            cw = fr[(fr <= win).all(axis=1)]
            series.append((f"{label} non-dom ({len(cw)}, central)",
                           color, marker, cw))
        fig = plt.figure(figsize=(13.4, 4.9))
        _row_3d(fig, series, digits, "mean CE")
        fig.suptitle(
            f"MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]}: "
            f"Pareto front in 3-D, {sub} (lower-left is better)   "
            f"(matched budget B={budget:,.0f}; adaptive CCP vs best "
            f"baseline r={best_bl['extra']['r']}; window <= ln 3)",
            fontsize=9.5)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.80,
                            bottom=0.05, wspace=0.10)
        fig.savefig(home / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(12.6, 4.2))
        _row_proj(fig, series, digits, log_axes=True)
        fig.suptitle(
            f"MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]} — "
            f"{sub}: pairwise projections (log-log, window <= ln 3)",
            fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(home / fname.replace(".png", "_proj.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- supplementary: where the TEST HV gap lives (Aug-26 request) ----
    ad_label = _style(next(sm for sm in legs
                           if sm["policy"] != "baseline"), rs)[0]
    bl_label = _style(best_bl, rs)[0]
    fr_a = metrics[ad_label]["front_test"]
    fr_a = fr_a[(fr_a <= LN3).all(axis=1)]
    fr_b = metrics[bl_label]["front_test"]
    fr_b = fr_b[(fr_b <= LN3).all(axis=1)]
    hv_gap_slices = _hv_slice_figure(home, fr_a, fr_b, ad_label, bl_label,
                                     digits, budget)

    # ---- supplementary: MODPO-style frontier sheet (Aug-26 request) ----
    surf_series = []
    for sm in sorted(show, key=lambda s: s["policy"] != "baseline"):
        label, color, _m = _style(sm, rs)
        cloud = sm["_test_ce"]
        cloud = cloud[(cloud <= LN3).all(axis=1)]
        fr = metrics[label]["front_test"]
        surf_series.append((label, color, cloud,
                            fr[(fr <= LN3).all(axis=1)]))
    _front_surface_figure(home, surf_series, digits, budget)
    _dominance_map_figure(home, fr_a, fr_b, ad_label, bl_label, digits,
                          budget)

    # ---- 4: test error fronts (all legs; Aug-26 reference style) --------
    err_series, err_max = [], 0.0
    for sm in sorted(legs, key=lambda s: s["policy"] != "baseline"):
        label, color, marker = _style(sm, rs)
        E = sm["_test_err"]
        fr = E[_nondominated_kd(E)]
        metrics[label]["front_test_err"] = fr
        err_max = max(err_max, float(fr.max()))
        err_series.append((f"{label} non-dom ({len(fr)})", color, marker,
                           fr))
    lim = min(1.0, err_max * 1.15 + 0.01)
    fig = plt.figure(figsize=(13.4, 4.9))
    _row_3d(fig, err_series, digits, "test err (1 - recall)",
            lims=(0.0, lim))
    fig.suptitle(
        f"MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]}: "
        f"per-class TEST error front in 3-D (lower-left is better)   "
        f"(matched budget B={budget:,.0f})", fontsize=9.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80,
                        bottom=0.05, wspace=0.10)
    fig.savefig(home / "front_err_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(12.6, 4.2))
    _row_proj(fig, err_series, digits, log_axes=False, lim=lim)
    fig.suptitle(
        f"MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]} — "
        f"TEST error fronts: pairwise projections (1 - recall)",
        fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(home / "front_err_test_proj.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 5: best balanced TEST CE discovered so far vs budget / CPU ----
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
        f"MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]} — "
        f"balanced TEST quality discovered along the run (prefix-best "
        f"mean per-class test CE), B={budget:,.0f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(home / "test_ce_vs_budget.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- metrics json + README ----------------------------------------
    union_tr = np.vstack([m["front_train"] for m in metrics.values()])
    union_tr = union_tr[_nondominated_kd(union_tr)]
    central_tr = union_tr[(union_tr <= LN3).all(axis=1)]
    ref3 = (LN3, LN3, LN3)
    out = {}
    for label, m in metrics.items():
        d = _front_metrics(m["front_train"], union_tr)
        cen = (_front_metrics(m["front_train"], central_tr)
               if central_tr.size else
               dict(igd_to_union=None, maxdist_to_union=None))
        fc = m["front_train"][(m["front_train"] <= LN3).all(axis=1)]
        te = m["front_test"][(m["front_test"] <= LN3).all(axis=1)]
        out[label] = {
            "n_points": m["n_points"],
            "n_front_train": int(m["front_train"].shape[0]),
            "n_front_test": int(m["front_test"].shape[0]),
            "n_front_test_err": int(m["front_test_err"].shape[0]),
            "igd_to_union_train": d["igd_to_union"],
            "maxdist_to_union_train": d["maxdist_to_union"],
            "igd_central_train": cen["igd_to_union"],
            "maxdist_central_train": cen["maxdist_to_union"],
            "hv_central_train": _hv_3d(fc, ref3),
            "hv_central_test": _hv_3d(te, ref3),
        }
    out["_hv_gap_test_slice_check"] = {
        "adaptive_minus_best_baseline": hv_gap_slices,
        "note": ("integral of the 2-D slice-area difference over z "
                 "(hv_slices_test.png); equals the difference of the two "
                 "hv_central_test values up to z-grid discretisation")}
    out["_conventions"] = {
        "central_bound": LN3,
        "hv_reference": [LN3, LN3, LN3],
        "note": ("central = all three losses <= ln 3 (guess-level CE of "
                 "the balanced triple); union/reference from TRAIN fronts "
                 "of all legs; 3-D HV by z-sweep over the 2-D staircase; "
                 "test values on ALL official t10k rows")}
    (home / "front_metrics.json").write_text(
        json.dumps(_json_ready(out), indent=2), encoding="utf-8")

    rows = "\n".join(
        f"| {label} | {v['n_points']:,} | {v['n_front_train']} "
        f"| {v['n_front_test']} | {v['hv_central_train']:.4f} "
        f"| {v['hv_central_test']:.4f} |"
        for label, v in out.items() if not label.startswith("_"))
    (home / "README.md").write_text(f"""# K3 MNIST triple {digits[0]} vs {digits[1]} vs {digits[2]} — pure fixed budget campaign (Aug 26, 2026)

Legs: baseline simplex grids r in {rs} + adaptive CCP, all at
s = {legs[0]['s']}, B = {budget:,.0f} grad-equivalents, batch
{legs[0]['config_instance']['msvrg_batch']}, per_class =
{legs[0]['per_class']} (balanced maximum), d = {legs[0]['d']}.
Quality: audit_v2 two-instrument meter (IPOPT strict multistart + heavy
CCP) at every checkpoint + dense simplex-grid lower-bound cross-check at
the final stack (the K=2 exact 1-D meter has no K=3 analogue).  Test
values: ALL official t10k rows of the three digits.  Figures (Aug-26
restyle, modelled on the breakable-bottles reference layout):
gn_vs_grads, gn_vs_cpu, front_train + front_test (adaptive CCP vs the
best baseline only: ONE row of three 3-D views at fixed angles
(22,-60)/(18,-140)/(34,115), linear axes, window <= ln 3; pairwise
log-log projections in the companion *_proj.png), front_err_test (all
legs, same layout + companion projections), test_ce_vs_budget.
Divergence arms of vertex / edge grid nodes lie outside the ln-3
window by design (no regularisation).

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
             sorted(CAMPAIGN_ROOT.glob("triple_*_B*")))
    if not homes:
        raise SystemExit("no triple homes found")
    for home in homes:
        if home.is_dir() and list(home.glob("*/summary.json")):
            make_figures(home)


if __name__ == "__main__":
    main()
