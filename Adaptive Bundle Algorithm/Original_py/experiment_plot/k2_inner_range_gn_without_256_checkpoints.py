"""k2_inner_range_gn_without_256_checkpoints.py — same-range worst-GN
evaluation for the K = 2 dots campaign (review of Sep 9 2026).

The SURF legs train on the trimmed dial w in [0.05, 0.95] while the main
metric maximises over the full dial [0, 1]; every SURF run's worst weight
is w = 1.  This post-processing evaluates every saved run (CCP, 9 uniform,
9 SURF; seed 41; B = 20,000) on BOTH ranges from the saved Gram stacks:

    G_full(S)  = max_{w in [0, 1]}      min_{θ in S} ‖w ∇F4(θ) + (1−w) ∇F9(θ)‖
    G_inner(S) = max_{w in [0.05, 0.95]} min_{θ in S} ‖ ... ‖

S is the bundle prefix delivered at each checkpoint (its own ck_m); only
the OUTER evaluation range changes — bundle points are never filtered by
their training weight, and the gradients are those of the ridge-
regularised objectives exactly as stored in grams.npz.  No training and
no gradient call is made.

The meter is the campaign's own exact 1-D meter (dense grid of true
envelope values, closed-form polish of the winning cell, proven upper
bound), generalised here to an arbitrary interval [w_lo, w_hi]; on [0, 1]
it reproduces run_pure_budget_K2's ``exact_gn_1d`` bit for bit (checked
at start-up).  Squared values (lam^T M lam) and norm values (their square
roots) are both stored and named explicitly.

Outputs (next to the runs):  inner_range_gn_K2.json  (per checkpoint
curves and finals, lower value / upper bound / worst weight, both ranges),
inner_range_gn_K2.md (final table), worst_gn_full_vs_inner.png (curves,
both ranges, budget and wall-clock axes), vertex_arm_K2.json (loss values
at the vertex and near-vertex weights of the uniform runs, evidence for the
"finite but long arm" statement).

Usage:
    python k2_inner_range_gn_without_256_checkpoints.py            # grid 20,001 per checkpoint, 200,001 at the end
    python k2_inner_range_gn_without_256_checkpoints.py --grid 200001
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout  # noqa: F401
from run_pure_budget_K2_without_256_checkpoints import (  # noqa: E402
    _env_at,
    _quad_coeffs,
    exact_gn_1d,
)
from run_surf_compare_K2_without_256_checkpoints import MAIN_HOME  # noqa: E402

CORE = "adam_1e-3_b0.9"
SEED = 41
RS = [10, 20, 30, 40, 50, 70, 90, 120, 140]
NS = [10, 20, 30, 40, 50, 70, 90, 120, 140]
INNER = (0.05, 0.95)


def exact_gn_interval(Ms, w_lo=0.0, w_hi=1.0, grid_points=200_001, chunk=2_000,
                      polish=True, certify=True):
    """max over w in [w_lo, w_hi] of min_i lam(w)^T M_i lam(w), lam(w) = (w, 1-w).
    Same construction as exact_gn_1d (dense grid of exact envelope values,
    closed-form polish of the winning cell, proven cell-wise upper bound),
    restricted to the interval.  Returns (value, w_star, upper_bound) on the
    SQUARED scale; value is a true function value (lower bound of the max)."""
    A, B, C = _quad_coeffs(Ms)
    G = int(grid_points)
    grid = np.linspace(float(w_lo), float(w_hi), G)
    env = np.empty(G)
    act = np.empty(G, dtype=np.int64)
    for lo in range(0, G, int(chunk)):
        w = grid[lo:lo + int(chunk)]
        Q = A[:, None] * w[None, :] ** 2 + B[:, None] * w[None, :] + C[:, None]
        env[lo:lo + w.size] = Q.min(axis=0)
        act[lo:lo + w.size] = Q.argmin(axis=0)
    jbest = int(env.argmax())
    best_v, best_w = float(env[jbest]), float(grid[jbest])
    h = (float(w_hi) - float(w_lo)) / (G - 1)
    if polish:
        wl, wr = max(float(w_lo), best_w - h), min(float(w_hi), best_w + h)
        cand = set()
        for w0 in (wl, best_w, wr):
            q = A * w0 * w0 + B * w0 + C
            cand.update(np.argsort(q)[:64].tolist())
        cand = np.asarray(sorted(cand), dtype=int)
        ws = [wl, wr, best_w]
        Ac, Bc, Cc = A[cand], B[cand], C[cand]
        for i in range(cand.size):
            dA = Ac[i] - Ac[i + 1:]
            dB = Bc[i] - Bc[i + 1:]
            dC = Cc[i] - Cc[i + 1:]
            with np.errstate(all="ignore"):
                disc = dB * dB - 4.0 * dA * dC
                ok = (np.abs(dA) > 1e-300) & (disc >= 0.0)
                sq = np.sqrt(np.where(ok, disc, 0.0))
                for sgn in (+1.0, -1.0):
                    r = np.where(ok, (-dB + sgn * sq) / (2.0 * dA), np.nan)
                    r = r[(r >= wl) & (r <= wr)]
                    ws.extend(float(t) for t in r)
                lin = (~ok) & (np.abs(dB) > 1e-300)
                r = np.where(lin, -dC / np.where(lin, dB, 1.0), np.nan)
                r = r[(r >= wl) & (r <= wr)]
                ws.extend(float(t) for t in r)
        ws = np.asarray(ws, dtype=float)
        env_ws = _env_at(A, B, C, ws)
        j = int(np.argmax(env_ws))
        if float(env_ws[j]) > best_v:
            best_v, best_w = float(env_ws[j]), float(ws[j])
    ub = best_v
    if certify:
        Ai, Bi = A[act], B[act]
        s_here = 2.0 * Ai * grid + Bi
        s_l = np.maximum(np.abs(s_here[:-1]), np.abs(2.0 * Ai[:-1] * grid[1:] + Bi[:-1]))
        s_r = np.maximum(np.abs(s_here[1:]), np.abs(2.0 * Ai[1:] * grid[:-1] + Bi[1:]))
        u_cell = np.minimum(env[:-1] + s_l * h, env[1:] + s_r * h)
        ub = float(max(float(u_cell.max()), float(env[0]), float(env[-1]), best_v))
    return best_v, best_w, ub


def _selfcheck(Ms, grid):
    """on [0, 1] the interval meter must reproduce exact_gn_1d exactly"""
    v0, w0, u0 = exact_gn_1d(Ms, grid_points=grid, certify=True)
    v1, w1, u1 = exact_gn_interval(Ms, 0.0, 1.0, grid_points=grid)
    # chunking must not change the result: same grid, same envelope values
    assert v0 == v1 and w0 == w1 and u0 == u1, (v0, v1, w0, w1, u0, u1)


def runs_of(home):
    legs = [("adaptive", None, home / f"adaptive_ccp_seed{SEED}")]
    legs += [("uniform", r, home / f"uniform_r{r}_seed{SEED}") for r in RS]
    legs += [("surf", n, home / f"surf_N{n}_seed{SEED}") for n in NS]
    return legs


def evaluate(home, grid_ck, grid_final):
    out = {"home": str(home), "seed": SEED, "inner_range": list(INNER), "full_range": [0.0, 1.0],
           "grid_points_checkpoints": grid_ck, "grid_points_final": grid_final,
           "note": "squared = lam^T M lam (max over w of the min over the bundle prefix); norm = sqrt(squared); "
                   "value is a true envelope value (lower bound of the max), upper is the proven cell-wise bound; "
                   "bundle prefix = the first ck_m[k] delivered points of the run; gradients of the ridge objectives as stored",
           "runs": {}}
    t0 = time.time()
    for fam, p, d in runs_of(home):
        sm = json.loads((d / "summary.json").read_text())
        Ms = np.asarray(np.load(d / "grams.npz")["gram_stack"], dtype=float)
        ck_m = [int(v) for v in sm["ck_m"]]
        g, c = [float(v) for v in sm["ck_grads"]], [float(v) for v in sm["ck_cpu"]]
        if fam == "adaptive":
            _selfcheck(Ms, grid_ck)
        rec = {"family": fam, "param": p, "dir": d.name, "ck_grads": g, "ck_cpu": c, "ck_m": ck_m,
               "stored_final_audit_squared": float(sm["final_audit"]),
               "stored_final_audit_upper_squared": float(sm.get("final_audit_upper", float("nan"))),
               "stored_w_star": float(sm.get("w_star", float("nan")))}
        for tag, (lo, hi) in (("full", (0.0, 1.0)), ("inner", INNER)):
            vals, ws, ubs = [], [], []
            for k, m in enumerate(ck_m):
                gp = grid_final if k == len(ck_m) - 1 else grid_ck
                v, w, u = exact_gn_interval(Ms[:m], lo, hi, grid_points=gp)
                vals.append(v); ws.append(w); ubs.append(u)
            rec[f"{tag}_squared"] = vals
            rec[f"{tag}_norm"] = [float(np.sqrt(max(v, 0.0))) for v in vals]
            rec[f"{tag}_upper_norm"] = [float(np.sqrt(max(u, 0.0))) for u in ubs]
            rec[f"{tag}_w_star"] = ws
            rec[f"{tag}_final"] = {"norm": rec[f"{tag}_norm"][-1], "upper_norm": rec[f"{tag}_upper_norm"][-1],
                                   "w_star": ws[-1], "squared": vals[-1], "upper_squared": ubs[-1]}
        out["runs"][d.name] = rec
        print(f"  {d.name:22s} full {rec['full_final']['norm']:.6e} (w*={rec['full_final']['w_star']:.4f})  "
              f"inner {rec['inner_final']['norm']:.6e} (w*={rec['inner_final']['w_star']:.4f})  "
              f"stored {np.sqrt(sm['final_audit']):.6e}  [{time.time() - t0:.0f}s]", flush=True)
    return out


def vertex_arm(home):
    """loss values at the exact vertices and the nearest interior weights of
    the uniform runs: evidence that the arm is finite (ridge) but long"""
    res = {}
    for r in RS:
        z = np.load(home / f"uniform_r{r}_seed{SEED}" / "grams.npz")
        lam = np.asarray(z["seg_lams"], float); F = np.asarray(z["fvals"], float)
        ok = np.isfinite(lam[:, 0])
        w = lam[ok, 0]; F = F[ok]
        step = 1.0 / r
        def stat(mask):
            return {"n": int(mask.sum()), "F4_min": float(F[mask, 0].min()) if mask.any() else None,
                    "F9_min": float(F[mask, 1].min()) if mask.any() else None,
                    "F4_median": float(np.median(F[mask, 0])) if mask.any() else None,
                    "F9_median": float(np.median(F[mask, 1])) if mask.any() else None}
        res[f"r{r}"] = {"w=1 (digit 4 only)": stat(np.isclose(w, 1.0)),
                        f"w=1-1/r={1 - step:.4f}": stat(np.isclose(w, 1.0 - step)),
                        "w=0 (digit 9 only)": stat(np.isclose(w, 0.0)),
                        f"w=1/r={step:.4f}": stat(np.isclose(w, step))}
    return res


def figure(out, home):
    fam_colors = {"adaptive": "#ff7f00"}
    blues = plt.get_cmap("Blues"); reds = plt.get_cmap("Reds")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), dpi=170)
    for row, tag in enumerate(("full", "inner")):
        for col, (key, xlabel) in enumerate((("ck_grads", "total gradient evaluations"), ("ck_cpu", "wall-clock seconds"))):
            ax = axes[row, col]
            for name, rec in out["runs"].items():
                x = np.asarray(rec[key]); y = np.asarray(rec[f"{tag}_norm"])
                if rec["family"] == "adaptive":
                    ax.plot(x, y, color=fam_colors["adaptive"], lw=2.4, zorder=5, label="adaptive λ-bundle")
                elif rec["family"] == "uniform":
                    k = RS.index(rec["param"]); ax.plot(x, y, color=blues(0.35 + 0.6 * k / (len(RS) - 1)), lw=1.1,
                                                        label=f"uniform r = {rec['param']}")
                else:
                    k = NS.index(rec["param"]); ax.plot(x, y, color=reds(0.35 + 0.6 * k / (len(NS) - 1)), lw=1.1,
                                                        label=f"SURF N = {rec['param']}")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel, fontsize=10)
            rng = "w ∈ [0, 1] (full dial)" if tag == "full" else f"w ∈ [{INNER[0]}, {INNER[1]}] (inner dial)"
            ax.set_ylabel(f"worst-case gradient norm, {rng}", fontsize=9.5)
            ax.grid(True, which="major", alpha=0.3, lw=0.6); ax.grid(False, which="minor")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8.5)
            if row == 0 and col == 0:
                ax.legend(fontsize=6.5, ncol=3, loc="upper right", frameon=True, framealpha=0.9)
    fig.suptitle("MNIST 4 vs 9, B = 20,000, seed 41: the same 19 runs evaluated on the full dial (top) "
                 "and on the inner dial (bottom)", fontsize=11)
    fig.text(0.5, 0.01, "Every point of a curve is max over the stated w-range of min over the bundle prefix delivered at that checkpoint; "
             "bundle points are never filtered by their training weight; gradients of the ridge objectives (μ = 1e-3).",
             ha="center", fontsize=8, color="#333333")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out_png = home / "worst_gn_full_vs_inner.png"
    fig.savefig(out_png, dpi=170); plt.close(fig)
    return out_png


def figure_split(out, home, tag):
    """one dial per figure: two panels (budget axis, wall-clock axis)"""
    blues = plt.get_cmap("Blues"); reds = plt.get_cmap("Reds")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=170)
    for ax, (key, xlabel) in zip(axes, (("ck_grads", "total gradient evaluations"), ("ck_cpu", "wall-clock seconds"))):
        for name, rec in out["runs"].items():
            x = np.asarray(rec[key]); y = np.asarray(rec[f"{tag}_norm"])
            if rec["family"] == "adaptive":
                ax.plot(x, y, color="#ff7f00", lw=2.4, zorder=5, label="adaptive λ-bundle")
            elif rec["family"] == "uniform":
                k = RS.index(rec["param"]); ax.plot(x, y, color=blues(0.35 + 0.6 * k / (len(RS) - 1)), lw=1.1, label=f"uniform r = {rec['param']}")
            else:
                k = NS.index(rec["param"]); ax.plot(x, y, color=reds(0.35 + 0.6 * k / (len(NS) - 1)), lw=1.1, label=f"SURF N = {rec['param']}")
        ax.set_yscale("log"); ax.set_xlabel(xlabel, fontsize=10.5)
        rng = "w ∈ [0, 1] (full dial)" if tag == "full" else f"w ∈ [{INNER[0]}, {INNER[1]}] (inner dial)"
        ax.set_ylabel(f"worst-case gradient norm, {rng}", fontsize=10)
        ax.grid(True, which="major", alpha=0.3, lw=0.6); ax.grid(False, which="minor")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].legend(fontsize=6.8, ncol=3, loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    out_png = home / f"worst_gn_{tag}_curves.png"
    fig.savefig(out_png, dpi=170); plt.close(fig)
    return out_png


def table(out, home):
    ad = out["runs"][f"adaptive_ccp_seed{SEED}"]
    af, ai = ad["full_final"]["norm"], ad["inner_final"]["norm"]
    lines = ["| Method | G_full (norm) | certified upper | worst w (full) | G_inner (norm) | certified upper | worst w (inner) | G_full / CCP | G_inner / CCP |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = [f"adaptive_ccp_seed{SEED}"] + [f"uniform_r{r}_seed{SEED}" for r in RS] + [f"surf_N{n}_seed{SEED}" for n in NS]
    for name in order:
        rec = out["runs"][name]; f, i = rec["full_final"], rec["inner_final"]
        lab = {"adaptive": "Adaptive λ-bundle", "uniform": f"Uniform grid, r = {rec['param']}", "surf": f"SURF, N = {rec['param']}"}[rec["family"]]
        lines.append(f"| {lab} | {f['norm']:.4e} | {f['upper_norm']:.4e} | {f['w_star']:.4f} | {i['norm']:.4e} | {i['upper_norm']:.4e} | {i['w_star']:.4f} | {f['norm'] / af:.1f}× | {i['norm'] / ai:.1f}× |")
    md = "\n".join(lines) + (f"\n\nB = 20,000, seed 41; G_full = max over w in [0, 1], G_inner = max over w in [0.05, 0.95], both of min over the "
                             f"complete delivered bundle of ‖w∇F4 + (1−w)∇F9‖ (ridge objectives, μ = 1e-3); values are true envelope "
                             f"values (lower bounds), 'certified upper' the proven upper bound of the exact 1-D meter on a "
                             f"{out['grid_points_final']:,}-point grid; ratios use the CCP value of the same range.\n")
    (home / "inner_range_gn_K2.md").write_text(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(MAIN_HOME / CORE))
    ap.add_argument("--grid", type=int, default=20_001, help="grid points per checkpoint")
    ap.add_argument("--grid-final", type=int, default=200_001, help="grid points at the final checkpoint")
    ap.add_argument("--figures-only", action="store_true", help="redraw from the saved JSON, no recomputation")
    a = ap.parse_args()
    home = Path(a.home)
    if a.figures_only:
        out = json.loads((home / "inner_range_gn_K2.json").read_text())
        for tag in ("full", "inner"):
            print(f"figure -> {figure_split(out, home, tag)}")
        print(f"figure -> {figure(out, home)}")
        return
    out = evaluate(home, a.grid, a.grid_final)
    out["vertex_arm"] = vertex_arm(home)
    (home / "inner_range_gn_K2.json").write_text(json.dumps(out, indent=1))
    (home / "vertex_arm_K2.json").write_text(json.dumps(out["vertex_arm"], indent=2))
    png = figure(out, home)
    for tag in ("full", "inner"):
        print(f"figure -> {figure_split(out, home, tag)}")
    md = table(out, home)
    print(f"figure -> {png}")
    print(md)


if __name__ == "__main__":
    main()
