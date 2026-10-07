"""K>2: one panel per Uniform resolution r, comparing GRAB's upper bound with the Uniform value along the Uniform run.

At every Gradient-Call checkpoint of the Uniform run within the budget B, the Uniform value lb(UD) (the GN at an actual
weight, a lower bound on the Uniform maximum) is compared with GRAB's upper bound ub(GRAB) at GRAB's last checkpoint
within the same Gradient Calls (figure ..._calls) or the same CPU time (figure ..._cpu); green where
lb(UD) > ub(GRAB).  Each panel title gives the smallest ratio lb(UD) / ub(GRAB) over these checkpoints; the checkpoint
attaining it is marked and enlarged in an inset.  In the time figure, the Uniform checkpoints earlier than GRAB's first
checkpoint after theta_0 are drawn as open circles and left out of the ratio (GRAB has only theta_0 there).
Writes figures/<task>_bounds_grid_calls.png, figures/<task>_bounds_grid_cpu.png and figures/<task>_bounds_grid.json.

    python scripts/make_bounds_grid.py fruittree_d6 [--results results] [--figures figures]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, points

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=tuple(config.TASKS))
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
ap.add_argument("--figures", default=str(_setup.ROOT / "figures"))
a = ap.parse_args()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

task = a.task
K = envs.build(task)["K"]
if K <= 2:
    raise SystemExit(f"{task}: K = {K}; the metric is exact")
spec = config.TASKS[task]; B = spec["budget"]; rs = spec["uniform"]["values"]
meta, curve = points.upper_curve(a.results, task, K)
gx = dict(calls=np.array([c["calls"] for c in curve], float), cpu=np.array([c["cpu"] for c in curve]))
gub = np.array([c["gn"] for c in curve]); t_first = float(gx["cpu"][1])  # GRAB's first checkpoint after theta_0
times = points.timing(a.results, task)
C = dict(grab="#ff7f0e", ud="#1f77b4", ok="#d9eed9")
ncol = 3; nrow = -(-len(rs) // ncol)
out = Path(a.figures); out.mkdir(parents=True, exist_ok=True)
summary = dict(task=task, budget=B, first_grab_checkpoint_cpu=t_first, calls={}, cpu={})
for key, xlabel in (("calls", "Gradient Calls"), ("cpu", "Time (s)")):
    fig, axs = plt.subplots(nrow, ncol, figsize=(14.5, 4.1 * nrow), sharey=True, squeeze=False)
    ymin, ymax = np.inf, 0.
    for r, ax in zip(rs, axs.flat):
        m = json.loads((Path(a.results) / task / "uniform" / f"r{r}.json").read_text())
        med, _ = points._cpu(times, f"uniform/r{r}", m)
        idx = [i for i, c in enumerate(m["checkpoints"]) if c.get("kind") == "calls" and c["component_gradients"] <= B]
        x = np.array([m["checkpoints"][i]["component_gradients"] if key == "calls" else med[i] for i in idx], float)
        lb = np.array([m["checkpoints"][i]["gn"] for i in idx])
        keep = x >= t_first if key == "cpu" else np.ones(len(x), bool)
        x_ex, lb_ex, x, lb = x[~keep], lb[~keep], x[keep], lb[keep]
        k = np.array([np.flatnonzero(gx[key] <= v + 1e-12)[-1] for v in x])
        ub = gub[k]; ratio = lb / ub; j = int(np.argmin(ratio))
        ymin, ymax = min(ymin, ub.min(), lb.min()), max(ymax, lb.max(), ub.max())
        summary[key][r] = dict(checkpoints=len(x), min_ratio=float(ratio[j]), at=float(x[j]), lb=float(lb[j]),
                               ub=float(ub[j]), grab_checkpoint_calls=int(curve[k[j]]["calls"]),
                               all_above=bool(np.all(ratio > 1)),
                               left_out=[dict(at=float(u), lb=float(v)) for u, v in zip(x_ex, lb_ex)])
        x0 = min(x.min(), x_ex.min()) if len(x_ex) else x.min()
        xl, xr = x0 / 1.25, x.max() * 1.1
        g0 = np.flatnonzero(gx[key] <= x0 + 1e-12)[-1]
        sel = (np.arange(len(curve)) >= g0) & (gx[key] <= x.max() + 1e-12)
        gxs = np.concatenate([[xl], gx[key][sel][1:], [x.max()]]); gys = np.concatenate([gub[sel], [gub[sel][-1]]])
        ax.fill_between(x, ub, lb, where=lb > ub, color=C["ok"], lw=0, zorder=1)
        ax.step(gxs, gys, where="post", color=C["grab"], lw=1.8, zorder=3)
        ax.plot(x, lb, "-", color=C["ud"], lw=1.6, zorder=3)
        if len(x_ex):
            ymax = max(ymax, float(gub[g0]), lb_ex.max())
            ax.plot(x_ex, lb_ex, "o", mfc="white", mec=C["ud"], mew=1.4, ms=6, zorder=6)
        ax.plot([x[j], x[j]], [ub[j], lb[j]], ":", color="k", lw=1.2, zorder=4)
        ax.plot([x[j]] * 2, [ub[j], lb[j]], "o", color="k", ms=3.5, zorder=5)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(xl, xr)
        ax.grid(True, which="major", color="#e6e5e0", lw=.8)
        tail = f" from {t_first:.3f} s" if key == "cpu" else ""
        ax.set_title(f"r = {r}:  lb(UD) / ub(GRAB) $\\geq$ {ratio[j]:.3f}{tail}", fontsize=12)
        # inset: the three checkpoints on either side of the smallest ratio
        lo_i, hi_i = max(0, j - 3), min(len(x), j + 4); w = np.zeros(len(x), bool); w[lo_i:hi_i] = True
        span = x[hi_i - 1] - x[lo_i]
        lo, hi = (x[lo_i] - .02 * span, x[hi_i - 1] + .05 * span) if span > 0 else (.75 * x[j], 1.25 * x[j])
        ins = ax.inset_axes([.05, .04, .5, .34])
        ga = np.flatnonzero(gx[key] <= lo + 1e-12); ga = ga[-1] if len(ga) else 0
        gw = (np.arange(len(curve)) >= ga) & (gx[key] <= hi)
        ins.fill_between(x[w], ub[w], lb[w], where=lb[w] > ub[w], color=C["ok"], lw=0)
        ins.step(np.concatenate([[lo], gx[key][gw][1:], [hi]]), np.concatenate([gub[gw], [gub[gw][-1]]]),
                 where="post", color=C["grab"], lw=1.5)
        ins.plot(x[w], lb[w], ".-", color=C["ud"], lw=1.3, ms=4)
        ins.plot([x[j], x[j]], [ub[j], lb[j]], ":", color="k", lw=1)
        ins.plot([x[j]] * 2, [ub[j], lb[j]], "o", color="k", ms=3)
        ins.set_yscale("log"); ins.set_xlim(lo, hi)
        yl, yh = min(ub[w].min(), lb[w].min(), gub[gw].min()), max(ub[w].max(), lb[w].max())
        ins.set_ylim(yl / 1.03, yh * 1.03); ins.minorticks_off(); ins.set_xticks([]); ins.set_yticks([])
        ins.text(.03, .05, f"{ratio[j]:.3f} at {x[j]:,.0f}" if key == "calls" else f"{ratio[j]:.3f} at {x[j]:.3f} s",
                 transform=ins.transAxes, fontsize=8)
        ax.indicate_inset_zoom(ins, edgecolor="0.4", lw=.6)
    for ax in axs.flat[len(rs):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel(xlabel, fontsize=11)
    axs[0, 0].set_ylim(ymin / 3.2, ymax * 1.15)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Bound on $\max_\lambda\,\mathrm{GN}(\lambda,B_t)$", fontsize=11)
    h = [plt.Line2D([], [], color=C["grab"], lw=2), plt.Line2D([], [], color=C["ud"], lw=2),
         plt.Rectangle((0, 0), 1, 1, color=C["ok"])]
    fig.legend(h, ["GRAB: upper bound ub", "Unif Discrtztn (r): lower bound lb (its value)", "lb(UD) > ub(GRAB)"],
               loc="upper center", ncol=3, fontsize=12, frameon=False)
    note = (f"Every Uniform checkpoint within B = {B:,} gradient calls; GRAB at its last checkpoint within the same "
            f"{'gradient calls' if key == 'calls' else 'CPU time'}; dot: smallest ratio (enlarged).")
    if key == "cpu":
        note += (f"\nOpen circles: Uniform checkpoints before {t_first:.4f} s, GRAB's first checkpoint after "
                 "$\\theta_0$; there GRAB has only $\\theta_0$, whose bound lies above them, so they are left out of the "
                 "ratio.")
    fig.text(.5, .01, note, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05 if key == "cpu" else .03, 1, .95))
    fig.savefig(out / f"{task}_bounds_grid_{key}.png", dpi=200); plt.close(fig)
(out / f"{task}_bounds_grid.json").write_text(json.dumps(summary, indent=1) + "\n")
for key in ("calls", "cpu"):
    print(f"{task} ({key}): smallest lb(UD) / ub(GRAB) per r: " +
          ", ".join(f"r={r} {v['min_ratio']:.3f}" for r, v in summary[key].items()) +
          ("" if key == "calls" else f"; left out (before {t_first:.4f} s): " +
           ", ".join(f"r={r} {len(v['left_out'])}" for r, v in summary[key].items())))
