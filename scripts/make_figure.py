"""Convergence figure of one task and the numbers reported in the paper.

Left: max_lambda GN(lambda, B_t) against Gradient Calls; right: against training CPU time.  Curve: the
adaptive run; squares / triangles: one point per Uniform r and SURF N (mogym.points).  Also prints and
writes figures/<task>_summary.json: the adaptive value at the end of its budget, each baseline's lowest
point and the ratio to the adaptive value (paper Table "MO-Gymnasium"), the adaptive GN at the same
Gradient Calls / CPU time as that point, and where the adaptive curve first reaches it.

    python scripts/make_figure.py fishwood [--results results] [--figures figures]
"""
import argparse
import json

import numpy as np

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, points

YL = r"$\max_{\lambda\in\Delta_K}\mathrm{GN}(\lambda,\mathcal{B}_t)$"
LABELED = {"bb": {5, 10, 15, 20, 21, 22, 23, 24}}  # tasks with many points: label only these r

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=tuple(config.TASKS))
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
ap.add_argument("--figures", default=str(_setup.ROOT / "figures"))
a = ap.parse_args()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

task = a.task
K = envs.build(task)["K"]
meta, curve = points.adaptive_curve(a.results, task, K)
pts = points.points(a.results, task, K)
U = sorted([p for p in pts if p["method"] == "Uniform"], key=lambda p: p["param"])
S = sorted([p for p in pts if p["method"] == "SURF"], key=lambda p: p["param"])

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), sharey=True)
for ax, f, xl in zip(axes, ("calls", "cpu"), ("Gradient Calls", "Time (s)")):
    xa = np.array([c[f] for c in curve], float)
    if f == "cpu":
        xa[0] = 0.
    if K == 3:
        ax.fill_between(xa, [c["lower"] for c in curve], [c["upper"] for c in curve], color="#d62728", alpha=.13, lw=0)
    ax.plot(xa, [c["gn"] for c in curve], color="#d62728", lw=2.2, label="Adaptive Bundle Method", zorder=4)
    for group, mk, col, name, pre, off in ((U, "s", "#2171b5", "Unif Discrtztn (r)", "r", (4, 4)),
                                           (S, "^", "#2ca02c", "SURF (N)", "N", (-4, -9))):
        if not group:
            continue
        ax.scatter([p[f] for p in group], [p["gn"] for p in group], marker=mk, s=40, facecolor="white",
                   edgecolor=col, lw=1.2, label=name, zorder=5)
        shown = sorted((p for p in group if p["param"] in LABELED.get(task, {p["param"]})), key=lambda p: p[f])
        for i, p in enumerate(shown):
            o = ((3, 5) if i % 2 == 0 else (3, -10)) if task in LABELED else off
            ax.annotate(f"{pre}={p['param']}", (p[f], p["gn"]), xytext=o, textcoords="offset points",
                        color=col, fontsize=6.3, ha="left" if pre == "r" else "right")
    ax.set_xlabel(xl); ax.set_ylabel(YL)
    ax.set_yscale("log"); ax.grid(alpha=.22, linestyle=":", which="both")
    ax.spines[["top", "right"]].set_visible(False); ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="y", labelleft=True); ax.margins(x=.045)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, .005), ncol=len(l), frameon=False, fontsize=9)
fig.tight_layout(rect=[0, .1, 1, 1])
out = _setup.Path(a.figures); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"{task}_convergence.png", dpi=230, bbox_inches="tight", pad_inches=.1)
plt.close(fig)

end = curve[-1]
summary = dict(task=task, adaptive=dict(calls=end["calls"], cpu=end["cpu"], gn=end["gn"], lower=end["lower"],
                                        upper=end["upper"], outer_iterations=len(meta["lambdas"])), baselines={})
print(f"{task}: adaptive {end['calls']:,} calls / {end['cpu']:.2f} s, GN {end['gn']:.4e}"
      + (f" [{end['lower']:.4e}, {end['upper']:.4e}]" if K == 3 else "")
      + f"; {len(meta['lambdas'])} outer iterations")
for method, group in (("Uniform", U), ("SURF", S)):
    if not group:
        continue
    p = min(group, key=lambda p: p["gn"])
    same_calls, same_time = points.best_within(curve, "calls", p["calls"]), points.best_within(curve, "cpu", p["cpu"])
    above = {f: [q["param"] for q in group if q["gn"] > points.best_within(curve, f, q[f])] for f in ("calls", "cpu")}
    row = dict(param=p["param"], gn=p["gn"], lower=p["lower"], upper=p["upper"], calls=p["calls"], cpu=p["cpu"],
               iterations_to_stop=p["iterations"], ratio_to_final_adaptive=p["gn"] / end["gn"],
               adaptive_same_calls=same_calls, ratio_same_calls=p["gn"] / same_calls,
               adaptive_same_time=same_time, ratio_same_time=p["gn"] / same_time,
               adaptive_reaches_at=dict(calls=points.first_reach(curve, "calls", p["gn"]),
                                        cpu=points.first_reach(curve, "cpu", p["gn"])),
               points_above_curve=dict(calls=above["calls"], time=above["cpu"], all=[q["param"] for q in group]))
    summary["baselines"][method] = row
    pre = "r" if method == "Uniform" else "N"
    print(f"  {method} lowest point {pre}={p['param']}: GN {p['gn']:.4e} at {p['calls']:,} calls / {p['cpu']:.2f} s "
          f"({p['iterations']} {'sweeps' if method == 'Uniform' else 'rounds'} to stop); ratio to final adaptive "
          f"{row['ratio_to_final_adaptive']:.2f}x; same calls {row['ratio_same_calls']:.2f}x, same time "
          f"{row['ratio_same_time']:.2f}x; adaptive reaches it at {row['adaptive_reaches_at']['calls']:,} calls / "
          f"{row['adaptive_reaches_at']['cpu']:.2f} s")
    print(f"    points above the adaptive curve: calls {len(above['calls'])}/{len(group)}, "
          f"time {len(above['cpu'])}/{len(group)}")
(out / f"{task}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
