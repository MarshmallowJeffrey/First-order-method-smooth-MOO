"""Convergence figure of one task and the numbers reported in the paper.

Left: max_lambda GN(lambda, B_t) against Gradient Calls; right: against training CPU time.  The style is that of
the MNIST figures of the paper: GRAB as an orange curve, one blue square per Uniform r and one red triangle per
SURF N (mogym.points), each family with a dashed descriptive trend c + a (x/s)^(-p) fitted by least squares on
the logarithms (s = median x) from its leftmost to its rightmost point, and number labels placed by
scripts/labels.py.  The x axis runs to the larger of the GRAB budget and 1.1 x the farthest point.

Also prints and writes figures/<task>_summary.json: the GRAB value at the end of its budget, each baseline's
lowest point and the ratio to the final GRAB value (paper Table "MO-Gymnasium"), the GRAB value at the same
Gradient Calls / CPU time as that point, where the GRAB curve first reaches it, and the points above the curve.

    python scripts/make_figure.py fishwood [--results results] [--figures figures]
"""
import argparse
import json

import numpy as np
from scipy.optimize import least_squares

import _setup  # noqa: F401  (one thread, repository on sys.path)
from mogym import config, envs, points

COL = {"grab": "#ff7f0e", "Uniform": "#1f77b4", "SURF": "#d62728"}
MARK = {"Uniform": "s", "SURF": "^"}
NAME = {"Uniform": "Unif Discrtztn (r)", "SURF": "SURF (N)"}
YLAB = r"$\max_{\lambda\in\Delta_K}\,\mathrm{GN}(\lambda,B_t)$"
FS = dict(label=16, tick=14, legend=14, num=11)


def fit_trend(x, y):
    """f(x) = c + a (x/s)^(-p), s = median of x, least squares on log f(x_i) - log y_i (multistart)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    s = float(np.median(x))
    lo, hi = np.array([-30.0, -40.0, 1e-3]), np.array([30.0, 30.0, 10.0])

    def resid(th):
        la, lc, p = th
        return np.log(np.exp(lc) + np.exp(la) * (x / s) ** (-p)) - np.log(y)
    best = None
    ymin, ymed = float(y.min()), float(np.median(y))
    for p0 in (0.3, 0.7, 1.2, 2.0, 3.0):
        for lc0 in (np.log(ymin / 2), np.log(ymin / 10), -20.0):
            for la0 in (np.log(ymed), np.log(max(ymed - np.exp(lc0), 1e-12))):
                th0 = np.clip([la0, lc0, p0], lo, hi)
                try:
                    r = least_squares(resid, th0, bounds=(lo, hi), method="trf", max_nfev=5000)
                except Exception:
                    continue
                if r.success and (best is None or r.cost < best.cost):
                    best = r
    la, lc, p = best.x
    return dict(a=float(np.exp(la)), c=float(np.exp(lc)), p=float(p), s=s, x_min=float(x.min()), x_max=float(x.max()))


def trend(fit, xs):
    return fit["c"] + fit["a"] * (xs / fit["s"]) ** (-fit["p"])


ap = argparse.ArgumentParser()
ap.add_argument("task", choices=tuple(config.TASKS))
ap.add_argument("--results", default=str(_setup.ROOT / "results"))
ap.add_argument("--figures", default=str(_setup.ROOT / "figures"))
a = ap.parse_args()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from labels import place  # noqa: E402

task = a.task
K = envs.build(task)["K"]
meta, curve = points.adaptive_curve(a.results, task, K)
pts = points.points(a.results, task, K)
groups = {m: sorted([p for p in pts if p["method"] == m], key=lambda p: p["param"]) for m in ("Uniform", "SURF")}
groups = {m: g for m, g in groups.items() if g}
ad = {f: np.array([c[f] for c in curve], float) for f in ("calls", "cpu", "gn")}


fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
per_axis, handles = [], {}
for ax, f, xlabel in ((axs[0], "calls", "Gradient Calls"), (axs[1], "cpu", "Time (s)")):
    m = ad["calls"] > 0
    handles["grab"], = ax.plot(ad[f][m], ad["gn"][m], "-", color=COL["grab"], lw=2.6, zorder=3)
    items, lines = [], []
    for meth, group in groups.items():
        xs, ys = [p[f] for p in group], [p["gn"] for p in group]
        handles[meth], = ax.plot(xs, ys, MARK[meth], color=COL[meth], ms=9, mec="white", mew=0.8, ls="", zorder=5)
        if len(group) >= 3:
            fit = fit_trend(xs, ys)
            tx = np.geomspace(fit["x_min"], fit["x_max"], 300)
            ax.plot(tx, trend(fit, tx), "--", color=COL[meth], lw=1.8, alpha=0.9, zorder=2)
            dense = np.linspace(fit["x_min"], fit["x_max"], 600)
            lines.append((dense, trend(fit, dense)))
        items += [dict(x=p[f], y=p["gn"], text=str(p["param"]), color=COL[meth], key=(meth, p["param"]))
                  for p in group]
    xmax = max(1.1 * max(p[f] for g in groups.values() for p in g), float(ad[f][-1]))
    ax.set_yscale("log"); ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, fontsize=FS["label"]); ax.tick_params(labelsize=FS["tick"])
    ax.grid(True, color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    mv = (ad[f] > 0) & (ad[f] <= xmax)
    per_axis.append((ax, items, lines + [(ad[f][mv], ad["gn"][mv])], float(ad["gn"][mv].min()), float(ad["gn"][mv].max())))
xr = axs[0].get_xlim()[1]  # no tick label at the panel gap: left-panel ticks within 5% of its right edge dropped
axs[0].set_xticks([t for t in mticker.MaxNLocator(nbins=5).tick_values(0, xr) if 0 <= t <= .95 * xr])
axs[0].set_ylabel(YLAB, fontsize=FS["label"])
y_lo = min(p[3] for p in per_axis) / 1.35
y_hi = 1.9 * max(it["y"] for p in per_axis for it in p[1])
if K >= 3 and np.log10(y_hi / y_lo) < 1.2:
    y_hi = max(y_hi, 1.15 * max(p[4] for p in per_axis))
axs[0].set_ylim(y_lo, y_hi)
if np.log10(y_hi / y_lo) < 2.2:  # short range: label the 2x, 3x and 5x ticks as well
    axs[0].yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2.0, 3.0, 5.0)))
    axs[0].yaxis.set_minor_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False, minor_thresholds=(3, 3)))
    axs[0].tick_params(axis="y", which="minor", labelsize=FS["tick"] - 1)
    axs[1].tick_params(axis="y", which="both", labelleft=False)
fig.legend([handles["grab"]] + [handles[m] for m in groups], ["GRAB"] + [NAME[m] for m in groups],
           loc="upper center", ncol=1 + len(groups), fontsize=FS["legend"], frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.15, top=0.87, wspace=0.07)
for ax, items, lines, _, _ in per_axis:  # labels last: they need the final limits and layout
    place(ax, items, lines, fontsize=FS["num"])
out = _setup.Path(a.figures); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"{task}_convergence.png", dpi=300)
plt.close(fig)

end = curve[-1]
summary = dict(task=task, grab=dict(calls=end["calls"], cpu=end["cpu"], gn=end["gn"], lower=end["lower"],
                                    upper=end["upper"], outer_iterations=len(meta["lambdas"])), baselines={})
print(f"{task}: GRAB {end['calls']:,} calls / {end['cpu']:.2f} s, GN {end['gn']:.4e}"
      + (f" [{end['lower']:.4e}, {end['upper']:.4e}]" if K == 3 else "")
      + f"; {len(meta['lambdas'])} outer iterations")
for method, group in groups.items():
    p = min(group, key=lambda p: p["gn"])
    same_calls, same_time = points.best_within(curve, "calls", p["calls"]), points.best_within(curve, "cpu", p["cpu"])
    above = {f: [q["param"] for q in group if q["gn"] > points.best_within(curve, f, q[f])] for f in ("calls", "cpu")}
    row = dict(param=p["param"], gn=p["gn"], lower=p["lower"], upper=p["upper"], calls=p["calls"], cpu=p["cpu"],
               iterations_to_stop=p["iterations"], ratio_to_final_grab=p["gn"] / end["gn"],
               grab_same_calls=same_calls, ratio_same_calls=p["gn"] / same_calls,
               grab_same_time=same_time, ratio_same_time=p["gn"] / same_time,
               grab_reaches_at=dict(calls=points.first_reach(curve, "calls", p["gn"]),
                                    cpu=points.first_reach(curve, "cpu", p["gn"])),
               points=[dict(param=q["param"], calls=q["calls"], cpu=q["cpu"], gn=q["gn"]) for q in group],
               points_above_curve=dict(calls=above["calls"], time=above["cpu"], all=[q["param"] for q in group]))
    summary["baselines"][method] = row
    pre = "r" if method == "Uniform" else "N"
    print(f"  {method} lowest point {pre}={p['param']}: GN {p['gn']:.4e} at {p['calls']:,} calls / {p['cpu']:.2f} s "
          f"({p['iterations']} {'sweeps' if method == 'Uniform' else 'rounds'} to stop); ratio to final GRAB "
          f"{row['ratio_to_final_grab']:.2f}x; same calls {row['ratio_same_calls']:.2f}x, same time "
          f"{row['ratio_same_time']:.2f}x")
    print(f"    points above the GRAB curve: calls {len(above['calls'])}/{len(group)}, time {len(above['cpu'])}/{len(group)}")
(out / f"{task}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
