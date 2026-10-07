"""Convergence figure of one task and its reported numbers.

Left: the worst-case gradient norm max_lambda GN(lambda, B_t) against Gradient Calls; right: against training CPU
time.  The style is that of the MNIST figures: GRAB in orange, one blue square per Uniform r and one red
triangle per SURF N (mogym.points), each family with a dashed descriptive trend c + a (x/s)^(-p) fitted by least
squares on the logarithms (s = median x) from its leftmost to its rightmost point, and number labels placed by
scripts/labels.py.  The x axis runs to the larger of the GRAB budget and 1.1 x the farthest point.
  K=2  exact values; GRAB is drawn as a curve.
  K>2  GRAB: its upper bound at every checkpoint (scripts/upper_bounds.py), drawn as a step line (a bound holds until
       the next checkpoint, since the bundle only grows); baselines: their values, each the GN at an actual weight
       and hence a lower bound on their maximum.  All comparisons below then use GRAB's upper bound.

CPU times are the medians over the timing repeats of scripts/time_repeats.py (the single run if it was not run).
Also prints and writes figures/<task>_summary.json: the GRAB value at the budget B, each baseline's lowest point and
the ratio to the final GRAB value, the ratios to GRAB at the same Gradient Calls and at the same CPU time (median and
range over the timing repeats), where the GRAB curve first reaches that point, and the points above the GRAB curve.
Configured runs that are not plotted (no plateau, or point beyond B) are listed.

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
BOUND = {"grab": " (upper bound)", "Uniform": " (lower bound)", "SURF": " (lower bound)"}
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
meta, curve = (points.upper_curve if K > 2 else points.adaptive_curve)(a.results, task, K)
pts, skipped = points.points(a.results, task)
groups = {m: sorted([p for p in pts if p["method"] == m], key=lambda p: p["param"]) for m in ("Uniform", "SURF")}
groups = {m: g for m, g in groups.items() if g}
ad = {f: np.array([c[f] for c in curve], float) for f in ("calls", "cpu", "gn")}


fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
per_axis, handles = [], {}
for ax, f, xlabel in ((axs[0], "calls", "Gradient Calls"), (axs[1], "cpu", "Time (s)")):
    m = ad["calls"] > 0
    xmax = max(1.1 * max(p[f] for g in groups.values() for p in g), float(ad[f][-1]))
    if K > 2:  # the bound at a checkpoint holds until the next one
        gx, gy = np.append(ad[f][m], xmax), np.append(ad["gn"][m], ad["gn"][m][-1])
        handles["grab"], = ax.step(gx, gy, where="post", color=COL["grab"], lw=2.6, zorder=3)
        grab_line = (np.repeat(gx, 2)[1:], np.repeat(gy, 2)[:-1])
    else:
        handles["grab"], = ax.plot(ad[f][m], ad["gn"][m], "-", color=COL["grab"], lw=2.6, zorder=3)
        grab_line = None
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
    ax.set_yscale("log"); ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, fontsize=FS["label"]); ax.tick_params(labelsize=FS["tick"])
    ax.grid(True, color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    mv = (ad[f] > 0) & (ad[f] <= xmax)
    per_axis.append((ax, items, lines + [grab_line if grab_line is not None else (ad[f][mv], ad["gn"][mv])],
                     float(ad["gn"][mv].min()), float(ad["gn"][mv].max())))
xr = axs[0].get_xlim()[1]  # no tick label at the panel gap: left-panel ticks within 5% of its right edge dropped
axs[0].set_xticks([t for t in mticker.MaxNLocator(nbins=5).tick_values(0, xr) if 0 <= t <= .95 * xr])
if K == 2:  # K>2: the legend says which values are upper and lower bounds
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
fig.legend([handles["grab"]] + [handles[m] for m in groups],
           ["GRAB" + (BOUND["grab"] if K > 2 else "")] + [NAME[m] + (BOUND[m] if K > 2 else "") for m in groups],
           loc="upper center", ncol=1 + len(groups), fontsize=FS["legend"], frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.subplots_adjust(left=0.105 if K == 2 else 0.075, right=0.985, bottom=0.15, top=0.87, wspace=0.07)
for ax, items, lines, _, _ in per_axis:  # labels last: they need the final limits and layout
    place(ax, items, lines, fontsize=FS["num"])
out = _setup.Path(a.figures); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"{task}_convergence.png", dpi=300)
plt.close(fig)

end = curve[-1]
summary = dict(task=task, budget=config.TASKS[task]["budget"], not_plotted=[list(x) for x in skipped],
               grab_value="exact" if K == 2 else "upper bound (mogym.bounds); baselines: their values (lower bounds)",
               grab=dict(calls=end["calls"], cpu=end["cpu"], cpu_range=[min(end["cpu_repeats"]), max(end["cpu_repeats"])],
                         timing_repeats=len(end["cpu_repeats"]), gn=end["gn"], outer_iterations=len(meta["lambdas"])),
               baselines={})
print(f"{task}: GRAB {end['calls']:,} calls / {end['cpu']:.2f} s, GN {'' if K == 2 else 'upper bound '}{end['gn']:.4e}; "
      f"{len(meta['lambdas'])} outer iterations")
for method, v, reason in skipped:
    print(f"  not plotted: {method} {v} ({reason})")
for method, group in groups.items():
    p = min(group, key=lambda p: p["gn"])
    same_calls, same_time = points.best_within(curve, "calls", p["calls"]), points.best_within(curve, "cpu", p["cpu"])
    # same-time ratio in every timing repeat (GRAB curve and point timed in the same repeat): median and range
    n_rep = min(len(p["cpu_repeats"]), len(curve[0]["cpu_repeats"]))
    rep_ratios = [p["gn"] / points.best_within([dict(c, cpu=c["cpu_repeats"][k]) for c in curve], "cpu",
                                               p["cpu_repeats"][k]) for k in range(n_rep)]
    above = {f: [q["param"] for q in group if q["gn"] > points.best_within(curve, f, q[f])] for f in ("calls", "cpu")}
    row = dict(param=p["param"], gn=p["gn"], calls=p["calls"], cpu=p["cpu"],
               cpu_range=[min(p["cpu_repeats"]), max(p["cpu_repeats"])], ratio_to_final_grab=p["gn"] / end["gn"],
               ratio_same_calls=p["gn"] / same_calls, ratio_same_time=dict(
                   median=float(np.median(rep_ratios)), min=float(min(rep_ratios)), max=float(max(rep_ratios))),
               grab_reaches_at=dict(calls=points.first_reach(curve, "calls", p["gn"]),
                                    cpu=points.first_reach(curve, "cpu", p["gn"])),
               points=[dict(param=q["param"], calls=q["calls"], cpu=q["cpu"], gn=q["gn"], iterations=q["iterations"])
                       for q in group],
               points_above_curve=dict(calls=above["calls"], time=above["cpu"]))
    summary["baselines"][method] = row
    pre = "r" if method == "Uniform" else "N"
    print(f"  {method} lowest point {pre}={p['param']}: GN {p['gn']:.4e} at {p['calls']:,} calls / {p['cpu']:.2f} s; "
          f"ratio to final GRAB {row['ratio_to_final_grab']:.2f}x; same calls {row['ratio_same_calls']:.2f}x, same time "
          f"{np.median(rep_ratios):.2f}x (range [{min(rep_ratios):.2f}, {max(rep_ratios):.2f}] over {n_rep} repeat(s))")
    print(f"    points above the GRAB curve: calls {len(above['calls'])}/{len(group)}, time {len(above['cpu'])}/{len(group)}")
(out / f"{task}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
