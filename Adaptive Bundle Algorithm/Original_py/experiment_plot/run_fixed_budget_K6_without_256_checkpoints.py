"""Fixed-budget comparison: adaptive trajectory vs baseline endpoint audits.

NEW FILE (July 26, 2026, session 12).  Design record: ``Note/Jul_26_note.md``
part 2 (user-approved design, same date).  Engine files untouched.

Protocol
--------
One budget axis, ONE instrument.  Each completed baseline configuration
(r, node_tol) is a single POINT: x = its realized total grad-equivalents
(resp. CPU seconds), y = the strict full-simplex 64-start GN* of its
final delivered set (the delivery audit already stored in its summary).
The adaptive method is ONE budget-mode run of ``algorithm_adaptive_fast``
stopped by ``max_grad_evals`` = the budget B; its trajectory is audited
POST-HOC: at every checkpoint, the strict 64-start GN* of the bundle
PREFIX that existed at that moment (the same instrument, warm-started
along prefixes, cost off both axes — symmetric to the baseline's
``delivered_gn_strict_history``).  Every abscissa on the figure is
therefore an equal-budget comparison read with a single meter; the
in-run searches only steer the algorithm (their time stays inside the
adaptive CPU axis, as always on this track).

Baseline points are LOADED from the v2 comparison home (tol0.02/ and
tol0.01/ per-r summaries); nothing baseline-side is re-run.

Why the in-run tier may use fewer starts than the audit: the run's
searches are TARGETING (choose the next lambda); the figure's meter is
the fixed 64-start instrument applied post-hoc.  ``--targeting-starts``
(default 24) trades per-round search cost against targeting quality;
the meter is unaffected.  Disclosed in the README this script writes.

Usage:
    python run_fixed_budget_K6_without_256_checkpoints.py
        [--budget 80912] [--rel-target 0.05] [--targeting-starts 24]
        [--eval-every 2000] [--max-outer 5000] [--tag TAG] [--smoke]
"""
import argparse
import json
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

from math import comb  # noqa: E402
from pathlib import Path  # noqa: E402

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from experiments import make_mlp_initial_point  # noqa: E402  (torch first)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bundle import prefer_fused_joint_oracle  # noqa: E402
from objectives_torch_fast import make_mlp_nonconvex_fast  # noqa: E402
from algorithm_fast_without_256_checkpoints import (  # noqa: E402
    algorithm_adaptive_fast,
    _maximise_GN_fast,
    ipopt_available,
)
from baseline_svrg_certified_without_256_checkpoints import _GramSet  # noqa: E402
from run_experiments import DATA_SEED, INIT_SEED, _json_ready  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent.parent / "output"
V2_HOME = OUTPUT_ROOT / "pure_budget_without_256_checkpoints_SVRG_IPOPT_Baseline/baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints"
AUDIT_STARTS = 64  # the fixed instrument; never overridden by CLI

TRIAL_INSTANCE = dict(
    K=6, p=20, n=50_000, hidden_sizes=[96, 96], activation="tanh",
    msvrg_batch=4096, sampler_seed=41,
)
TRIAL_ADAPTIVE = dict(
    epsilon=1e-3, lambda_tier_mode="strict",
    msvrg_step_const=0.1, msvrg_momentum=0.5, msvrg_epoch_len=None,
    msvrg_max_segments=10, msvrg_trigger_rho=0.7, msvrg_trigger_patience=2,
    prune_grid_r=10,
)
SMOKE_INSTANCE = dict(
    K=6, p=6, n=300, hidden_sizes=[8], activation="tanh",
    msvrg_batch=60, sampler_seed=41,
)
SMOKE_ADAPTIVE = dict(
    epsilon=1e-2, lambda_tier_mode="strict",
    msvrg_step_const=0.1, msvrg_momentum=0.5, msvrg_epoch_len=5,
    msvrg_max_segments=4, msvrg_trigger_rho=0.7, msvrg_trigger_patience=2,
    prune_grid_r=3,
)

_ADAPT_KW = dict(color="#2ca02c", marker="^", ms=4, lw=1.8)
_TOL_MARKERS = {"0.02": ("x", 52, 1.8), "0.01": ("D", 34, 1.3)}
FIG_AXES = (
    ("grad_equiv_total", "grad_evals_history",
     "total gradient evaluations, grad-equivalents",
     "fixed_budget_gn_vs_grads.png"),
    ("wall_seconds", "cpu_times", "CPU time, seconds",
     "fixed_budget_gn_vs_cpu.png"),
)


def _collect_baseline_points(K: int):
    """Load (cost, audit) endpoints from the v2 home's per-r summaries."""
    points = []
    for tol_dir, tol_label in (("tol0.02", "0.02"), ("tol0.01", "0.01")):
        base = V2_HOME / tol_dir
        if not base.exists():
            continue
        for p in sorted(base.glob("r*/summary.json")):
            with open(p, "r", encoding="utf-8") as fh:
                s = json.load(fh)
            r = int(s["resolution"])
            points.append({
                "r": r,
                "node_tol": tol_label,
                "n_nodes": comb(r + K - 1, K - 1),
                "grad_equiv_total": float(s["grad_equiv_total"]),
                "wall_seconds": float(s["wall_seconds"]),
                "delivered_gn_strict": float(s["delivered_gn_strict"]),
                "stop_reason": s.get("stop_reason", "?"),
                "censored_nodes": int(s.get("censored_nodes", -1)),
                "source": str(p.relative_to(OUTPUT_ROOT)),
            })
    return points


def _audit_prefixes(Ms: np.ndarray, m_history, K: int):
    """Strict 64-start GN* of each checkpoint's bundle prefix (off-axis)."""
    t0 = time.time()
    vals, lams = [], []
    prev = None
    for m in m_history:
        m = int(m)
        gs = _GramSet(list(Ms[:m]), K)
        v, lam = _maximise_GN_fast(gs, prev_lam=prev, tier="strict",
                                   max_starts=AUDIT_STARTS)
        prev = np.asarray(lam, dtype=float)
        vals.append(float(v))
        lams.append([float(t) for t in prev])
        print(f"  [audit] m={m:5d}  GN*={v:.6e}", flush=True)
    return vals, lams, time.time() - t0


def _plot(fig_axis, adaptive, bl_points, args, out_dir: Path, cfg: dict,
          budget: float) -> None:
    axis_key, hist_key, x_label, fname = fig_axis
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    plot_key = ("audited_gn_envelope" if "audited_gn_envelope" in adaptive
                else "audited_gn_history")
    hx = np.asarray(adaptive[hist_key], dtype=float)
    hy = np.maximum(np.asarray(adaptive[plot_key], dtype=float),
                    np.finfo(float).tiny)
    positive = [float(v) for v in hx if v > 0]
    positive += [float(p[axis_key]) for p in bl_points if p[axis_key] > 0]
    x0_pseudo = (min(positive) / 3.0) if positive else 1e-3
    hx = np.where(hx > 0, hx, x0_pseudo)
    ax.plot(hx, hy, label="adaptive bundle — strict 64-start prefix audit",
            **_ADAPT_KW)

    reds = plt.get_cmap("Reds")
    rs = sorted({p["r"] for p in bl_points})
    log_hx, log_hy = np.log(hx), np.log(hy)
    for p in bl_points:
        color = reds(0.45 + 0.5 * rs.index(p["r"]) / max(1, len(rs) - 1))
        marker, size, lw = _TOL_MARKERS[p["node_tol"]]
        x_end = float(p[axis_key])
        if x_end <= 0:
            x_end = x0_pseudo
        y_end = p["delivered_gn_strict"]
        if marker == "D":
            ax.scatter([x_end], [y_end], marker=marker, s=size, zorder=6,
                       facecolors="none", edgecolor=color, linewidth=lw)
        else:
            ax.scatter([x_end], [y_end], marker=marker, s=size, zorder=6,
                       color=color, linewidth=lw)
        note = f"r={p['r']} tol={p['node_tol']}"
        # Equal-budget ratio where the trajectory covers this abscissa
        # (log-log interpolation, same convention as the equal-time
        # ratios elsewhere on this track).  Points within 2% past the
        # last checkpoint (checkpoint quantisation) clamp to the final
        # audited value instead of reading "beyond budget".
        if x_end <= hx.max() * 1.02:
            x_cmp = min(max(x_end, hx.min()), hx.max())
            y_ad = float(np.exp(np.interp(np.log(x_cmp), log_hx, log_hy)))
            note += f"\nx{y_end / y_ad:.1f} vs curve"
        else:
            note += "\nbeyond budget"
        ax.annotate(note, xy=(x_end, y_end), xytext=(5, 6),
                    textcoords="offset points", fontsize=7.2,
                    color="#8b1a1a")

    for tol_label, (marker, size, lw) in _TOL_MARKERS.items():
        if any(p["node_tol"] == tol_label for p in bl_points):
            if marker == "D":
                ax.scatter([], [], marker=marker, s=size, facecolors="none",
                           edgecolor=reds(0.7), linewidth=lw,
                           label=f"baseline final audit (node_tol={tol_label})")
            else:
                ax.scatter([], [], marker=marker, s=size, color=reds(0.7),
                           linewidth=lw,
                           label=f"baseline final audit (node_tol={tol_label})")
    ax.axvline(budget if axis_key == "grad_equiv_total" else
               float(np.asarray(adaptive[hist_key], dtype=float)[-1]),
               color="dimgray", ls="-.", lw=1.0, alpha=0.7,
               label=("budget B" if axis_key == "grad_equiv_total"
                      else "adaptive total CPU"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("strict full-simplex GN* (64-start audit — one meter)")
    ax.set_title(
        f"MLP K={cfg['K']}, p={cfg['p']}, n={cfg['n']}, "
        f"h={cfg['hidden_sizes']}, {cfg['activation']} — fixed-budget "
        f"comparison, single instrument\n"
        f"B={budget:,.0f} grad-equivalents; adaptive: strict targeting "
        f"({args.targeting_starts} starts), rel_target={args.rel_target}, "
        f"b={cfg['msvrg_batch']}; without-256 track",
        fontsize=9.6,
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_readme(out_dir: Path, cfg, args, summary) -> None:
    fx = summary["adaptive_extras"]
    aud = summary["adaptive"]
    pts = summary["baseline_points"]
    ca = summary["config_adaptive"]
    rows = "\n".join(
        f"| {p['r']} | {p['node_tol']} | {p['n_nodes']:,} "
        f"| {p['grad_equiv_total']:.0f} | {p['wall_seconds']:.0f} "
        f"| {p['delivered_gn_strict']:.4e} | {p['stop_reason']} |"
        for p in pts)
    readme = f"""# Fixed-budget comparison (single strict instrument, without-256 track)

Produced by `Original_py/run_fixed_budget_K6_without_256_checkpoints.py`
(July 26, 2026, session 12; design record `Note/Jul_26_note.md` part 2).

Protocol: every completed baseline configuration (r, node_tol) is one
POINT — x = its realized cost, y = the strict full-simplex 64-start
delivery audit from its own summary.  The adaptive method is ONE
budget-mode run (B = {args.budget:,.0f} grad-equivalents, stop =
`max_grad_evals`; stop_reason=`{fx['stop_reason']}`); its curve is the
strict 64-start GN* of the bundle PREFIX at each checkpoint, audited
post-hoc (audit cost {summary['audit_seconds']:.1f}s, OFF both axes).
One instrument everywhere; every abscissa is an equal-budget read.

In-run searches are TARGETING only (tier=strict, {args.targeting_starts}
starts, time inside the adaptive CPU axis); the meter is always the
64-start audit.  Inner loop: Momentum-SVRG (b={cfg['msvrg_batch']},
step_const={ca['msvrg_step_const']}, beta={ca['msvrg_momentum']},
rel_target={args.rel_target}, max_segments={ca['msvrg_max_segments']});
segment-cap rounds are acceptable in budget mode (budget burns, point
still delivered) — cap rounds this run: {fx['inner_cap_hits']}.

Adaptive endpoint: audited GN* {aud['audited_gn_history'][-1]:.4e} at
{aud['grad_evals_history'][-1]:.0f} grad-equivalents /
{aud['cpu_times'][-1]:.1f} s (bundle m={fx['m_final']},
post-prune audit {summary['post_prune_audit']:.4e}).

## Baseline points (loaded, not re-run)

| r | node_tol | N nodes | grad-equiv | wall s | strict audit | stop |
|---|----------|---------|------------|--------|--------------|------|
{rows}

## Figures

- `fixed_budget_gn_vs_grads.png` — grad-equivalents axis (headline).
- `fixed_budget_gn_vs_cpu.png` — CPU axis (adaptive pays its search
  time on-axis; baseline points include their solve time).

Annotations "xN.N vs curve" = baseline audit / adaptive audited value
at the SAME abscissa (>1: adaptive better at that budget).
"beyond budget" = the point lies past the adaptive run's end.

The plotted adaptive curve is the MONOTONE LOWER-BOUND ENVELOPE of the
raw prefix audits (at each checkpoint, the max over that and all later
audits).  Valid because the true prefix GN* is non-increasing in m and
every audit is a lower bound; raw per-checkpoint audits stay in
`summary.json` (`audited_gn_history`).

## Caveats

Single instance (seeds {DATA_SEED}/{INIT_SEED}/41), single machine,
serial runs.  Audits are heuristic lower bounds of an NP-hard max
(64 starts, warm-started); under-search can only under-report — and it
is the SAME instrument for both methods.  MLP torch runs are not
bit-reproducible in this environment (session-12 finding); trajectories
are one realization each.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=80_912.0,
                        help="grad-equivalent budget (default: r15's "
                             "realized cost at tol 0.02)")
    parser.add_argument("--rel-target", type=float, default=0.05)
    parser.add_argument("--targeting-starts", type=int, default=24,
                        help="starts for the IN-RUN targeting searches "
                             "(the audit meter always uses 64)")
    parser.add_argument("--eval-every", type=float, default=2000.0)
    parser.add_argument("--max-outer", type=int, default=5000,
                        help="round fuse only; the budget is the stop")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="redraw figures/README from the stored "
                             "summary.json (refreshing baseline points); "
                             "runs nothing")
    args = parser.parse_args()

    if not ipopt_available():
        raise RuntimeError("IPOPT is required but unavailable.")

    cfg = dict(SMOKE_INSTANCE if args.smoke else TRIAL_INSTANCE)
    adap = dict(SMOKE_ADAPTIVE if args.smoke else TRIAL_ADAPTIVE)
    if args.smoke:
        args.budget = min(args.budget, 600.0)
        args.eval_every = 50.0
        args.targeting_starts = 8
        args.max_outer = 60

    dirname = f"fixed_budget_B{args.budget:.0f}"
    if args.tag:
        dirname += f"_{args.tag}"
    if args.smoke:
        dirname += "_SMOKE"
    out_dir = V2_HOME / dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        p = out_dir / "summary.json"
        if not p.exists():
            raise SystemExit(f"--replot: no summary.json under {out_dir}")
        with open(p, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        cfg = dict(summary["config_instance"])
        ca = summary["config_adaptive"]
        args.rel_target = ca["rel_target"]
        args.targeting_starts = ca["targeting_starts"]
        adaptive = summary["adaptive"]
        if "audited_gn_envelope" not in adaptive:
            raw = np.asarray(adaptive["audited_gn_history"], dtype=float)
            adaptive["audited_gn_envelope"] = [
                float(v) for v in np.maximum.accumulate(raw[::-1])[::-1]]
        summary["baseline_points"] = _collect_baseline_points(cfg["K"])
        (out_dir / "summary.json").write_text(
            json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
        for fig_axis in FIG_AXES:
            _plot(fig_axis, adaptive, summary["baseline_points"], args,
                  out_dir, cfg, args.budget)
        _write_readme(out_dir, cfg, args, summary)
        print(f"REPLOTTED -> {out_dir}", flush=True)
        return

    t_build = time.time()
    objectives, gradients, L, joint, stoch = make_mlp_nonconvex_fast(
        K=cfg["K"], p=cfg["p"], n=cfg["n"],
        hidden_sizes=cfg["hidden_sizes"], seed=DATA_SEED,
        activation=cfg["activation"], w_true_scale=1.0,
        batch_size=cfg["msvrg_batch"], sampler_seed=cfg["sampler_seed"],
    )
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(K=cfg["K"], p=cfg["p"],
                                hidden_sizes=cfg["hidden_sizes"],
                                seed=INIT_SEED)
    print(f"[build] instance in {time.time() - t_build:.1f}s", flush=True)

    print(f"=== fixed-budget adaptive run: B={args.budget:,.0f}, "
          f"strict({args.targeting_starts}) targeting, "
          f"rel_target={args.rel_target} ===", flush=True)
    t_run = time.time()
    fast = algorithm_adaptive_fast(
        K=cfg["K"], d=int(x0.size), objectives=objectives,
        grad_objectives=gradients, L=L, x0=x0, stoch_oracle=stoch,
        epsilon=adap["epsilon"], max_outer=args.max_outer,
        eval_every_n_grads=args.eval_every,
        max_grad_evals=args.budget,
        lambda_max_starts=args.targeting_starts,
        lambda_tier_mode=adap["lambda_tier_mode"],
        msvrg_step_const=adap["msvrg_step_const"],
        msvrg_momentum=adap["msvrg_momentum"],
        msvrg_epoch_len=adap["msvrg_epoch_len"],
        msvrg_max_segments=adap["msvrg_max_segments"],
        msvrg_trigger_rho=adap["msvrg_trigger_rho"],
        msvrg_trigger_patience=adap["msvrg_trigger_patience"],
        msvrg_rel_target=args.rel_target,
        prune_grid_r=adap["prune_grid_r"],
        joint_oracle=joint_oracle, verbose=True,
        return_pre_prune=True,
    )
    wall = time.time() - t_run
    print(f"[run] done in {wall:.1f}s (stop={fast['stop_reason']}, "
          f"lambda-search {fast['lambda_search_seconds']:.1f}s)", flush=True)

    pp = fast["pre_prune"]
    Ms = np.asarray(pp["gram_stack"], dtype=float)
    m_hist = [int(v) for v in fast["m_history"]]
    if m_hist[-1] != Ms.shape[0]:
        # Late additions between the last checkpoint and delivery
        # (e.g. final search winners) belong to the endpoint audit.
        m_hist[-1] = int(Ms.shape[0])

    print(f"[audit] strict {AUDIT_STARTS}-start prefix audits at "
          f"{len(m_hist)} checkpoints (off-axis)", flush=True)
    audited, audit_lams, audit_seconds = _audit_prefixes(
        Ms, m_hist, cfg["K"])
    # Monotone lower-bound envelope: the true prefix GN* is non-increasing
    # in m and every audit is a lower bound, so the max over LATER audits
    # is a valid, tighter lower bound at each checkpoint (the same
    # never-understate principle as the baseline's delivery audit).  Raw
    # values stay in the summary; figures plot the envelope.
    audited_env = [float(v) for v in
                   np.maximum.accumulate(np.asarray(audited)[::-1])[::-1]]

    kept = fast["prune_report"].get("kept_indices")
    if kept:
        gs_post = _GramSet(list(Ms[np.asarray(kept, dtype=int)]), cfg["K"])
        v_post, _ = _maximise_GN_fast(
            gs_post, prev_lam=np.asarray(audit_lams[-1], dtype=float),
            tier="strict", max_starts=AUDIT_STARTS)
        post_prune_audit = float(v_post)
    else:
        post_prune_audit = float(audited[-1])

    adaptive = {
        "grad_evals_history": _json_ready(fast["grad_evals_history"]),
        "cpu_times": _json_ready(fast["cpu_times"]),
        "m_history": m_hist,
        "audited_gn_history": audited,
        "audited_gn_envelope": audited_env,
        "audit_lambdas": audit_lams,
        "inrun_pc_history_targeting_tier": _json_ready(fast["pc_history"]),
        "cov_history_inrun": _json_ready(fast["cov_history"]),
    }
    bl_points = _collect_baseline_points(cfg["K"])
    summary = {
        "metric": (f"strict {AUDIT_STARTS}-start in-family audit, one "
                   "instrument for both methods; adaptive prefix audits "
                   "post-hoc, off both axes"),
        "config_instance": _json_ready(cfg),
        "config_adaptive": _json_ready({**adap,
                                        "rel_target": args.rel_target,
                                        "targeting_starts": args.targeting_starts,
                                        "budget": args.budget,
                                        "eval_every": args.eval_every,
                                        "max_outer": args.max_outer}),
        "data_seed": DATA_SEED, "init_seed": INIT_SEED,
        "adaptive": adaptive,
        "adaptive_extras": {
            "stop_reason": fast["stop_reason"],
            "grad_equiv_total": _json_ready(fast["grad_equiv_total"]),
            "wall_seconds": wall,
            "lambda_search_seconds": _json_ready(fast["lambda_search_seconds"]),
            "joint_calls": _json_ready(fast["joint_calls"]),
            "ifo_minibatch_total": _json_ready(fast["ifo_minibatch_total"]),
            "segments_history": _json_ready(fast["segments_history"]),
            "inner_target_history": _json_ready(fast["inner_target_history"]),
            "inner_cap_hits": _json_ready(fast["inner_cap_hits"]),
            "L_scale_final": _json_ready(fast["L_scale_final"]),
            "m_final": int(Ms.shape[0]),
            "prune_report": _json_ready(fast["prune_report"]),
        },
        "post_prune_audit": post_prune_audit,
        "audit_seconds": audit_seconds,
        "baseline_points": bl_points,
        "runtime_note": ("MLP torch runs are not bit-reproducible in this "
                         "environment (session-12 finding); one realization."),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "bundle_grams.npz", gram_stack=Ms,
                        m_history=np.asarray(m_hist),
                        audited_gn=np.asarray(audited))

    if bl_points:
        for fig_axis in (
            ("grad_equiv_total", "grad_evals_history",
             "total gradient evaluations, grad-equivalents",
             "fixed_budget_gn_vs_grads.png"),
            ("wall_seconds", "cpu_times", "CPU time, seconds",
             "fixed_budget_gn_vs_cpu.png"),
        ):
            _plot(fig_axis, adaptive, bl_points, args, out_dir, cfg,
                  args.budget)
    _write_readme(out_dir, cfg, args, summary)

    if args.smoke:
        assert all(np.isfinite(v) for v in audited)
        assert len(audited) == len(m_hist) == len(adaptive["cpu_times"])
        assert np.isfinite(post_prune_audit)
        print("SMOKE OK:", {"ckpts": len(m_hist), "m_final": Ms.shape[0],
                            "final_audit": round(audited[-1], 6)},
              flush=True)
    print(f"DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
