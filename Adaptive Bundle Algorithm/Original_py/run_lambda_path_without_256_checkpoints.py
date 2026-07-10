"""Lambda-adaptivity figure at K=3 (without-256-checkpoints track).

Purpose: visualise HOW the adaptive bundle method chooses its weights.
The weight domain at K=3 is the 2-simplex (a triangle).  The baseline's
weights are decided BEFORE seeing the problem: a uniform grid of
resolution r (C(r+2, 2) dots at K=3) — pure combinatorics of (K, r), so
no baseline run is needed and none is performed.  The adaptive method's
weight at every outer round is the solution of its max-min search (the
currently worst-served weight), so its weights are only known by RUNNING
it; `algorithm_adaptive` records them in `lambda_history`.  The figure
draws both on one triangle: uniform dots (baseline) versus the
round-ordered weight sequence (adaptive).

Configuration: the plateau sweep's K=3 run
(output/plateau/K3_p6_n30_h4_tanh_r6_B30000) — K=3, p=6, n=30,
hidden [4], tanh, seeds 7/8, max_inner=25, lambda_max_starts=64,
prune_inner=True, eval_every=1,200 — with two deliberate deviations:
the grid drawn for the baseline uses r=10 (66 nodes) as requested, and
the run is CERTIFIED mode (epsilon=0.001: stop as soon as the method's
own search value <= 2*eps/3) so the weight sequence ends at
certification instead of consuming the whole 30,000-gradient budget,
which stays in place as the fuse.  Runs on the without-256-checkpoints
track (self-reported checkpoints; irrelevant to which weights the
search selects, but keeps wall time down).

Usage:
    python run_lambda_path_without_256_checkpoints.py [--smoke]

Outputs (originals untouched):
    output/lambda_path_K3[_smoke]/
      lambda_path_triangle.png   the figure
      summary.json               curves + full lambda_history + step stats
      README.md                  what everything is (English)
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

# experiments first: it loads torch's OpenMP runtime before cyipopt.
from experiments import _make_mlp_problem, make_mlp_initial_point  # noqa: E402
from bundle import prefer_fused_joint_oracle  # noqa: E402
from baseline_without_256_checkpoints import _uniform_simplex_grid  # noqa: E402
from algorithm_without_256_checkpoints import algorithm_adaptive  # noqa: E402
from run_experiments import _result_curves, _json_ready  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent / "output"

GRID_RESOLUTION = 10  # the baseline dots' density (user-requested r)


def problem_config(smoke: bool) -> dict:
    # epsilon=0.001: certified mode — the run stops as soon as the method's
    # own search value drops to 2*eps/3, instead of consuming the whole
    # 30k-gradient budget (which stays in place as the fuse).
    base = dict(K=3, p=6, n=30, hidden_sizes=[4], activation="tanh",
                max_grad_evals=30_000, eval_every_n_grads=1_200,
                max_outer=1_000_000, max_inner=25, lambda_max_starts=64,
                prune_inner=True, epsilon=0.001)
    if smoke:
        base.update(max_grad_evals=3_000, eval_every_n_grads=300,
                    epsilon=0.01)
    return base


def simplex_to_plane(lams: np.ndarray) -> np.ndarray:
    """Project simplex points (m, 3) to 2D: vertices e1->(0,0), e2->(1,0),
    e3->(0.5, sqrt(3)/2) — an equilateral triangle."""
    lams = np.atleast_2d(np.asarray(lams, dtype=float))
    x = lams[:, 1] + 0.5 * lams[:, 2]
    y = (np.sqrt(3.0) / 2.0) * lams[:, 2]
    return np.column_stack([x, y])


def path_step_stats(lams: np.ndarray) -> dict:
    """L1 distances between consecutive lambda choices, to quantify how
    continuous the sequence is (small steps) versus jumping (large
    steps).  The simplex L1 diameter is 2; the r=10 grid spacing is
    1/10."""
    if len(lams) < 2:
        return {"n_rounds": int(len(lams))}
    steps = np.abs(np.diff(lams, axis=0)).sum(axis=1)
    return {
        "n_rounds": int(len(lams)),
        "l1_step_median": float(np.median(steps)),
        "l1_step_mean": float(np.mean(steps)),
        "l1_step_max": float(np.max(steps)),
        "share_of_steps_below_0.2": float(np.mean(steps < 0.2)),
        "share_of_steps_below_0.5": float(np.mean(steps < 0.5)),
    }


BASELINE_COLOR = "#1f77b4"   # blue, hollow circles (the fixed grid)
ADAPTIVE_COLOR = "#ff7f0e"   # orange (the adaptive weight sequence)
JUMP_COLOR = "0.72"          # light grey (de-emphasised global jumps)
CHAIN_THRESHOLD = 0.2        # L1; "local chain" = step below 2x grid spacing


def _draw_triangle_base(ax, grid, r, label=True, dot_size=26):
    """Triangle boundary + the baseline's uniform grid as hollow blue
    circles.  Returns the vertex coordinates."""
    tri = simplex_to_plane(np.eye(3))
    closed = np.vstack([tri, tri[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="black", lw=1.1, zorder=1)
    gxy = simplex_to_plane(grid)
    ax.scatter(gxy[:, 0], gxy[:, 1], s=dot_size, facecolors="none",
               edgecolors=BASELINE_COLOR, linewidths=1.1, alpha=0.85,
               zorder=2,
               label=(f"baseline weights: uniform grid, r={r} "
                      f"({len(grid)} nodes, fixed in advance)"
                      if label else None))
    ax.set_aspect("equal")
    ax.set_axis_off()
    return tri


def _chain_jump_segments(path_xy, lams):
    """Split consecutive-round segments into local chains (L1 step below
    CHAIN_THRESHOLD) and global jumps (the rest)."""
    steps = np.abs(np.diff(lams, axis=0)).sum(axis=1)
    segments = np.stack([path_xy[:-1], path_xy[1:]], axis=1)
    return steps, segments[steps < CHAIN_THRESHOLD], \
        segments[steps >= CHAIN_THRESHOLD]


def draw_main_figure(out_path, grid, lams, r, outcome_line):
    from matplotlib.collections import LineCollection
    path_xy = simplex_to_plane(lams)
    steps, chains, jumps = _chain_jump_segments(path_xy, lams)

    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    tri = _draw_triangle_base(ax, grid, r)
    ax.add_collection(LineCollection(jumps, colors=JUMP_COLOR, lw=0.6,
                                     alpha=0.55, zorder=3))
    ax.add_collection(LineCollection(chains, colors=ADAPTIVE_COLOR, lw=2.4,
                                     alpha=0.95, zorder=4))
    ax.scatter(path_xy[:, 0], path_xy[:, 1], color=ADAPTIVE_COLOR, s=22,
               zorder=5, edgecolors="none", alpha=0.9,
               label=f"adaptive weights: one per outer round "
                     f"({len(lams)} rounds)")
    # Legend proxies for the two segment styles.
    ax.plot([], [], color=ADAPTIVE_COLOR, lw=2.4,
            label=f"local chain: next weight moved < {CHAIN_THRESHOLD} in L1 "
                  f"({np.mean(steps < CHAIN_THRESHOLD):.0%} of steps)")
    ax.plot([], [], color=JUMP_COLOR, lw=0.8,
            label=f"global jump: next weight moved >= {CHAIN_THRESHOLD} "
                  f"({np.mean(steps >= CHAIN_THRESHOLD):.0%} of steps)")
    ax.scatter(*path_xy[0], marker="*", s=260, color="black", zorder=6,
               label="round 1 (start)")
    ax.scatter(*path_xy[-1], marker="X", s=120, color="black", zorder=6,
               label=f"round {len(lams)} (stop)")
    offsets = [(-0.04, -0.035), (-0.12, -0.035), (0.0, 0.02)]
    for k, (vx, vy) in enumerate(tri):
        dx, dy = offsets[k]
        ax.annotate(f"$e_{k + 1}$ (pure $F_{k + 1}$)", (vx, vy),
                    xytext=(vx + dx, vy + dy), fontsize=10)
    ax.set_title("Where each method spends its weight budget (K=3 simplex)\n"
                 + outcome_line, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(0.0, -0.02))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_step_size_figure(out_path, lams, r):
    """L1 step size per round: local chains vs jumps, the shrinking
    trend, and the reference levels (grid spacing, chain threshold)."""
    steps = np.abs(np.diff(lams, axis=0)).sum(axis=1)
    x = np.arange(2, len(lams) + 1)  # step i lands on round i+1
    colors = np.where(steps < CHAIN_THRESHOLD, ADAPTIVE_COLOR, JUMP_COLOR)

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.bar(x, steps, width=1.0, color=colors, linewidth=0)
    # Rolling median: the shrinking-step trend.
    win = 15
    if len(steps) >= win:
        kernel_med = np.array([np.median(steps[max(0, i - win + 1):i + 1])
                               for i in range(len(steps))])
        ax.plot(x, kernel_med, color="black", lw=1.8,
                label=f"rolling median (window {win})")
    # Quarter means: the numbers quoted in the analysis.
    q = len(steps) // 4
    for qi in range(4):
        lo = qi * q
        hi = (qi + 1) * q if qi < 3 else len(steps)
        mean = steps[lo:hi].mean()
        ax.hlines(mean, x[lo], x[hi - 1], colors="#d62728", lw=2.2,
                  linestyles="-",
                  label="quarter mean" if qi == 0 else None)
    ax.axhline(1.0 / r, color=BASELINE_COLOR, ls="--", lw=1.2,
               label=f"baseline grid spacing 1/r = {1.0 / r:g}")
    ax.axhline(CHAIN_THRESHOLD, color="0.4", ls=":", lw=1.2,
               label=f"chain/jump threshold {CHAIN_THRESHOLD}")
    ax.set_xlabel("outer round")
    ax.set_ylabel("L1 distance from previous weight\n(simplex diameter = 2)")
    ax.set_title("Consecutive weight movement: orange = local chain step, "
                 "grey = global jump; the trend shrinks as coverage fills in",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_quarters_figure(out_path, grid, lams, r):
    """Four small triangles: the weight sequence split into consecutive
    quarters of the run, showing coverage filling in and later quarters
    moving more locally."""
    from matplotlib.collections import LineCollection
    n = len(lams)
    bounds = [(i * n) // 4 for i in range(4)] + [n]
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.4))
    for qi, ax in enumerate(axes):
        lo, hi = bounds[qi], bounds[qi + 1]
        part = lams[lo:hi]
        part_xy = simplex_to_plane(part)
        _draw_triangle_base(ax, grid, r, label=False, dot_size=10)
        steps, chains, jumps = _chain_jump_segments(part_xy, part)
        ax.add_collection(LineCollection(jumps, colors=JUMP_COLOR, lw=0.5,
                                         alpha=0.5, zorder=3))
        ax.add_collection(LineCollection(chains, colors=ADAPTIVE_COLOR,
                                         lw=2.0, alpha=0.95, zorder=4))
        ax.scatter(part_xy[:, 0], part_xy[:, 1], color=ADAPTIVE_COLOR,
                   s=16, zorder=5, edgecolors="none", alpha=0.9)
        ax.set_title(f"rounds {lo + 1}-{hi}\n"
                     f"mean step {steps.mean():.2f}, "
                     f"chains {np.mean(steps < CHAIN_THRESHOLD):.0%}",
                     fontsize=9)
    fig.suptitle("The weight sequence quarter by quarter: coverage fills "
                 "in, movement becomes local (hollow blue = the fixed "
                 "baseline grid)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = problem_config(args.smoke)
    out_dir = OUTPUT_ROOT / ("lambda_path_K3_smoke" if args.smoke
                             else "lambda_path_K3")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    objectives, gradients, L, joint = _make_mlp_problem(
        K=cfg["K"], p=cfg["p"], n=cfg["n"], hidden_sizes=cfg["hidden_sizes"],
        seed=7, activation=cfg["activation"], w_true_scale=1.0)
    joint_oracle = prefer_fused_joint_oracle(joint)
    x0 = make_mlp_initial_point(K=cfg["K"], p=cfg["p"],
                                hidden_sizes=cfg["hidden_sizes"], seed=8)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = algorithm_adaptive(
            K=cfg["K"], d=int(x0.size), objectives=objectives,
            grad_objectives=gradients, L=L, x0=x0,
            max_outer=cfg["max_outer"], max_inner=cfg["max_inner"],
            epsilon=cfg["epsilon"],
            eval_every_n_grads=cfg["eval_every_n_grads"],
            target_cov=None, lambda_solver="ipopt", require_ipopt=True,
            lambda_max_starts=cfg["lambda_max_starts"],
            prune_inner=cfg["prune_inner"],
            max_grad_evals=cfg["max_grad_evals"],
            joint_oracle=joint_oracle, verbose=True)

    lams = np.asarray(res["lambda_history"], dtype=float)
    used = int(res["grad_evals_history"][-1])
    stats = path_step_stats(lams)
    grid = _uniform_simplex_grid(cfg["K"], GRID_RESOLUTION)

    certified = (cfg["epsilon"] is not None
                 and bool(res["pc_history"][-1] <= 2 * cfg["epsilon"] / 3))
    outcome_line = (
        f"adaptive: eps={cfg['epsilon']}, "
        + (f"CERTIFIED at {used} grads "
           f"(of the {cfg['max_grad_evals']} fuse)" if certified
           else f"fuse hit at {used} grads (not certified)")
        + f", {stats['n_rounds']} rounds, "
          f"final search value {res['pc_history'][-1]:.2e}")
    draw_main_figure(out_dir / "lambda_path_triangle.png", grid, lams,
                     GRID_RESOLUTION, outcome_line)
    draw_step_size_figure(out_dir / "lambda_step_sizes.png", lams,
                          GRID_RESOLUTION)
    draw_quarters_figure(out_dir / "lambda_path_quarters.png", grid, lams,
                         GRID_RESOLUTION)

    summary = {
        "config": _json_ready(cfg),
        "grid_resolution_for_figure": GRID_RESOLUTION,
        "data_seed": 7, "init_seed": 8, "d": int(x0.size),
        "checkpoint_metric": "self_reported (without-256 track)",
        "mode": "certified (epsilon), budget kept as the fuse",
        "certified": certified,
        "stop_search_value": float(res["pc_history"][-1]),
        "grad_evals_total": used,
        "bundle_size": int(len(res["bundle"].points)),
        "L_scale_final": _json_ready(res["L_scale_final"]),
        "inner_cap_hits": _json_ready(res["inner_cap_hits"]),
        "final_self_reported_gn": float(res["pc_history"][-1]),
        "lambda_history": _json_ready(lams),
        "path_step_stats": _json_ready(stats),
        "baseline_grid_size": int(grid.shape[0]),
        "curves": _result_curves(res),
        "runtime_warnings": sorted({str(w.message) for w in caught}),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""# Lambda adaptivity at K=3: uniform grid vs adaptive weight sequence

Produced by `Original_py/run_lambda_path_without_256_checkpoints.py`
(module docstring = full specification).  One adaptive CERTIFIED run
(epsilon={cfg['epsilon']}, stop at own-search value <= 2*eps/3, the
{cfg['max_grad_evals']}-gradient budget kept as the fuse) on the plateau
testbed: K=3, p={cfg['p']}, n={cfg['n']},
hidden={cfg['hidden_sizes']}, tanh, seeds 7/8, max_inner={cfg['max_inner']},
lambda_max_starts={cfg['lambda_max_starts']} — otherwise the
configuration of `output/plateau/K3_p6_n30_h4_tanh_r6_B30000`.
Outcome: {outcome_line}.

- `lambda_path_triangle.png` — the K=3 weight simplex as a triangle.
  Hollow blue circles: the baseline's weights (uniform grid,
  r={GRID_RESOLUTION}, {grid.shape[0]} nodes) — fixed by (K, r) before
  seeing the problem, so no baseline run is needed or performed.
  Orange dots: the weight the adaptive method's max-min search selected
  at each outer round (`lambda_history`; star = round 1, X = final
  round).  Consecutive-round movements are drawn in two styles: L1 step
  below {CHAIN_THRESHOLD} = thick orange segment (a LOCAL CHAIN — the
  worst weight slid to a neighbour), L1 step at or above
  {CHAIN_THRESHOLD} = thin grey line (a GLOBAL JUMP — the budget
  teleported to a new worst region).
- `lambda_step_sizes.png` — the L1 distance between consecutive weights
  per round (orange bars = chain steps, grey = jumps), with the grid
  spacing 1/r and the chain threshold as reference lines, a rolling
  median, and each quarter's mean — the shrinking-movement trend the
  analysis quotes.
- `lambda_path_quarters.png` — the sequence split into four consecutive
  quarters, one small triangle each: coverage fills in and movement
  becomes local over time.
- `summary.json` — the run record, the full `lambda_history`, and
  consecutive-step statistics (L1 distances between successive weights;
  the simplex L1 diameter is 2, the r={GRID_RESOLUTION} grid spacing is
  {1.0 / GRID_RESOLUTION}): median
  {stats.get('l1_step_median', float('nan')):.3f}, mean
  {stats.get('l1_step_mean', float('nan')):.3f}, max
  {stats.get('l1_step_max', float('nan')):.3f}, share below 0.2:
  {stats.get('share_of_steps_below_0.2', float('nan')):.0%}.
- `EXPLANATION_ZH.md` — why the adaptive weights are non-uniform and in
  what sense they do / do not form a continuous line (mechanism
  write-up, Chinese).

Caveat: each round's weight is the argmax found by the method's own
{cfg['lambda_max_starts']}-start search (a heuristic maximiser); the
without-256 track only changes what checkpoints record, not which
weights the search selects.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"DONE in {time.time() - t0:.0f}s -> {out_dir}")
    print("path stats:", json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
