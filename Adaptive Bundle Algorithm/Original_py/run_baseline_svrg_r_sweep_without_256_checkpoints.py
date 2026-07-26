"""Baseline r-sweep: SVRG-certified uniform grids vs the v3 fast curve.

NEW FILE (July 20, 2026).  Design record: ``Note/Jul_20_note.md``.
Original files untouched; the engine is
``baseline_svrg_certified_without_256_checkpoints.baseline_svrg_certified``.

EDITED July 25, 2026 (user-authorised, see ``Note/Jul_25_note.md``):
the baseline trajectory LINES are restored on both figures.  Each r's
run is drawn as a checkpoint curve of its own grid meter (max over
grid nodes of the best-known value — the July-8 baseline's metric),
ending in a dotted vertical connector up to the filled square = the
delivered set's strict full-simplex score.  The vertical gap IS the
measured between-node error of that grid.

For each r in ``--r-list`` (default 10,20,30,40,50) this runner certifies
the uniform Δ_K grid at resolution r with the SAME Momentum-SVRG inner
solver, batch size, momentum, step rule, safeguard and accounting as the
v3 fast trial (same instance, seeds 7/8, b=4096), then scores the
delivered point set ONCE with the strict-tier in-family λ-search
(64 starts).  Each r becomes ONE point on two scatter figures
(grad-equivalents axis / CPU axis) against the UNCHANGED v3 fast
adaptive curve, which is reused from its summary.json (disclosed).

Still without-256-checkpoints: no external 256-start yardstick anywhere;
the delivery-time score is the method-family's own strict search, run
once per r, excluded from both cost axes.

The sweep is resumable: an r whose summary.json already exists is
skipped (--force re-runs).  Figures + sweep summary + READMEs are
rewritten after every completed r, so partial sweeps still deliver.

Usage:
    python run_baseline_svrg_r_sweep_without_256_checkpoints.py
        [--r-list 10,20,30,40,50] [--node-tol 0.02]
        [--solve-target-frac 0.25] [--share-mode gram|none]
        [--max-wall-per-r 14400] [--max-grads-per-r 2000000]
        [--smoke] [--force]
"""
import argparse
import json
import os
import time
import warnings
from math import comb
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

# experiments first: it loads objectives_torch (and torch's OpenMP
# runtime) before any cyipopt import — same convention as the trial
# runners.
from experiments import make_mlp_initial_point  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bundle import prefer_fused_joint_oracle  # noqa: E402
from objectives_torch_fast import make_mlp_nonconvex_fast  # noqa: E402
from baseline_svrg_certified_without_256_checkpoints import (  # noqa: E402
    baseline_svrg_certified,
)
from algorithm_fast_without_256_checkpoints import ipopt_available  # noqa: E402
from run_experiments import DATA_SEED, INIT_SEED, _json_ready  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent / "output"
# Renamed July 25 per the user's spec: the folder name states what the
# figures contain — multiple-r Momentum-SVRG baselines vs the fast
# adaptive method (see Note/Jul_25_note.md).
SWEEP_DIRNAME = "baseline_svrg_multi_r_vs_fast_without_256_checkpoints"
FAST_REF_CANDIDATES = [
    OUTPUT_ROOT / "fast_method_trials" / "v3_rel_target_two_tier",
    OUTPUT_ROOT / "fast_method_trials"
    / ("trial_K6_d11910_h96x96_tanh_n50000_eps0.001_fast_msvrg"
       "_without_256_checkpoints_v3"),
]

# Measured paces on this machine, used ONLY for the figure's floor lines
# and forecasts (never for any reported measurement): July 11 baseline
# log: 180,180 grads in 4157.5 s -> 0.1384 s per K-grad joint call.
JOINT_CALL_SECONDS = 0.1384

TRIAL_INSTANCE = dict(
    K=6, p=20, n=50_000, hidden_sizes=[96, 96], activation="tanh",
    msvrg_batch=4096, sampler_seed=41,
)
TRIAL_MSVRG = dict(
    msvrg_step_const=0.1, msvrg_momentum=0.5, msvrg_epoch_len=None,
    msvrg_max_segments=10, msvrg_trigger_rho=0.7, msvrg_trigger_patience=2,
)
SMOKE_INSTANCE = dict(
    K=6, p=6, n=300, hidden_sizes=[8], activation="tanh",
    msvrg_batch=60, sampler_seed=41,
)
SMOKE_MSVRG = dict(
    msvrg_step_const=0.1, msvrg_momentum=0.5, msvrg_epoch_len=5,
    msvrg_max_segments=4, msvrg_trigger_rho=0.7, msvrg_trigger_patience=2,
)

_FAST_KW = dict(color="#2ca02c", marker="^", ms=4, lw=1.8)


def _load_fast_curves():
    for cand in FAST_REF_CANDIDATES:
        p = cand / "summary.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                s = json.load(fh)
            return s["fast_adaptive"], str(cand)
    raise FileNotFoundError(
        "v3 fast summary.json not found in either candidate directory."
    )


def _run_one_r(r: int, cfg: dict, msvrg: dict, args, out_dir: Path) -> dict:
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
    d = int(x0.size)
    print(f"[r={r}] instance rebuilt in {time.time() - t_build:.1f}s "
          f"(fresh sampler, seed {cfg['sampler_seed']})", flush=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = baseline_svrg_certified(
            K=cfg["K"], d=d, objectives=objectives,
            grad_objectives=gradients, L=L, x0=x0,
            resolution=r, stoch_oracle=stoch,
            node_tol=args.node_tol,
            solve_target=args.node_tol * args.solve_target_frac,
            share_mode=args.share_mode,
            **msvrg,
            max_grad_evals=args.max_grads_per_r,
            max_wall_seconds=args.max_wall_per_r,
            eval_every_n_grads=args.ckpt_every_grads,
            lambda_max_starts=64,
            joint_oracle=joint_oracle,
            verify_certificates=True,
            return_points=False,
            return_grams=args.save_grams,
            verbose=True,
        )
    res["runtime_warnings"] = sorted({str(w.message) for w in caught})
    res["cfg_instance"] = dict(cfg)
    res["d"] = d
    res["data_seed"] = DATA_SEED
    res["init_seed"] = INIT_SEED

    r_dir = out_dir / f"r{r:02d}"
    r_dir.mkdir(parents=True, exist_ok=True)
    # Jul 25: --save-grams dumps the delivery-audit arrays to npz (a few
    # MB) and strips them from the JSON summary.  Consumed by
    # plot_between_node_gap_without_256_checkpoints.py.
    grams = res.pop("delivered_grams_stack", None)
    node_best_val = res.pop("node_best_val", None)
    grid = res.pop("grid", None)
    if args.save_grams and grams is not None:
        np.savez_compressed(
            r_dir / "delivery_audit.npz",
            delivered_grams=grams, node_best_val=node_best_val,
            grid=grid, lambda_star=np.asarray(res["lambda_star"]),
            node_tol=args.node_tol,
            delivered_gn_strict=res["delivered_gn_strict"],
        )
    (r_dir / "summary.json").write_text(
        json.dumps(_json_ready(res), indent=2), encoding="utf-8")
    return res


def _fully_certified(res: dict) -> bool:
    return (res["stop_reason"] == "completed"
            and res["censored_nodes"] == 0
            and res["n_unserved"] == 0)


def _plot_sweep(results: dict, fast: dict, fast_path: str, args,
                out_dir: Path, cfg: dict) -> None:
    K = cfg["K"]
    for axis_key, x_label, fname, floor_per_node, x_log in (
        ("grad_equiv_total", "total gradient evaluations, grad-equivalents",
         "gn_vs_grad_evals_baseline_r_sweep_vs_fast.png", float(K), True),
        ("wall_seconds", "CPU time, seconds",
         "gn_vs_cpu_time_baseline_r_sweep_vs_fast.png",
         JOINT_CALL_SECONDS, True),
    ):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        hist_key = ("grad_evals_history" if axis_key == "grad_equiv_total"
                    else "cpu_times")
        rs = sorted(results)

        # Shared pseudo-abscissa for x=0 checkpoints on the log axes
        # (same convention as the trial figures: the x=0 start is real,
        # a log axis just cannot draw it at 0).
        positive = [float(v) for v in np.asarray(fast[hist_key], dtype=float)
                    if v > 0]
        for r in rs:
            hx = results[r].get(hist_key) or []
            positive.extend(float(v) for v in hx if v > 0)
        x0_pseudo = (min(positive) / 3.0) if positive else 1e-3

        fx = np.asarray(fast[hist_key], dtype=float)
        fy = np.maximum(np.asarray(fast["cov_history"], dtype=float),
                        np.finfo(float).tiny)
        if x_log:
            fx = np.where(fx > 0, fx, x0_pseudo)
        ax.plot(fx, fy, label="Fast adaptive", **_FAST_KW)

        reds = plt.get_cmap("Reds")
        for j, r in enumerate(rs):
            res = results[r]
            color = reds(0.45 + (0.5 * j / max(1, len(rs) - 1)))
            x_end = float(res[axis_key])
            if x_log and x_end <= 0:
                x_end = x0_pseudo

            # Trajectory line, COMPARABLE meter (Jul 25 fix, Note §6):
            # strict full-simplex GN* of the delivered-set prefix at each
            # checkpoint — the same meter family as the fast curve, so
            # the lines share axes without cross-meter misreads.  The
            # grid meter (cov_history) stays in the summaries only.
            hist_y = res.get("delivered_gn_strict_history")
            if hist_y:
                hx = np.asarray(res[hist_key], dtype=float)
                hy = np.maximum(np.asarray(hist_y, dtype=float),
                                np.finfo(float).tiny)
                if x_log:
                    hx = np.where(hx > 0, hx, x0_pseudo)
                ax.plot(hx, hy, color=color, lw=1.5, alpha=0.95, zorder=3)

            filled = _fully_certified(res)
            # Jul 26 (user): small circles instead of large squares — the
            # marker only has to be distinguishable, not dominant.
            ax.scatter([x_end], [res["delivered_gn_strict"]],
                       marker="o", s=34, zorder=5,
                       facecolors=(color if filled else "none"),
                       edgecolor=("black" if filled else color),
                       linewidth=(0.7 if filled else 1.3))
            n_nodes = comb(r + K - 1, K - 1)
            ax.annotate(f"r={r}\nN={n_nodes:,}",
                        xy=(x_end, res["delivered_gn_strict"]),
                        xytext=(5, 7), textcoords="offset points",
                        fontsize=7.6, color="#8b1a1a")

        # Legend proxies for the baseline family: one entry per element,
        # not per r.  Jul 25 (user): no parenthetical text in labels; the
        # open-square entry appears ONLY if some run actually fused, and
        # the floor verticals carry no legend entry at all.
        proxy_c = reds(0.7)
        ax.plot([], [], color=proxy_c, lw=1.5, label="Baseline-SVRG")
        ax.scatter([], [], marker="o", s=34, facecolors=proxy_c,
                   edgecolor="black", linewidth=0.7,
                   label="final delivered set")
        if any(not _fully_certified(results[r]) for r in rs):
            ax.scatter([], [], marker="o", s=34, facecolors="none",
                       edgecolor=proxy_c, linewidth=1.3,
                       label="fused or censored")

        # Jul 26 (user): reference line at the fast method's best level, so
        # every baseline endpoint can be read as above or below it at a
        # glance.  fy is the fast curve's raw per-checkpoint value; its
        # best-so-far minimum is the level actually delivered.
        fast_best = float(np.min(fy))
        ax.axhline(fast_best, color=_FAST_KW["color"], ls="--", lw=1.2,
                   alpha=0.9,
                   label=f"fast adaptive best = {fast_best:.4f}")

        # No-reuse Algorithm-1 floor: >= one full joint call per node
        # (paper-faithful per-node oracle acceptance, no cached-Jacobian
        # reuse).  Drawn for the swept r values only, unlabelled.
        for r in args.r_list:
            n_nodes = comb(r + K - 1, K - 1)
            ax.axvline(n_nodes * floor_per_node, color="dimgray", ls=":",
                       lw=0.9, alpha=0.55)

        if x_log:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"full-simplex $GN^*$")
        ax.set_title(
            f"MLP K={K}, p={cfg['p']}, n={cfg['n']}, "
            f"h={cfg['hidden_sizes']}, {cfg['activation']} — "
            f"Baseline-SVRG r-sweep vs fast adaptive\n"
            f"node_tol={args.node_tol}, solve_target="
            f"{args.node_tol * args.solve_target_frac:g}, "
            f"share_mode={args.share_mode}, without-256 track",
            fontsize=10,
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _write_readmes(results: dict, fast_path: str, args, out_dir: Path,
                   cfg: dict) -> None:
    K = cfg["K"]
    rows = []
    for r in sorted(results):
        res = results[r]
        n_nodes = comb(r + K - 1, K - 1)
        rows.append(
            f"| {r} | {n_nodes:,} | {res['delivered_gn_strict']:.4e} "
            f"| {res.get('grid_worst_best_val', float('nan')):.4e} "
            f"| {res['grad_equiv_total']:.0f} | {res['wall_seconds']:.0f} "
            f"| {res['solved_nodes']} | {res['served_by_share']} "
            f"| {res['censored_nodes']} | {res['stop_reason']} |")
    table = (
        "| r | N nodes | delivered GN* (strict, full simplex) "
        "| grid cert end | grad-equivalents "
        "| wall s | solved | served-by-share | censored | stop |\n"
        "|---|---------|--------------------------------------"
        "|---------------|------------------"
        "|--------|--------|-----------------|----------|------|\n"
        + "\n".join(rows))

    readme = f"""# Baseline r-sweep with the v3 Momentum-SVRG inner solver (without-256 track)

Produced by `Original_py/run_baseline_svrg_r_sweep_without_256_checkpoints.py`
(July 20, 2026).  Design record: `Note/Jul_20_note.md`.  Engine:
`Original_py/baseline_svrg_certified_without_256_checkpoints.py` (new
file; originals untouched).

Each r is ONE run of Algorithm 1 on the uniform grid at resolution r,
with the SAME inner solver, batch (b={cfg['msvrg_batch']}), momentum,
step rule, descent safeguard and gradient-equivalent accounting as the
v3 fast trial (same instance, seeds {DATA_SEED}/{INIT_SEED}).  Per-node
service certificate: some delivered point x has lambda^T G(x) lambda <=
node_tol = {args.node_tol} (exact, full-gradient Grams — randomness can
never fake a certificate).  Each solved node is pushed to solve_target =
{args.node_tol * args.solve_target_frac:g} (= {args.solve_target_frac} x
node_tol, mirroring v3's rel_target=0.25).  share_mode =
`{args.share_mode}`: delivered Grams serve all still-unserved nodes by
cached lambda-reweighting (zero oracle calls — the same cache discipline
the fast method's lambda-search runs on).

Scatter y-value = GN* of the DELIVERED point set from ONE strict-tier
64-start in-family lambda-search at delivery time (metric cost excluded
from both axes; NOT the external 256-start yardstick — the track rule
holds).  x-value = the run's total grad-equivalents / wall seconds.

Fast adaptive curve reused unchanged from `{fast_path}` (disclosed;
its plotted values are the run's own CHEAP-tier searches — an
under-search of a maximiser can only under-report, per its README; the
user chose to keep that meter as-is on July 20).

## Results so far

{table}

Dotted vertical lines on the figures: the no-reuse Algorithm-1 floor
(>= one full joint call per node, N(r) x {K} grad-equivalents resp.
N(r) x {JOINT_CALL_SECONDS} s at the July-11 measured pace) — what the
paper-faithful per-node-oracle baseline would pay BEFORE any solving.

## Figures

- `gn_vs_grad_evals_baseline_r_sweep_vs_fast.png`
- `gn_vs_cpu_time_baseline_r_sweep_vs_fast.png`

Each r is drawn as a trajectory LINE plus one endpoint SQUARE.  Both
are on the COMPARABLE meter (July 25 fix, Note/Jul_25_note.md §6): the
line is the strict full-simplex 64-start GN* of the delivered-set
prefix at each checkpoint, computed post-hoc on cached Grams (cost in
`metric_seconds`, excluded from both axes); the square is its final
value.  This is the same meter family as the fast curve's y-axis, so
the lines share axes without cross-meter misreads.  The run's own GRID
meter (max over grid nodes of best-known value — the July-8 lag
semantics) is NOT plotted here; it is kept in each summary
(`cov_history`) and its endpoint appears in the table above as "grid
cert end".  The gap between "grid cert end" and "delivered GN*" is the
measured between-node error of that grid.  Open squares (if any) mark
fused or censored runs: their point is a lower bound on that r's cost
at this tolerance, not a converged measurement.

## Caveats

Single instance, seeds {DATA_SEED}/{INIT_SEED}; single machine;
delivered points are re-derivable exactly (deterministic seeds) but not
stored (memory).  The SVRG inner guarantee is expectation-type; every
certificate value is exact (full-gradient Grams).  Fresh sampler
(seed {cfg['sampler_seed']}) per r: runs are independently reproducible.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    rows_zh = table  # same table
    readme_zh = f"""# Baseline r-扫描（v3 Momentum-SVRG 内层，without-256 轨道）

由 `Original_py/run_baseline_svrg_r_sweep_without_256_checkpoints.py`
生成（2026-07-20）。设计记录：`Note/Jul_20_note.md`。引擎：
`Original_py/baseline_svrg_certified_without_256_checkpoints.py`
（新文件，原文件未动）。

每个 r 是 Algorithm 1 在分辨率 r 均匀网格上的一次完整运行，内层求解器
与 v3 fast 试验完全一致（同实例、种子 {DATA_SEED}/{INIT_SEED}、
b={cfg['msvrg_batch']}、同动量/步长/下降 safeguard/梯度当量记账）。
每节点服务证书：存在交付点 x 使 lambda^T G(x) lambda <= node_tol =
{args.node_tol}（精确、full-gradient Gram——随机性无法伪造证书）。被
求解的节点推进到 solve_target = {args.node_tol * args.solve_target_frac:g}
（= {args.solve_target_frac} x node_tol，对应 v3 的 rel_target=0.25）。
share_mode=`{args.share_mode}`：交付点的 Gram 以缓存 lambda 重加权服务
所有未服务节点（零 oracle 调用——与 fast 方法 lambda-search 的缓存
纪律同一原则）。

散点 y 值 = 交付点集在交付时刻一次严格档 64 起点族内 lambda-search 的
GN*（测量成本不进两条成本轴；不是外部 256 起点尺子——轨道规矩不变）。
x 值 = 该 run 的总 grad-equivalents / 墙钟秒。

Fast adaptive 曲线原样复用自 `{fast_path}`（已披露；其画的是粗档
搜索值，欠搜索只会低报——见其 README；用户 7 月 20 日决定保持该口径）。

## 当前结果

{rows_zh}

图中竖虚线：不复用缓存的 Algorithm-1 下限（每节点至少一次 full joint
调用，N(r) x {K} grad-equivalents / N(r) x {JOINT_CALL_SECONDS} s，
按 7 月 11 日实测速率）——忠实按论文逐节点 oracle 验收的 baseline 在
任何求解之前就要付的量。

## 图

- `gn_vs_grad_evals_baseline_r_sweep_vs_fast.png`
- `gn_vs_cpu_time_baseline_r_sweep_vs_fast.png`

每个 r 画成一条轨迹线 + 一个终点方块，两者都在可比口径上
（7 月 25 日修正，见 Note/Jul_25_note.md §6）：线是每个 checkpoint
时刻"已交付点集前缀"的严格档 64 起点全 simplex GN*（事后在缓存 Gram
上计算，成本计入 `metric_seconds`，不进两条成本轴）；方块是其最终值。
这与 fast 曲线的 y 轴同属一个口径族，两方法的线可以共享坐标轴而不产生
跨口径误读。run 自己的网格口径（所有节点最好已知值的最大值，7 月 8 日
滞后语义）不再入主图，保留在各 summary 的 `cov_history`，其终点在上表
"grid cert end" 列。"grid cert end" 与 "delivered GN*" 之差就是该网格
被实测出的节点间误差。空心方块（如有）= 触保险丝/censored 的 run：该
点是此容差下该 r 成本的下界，不是收敛测量。

## 注意

单实例（种子 {DATA_SEED}/{INIT_SEED}）、单机器；交付点坐标未存盘
（内存原因），但种子确定、可精确重导出。SVRG 内层保证为期望型；所有
证书值均精确（full-gradient Gram）。每个 r 用新采样器（种子
{cfg['sampler_seed']}），各 r 可独立复现。
"""
    (out_dir / "README_ZH.md").write_text(readme_zh, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default r list fixed July 25 (user decision): {10, 12, 15, 20}.
    parser.add_argument("--r-list", type=str, default="10,12,15,20")
    parser.add_argument("--node-tol", type=float, default=0.02)
    parser.add_argument("--solve-target-frac", type=float, default=0.25)
    parser.add_argument("--share-mode", type=str, default="gram",
                        choices=["gram", "none"])
    parser.add_argument("--max-wall-per-r", type=float, default=14_400.0,
                        help="per-r wall fuse in seconds (default 4 h)")
    parser.add_argument("--max-grads-per-r", type=float, default=2_000_000.0)
    parser.add_argument("--ckpt-every-grads", type=float, default=4_500.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="re-run r values whose summary.json exists")
    parser.add_argument("--save-grams", action="store_true",
                        help="dump delivery-audit arrays (Grams, per-node "
                             "values, grid) to delivery_audit.npz per r")
    parser.add_argument("--replot", action="store_true",
                        help="regenerate figures/READMEs from the stored "
                             "per-r summaries and exit; runs nothing")
    args = parser.parse_args()
    args.r_list = [int(tok) for tok in args.r_list.split(",") if tok.strip()]

    if not ipopt_available():
        raise RuntimeError("IPOPT is required but unavailable.")

    cfg = dict(SMOKE_INSTANCE if args.smoke else TRIAL_INSTANCE)
    out_dir = OUTPUT_ROOT / (SWEEP_DIRNAME + ("_SMOKE" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    msvrg = dict(SMOKE_MSVRG if args.smoke else TRIAL_MSVRG)

    if args.smoke:
        args.r_list = [2, 3]
        args.node_tol = 0.5
        args.max_wall_per_r = 120.0
        args.max_grads_per_r = 50_000.0
        args.ckpt_every_grads = 50.0

    fast, fast_path = (None, "")
    if not args.smoke:
        fast, fast_path = _load_fast_curves()

    results: dict = {}
    # Pick up previously completed r values so figures stay cumulative.
    for r in args.r_list:
        p = out_dir / f"r{r:02d}" / "summary.json"
        if p.exists() and not (args.force and not args.replot):
            with open(p, "r", encoding="utf-8") as fh:
                results[r] = json.load(fh)
            print(f"[r={r}] found existing summary, skipping (use --force "
                  f"to re-run)", flush=True)

    if args.replot:
        # Jul 25: figures are normally rewritten after each completed leg,
        # so an all-cached invocation would never redraw.  --replot draws
        # from the stored summaries and exits without running anything.
        if not results:
            raise SystemExit("--replot: no stored summaries for "
                             f"r in {args.r_list} under {out_dir}")
        _plot_sweep(results, fast, fast_path, args, out_dir, cfg)
        _write_readmes(results, fast_path, args, out_dir, cfg)
        print(f"REPLOTTED r={sorted(results)} -> {out_dir}", flush=True)
        return

    for r in args.r_list:
        if r in results:
            continue
        print(f"=== baseline-SVRG r={r} (node_tol={args.node_tol}, "
              f"share={args.share_mode}) ===", flush=True)
        t0 = time.time()
        res = _run_one_r(r, cfg, msvrg, args, out_dir)
        print(f"[r={r}] DONE in {time.time() - t0:.0f}s | "
              f"GN*={res['delivered_gn_strict']:.4e} | "
              f"grads={res['grad_equiv_total']:.0f} | "
              f"solved={res['solved_nodes']} | "
              f"shared={res['served_by_share']} | "
              f"censored={res['censored_nodes']} | "
              f"stop={res['stop_reason']}", flush=True)
        results[r] = res

        sweep_summary = {
            "metric": "self_reported family (no 256-start checkpoint solve); "
                      "scatter y = one strict 64-start delivery-time search",
            "args": {k: (v if not isinstance(v, list) else list(v))
                     for k, v in vars(args).items()},
            "instance": _json_ready(cfg),
            "joint_call_seconds_floor_basis": JOINT_CALL_SECONDS,
            "fast_reference": fast_path,
            "results": {str(rr): _json_ready(results[rr]) for rr in results},
        }
        (out_dir / "sweep_summary.json").write_text(
            json.dumps(sweep_summary, indent=2), encoding="utf-8")
        if not args.smoke:
            _plot_sweep(results, fast, fast_path, args, out_dir, cfg)
            _write_readmes(results, fast_path, args, out_dir, cfg)

    if args.smoke:
        # Machinery assertions: every node accounted for, certificates hold.
        for r, res in results.items():
            assert res["n_served"] + res["n_unserved"] == res["n_nodes"]
            assert res["verify_report"] is None or \
                res["verify_report"]["certificates_hold"]
            assert np.isfinite(res["delivered_gn_strict"])
        print("SMOKE OK:", {r: (res["n_served"], res["n_nodes"],
                                round(res["delivered_gn_strict"], 4))
                            for r, res in sorted(results.items())},
              flush=True)

    print(f"ALL DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
