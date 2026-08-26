"""Fast-method TRIAL: K=6 at 96x96, epsilon mode, self-reported checkpoints.

NEW FILE (July 15, 2026).  Runs the ACCELERATED adaptive method
(``algorithm_fast_without_256_checkpoints.algorithm_adaptive_fast``:
Gram-path λ-search + Momentum-SVRG inner loop + gradient-equivalent
accounting + delivery-time pruning) on the SAME problem instance as the
July 11 trial ``run_trial_K6_without_256_checkpoints.py`` (K=6, p=20,
n=50000, hidden 96x96, tanh, seeds 7/8), with the user-fixed
``epsilon=0.001``, and produces four comparison figures:

    1.1  baseline (r=10)          vs  fast adaptive   (grad axis, CPU axis)
    1.2  original adaptive method vs  fast adaptive   (grad axis, CPU axis)

REUSED CURVES, disclosed: the baseline and original-adaptive curves are
read from ``output/trial_K6_d11910_h96x96_tanh_n50000_B180180_without_256_
checkpoints/summary.json`` (July 11/12 run) rather than re-run:

* Same problem instance and x0 (seeds 7/8), same r=10, same fuse
  max_outer=150, same lambda_max_starts=64, same self-reported metric.
* The old adaptive run used budget mode (epsilon=None).  With
  epsilon=0.001 the trajectory over the entire recorded range is
  IDENTICAL: the only epsilon-mode additions are the outer stop test
  (pc <= 2eps/3 ~ 6.7e-4) and the inner eps/3 early-stop (3.3e-4), and
  the old run's GN never went below 0.147 — two orders of magnitude
  above either threshold, so neither test could have fired.  (The old
  budget-mode curve, if anything, slightly flatters the original method
  on the CPU axis: epsilon mode would have added a per-inner-step GN
  check it did not pay.)
* CPU-axis caveat: cross-run comparison on the same machine; the July 11
  run logged its load and its oracle pace agreed with calibration within
  6.5% (see the old folder's README / machine_load_log.txt).

Gradient axis: the fast method is charged in GRAD-EQUIVALENTS
(1 joint-oracle call = K; one minibatch Momentum-SVRG step = 2b·K/n,
because the K per-class losses partition the n samples).  The legacy
curves' axis is exactly the same unit (their steps are all full joint
calls), so the axes are directly comparable.

λ-search tiers (v3 default, user-approved July 16): lambda_tier_mode=
"two_tier" — ordinary rounds search on the CHEAP tier (centroid + K
vertices + prev_lam ≈ 8 starts, tol 1e-4) and their values are PLOTTED
DIRECTLY (user decision; the 8-start meter under-searches a maximiser so
it can only under-report — disclosed in the README; the
"strict-on-recorded-rounds" guard is held as a backup, see
Note/Jul_16_note.md).  Only the STRICT tier (64 starts, tol 1e-8) may
sign a stopping certificate: a cheap value at or below 2eps/3 triggers a
strict re-solve, and the run stops only if the strict value confirms.

Relative inner target (v3, user-approved July 16; Algorithm-2 VARIANT):
each round's inner-loop target is max(eps/3, rel_target * pc_val) — "cut
the selected worst direction to a quarter", floored by the paper's
absolute eps/3 so the endgame matches the original algorithm exactly.
Responds to v2's cap_hits=150 (the absolute eps/3 is unreachable from a
cold start).  The stopping certificate is unaffected.

Usage:
    python run_trial_K6_fast_without_256_checkpoints.py [--smoke]
        [--batch B] [--beta B] [--step-const C] [--tier-mode M]
        [--max-outer N] [--rel-target G] [--variant-tag TAG]
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

# experiments first: it loads objectives_torch (and torch's OpenMP runtime)
# before any cyipopt import — same convention as the July 11 runner.
import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from experiments import (  # noqa: E402
    make_mlp_initial_point,
    detect_plateau,
    _plot_plateau_pair,
    _BL_KW,
    _A2_KW,
)
from bundle import prefer_fused_joint_oracle  # noqa: E402
from objectives_torch_fast import make_mlp_nonconvex_fast  # noqa: E402
from algorithm_fast_without_256_checkpoints import (  # noqa: E402
    algorithm_adaptive_fast,
    ipopt_available,
)
from run_experiments import (  # noqa: E402
    DATA_SEED,
    INIT_SEED,
    best_so_far,
    symmetric_time_to_target,
    _json_ready,
)

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent.parent / "output"
REF_TAG = "trial_K6_d11910_h96x96_tanh_n50000_B180180_without_256_checkpoints"
REF_DIR = OUTPUT_ROOT / REF_TAG

_FAST_KW = dict(color="#2ca02c", marker="^", ms=4, lw=1.8,
                label="adaptive bundle (fast: Gram + Momentum-SVRG)")
_ORIG_KW = dict(**{**_A2_KW, "label": "adaptive bundle (original)"})

TRIAL_CONFIG = dict(
    # ---- user-fixed base parameters ----
    K=6, p=20, n=50_000, hidden_sizes=[96, 96], activation="tanh",
    coarse_resolution=10,          # baseline grid r (reused curves) + prune grid
    epsilon=1e-3,
    # ---- kept aligned with the July 11 trial ----
    max_grad_evals=180_180.0,      # nominal cap (grad-equivalents), same B
    adaptive_eval_every_n_grads=600.0,
    lambda_max_starts=64,          # strict tier, same as the old run's search
    # ---- v3 settings (user-approved July 16; see module docstring) ----
    max_outer=500,                 # raised fuse: chase eps for real
    lambda_tier_mode="two_tier",   # cheap rounds + strict-only certificates
    msvrg_rel_target=0.25,         # inner target = max(eps/3, 0.25*pc_val)
    # ---- Momentum-SVRG parameters (v2-validated combination) ----
    msvrg_batch=4096,
    msvrg_epoch_len=None,          # None -> ceil(n/b) = 13
    msvrg_step_const=0.1,
    msvrg_momentum=0.5,
    msvrg_trigger_rho=0.7,
    msvrg_trigger_patience=2,
    msvrg_max_segments=10,
    prune_grid_r=10,
    sampler_seed=41,
)

SMOKE_CONFIG = dict(
    K=6, p=6, n=300, hidden_sizes=[8], activation="tanh",
    coarse_resolution=2, epsilon=1e-2,
    max_outer=8, max_grad_evals=None, adaptive_eval_every_n_grads=None,
    lambda_max_starts=8, lambda_tier_mode="two_tier",
    msvrg_rel_target=0.25,
    msvrg_batch=60, msvrg_epoch_len=5, msvrg_step_const=0.1,
    msvrg_momentum=0.9, msvrg_trigger_rho=0.7, msvrg_trigger_patience=2,
    msvrg_max_segments=4, prune_grid_r=3, sampler_seed=41,
)

DETECT = dict(window=4, relative_improvement_tol=0.05, consecutive_windows=2)


def _load_reference_curves():
    """Baseline + original-adaptive curves from the July 11 summary."""
    with open(REF_DIR / "summary.json", "r", encoding="utf-8") as fh:
        ref = json.load(fh)
    return ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny problem, separate output folder, no reuse")
    parser.add_argument("--batch", type=int, default=None,
                        help="override msvrg_batch")
    parser.add_argument("--beta", type=float, default=None,
                        help="override msvrg_momentum")
    parser.add_argument("--step-const", type=float, default=None,
                        help="override msvrg_step_const")
    parser.add_argument("--tier-mode", type=str, default=None,
                        choices=["strict", "two_tier"],
                        help="override lambda_tier_mode")
    parser.add_argument("--max-outer", type=int, default=None,
                        help="override max_outer (round fuse)")
    parser.add_argument("--rel-target", type=float, default=None,
                        help="override msvrg_rel_target (γ); pass 0 to "
                             "disable the relative target (pure eps/3)")
    parser.add_argument("--variant-tag", type=str, default="",
                        help="suffix appended to the output folder name")
    args = parser.parse_args()
    cfg = dict(SMOKE_CONFIG if args.smoke else TRIAL_CONFIG)
    if args.batch is not None:
        cfg["msvrg_batch"] = args.batch
    if args.beta is not None:
        cfg["msvrg_momentum"] = args.beta
    if args.step_const is not None:
        cfg["msvrg_step_const"] = args.step_const
    if args.tier_mode is not None:
        cfg["lambda_tier_mode"] = args.tier_mode
    if args.max_outer is not None:
        cfg["max_outer"] = args.max_outer
    if args.rel_target is not None:
        cfg["msvrg_rel_target"] = (None if args.rel_target == 0.0
                                   else args.rel_target)

    if not ipopt_available():
        raise RuntimeError("IPOPT is required but unavailable.")

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

    hid = "x".join(str(h) for h in cfg["hidden_sizes"])
    tag = (f"trial_K{cfg['K']}_d{d}_h{hid}_{cfg['activation']}_n{cfg['n']}"
           f"_eps{cfg['epsilon']}_fast_msvrg_without_256_checkpoints")
    if args.variant_tag:
        tag += f"_{args.variant_tag}"
    if args.smoke:
        tag += "_SMOKE"
    # Since the July 16 reorganisation all fast-series runs live under
    # output/fast_method_trials/ (see its README for the naming map).
    out_dir = OUTPUT_ROOT / "fast_method_trials_v1v2v3_IPOPT" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== K={cfg['K']} FAST trial: {tag} ===", flush=True)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fast = algorithm_adaptive_fast(
            K=cfg["K"], d=d, objectives=objectives,
            grad_objectives=gradients, L=L, x0=x0,
            stoch_oracle=stoch,
            epsilon=cfg["epsilon"],
            max_outer=cfg["max_outer"],
            eval_every_n_grads=cfg["adaptive_eval_every_n_grads"],
            max_grad_evals=cfg["max_grad_evals"],
            lambda_max_starts=cfg["lambda_max_starts"],
            lambda_tier_mode=cfg["lambda_tier_mode"],
            msvrg_step_const=cfg["msvrg_step_const"],
            msvrg_momentum=cfg["msvrg_momentum"],
            msvrg_epoch_len=cfg["msvrg_epoch_len"],
            msvrg_max_segments=cfg["msvrg_max_segments"],
            msvrg_trigger_rho=cfg["msvrg_trigger_rho"],
            msvrg_trigger_patience=cfg["msvrg_trigger_patience"],
            msvrg_rel_target=cfg["msvrg_rel_target"],
            prune_grid_r=cfg["prune_grid_r"],
            joint_oracle=joint_oracle, verbose=True,
        )
    wall = time.time() - t0
    warning_messages = sorted({str(w.message) for w in caught})
    print(f"fast run finished in {wall:.1f}s "
          f"(stop={fast['stop_reason']}, "
          f"lambda-search {fast['lambda_search_seconds']:.1f}s)", flush=True)

    fast_curves = {
        "cov_history": _json_ready(fast["cov_history"]),
        "cpu_times": _json_ready(fast["cpu_times"]),
        "grad_evals_history": _json_ready(fast["grad_evals_history"]),
        "best_so_far": _json_ready(best_so_far(fast["cov_history"])),
        "L_scale_final": _json_ready(fast["L_scale_final"]),
        "inner_cap_hits": _json_ready(fast["inner_cap_hits"]),
    }
    fast_plateau = detect_plateau(
        fast["cov_history"], fast["grad_evals_history"], fast["cpu_times"],
        **DETECT)

    summary = {
        "metric": "self_reported (no 256-start checkpoint solve)",
        "trial_note": (
            "K=6 FAST trial (July 15): accelerated adaptive method = "
            "Gram-path lambda-search + Momentum-SVRG inner loop + "
            "grad-equivalent accounting + delivery-time pruning; "
            "epsilon=0.001; baseline and original-adaptive curves reused "
            f"from {REF_TAG} (same instance, seeds "
            f"{DATA_SEED}/{INIT_SEED}; see README)."),
        "config": _json_ready(cfg),
        "data_seed": DATA_SEED,
        "init_seed": INIT_SEED,
        "sampler_seed": cfg["sampler_seed"],
        "d": d,
        "fast_adaptive": fast_curves,
        "fast_extras": {
            "stop_reason": fast["stop_reason"],
            "joint_calls": fast["joint_calls"],
            "ifo_minibatch_total": fast["ifo_minibatch_total"],
            "grad_equiv_total": fast["grad_equiv_total"],
            "lambda_search_seconds": fast["lambda_search_seconds"],
            "wall_seconds": wall,
            "segments_history": _json_ready(fast["segments_history"]),
            "tier_history_counts": {
                t: fast["tier_history"].count(t)
                for t in set(fast["tier_history"])},
            "inner_target_history": _json_ready(fast["inner_target_history"]),
            "msvrg_rel_target": _json_ready(fast["msvrg_rel_target"]),
            "pc_history": _json_ready(fast["pc_history"]),
            "prune_report": _json_ready(fast["prune_report"]),
            "final_lambda": _json_ready(fast["lambda_history"][-1]
                                        if fast["lambda_history"] else None),
        },
        "plateaus": {"fast_adaptive": _json_ready(fast_plateau)},
        "runtime_warnings": warning_messages,
    }

    if not args.smoke:
        ref = _load_reference_curves()
        bl, a2 = ref["baseline"], ref["adaptive"]
        bl_plateau = detect_plateau(
            bl["cov_history"], bl["grad_evals_history"], bl["cpu_times"],
            **DETECT)
        a2_plateau = detect_plateau(
            a2["cov_history"], a2["grad_evals_history"], a2["cpu_times"],
            **DETECT)
        summary["reused_reference"] = {
            "path": str(REF_DIR),
            "baseline": bl, "adaptive_original": a2,
            "note": ("curves recorded July 11/12 on the same machine/instance;"
                     " budget-mode trajectory == epsilon-mode trajectory over"
                     " the recorded range (GN >= 0.147 >> 2eps/3=6.7e-4)"),
        }
        summary["plateaus"]["baseline"] = _json_ready(bl_plateau)
        summary["plateaus"]["adaptive_original"] = _json_ready(a2_plateau)
        summary["time_to_target_vs_baseline"] = _json_ready(
            symmetric_time_to_target(bl, {
                "cov_history": fast["cov_history"],
                "cpu_times": fast["cpu_times"],
                "grad_evals_history": fast["grad_evals_history"]}))
        summary["time_to_target_vs_original_adaptive"] = _json_ready(
            symmetric_time_to_target(a2, {
                "cov_history": fast["cov_history"],
                "cpu_times": fast["cpu_times"],
                "grad_evals_history": fast["grad_evals_history"]}))

        title_base = (f"MLP K={cfg['K']} fast trial (p={cfg['p']}, "
                      f"n={cfg['n']}, h={cfg['hidden_sizes']}, "
                      f"{cfg['activation']}, eps={cfg['epsilon']}, "
                      f"self-reported)")
        fast_result_for_plot = {
            "cov_history": fast["cov_history"],
            "cpu_times": fast["cpu_times"],
            "grad_evals_history": fast["grad_evals_history"],
        }
        bl_kw = {**_BL_KW,
                 "label": f"uniform discretisation (r={cfg['coarse_resolution']})"}
        _plot_plateau_pair(
            bl, bl_plateau, "Baseline", bl_kw,
            fast_result_for_plot, fast_plateau, "Fast adaptive", _FAST_KW,
            x_history_key="grad_evals_history",
            x_label="total gradient evaluations (grad-equivalents)",
            title=title_base + ": Baseline vs Fast adaptive",
            out_path=str(out_dir / "gn_vs_grad_evals_baseline_vs_fast.png"),
        )
        _plot_plateau_pair(
            bl, bl_plateau, "Baseline", bl_kw,
            fast_result_for_plot, fast_plateau, "Fast adaptive", _FAST_KW,
            x_history_key="cpu_times",
            x_label="CPU time (s, log scale)",
            title=title_base + ": Baseline vs Fast adaptive (CPU time)",
            out_path=str(out_dir / "gn_vs_cpu_time_baseline_vs_fast.png"),
            x_log=True, mark_equal_time=True,
        )
        _plot_plateau_pair(
            a2, a2_plateau, "Original adaptive", _ORIG_KW,
            fast_result_for_plot, fast_plateau, "Fast adaptive", _FAST_KW,
            x_history_key="grad_evals_history",
            x_label="total gradient evaluations (grad-equivalents)",
            title=title_base + ": Original vs Fast adaptive",
            out_path=str(out_dir / "gn_vs_grad_evals_adaptive_orig_vs_fast.png"),
        )
        _plot_plateau_pair(
            a2, a2_plateau, "Original adaptive", _ORIG_KW,
            fast_result_for_plot, fast_plateau, "Fast adaptive", _FAST_KW,
            x_history_key="cpu_times",
            x_label="CPU time (s, log scale)",
            title=title_base + ": Original vs Fast adaptive (CPU time)",
            out_path=str(out_dir / "gn_vs_cpu_time_adaptive_orig_vs_fast.png"),
            x_log=True, mark_equal_time=True,
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    _write_readmes(out_dir, cfg, d, summary, args.smoke)
    print(f"DONE in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


def _write_readmes(out_dir: Path, cfg: dict, d: int, summary: dict,
                   smoke: bool) -> None:
    fx = summary["fast_extras"]
    fc = summary["fast_adaptive"]
    final_best = fc["best_so_far"][-1] if fc["best_so_far"] else float("nan")
    lam_share = (100.0 * fx["lambda_search_seconds"] / fx["wall_seconds"]
                 if fx["wall_seconds"] > 0 else float("nan"))
    ttb = summary.get("time_to_target_vs_baseline", {})
    tta = summary.get("time_to_target_vs_original_adaptive", {})
    prune = fx["prune_report"]

    if cfg["lambda_tier_mode"] == "two_tier":
        tier_note_en = (
            "Ordinary rounds search on the CHEAP tier (~K+2 starts, tol "
            "1e-4) and their values are plotted directly (user decision): "
            "an under-search of a maximiser can only UNDER-report, and the "
            "legacy curves carry the 64-start meter — keep this in mind on "
            "cross-curve reads (backup guard recorded in "
            "Note/Jul_16_note.md).  Certificates are strict-tier only: a "
            "cheap value at or below 2eps/3 triggers a strict re-solve and "
            "the run stops only on strict confirmation.")
        tier_note_zh = (
            "平时用粗档（约 K+2 起点、tol 1e-4）搜索，其值直接进图"
            "（用户决定）：求最大化时搜索不足只会低报，而旧曲线是 64 起点"
            "口径——跨曲线解读时注意这一点（backup 方案记录于 "
            "Note/Jul_16_note.md）。证书只由严格档签发：粗档值 ≤ 2eps/3 "
            "时触发严格档复核，复核确认才停机。")
    else:
        tier_note_en = (
            "All rounds used the STRICT tier (64 starts): the self-reported "
            "metric stays exactly on the legacy 64-start yardstick.")
        tier_note_zh = (
            "全程严格档（64 起点）：自报告口径与旧曲线完全一致。")

    readme = f"""# K={cfg['K']} FAST trial (Gram + Momentum-SVRG, eps={cfg['epsilon']}, without-256 track)

Produced by `Original_py/run_trial_K6_fast_without_256_checkpoints.py`
(July 15, 2026).  The module docstring is the full specification; the
July 15 plan document (Desktop, ZH/EN) is the design reference.  All
`_fast` code files are NEW; the original files are untouched.

## What ran

The ACCELERATED adaptive method (`algorithm_adaptive_fast`), all four
plan items live:

1. **Gram-path λ-search** — exact rewrite GN(λ)=min_i λ^T M_i λ;
   measured λ-search share this run: **{fx['lambda_search_seconds']:.1f} s
   of {fx['wall_seconds']:.1f} s wall ({lam_share:.1f}%)** vs ~95% in the
   July 11 original run.
2. **Two-tier λ-search + stop-verify** —
   `lambda_tier_mode="{cfg['lambda_tier_mode']}"`.  {tier_note_en}
   Tier counts: {fx['tier_history_counts']}.
3. **Momentum-SVRG inner loop** — segments of stratified minibatch
   variance-reduced heavy-ball steps (b={cfg['msvrg_batch']},
   p_seg={cfg['msvrg_epoch_len'] or 'ceil(n/b)'},
   step_const={cfg['msvrg_step_const']}, beta={cfg['msvrg_momentum']},
   rho={cfg['msvrg_trigger_rho']}, patience={cfg['msvrg_trigger_patience']},
   max_segments={cfg['msvrg_max_segments']},
   rel_target={cfg['msvrg_rel_target']}).  Per-round inner target =
   max(eps/3, rel_target*pc_val) when rel_target is set — an Algorithm-2
   VARIANT responding to v2's cap_hits (the certificate is unaffected;
   the stop line never moves).  Every bundle admission and every
   acceptance test runs on FULL gradients (randomness cannot fake a
   certificate).
4. **Delivery-time pruning** — bundle {prune['m_before']} -> {prune['m_after']}
   points by lambda-activation on the r={cfg['prune_grid_r']} simplex grid
   (+ final search winners); probe GN values bitwise unchanged.

Outcome: stop_reason=`{fx['stop_reason']}`, final self-reported best-so-far
GN = **{final_best:.4e}**, grad-equivalents used = {fx['grad_equiv_total']:.0f}
(joint calls {fx['joint_calls']}, minibatch IFO {fx['ifo_minibatch_total']}),
wall {fx['wall_seconds']:.1f} s.  L_scale_final={fc['L_scale_final']},
inner_cap_hits={fc['inner_cap_hits']}.

## Axes and accounting

Gradient axis = GRAD-EQUIVALENTS: one joint-oracle call = K (its n
per-sample gradients cover the K disjoint per-class losses once); one
minibatch Momentum-SVRG step = 2b·K/n.  The reused curves' axis is the
same unit (all their steps are full joint calls), so the axes compare
directly.  CPU axis: checkpoint bookkeeping excluded, as everywhere on
this track.

## Reused reference curves (disclosed)

Baseline (r=10) and ORIGINAL adaptive curves come from
`output/{REF_TAG}/summary.json` (July 11/12), NOT re-run: same problem
instance and x0 (seeds {summary['data_seed']}/{summary['init_seed']}),
same fuse (max_outer=150), same 64-start self-reported metric.  The old
adaptive run was budget mode (epsilon=None); with epsilon=0.001 the
trajectory over the recorded range is identical (its GN never went below
0.147 >> 2eps/3 = 6.7e-4, so neither epsilon test could fire) — and the
budget-mode CPU axis, if anything, flatters the ORIGINAL method (epsilon
mode would have charged it an extra per-step GN check).  CPU comparison
is cross-run on the same machine; the July 11 folder logs machine load
and an oracle-pace calibration within 6.5%.

## Headline comparisons (self-reported meters)

vs baseline: target GN {ttb.get('target_gn', float('nan')):.4f} reached by
baseline at {ttb.get('baseline_cpu_to_target')} s /
{ttb.get('baseline_grads_to_target')} grads, by fast adaptive at
{ttb.get('adaptive_cpu_to_target')} s / {ttb.get('adaptive_grads_to_target')}
grad-equivalents (ratios: CPU {ttb.get('cpu_ratio_baseline_over_adaptive')},
grads {ttb.get('grad_ratio_baseline_over_adaptive')}).

vs original adaptive: target GN {tta.get('target_gn', float('nan')):.4f}
reached by original at {tta.get('baseline_cpu_to_target')} s /
{tta.get('baseline_grads_to_target')} grads, by fast at
{tta.get('adaptive_cpu_to_target')} s / {tta.get('adaptive_grads_to_target')}
grad-equivalents.  (In `time_to_target_vs_original_adaptive` the
"baseline_*" fields refer to the ORIGINAL adaptive method — the helper's
first-slot naming.)

## Figures

- `gn_vs_grad_evals_baseline_vs_fast.png`, `gn_vs_cpu_time_baseline_vs_fast.png`
- `gn_vs_grad_evals_adaptive_orig_vs_fast.png`, `gn_vs_cpu_time_adaptive_orig_vs_fast.png`

CPU figures: log time axis; vertical line = equal-budget point.

## Caveats, stated once

Self-reported meters (baseline never looks between its grid nodes; the
adaptive values are heuristic lower bounds of an NP-hard max — the
lambda-search is a maximiser, so an under-searched value UNDERSTATES the
criterion).  Single instance, seeds {summary['data_seed']}/{summary['init_seed']};
single machine; cross-run CPU comparison for the reused curves.  The
Momentum-SVRG inner guarantee is expectation-type; all acceptance tests
are exact (full gradients).  Inner rounds that hit the segment cap void
the Algorithm-2 termination argument for those rounds
(inner_cap_hits={fc['inner_cap_hits']}; warning recorded).
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    readme_zh = f"""# K={cfg['K']} FAST 试验（Gram + Momentum-SVRG，eps={cfg['epsilon']}，without-256 轨道）

由 `Original_py/run_trial_K6_fast_without_256_checkpoints.py` 生成
（2026-07-15）。模块 docstring 是完整规格；设计参考为 7 月 15 日的双语
方案文档（桌面 ZH/EN 两版）。所有 `_fast` 代码均为新文件，原文件未动。

## 跑了什么

加速版 adaptive 方法（`algorithm_adaptive_fast`），方案四项全部生效：

1. **Gram 化 λ-search**——精确恒等改写 GN(λ)=min_i λ^T M_i λ；本次实测
   λ-search 耗时 **{fx['lambda_search_seconds']:.1f} s / 总 {fx['wall_seconds']:.1f} s
   （{lam_share:.1f}%）**，对比 7 月 11 日原版 run 的 ~95%。
2. **两档 λ-search + 停机复核**——
   `lambda_tier_mode="{cfg['lambda_tier_mode']}"`。{tier_note_zh}
   档位统计：{fx['tier_history_counts']}。
3. **Momentum-SVRG 内层**——分层 minibatch 方差缩减 + heavy-ball 动量的
   分段结构（b={cfg['msvrg_batch']}，p_seg={cfg['msvrg_epoch_len'] or 'ceil(n/b)'}，
   step_const={cfg['msvrg_step_const']}，beta={cfg['msvrg_momentum']}，
   rho={cfg['msvrg_trigger_rho']}，patience={cfg['msvrg_trigger_patience']}，
   max_segments={cfg['msvrg_max_segments']}，
   rel_target={cfg['msvrg_rel_target']}）。启用 rel_target 时每轮内层目标 =
   max(eps/3, rel_target×pc_val)——针对 v2 cap_hits 的 Algorithm-2 变体
   （证书不受影响，停机线不动）。入 bundle 与验收全部基于 FULL 梯度
   （随机性无法伪造证书）。
4. **交付时修剪**——bundle {prune['m_before']} -> {prune['m_after']} 点
   （r={cfg['prune_grid_r']} simplex 网格 + 末轮搜索赢家的 λ-激活检测）；
   探针 λ 上的 GN 值逐位不变。

结果：stop_reason=`{fx['stop_reason']}`，最终自报告 best-so-far GN =
**{final_best:.4e}**，消耗 grad-equivalents = {fx['grad_equiv_total']:.0f}
（joint 调用 {fx['joint_calls']} 次，minibatch IFO {fx['ifo_minibatch_total']}），
墙钟 {fx['wall_seconds']:.1f} s。L_scale_final={fc['L_scale_final']}，
inner_cap_hits={fc['inner_cap_hits']}。

## 轴与记账口径

梯度轴 = GRAD-EQUIVALENTS：一次 joint oracle = K（其 n 个 per-sample
梯度恰好把 K 个互不相交的 per-class loss 各覆盖一遍）；一步 minibatch
Momentum-SVRG = 2b·K/n。复用曲线的轴是同一单位（其所有步都是 full joint
调用），两轴直接可比。CPU 轴照本轨道惯例剔除 checkpoint 开销。

## 复用的参照曲线（披露）

Baseline（r=10）与原版 adaptive 曲线取自
`output/{REF_TAG}/summary.json`（7 月 11/12），未重跑：同一问题实例与
x0（种子 {summary['data_seed']}/{summary['init_seed']}）、同一 fuse
（max_outer=150）、同一 64 起点自报告口径。旧 adaptive run 为 budget 模式
（epsilon=None）；在 epsilon=0.001 下其已记录区间的轨迹逐位相同（其 GN
最低 0.147，远高于 2eps/3=6.7e-4，两处 epsilon 检查均不可能触发）——且
budget 模式的 CPU 轴对原版方法只会更有利（epsilon 模式还要多付每步一次
GN 检查）。CPU 对比为同机跨 run；7 月 11 日文件夹记录了机器负载与
oracle 速率校准（偏差 6.5% 以内）。

## 关键对比（自报告口径）

对 baseline：共同目标 GN {ttb.get('target_gn', float('nan')):.4f}，baseline 于
{ttb.get('baseline_cpu_to_target')} s / {ttb.get('baseline_grads_to_target')}
grads 达到，fast 于 {ttb.get('adaptive_cpu_to_target')} s /
{ttb.get('adaptive_grads_to_target')} grad-equivalents 达到
（比值：CPU {ttb.get('cpu_ratio_baseline_over_adaptive')}，
梯度 {ttb.get('grad_ratio_baseline_over_adaptive')}）。

对原版 adaptive：共同目标 GN {tta.get('target_gn', float('nan')):.4f}，原版于
{tta.get('baseline_cpu_to_target')} s / {tta.get('baseline_grads_to_target')}
grads 达到，fast 于 {tta.get('adaptive_cpu_to_target')} s /
{tta.get('adaptive_grads_to_target')} grad-equivalents 达到。
（`time_to_target_vs_original_adaptive` 中的 "baseline_*" 字段指原版
adaptive——沿用工具函数的第一槽位命名。）

## 图

- `gn_vs_grad_evals_baseline_vs_fast.png`、`gn_vs_cpu_time_baseline_vs_fast.png`
- `gn_vs_grad_evals_adaptive_orig_vs_fast.png`、`gn_vs_cpu_time_adaptive_orig_vs_fast.png`

CPU 图为对数时间轴；竖线 = 等预算点。

## 注意事项（一次说清）

自报告口径（baseline 从不看网格节点之间；adaptive 值是 NP-hard max 的
启发式下界——λ-search 是求最大化器，搜索不足只会让报告值偏低、即偏
保守）。单实例（种子 {summary['data_seed']}/{summary['init_seed']}）、单机器；
复用曲线为跨 run CPU 对比。Momentum-SVRG 的内层保证是期望型；所有验收
检查均为精确（full 梯度）。触到段数帽的轮次不适用 Algorithm 2 的终止
论证（inner_cap_hits={fc['inner_cap_hits']}；warning 已记录）。
"""
    (out_dir / "README_ZH.md").write_text(readme_zh, encoding="utf-8")


if __name__ == "__main__":
    main()
