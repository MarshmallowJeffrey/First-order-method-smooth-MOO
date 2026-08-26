"""rebuild_cert_partial.py – salvage the user-stopped cert run from its log.

ONE-OFF (July 16, 2026).  The certification attempt (--variant-tag cert,
max_outer=3000) was stopped by the user at round ~1411 after the per-round
cost climbed to ~24 s (stacked-copy waste + segment-count growth; see
Note/Jul_16_note.md).  The process never wrote summary.json, but every
checkpoint line survives in run_log.txt:

    Fast outer R/3000 | t=Ts | bundle=M | grad_equiv=G | self-reported pc=P

This script rebuilds {cov_history, cpu_times, grad_evals_history} from
those lines (checkpoint-0 backfill per track convention), runs the
standard plateau detection, draws the two standard comparison figures
against the ORIGINAL adaptive curve (July 11 reuse, unchanged
disclosure), and writes summary_partial.json + a bilingual README into
the cert folder.  No algorithmic quantity is recomputed — everything
plotted is exactly what the run itself reported at its checkpoints.
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import _layout  # noqa: F401  (Aug-25 layout bootstrap; see _layout.py)
from experiments import detect_plateau, _plot_plateau_pair  # noqa: E402
from run_experiments import best_so_far, _json_ready  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "output"
# Post-reorganisation path (July 16): the cert folder now lives under
# fast_method_trials/ with a short name.  This one-off already ran against
# the old path; the constant is updated so a re-run targets the new home.
CERT_DIR = OUT / "fast_method_trials_v1v2v3_IPOPT" / "cert_attempt_partial"
LOG = OUT / "fast_trial_cert_run_log.txt"
REF = OUT / ("trial_K6_d11910_h96x96_tanh_n50000_B180180"
             "_without_256_checkpoints")

_ORIG_KW = dict(color="#1f77b4", marker="o", ms=4, lw=1.8)
_FAST_KW = dict(color="#2ca02c", marker="^", ms=4, lw=1.8)
DETECT = dict(window=4, relative_improvement_tol=0.05, consecutive_windows=2)

PAT = re.compile(
    r"Fast outer (\d+)/3000 \| t=([\d.]+)s \| bundle=(\d+) "
    r"\| grad_equiv=([\d.]+) \| self-reported pc=([\deE.+-]+|nan)")


def main() -> None:
    rounds, cpu, equiv, pc, bundle_m = [], [], [], [], []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        m = PAT.search(line)
        if not m:
            continue
        rounds.append(int(m.group(1)))
        cpu.append(float(m.group(2)))
        bundle_m.append(int(m.group(3)))
        equiv.append(float(m.group(4)))
        v = m.group(5)
        pc.append(float("nan") if v == "nan" else float(v))
    if not rounds:
        raise RuntimeError("no checkpoint lines parsed")

    # Checkpoint-0 backfill (track convention: same {x0} bundle as round 1).
    if pc and pc[0] != pc[0]:
        nxt = next(v for v in pc[1:] if v == v)
        pc[0] = nxt

    cert = {"cov_history": pc, "cpu_times": cpu,
            "grad_evals_history": equiv}
    cert_plateau = detect_plateau(pc, equiv, cpu, **DETECT)

    with open(REF / "summary.json", encoding="utf-8") as fh:
        ref = json.load(fh)
    orig = ref["adaptive"]
    orig_plateau = detect_plateau(
        orig["cov_history"], orig["grad_evals_history"], orig["cpu_times"],
        **DETECT)

    title = ("MLP K=6 cert attempt, PARTIAL (user-stopped at round "
             f"{rounds[-1]}/3000, eps=0.001, self-reported)")
    orig_kw = {**_ORIG_KW, "label": "adaptive bundle (original)"}
    fast_kw = {**_FAST_KW,
               "label": "fast adaptive (cert attempt, partial)"}
    _plot_plateau_pair(
        orig, orig_plateau, "Original adaptive", orig_kw,
        cert, cert_plateau, "Fast cert (partial)", fast_kw,
        x_history_key="grad_evals_history",
        x_label="total gradient evaluations (grad-equivalents)",
        title=title + ": Original vs Fast",
        out_path=str(CERT_DIR / "gn_vs_grad_evals_adaptive_orig_vs_cert_partial.png"),
    )
    _plot_plateau_pair(
        orig, orig_plateau, "Original adaptive", orig_kw,
        cert, cert_plateau, "Fast cert (partial)", fast_kw,
        x_history_key="cpu_times",
        x_label="CPU time (s, log scale)",
        title=title + ": Original vs Fast (CPU time)",
        out_path=str(CERT_DIR / "gn_vs_cpu_time_adaptive_orig_vs_cert_partial.png"),
        x_log=True, mark_equal_time=True,
    )

    bsf = best_so_far(pc)
    summary = {
        "metric": "self_reported (no 256-start checkpoint solve)",
        "status": "PARTIAL — user-stopped at round "
                  f"{rounds[-1]} of 3000 (July 16); curves rebuilt from "
                  "run_log.txt checkpoint lines, no quantity recomputed",
        "why_stopped": (
            "per-round cost had climbed 0.6s -> ~24s: (a) stacked-copy "
            "waste in the MSVRG inner loop (O(m*K*d) np.asarray per "
            "segment; fix planned), (b) segments/round rose 1 -> ~7.6 as "
            "the relative target hardened at low GN, bundle at 4088 "
            "points; and the best-so-far slope had decayed 1.2 -> 0.32 "
            "dex per 1000 rounds, so certification within the 3000-round "
            "fuse had become unlikely."),
        "config_note": ("same as v3 (two_tier, cheap plotted, "
                        "rel_target=0.25, v2 MSVRG params) with "
                        "max_outer=3000"),
        "rounds_recorded": rounds,
        "bundle_size_at_checkpoints": bundle_m,
        "fast_cert_partial": {
            "cov_history": _json_ready(pc),
            "cpu_times": _json_ready(cpu),
            "grad_evals_history": _json_ready(equiv),
            "best_so_far": _json_ready(bsf),
        },
        "final_best_so_far": float(bsf[-1]),
        "final_cpu_seconds": cpu[-1],
        "final_grad_equiv": equiv[-1],
        "plateaus": {"fast_cert_partial": _json_ready(cert_plateau),
                     "adaptive_original": _json_ready(orig_plateau)},
        "reused_reference": str(REF),
    }
    (CERT_DIR / "summary_partial.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""# Certification attempt — PARTIAL (user-stopped, round {rounds[-1]}/3000)

v3 configuration with max_outer=3000, stopped by the user on July 16 at
round {rounds[-1]} (~{cpu[-1]:.0f} s CPU, best-so-far GN
{bsf[-1]:.4e}, {equiv[-1]:.0f} grad-equivalents, bundle {bundle_m[-1]}
points).  Curves in `summary_partial.json` and the two figures are
rebuilt from the run's own checkpoint log lines — nothing recomputed.

Why stopped: per-round cost had grown 0.6 s -> ~24 s (stacked-copy waste
in the inner loop — fix planned — plus segments/round rising to ~7.6 as
the relative target hardens at low GN), and the best-so-far slope had
decayed from ~1.2 to ~0.32 dex per 1,000 rounds, making certification
within the 3,000-round fuse unlikely.  Full read-out:
`Note/Jul_16_note.md`.

The partial result still extends v3: best-so-far {bsf[-1]:.3e} vs v3's
5.81e-2 and the original method's 1.47e-1 (10,375 s).

---

# 认证尝试——部分结果（用户于第 {rounds[-1]}/3000 轮停止）

v3 配置 + max_outer=3000，7 月 16 日由用户在第 {rounds[-1]} 轮停止
（CPU 约 {cpu[-1]:.0f} s，best-so-far GN {bsf[-1]:.4e}，
{equiv[-1]:.0f} 梯度当量，bundle {bundle_m[-1]} 点）。曲线与两张图由
run 自身的检查点日志逐行重建，未重算任何量。

停止原因：每轮成本从 0.6 s 涨到约 24 s（内层堆叠拷贝浪费——已列入修复
——叠加低 GN 水位下相对目标变难、每轮段数回升到约 7.6），且 best-so-far
斜率从每千轮约 1.2 个数量级衰减到约 0.32，3000 轮内认证已不太可能。
完整记录见 `Note/Jul_16_note.md`。

部分结果依然推进了 v3：best-so-far {bsf[-1]:.3e}，对比 v3 的 5.81e-2
与原版方法的 1.47e-1（10,375 s）。
"""
    (CERT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"rebuilt {len(rounds)} checkpoints -> {CERT_DIR}")
    print(f"final best {bsf[-1]:.4e} at {cpu[-1]:.0f}s / {equiv[-1]:.0f} equiv")


if __name__ == "__main__":
    main()
