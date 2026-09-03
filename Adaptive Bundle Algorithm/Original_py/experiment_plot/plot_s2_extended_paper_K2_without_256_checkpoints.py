"""plot_s2_extended_paper_K2_without_256_checkpoints.py — paper version of
the K = 2 step-rule (optimizer-core) selection figure.

Same data as stage_s2_extend of run_stepper_pre_experiment_K2 (all 11
step-rule configurations, 3 sampling seeds, B = 10,000, ridge mu = 1e-3,
pair 4v9, adaptive-CCP trajectory), redrawn WITHOUT the B = 2,500
reference line and with readable legend labels (user request Sep 3: the
2,500 screen budget must not appear in the paper).  Reads the per-seed
summary.json files; nothing is rerun.

output: stepper_pre_experiment/extended_B10000_mu0.001/s2_extended_curves_paper.png
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _layout  # noqa: F401
from run_stepper_pre_experiment_K2_without_256_checkpoints import (  # noqa: E402
    PAIR,
    S2_EXT_BUDGET,
    S2_EXT_CONFIGS,
    S2_EXT_MU,
    S2_HOME,
    S2_SEEDS,
    _cfg_tag,
)

HOME = S2_HOME / f"extended_B{int(S2_EXT_BUDGET)}_mu{S2_EXT_MU:g}"
CHOSEN = "adam_alpha0.001_beta20.9"


def _label(tag: str) -> str:
    """Readable legend label for a config tag."""
    if tag == "const":
        return "const (incumbent)"
    if tag == "bb":
        return "BB"
    if tag.startswith("adagrad_mult"):
        return f"AdaGrad x{tag[len('adagrad_mult'):]}"
    if tag.startswith("adam_alpha"):
        a, b2 = tag[len("adam_alpha"):].split("_beta2")
        return f"Adam({float(a):g}, {float(b2):g})"
    return tag


def _style(tag: str):
    if tag == CHOSEN:
        return dict(lw=2.4, color="#d62728", ls="-", zorder=5)
    if tag == "const":
        return dict(lw=1.8, color="black", ls="--", zorder=4)
    if tag == "adagrad_mult10":
        return dict(lw=1.8, color="#1f77b4", ls="-", zorder=4)
    return dict(lw=1.0, alpha=0.75)


def main():
    curves = {}
    for name, scfg in S2_EXT_CONFIGS:
        tag = _cfg_tag(name, scfg)
        sms = [json.loads((HOME / f"{tag}_seed{s}" / "summary.json").read_text())
               for s in S2_SEEDS]
        cs = [np.asarray(sm["audited_gn_norm_history"], dtype=float) for sm in sms]
        L = min(len(c) for c in cs)
        g = np.asarray(sms[0]["ck_grads"], dtype=float)[:L]
        cpu = np.mean([np.asarray(sm["ck_cpu"], dtype=float)[:L] for sm in sms], axis=0)
        curves[tag] = (g, cpu, np.mean([c[:L] for c in cs], axis=0))

    # draw the highlighted lines last so they sit on top
    order = sorted(curves, key=lambda t: (t in (CHOSEN, "const", "adagrad_mult10"), t))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for tag in order:
        g, cpu, m = curves[tag]
        st = _style(tag)
        axes[0].plot(g, m, label=_label(tag), **st)
        axes[1].plot(cpu, m, label=_label(tag), **st)
    axes[0].set_xlabel("total gradient evaluations (grad_equiv)")
    axes[1].set_xlabel("CPU seconds")
    for ax in axes:
        ax.set_ylabel("best-so-far worst GN (norm)")
        ax.set_yscale("log")
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title(f"MNIST {PAIR[0]} vs {PAIR[1]}, ridge mu={S2_EXT_MU:g}, "
                      f"B={int(S2_EXT_BUDGET):,}, mean of {len(S2_SEEDS)} seeds",
                      fontsize=10)
    fig.tight_layout()
    out = HOME / "s2_extended_curves_paper.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[plot] ->", out, flush=True)


if __name__ == "__main__":
    main()
