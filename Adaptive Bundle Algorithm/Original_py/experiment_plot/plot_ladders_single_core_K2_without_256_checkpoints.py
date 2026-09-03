"""plot_ladders_single_core_K2_without_256_checkpoints.py — the K = 2 S3
ladder figure for ONE optimization core (user request Sep 4: the report
shows the adam 1e-3/0.9 core only).  Reads ladders_summary.json written
by run_surf_compare_K2_without_256_checkpoints.py --stage ladders and
writes ladders_<core>.png next to it, same size as ladders.png.

Usage:
    python plot_ladders_single_core_K2_without_256_checkpoints.py            # adam core
    python plot_ladders_single_core_K2_without_256_checkpoints.py --core adagrad_x10
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _layout  # noqa: F401
from run_surf_compare_K2_without_256_checkpoints import (  # noqa: E402
    LADDER_HOME,
    LADDER_RS,
    PAIR,
)

LABEL = {"adam_1e-3_b0.9": "adam α=10⁻³, β₂=0.9",
         "adagrad_x10": "adagrad ×10"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", default="adam_1e-3_b0.9")
    a = ap.parse_args()
    d = json.loads((LADDER_HOME / "ladders_summary.json").read_text())
    b = d["board"][a.core]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(LADDER_RS, [b["uniform"][str(r)]["mean"] for r in LADDER_RS],
                 marker="o", color="#ff7f0e", label=LABEL.get(a.core, a.core))
    axes[1].plot(LADDER_RS, [b["surf"][str(N)]["mean"] for N in LADDER_RS],
                 marker="o", color="#ff7f0e", label=LABEL.get(a.core, a.core))
    axes[0].set_xlabel("uniform grid r"); axes[1].set_xlabel("SURF N")
    for ax in axes:
        ax.set_ylabel("final worst GN (norm), mean of 3 seeds")
        ax.set_yscale("log"); ax.legend(fontsize=9); ax.set_xticks(LADDER_RS)
    axes[0].set_title(f"S3 ladders, pair {PAIR[0]}v{PAIR[1]}, B=2500, "
                      f"core {LABEL.get(a.core, a.core)}", fontsize=10)
    fig.tight_layout()
    out = LADDER_HOME / f"ladders_{a.core}.png"
    fig.savefig(out, dpi=150)
    print("[ladders-single] ->", out, flush=True)


if __name__ == "__main__":
    main()
