"""Aggregate independently-run MLP crossover notebooks."""

from pathlib import Path
from typing import Dict, List, Optional
import json

import matplotlib.pyplot as plt
import numpy as np


PARALLEL_TAGS = ("h64x64", "h80x80", "h96x96")


def plot_parallel_crossover(output_base: Path) -> Optional[Path]:
    """Plot valid same-target comparisons found in parallel output folders."""
    parallel_root = Path(output_base) / "parallel"
    rows: List[Dict] = []
    missing = []
    for tag in PARALLEL_TAGS:
        summary_path = parallel_root / tag / "summary.json"
        if not summary_path.exists():
            missing.append(tag)
            continue
        with summary_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.extend(payload)

    if not rows:
        print("No parallel results are available yet.")
        return None

    rows.sort(key=lambda row: int(row["d"]))
    valid = [row for row in rows if row["adaptive_reached_target"]]
    invalid = [row for row in rows if not row["adaptive_reached_target"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    panels = (
        ("time_ratio_baseline_over_adaptive", "Baseline / Adaptive wall time",
         "CPU-time crossover (>1: Adaptive wins)", "#9467bd"),
        ("grad_ratio_baseline_over_adaptive", "Baseline / Adaptive gradient evaluations",
         "Gradient-efficiency ratio (>1: Adaptive wins)", "#1f77b4"),
    )

    for axis, (key, ylabel, title, color) in zip(axes, panels):
        if valid:
            x = np.asarray([row["d"] for row in valid], dtype=float)
            y = np.asarray([row[key] for row in valid], dtype=float)
            axis.plot(x, y, marker="o", color=color)
            for xi, yi, row in zip(x, y, valid):
                label = "x".join(map(str, row["hidden_sizes"]))
                axis.annotate(
                    label, (xi, yi), xytext=(4, 4),
                    textcoords="offset points", fontsize=8,
                )
            axis.set_xlim(x.min() * 0.8, x.max() * 1.25)
        else:
            axis.text(
                0.5, 0.5, "No architecture reached the common GN* target",
                transform=axis.transAxes, ha="center", va="center",
            )
        axis.axhline(1.0, color="black", ls="--", lw=1)
        axis.set_xscale("log")
        axis.set_xlabel("MLP parameter count d")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25, which="both")

    output = parallel_root / "complexity_crossover_parallel_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)

    if missing:
        print("Not finished yet:", ", ".join(missing))
    for row in invalid:
        architecture = "x".join(map(str, row["hidden_sizes"]))
        print(
            f"Excluded {architecture}: Adaptive did not reach "
            f"target GN*={row['target_gn']:.4e}."
        )
    print("Parallel summary plot:", output)
    return output
