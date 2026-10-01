"""Create a matched, paper-style reward-safety figure from saved evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required to plot reward-safety panels.") from exc


@dataclass(frozen=True)
class RewardPoint:
    method: str
    reward: float
    safety: float


METHOD_STYLES = {
    "Uniform DPO-LW": {"color": "#1F77B4", "marker": "s", "linestyle": "--"},
    "Adaptive bundle": {"color": "#FF7F0E", "marker": "o", "linestyle": "-"},
    "Adaptive exact_k2": {"color": "#FF7F0E", "marker": "o", "linestyle": "-"},
    "Adaptive IPOPT": {"color": "#E45756", "marker": "o", "linestyle": "-"},
    "SURF": {"color": "#D62728", "marker": "^", "linestyle": "--"},
    "SFT": {"color": "#8A8A8A", "marker": "D", "linestyle": "None"},
}
FALLBACK_COLORS = ["#9467BD", "#17BECF", "#8C564B"]
FALLBACK_MARKERS = ["P", "X", "v"]


def read_points(path: Path) -> List[RewardPoint]:
    points: List[RewardPoint] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            points.append(
                RewardPoint(
                    method=str(row["method"]),
                    reward=float(row["mean_reward"]),
                    safety=float(row["mean_safety"]),
                )
            )
    return points


def nondominated(points: Sequence[RewardPoint]) -> List[RewardPoint]:
    frontier: List[RewardPoint] = []
    for idx, point in enumerate(points):
        dominated = any(
            other.reward >= point.reward
            and other.safety >= point.safety
            and (other.reward > point.reward or other.safety > point.safety)
            for other_idx, other in enumerate(points)
            if other_idx != idx
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: point.safety)


def style_for(method: str, fallback_index: int) -> Dict[str, str]:
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    return {
        "color": FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)],
        "marker": FALLBACK_MARKERS[fallback_index % len(FALLBACK_MARKERS)],
        "linestyle": "-",
    }


def axis_limits(panels: Sequence[Sequence[RewardPoint]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    safeties = np.asarray([point.safety for panel in panels for point in panel], dtype=float)
    rewards = np.asarray([point.reward for panel in panels for point in panel], dtype=float)
    safety_span = max(float(safeties.max() - safeties.min()), 1e-6)
    reward_span = max(float(rewards.max() - rewards.min()), 1e-6)
    return (
        (float(safeties.min() - 0.06 * safety_span), float(safeties.max() + 0.06 * safety_span)),
        (float(rewards.min() - 0.06 * reward_span), float(rewards.max() + 0.06 * reward_span)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        nargs=2,
        action="append",
        required=True,
        metavar=("POINTS_CSV", "TITLE"),
        help="Path to reward_pareto_points.csv and the panel title. Repeat twice.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="External reward-model evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.panel) != 2:
        raise SystemExit("Exactly two --panel arguments are required for seen and held-out evaluation.")

    panels = [(Path(path), title, read_points(Path(path))) for path, title in args.panel]
    if any(not points for _, _, points in panels):
        raise SystemExit("Each reward CSV must contain at least one point.")
    x_limits, y_limits = axis_limits([points for _, _, points in panels])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=240, sharex=True, sharey=True)
    legend_handles = []
    legend_labels = []
    for panel_index, ((_, panel_title, points), ax) in enumerate(zip(panels, axes)):
        methods: List[str] = []
        for point in points:
            if point.method not in methods:
                methods.append(point.method)
        methods.sort(key=lambda method: (method == "SFT", method))

        for method_index, method in enumerate(methods):
            method_points = [point for point in points if point.method == method]
            style = style_for(method, method_index)
            ax.scatter(
                [point.safety for point in method_points],
                [point.reward for point in method_points],
                s=26 if method != "SFT" else 60,
                marker=style["marker"],
                color=style["color"],
                alpha=0.20 if method != "SFT" else 0.85,
                edgecolor="white",
                linewidth=0.6,
                label="_nolegend_",
                zorder=1,
            )
            if method == "SFT":
                if panel_index == 0:
                    handle = ax.scatter([], [], s=60, marker="D", color=style["color"], label="SFT")
                    legend_handles.append(handle)
                    legend_labels.append("SFT")
                continue
            frontier = nondominated(method_points)
            if not frontier:
                continue
            (line,) = ax.plot(
                [point.safety for point in frontier],
                [point.reward for point in frontier],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.35,
                marker=style["marker"],
                markersize=6.0,
                markeredgecolor="white",
                markeredgewidth=0.65,
                label=method,
                zorder=3,
            )
            if panel_index == 0:
                legend_handles.append(line)
                legend_labels.append(method)

        ax.set_title(panel_title, fontsize=11, pad=8)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.28, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Mean safety = -cost (higher is better)", fontsize=10)

    axes[0].set_ylabel("Mean reward (higher is better)", fontsize=10)
    fig.suptitle(args.title, fontsize=13, y=1.02)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.97),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
