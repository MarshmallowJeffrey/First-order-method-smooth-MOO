from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required for plotting.") from exc


COLORS = {
    "DPO-LW uniform": "#4C78A8",
    "SURF": "#E45756",
    "Adaptive bundle": "#F58518",
}


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def norm_from_record(record: Dict) -> Optional[float]:
    value = first_present(record.get("best_gradient_norm"), record.get("gradient_norm"))
    if value is not None:
        return float(value)
    squared = first_present(record.get("best_gn_star"), record.get("gn_star"))
    return None if squared is None else float(np.sqrt(max(float(squared), 0.0)))


def x_from_record(record: Dict, x_axis: str) -> Optional[float]:
    if x_axis == "objective_gradient_evals":
        value = first_present(
            record.get("objective_gradient_evals"),
            record.get("objective_gradient_evals_after"),
        )
    else:
        value = first_present(
            record.get("elapsed_wall_seconds"),
            record.get("elapsed_wall_seconds_after"),
        )
    return None if value is None else float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot pre-specified GN-target stopping points."
    )
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "RUN_DIR"),
        required=True,
        help="One label and one run directory containing plateau_summary.json.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--x_axis",
        choices=("objective_gradient_evals", "elapsed_wall_seconds"),
        default="objective_gradient_evals",
    )
    parser.add_argument("--title", default="Plateau-level stopping comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for label, run_dir_text in args.run:
        run_dir = Path(run_dir_text)
        summary = read_json(run_dir / "plateau_summary.json")
        if not summary:
            summary = read_json(run_dir / "adaptive_final_state.json")
        if not summary:
            raise FileNotFoundError(f"No plateau summary found in {run_dir}")

        hit = summary.get("target_hit") or summary
        reached = bool(summary.get("target_reached", False))
        x_value = x_from_record(hit, args.x_axis)
        if x_value is None:
            x_value = x_from_record(summary, args.x_axis)
        y_value = norm_from_record(hit)
        if y_value is None:
            y_value = norm_from_record(summary)
        if x_value is None or y_value is None:
            raise ValueError(f"Missing plateau coordinates in {run_dir}")

        rows.append(
            {
                "label": label,
                "method": str(summary.get("method", "unknown")),
                "run_dir": str(run_dir),
                "target_reached": reached,
                "gn_target_norm": summary.get("gn_target_norm"),
                "plateau_gradient_norm": y_value,
                args.x_axis: x_value,
            }
        )

    csv_path = args.output_dir / "plateau_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    seen_methods = set()
    for row in rows:
        method = row["method"]
        color = COLORS.get(method, "#72B7B2")
        marker = "o" if row["target_reached"] else "x"
        label = method if method not in seen_methods else None
        seen_methods.add(method)
        ax.scatter(
            row[args.x_axis],
            row["plateau_gradient_norm"],
            s=64,
            marker=marker,
            color=color,
            label=label,
            zorder=3,
        )
        target = row["gn_target_norm"]
        suffix = "" if target is None else f", tau={float(target):.3g}"
        if not row["target_reached"]:
            suffix += ", not reached"
        ax.annotate(
            f"{row['label']}{suffix}",
            (row[args.x_axis], row["plateau_gradient_norm"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title(args.title)
    ax.set_ylabel("Best-so-far worst-case gradient norm")
    ax.set_xlabel(
        "Objective gradient evaluations"
        if args.x_axis == "objective_gradient_evals"
        else "Elapsed wall time (seconds)"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_path = args.output_dir / "plateau_hits.png"
    fig.savefig(figure_path)
    plt.close(fig)

    print(csv_path)
    print(figure_path)


if __name__ == "__main__":
    main()
