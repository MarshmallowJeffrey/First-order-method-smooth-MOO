from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required for timing plots.") from exc


PHASES = [
    ("lambda_solver", "Lambda solver"),
    ("lambda_diagnostics", "Lambda diagnostics"),
    ("inner_update", "Inner update"),
    ("inner_oracle", "Oracle eval"),
    ("bundle_cap", "Cap replacement"),
    ("other", "Other"),
]


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_run(run_dir: Path, label: str) -> List[Dict]:
    history_path = run_dir / "adaptive_history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing adaptive_history.jsonl: {history_path}")

    records = []
    for row in read_jsonl(history_path):
        if row.get("phase") == "outer_stop":
            continue
        if "outer" not in row or "gn_star" not in row:
            continue

        updates_before = as_int(row.get("parameter_updates_before"))
        updates_after = as_int(row.get("parameter_updates_after"), updates_before)
        update_delta = max(1, updates_after - updates_before)
        elapsed_before = as_float(row.get("elapsed_wall_seconds_before"))
        elapsed_after = as_float(row.get("elapsed_wall_seconds_after"), elapsed_before)
        fallback_total = max(0.0, elapsed_after - elapsed_before)
        total = as_float(row.get("timing_outer_total_seconds"), fallback_total)

        lambda_solver = as_float(row.get("timing_lambda_solver_seconds"))
        lambda_raw_solver = as_float(row.get("timing_lambda_raw_solver_seconds"))
        lambda_diagnostics = as_float(row.get("timing_lambda_diagnostics_seconds"))
        inner_update = as_float(row.get("timing_inner_update_seconds"))
        inner_oracle = as_float(row.get("timing_inner_oracle_seconds"))
        inner_accept = as_float(row.get("timing_inner_accept_seconds"))
        prune_inner = as_float(row.get("timing_prune_inner_seconds"))
        bundle_cap = as_float(row.get("timing_bundle_cap_seconds"))
        cap_solver = as_float(row.get("timing_bundle_cap_solver_seconds"))
        cap_solver_calls = as_int(row.get("timing_bundle_cap_solver_calls"))
        other = as_float(row.get("timing_other_overhead_seconds"))

        if other <= 0.0 and total > 0.0:
            known = (
                lambda_solver
                + lambda_raw_solver
                + lambda_diagnostics
                + inner_update
                + inner_oracle
                + inner_accept
                + prune_inner
                + bundle_cap
            )
            other = max(0.0, total - known)

        records.append({
            "label": label,
            "run_dir": str(run_dir),
            "lambda_solver_name": row.get("lambda_solver", ""),
            "gn_certificate_type": row.get("gn_certificate_type", ""),
            "outer": as_int(row.get("outer")),
            "bundle_size_before": as_int(row.get("bundle_size_before"), as_int(row.get("bundle_size"))),
            "bundle_size_after": as_int(row.get("bundle_size"), as_int(row.get("bundle_size_before"))),
            "M_t": as_int(row.get("M_t"), update_delta),
            "parameter_updates_before": updates_before,
            "parameter_updates_after": updates_after,
            "update_delta": update_delta,
            "objective_gradient_evals_after": as_int(row.get("objective_gradient_evals_after")),
            "gn_star": as_float(row.get("gn_star")),
            "timing_outer_total_seconds": total,
            "seconds_per_update": total / update_delta,
            "lambda_solver": lambda_solver + lambda_raw_solver,
            "lambda_solver_seconds": lambda_solver + lambda_raw_solver,
            "lambda_diagnostics": lambda_diagnostics,
            "inner_update": inner_update,
            "inner_oracle": inner_oracle,
            "inner_accept": inner_accept,
            "prune_inner": prune_inner,
            "bundle_cap": bundle_cap,
            "bundle_cap_solver": cap_solver,
            "bundle_cap_solver_calls": cap_solver_calls,
            "other": other,
            "lambda_solver_per_update": (lambda_solver + lambda_raw_solver) / update_delta,
            "lambda_diagnostics_per_update": lambda_diagnostics / update_delta,
            "inner_update_per_update": inner_update / update_delta,
            "inner_oracle_per_update": inner_oracle / update_delta,
            "inner_accept_per_update": inner_accept / update_delta,
            "prune_inner_per_update": prune_inner / update_delta,
            "bundle_cap_per_update": bundle_cap / update_delta,
            "bundle_cap_solver_per_update": cap_solver / update_delta,
            "other_per_update": other / update_delta,
        })
    return records


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bin_label(size: int, cuts: Sequence[int]) -> str:
    previous = 0
    for cut in cuts:
        if size <= cut:
            return f"{previous + 1}-{cut}"
        previous = cut
    return f">{cuts[-1]}"


def summarize_by_bin(rows: Sequence[Dict], cuts: Sequence[int]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (row["label"], bin_label(int(row["bundle_size_before"]), cuts))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (label, bundle_bin), group in sorted(grouped.items()):
        total_updates = sum(as_int(r["update_delta"]) for r in group)
        item = {
            "label": label,
            "bundle_bin": bundle_bin,
            "num_outers": len(group),
            "total_updates": total_updates,
            "mean_bundle_size_before": float(np.mean([r["bundle_size_before"] for r in group])),
            "mean_seconds_per_update": float(np.mean([r["seconds_per_update"] for r in group])),
            "mean_lambda_solver_per_update": float(np.mean([r["lambda_solver_per_update"] for r in group])),
            "mean_lambda_diagnostics_per_update": float(np.mean([r["lambda_diagnostics_per_update"] for r in group])),
            "mean_inner_update_per_update": float(np.mean([r["inner_update_per_update"] for r in group])),
            "mean_inner_oracle_per_update": float(np.mean([r["inner_oracle_per_update"] for r in group])),
            "mean_inner_accept_per_update": float(np.mean([r["inner_accept_per_update"] for r in group])),
            "mean_prune_inner_per_update": float(np.mean([r["prune_inner_per_update"] for r in group])),
            "mean_bundle_cap_per_update": float(np.mean([r["bundle_cap_per_update"] for r in group])),
            "mean_bundle_cap_solver_per_update": float(np.mean([r["bundle_cap_solver_per_update"] for r in group])),
            "mean_other_per_update": float(np.mean([r["other_per_update"] for r in group])),
        }
        summary.append(item)
    return summary


def plot_per_update_vs_bundle(rows: Sequence[Dict], output_dir: Path) -> None:
    labels = list(dict.fromkeys(row["label"] for row in rows))
    plt.figure(figsize=(10, 6))
    for label in labels:
        group = [row for row in rows if row["label"] == label]
        x = [row["bundle_size_before"] for row in group]
        y = [row["seconds_per_update"] for row in group]
        plt.plot(x, y, "-o", linewidth=2.2, markersize=5, label=label)
    plt.xlabel("Bundle size before outer update")
    plt.ylabel("Seconds per parameter update")
    plt.title("Per-update wall time vs bundle size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "timing_per_update_vs_bundle_size.png", dpi=220)
    plt.close()


def plot_solver_vs_bundle(rows: Sequence[Dict], output_dir: Path) -> None:
    labels = list(dict.fromkeys(row["label"] for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for label in labels:
        group = [row for row in rows if row["label"] == label]
        x = [row["bundle_size_before"] for row in group]
        axes[0].plot(
            x,
            [row["lambda_solver"] for row in group],
            "-o",
            linewidth=2.0,
            markersize=4,
            label=label,
        )
        axes[1].plot(
            x,
            [row["bundle_cap_solver"] for row in group],
            "-o",
            linewidth=2.0,
            markersize=4,
            label=label,
        )
    axes[0].set_title("Outer lambda-selection solver time")
    axes[0].set_ylabel("Seconds per outer")
    axes[1].set_title("Cap global-swap solver time")
    for ax in axes:
        ax.set_xlabel("Bundle size before outer update")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "timing_solver_vs_bundle_size.png", dpi=220)
    plt.close(fig)


def plot_phase_stack(summary: Sequence[Dict], output_dir: Path) -> None:
    if not summary:
        return
    labels = [f"{row['label']}\n{row['bundle_bin']}" for row in summary]
    x = np.arange(len(summary))
    bottom = np.zeros(len(summary), dtype=np.float64)
    phase_keys = [
        ("mean_lambda_solver_per_update", "Lambda solver"),
        ("mean_lambda_diagnostics_per_update", "Lambda diagnostics"),
        ("mean_inner_update_per_update", "Inner update"),
        ("mean_inner_oracle_per_update", "Oracle eval"),
        ("mean_inner_accept_per_update", "Inner accept"),
        ("mean_prune_inner_per_update", "Prune inner"),
        ("mean_bundle_cap_per_update", "Cap replacement"),
        ("mean_other_per_update", "Other"),
    ]
    plt.figure(figsize=(max(10, len(summary) * 0.8), 6))
    for key, label in phase_keys:
        values = np.asarray([row[key] for row in summary], dtype=np.float64)
        plt.bar(x, values, bottom=bottom, label=label)
        bottom += values
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Mean seconds per parameter update")
    plt.title("Timing breakdown by bundle-size bin")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "timing_phase_stack_by_bundle_bin.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot adaptive-bundle timing breakdown by bundle size."
    )
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("DIR", "LABEL"),
        required=True,
        help="Adaptive output directory and display label. Can be repeated.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--bundle_bins",
        default="25,50",
        help="Comma-separated upper cutoffs for bundle-size bins.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cuts = [int(x) for x in args.bundle_bins.split(",") if x.strip()]
    if not cuts:
        cuts = [25, 50]

    rows: List[Dict] = []
    for run_dir, label in args.run:
        rows.extend(load_run(Path(run_dir), label))
    if not rows:
        raise SystemExit("No adaptive timing rows found.")

    summary = summarize_by_bin(rows, cuts)
    write_csv(output_dir / "timing_outer_breakdown.csv", rows)
    write_csv(output_dir / "timing_summary_by_bundle_bin.csv", summary)
    plot_per_update_vs_bundle(rows, output_dir)
    plot_solver_vs_bundle(rows, output_dir)
    plot_phase_stack(summary, output_dir)

    print(f"saved timing breakdown to {output_dir}")
    print(f"rows: {len(rows)}")
    for row in summary:
        print(
            row["label"],
            row["bundle_bin"],
            "mean sec/update=",
            f"{row['mean_seconds_per_update']:.3f}",
            "lambda_solver=",
            f"{row['mean_lambda_solver_per_update']:.3f}",
            "lambda_diag=",
            f"{row['mean_lambda_diagnostics_per_update']:.3f}",
            "cap=",
            f"{row['mean_bundle_cap_per_update']:.3f}",
        )


if __name__ == "__main__":
    main()
