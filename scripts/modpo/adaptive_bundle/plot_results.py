from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc


DEFAULT_DPO_LW_DIR = (
    "./output/PKU-Alignment/PKU-SafeRLHF-10K/dpo_lw/qwen2_0_5b_2k_r5"
)
DEFAULT_ADAPTIVE_DIR = (
    "./output/PKU-Alignment/PKU-SafeRLHF-10K/adaptive_bundle/qwen2_0_5b_2k"
)
DEFAULT_OUTPUT_DIR = (
    "./output/PKU-Alignment/PKU-SafeRLHF-10K/figures/qwen2_0_5b_2k"
)
ADAPTIVE_METHOD_COLORS = [
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#B279A2",
    "#FF9DA6",
]


@dataclass
class ParetoPoint:
    method: str
    run: str
    helpful_loss: float
    harmless_loss: float
    lambda_helpful: Optional[float] = None
    lambda_harmless: Optional[float] = None
    step: Optional[int] = None
    outer: Optional[int] = None
    gradient_eval: Optional[int] = None
    parameter_update: Optional[int] = None
    objective_gradient_evals: Optional[int] = None
    source: str = ""


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def group_points_by_method(points: Sequence[ParetoPoint]) -> Dict[str, List[ParetoPoint]]:
    groups: Dict[str, List[ParetoPoint]] = {}
    for point in points:
        groups.setdefault(point.method, []).append(point)
    return groups


def group_rows_by_method(rows: Sequence[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("method", "Adaptive bundle")), []).append(row)
    return groups


def budget_x_from_point(point: ParetoPoint) -> int:
    value = first_present(
        point.objective_gradient_evals,
        point.parameter_update,
        point.gradient_eval,
        point.step,
        point.outer,
        0,
    )
    return int(value)


def budget_x_from_row(row: Dict, fallback: int = 0) -> int:
    value = first_present(
        row.get("objective_gradient_evals"),
        row.get("parameter_updates"),
        row.get("parameter_update"),
        row.get("cumulative_parameter_updates"),
        row.get("gradient_eval"),
        row.get("outer"),
        fallback,
    )
    return int(value)


def tail_mean(records: Sequence[Dict], key: str, tail_window: int) -> float:
    usable = [record[key] for record in records if key in record and record[key] is not None]
    if not usable:
        raise ValueError(f"No values found for `{key}`.")
    tail = usable[-max(1, tail_window):]
    return float(np.mean(np.asarray(tail, dtype=np.float64)))


def moving_average(values: Sequence[float], window: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(1, len(y) + 1, dtype=np.int64)
    if window <= 1 or len(y) < window:
        return x, y
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(y, kernel, mode="valid")
    return x[window - 1 :], smoothed


def best_so_far(values: Sequence[float]) -> np.ndarray:
    """Return the monotone best-so-far envelope for minimization metrics."""
    return np.minimum.accumulate(np.asarray(values, dtype=np.float64))


def elapsed_x_from_row(row: Dict) -> Optional[float]:
    value = first_present(
        row.get("elapsed_wall_seconds"),
        row.get("elapsed_wall_seconds_after"),
        row.get("elapsed_seconds"),
        row.get("runtime_seconds"),
        row.get("wall_time_seconds"),
    )
    return float(value) if value is not None else None


def relative_elapsed_axis(x: np.ndarray, *, log_x: bool) -> np.ndarray:
    """Align elapsed-time curves to their own first logged checkpoint.

    Absolute wall-clock time includes method-specific startup overhead such as
    model loading and dataset construction. For method comparison plots, we want
    both curves to start from a common time origin. Log plots cannot include
    zero, so we shift the relative axis by one second.
    """
    if len(x) == 0:
        return x
    origin = float(np.nanmin(x))
    relative = x - origin
    if log_x:
        relative = relative + 1.0
    return relative


def gn_value_from_row(row: Dict, gn_metric: str) -> float:
    """Convert logged squared GN* values to the requested plotting scale."""
    value = float(row["gn_star"])
    if gn_metric == "norm":
        return float(np.sqrt(max(value, 0.0)))
    if gn_metric == "squared":
        return value
    raise ValueError(f"Unknown gn_metric: {gn_metric!r}")


def gn_metric_ylabel(gn_metric: str) -> str:
    if gn_metric == "norm":
        return "Best-so-far gradient norm"
    if gn_metric == "squared":
        return "Best-so-far GN*"
    raise ValueError(f"Unknown gn_metric: {gn_metric!r}")


def gn_metric_title(gn_metric: str, suffix: str = "") -> str:
    base = (
        "Best-so-far worst-case gradient norm"
        if gn_metric == "norm"
        else "Best-so-far worst-case GN*"
    )
    return f"{base}{suffix}"


def load_dpo_lw_runs(
    dpo_lw_dir: Optional[Path],
    tail_window: int,
) -> Tuple[List[ParetoPoint], List[Dict]]:
    if dpo_lw_dir is None or not dpo_lw_dir.exists():
        return [], []

    points: List[ParetoPoint] = []
    curves: List[Dict] = []
    history_paths = sorted(dpo_lw_dir.glob("lambda_helpful_*_harmless_*/training_history.jsonl"))
    for history_path in history_paths:
        records = read_jsonl(history_path)
        if not records:
            continue

        run_dir = history_path.parent
        config = read_json(run_dir / "dpo_lw_config.json")
        last = records[-1]
        lambda_helpful = first_present(
            config.get("lambda_helpful"),
            last.get("lambda_helpful"),
        )
        lambda_harmless = first_present(
            config.get("lambda_harmless"),
            last.get("lambda_harmless"),
        )
        step = first_present(last.get("optimizer_step"), last.get("micro_step"))
        parameter_update = first_present(
            last.get("parameter_updates"),
            last.get("parameter_update"),
            last.get("optimizer_step"),
        )
        objective_gradient_evals = first_present(
            last.get("objective_gradient_evals"),
            2 * int(parameter_update) if parameter_update is not None else None,
        )

        point = ParetoPoint(
            method="DPO-LW",
            run=run_dir.name,
            helpful_loss=tail_mean(records, "helpful_loss", tail_window),
            harmless_loss=tail_mean(records, "harmless_loss", tail_window),
            lambda_helpful=float(lambda_helpful) if lambda_helpful is not None else None,
            lambda_harmless=float(lambda_harmless) if lambda_harmless is not None else None,
            step=int(step) if step is not None else None,
            parameter_update=int(parameter_update) if parameter_update is not None else None,
            objective_gradient_evals=(
                int(objective_gradient_evals)
                if objective_gradient_evals is not None
                else None
            ),
            source=f"last_{max(1, tail_window)}_records_mean",
        )
        points.append(point)
        curves.append(
            {
                "run": run_dir.name,
                "lambda_helpful": point.lambda_helpful,
                "lambda_harmless": point.lambda_harmless,
                "records": records,
            }
        )

    points.sort(key=lambda point: -1.0 if point.lambda_helpful is None else point.lambda_helpful)
    curves.sort(key=lambda curve: -1.0 if curve["lambda_helpful"] is None else curve["lambda_helpful"])
    return points, curves


def load_uniform_gn_run(
    dpo_lw_dir: Optional[Path],
) -> Tuple[List[ParetoPoint], List[Dict]]:
    if dpo_lw_dir is None or not dpo_lw_dir.exists():
        return [], []

    records = read_jsonl(dpo_lw_dir / "uniform_gn_history.jsonl")
    if not records:
        return [], []

    points: List[ParetoPoint] = []
    gn_rows: List[Dict] = []
    for idx, record in enumerate(records, start=1):
        fvals = record.get("fvals") or []
        if len(fvals) < 2:
            continue
        lambda_train = record.get("lambda_train") or []
        lambda_helpful = float(lambda_train[0]) if len(lambda_train) > 0 else None
        lambda_harmless = float(lambda_train[1]) if len(lambda_train) > 1 else None
        checkpoint_index = int(record.get("checkpoint_index", record.get("gradient_eval", idx)))
        oracle_gradient_eval = int(record.get("oracle_gradient_eval", record.get("gradient_eval", idx)))
        parameter_updates = int(first_present(
            record.get("parameter_updates"),
            record.get("parameter_update"),
            record.get("cumulative_parameter_updates"),
            checkpoint_index,
        ))
        num_objectives = int(record.get("num_objectives", 2))
        objective_gradient_evals = int(
            record.get("objective_gradient_evals", num_objectives * parameter_updates)
        )

        points.append(
            ParetoPoint(
                method="Uniform DPO-LW",
                run=record.get("run", f"uniform_eval_{idx}"),
                helpful_loss=float(fvals[0]),
                harmless_loss=float(fvals[1]),
                lambda_helpful=lambda_helpful,
                lambda_harmless=lambda_harmless,
                gradient_eval=oracle_gradient_eval,
                parameter_update=parameter_updates,
                objective_gradient_evals=objective_gradient_evals,
                source="fixed_oracle",
            )
        )
        gn_rows.append(
            {
                "phase": record.get("phase"),
                "gradient_eval": oracle_gradient_eval,
                "oracle_gradient_eval": oracle_gradient_eval,
                "checkpoint_index": checkpoint_index,
                "parameter_updates": parameter_updates,
                "objective_gradient_evals": objective_gradient_evals,
                "elapsed_wall_seconds": record.get("elapsed_wall_seconds"),
                "elapsed_wall_seconds_after": record.get("elapsed_wall_seconds_after"),
                "gn_star": record.get("gn_star"),
                "lambda_gn_star": record.get("lambda_gn_star"),
                "run": record.get("run", f"uniform_eval_{idx}"),
                "lambda_helpful": lambda_helpful,
                "lambda_harmless": lambda_harmless,
            }
        )

    points.sort(key=lambda point: -1.0 if point.lambda_helpful is None else point.lambda_helpful)
    gn_rows.sort(key=budget_x_from_row)
    return points, gn_rows


def add_adaptive_point(
    points: List[ParetoPoint],
    fvals: Sequence[float],
    outer: int,
    run: str,
    method: str,
    source: str,
    lambda_helpful: Optional[float],
    lambda_harmless: Optional[float],
    step: Optional[int],
    gradient_eval: Optional[int],
    parameter_update: Optional[int],
    objective_gradient_evals: Optional[int],
) -> None:
    if len(fvals) < 2:
        return
    points.append(
        ParetoPoint(
            method=method,
            run=run,
            helpful_loss=float(fvals[0]),
            harmless_loss=float(fvals[1]),
            lambda_helpful=lambda_helpful,
            lambda_harmless=lambda_harmless,
            outer=outer,
            step=step,
            gradient_eval=gradient_eval,
            parameter_update=parameter_update,
            objective_gradient_evals=objective_gradient_evals,
            source=source,
        )
    )


def load_adaptive_run(
    adaptive_dir: Optional[Path],
    method_label: str = "Adaptive bundle",
) -> Tuple[List[ParetoPoint], List[Dict], List[Dict]]:
    if adaptive_dir is None or not adaptive_dir.exists():
        return [], [], []

    history_path = adaptive_dir / "adaptive_history.jsonl"
    records = read_jsonl(history_path)
    if not records:
        return [], [], []
    is_surf_run = (
        method_label.lower() == "surf"
        or any(record.get("method") == "SURF" or record.get("surf_outer") is not None for record in records)
    )
    final_surf_outer = None
    if is_surf_run:
        surf_outers = [
            int(record["surf_outer"])
            for record in records
            if record.get("surf_outer") is not None
        ]
        final_surf_outer = max(surf_outers) if surf_outers else None

    points: List[ParetoPoint] = []
    lambda_rows: List[Dict] = []
    gn_rows: List[Dict] = []
    initial = read_json(adaptive_dir / "adaptive_initial_oracle.json")
    if "fvals" in initial:
        initial_parameter_updates = int(first_present(
            initial.get("parameter_updates"),
            initial.get("parameter_update"),
            0,
        ))
        initial_oracle_gradient_eval = int(
            initial.get("oracle_gradient_eval", initial.get("gradient_eval", 1))
        )
        initial_objective_gradient_evals = int(
            initial.get("objective_gradient_evals", 2 * initial_parameter_updates)
        )
        add_adaptive_point(
            points,
            initial["fvals"],
            outer=0,
            run="initial",
            method=method_label,
            source="initial_fixed_oracle",
            lambda_helpful=None,
            lambda_harmless=None,
            step=0,
            gradient_eval=initial_oracle_gradient_eval,
            parameter_update=initial_parameter_updates,
            objective_gradient_evals=initial_objective_gradient_evals,
        )

    fallback_parameter_updates = 0
    for record in records:
        outer = int(record.get("outer", len(lambda_rows) + 1))
        lam = record.get("lambda") or []
        lambda_helpful = float(lam[0]) if len(lam) > 0 else None
        lambda_harmless = float(lam[1]) if len(lam) > 1 else None
        total_inner_steps = record.get("total_inner_steps")
        total_inner_steps = int(total_inner_steps) if total_inner_steps is not None else None
        inner_records = record.get("inner", [])
        gradient_evals_before = int(
            record.get(
                "oracle_gradient_evals_before",
                record.get("gradient_evals_before", outer),
            )
        )
        gradient_evals_after = int(
            record.get(
                "oracle_gradient_evals_after",
                record.get("gradient_evals_after", gradient_evals_before),
            )
        )
        parameter_updates_before = int(first_present(
            record.get("parameter_updates_before"),
            record.get("parameter_update_before"),
            fallback_parameter_updates,
        ))
        parameter_updates_after = int(first_present(
            record.get("parameter_updates_after"),
            record.get("parameter_update_after"),
            total_inner_steps,
            parameter_updates_before + len(inner_records),
        ))
        num_objectives = int(record.get("num_objectives", 2))
        objective_gradient_evals_before = int(
            record.get(
                "objective_gradient_evals_before",
                num_objectives * parameter_updates_before,
            )
        )
        objective_gradient_evals_after = int(
            record.get(
                "objective_gradient_evals_after",
                num_objectives * parameter_updates_after,
            )
        )

        lambda_rows.append(
            {
                "method": method_label,
                "adaptive_dir": str(adaptive_dir),
                "outer": outer,
                "lambda_helpful": lambda_helpful,
                "lambda_harmless": lambda_harmless,
                "gradient_eval": gradient_evals_before,
                "oracle_gradient_eval": gradient_evals_before,
                "parameter_updates": parameter_updates_before,
                "objective_gradient_evals": objective_gradient_evals_before,
            }
        )
        gn_rows.append(
            {
                "method": method_label,
                "adaptive_dir": str(adaptive_dir),
                "phase": record.get("phase"),
                "outer": outer,
                "gradient_eval": gradient_evals_before,
                "oracle_gradient_eval": gradient_evals_before,
                "parameter_updates": parameter_updates_before,
                "parameter_updates_after": parameter_updates_after,
                "objective_gradient_evals": objective_gradient_evals_before,
                "objective_gradient_evals_after": objective_gradient_evals_after,
                "elapsed_wall_seconds": first_present(
                    record.get("elapsed_wall_seconds_before"),
                    record.get("elapsed_wall_seconds"),
                ),
                "elapsed_wall_seconds_after": first_present(
                    record.get("elapsed_wall_seconds_after"),
                    record.get("elapsed_wall_seconds"),
                ),
                "gn_star": record.get("gn_star"),
                "bundle_size": record.get("bundle_size"),
            }
        )

        include_record_point = "fvals" in record
        if is_surf_run and final_surf_outer is not None:
            include_record_point = (
                include_record_point
                and record.get("surf_outer") is not None
                and int(record["surf_outer"]) == int(final_surf_outer)
            )
        if include_record_point:
            run_name = str(record.get("run") or f"outer_{outer}")
            add_adaptive_point(
                points,
                record["fvals"],
                outer,
                run_name,
                method_label,
                "outer_record",
                lambda_helpful,
                lambda_harmless,
                parameter_updates_after,
                gradient_evals_after,
                parameter_updates_after,
                objective_gradient_evals_after,
            )

        for inner_idx, inner_record in enumerate(inner_records, start=1):
            if "fvals" not in inner_record:
                continue
            if inner_record.get("candidate_accepted") is False:
                continue
            if (
                record.get("prune_inner")
                and record.get("bundle_update_mode") == "append"
                and record.get("retained_bundle_index") is not None
            ):
                if inner_record.get("bundle_index") != record.get("retained_bundle_index"):
                    continue
            inner_gradient_eval = inner_record.get("gradient_eval")
            if inner_gradient_eval is None:
                inner_gradient_eval = gradient_evals_before + inner_idx
            inner_parameter_update = int(first_present(
                inner_record.get("parameter_updates"),
                inner_record.get("parameter_update"),
                parameter_updates_before + inner_idx,
            ))
            inner_objective_gradient_evals = int(
                inner_record.get(
                    "objective_gradient_evals",
                    num_objectives * inner_parameter_update,
                )
            )
            add_adaptive_point(
                points,
                inner_record["fvals"],
                outer,
                f"outer_{outer}_inner_{inner_idx}",
                method_label,
                "inner_oracle",
                lambda_helpful,
                lambda_harmless,
                inner_parameter_update,
                int(inner_gradient_eval) if inner_gradient_eval is not None else None,
                inner_parameter_update,
                inner_objective_gradient_evals,
            )
        fallback_parameter_updates = parameter_updates_after

    return points, lambda_rows, gn_rows


def save_summary(points: Sequence[ParetoPoint], output_dir: Path) -> Path:
    path = output_dir / "results_summary.csv"
    fields = [
        "method",
        "run",
        "helpful_loss",
        "harmless_loss",
        "lambda_helpful",
        "lambda_harmless",
        "step",
        "outer",
        "gradient_eval",
        "parameter_update",
        "objective_gradient_evals",
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "method": point.method,
                    "run": point.run,
                    "helpful_loss": point.helpful_loss,
                    "harmless_loss": point.harmless_loss,
                    "lambda_helpful": point.lambda_helpful,
                    "lambda_harmless": point.lambda_harmless,
                    "step": point.step,
                    "outer": point.outer,
                    "gradient_eval": point.gradient_eval,
                    "parameter_update": point.parameter_update,
                    "objective_gradient_evals": point.objective_gradient_evals,
                    "source": point.source,
                }
            )
    return path


def save_representatives(
    dpo_points: Sequence[ParetoPoint],
    adaptive_points: Sequence[ParetoPoint],
    output_dir: Path,
    lambda_round_decimals: int,
) -> Path:
    path = output_dir / "pareto_representatives.csv"
    representatives = [
        *best_observed_per_lambda(dpo_points, lambda_round_decimals),
        *best_observed_per_lambda(adaptive_points, lambda_round_decimals),
    ]
    fields = [
        "method",
        "run",
        "helpful_loss",
        "harmless_loss",
        "lambda_helpful",
        "lambda_harmless",
        "scalarized_loss",
        "gradient_eval",
        "parameter_update",
        "objective_gradient_evals",
        "outer",
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in representatives:
            writer.writerow(
                {
                    "method": point.method,
                    "run": point.run,
                    "helpful_loss": point.helpful_loss,
                    "harmless_loss": point.harmless_loss,
                    "lambda_helpful": point.lambda_helpful,
                    "lambda_harmless": point.lambda_harmless,
                    "scalarized_loss": scalarized_loss(point),
                    "gradient_eval": point.gradient_eval,
                    "parameter_update": point.parameter_update,
                    "objective_gradient_evals": point.objective_gradient_evals,
                    "outer": point.outer,
                    "source": point.source,
                }
            )
    return path


def save_frontiers(
    dpo_points: Sequence[ParetoPoint],
    adaptive_points: Sequence[ParetoPoint],
    output_dir: Path,
    lambda_round_decimals: int,
) -> Path:
    path = output_dir / "pareto_frontiers.csv"
    dpo_frontier: List[ParetoPoint] = []
    for group in group_points_by_method(
        best_observed_per_lambda(dpo_points, lambda_round_decimals)
    ).values():
        dpo_frontier.extend(nondominated_points(group))
    adaptive_frontier: List[ParetoPoint] = []
    for group in group_points_by_method(
        best_observed_per_lambda(adaptive_points, lambda_round_decimals)
    ).values():
        adaptive_frontier.extend(nondominated_points(group))
    fields = [
        "method",
        "run",
        "helpful_loss",
        "harmless_loss",
        "lambda_helpful",
        "lambda_harmless",
        "scalarized_loss",
        "objective_gradient_evals",
        "outer",
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in [*dpo_frontier, *adaptive_frontier]:
            writer.writerow(
                {
                    "method": point.method,
                    "run": point.run,
                    "helpful_loss": point.helpful_loss,
                    "harmless_loss": point.harmless_loss,
                    "lambda_helpful": point.lambda_helpful,
                    "lambda_harmless": point.lambda_harmless,
                    "scalarized_loss": scalarized_loss(point),
                    "objective_gradient_evals": point.objective_gradient_evals,
                    "outer": point.outer,
                    "source": point.source,
                }
            )
    return path


def setup_axes(ax, title: Optional[str] = None) -> None:
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=10)


def nondominated_points(points: Sequence[ParetoPoint]) -> List[ParetoPoint]:
    """Return points not dominated in the two-loss minimization plane."""
    if not points:
        return []
    values = np.asarray(
        [[point.helpful_loss, point.harmless_loss] for point in points],
        dtype=np.float64,
    )
    keep = []
    for idx, value in enumerate(values):
        no_worse = np.all(values <= value, axis=1)
        strictly_better = np.any(values < value, axis=1)
        dominated = bool(np.any(no_worse & strictly_better))
        if not dominated:
            keep.append(points[idx])
    keep.sort(key=lambda point: point.helpful_loss)
    return keep


def objective_ordered_path(points: Sequence[ParetoPoint]) -> List[ParetoPoint]:
    """Order plotted representatives into a readable objective-space path."""
    return sorted(points, key=lambda point: (point.helpful_loss, point.harmless_loss))


def lambda_color_values(points: Sequence[ParetoPoint]) -> np.ndarray:
    return np.asarray(
        [
            np.nan if point.lambda_helpful is None else point.lambda_helpful
            for point in points
        ],
        dtype=np.float64,
    )


def lambda_key(point: ParetoPoint, decimals: int) -> Optional[Tuple[float, float]]:
    if point.lambda_helpful is None:
        return None
    lambda_harmless = (
        1.0 - point.lambda_helpful
        if point.lambda_harmless is None
        else point.lambda_harmless
    )
    return (
        round(float(point.lambda_helpful), decimals),
        round(float(lambda_harmless), decimals),
    )


def format_lambda_label(value: float) -> str:
    """Format lambda labels without hiding near-endpoint SURF weights."""
    value = min(max(float(value), 0.0), 1.0)
    if np.isclose(value, 0.0) or np.isclose(value, 1.0):
        return f"{value:.1f}"
    edge_distance = min(value, 1.0 - value)
    if edge_distance < 0.01:
        return f"{value:.3f}"
    if edge_distance < 0.1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def scalarized_loss(point: ParetoPoint) -> float:
    if point.lambda_helpful is None:
        return float("inf")
    lambda_harmless = (
        1.0 - point.lambda_helpful
        if point.lambda_harmless is None
        else point.lambda_harmless
    )
    return (
        float(point.lambda_helpful) * point.helpful_loss
        + float(lambda_harmless) * point.harmless_loss
    )


def best_observed_per_lambda(
    points: Sequence[ParetoPoint],
    decimals: int,
) -> List[ParetoPoint]:
    """Keep one best-observed candidate for each lambda value.

    The true Pareto point for a lambda is the optimizer of the scalarized
    objective. From logs we only know evaluated candidates, so the plotting
    representative is the lowest observed lambda-weighted DPO loss.
    """
    best: Dict[Tuple[str, float, float], ParetoPoint] = {}
    for point in points:
        key = lambda_key(point, decimals)
        if key is None:
            continue
        key = (point.method, *key)
        incumbent = best.get(key)
        if incumbent is None or scalarized_loss(point) < scalarized_loss(incumbent):
            best[key] = point
    representatives = list(best.values())
    representatives.sort(key=lambda point: point.lambda_helpful if point.lambda_helpful is not None else -1.0)
    return representatives


def initial_points(points: Sequence[ParetoPoint]) -> List[ParetoPoint]:
    return [point for point in points if point.lambda_helpful is None]


def scatter_lambda_points(
    ax,
    points: Sequence[ParetoPoint],
    *,
    marker: str,
    label: str,
    alpha: float,
    size: float,
    cmap,
    norm,
    zorder: int,
) -> None:
    if not points:
        return

    with_lambda = [point for point in points if point.lambda_helpful is not None]
    without_lambda = [point for point in points if point.lambda_helpful is None]

    if with_lambda:
        ax.scatter(
            [point.helpful_loss for point in with_lambda],
            [point.harmless_loss for point in with_lambda],
            c=lambda_color_values(with_lambda),
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.35,
            label=label,
            zorder=zorder,
        )

    if without_lambda:
        ax.scatter(
            [point.helpful_loss for point in without_lambda],
            [point.harmless_loss for point in without_lambda],
            marker="D",
            s=size * 0.9,
            alpha=0.8,
            facecolor="#8A8A8A",
            edgecolor="black",
            linewidth=0.35,
            label="Initial point",
            zorder=zorder,
        )


def plot_pareto_front(
    dpo_points: Sequence[ParetoPoint],
    adaptive_points: Sequence[ParetoPoint],
    output_dir: Path,
    title: Optional[str],
    annotate: bool,
    lambda_round_decimals: int,
) -> Optional[Path]:
    if not dpo_points and not adaptive_points:
        return None

    dpo_representatives = best_observed_per_lambda(dpo_points, lambda_round_decimals)
    adaptive_representatives = best_observed_per_lambda(adaptive_points, lambda_round_decimals)
    initials = initial_points([*dpo_points, *adaptive_points])

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    setup_axes(ax, title or "Best observed Pareto front from DPO objective losses")
    ax.set_xlabel("Helpful DPO loss (lower is better)")
    ax.set_ylabel("Harmless DPO loss (lower is better)")
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)

    if dpo_representatives:
        scatter_lambda_points(
            ax,
            dpo_representatives,
            marker="s",
            label="Uniform DPO-LW best per lambda",
            alpha=0.85,
            size=62,
            cmap=cmap,
            norm=norm,
            zorder=3,
        )
        frontier = nondominated_points(dpo_representatives)
        if len(frontier) >= 2:
            ax.plot(
                [point.helpful_loss for point in frontier],
                [point.harmless_loss for point in frontier],
                color="#4C78A8",
                alpha=0.9,
                linewidth=2.3,
                marker="s",
                markersize=3.6,
                label="Uniform DPO-LW frontier",
                zorder=5,
            )
        if annotate:
            for point in frontier:
                if point.lambda_helpful is None:
                    continue
                ax.annotate(
                    format_lambda_label(point.lambda_helpful),
                    (point.helpful_loss, point.harmless_loss),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

    adaptive_markers = ["o", "^", "P", "X", "v", "*"]
    for method_idx, (method, method_points) in enumerate(group_points_by_method(adaptive_representatives).items()):
        marker = adaptive_markers[method_idx % len(adaptive_markers)]
        line_color = ADAPTIVE_METHOD_COLORS[method_idx % len(ADAPTIVE_METHOD_COLORS)]
        scatter_lambda_points(
            ax,
            method_points,
            marker=marker,
            label=f"{method} best per lambda",
            alpha=0.9,
            size=60,
            cmap=cmap,
            norm=norm,
            zorder=4 + method_idx,
        )
        frontier = nondominated_points(method_points)
        if len(frontier) >= 2:
            ax.plot(
                [point.helpful_loss for point in frontier],
                [point.harmless_loss for point in frontier],
                color=line_color,
                alpha=0.9,
                linewidth=2.3,
                marker=marker,
                markersize=3.6,
                label=f"{method} frontier",
                zorder=5 + method_idx,
            )
        if annotate:
            for point in frontier:
                if point.lambda_helpful is None:
                    continue
                ax.annotate(
                    format_lambda_label(point.lambda_helpful),
                    (point.helpful_loss, point.harmless_loss),
                    textcoords="offset points",
                    xytext=(4, -9 - 3 * method_idx),
                    fontsize=8,
                    color=line_color,
                )

    if initials:
        unique_initials = []
        seen_initials = set()
        for point in initials:
            key = (round(point.helpful_loss, 8), round(point.harmless_loss, 8))
            if key not in seen_initials:
                unique_initials.append(point)
                seen_initials.add(key)
        scatter_lambda_points(
            ax,
            unique_initials,
            marker="D",
            label="Initial point",
            alpha=0.8,
            size=54,
            cmap=cmap,
            norm=norm,
            zorder=5,
        )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("lambda_helpful")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "pareto_front.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def publication_pareto_style(method: str) -> Dict[str, object]:
    """Use the same method encoding across GN and DPO-loss paper figures."""
    normalized = method.lower()
    if normalized in {"dpo-lw", "uniform dpo-lw"} or "uniform" in normalized:
        return {"color": "#1F77B4", "marker": "s", "linestyle": "--"}
    if "surf" in normalized:
        return {"color": "#D62728", "marker": "^", "linestyle": "--"}
    return {"color": "#FF7F0E", "marker": "o", "linestyle": "-"}


def publication_method_label(method: str) -> str:
    return "Uniform DPO-LW" if method == "DPO-LW" else method


def sparse_frontier_annotations(points: Sequence[ParetoPoint]) -> List[ParetoPoint]:
    """Keep at most five well-spaced lambda labels for a readable paper plot."""
    with_lambda = [point for point in points if point.lambda_helpful is not None]
    if len(with_lambda) <= 5:
        return with_lambda
    indices = np.linspace(0, len(with_lambda) - 1, num=5, dtype=int)
    selected: List[ParetoPoint] = []
    seen = set()
    for index in indices:
        point = with_lambda[int(index)]
        key = round(float(point.lambda_helpful), 3)
        if key not in seen:
            selected.append(point)
            seen.add(key)
    return selected


def plot_publication_pareto_front(
    dpo_points: Sequence[ParetoPoint],
    adaptive_points: Sequence[ParetoPoint],
    output_dir: Path,
    *,
    annotate: bool,
    lambda_round_decimals: int,
) -> Optional[Path]:
    """Save a compact, method-first DPO-loss Pareto figure for the paper."""
    if not dpo_points and not adaptive_points:
        return None

    grouped: Dict[str, List[ParetoPoint]] = {}
    if dpo_points:
        grouped["Uniform DPO-LW"] = best_observed_per_lambda(
            dpo_points, lambda_round_decimals
        )
    for method, method_points in group_points_by_method(adaptive_points).items():
        grouped[method] = best_observed_per_lambda(method_points, lambda_round_decimals)

    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=240)
    legend_handles = []
    legend_labels = []
    for method, representatives in grouped.items():
        if not representatives:
            continue
        style = publication_pareto_style(method)
        xs = [point.helpful_loss for point in representatives]
        ys = [point.harmless_loss for point in representatives]
        # Keep every representative visible as context without competing with the frontier.
        ax.scatter(
            xs,
            ys,
            s=28,
            marker=style["marker"],
            color=style["color"],
            alpha=0.20,
            linewidths=0,
            zorder=1,
        )
        frontier = nondominated_points(representatives)
        if not frontier:
            continue
        (line,) = ax.plot(
            [point.helpful_loss for point in frontier],
            [point.harmless_loss for point in frontier],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.35,
            marker=style["marker"],
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.65,
            label=publication_method_label(method),
            zorder=3,
        )
        legend_handles.append(line)
        legend_labels.append(publication_method_label(method))
        if annotate:
            for point in sparse_frontier_annotations(frontier):
                ax.annotate(
                    format_lambda_label(float(point.lambda_helpful)),
                    (point.helpful_loss, point.harmless_loss),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=8,
                    color=style["color"],
                )

    initials = initial_points([*dpo_points, *adaptive_points])
    unique_initials = []
    seen_initials = set()
    for point in initials:
        key = (round(point.helpful_loss, 8), round(point.harmless_loss, 8))
        if key not in seen_initials:
            unique_initials.append(point)
            seen_initials.add(key)
    if unique_initials:
        initial_handle = ax.scatter(
            [point.helpful_loss for point in unique_initials],
            [point.harmless_loss for point in unique_initials],
            marker="D",
            s=54,
            facecolor="#9A9A9A",
            edgecolor="white",
            linewidth=0.6,
            label="SFT initialization",
            zorder=4,
        )
        legend_handles.append(initial_handle)
        legend_labels.append("SFT initialization")

    ax.set_xlabel("Helpful DPO loss (lower is better)", fontsize=11)
    ax.set_ylabel("Harmless DPO loss (lower is better)", fontsize=11)
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            frameon=False,
            fontsize=9,
        )
    fig.tight_layout()
    path = output_dir / "pareto_front_publication.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def group_points_by_lambda(
    points: Sequence[ParetoPoint],
    decimals: int,
) -> Dict[Tuple[str, float, float], List[ParetoPoint]]:
    groups: Dict[Tuple[str, float, float], List[ParetoPoint]] = {}
    for point in points:
        key = lambda_key(point, decimals)
        if key is None:
            continue
        groups.setdefault((point.method, *key), []).append(point)
    for group in groups.values():
        group.sort(
            key=lambda point: (
                budget_x_from_point(point),
                point.outer if point.outer is not None else 0,
                point.step if point.step is not None else 0,
                point.run,
            )
        )
    return groups


def plot_adaptive_lambda_trajectories(
    adaptive_points: Sequence[ParetoPoint],
    output_dir: Path,
    lambda_round_decimals: int,
    annotate: bool,
) -> Optional[Path]:
    groups = group_points_by_lambda(adaptive_points, lambda_round_decimals)
    if not groups:
        return None

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    setup_axes(ax, "Adaptive bundle trajectories by lambda")
    ax.set_xlabel("Helpful DPO loss (lower is better)")
    ax.set_ylabel("Harmless DPO loss (lower is better)")

    for (method, lambda_helpful, lambda_harmless), group in sorted(groups.items()):
        color = cmap(norm(lambda_helpful))
        xs = [point.helpful_loss for point in group]
        ys = [point.harmless_loss for point in group]
        label = f"{method} lambda=({lambda_helpful:g},{lambda_harmless:g})"
        if len(group) > 1:
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.4,
                alpha=0.6,
                label=label,
                zorder=2,
            )
        ax.scatter(
            xs,
            ys,
            c=[lambda_helpful] * len(group),
            cmap=cmap,
            norm=norm,
            marker="o",
            s=48,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        if annotate:
            for point in group:
                label_value = first_present(
                    point.objective_gradient_evals,
                    point.parameter_update,
                    point.gradient_eval,
                    point.outer,
                )
                if label_value is None:
                    continue
                ax.annotate(
                    str(label_value),
                    (point.helpful_loss, point.harmless_loss),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color="#333333",
                )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("lambda_helpful")
    handles, labels = ax.get_legend_handles_labels()
    if handles and len(handles) <= 8:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "adaptive_lambda_trajectories.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_uniform_lambda_trajectories(
    curves: Sequence[Dict],
    output_dir: Path,
    smooth_window: int,
    annotate: bool,
) -> Optional[Path]:
    if not curves:
        return None

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    setup_axes(ax, "Uniform DPO-LW trajectories by lambda")
    ax.set_xlabel("Helpful DPO loss (lower is better)")
    ax.set_ylabel("Harmless DPO loss (lower is better)")

    for curve in curves:
        lambda_helpful = curve["lambda_helpful"]
        lambda_harmless = curve["lambda_harmless"]
        if lambda_helpful is None:
            continue

        records = [
            record
            for record in curve["records"]
            if "helpful_loss" in record and "harmless_loss" in record
        ]
        if not records:
            continue

        helpful_values = [float(record["helpful_loss"]) for record in records]
        harmless_values = [float(record["harmless_loss"]) for record in records]
        _, helpful_smoothed = moving_average(helpful_values, smooth_window)
        _, harmless_smoothed = moving_average(harmless_values, smooth_window)
        length = min(len(helpful_smoothed), len(harmless_smoothed))
        if length == 0:
            continue

        xs = helpful_smoothed[:length]
        ys = harmless_smoothed[:length]
        color = cmap(norm(float(lambda_helpful)))
        label = (
            f"lambda=({format_lambda_label(lambda_helpful)},"
            f"{format_lambda_label(lambda_harmless)})"
        )
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.35,
            alpha=0.75,
            label=label,
            zorder=2,
        )
        ax.scatter(
            [xs[0], xs[-1]],
            [ys[0], ys[-1]],
            c=[lambda_helpful, lambda_helpful],
            cmap=cmap,
            norm=norm,
            marker="s",
            s=[34, 58],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )
        if annotate:
            ax.annotate(
                format_lambda_label(lambda_helpful),
                (xs[-1], ys[-1]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color="#333333",
            )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("lambda_helpful")
    handles, labels = ax.get_legend_handles_labels()
    if handles and len(handles) <= 12:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "uniform_lambda_trajectories.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_lambda_path(lambda_rows: Sequence[Dict], output_dir: Path) -> Optional[Path]:
    if not lambda_rows:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    setup_axes(ax, "Adaptive bundle lambda path")
    for method_idx, (method, method_rows) in enumerate(group_rows_by_method(lambda_rows).items()):
        color = ADAPTIVE_METHOD_COLORS[method_idx % len(ADAPTIVE_METHOD_COLORS)]
        rows = sorted(method_rows, key=lambda row: row["outer"])
        outer = np.asarray([row["outer"] for row in rows], dtype=np.int64)
        helpful = np.asarray([row["lambda_helpful"] for row in rows], dtype=np.float64)
        harmless = np.asarray([row["lambda_harmless"] for row in rows], dtype=np.float64)
        ax.plot(
            outer,
            helpful,
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"{method} lambda_helpful",
        )
        ax.plot(
            outer,
            harmless,
            marker="s",
            linewidth=1.8,
            linestyle="--",
            color=color,
            label=f"{method} lambda_harmless",
        )
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Lambda")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = output_dir / "lambda_path.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dpo_training_curves(
    curves: Sequence[Dict],
    output_dir: Path,
    smooth_window: int,
) -> Optional[Path]:
    if not curves:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), dpi=180, sharex=True)
    curve_specs = [
        ("loss", "Weighted loss"),
        ("helpful_loss", "Helpful DPO loss"),
        ("harmless_loss", "Harmless DPO loss"),
    ]

    cmap = plt.get_cmap("viridis")
    for curve in curves:
        records = curve["records"]
        lambda_helpful = curve["lambda_helpful"]
        color_value = 0.0 if lambda_helpful is None else float(lambda_helpful)
        color = cmap(color_value)
        label = (
            curve["run"]
            if lambda_helpful is None
            else f"lambda_helpful={format_lambda_label(lambda_helpful)}"
        )

        for ax, (key, ylabel) in zip(axes, curve_specs):
            values = [float(record[key]) for record in records if key in record]
            if not values:
                continue
            x, y = moving_average(values, smooth_window)
            ax.plot(x, y, linewidth=1.4, color=color, alpha=0.9, label=label)
            ax.set_ylabel(ylabel)
            setup_axes(ax)

    axes[-1].set_xlabel("Logged micro step")
    axes[0].set_title("DPO-LW training curves")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(3, len(handles)),
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.16)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    path = output_dir / "dpo_lw_training_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_adaptive_trace(
    adaptive_points: Sequence[ParetoPoint],
    gn_rows: Sequence[Dict],
    output_dir: Path,
    *,
    gn_metric: str = "norm",
) -> Optional[Path]:
    if not adaptive_points and not gn_rows:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.4), dpi=180, sharex=False)

    if adaptive_points:
        for method_idx, (method, method_points) in enumerate(group_points_by_method(adaptive_points).items()):
            color = ADAPTIVE_METHOD_COLORS[method_idx % len(ADAPTIVE_METHOD_COLORS)]
            ordered = sorted(
                method_points,
                key=lambda point: (
                    budget_x_from_point(point),
                    point.outer if point.outer is not None else 0,
                    point.step if point.step is not None else 0,
                    point.run,
                ),
            )
            x = np.asarray(
                [
                    budget_x_from_point(point)
                    for point in ordered
                ],
                dtype=np.int64,
            )
            axes[0].plot(
                x,
                [point.helpful_loss for point in ordered],
                marker="o",
                linewidth=1.6,
                color=color,
                label=f"{method} helpful_loss",
            )
            axes[0].plot(
                x,
                [point.harmless_loss for point in ordered],
                marker="s",
                linewidth=1.6,
                linestyle="--",
                color=color,
                label=f"{method} harmless_loss",
            )
        axes[0].set_xlabel("Objective gradient evaluations")
        axes[0].set_ylabel("DPO loss")
        axes[0].legend(frameon=False)
    setup_axes(axes[0], "Adaptive bundle objective trace")

    for method_idx, (method, method_rows) in enumerate(group_rows_by_method(gn_rows).items()):
        color = ADAPTIVE_METHOD_COLORS[method_idx % len(ADAPTIVE_METHOD_COLORS)]
        usable_gn = [
            row
            for row in method_rows
            if row.get("gn_star") is not None and row.get("outer") is not None
            and row.get("phase") != "warm_start"
        ]
        if not usable_gn:
            continue
        parameter_updates = [budget_x_from_row(row) for row in usable_gn]
        gn_star = best_so_far([gn_value_from_row(row, gn_metric) for row in usable_gn])
        axes[1].plot(
            parameter_updates,
            gn_star,
            marker="o",
            linewidth=1.6,
            color=color,
            label=method,
        )
        axes[1].set_xlabel("Objective gradient evaluations")
        axes[1].set_ylabel(gn_metric_ylabel(gn_metric))
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(frameon=False)
    setup_axes(
        axes[1],
        f"{gn_metric_ylabel(gn_metric)} over objective gradient evaluations",
    )

    fig.tight_layout()
    path = output_dir / "adaptive_training_trace.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def adaptive_checkpoint_gn_rows(rows: Sequence[Dict]) -> List[Dict]:
    """Approximate MOA-style post-update adaptive GN* checkpoints.

    Adaptive logs store the full worst-case GN* at the start of each outer
    iteration. Therefore outer t+1 is the first full GN* recomputation after
    the update performed during outer t. We shift those values back to the
    previous row's after-update budget.
    """
    usable = [
        row
        for row in rows
        if row.get("gn_star") is not None and row.get("phase") != "warm_start"
    ]
    usable = sorted(usable, key=budget_x_from_row)
    if not usable:
        return []
    if any(row.get("phase") == "surf_outer_end" or row.get("method") == "SURF" for row in usable):
        checkpoints = []
        for row in usable:
            checkpoint = dict(row)
            checkpoint["checkpoint_kind"] = "post_surf_outer"
            checkpoints.append(checkpoint)
        return checkpoints

    checkpoints: List[Dict] = []
    first = dict(usable[0])
    first["checkpoint_kind"] = "initial_before_outer"
    checkpoints.append(first)

    for prev, current in zip(usable, usable[1:]):
        shifted = dict(current)
        shifted["checkpoint_kind"] = "post_previous_outer"
        shifted["objective_gradient_evals"] = first_present(
            prev.get("objective_gradient_evals_after"),
            prev.get("objective_gradient_evals"),
        )
        shifted["parameter_updates"] = first_present(
            prev.get("parameter_updates_after"),
            prev.get("parameter_updates"),
        )
        shifted["elapsed_wall_seconds"] = first_present(
            prev.get("elapsed_wall_seconds_after"),
            prev.get("elapsed_wall_seconds"),
        )
        checkpoints.append(shifted)

    return checkpoints


def gn_series(
    rows: Sequence[Dict],
    *,
    use_elapsed_time: bool,
    adaptive: bool,
    gn_metric: str = "norm",
) -> Tuple[np.ndarray, np.ndarray]:
    usable = adaptive_checkpoint_gn_rows(rows) if adaptive else [
        row for row in rows if row.get("gn_star") is not None
    ]
    usable = sorted(usable, key=budget_x_from_row)
    x_values: List[float] = []
    y_values: List[float] = []
    for idx, row in enumerate(usable):
        x_value = elapsed_x_from_row(row) if use_elapsed_time else budget_x_from_row(row, idx + 1)
        if x_value is None:
            continue
        x_values.append(float(x_value))
        y_values.append(gn_value_from_row(row, gn_metric))
    if not x_values:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    order = np.argsort(np.asarray(x_values, dtype=np.float64))
    x = np.asarray(x_values, dtype=np.float64)[order]
    y = np.asarray(y_values, dtype=np.float64)[order]
    return x, best_so_far(y)


def plot_gn_star_comparison(
    adaptive_gn_rows: Sequence[Dict],
    uniform_gn_rows: Sequence[Dict],
    output_dir: Path,
    *,
    use_elapsed_time: bool = False,
    log_x: bool = False,
    relative_time: bool = True,
    gn_metric: str = "norm",
) -> Optional[Path]:
    adaptive_series: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for method, method_rows in group_rows_by_method(adaptive_gn_rows).items():
        adaptive_x, adaptive_y = gn_series(
            method_rows,
            use_elapsed_time=use_elapsed_time,
            adaptive=True,
            gn_metric=gn_metric,
        )
        if use_elapsed_time and relative_time:
            adaptive_x = relative_elapsed_axis(adaptive_x, log_x=log_x)
        if log_x:
            adaptive_mask = adaptive_x > 0
            adaptive_x = adaptive_x[adaptive_mask]
            adaptive_y = adaptive_y[adaptive_mask]
        if len(adaptive_x) > 0:
            adaptive_series.append((method, adaptive_x, adaptive_y))

    uniform_x, uniform_y = gn_series(
        uniform_gn_rows,
        use_elapsed_time=use_elapsed_time,
        adaptive=False,
        gn_metric=gn_metric,
    )
    if use_elapsed_time and relative_time:
        uniform_x = relative_elapsed_axis(uniform_x, log_x=log_x)

    if log_x:
        uniform_mask = uniform_x > 0
        uniform_x = uniform_x[uniform_mask]
        uniform_y = uniform_y[uniform_mask]

    if not adaptive_series and len(uniform_x) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    setup_axes(
        ax,
        gn_metric_title(gn_metric, " comparison")
        if not use_elapsed_time
        else (
            gn_metric_title(gn_metric, " vs log elapsed time")
            if log_x
            else gn_metric_title(gn_metric, " vs elapsed time")
        ),
    )
    ax.set_xlabel(
        (
            (
                "Relative elapsed wall time (seconds, log scale)"
                if relative_time
                else "Elapsed wall time (seconds, log scale)"
            )
            if log_x
            else (
                "Relative elapsed wall time (seconds)"
                if relative_time
                else "Elapsed wall time (seconds)"
            )
        )
        if use_elapsed_time
        else "Objective gradient evaluations (= parameter updates * K)"
    )
    ax.set_ylabel(gn_metric_ylabel(gn_metric))
    if log_x:
        ax.set_xscale("log")

    for method_idx, (method, adaptive_x, adaptive_y) in enumerate(adaptive_series):
        color = ADAPTIVE_METHOD_COLORS[method_idx % len(ADAPTIVE_METHOD_COLORS)]
        ax.plot(
            adaptive_x,
            adaptive_y,
            marker="o",
            linewidth=1.8,
            color=color,
            label=method,
        )

    if len(uniform_x) > 0:
        ax.plot(
            uniform_x,
            uniform_y,
            marker="s",
            linewidth=1.8,
            color="#4C78A8",
            label="Uniform DPO-LW",
        )

    ax.legend(frameon=False)
    fig.tight_layout()
    if use_elapsed_time and log_x:
        filename = (
            "gn_star_log_time_comparison.png"
            if relative_time
            else "gn_star_log_absolute_time_comparison.png"
        )
    elif use_elapsed_time:
        filename = (
            "gn_star_time_comparison.png"
            if relative_time
            else "gn_star_absolute_time_comparison.png"
        )
    else:
        filename = "gn_star_comparison.png"
    path = output_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def publication_gn_style(method: str, *, uniform: bool) -> Dict[str, object]:
    """Return a stable, paper-oriented visual style for GN trajectories."""
    normalized = method.lower()
    if uniform:
        return {
            "color": "#1F77B4",
            "linestyle": "--",
            "marker": "s",
            "markersize": 4.5,
        }
    if "surf" in normalized:
        return {
            "color": "#D62728",
            "linestyle": "--",
            "marker": "^",
            "markersize": 4.8,
        }
    return {
        "color": "#FF7F0E",
        "linestyle": "-",
        "marker": None,
        "markersize": 0.0,
    }


def plot_publication_gn_comparison(
    adaptive_gn_rows: Sequence[Dict],
    uniform_gn_rows: Sequence[Dict],
    output_dir: Path,
    *,
    relative_time: bool,
    gn_metric: str = "norm",
) -> Optional[Path]:
    """Plot the GN trajectories as matched gradient-budget and time panels.

    This is intentionally a visual restyling of the existing three-method
    comparison, not an r/N sweep: every line retains the checkpoints actually
    logged by its corresponding run.
    """
    series: List[Tuple[str, np.ndarray, np.ndarray, bool]] = []
    for method, method_rows in group_rows_by_method(adaptive_gn_rows).items():
        calls_x, y = gn_series(
            method_rows,
            use_elapsed_time=False,
            adaptive=True,
            gn_metric=gn_metric,
        )
        time_x, time_y = gn_series(
            method_rows,
            use_elapsed_time=True,
            adaptive=True,
            gn_metric=gn_metric,
        )
        if relative_time:
            time_x = relative_elapsed_axis(time_x, log_x=False)
        if len(calls_x) > 0 and len(time_x) > 0:
            # Both calls use the same checkpoint rows, so y and time_y agree.
            series.append((method, calls_x, y, False))
            series.append((method, time_x, time_y, True))

    uniform_calls_x, uniform_y = gn_series(
        uniform_gn_rows,
        use_elapsed_time=False,
        adaptive=False,
        gn_metric=gn_metric,
    )
    uniform_time_x, uniform_time_y = gn_series(
        uniform_gn_rows,
        use_elapsed_time=True,
        adaptive=False,
        gn_metric=gn_metric,
    )
    if relative_time:
        uniform_time_x = relative_elapsed_axis(uniform_time_x, log_x=False)
    if len(uniform_calls_x) > 0 and len(uniform_time_x) > 0:
        series.append(("Uniform DPO-LW", uniform_calls_x, uniform_y, False))
        series.append(("Uniform DPO-LW", uniform_time_x, uniform_time_y, True))

    if not series:
        return None

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.2),
        dpi=240,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    handles = []
    labels = []
    for panel_idx, ax in enumerate(axes):
        for method, x, y, is_time in series:
            if is_time != bool(panel_idx):
                continue
            style = publication_gn_style(method, uniform=method == "Uniform DPO-LW")
            (line,) = ax.plot(
                x,
                y,
                linewidth=2.0 if method == "Uniform DPO-LW" else 2.3,
                markeredgecolor="white" if style["marker"] else None,
                markeredgewidth=0.65 if style["marker"] else 0.0,
                **style,
                label=method,
            )
            if panel_idx == 0 and method not in labels:
                handles.append(line)
                labels.append(method)

        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.28, linewidth=0.7)
        ax.grid(False, which="minor")
        ax.tick_params(axis="both", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(
            "Objective gradient evaluations"
            if panel_idx == 0
            else (
                "Relative wall-clock time (s)"
                if relative_time
                else "Wall-clock time (s)"
            ),
            fontsize=11,
        )

    axes[0].set_ylabel(gn_metric_ylabel(gn_metric), fontsize=11)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=2.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = output_dir / "gn_publication_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def epsilon_value_from_run_name(name: str) -> Optional[float]:
    match = re.search(r"eps_([^_]+)", name)
    if match is None:
        return None
    token = match.group(1)
    try:
        if "em" in token:
            base, exponent = token.split("em", 1)
            return float(base.replace("p", ".")) * (10 ** (-int(exponent)))
        return float(token.replace("p", "."))
    except ValueError:
        return None


def collect_latest_epsilon_runs(root: Path, run_glob: str) -> List[Tuple[float, Path]]:
    latest_by_epsilon: Dict[float, Path] = {}
    for history_path in root.glob(f"{run_glob}/adaptive_history.jsonl"):
        run_dir = history_path.parent
        epsilon = epsilon_value_from_run_name(run_dir.name)
        if epsilon is None:
            continue
        previous = latest_by_epsilon.get(epsilon)
        if previous is None or run_dir.stat().st_mtime > previous.stat().st_mtime:
            latest_by_epsilon[epsilon] = run_dir
    return sorted(latest_by_epsilon.items(), key=lambda item: item[0])


def final_update_count(gn_rows: Sequence[Dict]) -> int:
    if not gn_rows:
        return 0
    last = gn_rows[-1]
    return int(first_present(
        last.get("parameter_updates_after"),
        last.get("parameter_updates"),
        0,
    ))


def plot_epsilon_bundle_sweep(
    adaptive_root: Path,
    output_dir: Path,
    *,
    run_glob: str,
    gn_metric: str = "norm",
) -> List[Path]:
    epsilon_runs = collect_latest_epsilon_runs(adaptive_root, run_glob)
    if not epsilon_runs:
        raise SystemExit(
            f"No epsilon adaptive runs found under {adaptive_root} with glob {run_glob!r}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    loaded: List[Tuple[float, Path, List[Dict]]] = []
    for epsilon, run_dir in epsilon_runs:
        _, _, gn_rows = load_adaptive_run(run_dir)
        if gn_rows:
            loaded.append((epsilon, run_dir, gn_rows))
    if not loaded:
        raise SystemExit("Found epsilon runs, but none had usable adaptive GN* rows.")

    summary_path = output_dir / "epsilon_sweep_bundle_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epsilon",
            "adaptive_run",
            "updates",
            "objective_gradient_evals",
            "best_gradient_norm" if gn_metric == "norm" else "best_gn_star",
        ])
        for epsilon, run_dir, gn_rows in loaded:
            _, y_eval = gn_series(
                gn_rows,
                use_elapsed_time=False,
                adaptive=True,
                gn_metric=gn_metric,
            )
            updates = final_update_count(gn_rows)
            writer.writerow([
                epsilon,
                run_dir.name,
                updates,
                2 * updates,
                float(y_eval[-1]) if len(y_eval) else "",
            ])

    def _plot(*, use_elapsed_time: bool, log_x: bool, filename: str) -> Optional[Path]:
        fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=180)
        setup_axes(ax, f"Adaptive bundle epsilon sweep ({gn_metric_ylabel(gn_metric)})")
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(loaded))))

        plotted = False
        for (epsilon, _run_dir, gn_rows), color in zip(loaded, colors):
            x, y = gn_series(
                gn_rows,
                use_elapsed_time=use_elapsed_time,
                adaptive=True,
                gn_metric=gn_metric,
            )
            if use_elapsed_time:
                x = relative_elapsed_axis(x, log_x=log_x)
            if log_x:
                mask = x > 0
                x = x[mask]
                y = y[mask]
            if len(x) == 0:
                continue
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.9,
                markersize=4.8,
                color=color,
                label=f"epsilon={epsilon:g}",
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return None

        if log_x:
            ax.set_xscale("log")
            ax.set_xlabel("Relative elapsed wall time (seconds, log scale)")
        elif use_elapsed_time:
            ax.set_xlabel("Relative elapsed wall time (seconds)")
        else:
            ax.set_xlabel("Objective gradient evaluations (= parameter updates * K)")
        ax.set_ylabel(gn_metric_ylabel(gn_metric))
        ax.legend(frameon=False)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        return path

    saved_paths: List[Path] = [summary_path]
    for path in [
        _plot(
            use_elapsed_time=False,
            log_x=False,
            filename="epsilon_sweep_bundle_by_gradient_evals.png",
        ),
        _plot(
            use_elapsed_time=True,
            log_x=True,
            filename="epsilon_sweep_bundle_by_log_time.png",
        ),
    ]:
        if path is not None:
            saved_paths.append(path)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Pareto and training traces from adaptive bundle and DPO-LW logs."
    )
    parser.add_argument("--dpo_lw_dir", type=Path, default=Path(DEFAULT_DPO_LW_DIR))
    parser.add_argument("--adaptive_dir", type=Path, default=Path(DEFAULT_ADAPTIVE_DIR))
    parser.add_argument(
        "--adaptive_run",
        nargs=2,
        action="append",
        metavar=("DIR", "LABEL"),
        default=None,
        help=(
            "Adaptive output directory and plot label. Can be repeated to compare "
            "multiple adaptive solvers. When set, --adaptive_dir is ignored."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--epsilon_adaptive_root",
        type=Path,
        default=None,
        help=(
            "If set, scan epsilon adaptive bundle runs under this root and plot "
            "bundle-only epsilon sweep curves."
        ),
    )
    parser.add_argument(
        "--epsilon_run_glob",
        type=str,
        default="eps_*outer*_inner*",
        help="Directory glob, relative to --epsilon_adaptive_root, for epsilon runs.",
    )
    parser.add_argument(
        "--tail_window",
        type=int,
        default=20,
        help="Average the last N DPO-LW log records for each Pareto point.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=10,
        help="Moving-average window for DPO-LW training curves.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate Pareto points with lambda labels and trajectory points with gradient evals.",
    )
    parser.add_argument(
        "--lambda_round_decimals",
        type=int,
        default=4,
        help="Round lambdas to this many decimals when grouping candidates.",
    )
    parser.add_argument(
        "--absolute_elapsed_time",
        action="store_true",
        help=(
            "Plot elapsed-time GN* curves on the raw wall-clock axis instead of "
            "shifting each method to start at zero."
        ),
    )
    parser.add_argument(
        "--publication_gn",
        action="store_true",
        help=(
            "Additionally save a two-panel, publication-style GN comparison "
            "(gradient evaluations and elapsed time)."
        ),
    )
    parser.add_argument(
        "--publication_pareto",
        action="store_true",
        help=(
            "Additionally save a compact, publication-style DPO-loss Pareto figure "
            "with muted checkpoint clouds and prominent nondominated frontiers."
        ),
    )
    parser.add_argument(
        "--gn_metric",
        choices=("norm", "squared"),
        default="norm",
        help=(
            "Scale used for GN plots. Logs store squared gradient norms; "
            "`norm` plots sqrt(GN*) and `squared` reproduces the old plots."
        ),
    )
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.epsilon_adaptive_root is not None:
        saved_paths = plot_epsilon_bundle_sweep(
            args.epsilon_adaptive_root,
            args.output_dir,
            run_glob=args.epsilon_run_glob,
            gn_metric=args.gn_metric,
        )
        print("Saved epsilon bundle sweep plots:")
        for path in saved_paths:
            print(f"  {path}")
        return

    logged_dpo_points, dpo_curves = load_dpo_lw_runs(args.dpo_lw_dir, args.tail_window)
    uniform_oracle_points, uniform_gn_rows = load_uniform_gn_run(args.dpo_lw_dir)
    dpo_points = uniform_oracle_points if uniform_oracle_points else logged_dpo_points
    if args.adaptive_run:
        adaptive_specs = [
            (Path(run_dir), label)
            for run_dir, label in args.adaptive_run
        ]
    else:
        adaptive_specs = [(args.adaptive_dir, "Adaptive bundle")]

    adaptive_points: List[ParetoPoint] = []
    lambda_rows: List[Dict] = []
    adaptive_gn_rows: List[Dict] = []
    loaded_adaptive_labels: List[str] = []
    for adaptive_dir, label in adaptive_specs:
        run_points, run_lambda_rows, run_gn_rows = load_adaptive_run(adaptive_dir, label)
        adaptive_points.extend(run_points)
        lambda_rows.extend(run_lambda_rows)
        adaptive_gn_rows.extend(run_gn_rows)
        if run_points or run_lambda_rows or run_gn_rows:
            loaded_adaptive_labels.append(label)
    all_points = [*dpo_points, *adaptive_points]

    saved_paths: List[Path] = []
    summary_path = save_summary(all_points, args.output_dir)
    saved_paths.append(summary_path)
    representatives_path = save_representatives(
        dpo_points,
        adaptive_points,
        args.output_dir,
        args.lambda_round_decimals,
    )
    saved_paths.append(representatives_path)
    frontiers_path = save_frontiers(
        dpo_points,
        adaptive_points,
        args.output_dir,
        args.lambda_round_decimals,
    )
    saved_paths.append(frontiers_path)

    for path in [
        plot_pareto_front(
            dpo_points,
            adaptive_points,
            args.output_dir,
            args.title,
            args.annotate,
            args.lambda_round_decimals,
        ),
        *(
            [
                plot_publication_pareto_front(
                    dpo_points,
                    adaptive_points,
                    args.output_dir,
                    annotate=args.annotate,
                    lambda_round_decimals=args.lambda_round_decimals,
                )
            ]
            if args.publication_pareto
            else []
        ),
        plot_adaptive_lambda_trajectories(
            adaptive_points,
            args.output_dir,
            args.lambda_round_decimals,
            args.annotate,
        ),
        plot_uniform_lambda_trajectories(
            dpo_curves,
            args.output_dir,
            args.smooth_window,
            args.annotate,
        ),
        plot_lambda_path(lambda_rows, args.output_dir),
        plot_dpo_training_curves(dpo_curves, args.output_dir, args.smooth_window),
        plot_adaptive_trace(
            adaptive_points,
            adaptive_gn_rows,
            args.output_dir,
            gn_metric=args.gn_metric,
        ),
        plot_gn_star_comparison(
            adaptive_gn_rows,
            uniform_gn_rows,
            args.output_dir,
            gn_metric=args.gn_metric,
        ),
        plot_gn_star_comparison(
            adaptive_gn_rows,
            uniform_gn_rows,
            args.output_dir,
            use_elapsed_time=True,
            relative_time=not args.absolute_elapsed_time,
            gn_metric=args.gn_metric,
        ),
        plot_gn_star_comparison(
            adaptive_gn_rows,
            uniform_gn_rows,
            args.output_dir,
            use_elapsed_time=True,
            log_x=True,
            relative_time=not args.absolute_elapsed_time,
            gn_metric=args.gn_metric,
        ),
        *(
            [
                plot_publication_gn_comparison(
                    adaptive_gn_rows,
                    uniform_gn_rows,
                    args.output_dir,
                    relative_time=not args.absolute_elapsed_time,
                    gn_metric=args.gn_metric,
                )
            ]
            if args.publication_gn
            else []
        ),
    ]:
        if path is not None:
            saved_paths.append(path)

    if not all_points:
        adaptive_hint = (
            ", ".join(f"{label}={path}" for path, label in adaptive_specs)
            if adaptive_specs
            else str(args.adaptive_dir)
        )
        print(
            "No Pareto points found. Expected DPO-LW logs under "
            f"{args.dpo_lw_dir} or adaptive logs under {adaptive_hint}."
        )
    else:
        dpo_source = "fixed-oracle" if uniform_oracle_points else "logged-tail"
        adaptive_label_text = (
            ", ".join(loaded_adaptive_labels)
            if loaded_adaptive_labels
            else "none"
        )
        print(
            f"Loaded {len(dpo_points)} DPO-LW points ({dpo_source}) "
            f"and {len(adaptive_points)} adaptive points ({adaptive_label_text})."
        )

    print("Saved:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
