from __future__ import annotations

import argparse
import csv
import json
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
                "gradient_eval": oracle_gradient_eval,
                "oracle_gradient_eval": oracle_gradient_eval,
                "checkpoint_index": checkpoint_index,
                "parameter_updates": parameter_updates,
                "objective_gradient_evals": objective_gradient_evals,
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
            method="Adaptive bundle",
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
) -> Tuple[List[ParetoPoint], List[Dict], List[Dict]]:
    if adaptive_dir is None or not adaptive_dir.exists():
        return [], [], []

    history_path = adaptive_dir / "adaptive_history.jsonl"
    records = read_jsonl(history_path)
    if not records:
        return [], [], []

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
                "outer": outer,
                "gradient_eval": gradient_evals_before,
                "oracle_gradient_eval": gradient_evals_before,
                "parameter_updates": parameter_updates_before,
                "objective_gradient_evals": objective_gradient_evals_before,
                "gn_star": record.get("gn_star"),
                "bundle_size": record.get("bundle_size"),
            }
        )

        if "fvals" in record:
            add_adaptive_point(
                points,
                record["fvals"],
                outer,
                f"outer_{outer}",
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
    best: Dict[Tuple[float, float], ParetoPoint] = {}
    for point in points:
        key = lambda_key(point, decimals)
        if key is None:
            continue
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
        ax.plot(
            [point.helpful_loss for point in frontier],
            [point.harmless_loss for point in frontier],
            color="#4C78A8",
            alpha=0.85,
            linewidth=1.8,
            label="Uniform DPO-LW frontier",
            zorder=2,
        )
        if annotate:
            for point in frontier:
                if point.lambda_helpful is None:
                    continue
                ax.annotate(
                    f"{point.lambda_helpful:.1f}",
                    (point.helpful_loss, point.harmless_loss),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

    if adaptive_representatives:
        scatter_lambda_points(
            ax,
            adaptive_representatives,
            marker="o",
            label="Adaptive bundle best per lambda",
            alpha=0.9,
            size=60,
            cmap=cmap,
            norm=norm,
            zorder=4,
        )
        frontier = nondominated_points(adaptive_representatives)
        ax.plot(
            [point.helpful_loss for point in frontier],
            [point.harmless_loss for point in frontier],
            color="#F58518",
            alpha=0.85,
            linewidth=1.8,
            label="Adaptive bundle frontier",
            zorder=2,
        )
        if annotate:
            for point in frontier:
                if point.lambda_helpful is None:
                    continue
                ax.annotate(
                    f"{point.lambda_helpful:.1f}",
                    (point.helpful_loss, point.harmless_loss),
                    textcoords="offset points",
                    xytext=(4, -9),
                    fontsize=8,
                    color="#B85C00",
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


def group_points_by_lambda(
    points: Sequence[ParetoPoint],
    decimals: int,
) -> Dict[Tuple[float, float], List[ParetoPoint]]:
    groups: Dict[Tuple[float, float], List[ParetoPoint]] = {}
    for point in points:
        key = lambda_key(point, decimals)
        if key is None:
            continue
        groups.setdefault(key, []).append(point)
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

    for (lambda_helpful, lambda_harmless), group in sorted(groups.items()):
        color = cmap(norm(lambda_helpful))
        xs = [point.helpful_loss for point in group]
        ys = [point.harmless_loss for point in group]
        label = f"lambda=({lambda_helpful:g},{lambda_harmless:g})"
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
        label = f"lambda=({lambda_helpful:.1f},{lambda_harmless:.1f})"
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
                f"{lambda_helpful:.1f}",
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

    rows = sorted(lambda_rows, key=lambda row: row["outer"])
    outer = np.asarray([row["outer"] for row in rows], dtype=np.int64)
    helpful = np.asarray([row["lambda_helpful"] for row in rows], dtype=np.float64)
    harmless = np.asarray([row["lambda_harmless"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    setup_axes(ax, "Adaptive bundle lambda path")
    ax.plot(outer, helpful, marker="o", linewidth=1.8, label="lambda_helpful")
    ax.plot(outer, harmless, marker="s", linewidth=1.8, label="lambda_harmless")
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
            else f"lambda_helpful={lambda_helpful:.1f}"
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
) -> Optional[Path]:
    if not adaptive_points and not gn_rows:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.4), dpi=180, sharex=False)

    if adaptive_points:
        ordered = sorted(
            adaptive_points,
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
            label="helpful_loss",
        )
        axes[0].plot(
            x,
            [point.harmless_loss for point in ordered],
            marker="s",
            linewidth=1.6,
            label="harmless_loss",
        )
        axes[0].set_xlabel("Objective gradient evaluations")
        axes[0].set_ylabel("DPO loss")
        axes[0].legend(frameon=False)
    setup_axes(axes[0], "Adaptive bundle objective trace")

    usable_gn = [
        row
        for row in gn_rows
        if row.get("gn_star") is not None and row.get("outer") is not None
    ]
    if usable_gn:
        parameter_updates = [budget_x_from_row(row) for row in usable_gn]
        gn_star = [float(row["gn_star"]) for row in usable_gn]
        axes[1].plot(parameter_updates, gn_star, marker="o", linewidth=1.6, color="#E45756")
        axes[1].set_xlabel("Objective gradient evaluations")
        axes[1].set_ylabel("GN*")
    setup_axes(axes[1], "GN* over objective gradient evaluations")

    fig.tight_layout()
    path = output_dir / "adaptive_training_trace.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gn_star_comparison(
    adaptive_gn_rows: Sequence[Dict],
    uniform_gn_rows: Sequence[Dict],
    output_dir: Path,
) -> Optional[Path]:
    adaptive = [
        row
        for row in adaptive_gn_rows
        if row.get("gn_star") is not None
    ]
    uniform = [
        row
        for row in uniform_gn_rows
        if row.get("gn_star") is not None
    ]
    if not adaptive and not uniform:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    setup_axes(ax, "GN* comparison at matched gradient-eval budget")
    ax.set_xlabel("Objective gradient evaluations (= parameter updates * K)")
    ax.set_ylabel("GN*")

    if adaptive:
        adaptive = sorted(
            adaptive,
            key=budget_x_from_row,
        )
        ax.plot(
            [budget_x_from_row(row, idx + 1) for idx, row in enumerate(adaptive)],
            [float(row["gn_star"]) for row in adaptive],
            marker="o",
            linewidth=1.8,
            color="#F58518",
            label="Adaptive bundle",
        )

    if uniform:
        uniform = sorted(uniform, key=budget_x_from_row)
        ax.plot(
            [budget_x_from_row(row) for row in uniform],
            [float(row["gn_star"]) for row in uniform],
            marker="s",
            linewidth=1.8,
            color="#4C78A8",
            label="Uniform DPO-LW",
        )

    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "gn_star_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Pareto and training traces from adaptive bundle and DPO-LW logs."
    )
    parser.add_argument("--dpo_lw_dir", type=Path, default=Path(DEFAULT_DPO_LW_DIR))
    parser.add_argument("--adaptive_dir", type=Path, default=Path(DEFAULT_ADAPTIVE_DIR))
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
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
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logged_dpo_points, dpo_curves = load_dpo_lw_runs(args.dpo_lw_dir, args.tail_window)
    uniform_oracle_points, uniform_gn_rows = load_uniform_gn_run(args.dpo_lw_dir)
    dpo_points = uniform_oracle_points if uniform_oracle_points else logged_dpo_points
    adaptive_points, lambda_rows, adaptive_gn_rows = load_adaptive_run(args.adaptive_dir)
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

    for path in [
        plot_pareto_front(
            dpo_points,
            adaptive_points,
            args.output_dir,
            args.title,
            args.annotate,
            args.lambda_round_decimals,
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
        plot_adaptive_trace(adaptive_points, adaptive_gn_rows, args.output_dir),
        plot_gn_star_comparison(adaptive_gn_rows, uniform_gn_rows, args.output_dir),
    ]:
        if path is not None:
            saved_paths.append(path)

    if not all_points:
        print(
            "No Pareto points found. Expected DPO-LW logs under "
            f"{args.dpo_lw_dir} or adaptive logs under {args.adaptive_dir}."
        )
    else:
        dpo_source = "fixed-oracle" if uniform_oracle_points else "logged-tail"
        print(
            f"Loaded {len(dpo_points)} DPO-LW points ({dpo_source}) "
            f"and {len(adaptive_points)} adaptive points."
        )

    print("Saved:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
