from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tyro
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, get_scheduler

from scripts.modpo.adaptive_bundle.bundle_core import (
    FirstOrderBundle,
    LAMBDA_SOLVERS,
    ipopt_available,
    ipopt_import_error,
    maximise_gn,
)
from scripts.modpo.adaptive_bundle.dpo_lw import (
    CyclingLoader,
    ScriptArguments as DpoLwArguments,
    backward_weighted_dpo_update,
    build_trainer,
    format_weight,
    make_fixed_batches,
    parse_float_list,
    prepare_datasets,
    save_jsonl,
    select_subset,
)
from src.trainer.modpo_trainer import MODPODataCollatorWithPadding
from src.utils import disable_progress_bar_non_local_main, print_local_main, set_seeds

disable_progress_bar_non_local_main()


@dataclass
class ScriptArguments(DpoLwArguments):
    """SURF baseline using this repo's DPO-LW training/data/eval stack.

    SURF controls only the scalarization weights. Each slot keeps its own LoRA
    parameter vector and AdamW optimizer state, so outer t+1 starts slot n from
    the checkpoint produced by slot n at outer t.
    """

    surf_num_segments: Optional[int] = field(
        default=5,
        metadata={"help": "SURF uses surf_num_segments+1 scalarization slots."},
    )
    surf_max_outer: Optional[int] = field(default=7)
    surf_steps_per_slot_per_outer: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Parameter updates per slot per SURF outer iteration. Defaults "
                "to --max_steps so existing DPO-LW configs can be reused."
            )
        },
    )
    surf_alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "CDF refinement blend: F_next=(1-alpha)F+alpha*F_tilde."},
    )
    surf_cdf_grid_size: Optional[int] = field(default=2001)
    surf_use_pchip: Optional[bool] = field(
        default=True,
        metadata={"help": "Use monotone PCHIP interpolation for the arclength surrogate CDF when scipy is available."},
    )
    surf_monotone_eps: Optional[float] = field(default=1e-8)
    surf_force_endpoints: Optional[bool] = field(default=True)
    surf_save_every_outer: Optional[int] = field(
        default=1,
        metadata={"help": "Save every N SURF outers; the final outer is always saved. Use 0 to save final only."},
    )
    surf_warm_start_strategy: Optional[str] = field(
        default="same_slot",
        metadata={"help": "Currently supports 'same_slot' and 'from_sft_each_outer'."},
    )
    lambda_solver: Optional[str] = field(default="exact_k2")
    require_ipopt: Optional[bool] = field(default=False)

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/surf_dpo",
            overwrite_output_dir=True,
            seed=42,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-4,
            bf16=torch.cuda.is_available(),
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
            logging_strategy="no",
            save_strategy="no",
        )
    )


def make_uniform_cdf_grid(grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if grid_size < 2:
        raise ValueError("surf_cdf_grid_size must be at least 2.")
    grid = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
    return grid, grid.copy()


def invert_cdf(F: np.ndarray, grid: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    if F.shape != grid.shape:
        raise ValueError("CDF values and grid must have the same shape.")
    return np.interp(np.clip(quantiles, 0.0, 1.0), F, grid)


def enforce_monotone_cdf(
    F: np.ndarray,
    *,
    eps: float,
    force_endpoints: bool,
) -> np.ndarray:
    cdf = np.asarray(F, dtype=np.float64).copy()
    if force_endpoints:
        cdf[0] = 0.0
        cdf[-1] = 1.0
    for idx in range(1, len(cdf)):
        if cdf[idx] < cdf[idx - 1] + eps:
            cdf[idx] = cdf[idx - 1] + eps
    if force_endpoints:
        cdf[0] = 0.0
        cdf[-1] = 1.0
    max_value = float(np.max(cdf))
    if max_value > 1.0 + 1e-12:
        cdf = cdf / max_value
    return np.clip(cdf, 0.0, 1.0)


def segment_lengths(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("SURF currently expects two objective coordinates.")
    if points.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def build_surrogate_cdf_from_points(
    weights: np.ndarray,
    objective_points: np.ndarray,
    grid: np.ndarray,
    *,
    use_pchip: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    objective_points = np.asarray(objective_points, dtype=np.float64)
    if weights.shape[0] != objective_points.shape[0]:
        raise ValueError("weights and objective_points must have the same length.")

    order = np.argsort(weights)
    sorted_weights = weights[order]
    sorted_points = objective_points[order]
    lengths = segment_lengths(sorted_points)

    arclength_at_weight = np.zeros(len(sorted_weights), dtype=np.float64)
    if len(lengths):
        arclength_at_weight[1:] = np.cumsum(lengths)
    total_length = float(arclength_at_weight[-1]) if len(arclength_at_weight) else 0.0
    if total_length <= 0.0:
        cdf_at_weight = np.linspace(0.0, 1.0, len(sorted_weights), dtype=np.float64)
    else:
        cdf_at_weight = arclength_at_weight / total_length

    if use_pchip:
        try:
            from scipy.interpolate import PchipInterpolator

            interpolator = PchipInterpolator(sorted_weights, cdf_at_weight, extrapolate=False)
            cdf_grid = np.asarray(interpolator(grid), dtype=np.float64)
            missing = np.isnan(cdf_grid)
            if np.any(missing):
                cdf_grid[missing] = np.interp(grid[missing], sorted_weights, cdf_at_weight)
        except ImportError:
            cdf_grid = np.interp(grid, sorted_weights, cdf_at_weight)
    else:
        cdf_grid = np.interp(grid, sorted_weights, cdf_at_weight)

    return np.clip(cdf_grid, 0.0, 1.0), arclength_at_weight, lengths


def blend_cdfs(F_prev: np.ndarray, F_tilde: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("surf_alpha must be in (0, 1].")
    return (1.0 - alpha) * np.asarray(F_prev, dtype=np.float64) + alpha * np.asarray(F_tilde, dtype=np.float64)


def coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return float("nan")
    mean = float(np.mean(values))
    if mean <= 0.0:
        return float("nan")
    return float(np.std(values, ddof=0) / mean)


def gap_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return float("nan")
    smallest = float(np.min(values))
    if smallest <= 0.0:
        return float("inf")
    return float(np.max(values) / smallest)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: str | Path, payload) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def surf_slot_run_name(surf_outer: int, slot_index: int, helpful_weight: float, harmless_weight: float) -> str:
    return (
        f"surf_outer_{surf_outer:03d}_slot_{slot_index:03d}"
        f"_lambda_helpful_{format_weight(helpful_weight)}"
        f"_harmless_{format_weight(harmless_weight)}"
    )


def should_save_outer(args: ScriptArguments, surf_outer: int) -> bool:
    if surf_outer == int(args.surf_max_outer):
        return True
    interval = int(args.surf_save_every_outer or 0)
    return interval > 0 and surf_outer % interval == 0


def validate_args(args: ScriptArguments) -> None:
    if args.update_data_source not in {"train", "oracle"}:
        raise ValueError("update_data_source must be either 'train' or 'oracle'.")
    if args.surf_warm_start_strategy not in {"same_slot", "from_sft_each_outer"}:
        raise ValueError("surf_warm_start_strategy must be 'same_slot' or 'from_sft_each_outer'.")
    if int(args.surf_num_segments or 0) < 1:
        raise ValueError("surf_num_segments must be at least 1.")
    if int(args.surf_max_outer or 0) < 1:
        raise ValueError("surf_max_outer must be at least 1.")
    if args.surf_steps_per_slot_per_outer is None:
        args.surf_steps_per_slot_per_outer = int(args.max_steps)
    if int(args.surf_steps_per_slot_per_outer) < 0:
        raise ValueError("surf_steps_per_slot_per_outer must be non-negative.")
    if args.lambda_solver not in LAMBDA_SOLVERS:
        raise ValueError(
            "lambda_solver must be one of: "
            + ", ".join(sorted(LAMBDA_SOLVERS))
            + "."
        )
    if args.require_ipopt and args.lambda_solver == "ipopt" and not ipopt_available():
        raise RuntimeError(
            "IPOPT was required for SURF GN evaluation, but cyipopt/IPOPT is unavailable. "
            f"Import error: {ipopt_import_error()!r}"
        )
    if args.bundle_dtype not in {"float32", "float64"}:
        raise ValueError("bundle_dtype must be either 'float32' or 'float64'.")
    if args.gn_target_norm is not None and float(args.gn_target_norm) <= 0.0:
        raise ValueError("gn_target_norm must be positive when provided.")


def save_slot_config(
    args: ScriptArguments,
    run_dir: str,
    *,
    surf_outer: int,
    slot_index: int,
    helpful_weight: float,
    harmless_weight: float,
) -> None:
    write_json(
        os.path.join(run_dir, "dpo_lw_config.json"),
        {
            **asdict(args),
            "method": "SURF",
            "surf_outer": surf_outer,
            "slot_index": slot_index,
            "lambda_helpful": helpful_weight,
            "lambda_harmless": harmless_weight,
            "run_max_steps": int(args.surf_steps_per_slot_per_outer),
            "warm_start_strategy": args.surf_warm_start_strategy,
        },
    )


def main() -> None:
    run_start_time = time.perf_counter()
    args = tyro.cli(ScriptArguments)
    validate_args(args)
    set_seeds(args.seed)

    output_dir = args.training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for filename in [
        "adaptive_history.jsonl",
        "surf_history.jsonl",
        "surf_weight_history.jsonl",
        "surf_metric_history.jsonl",
        "surf_cdf_history.jsonl",
    ]:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path) and args.training_args.overwrite_output_dir:
            os.remove(path)

    config_payload = {
        **asdict(args),
        "method": "SURF",
        "algorithm": "CDF arclength refinement with same-slot warm start",
    }
    write_json(os.path.join(output_dir, "surf_config.json"), config_payload)
    write_json(os.path.join(output_dir, "adaptive_config.json"), config_payload)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print_local_main("preparing SURF DPO-LW datasets...")
    helpful_train, harmless_train, helpful_eval = prepare_datasets(args, tokenizer)
    data_collator = MODPODataCollatorWithPadding(tokenizer)

    smoothness = parse_float_list(args.smoothness)
    if len(smoothness) != 2:
        raise ValueError("SURF GN comparison expects exactly two smoothness constants.")
    num_objectives = len(smoothness)

    helpful_oracle = select_subset(
        helpful_train,
        args.oracle_subset_size_per_objective,
        args.seed + 2,
        "helpful fixed oracle for SURF",
    )
    harmless_oracle = select_subset(
        harmless_train,
        args.oracle_subset_size_per_objective,
        args.seed + 2 if args.agreement_ratio is not None else args.seed + 3,
        "harmless fixed oracle for SURF",
    )
    objective_batch_groups = {
        "helpful": make_fixed_batches(helpful_oracle, args.oracle_batch_size, data_collator),
        "harmless": make_fixed_batches(harmless_oracle, args.oracle_batch_size, data_collator),
    }

    print_local_main(
        "running SURF DPO-LW baseline: "
        f"slots={int(args.surf_num_segments) + 1}, "
        f"outer={args.surf_max_outer}, "
        f"steps/slot/outer={args.surf_steps_per_slot_per_outer}, "
        f"warm_start={args.surf_warm_start_strategy}, "
        f"update_data={args.update_data_source}"
    )

    trainer = build_trainer(args, tokenizer, data_collator, helpful_train, helpful_eval)
    if Accelerator().is_local_main_process and args.peft_config:
        trainer.model.print_trainable_parameters()

    helpful_loader = CyclingLoader(DataLoader(
        helpful_train,
        batch_size=args.per_objective_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator,
    ))
    harmless_loader = CyclingLoader(DataLoader(
        harmless_train,
        batch_size=args.per_objective_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator,
    ))

    trainable_params = [param for param in trainer.model.parameters() if param.requires_grad]
    initial_vector = trainer.get_trainable_parameter_vector(cpu=True).numpy()
    slot_count = int(args.surf_num_segments) + 1
    steps_per_slot_outer = int(args.surf_steps_per_slot_per_outer)
    target_steps_per_slot = int(args.surf_max_outer) * steps_per_slot_outer

    solution_vectors = [initial_vector.copy() for _ in range(slot_count)]
    slot_update_counts = [0 for _ in range(slot_count)]
    optimizers = [
        torch.optim.AdamW(
            trainable_params,
            lr=args.training_args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for _ in range(slot_count)
    ]
    schedulers = []
    for optimizer in optimizers:
        scheduler_steps = max(1, target_steps_per_slot)
        warmup_steps = int(target_steps_per_slot * args.warmup_ratio)
        schedulers.append(
            get_scheduler(
                args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=scheduler_steps,
            )
        )

    trainer.set_trainable_parameter_vector(initial_vector)
    initial_oracle = trainer.multi_objective_gradient_oracle_over_batches(
        objective_batch_groups,
        as_numpy=True,
    )
    gn_bundle = FirstOrderBundle(
        K=num_objectives,
        d=int(initial_oracle["x"].shape[0]),
        L=np.asarray(smoothness, dtype=np.float64),
        dtype=np.dtype(args.bundle_dtype),
    )
    for _ in range(slot_count):
        gn_bundle.add(
            initial_oracle["x"],
            initial_oracle["fvals"],
            initial_oracle["grads"],
        )
    write_json(
        os.path.join(output_dir, "adaptive_initial_oracle.json"),
        {
            "method": "SURF",
            "fvals": initial_oracle["fvals"].tolist(),
            "objective_names": list(initial_oracle.get("objective_names", ["helpful", "harmless"])),
            "parameter_updates": 0,
            "objective_gradient_evals": 0,
            "oracle_gradient_eval": 1,
            "elapsed_wall_seconds": time.perf_counter() - run_start_time,
        },
    )

    cdf_grid, current_cdf = make_uniform_cdf_grid(int(args.surf_cdf_grid_size))
    quantiles = np.linspace(0.0, 1.0, slot_count, dtype=np.float64)
    cumulative_parameter_updates = 0
    checkpoint_counter = 0
    oracle_gradient_evals = 1
    best_gn_star = float("inf")
    target_reached = False
    target_hit_record = None
    weights_history: List[Dict] = []
    metrics_history: List[Dict] = []
    cdf_history: List[Dict] = [
        {
            "surf_outer": 0,
            "cdf_grid": cdf_grid.tolist(),
            "cdf_values": current_cdf.tolist(),
        }
    ]
    pf_history: List[Dict] = []

    total_updates = int(args.surf_max_outer) * slot_count * steps_per_slot_outer
    progress = tqdm(total=total_updates, disable=not Accelerator().is_local_main_process)

    try:
        for surf_outer in range(1, int(args.surf_max_outer) + 1):
            helpful_weights = invert_cdf(current_cdf, cdf_grid, quantiles)
            if args.surf_force_endpoints:
                helpful_weights[0] = 0.0
                helpful_weights[-1] = 1.0
            helpful_weights = np.clip(helpful_weights, 0.0, 1.0)
            weight_pairs = [(float(w), float(1.0 - w)) for w in helpful_weights]

            weight_record = {
                "method": "SURF",
                "surf_outer": surf_outer,
                "quantiles": quantiles.tolist(),
                "lambda_helpful": helpful_weights.tolist(),
                "lambda_pairs": [[h, hh] for h, hh in weight_pairs],
                "elapsed_wall_seconds": time.perf_counter() - run_start_time,
            }
            weights_history.append(weight_record)
            save_jsonl(os.path.join(output_dir, "surf_weight_history.jsonl"), weight_record)

            outer_records = []
            objective_points = []
            outer_dir = os.path.join(output_dir, f"surf_outer_{surf_outer:03d}")
            os.makedirs(outer_dir, exist_ok=True)
            save_this_outer = should_save_outer(args, surf_outer)

            for slot_index, (helpful_weight, harmless_weight) in enumerate(weight_pairs):
                slot_t0 = time.perf_counter()
                if args.surf_warm_start_strategy == "from_sft_each_outer":
                    trainer.set_trainable_parameter_vector(initial_vector)
                else:
                    trainer.set_trainable_parameter_vector(solution_vectors[slot_index])

                optimizer = optimizers[slot_index]
                scheduler = schedulers[slot_index]
                run_name = surf_slot_run_name(
                    surf_outer,
                    slot_index,
                    helpful_weight,
                    harmless_weight,
                )
                run_dir = os.path.join(outer_dir, run_name)
                os.makedirs(run_dir, exist_ok=True)
                save_slot_config(
                    args,
                    run_dir,
                    surf_outer=surf_outer,
                    slot_index=slot_index,
                    helpful_weight=helpful_weight,
                    harmless_weight=harmless_weight,
                )
                history_path = os.path.join(run_dir, "training_history.jsonl")
                if os.path.exists(history_path) and args.training_args.overwrite_output_dir:
                    os.remove(history_path)

                for local_step in range(1, steps_per_slot_outer + 1):
                    optimizer.zero_grad(set_to_none=True)
                    update_stats = backward_weighted_dpo_update(
                        trainer=trainer,
                        helpful_weight=helpful_weight,
                        harmless_weight=harmless_weight,
                        update_data_source=args.update_data_source,
                        helpful_loader=helpful_loader,
                        harmless_loader=harmless_loader,
                        objective_batch_groups=objective_batch_groups,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                    )
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    slot_update_counts[slot_index] += 1
                    cumulative_parameter_updates += 1
                    train_record = {
                        "method": "SURF",
                        "surf_outer": surf_outer,
                        "slot_index": slot_index,
                        "local_step": local_step,
                        "optimizer_step": slot_update_counts[slot_index],
                        "parameter_update": slot_update_counts[slot_index],
                        "parameter_updates": slot_update_counts[slot_index],
                        "cumulative_parameter_updates": cumulative_parameter_updates,
                        "objective_gradient_evals": num_objectives * cumulative_parameter_updates,
                        "elapsed_wall_seconds": time.perf_counter() - run_start_time,
                        "lambda_helpful": helpful_weight,
                        "lambda_harmless": harmless_weight,
                        "loss": float(update_stats["train_loss"]),
                        "helpful_loss": float(update_stats["train_helpful_loss"]),
                        "harmless_loss": float(update_stats["train_harmless_loss"]),
                        "update_data_source": args.update_data_source,
                        "update_batches": int(update_stats["update_batches"]),
                        "update_examples": int(update_stats["update_examples"]),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                    }
                    save_jsonl(history_path, train_record)
                    if total_updates > 0:
                        progress.set_description(
                            f"surf={surf_outer}/{args.surf_max_outer} "
                            f"slot={slot_index + 1}/{slot_count} "
                            f"loss={train_record['loss']:.4f}"
                        )
                        progress.update(1)

                solution_vectors[slot_index] = trainer.get_trainable_parameter_vector(cpu=True).numpy()
                oracle = trainer.multi_objective_gradient_oracle_over_batches(
                    objective_batch_groups,
                    as_numpy=True,
                )
                oracle_gradient_evals += 1
                objective_points.append(np.asarray(oracle["fvals"], dtype=np.float64))
                gn_bundle.replace(slot_index, oracle["x"], oracle["fvals"], oracle["grads"])

                checkpoint_dir = None
                logged_checkpoint_dir = None
                if save_this_outer:
                    checkpoint_dir = os.path.join(run_dir, "final_checkpoint")
                    trainer.model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logged_checkpoint_dir = os.path.relpath(checkpoint_dir, output_dir)

                checkpoint_counter += 1
                record = {
                    "method": "SURF",
                    "run": run_name,
                    "outer": checkpoint_counter,
                    "surf_outer": surf_outer,
                    "slot_index": slot_index,
                    "slot_count": slot_count,
                    "lambda": [helpful_weight, harmless_weight],
                    "lambda_helpful": helpful_weight,
                    "lambda_harmless": harmless_weight,
                    "checkpoint_dir": logged_checkpoint_dir,
                    "fvals": oracle["fvals"].tolist(),
                    "objective_names": list(oracle.get("objective_names", ["helpful", "harmless"])),
                    "parameter_updates_before": cumulative_parameter_updates - steps_per_slot_outer,
                    "parameter_updates_after": cumulative_parameter_updates,
                    "objective_gradient_evals_before": num_objectives
                    * (cumulative_parameter_updates - steps_per_slot_outer),
                    "objective_gradient_evals_after": num_objectives * cumulative_parameter_updates,
                    "oracle_gradient_evals_after": oracle_gradient_evals,
                    "elapsed_wall_seconds_after": time.perf_counter() - run_start_time,
                    "timing_slot_seconds": time.perf_counter() - slot_t0,
                    "update_counts": list(slot_update_counts),
                    "warm_start_strategy": args.surf_warm_start_strategy,
                    "num_objectives": num_objectives,
                    "bundle_size": gn_bundle.m,
                }
                outer_records.append(record)

            objective_points_array = np.stack(objective_points, axis=0)
            surrogate_cdf, arclength_at_weight, lengths = build_surrogate_cdf_from_points(
                helpful_weights,
                objective_points_array,
                cdf_grid,
                use_pchip=bool(args.surf_use_pchip),
            )
            next_cdf = enforce_monotone_cdf(
                blend_cdfs(current_cdf, surrogate_cdf, float(args.surf_alpha)),
                eps=float(args.surf_monotone_eps),
                force_endpoints=bool(args.surf_force_endpoints),
            )
            gn_star, gn_lambda = maximise_gn(
                gn_bundle,
                max_starts=args.lambda_max_starts,
                solver=args.lambda_solver,
                require_ipopt=args.require_ipopt,
            )
            best_gn_star = min(best_gn_star, float(gn_star))
            best_gradient_norm = float(np.sqrt(max(best_gn_star, 0.0)))
            target_reached = bool(
                args.gn_target_norm is not None
                and best_gradient_norm <= float(args.gn_target_norm)
            )

            metric_record = {
                "method": "SURF",
                "surf_outer": surf_outer,
                "parameter_updates": cumulative_parameter_updates,
                "objective_gradient_evals": num_objectives * cumulative_parameter_updates,
                "oracle_gradient_evals": oracle_gradient_evals,
                "elapsed_wall_seconds": time.perf_counter() - run_start_time,
                "lambda_helpful": helpful_weights.tolist(),
                "objective_points": objective_points_array.tolist(),
                "segment_lengths": lengths.tolist(),
                "arclength_at_weight": arclength_at_weight.tolist(),
                "segment_cv": coefficient_of_variation(lengths),
                "gap_ratio": gap_ratio(lengths),
                "gn_star": float(gn_star),
                "best_gn_star": float(best_gn_star),
                "best_gradient_norm": best_gradient_norm,
                "gn_target_norm": args.gn_target_norm,
                "target_reached": target_reached,
                "lambda_gn_star": gn_lambda.tolist(),
                "lambda_solver": args.lambda_solver,
                "gn_certificate_type": (
                    "exact_two_objective_full_simplex"
                    if args.lambda_solver == "exact_k2"
                    else "local_solver_lower_bound"
                ),
            }
            metrics_history.append(metric_record)
            if target_reached and target_hit_record is None:
                target_hit_record = dict(metric_record)
            pf_history.append({
                "surf_outer": surf_outer,
                "points": [
                    {
                        "slot_index": idx,
                        "lambda_helpful": float(weight_pairs[idx][0]),
                        "lambda_harmless": float(weight_pairs[idx][1]),
                        "fvals": objective_points_array[idx].tolist(),
                    }
                    for idx in range(slot_count)
                ],
            })
            cdf_record = {
                "surf_outer": surf_outer,
                "cdf_grid": cdf_grid.tolist(),
                "cdf_values": next_cdf.tolist(),
                "surrogate_cdf_values": surrogate_cdf.tolist(),
            }
            cdf_history.append(cdf_record)

            outer_records[-1].update(
                {
                    "gn_star": float(gn_star),
                    "lambda_gn_star": gn_lambda.tolist(),
                    "lambda_solver": args.lambda_solver,
                    "gn_certificate_type": metric_record["gn_certificate_type"],
                    "segment_cv": metric_record["segment_cv"],
                    "gap_ratio": metric_record["gap_ratio"],
                    "best_gn_star": metric_record["best_gn_star"],
                    "best_gradient_norm": metric_record["best_gradient_norm"],
                    "gn_target_norm": args.gn_target_norm,
                    "target_reached": target_reached,
                    "phase": "surf_outer_end",
                }
            )
            for record in outer_records:
                save_jsonl(os.path.join(output_dir, "adaptive_history.jsonl"), record)
                save_jsonl(os.path.join(output_dir, "surf_history.jsonl"), record)
            save_jsonl(os.path.join(output_dir, "surf_metric_history.jsonl"), metric_record)
            save_jsonl(os.path.join(output_dir, "surf_cdf_history.jsonl"), cdf_record)

            print_local_main(
                f"surf_outer={surf_outer} "
                f"updates={cumulative_parameter_updates} "
                f"weights={[round(float(w), 4) for w in helpful_weights]} "
                f"gn*={float(gn_star):.4e} "
                f"cv={metric_record['segment_cv']:.4f} "
                f"gap={metric_record['gap_ratio']:.4f}"
            )
            current_cdf = next_cdf
            if target_reached:
                print_local_main(
                    "SURF GN target reached: "
                    f"best_norm={best_gradient_norm:.4e}, "
                    f"target={float(args.gn_target_norm):.4e}, "
                    f"updates={cumulative_parameter_updates}"
                )
                break

    finally:
        progress.close()
        del trainer
        for optimizer in optimizers:
            del optimizer
        for scheduler in schedulers:
            del scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(os.path.join(output_dir, "surf_weight_history.json"), weights_history)
    write_json(os.path.join(output_dir, "surf_metric_history.json"), metrics_history)
    write_json(os.path.join(output_dir, "surf_cdf_history.json"), cdf_history)
    write_json(os.path.join(output_dir, "pf_history.json"), pf_history)
    if args.gn_target_norm is not None:
        write_json(
            os.path.join(output_dir, "plateau_summary.json"),
            {
                "method": "SURF",
                "gn_target_norm": float(args.gn_target_norm),
                "target_reached": bool(target_reached),
                "target_hit": target_hit_record,
                "best_gn_star": None if not np.isfinite(best_gn_star) else float(best_gn_star),
                "best_gradient_norm": (
                    None
                    if not np.isfinite(best_gn_star)
                    else float(np.sqrt(max(best_gn_star, 0.0)))
                ),
                "parameter_updates": cumulative_parameter_updates,
                "objective_gradient_evals": num_objectives * cumulative_parameter_updates,
            },
        )
    print_local_main(f"saved SURF run to {output_dir}")


if __name__ == "__main__":
    main()
