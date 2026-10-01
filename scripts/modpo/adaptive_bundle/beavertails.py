from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

from scripts.modpo.adaptive_bundle.bundle_core import (
    FirstOrderBundle,
    LAMBDA_SOLVERS,
    active_gn_source,
    bundle_gradient_diagnostics,
    diversify_lambda_on_grid,
    gn_grid_diagnostics,
    gn_value_at_lambda,
    ipopt_available,
    ipopt_import_error,
    maximise_gn,
    prune_last_candidates,
    project_truncated_simplex,
    t_map_step,
)
import numpy as np
import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, get_scheduler

from scripts.modpo.adaptive_bundle.trainer import AdaptiveBundleMODPOTrainer
from src.data.configs import DATASET_CONFIGS
from src.data.raw_data.safe_rlhf import agreement_stats, sample_agreement_mixture
from src.trainer.modpo_trainer import MODPODataCollatorWithPadding, MODPODataMapFunc
from src.utils import (
    disable_progress_bar_non_local_main,
    param_sharding_enabled,
    print_local_main,
    set_seeds,
)

disable_progress_bar_non_local_main()


QWEN_PROMPT_TEMPLATE = "<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n"


def find_matching_lambda(bundle_lambdas, lam, tol: float) -> Optional[int]:
    if tol < 0.0:
        raise ValueError("lambda_match_tol must be non-negative")
    lam_arr = np.asarray(lam, dtype=np.float64)
    for idx, stored_lam in enumerate(bundle_lambdas):
        if stored_lam is None:
            continue
        stored_arr = np.asarray(stored_lam, dtype=np.float64)
        if stored_arr.shape == lam_arr.shape and np.max(np.abs(stored_arr - lam_arr)) <= tol:
            return idx
    return None


def scalarized_objective(fvals, lam) -> float:
    return float(np.asarray(fvals, dtype=np.float64) @ np.asarray(lam, dtype=np.float64))


def scalarized_gradient_norm_sq(bundle: FirstOrderBundle, index: int, lam) -> float:
    grad = np.asarray(bundle.grads[index], dtype=np.float64)
    grad_lam = np.einsum("kd,k->d", grad, np.asarray(lam, dtype=np.float64), optimize=True)
    return float(np.dot(grad_lam, grad_lam))


def active_gn_source_prefix(bundle: FirstOrderBundle, lam, prefix_m: int) -> tuple[int, float]:
    """Active worst-case source restricted to the first prefix_m bundle points."""
    upper = min(int(prefix_m), bundle.m)
    if upper < 1:
        raise ValueError("Cannot select an active source from an empty bundle prefix.")
    lam_arr = np.asarray(lam, dtype=np.float64)
    grads = np.asarray(bundle.grads[:upper], dtype=np.float64)
    grad_lam = np.einsum("mkd,k->md", grads, lam_arr, optimize=True)
    gnorms_sq = np.einsum("md,md->m", grad_lam, grad_lam, optimize=True)
    idx = int(np.argmin(gnorms_sq))
    return idx, float(gnorms_sq[idx])


def clone_bundle_indices(bundle: FirstOrderBundle, indices) -> FirstOrderBundle:
    """Build a small temporary bundle from selected entries."""
    cloned = FirstOrderBundle(
        K=bundle.K,
        d=bundle.d,
        L=bundle.L,
        dtype=bundle.dtype,
        lambda_projection_dim=(
            int(bundle.lambda_projection_dim)
            if bundle.lambda_projection_active and bundle.lambda_projection_dim is not None
            else None
        ),
        lambda_projection_seed=int(bundle.lambda_projection_seed),
    )
    for idx in indices:
        cloned.add(bundle.points[idx], bundle.fvals[idx], bundle.grads[idx])
    return cloned


def best_global_cap_swap(
    bundle: FirstOrderBundle,
    candidate_idx: int,
    prefix_m: int,
    *,
    prev_lam,
    max_starts: int,
    solver: str,
    require_ipopt: bool,
    lambda_normalization: str,
    lambda_min: float,
    use_projection: bool,
) -> dict:
    """Try replacing every old bundle point by the new point and minimize GN*."""
    prefix_m = min(int(prefix_m), bundle.m)
    if prefix_m < 1:
        raise ValueError("Cannot run a global cap swap with an empty prefix.")
    if candidate_idx < 0 or candidate_idx >= bundle.m:
        raise IndexError(f"candidate_idx out of range: {candidate_idx}")

    base_indices = list(range(prefix_m))
    base_bundle = clone_bundle_indices(bundle, base_indices)
    total_solver_seconds = 0.0
    base_solver_t0 = time.perf_counter()
    base_gn, base_lam = maximise_gn(
        base_bundle,
        prev_lam=prev_lam,
        max_starts=max_starts,
        solver=solver,
        require_ipopt=require_ipopt,
        lambda_normalization=lambda_normalization,
        lambda_min=lambda_min,
        use_projection=use_projection,
        entropy_tau=0.0,
    )
    base_solver_seconds = time.perf_counter() - base_solver_t0
    total_solver_seconds += base_solver_seconds

    best_replace_idx = None
    best_gn = float(base_gn)
    best_lam = base_lam.copy()
    candidates = []
    for replace_idx in range(prefix_m):
        swapped_indices = base_indices.copy()
        swapped_indices[replace_idx] = int(candidate_idx)
        swapped_bundle = clone_bundle_indices(bundle, swapped_indices)
        swap_solver_t0 = time.perf_counter()
        swapped_gn, swapped_lam = maximise_gn(
            swapped_bundle,
            prev_lam=prev_lam,
            max_starts=max_starts,
            solver=solver,
            require_ipopt=require_ipopt,
            lambda_normalization=lambda_normalization,
            lambda_min=lambda_min,
            use_projection=use_projection,
            entropy_tau=0.0,
        )
        swap_solver_seconds = time.perf_counter() - swap_solver_t0
        total_solver_seconds += swap_solver_seconds
        row = {
            "replace_idx": int(replace_idx),
            "gn_star": float(swapped_gn),
            "lambda": swapped_lam.tolist(),
            "solver_seconds": float(swap_solver_seconds),
        }
        candidates.append(row)
        if np.isfinite(swapped_gn) and float(swapped_gn) < best_gn:
            best_replace_idx = int(replace_idx)
            best_gn = float(swapped_gn)
            best_lam = swapped_lam.copy()

    return {
        "base_gn_star": float(base_gn),
        "base_lambda": base_lam.tolist(),
        "best_replace_idx": best_replace_idx,
        "best_gn_star": float(best_gn),
        "best_lambda": best_lam.tolist(),
        "candidates": candidates,
        "base_solver_seconds": float(base_solver_seconds),
        "total_solver_seconds": float(total_solver_seconds),
        "solver_calls": int(prefix_m + 1),
    }


@dataclass
class ScriptArguments:
    sft_model_name: str = field(default="Qwen/Qwen2.5-0.5B-Instruct")
    use_flash_attention_2: Optional[bool] = field(default=False)
    prompt_template: Optional[str] = field(default=QWEN_PROMPT_TEMPLATE)
    better_dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    safer_dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K-safer")
    dataset_caching: Optional[bool] = field(default=False)
    sanity_check: Optional[bool] = field(default=False)

    beta: Optional[float] = field(default=0.1)
    max_length: Optional[int] = field(default=384)
    num_proc: Optional[int] = field(default=4)
    train_subset_size_per_objective: Optional[int] = field(default=2000)
    oracle_subset_size_per_objective: Optional[int] = field(default=128)
    oracle_batch_size: Optional[int] = field(default=4)
    seed: Optional[int] = field(default=42)
    consistent_preferences_only: Optional[bool] = field(default=False)
    shared_objective_subset: Optional[bool] = field(default=False)
    agreement_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional controlled agreement ratio for PKU-SafeRLHF rows. "
                "For example, 0.7 builds shared helpful/harmless prompts with "
                "70% better==safer and 30% better!=safer."
            )
        },
    )

    max_outer: Optional[int] = field(default=20)
    max_inner: Optional[int] = field(default=25)
    algorithm_mode: Optional[str] = field(default="llm")
    stop_rule: Optional[str] = field(default="none")
    epsilon: Optional[float] = field(default=None)
    relative_rho: Optional[float] = field(default=0.5)
    gn_target_norm: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional pre-specified worst-case gradient-norm target. "
                "The adaptive run stops at its first best-so-far GN hit."
            )
        },
    )
    update_rule: Optional[str] = field(default="adamw")
    bundle_update_mode: Optional[str] = field(default="lambda_aware")
    update_data_source: Optional[str] = field(default="oracle")
    per_objective_batch_size: Optional[int] = field(default=2)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    warmup_ratio: Optional[float] = field(default=0.03)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    weight_decay: Optional[float] = field(default=0.0)
    max_grad_norm: Optional[float] = field(default=1.0)
    lambda_max_starts: Optional[int] = field(default=64)
    lambda_solver: Optional[str] = field(default="ipopt")
    require_ipopt: Optional[bool] = field(default=True)
    lambda_normalization: Optional[str] = field(default="none")
    lambda_min: Optional[float] = field(default=0.0)
    lambda_entropy_tau: Optional[float] = field(default=0.0)
    lambda_diversity_strength: Optional[float] = field(default=0.0)
    lambda_diversity_grid_points: Optional[int] = field(default=101)
    lambda_diversity_recent_window: Optional[int] = field(default=3)
    lambda_projection_dim: Optional[int] = field(default=0)
    lambda_projection_seed: Optional[int] = field(default=0)
    lambda_match_tol: Optional[float] = field(default=1e-4)
    lambda_stall_patience: Optional[int] = field(default=0)
    lambda_stall_abs_delta: Optional[float] = field(default=0.0)
    lambda_stall_rel_delta: Optional[float] = field(default=0.0)
    lambda_stall_cooldown: Optional[int] = field(default=1)
    lambda_stall_grid_points: Optional[int] = field(default=101)
    lambda_stall_match_tol: Optional[float] = field(default=0.05)
    bundle_warm_start_steps: Optional[int] = field(default=0)
    bundle_warm_start_lambdas: Optional[str] = field(default="0.0;0.25;0.5;0.75;1.0")
    resume: Optional[bool] = field(default=False)
    resume_state_path: Optional[str] = field(default=None)
    prefix_save_bundle_sizes: Optional[str] = field(default=None)
    prefix_state_dir: Optional[str] = field(default=None)
    smoothness: Optional[str] = field(default="1.0,1.0")
    l_scale: Optional[float] = field(default=1.0)
    descent_atol: Optional[float] = field(default=1e-6)
    descent_rtol: Optional[float] = field(default=1e-6)
    prune_inner: Optional[bool] = field(default=False)
    max_bundle_size: Optional[int] = field(default=0)
    bundle_cap_mode: Optional[str] = field(default="replace_active_if_better")
    save_every_outer: Optional[int] = field(default=0)
    bundle_dtype: Optional[str] = field(default="float32")

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/adaptive_bundle",
            overwrite_output_dir=True,
            seed=42,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=1e-4,
            bf16=torch.cuda.is_available(),
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="no",
        )
    )

    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    )


class CyclingLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def parse_float_list(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_lambda_schedule(value: Optional[str], num_objectives: int):
    if value is None or value.strip().lower() in {"", "none", "false"}:
        return []

    lambdas = []
    for raw_item in value.split(";"):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        values = parse_float_list(raw_item)
        if len(values) == 1 and num_objectives == 2:
            helpful_weight = values[0]
            values = [helpful_weight, 1.0 - helpful_weight]
        if len(values) != num_objectives:
            raise ValueError(
                "Each warm-start lambda must either be one helpful weight "
                f"(for K=2) or have {num_objectives} comma-separated values."
            )
        lam = np.asarray(values, dtype=np.float64)
        if np.any(~np.isfinite(lam)) or np.any(lam < 0.0):
            raise ValueError("Warm-start lambdas must be finite and non-negative.")
        total = float(lam.sum())
        if total <= 0.0:
            raise ValueError("Warm-start lambdas must have positive mass.")
        lambdas.append(lam / total)
    return lambdas


def select_subset(dataset, size, seed, name):
    if size is None or size <= 0 or size >= len(dataset):
        print_local_main(f"{name}: using {len(dataset)} samples")
        return dataset
    dataset = dataset.shuffle(seed=seed)
    print_local_main(f"{name}: selected {size} / {len(dataset)} samples")
    return dataset.select(range(size))


def preprocess_preference_dataset(dataset, tokenizer, max_length, num_proc):
    map_func = MODPODataMapFunc(tokenizer)
    dataset = dataset.map(
        map_func,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.filter(
        lambda sample: (
            len(sample["prompt_chosen_input_ids"]) <= max_length
            and len(sample["prompt_rejected_input_ids"]) <= max_length
        ),
        num_proc=num_proc,
    )
    return dataset


def map_raw_preference_dataset(raw_dataset, rdp, name: str):
    print_local_main(f"mapping {name} raw rows to preference format...")
    return raw_dataset.map(
        rdp._dataset_to_preference_formatter,
        num_proc=rdp.num_proc,
        remove_columns=raw_dataset.column_names,
    )


def make_controlled_agreement_pools(
    better_rdp,
    safer_rdp,
    split: str,
    size,
    agreement_ratio: float,
    seed: int,
):
    raw_dataset = better_rdp._get_raw_dataset(split=split)
    raw_stats = agreement_stats(raw_dataset)
    mixed_raw = sample_agreement_mixture(
        raw_dataset,
        size=size,
        agreement_ratio=agreement_ratio,
        seed=seed,
    )
    mixed_stats = agreement_stats(mixed_raw)
    print_local_main(
        "controlled agreement mixture: "
        f"raw agree={raw_stats['agree']}/{raw_stats['total']} "
        f"({raw_stats['agree_ratio']:.2%}), "
        f"selected agree={mixed_stats['agree']}/{mixed_stats['total']} "
        f"({mixed_stats['agree_ratio']:.2%}), "
        f"disagree={mixed_stats['disagree']}/{mixed_stats['total']} "
        f"({mixed_stats['disagree_ratio']:.2%})"
    )
    better_pool = map_raw_preference_dataset(
        mixed_raw,
        better_rdp,
        "helpful/better train pool",
    )
    safer_pool = map_raw_preference_dataset(
        mixed_raw,
        safer_rdp,
        "harmless/safer train pool",
    )
    return better_pool, safer_pool


def make_fixed_batches(dataset, batch_size, collator):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )
    return list(dataloader)


def save_jsonl(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def resume_paths(output_dir: str, resume_state_path: Optional[str] = None):
    base = resume_state_path or os.path.join(output_dir, "adaptive_resume_state")
    root, ext = os.path.splitext(base)
    if ext in {".npz", ".json", ".pt"}:
        base = root
    return {
        "base": base,
        "npz": f"{base}.npz",
        "json": f"{base}.json",
        "optimizer": f"{base}_optimizer.pt",
    }


def parse_positive_int_set(value: Optional[str], *, name: str) -> set[int]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text or text.lower() in {"none", "false", "0"}:
        return set()
    out = set()
    for raw in text.replace(";", ",").split(","):
        part = raw.strip()
        if not part:
            continue
        try:
            item = int(part)
        except ValueError as exc:
            raise ValueError(f"{name} must contain positive integers, got {part!r}") from exc
        if item <= 0:
            raise ValueError(f"{name} entries must be positive, got {item}")
        out.add(item)
    return out


def prefix_resume_paths(prefix_state_dir: str, bundle_size: int):
    base = os.path.join(prefix_state_dir, f"bundle_size_{int(bundle_size):04d}")
    return resume_paths(prefix_state_dir, base)


def copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_dst = f"{dst}.tmp"
    shutil.copyfile(src, tmp_dst)
    os.replace(tmp_dst, dst)
    return True


def save_resume_state(
    paths,
    bundle: FirstOrderBundle,
    bundle_lambdas,
    lambda_history,
    lambda_stall_states,
    prev_lam,
    next_outer: int,
    total_parameter_updates: int,
    total_oracle_gradient_evals: int,
    total_inner_steps: int,
    current_l_scale: float,
    safeguard_violations: int,
    safeguard_warned: bool,
    relative_outer_reference_gn,
    elapsed_wall_seconds: Optional[float] = None,
    optimizer=None,
    scheduler=None,
):
    points = np.stack(bundle.points, axis=0)
    fvals = np.stack(bundle.fvals, axis=0)
    grads = np.stack(bundle.grads, axis=0)
    lambdas = (
        np.stack(bundle_lambdas, axis=0)
        if bundle_lambdas
        else np.empty((0, bundle.K), dtype=np.float64)
    )
    history = (
        np.stack(lambda_history, axis=0)
        if lambda_history
        else np.empty((0, bundle.K), dtype=np.float64)
    )
    prev = (
        np.asarray(prev_lam, dtype=np.float64)
        if prev_lam is not None
        else np.empty((0,), dtype=np.float64)
    )

    os.makedirs(os.path.dirname(paths["npz"]), exist_ok=True)
    tmp_npz = f"{paths['npz']}.tmp"
    with open(tmp_npz, "wb") as handle:
        np.savez_compressed(
            handle,
            points=points,
            fvals=fvals,
            grads=grads,
            bundle_lambdas=lambdas,
            lambda_history=history,
            prev_lam=prev,
        )
    os.replace(tmp_npz, paths["npz"])

    metadata = {
        "version": 1,
        "next_outer": int(next_outer),
        "total_parameter_updates": int(total_parameter_updates),
        "total_oracle_gradient_evals": int(total_oracle_gradient_evals),
        "total_inner_steps": int(total_inner_steps),
        "current_l_scale": float(current_l_scale),
        "safeguard_violations": int(safeguard_violations),
        "safeguard_warned": bool(safeguard_warned),
        "relative_outer_reference_gn": (
            float(relative_outer_reference_gn)
            if relative_outer_reference_gn is not None
            else None
        ),
        "lambda_stall_states": lambda_stall_states,
        "bundle_size": int(bundle.m),
        "elapsed_wall_seconds": (
            float(elapsed_wall_seconds) if elapsed_wall_seconds is not None else None
        ),
    }
    tmp_json = f"{paths['json']}.tmp"
    save_json(tmp_json, metadata)
    os.replace(tmp_json, paths["json"])

    if optimizer is not None:
        payload = {"optimizer": optimizer.state_dict()}
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        tmp_optimizer = f"{paths['optimizer']}.tmp"
        torch.save(payload, tmp_optimizer)
        os.replace(tmp_optimizer, paths["optimizer"])


def load_resume_state(paths, bundle: FirstOrderBundle):
    if not os.path.exists(paths["npz"]) or not os.path.exists(paths["json"]):
        raise FileNotFoundError(
            "Resume requested but adaptive resume files are missing: "
            f"{paths['npz']} and/or {paths['json']}"
        )
    with open(paths["json"]) as handle:
        metadata = json.load(handle)
    arrays = np.load(paths["npz"])
    points = arrays["points"]
    fvals = arrays["fvals"]
    grads = arrays["grads"]
    if points.shape[0] != fvals.shape[0] or points.shape[0] != grads.shape[0]:
        raise ValueError("Resume bundle arrays have inconsistent lengths")
    for idx in range(points.shape[0]):
        bundle.add(points[idx], fvals[idx], grads[idx])
    bundle_lambdas = [lam.copy() for lam in arrays["bundle_lambdas"]]
    lambda_history = [lam.copy() for lam in arrays["lambda_history"]]
    prev_arr = arrays["prev_lam"]
    prev_lam = prev_arr.copy() if prev_arr.shape == (bundle.K,) else None
    lambda_stall_states = metadata.get("lambda_stall_states", [])
    return metadata, bundle_lambdas, lambda_history, lambda_stall_states, prev_lam


def validate_stop_rule(args: ScriptArguments) -> None:
    if args.algorithm_mode not in {"llm", "theory"}:
        raise ValueError("algorithm_mode must be either 'llm' or 'theory'.")
    if args.stop_rule not in {"none", "absolute", "relative"}:
        raise ValueError("stop_rule must be one of: 'none', 'absolute', 'relative'.")
    if args.stop_rule == "absolute":
        if args.epsilon is None or not np.isfinite(args.epsilon) or args.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive when stop_rule='absolute'.")
    if args.stop_rule == "relative":
        if args.relative_rho is None or not np.isfinite(args.relative_rho):
            raise ValueError("relative_rho must be finite when stop_rule='relative'.")
        if args.relative_rho <= 0.0 or args.relative_rho >= 1.0:
            raise ValueError("relative_rho must be in (0, 1) when stop_rule='relative'.")
    if args.gn_target_norm is not None and float(args.gn_target_norm) <= 0.0:
        raise ValueError("gn_target_norm must be positive when provided.")


def stop_threshold(args: ScriptArguments, kind: str, reference: Optional[float] = None) -> Optional[float]:
    if args.stop_rule == "none":
        return None
    if args.stop_rule == "absolute":
        factor = 2.0 / 3.0 if kind == "outer" else 1.0 / 3.0
        return float(args.epsilon) * factor
    if args.stop_rule == "relative":
        if reference is None:
            return None
        return float(args.relative_rho) * float(reference)
    raise ValueError(f"Unknown stop rule: {args.stop_rule}")


def stop_reached(value: float, threshold: Optional[float]) -> bool:
    return threshold is not None and np.isfinite(value) and value < threshold


def lambda_certificate_type(args: ScriptArguments, bundle: FirstOrderBundle, entropy_tau: float) -> str:
    if args.lambda_solver == "exact_k2":
        raw_full_simplex = (
            args.lambda_normalization == "none"
            and not bundle.lambda_projection_active
            and float(args.lambda_min) == 0.0
            and float(entropy_tau) == 0.0
        )
        if raw_full_simplex:
            return "exact_two_objective_full_simplex"
        return "exact_two_objective_modified_geometry"
    return "local_solver_lower_bound"


def unique_ints(values):
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        int_value = int(value)
        if int_value not in seen:
            seen.add(int_value)
            result.append(int_value)
    return result


def lambda_close(lam_a, lam_b, tol: float) -> bool:
    arr_a = np.asarray(lam_a, dtype=np.float64)
    arr_b = np.asarray(lam_b, dtype=np.float64)
    return arr_a.shape == arr_b.shape and float(np.max(np.abs(arr_a - arr_b))) <= tol


def find_lambda_stall_state(stall_states, lam, tol: float):
    for state in stall_states:
        if lambda_close(state["lambda"], lam, tol):
            return state
    return None


def active_stalled_lambdas(stall_states, outer: int):
    return [
        np.asarray(state["lambda"], dtype=np.float64)
        for state in stall_states
        if int(state.get("blocked_until_outer", 0)) >= outer
    ]


def lambda_is_blocked(lam, blocked_lambdas, tol: float) -> bool:
    return any(lambda_close(lam, blocked, tol) for blocked in blocked_lambdas)


def choose_unblocked_lambda_from_grid(
    bundle: FirstOrderBundle,
    blocked_lambdas,
    *,
    outer: int,
    lambda_normalization: str,
    lambda_min: float,
    entropy_tau: float,
    num_points: int,
    match_tol: float,
    use_projection: bool,
):
    base_info = {
        "enabled": bool(blocked_lambdas),
        "selected_by": "gn",
        "outer": int(outer),
        "blocked_lambdas": [blocked.tolist() for blocked in blocked_lambdas],
        "match_tol": float(match_tol),
        "num_points": int(num_points),
        "chosen_lambda": None,
        "chosen_gn": None,
        "chosen_objective": None,
    }
    if not blocked_lambdas or bundle.K != 2:
        return None, base_info

    rows = gn_grid_diagnostics(
        bundle,
        num_points=num_points,
        lambda_normalization=lambda_normalization,
        lambda_min=lambda_min,
        use_projection=use_projection,
    )
    best_row = None
    best_score = -np.inf
    for row in rows:
        lam = np.asarray(
            [row["lambda_helpful"], row["lambda_harmless"]],
            dtype=np.float64,
        )
        if lambda_is_blocked(lam, blocked_lambdas, match_tol):
            continue
        gn_value = float(row["gn"])
        if not np.isfinite(gn_value):
            continue
        score = gn_value
        if entropy_tau > 0.0:
            lam_safe = np.clip(lam, np.finfo(np.float64).tiny, 1.0)
            score += float(entropy_tau) * float(-np.sum(lam_safe * np.log(lam_safe)))
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        base_info["selected_by"] = "all_grid_lambdas_blocked"
        return None, base_info

    chosen = project_truncated_simplex(
        [best_row["lambda_helpful"], best_row["lambda_harmless"]],
        lambda_min=lambda_min,
    )
    base_info.update({
        "selected_by": "stall_unblocked_grid",
        "chosen_lambda": chosen.tolist(),
        "chosen_gn": float(best_row["gn"]),
        "chosen_objective": float(best_score),
    })
    return chosen, base_info


def update_lambda_stall_states(
    stall_states,
    lam,
    gn_before: float,
    gn_after: float,
    *,
    outer: int,
    patience: int,
    abs_delta: float,
    rel_delta: float,
    cooldown: int,
    match_tol: float,
):
    improvement = float(gn_before) - float(gn_after)
    required = max(float(abs_delta), float(rel_delta) * max(abs(float(gn_before)), np.finfo(np.float64).eps))
    stalled = bool(np.isfinite(improvement) and improvement <= required)
    state = find_lambda_stall_state(stall_states, lam, match_tol)
    if state is None:
        state = {
            "lambda": np.asarray(lam, dtype=np.float64).tolist(),
            "stall_count": 0,
            "blocked_until_outer": 0,
            "last_outer": None,
            "last_gn_before": None,
            "last_gn_after": None,
            "last_improvement": None,
            "last_required_improvement": None,
            "last_stalled": False,
        }
        stall_states.append(state)

    if stalled:
        state["stall_count"] = int(state.get("stall_count", 0)) + 1
    else:
        state["stall_count"] = 0

    if patience > 0 and cooldown > 0 and int(state["stall_count"]) >= patience:
        state["blocked_until_outer"] = int(outer + cooldown)

    state.update({
        "last_outer": int(outer),
        "last_gn_before": float(gn_before),
        "last_gn_after": float(gn_after),
        "last_improvement": float(improvement),
        "last_required_improvement": float(required),
        "last_stalled": stalled,
    })
    return {
        "enabled": patience > 0,
        "lambda": np.asarray(lam, dtype=np.float64).tolist(),
        "stalled": stalled,
        "improvement": float(improvement),
        "required_improvement": float(required),
        "stall_count": int(state["stall_count"]),
        "blocked_until_outer": int(state.get("blocked_until_outer", 0)),
    }


def weighted_dpo_loss(trainer, helpful_batch, harmless_batch, helpful_weight, harmless_weight):
    helpful_batch = trainer._prepare_inputs(helpful_batch)
    harmless_batch = trainer._prepare_inputs(harmless_batch)

    helpful_loss = trainer.dpo_objective_loss(trainer.model, helpful_batch)
    harmless_loss = trainer.dpo_objective_loss(trainer.model, harmless_batch)
    loss = helpful_weight * helpful_loss + harmless_weight * harmless_loss
    return loss, helpful_loss.detach(), harmless_loss.detach()


def dpo_batch_examples(batch) -> int:
    return int(batch["input_ids"].shape[0] // 2)


def backward_weighted_dpo_update(
    trainer,
    lam,
    update_data_source,
    helpful_loader,
    harmless_loader,
    objective_batch_groups,
    gradient_accumulation_steps,
):
    helpful_weight = float(lam[0])
    harmless_weight = float(lam[1])

    if update_data_source == "oracle":
        losses = {}
        update_batches = 0
        update_examples = 0
        objective_specs = [
            ("helpful", helpful_weight),
            ("harmless", harmless_weight),
        ]
        for objective_name, objective_weight in objective_specs:
            batches = list(objective_batch_groups[objective_name])
            if not batches:
                raise ValueError(f"Objective {objective_name!r} has no oracle batches.")
            batch_examples = []
            for batch in batches:
                batch_examples.append(dpo_batch_examples(batch))
            total_examples = sum(batch_examples)
            if total_examples <= 0:
                raise ValueError(f"Objective {objective_name!r} has no oracle examples.")

            loss_sum = 0.0
            for batch, count in zip(batches, batch_examples):
                prepared_batch = trainer._prepare_inputs(batch)
                loss = trainer.dpo_objective_loss(trainer.model, prepared_batch)
                loss_sum += float(loss.detach().cpu()) * count
                scaled_loss = objective_weight * loss * (count / total_examples)
                scaled_loss.backward()
                update_batches += 1
                update_examples += count
            losses[objective_name] = loss_sum / total_examples

        helpful_loss = float(losses["helpful"])
        harmless_loss = float(losses["harmless"])
        train_loss = helpful_weight * helpful_loss + harmless_weight * harmless_loss
        return {
            "train_loss": train_loss,
            "train_helpful_loss": helpful_loss,
            "train_harmless_loss": harmless_loss,
            "update_batches": update_batches,
            "update_examples": update_examples,
        }

    micro_losses = []
    micro_helpful_losses = []
    micro_harmless_losses = []
    update_examples = 0
    for _ in range(gradient_accumulation_steps):
        helpful_batch = helpful_loader.next()
        harmless_batch = harmless_loader.next()
        loss, helpful_loss, harmless_loss = weighted_dpo_loss(
            trainer,
            helpful_batch,
            harmless_batch,
            helpful_weight,
            harmless_weight,
        )
        micro_losses.append(float(loss.detach().cpu()))
        micro_helpful_losses.append(float(helpful_loss.cpu()))
        micro_harmless_losses.append(float(harmless_loss.cpu()))
        update_examples += dpo_batch_examples(helpful_batch) + dpo_batch_examples(harmless_batch)
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

    return {
        "train_loss": float(np.mean(micro_losses)),
        "train_helpful_loss": float(np.mean(micro_helpful_losses)),
        "train_harmless_loss": float(np.mean(micro_harmless_losses)),
        "update_batches": int(2 * gradient_accumulation_steps),
        "update_examples": update_examples,
    }


def main():
    run_start_time = time.perf_counter()
    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.seed)
    os.makedirs(script_args.training_args.output_dir, exist_ok=True)
    validate_stop_rule(script_args)

    smoothness = parse_float_list(script_args.smoothness)
    if len(smoothness) != 2:
        raise ValueError("The BeaverTails runner expects exactly two smoothness constants.")
    if script_args.bundle_dtype not in {"float32", "float64"}:
        raise ValueError("bundle_dtype must be either 'float32' or 'float64'.")
    if not np.isfinite(script_args.l_scale) or script_args.l_scale <= 0.0:
        raise ValueError("l_scale must be finite and strictly positive.")
    if not np.isfinite(script_args.descent_atol) or script_args.descent_atol < 0.0:
        raise ValueError("descent_atol must be finite and non-negative.")
    if not np.isfinite(script_args.descent_rtol) or script_args.descent_rtol < 0.0:
        raise ValueError("descent_rtol must be finite and non-negative.")
    if script_args.lambda_solver not in LAMBDA_SOLVERS:
        raise ValueError(
            "lambda_solver must be one of: "
            + ", ".join(sorted(LAMBDA_SOLVERS))
            + "."
        )
    if script_args.lambda_normalization not in {"none", "global_mean"}:
        raise ValueError("lambda_normalization must be either 'none' or 'global_mean'.")
    if (
        not np.isfinite(script_args.lambda_min)
        or script_args.lambda_min < 0.0
        or script_args.lambda_min >= 1.0 / len(smoothness)
    ):
        raise ValueError("lambda_min must be finite and in [0, 1 / num_objectives).")
    if (
        script_args.lambda_entropy_tau is None
        or not np.isfinite(script_args.lambda_entropy_tau)
        or script_args.lambda_entropy_tau < 0.0
    ):
        raise ValueError("lambda_entropy_tau must be finite and non-negative.")
    if (
        script_args.lambda_diversity_strength is None
        or not np.isfinite(script_args.lambda_diversity_strength)
        or script_args.lambda_diversity_strength < 0.0
    ):
        raise ValueError("lambda_diversity_strength must be finite and non-negative.")
    if script_args.lambda_diversity_grid_points is None or script_args.lambda_diversity_grid_points < 2:
        raise ValueError("lambda_diversity_grid_points must be at least 2.")
    if script_args.lambda_diversity_recent_window is None or script_args.lambda_diversity_recent_window < 1:
        raise ValueError("lambda_diversity_recent_window must be at least 1.")
    if script_args.lambda_projection_dim is None:
        script_args.lambda_projection_dim = 0
    if script_args.lambda_projection_dim < 0:
        raise ValueError("lambda_projection_dim must be non-negative.")
    if script_args.lambda_projection_seed is None:
        script_args.lambda_projection_seed = 0
    if script_args.agreement_ratio is not None:
        if (
            not np.isfinite(script_args.agreement_ratio)
            or script_args.agreement_ratio < 0.0
            or script_args.agreement_ratio > 1.0
        ):
            raise ValueError("agreement_ratio must be finite and in [0, 1].")
        if script_args.consistent_preferences_only:
            raise ValueError(
                "agreement_ratio cannot be combined with consistent_preferences_only; "
                "set ADAPTIVE_CONSISTENT_PREFERENCES_ONLY=False."
            )
    if script_args.consistent_preferences_only:
        os.environ["SAFE_RLHF_AGREEMENT_ONLY"] = "1"
    if script_args.update_rule == "adam":
        script_args.update_rule = "adamw"
    if script_args.update_rule not in {"adamw", "t_map"}:
        raise ValueError("update_rule must be either 'adamw' or 't_map'.")
    if script_args.bundle_update_mode not in {"lambda_aware", "replace_source", "append"}:
        raise ValueError("bundle_update_mode must be one of: 'lambda_aware', 'replace_source', 'append'.")
    theory_adam_update = (
        script_args.algorithm_mode == "theory"
        and script_args.update_rule == "adamw"
        and script_args.bundle_update_mode == "append"
    )
    if script_args.algorithm_mode == "theory" and script_args.update_rule != "t_map" and not theory_adam_update:
        warnings.warn(
            "algorithm_mode='theory' is intended for update_rule='t_map'. "
            "Continuing with the explicitly configured update_rule.",
            RuntimeWarning,
            stacklevel=2,
        )
    if script_args.algorithm_mode == "theory" and script_args.bundle_update_mode != "append":
        warnings.warn(
            "algorithm_mode='theory' is intended for bundle_update_mode='append'. "
            "Continuing with the explicitly configured bundle_update_mode.",
            RuntimeWarning,
            stacklevel=2,
        )
    if script_args.update_rule == "t_map" and script_args.update_data_source == "oracle":
        warnings.warn(
            "update_data_source='oracle' is ignored for update_rule='t_map'; "
            "using update_data_source='train' in the run metadata.",
            RuntimeWarning,
            stacklevel=2,
        )
        script_args.update_data_source = "train"
    if script_args.update_data_source not in {"train", "oracle"}:
        raise ValueError("update_data_source must be either 'train' or 'oracle'.")
    if script_args.update_data_source == "oracle" and script_args.update_rule != "adamw":
        raise ValueError("update_data_source='oracle' is currently implemented for AdamW updates only.")
    if not np.isfinite(script_args.lambda_match_tol) or script_args.lambda_match_tol < 0.0:
        raise ValueError("lambda_match_tol must be finite and non-negative.")
    if script_args.lambda_stall_patience is None:
        script_args.lambda_stall_patience = 0
    if script_args.lambda_stall_patience < 0:
        raise ValueError("lambda_stall_patience must be non-negative.")
    if script_args.lambda_stall_abs_delta is None or not np.isfinite(script_args.lambda_stall_abs_delta):
        raise ValueError("lambda_stall_abs_delta must be finite.")
    if script_args.lambda_stall_abs_delta < 0.0:
        raise ValueError("lambda_stall_abs_delta must be non-negative.")
    if script_args.lambda_stall_rel_delta is None or not np.isfinite(script_args.lambda_stall_rel_delta):
        raise ValueError("lambda_stall_rel_delta must be finite.")
    if script_args.lambda_stall_rel_delta < 0.0:
        raise ValueError("lambda_stall_rel_delta must be non-negative.")
    if script_args.lambda_stall_cooldown is None:
        script_args.lambda_stall_cooldown = 1
    if script_args.lambda_stall_cooldown < 0:
        raise ValueError("lambda_stall_cooldown must be non-negative.")
    if script_args.lambda_stall_grid_points is None or script_args.lambda_stall_grid_points < 2:
        raise ValueError("lambda_stall_grid_points must be at least 2.")
    if script_args.lambda_stall_match_tol is None or not np.isfinite(script_args.lambda_stall_match_tol):
        raise ValueError("lambda_stall_match_tol must be finite.")
    if script_args.lambda_stall_match_tol < 0.0:
        raise ValueError("lambda_stall_match_tol must be non-negative.")
    if script_args.bundle_warm_start_steps is None:
        script_args.bundle_warm_start_steps = 0
    if script_args.bundle_warm_start_steps < 0:
        raise ValueError("bundle_warm_start_steps must be non-negative.")
    warm_start_lambdas = parse_lambda_schedule(
        script_args.bundle_warm_start_lambdas,
        len(smoothness),
    )
    if script_args.bundle_warm_start_steps > 0 and script_args.update_rule != "adamw":
        raise ValueError("bundle warm-start is currently implemented for AdamW updates only.")
    if script_args.per_objective_batch_size is None or script_args.per_objective_batch_size < 1:
        raise ValueError("per_objective_batch_size must be at least 1.")
    if script_args.gradient_accumulation_steps is None or script_args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    if not np.isfinite(script_args.warmup_ratio) or script_args.warmup_ratio < 0.0:
        raise ValueError("warmup_ratio must be finite and non-negative.")
    if script_args.max_grad_norm is not None and script_args.max_grad_norm < 0.0:
        raise ValueError("max_grad_norm must be non-negative when provided.")
    if script_args.require_ipopt and script_args.lambda_solver == "ipopt" and not ipopt_available():
        raise RuntimeError(
            "IPOPT was required for adaptive bundle lambda selection, but "
            "cyipopt/IPOPT is unavailable. Install IPOPT + cyipopt on the "
            "training machine before running this experiment. "
            f"Import error: {ipopt_import_error()!r}"
        )
    if (
        script_args.update_rule == "adamw"
        and script_args.prune_inner
        and script_args.bundle_update_mode != "append"
    ):
        warnings.warn(
            "prune_inner only applies to append bundle updates. "
            "It will not prune lambda_aware or replace_source updates.",
            RuntimeWarning,
            stacklevel=2,
        )
    if script_args.max_bundle_size is None:
        script_args.max_bundle_size = 0
    if script_args.max_bundle_size < 0:
        raise ValueError("max_bundle_size must be non-negative; use 0 to disable the cap.")
    if script_args.max_bundle_size == 1:
        warnings.warn(
            "max_bundle_size=1 leaves no room beyond the initial bundle point. "
            "Use a larger cap such as 20 or 25 for adaptive path experiments.",
            RuntimeWarning,
            stacklevel=2,
        )
    valid_bundle_cap_modes = {
        "replace_active_if_better",
        "global_swap_if_better",
        "global_swap",
    }
    if script_args.bundle_cap_mode not in valid_bundle_cap_modes:
        raise ValueError(
            "bundle_cap_mode must be one of: "
            + ", ".join(sorted(valid_bundle_cap_modes))
        )
    if script_args.max_bundle_size > 0 and script_args.bundle_update_mode != "append":
        warnings.warn(
            "max_bundle_size currently only applies to bundle_update_mode='append'.",
            RuntimeWarning,
            stacklevel=2,
        )
    if (
        script_args.max_bundle_size > 0
        and script_args.bundle_update_mode == "append"
        and not script_args.prune_inner
    ):
        warnings.warn(
            "max_bundle_size is designed for append updates with prune_inner=True; "
            "without pruning, multiple candidates may be appended before the cap is checked.",
            RuntimeWarning,
            stacklevel=2,
        )

    print_local_main("loading model...")
    device_kwargs = {}
    if torch.cuda.is_available() and not param_sharding_enabled():
        device_kwargs["device_map"] = {"": Accelerator().local_process_index}
    sft_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **device_kwargs,
    )
    sft_model.config.update({
        "use_cache": False,
        "pad_token_id": sft_model.config.eos_token_id,
    })

    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not script_args.dataset_caching:
        from datasets import disable_caching
        disable_caching()

    better_rdp = DATASET_CONFIGS[script_args.better_dataset_name](
        prompt_template=script_args.prompt_template,
        sanity_check=script_args.sanity_check,
    )
    safer_rdp = DATASET_CONFIGS[script_args.safer_dataset_name](
        prompt_template=script_args.prompt_template,
        sanity_check=script_args.sanity_check,
    )

    print_local_main("loading and subsampling objective datasets...")
    if script_args.agreement_ratio is not None:
        better_pool, safer_pool = make_controlled_agreement_pools(
            better_rdp,
            safer_rdp,
            split="train",
            size=script_args.train_subset_size_per_objective,
            agreement_ratio=float(script_args.agreement_ratio),
            seed=script_args.seed,
        )
    else:
        better_pool = select_subset(
            better_rdp.get_preference_dataset(split="train"),
            script_args.train_subset_size_per_objective,
            script_args.seed,
            "helpful/better train pool",
        )
        safer_pool = select_subset(
            safer_rdp.get_preference_dataset(split="train"),
            script_args.train_subset_size_per_objective,
            script_args.seed if script_args.shared_objective_subset else script_args.seed + 1,
            "harmless/safer train pool",
        )

    print_local_main("preprocessing objective datasets...")
    better_train = preprocess_preference_dataset(
        better_pool,
        tokenizer,
        script_args.max_length,
        script_args.num_proc,
    )
    safer_train = preprocess_preference_dataset(
        safer_pool,
        tokenizer,
        script_args.max_length,
        script_args.num_proc,
    )
    better_eval = preprocess_preference_dataset(
        better_rdp.get_preference_dataset(split="validation"),
        tokenizer,
        script_args.max_length,
        script_args.num_proc,
    )

    data_collator = MODPODataCollatorWithPadding(tokenizer)
    better_oracle = select_subset(
        better_train,
        script_args.oracle_subset_size_per_objective,
        script_args.seed + 2,
        "helpful/better fixed oracle",
    )
    safer_oracle = select_subset(
        safer_train,
        script_args.oracle_subset_size_per_objective,
        (
            script_args.seed + 2
            if script_args.shared_objective_subset or script_args.agreement_ratio is not None
            else script_args.seed + 3
        ),
        "harmless/safer fixed oracle",
    )
    objective_batch_groups = {
        "helpful": make_fixed_batches(better_oracle, script_args.oracle_batch_size, data_collator),
        "harmless": make_fixed_batches(safer_oracle, script_args.oracle_batch_size, data_collator),
    }

    trainer = AdaptiveBundleMODPOTrainer(
        model=sft_model,
        beta=script_args.beta,
        args=script_args.training_args,
        train_dataset=better_train,
        eval_dataset=better_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        peft_config=script_args.peft_config,
        max_length=script_args.max_length,
        num_proc=script_args.num_proc,
        generate_during_eval=False,
    )
    if Accelerator().is_local_main_process and script_args.peft_config:
        trainer.model.print_trainable_parameters()
    trainer.model.train()

    helpful_loader = CyclingLoader(DataLoader(
        better_train,
        batch_size=script_args.per_objective_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator,
    ))
    harmless_loader = CyclingLoader(DataLoader(
        safer_train,
        batch_size=script_args.per_objective_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator,
    ))
    trainable_params = [param for param in trainer.model.parameters() if param.requires_grad]
    optimizer = None
    scheduler = None
    if script_args.update_rule == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=script_args.training_args.learning_rate,
            weight_decay=script_args.weight_decay,
        )
        total_warm_start_steps = script_args.bundle_warm_start_steps * len(warm_start_lambdas)
        total_optimizer_steps = max(1, script_args.max_outer * script_args.max_inner + total_warm_start_steps)
        warmup_steps = int(total_optimizer_steps * script_args.warmup_ratio)
        scheduler = get_scheduler(
            script_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
        optimizer.zero_grad(set_to_none=True)

    def oracle_call():
        return trainer.multi_objective_gradient_oracle_over_batches(
            objective_batch_groups,
            as_numpy=True,
        )

    resume_state_files = resume_paths(
        script_args.training_args.output_dir,
        script_args.resume_state_path,
    )

    config_path = os.path.join(script_args.training_args.output_dir, "adaptive_config.json")
    with open(config_path, "w") as handle:
        json.dump(asdict(script_args), handle, indent=2, default=str)

    num_objectives = 2
    objective_names = ["helpful", "harmless"]
    initial_lambda = np.full(num_objectives, 1.0 / num_objectives, dtype=np.float64)
    initial_oracle_path = os.path.join(script_args.training_args.output_dir, "adaptive_initial_oracle.json")

    history_path = os.path.join(script_args.training_args.output_dir, "adaptive_history.jsonl")
    if (
        os.path.exists(history_path)
        and script_args.training_args.overwrite_output_dir
        and not script_args.resume
    ):
        os.remove(history_path)
    solution_path_file = os.path.join(script_args.training_args.output_dir, "adaptive_solution_path.json")
    prefix_bundle_sizes = parse_positive_int_set(
        script_args.prefix_save_bundle_sizes,
        name="prefix_save_bundle_sizes",
    )
    prefix_state_dir = script_args.prefix_state_dir or os.path.join(
        script_args.training_args.output_dir,
        "prefix_states",
    )
    saved_prefix_bundle_sizes = set()

    stop_reason = "max_outer"
    stopped_outer = None
    final_gn_star = None
    best_gn_star = float("inf")
    target_hit = None

    if script_args.resume:
        prefix_history_file = f"{resume_state_files['base']}_history.jsonl"
        if os.path.exists(prefix_history_file) and (
            script_args.training_args.overwrite_output_dir or not os.path.exists(history_path)
        ):
            copy_if_exists(prefix_history_file, history_path)
        prefix_solution_file = f"{resume_state_files['base']}_solution_path.json"
        if os.path.exists(prefix_solution_file) and (
            script_args.training_args.overwrite_output_dir
            or not os.path.exists(solution_path_file)
        ):
            copy_if_exists(prefix_solution_file, solution_path_file)
        with np.load(resume_state_files["npz"]) as resume_arrays:
            if "points" not in resume_arrays:
                raise ValueError(f"Resume file missing points: {resume_state_files['npz']}")
            resume_d = int(resume_arrays["points"].shape[1])
        bundle = FirstOrderBundle(
            K=num_objectives,
            d=resume_d,
            L=np.asarray(smoothness, dtype=np.float64),
            dtype=np.dtype(script_args.bundle_dtype),
            lambda_projection_dim=(
                int(script_args.lambda_projection_dim)
                if script_args.lambda_projection_dim and script_args.lambda_projection_dim > 0
                else None
            ),
            lambda_projection_seed=int(script_args.lambda_projection_seed),
        )
        metadata, bundle_lambdas, lambda_history, lambda_stall_states, prev_lam = load_resume_state(
            resume_state_files,
            bundle,
        )
        total_parameter_updates = int(metadata["total_parameter_updates"])
        total_oracle_gradient_evals = int(metadata["total_oracle_gradient_evals"])
        total_inner_steps = int(metadata["total_inner_steps"])
        current_l_scale = float(metadata["current_l_scale"])
        safeguard_violations = int(metadata["safeguard_violations"])
        safeguard_warned = bool(metadata.get("safeguard_warned", False))
        relative_outer_reference_gn = metadata.get("relative_outer_reference_gn")
        elapsed_offset_seconds = float(metadata.get("elapsed_wall_seconds") or 0.0)
        if elapsed_offset_seconds > 0.0:
            run_start_time = time.perf_counter() - elapsed_offset_seconds
        start_outer = int(metadata["next_outer"])
        if os.path.exists(solution_path_file):
            with open(solution_path_file) as handle:
                solution_path = json.load(handle)
        else:
            solution_path = {
                "version": 1,
                "algorithm_mode": script_args.algorithm_mode,
                "update_rule": script_args.update_rule,
                "bundle_update_mode": script_args.bundle_update_mode,
                "lambda_solver": script_args.lambda_solver,
                "stop_rule": script_args.stop_rule,
                "epsilon": script_args.epsilon,
                "relative_rho": script_args.relative_rho,
                "num_objectives": num_objectives,
                "objective_names": objective_names,
                "lambda_projection": bundle.lambda_projection_info(),
                "anchors": [],
                "termination": None,
            }
        if optimizer is not None and os.path.exists(resume_state_files["optimizer"]):
            opt_state = torch.load(resume_state_files["optimizer"], map_location="cpu")
            optimizer.load_state_dict(opt_state["optimizer"])
            if scheduler is not None and "scheduler" in opt_state:
                scheduler.load_state_dict(opt_state["scheduler"])
        trainer.set_trainable_parameter_vector(bundle.points[-1])
        print_local_main(
            f"resuming adaptive bundle from outer={start_outer} "
            f"updates={total_parameter_updates} bundle={bundle.m}"
        )
    else:
        print_local_main("building initial bundle point...")
        initial = oracle_call()
        total_parameter_updates = 0
        total_oracle_gradient_evals = 1
        bundle = FirstOrderBundle(
            K=num_objectives,
            d=int(initial["x"].shape[0]),
            L=np.asarray(smoothness, dtype=np.float64),
            dtype=np.dtype(script_args.bundle_dtype),
            lambda_projection_dim=(
                int(script_args.lambda_projection_dim)
                if script_args.lambda_projection_dim and script_args.lambda_projection_dim > 0
                else None
            ),
            lambda_projection_seed=int(script_args.lambda_projection_seed),
        )
        bundle.add(initial["x"], initial["fvals"], initial["grads"])
        bundle_lambdas = [initial_lambda.copy()]
        lambda_stall_states = []
        with open(initial_oracle_path, "w") as handle:
            json.dump(
                {
                    "gradient_eval": 1,
                    "oracle_gradient_eval": total_oracle_gradient_evals,
                    "parameter_update": total_parameter_updates,
                    "parameter_updates": total_parameter_updates,
                    "objective_gradient_evals": num_objectives * total_parameter_updates,
                    "elapsed_wall_seconds": time.perf_counter() - run_start_time,
                    "num_objectives": num_objectives,
                    "lambda_projection": bundle.lambda_projection_info(),
                    "lambda": initial_lambda.tolist(),
                    "fvals": initial["fvals"].tolist(),
                    "objective_names": initial["objective_names"],
                },
                handle,
                indent=2,
            )
        solution_path = {
            "version": 1,
            "algorithm_mode": script_args.algorithm_mode,
            "update_rule": script_args.update_rule,
            "bundle_update_mode": script_args.bundle_update_mode,
            "lambda_solver": script_args.lambda_solver,
            "stop_rule": script_args.stop_rule,
            "epsilon": script_args.epsilon,
            "relative_rho": script_args.relative_rho,
            "num_objectives": num_objectives,
            "objective_names": objective_names,
            "lambda_projection": bundle.lambda_projection_info(),
            "anchors": [
                {
                    "phase": "initial",
                    "outer": 0,
                    "lambda": initial_lambda.tolist(),
                    "bundle_indices": [0],
                    "new_bundle_indices": [0],
                    "updated_bundle_indices": [],
                    "M_t": 0,
                    "parameter_updates_before": 0,
                    "parameter_updates_after": 0,
                    "oracle_gradient_evals_before": 1,
                    "oracle_gradient_evals_after": 1,
                    "objective_gradient_evals_before": 0,
                    "objective_gradient_evals_after": 0,
                    "gn_at_lambda": gn_value_at_lambda(
                        bundle,
                        initial_lambda,
                        lambda_normalization="none",
                        lambda_min=script_args.lambda_min,
                        use_projection=False,
                    ),
                    "fvals": initial["fvals"].tolist(),
                }
            ],
            "termination": None,
        }
        relative_outer_reference_gn = None
        prev_lam = None
        lambda_history = []
        total_inner_steps = 0
        current_l_scale = float(script_args.l_scale)
        safeguard_violations = 0
        safeguard_warned = False
        start_outer = 1

    if (not script_args.resume) and script_args.bundle_warm_start_steps > 0 and warm_start_lambdas:
        print_local_main(
            "running bundle warm-start with lambdas="
            f"{[np.round(lam, 4).tolist() for lam in warm_start_lambdas]} "
            f"steps={script_args.bundle_warm_start_steps}"
        )
        for warm_idx, lam in enumerate(warm_start_lambdas, start=1):
            lam = np.asarray(lam, dtype=np.float64)
            warm_inner_records = []
            warm_anchor_bundle_indices = []
            warm_new_bundle_indices = []
            warm_updated_bundle_indices = []
            warm_parameter_updates_before = total_parameter_updates
            warm_oracle_gradient_evals_before = total_oracle_gradient_evals
            warm_objective_gradient_evals_before = num_objectives * total_parameter_updates
            warm_elapsed_wall_seconds_before = time.perf_counter() - run_start_time
            for warm_step in range(1, script_args.bundle_warm_start_steps + 1):
                lambda_match_idx = find_matching_lambda(
                    bundle_lambdas,
                    lam,
                    float(script_args.lambda_match_tol),
                )
                lambda_known = lambda_match_idx is not None
                source_idx = int(lambda_match_idx) if lambda_known else 0
                source_grad_norm_sq = scalarized_gradient_norm_sq(bundle, source_idx, lam)
                source_f_lambda = scalarized_objective(bundle.fvals[source_idx], lam)
                trainer.set_trainable_parameter_vector(bundle.points[source_idx])

                update_stats = backward_weighted_dpo_update(
                    trainer,
                    lam,
                    script_args.update_data_source,
                    helpful_loader,
                    harmless_loader,
                    objective_batch_groups,
                    script_args.gradient_accumulation_steps,
                )
                if script_args.max_grad_norm and script_args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, script_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                oracle = oracle_call()
                f_lambda_new = scalarized_objective(oracle["fvals"], lam)
                descent_slack = f_lambda_new - source_f_lambda
                descent_tolerance = (
                    script_args.descent_atol
                    + script_args.descent_rtol * (1.0 + abs(source_f_lambda))
                )
                candidate_accepted = descent_slack <= descent_tolerance
                bundle_replaced = False
                bundle_appended = False
                bundle_index = None
                new_bundle_index = None
                updated_bundle_index = None
                if candidate_accepted:
                    if lambda_match_idx is None:
                        bundle.add(oracle["x"], oracle["fvals"], oracle["grads"])
                        bundle_lambdas.append(lam.copy())
                        bundle_appended = True
                        bundle_index = bundle.m - 1
                        new_bundle_index = bundle_index
                        bundle_action = "warm_start_add_lambda"
                    else:
                        bundle.replace(lambda_match_idx, oracle["x"], oracle["fvals"], oracle["grads"])
                        bundle_lambdas[lambda_match_idx] = lam.copy()
                        bundle_replaced = True
                        bundle_index = int(lambda_match_idx)
                        updated_bundle_index = bundle_index
                        bundle_action = "warm_start_replace_lambda_representative"
                else:
                    trainer.set_trainable_parameter_vector(bundle.points[source_idx])
                    bundle_action = "warm_start_reject"
                warm_anchor_bundle_indices.append(bundle_index)
                warm_new_bundle_indices.append(new_bundle_index)
                warm_updated_bundle_indices.append(updated_bundle_index)

                total_parameter_updates += 1
                total_oracle_gradient_evals += 1
                total_inner_steps += 1
                warm_inner_records.append({
                    "inner_step": warm_step,
                    "update_rule": script_args.update_rule,
                    "gradient_eval": total_oracle_gradient_evals,
                    "oracle_gradient_eval": total_oracle_gradient_evals,
                    "parameter_update": total_parameter_updates,
                    "parameter_updates": total_parameter_updates,
                    "objective_gradient_evals": num_objectives * total_parameter_updates,
                    "elapsed_wall_seconds": time.perf_counter() - run_start_time,
                    "bundle_size": bundle.m,
                    "bundle_update_mode": script_args.bundle_update_mode,
                    "bundle_action": bundle_action,
                    "bundle_replaced": bundle_replaced,
                    "bundle_appended": bundle_appended,
                    "bundle_index": bundle_index,
                    "new_bundle_index": new_bundle_index,
                    "updated_bundle_index": updated_bundle_index,
                    "candidate_accepted": candidate_accepted,
                    "lambda_known": lambda_known,
                    "lambda_match_idx": lambda_match_idx,
                    "lambda_match_tol": script_args.lambda_match_tol,
                    "source_idx": source_idx,
                    "source_grad_norm_sq": source_grad_norm_sq,
                    "source_f_lambda": source_f_lambda,
                    "f_lambda_new": f_lambda_new,
                    "descent_slack": descent_slack,
                    "descent_tolerance": descent_tolerance,
                    "descent_atol": script_args.descent_atol,
                    "descent_rtol": script_args.descent_rtol,
                    "train_loss": update_stats["train_loss"],
                    "train_helpful_loss": update_stats["train_helpful_loss"],
                    "train_harmless_loss": update_stats["train_harmless_loss"],
                    "update_data_source": script_args.update_data_source,
                    "update_batches": update_stats["update_batches"],
                    "update_examples": update_stats["update_examples"],
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "fvals": oracle["fvals"].tolist(),
                })

            lambda_history.append(lam.copy())
            warm_gn_after = gn_value_at_lambda(
                bundle,
                lam,
                lambda_normalization="none",
                lambda_min=script_args.lambda_min,
                use_projection=False,
            )
            warm_elapsed_wall_seconds_after = time.perf_counter() - run_start_time
            solution_path["anchors"].append({
                "phase": "warm_start",
                "outer": 0,
                "warm_start_index": warm_idx,
                "lambda": lam.tolist(),
                "bundle_indices": unique_ints(warm_anchor_bundle_indices),
                "new_bundle_indices": unique_ints(warm_new_bundle_indices),
                "updated_bundle_indices": unique_ints(warm_updated_bundle_indices),
                "M_t": int(script_args.bundle_warm_start_steps),
                "inner_stop_reached": False,
                "inner_stop_reason": "warm_start_fixed_steps",
                "parameter_updates_before": warm_parameter_updates_before,
                "parameter_updates_after": total_parameter_updates,
                "oracle_gradient_evals_before": warm_oracle_gradient_evals_before,
                "oracle_gradient_evals_after": total_oracle_gradient_evals,
                "objective_gradient_evals_before": warm_objective_gradient_evals_before,
                "objective_gradient_evals_after": num_objectives * total_parameter_updates,
                "elapsed_wall_seconds_before": warm_elapsed_wall_seconds_before,
                "elapsed_wall_seconds_after": warm_elapsed_wall_seconds_after,
                "gn_at_lambda_after": warm_gn_after,
            })
            lambda_diagnostics = {
                "lambda_solver": script_args.lambda_solver,
                "lambda_normalization": script_args.lambda_normalization,
                "lambda_min": script_args.lambda_min,
                "lambda_entropy_tau": 0.0,
                "lambda_entropy": float(-np.sum(
                    np.clip(lam, np.finfo(np.float64).tiny, 1.0)
                    * np.log(np.clip(lam, np.finfo(np.float64).tiny, 1.0))
                )),
                "lambda_before_diversity": lam.tolist(),
                "lambda_diversity": {"enabled": False, "score": None},
                "lambda_projection": bundle.lambda_projection_info(),
                "gradient": bundle_gradient_diagnostics(bundle),
                "gn_grid": gn_grid_diagnostics(
                    bundle,
                    lambda_normalization="none",
                    lambda_min=script_args.lambda_min,
                    use_projection=False,
                ),
                "selection_gn_grid": gn_grid_diagnostics(
                    bundle,
                    lambda_normalization=script_args.lambda_normalization,
                    lambda_min=script_args.lambda_min,
                    use_projection=True,
                ),
            }
            warm_record = {
                "phase": "warm_start",
                "outer": 0,
                "warm_start_index": warm_idx,
                "lambda": lam.tolist(),
                "update_rule": script_args.update_rule,
                "bundle_update_mode": script_args.bundle_update_mode,
                "update_data_source": script_args.update_data_source,
                "lambda_solver": script_args.lambda_solver,
                "lambda_normalization": script_args.lambda_normalization,
                "lambda_min": script_args.lambda_min,
                "lambda_entropy_tau": 0.0,
                "lambda_entropy": lambda_diagnostics["lambda_entropy"],
                "lambda_before_diversity": lam.tolist(),
                "lambda_diversity": {"enabled": False, "score": None},
                "lambda_projection": bundle.lambda_projection_info(),
                "gn_star": gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization="none",
                    lambda_min=script_args.lambda_min,
                    use_projection=False,
                ),
                "raw_gn_star": gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization="none",
                    lambda_min=script_args.lambda_min,
                    use_projection=False,
                ),
                "raw_gn_star_lambda": lam.tolist(),
                "raw_gn_at_selected_lambda": gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization="none",
                    lambda_min=script_args.lambda_min,
                    use_projection=False,
                ),
                "lambda_selection_objective_star": gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization=script_args.lambda_normalization,
                    lambda_min=script_args.lambda_min,
                    use_projection=True,
                ),
                "lambda_selection_gn_star": gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization=script_args.lambda_normalization,
                    lambda_min=script_args.lambda_min,
                    use_projection=True,
                ),
                "lambda_selection_gn_at_selected_lambda": gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization=script_args.lambda_normalization,
                    lambda_min=script_args.lambda_min,
                    use_projection=True,
                ),
                "gradient_evals_before": warm_oracle_gradient_evals_before,
                "gradient_evals_after": total_oracle_gradient_evals,
                "oracle_gradient_evals_before": warm_oracle_gradient_evals_before,
                "oracle_gradient_evals_after": total_oracle_gradient_evals,
                "parameter_updates_before": warm_parameter_updates_before,
                "parameter_updates_after": total_parameter_updates,
                "objective_gradient_evals_before": warm_objective_gradient_evals_before,
                "objective_gradient_evals_after": num_objectives * total_parameter_updates,
                "elapsed_wall_seconds": warm_elapsed_wall_seconds_before,
                "elapsed_wall_seconds_before": warm_elapsed_wall_seconds_before,
                "elapsed_wall_seconds_after": warm_elapsed_wall_seconds_after,
                "bundle_size": bundle.m,
                "bundle_size_before": None,
                "bundle_lambda_count": len(bundle_lambdas),
                "num_objectives": num_objectives,
                "objective_names": objective_names,
                "lambda_diagnostics": lambda_diagnostics,
                "total_inner_steps": total_inner_steps,
                "inner": warm_inner_records,
            }
            save_jsonl(history_path, warm_record)
            print_local_main(
                f"warm_start={warm_idx} lambda={np.round(lam, 4).tolist()} "
                f"updates={total_parameter_updates} bundle={bundle.m}"
            )

    for outer in range(start_outer, script_args.max_outer + 1):
        outer_wall_t0 = time.perf_counter()
        lambda_entropy_tau_current = float(script_args.lambda_entropy_tau)
        blocked_lambdas = active_stalled_lambdas(lambda_stall_states, outer)
        lambda_solver_t0 = time.perf_counter()
        selection_pc_value, gn_lam = maximise_gn(
            bundle,
            prev_lam=prev_lam,
            max_starts=script_args.lambda_max_starts,
            solver=script_args.lambda_solver,
            require_ipopt=script_args.require_ipopt,
            lambda_normalization=script_args.lambda_normalization,
            lambda_min=script_args.lambda_min,
            use_projection=True,
            entropy_tau=lambda_entropy_tau_current,
        )
        timing_lambda_solver_seconds = time.perf_counter() - lambda_solver_t0
        lam, lambda_diversity = diversify_lambda_on_grid(
            bundle,
            gn_lam,
            lambda_history,
            lambda_normalization=script_args.lambda_normalization,
            lambda_min=script_args.lambda_min,
            num_points=script_args.lambda_diversity_grid_points,
            diversity_strength=script_args.lambda_diversity_strength,
            recent_window=script_args.lambda_diversity_recent_window,
            use_projection=True,
        )
        lambda_stall_selection = {
            "enabled": script_args.lambda_stall_patience > 0,
            "selected_by": "gn",
            "active_blocked_lambdas": [blocked.tolist() for blocked in blocked_lambdas],
            "base_lambda": lam.tolist(),
            "chosen_lambda": lam.tolist(),
            "match_tol": float(script_args.lambda_stall_match_tol),
            "patience": int(script_args.lambda_stall_patience),
            "cooldown": int(script_args.lambda_stall_cooldown),
        }
        if (
            script_args.lambda_stall_patience > 0
            and lambda_is_blocked(lam, blocked_lambdas, float(script_args.lambda_stall_match_tol))
        ):
            unblocked_lam, stall_grid_info = choose_unblocked_lambda_from_grid(
                bundle,
                blocked_lambdas,
                outer=outer,
                lambda_normalization=script_args.lambda_normalization,
                lambda_min=script_args.lambda_min,
                entropy_tau=lambda_entropy_tau_current,
                num_points=script_args.lambda_stall_grid_points,
                match_tol=script_args.lambda_stall_match_tol,
                use_projection=True,
            )
            lambda_stall_selection.update(stall_grid_info)
            lambda_stall_selection["base_lambda"] = lam.tolist()
            if unblocked_lam is not None:
                lam = unblocked_lam
                lambda_stall_selection["chosen_lambda"] = lam.tolist()
        if script_args.lambda_normalization == "none" and not bundle.lambda_projection_active:
            pc_value = selection_pc_value
            raw_pc_lam = gn_lam.copy()
            timing_lambda_raw_solver_seconds = 0.0
        else:
            lambda_raw_solver_t0 = time.perf_counter()
            pc_value, raw_pc_lam = maximise_gn(
                bundle,
                prev_lam=prev_lam,
                max_starts=script_args.lambda_max_starts,
                solver=script_args.lambda_solver,
                require_ipopt=script_args.require_ipopt,
                lambda_normalization="none",
                lambda_min=script_args.lambda_min,
                use_projection=False,
                entropy_tau=0.0,
            )
            timing_lambda_raw_solver_seconds = time.perf_counter() - lambda_raw_solver_t0
        raw_gn_at_selected_lam = gn_value_at_lambda(
            bundle,
            lam,
            lambda_normalization="none",
            lambda_min=script_args.lambda_min,
            use_projection=False,
        )
        selection_gn_at_selected_lam = gn_value_at_lambda(
            bundle,
            lam,
            lambda_normalization=script_args.lambda_normalization,
            lambda_min=script_args.lambda_min,
            use_projection=True,
        )
        lam_safe = np.clip(lam.astype(np.float64, copy=False), np.finfo(np.float64).tiny, 1.0)
        lambda_entropy = float(-np.sum(lam_safe * np.log(lam_safe)))
        gn_certificate_type = lambda_certificate_type(
            script_args,
            bundle,
            lambda_entropy_tau_current,
        )
        prev_lam = lam.copy()
        lambda_history.append(lam.copy())
        lambda_diagnostics_t0 = time.perf_counter()
        gradient_diagnostics = bundle_gradient_diagnostics(bundle)
        raw_gn_grid_diagnostics = gn_grid_diagnostics(
            bundle,
            lambda_normalization="none",
            lambda_min=script_args.lambda_min,
            use_projection=False,
        )
        selection_gn_grid_diagnostics = gn_grid_diagnostics(
            bundle,
            lambda_normalization=script_args.lambda_normalization,
            lambda_min=script_args.lambda_min,
            use_projection=True,
        )
        timing_lambda_diagnostics_seconds = time.perf_counter() - lambda_diagnostics_t0
        lambda_diagnostics = {
            "lambda_solver": script_args.lambda_solver,
            "gn_certificate_type": gn_certificate_type,
            "lambda_normalization": script_args.lambda_normalization,
            "lambda_min": script_args.lambda_min,
            "lambda_entropy_tau": lambda_entropy_tau_current,
            "lambda_entropy": lambda_entropy,
            "lambda_before_diversity": gn_lam.tolist(),
            "lambda_diversity": lambda_diversity,
            "lambda_stall_selection": lambda_stall_selection,
            "lambda_projection": bundle.lambda_projection_info(),
            "gradient": gradient_diagnostics,
            "gn_grid": raw_gn_grid_diagnostics,
            "selection_gn_grid": selection_gn_grid_diagnostics,
            "timing_lambda_solver_seconds": timing_lambda_solver_seconds,
            "timing_lambda_raw_solver_seconds": timing_lambda_raw_solver_seconds,
            "timing_lambda_diagnostics_seconds": timing_lambda_diagnostics_seconds,
        }
        if script_args.stop_rule == "relative" and relative_outer_reference_gn is None:
            relative_outer_reference_gn = pc_value
        outer_stop_threshold = stop_threshold(
            script_args,
            "outer",
            reference=relative_outer_reference_gn,
        )
        best_gn_star = min(best_gn_star, float(pc_value))
        best_gradient_norm = float(np.sqrt(max(best_gn_star, 0.0)))
        target_reached = bool(
            script_args.gn_target_norm is not None
            and best_gradient_norm <= float(script_args.gn_target_norm)
        )
        if stop_reached(pc_value, outer_stop_threshold) or target_reached:
            stop_reason = (
                "gn_target_norm_reached"
                if target_reached
                else f"{script_args.stop_rule}_outer_gn_below_threshold"
            )
            stopped_outer = outer
            final_gn_star = pc_value
            stop_elapsed_wall_seconds = time.perf_counter() - run_start_time
            timing_outer_total_seconds = time.perf_counter() - outer_wall_t0
            record = {
                "phase": "outer_stop",
                "outer": outer,
                "lambda": lam.tolist(),
                "update_rule": script_args.update_rule,
                "bundle_update_mode": script_args.bundle_update_mode,
                "update_data_source": script_args.update_data_source,
                "lambda_solver": script_args.lambda_solver,
                "lambda_normalization": script_args.lambda_normalization,
                "lambda_min": script_args.lambda_min,
                "lambda_entropy_tau": lambda_entropy_tau_current,
                "lambda_entropy": lambda_entropy,
                "lambda_before_diversity": gn_lam.tolist(),
                "lambda_diversity": lambda_diversity,
                "lambda_stall_selection": lambda_stall_selection,
                "lambda_stall_states": lambda_stall_states,
                "lambda_projection": bundle.lambda_projection_info(),
                "gn_certificate_type": gn_certificate_type,
                "gn_star": pc_value,
                "raw_gn_star": pc_value,
                "raw_gn_star_lambda": raw_pc_lam.tolist(),
                "raw_gn_at_selected_lambda": raw_gn_at_selected_lam,
                "lambda_selection_objective_star": selection_pc_value,
                "lambda_selection_gn_star": selection_pc_value,
                "lambda_selection_gn_at_selected_lambda": selection_gn_at_selected_lam,
                "stop_rule": script_args.stop_rule,
                "outer_stop_threshold": outer_stop_threshold,
                "relative_outer_reference_gn": relative_outer_reference_gn,
                "best_gn_star": best_gn_star,
                "best_gradient_norm": best_gradient_norm,
                "gn_target_norm": script_args.gn_target_norm,
                "target_reached": target_reached,
                "stop_reason": stop_reason,
                "gradient_evals_before": total_oracle_gradient_evals,
                "gradient_evals_after": total_oracle_gradient_evals,
                "oracle_gradient_evals_before": total_oracle_gradient_evals,
                "oracle_gradient_evals_after": total_oracle_gradient_evals,
                "parameter_updates_before": total_parameter_updates,
                "parameter_updates_after": total_parameter_updates,
                "objective_gradient_evals_before": num_objectives * total_parameter_updates,
                "objective_gradient_evals_after": num_objectives * total_parameter_updates,
                "elapsed_wall_seconds": stop_elapsed_wall_seconds,
                "elapsed_wall_seconds_before": stop_elapsed_wall_seconds,
                "elapsed_wall_seconds_after": stop_elapsed_wall_seconds,
                "timing_outer_total_seconds": timing_outer_total_seconds,
                "timing_lambda_solver_seconds": timing_lambda_solver_seconds,
                "timing_lambda_raw_solver_seconds": timing_lambda_raw_solver_seconds,
                "timing_lambda_diagnostics_seconds": timing_lambda_diagnostics_seconds,
                "timing_inner_total_seconds": 0.0,
                "timing_inner_update_seconds": 0.0,
                "timing_inner_oracle_seconds": 0.0,
                "timing_inner_accept_seconds": 0.0,
                "timing_prune_inner_seconds": 0.0,
                "timing_bundle_cap_seconds": 0.0,
                "timing_bundle_cap_solver_seconds": 0.0,
                "timing_bundle_cap_solver_calls": 0,
                "timing_other_overhead_seconds": max(
                    0.0,
                    timing_outer_total_seconds
                    - timing_lambda_solver_seconds
                    - timing_lambda_raw_solver_seconds
                    - timing_lambda_diagnostics_seconds,
                ),
                "bundle_size": bundle.m,
                "bundle_size_before": bundle.m,
                "bundle_lambda_count": len(bundle_lambdas),
                "num_objectives": num_objectives,
                "objective_names": objective_names,
                "lambda_diagnostics": lambda_diagnostics,
                "inner": [],
            }
            save_jsonl(history_path, record)
            if target_reached:
                target_hit = dict(record)
            reported_threshold = (
                float(script_args.gn_target_norm)
                if target_reached
                else float(outer_stop_threshold)
            )
            print_local_main(
                f"outer={outer} stopping: gn*={pc_value:.4e} "
                f"threshold={reported_threshold:.4e} "
                f"rule={'gn_target_norm' if target_reached else script_args.stop_rule}"
            )
            break

        bundle_size_before = bundle.m
        parameter_updates_before = total_parameter_updates
        oracle_gradient_evals_before = total_oracle_gradient_evals
        objective_gradient_evals_before = num_objectives * total_parameter_updates
        outer_elapsed_wall_seconds_before = time.perf_counter() - run_start_time
        l_scale_before_outer = current_l_scale
        safeguard_violations_before = safeguard_violations
        inner_records = []
        outer_anchor_bundle_indices = []
        outer_new_bundle_indices = []
        outer_updated_bundle_indices = []
        inner_gn_reference = raw_gn_at_selected_lam
        inner_stop_threshold = stop_threshold(
            script_args,
            "inner",
            reference=inner_gn_reference,
        )
        inner_stop_reached = False
        inner_stop_reason = "max_inner"
        inner_gn_at_lambda_after = inner_gn_reference
        M_t = 0
        timing_inner_total_seconds = 0.0
        timing_inner_update_seconds = 0.0
        timing_inner_oracle_seconds = 0.0
        timing_inner_accept_seconds = 0.0
        for inner_step in range(1, script_args.max_inner + 1):
            inner_wall_t0 = time.perf_counter()
            bundle_size_before_inner = bundle.m
            inner_update_t0 = time.perf_counter()
            l_scale_before_step = current_l_scale
            source_idx = bundle.m - 1
            source_grad_norm_sq = None
            source_f_lambda = None
            t_map_u_star = None
            L_lambda = None
            f_lambda_new = None
            descent_slack = None
            descent_tolerance = None
            safeguard_triggered = False
            candidate_accepted = True
            bundle_replaced = False
            bundle_appended = False
            bundle_action = "none"
            lambda_match_idx = None
            lambda_known = False
            train_loss_value = None
            train_helpful_loss = None
            train_harmless_loss = None
            update_batches = None
            update_examples = None
            learning_rate = None
            bundle_index = None
            new_bundle_index = None
            updated_bundle_index = None

            if script_args.update_rule == "t_map":
                x_new, source_idx, source_grad_norm_sq, t_map_u_star, L_lambda = t_map_step(
                    bundle,
                    lam,
                    L_scale=current_l_scale,
                )
                source_f_lambda = float(np.asarray(bundle.fvals[source_idx], dtype=np.float64) @ lam)
                trainer.set_trainable_parameter_vector(x_new)
            else:
                if script_args.bundle_update_mode == "lambda_aware":
                    lambda_match_idx = find_matching_lambda(
                        bundle_lambdas,
                        lam,
                        float(script_args.lambda_match_tol),
                    )
                    lambda_known = lambda_match_idx is not None
                    if lambda_known:
                        source_idx = int(lambda_match_idx)
                        source_grad_norm_sq = scalarized_gradient_norm_sq(bundle, source_idx, lam)
                    else:
                        source_idx, source_grad_norm_sq = active_gn_source(
                            bundle,
                            lam,
                            lambda_normalization=script_args.lambda_normalization,
                            lambda_min=script_args.lambda_min,
                            use_projection=True,
                        )
                    source_f_lambda = scalarized_objective(bundle.fvals[source_idx], lam)
                    trainer.set_trainable_parameter_vector(bundle.points[source_idx])
                elif script_args.bundle_update_mode == "replace_source":
                    source_idx, source_grad_norm_sq = active_gn_source(
                        bundle,
                        lam,
                        lambda_normalization=script_args.lambda_normalization,
                        lambda_min=script_args.lambda_min,
                        use_projection=True,
                    )
                    source_f_lambda = scalarized_objective(bundle.fvals[source_idx], lam)
                    trainer.set_trainable_parameter_vector(bundle.points[source_idx])
                else:
                    source_idx, source_grad_norm_sq = active_gn_source(
                        bundle,
                        lam,
                        lambda_normalization=script_args.lambda_normalization,
                        lambda_min=script_args.lambda_min,
                        use_projection=True,
                    )
                    source_f_lambda = scalarized_objective(bundle.fvals[source_idx], lam)
                    trainer.set_trainable_parameter_vector(bundle.points[source_idx])
                update_stats = backward_weighted_dpo_update(
                    trainer,
                    lam,
                    script_args.update_data_source,
                    helpful_loader,
                    harmless_loader,
                    objective_batch_groups,
                    script_args.gradient_accumulation_steps,
                )
                if script_args.max_grad_norm and script_args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, script_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                learning_rate = float(scheduler.get_last_lr()[0])
                train_loss_value = update_stats["train_loss"]
                train_helpful_loss = update_stats["train_helpful_loss"]
                train_harmless_loss = update_stats["train_harmless_loss"]
                update_batches = update_stats["update_batches"]
                update_examples = update_stats["update_examples"]

            inner_update_seconds = time.perf_counter() - inner_update_t0
            timing_inner_update_seconds += inner_update_seconds
            inner_oracle_t0 = time.perf_counter()
            oracle = oracle_call()
            inner_oracle_seconds = time.perf_counter() - inner_oracle_t0
            timing_inner_oracle_seconds += inner_oracle_seconds
            f_lambda_new = scalarized_objective(oracle["fvals"], lam)
            if script_args.update_rule == "t_map":
                descent_slack = f_lambda_new - t_map_u_star
                descent_tolerance = (
                    script_args.descent_atol
                    + script_args.descent_rtol * (1.0 + abs(t_map_u_star))
                )
                safeguard_triggered = descent_slack > descent_tolerance
                if safeguard_triggered:
                    safeguard_violations += 1
                    current_l_scale *= 2.0
                    if not safeguard_warned:
                        warnings.warn(
                            "Descent-lemma check failed: ADAPTIVE_SMOOTHNESS / "
                            "ADAPTIVE_L_SCALE underestimate local curvature. "
                            "The adaptive bundle runner is doubling L_scale and "
                            "continuing with a smaller T-map step size.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        safeguard_warned = True
                    if current_l_scale > 2.0 ** 60:
                        raise RuntimeError(
                            "Descent-lemma safeguard scaled L by more than 2^60. "
                            "The DPO objectives do not appear to satisfy the "
                            "current smoothness model along the iterates."
                        )
                candidate_accepted = not safeguard_triggered
            elif script_args.bundle_update_mode in {"lambda_aware", "replace_source"}:
                descent_slack = f_lambda_new - source_f_lambda
                descent_tolerance = (
                    script_args.descent_atol
                    + script_args.descent_rtol * (1.0 + abs(source_f_lambda))
                )
                candidate_accepted = descent_slack <= descent_tolerance

            if script_args.bundle_update_mode == "lambda_aware":
                if candidate_accepted:
                    if lambda_match_idx is None:
                        bundle.add(oracle["x"], oracle["fvals"], oracle["grads"])
                        bundle_lambdas.append(lam.copy())
                        bundle_appended = True
                        bundle_index = bundle.m - 1
                        new_bundle_index = bundle_index
                        bundle_action = "add_new_lambda"
                    else:
                        bundle.replace(lambda_match_idx, oracle["x"], oracle["fvals"], oracle["grads"])
                        bundle_lambdas[lambda_match_idx] = lam.copy()
                        bundle_replaced = True
                        bundle_index = int(lambda_match_idx)
                        updated_bundle_index = bundle_index
                        bundle_action = "replace_lambda_representative"
                else:
                    trainer.set_trainable_parameter_vector(bundle.points[source_idx])
                    bundle_action = "reject"
            elif script_args.bundle_update_mode == "replace_source":
                if candidate_accepted:
                    bundle.replace(source_idx, oracle["x"], oracle["fvals"], oracle["grads"])
                    bundle_replaced = True
                    bundle_index = int(source_idx)
                    updated_bundle_index = bundle_index
                    bundle_action = "replace_source"
                else:
                    trainer.set_trainable_parameter_vector(bundle.points[source_idx])
                    bundle_action = "reject"
            else:
                if candidate_accepted:
                    bundle.add(oracle["x"], oracle["fvals"], oracle["grads"])
                    bundle_lambdas.append(lam.copy())
                    bundle_appended = True
                    bundle_index = bundle.m - 1
                    new_bundle_index = bundle_index
                    bundle_action = "append"
                else:
                    trainer.set_trainable_parameter_vector(bundle.points[source_idx])
                    bundle_action = "reject"
            outer_anchor_bundle_indices.append(bundle_index)
            outer_new_bundle_indices.append(new_bundle_index)
            outer_updated_bundle_indices.append(updated_bundle_index)
            inner_gn_at_lambda_after = gn_value_at_lambda(
                bundle,
                lam,
                lambda_normalization="none",
                lambda_min=script_args.lambda_min,
                use_projection=False,
            )
            selection_inner_gn_at_lambda_after = gn_value_at_lambda(
                bundle,
                lam,
                lambda_normalization=script_args.lambda_normalization,
                lambda_min=script_args.lambda_min,
                use_projection=True,
            )
            total_parameter_updates += 1
            total_oracle_gradient_evals += 1
            total_inner_steps += 1
            M_t = inner_step
            current_inner_stop_reached = stop_reached(
                inner_gn_at_lambda_after,
                inner_stop_threshold,
            )
            if current_inner_stop_reached:
                inner_stop_reached = True
                inner_stop_reason = f"{script_args.stop_rule}_inner_gn_below_threshold"
            inner_total_seconds = time.perf_counter() - inner_wall_t0
            inner_accept_seconds = max(
                0.0,
                inner_total_seconds - inner_update_seconds - inner_oracle_seconds,
            )
            timing_inner_total_seconds += inner_total_seconds
            timing_inner_accept_seconds += inner_accept_seconds
            inner_records.append({
                "inner_step": inner_step,
                "update_rule": script_args.update_rule,
                "gradient_eval": total_oracle_gradient_evals,
                "oracle_gradient_eval": total_oracle_gradient_evals,
                "parameter_update": total_parameter_updates,
                "parameter_updates": total_parameter_updates,
                "objective_gradient_evals": num_objectives * total_parameter_updates,
                "elapsed_wall_seconds": time.perf_counter() - run_start_time,
                "bundle_size": bundle.m,
                "bundle_size_before": bundle_size_before_inner,
                "bundle_size_after": bundle.m,
                "bundle_update_mode": script_args.bundle_update_mode,
                "bundle_action": bundle_action,
                "bundle_replaced": bundle_replaced,
                "bundle_appended": bundle_appended,
                "bundle_index": bundle_index,
                "new_bundle_index": new_bundle_index,
                "updated_bundle_index": updated_bundle_index,
                "candidate_accepted": candidate_accepted,
                "lambda_known": lambda_known,
                "lambda_match_idx": lambda_match_idx,
                "lambda_match_tol": script_args.lambda_match_tol,
                "source_idx": source_idx,
                "source_grad_norm_sq": source_grad_norm_sq,
                "source_f_lambda": source_f_lambda,
                "t_map_u_star": t_map_u_star,
                "L_lambda": L_lambda,
                "l_scale_before": l_scale_before_step,
                "l_scale_after": current_l_scale,
                "f_lambda_new": f_lambda_new,
                "descent_slack": descent_slack,
                "descent_tolerance": descent_tolerance,
                "descent_atol": script_args.descent_atol,
                "descent_rtol": script_args.descent_rtol,
                "safeguard_triggered": safeguard_triggered,
                "safeguard_violations": safeguard_violations,
                "gn_at_lambda_before_outer": inner_gn_reference,
                "gn_at_lambda_after": inner_gn_at_lambda_after,
                "lambda_selection_gn_at_lambda_after": selection_inner_gn_at_lambda_after,
                "inner_stop_threshold": inner_stop_threshold,
                "inner_stop_reached": current_inner_stop_reached,
                "inner_stop_rule": script_args.stop_rule,
                "train_loss": train_loss_value,
                "train_helpful_loss": train_helpful_loss,
                "train_harmless_loss": train_harmless_loss,
                "update_data_source": script_args.update_data_source,
                "update_batches": update_batches,
                "update_examples": update_examples,
                "learning_rate": learning_rate,
                "fvals": oracle["fvals"].tolist(),
                "timing_total_seconds": inner_total_seconds,
                "timing_update_seconds": inner_update_seconds,
                "timing_oracle_seconds": inner_oracle_seconds,
                "timing_accept_seconds": inner_accept_seconds,
            })
            if current_inner_stop_reached:
                break

        pruned_inner_candidate_count = 0
        retained_bundle_index = None
        appended_candidate_count = (
            max(0, bundle.m - bundle_size_before)
            if script_args.bundle_update_mode == "append"
            else 0
        )
        bundle_cap_action = (
            "disabled"
            if not script_args.max_bundle_size or script_args.max_bundle_size <= 0
            else "within_cap"
        )
        bundle_cap_source_idx = None
        bundle_cap_source_gn = None
        bundle_cap_candidate_idx = None
        bundle_cap_candidate_gn = None
        bundle_cap_swap_idx = None
        bundle_cap_gn_star_before = None
        bundle_cap_gn_star_after = None
        bundle_cap_gn_star_lambda_before = None
        bundle_cap_gn_star_lambda_after = None
        bundle_cap_removed_count = 0
        bundle_cap_solver_seconds = 0.0
        bundle_cap_solver_calls = 0
        timing_prune_inner_seconds = 0.0
        timing_bundle_cap_seconds = 0.0
        prune_inner_t0 = time.perf_counter()
        if script_args.prune_inner and script_args.bundle_update_mode == "append":
            retained_bundle_index = prune_last_candidates(
                bundle,
                bundle_size_before,
                appended_candidate_count,
                lam,
            )
            if retained_bundle_index is not None:
                pruned_inner_candidate_count = appended_candidate_count - 1
                del bundle_lambdas[bundle_size_before:]
                bundle_lambdas.append(lam.copy())
                outer_anchor_bundle_indices = [retained_bundle_index]
                outer_new_bundle_indices = [retained_bundle_index]
                outer_updated_bundle_indices = []
                inner_gn_at_lambda_after = gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization="none",
                    lambda_min=script_args.lambda_min,
                    use_projection=False,
                )
                selection_inner_gn_at_lambda_after = gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization=script_args.lambda_normalization,
                    lambda_min=script_args.lambda_min,
                    use_projection=True,
                )
                trainer.set_trainable_parameter_vector(bundle.points[-1])
        timing_prune_inner_seconds = time.perf_counter() - prune_inner_t0

        candidate_bundle_index = retained_bundle_index
        if (
            candidate_bundle_index is None
            and script_args.bundle_update_mode == "append"
            and appended_candidate_count == 1
            and bundle.m > bundle_size_before
        ):
            candidate_bundle_index = bundle.m - 1

        if (
            script_args.max_bundle_size
            and script_args.max_bundle_size > 0
            and script_args.bundle_update_mode == "append"
            and bundle.m > script_args.max_bundle_size
        ):
            bundle_cap_t0 = time.perf_counter()
            if candidate_bundle_index is None or bundle_size_before < 1:
                bundle_cap_action = "not_applied_no_single_candidate"
            else:
                bundle_cap_candidate_idx = int(candidate_bundle_index)
                bundle_cap_candidate_gn = scalarized_gradient_norm_sq(
                    bundle,
                    bundle_cap_candidate_idx,
                    lam,
                )
                bundle_cap_source_idx, bundle_cap_source_gn = active_gn_source_prefix(
                    bundle,
                    lam,
                    bundle_size_before,
                )
                if script_args.bundle_cap_mode in {"global_swap_if_better", "global_swap"}:
                    swap_info = best_global_cap_swap(
                        bundle,
                        bundle_cap_candidate_idx,
                        bundle_size_before,
                        prev_lam=lam,
                        max_starts=script_args.lambda_max_starts,
                        solver=script_args.lambda_solver,
                        require_ipopt=script_args.require_ipopt,
                        lambda_normalization=script_args.lambda_normalization,
                        lambda_min=script_args.lambda_min,
                        use_projection=True,
                    )
                    bundle_cap_gn_star_before = swap_info["base_gn_star"]
                    bundle_cap_gn_star_after = swap_info["best_gn_star"]
                    bundle_cap_gn_star_lambda_before = swap_info["base_lambda"]
                    bundle_cap_gn_star_lambda_after = swap_info["best_lambda"]
                    bundle_cap_swap_idx = swap_info["best_replace_idx"]
                    bundle_cap_solver_seconds = float(
                        swap_info.get("total_solver_seconds", 0.0)
                    )
                    bundle_cap_solver_calls = int(swap_info.get("solver_calls", 0))
                    accept_swap = (
                        bundle_cap_swap_idx is not None
                        and (
                            script_args.bundle_cap_mode == "global_swap"
                            or bundle_cap_gn_star_after < bundle_cap_gn_star_before
                        )
                    )
                    if accept_swap:
                        candidate_x = bundle.points[bundle_cap_candidate_idx].copy()
                        candidate_fvals = bundle.fvals[bundle_cap_candidate_idx].copy()
                        candidate_grads = bundle.grads[bundle_cap_candidate_idx].copy()
                        bundle.replace(
                            bundle_cap_swap_idx,
                            candidate_x,
                            candidate_fvals,
                            candidate_grads,
                        )
                        bundle.pop()
                        if bundle_lambdas:
                            bundle_lambdas[bundle_cap_swap_idx] = lam.copy()
                            bundle_lambdas.pop()
                        retained_bundle_index = int(bundle_cap_swap_idx)
                        outer_anchor_bundle_indices = [retained_bundle_index]
                        outer_new_bundle_indices = [retained_bundle_index]
                        outer_updated_bundle_indices = [retained_bundle_index]
                        bundle_cap_action = script_args.bundle_cap_mode
                        trainer.set_trainable_parameter_vector(bundle.points[retained_bundle_index])
                    else:
                        bundle.pop()
                        if bundle_lambdas:
                            bundle_lambdas.pop()
                        retained_bundle_index = None
                        outer_anchor_bundle_indices = []
                        outer_new_bundle_indices = []
                        outer_updated_bundle_indices = []
                        bundle_cap_action = "drop_candidate_no_global_swap_improvement"
                        if bundle_cap_source_idx is not None:
                            trainer.set_trainable_parameter_vector(bundle.points[bundle_cap_source_idx])
                elif bundle_cap_candidate_gn < bundle_cap_source_gn:
                    candidate_x = bundle.points[bundle_cap_candidate_idx].copy()
                    candidate_fvals = bundle.fvals[bundle_cap_candidate_idx].copy()
                    candidate_grads = bundle.grads[bundle_cap_candidate_idx].copy()
                    bundle.replace(
                        bundle_cap_source_idx,
                        candidate_x,
                        candidate_fvals,
                        candidate_grads,
                    )
                    bundle.pop()
                    if bundle_lambdas:
                        bundle_lambdas[bundle_cap_source_idx] = lam.copy()
                        bundle_lambdas.pop()
                    retained_bundle_index = int(bundle_cap_source_idx)
                    outer_anchor_bundle_indices = [retained_bundle_index]
                    outer_new_bundle_indices = [retained_bundle_index]
                    outer_updated_bundle_indices = [retained_bundle_index]
                    bundle_cap_action = "replace_active_if_better"
                    trainer.set_trainable_parameter_vector(bundle.points[retained_bundle_index])
                else:
                    bundle.pop()
                    if bundle_lambdas:
                        bundle_lambdas.pop()
                    retained_bundle_index = None
                    outer_anchor_bundle_indices = []
                    outer_new_bundle_indices = []
                    outer_updated_bundle_indices = []
                    bundle_cap_action = "drop_candidate_not_better"
                    trainer.set_trainable_parameter_vector(bundle.points[bundle_cap_source_idx])
                bundle_cap_removed_count = 1
                inner_gn_at_lambda_after = gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization="none",
                    lambda_min=script_args.lambda_min,
                    use_projection=False,
                )
                selection_inner_gn_at_lambda_after = gn_value_at_lambda(
                    bundle,
                    lam,
                    lambda_normalization=script_args.lambda_normalization,
                    lambda_min=script_args.lambda_min,
                    use_projection=True,
                )
            timing_bundle_cap_seconds = time.perf_counter() - bundle_cap_t0

        lambda_stall_update = {
            "enabled": script_args.lambda_stall_patience > 0,
            "lambda": lam.tolist(),
            "stalled": False,
            "improvement": float(inner_gn_reference - inner_gn_at_lambda_after),
            "required_improvement": 0.0,
            "stall_count": 0,
            "blocked_until_outer": 0,
        }
        if script_args.lambda_stall_patience > 0:
            lambda_stall_update = update_lambda_stall_states(
                lambda_stall_states,
                lam,
                inner_gn_reference,
                inner_gn_at_lambda_after,
                outer=outer,
                patience=int(script_args.lambda_stall_patience),
                abs_delta=float(script_args.lambda_stall_abs_delta),
                rel_delta=float(script_args.lambda_stall_rel_delta),
                cooldown=int(script_args.lambda_stall_cooldown),
                match_tol=float(script_args.lambda_stall_match_tol),
            )
        outer_elapsed_wall_seconds_after = time.perf_counter() - run_start_time
        timing_outer_total_seconds = time.perf_counter() - outer_wall_t0
        timing_parameter_update_delta = max(
            1,
            int(total_parameter_updates - parameter_updates_before),
        )
        timing_other_overhead_seconds = max(
            0.0,
            timing_outer_total_seconds
            - timing_lambda_solver_seconds
            - timing_lambda_raw_solver_seconds
            - timing_lambda_diagnostics_seconds
            - timing_inner_total_seconds
            - timing_prune_inner_seconds
            - timing_bundle_cap_seconds,
        )

        outer_bundle_indices = unique_ints(outer_anchor_bundle_indices)
        outer_new_bundle_indices = unique_ints(outer_new_bundle_indices)
        outer_updated_bundle_indices = unique_ints(outer_updated_bundle_indices)
        outer_checkpoint_dir = (
            os.path.join(script_args.training_args.output_dir, f"outer_{outer}")
            if script_args.save_every_outer and outer % script_args.save_every_outer == 0
            else None
        )
        solution_path["anchors"].append({
            "phase": "outer",
            "outer": outer,
            "lambda": lam.tolist(),
            "lambda_before_diversity": gn_lam.tolist(),
            "M_t": int(M_t),
            "inner_stop_reached": inner_stop_reached,
            "inner_stop_reason": inner_stop_reason,
            "inner_stop_threshold": inner_stop_threshold,
            "prune_inner": script_args.prune_inner,
            "pruned_inner_candidate_count": pruned_inner_candidate_count,
            "retained_bundle_index": retained_bundle_index,
            "max_bundle_size": script_args.max_bundle_size,
            "bundle_cap_mode": script_args.bundle_cap_mode,
            "bundle_cap_action": bundle_cap_action,
            "bundle_cap_source_idx": bundle_cap_source_idx,
            "bundle_cap_source_gn": bundle_cap_source_gn,
            "bundle_cap_candidate_idx": bundle_cap_candidate_idx,
            "bundle_cap_candidate_gn": bundle_cap_candidate_gn,
            "bundle_cap_swap_idx": bundle_cap_swap_idx,
            "bundle_cap_gn_star_before": bundle_cap_gn_star_before,
            "bundle_cap_gn_star_after": bundle_cap_gn_star_after,
            "bundle_cap_gn_star_lambda_before": bundle_cap_gn_star_lambda_before,
            "bundle_cap_gn_star_lambda_after": bundle_cap_gn_star_lambda_after,
            "bundle_cap_removed_count": bundle_cap_removed_count,
            "gn_certificate_type": gn_certificate_type,
            "gn_star_before": pc_value,
            "raw_gn_star_lambda": raw_pc_lam.tolist(),
            "gn_at_lambda_before": inner_gn_reference,
            "gn_at_lambda_after": inner_gn_at_lambda_after,
            "lambda_stall_selection": lambda_stall_selection,
            "lambda_stall_update": lambda_stall_update,
            "bundle_indices": outer_bundle_indices,
            "new_bundle_indices": outer_new_bundle_indices,
            "updated_bundle_indices": outer_updated_bundle_indices,
            "bundle_size_before": bundle_size_before,
            "bundle_size_after": bundle.m,
            "parameter_updates_before": parameter_updates_before,
            "parameter_updates_after": total_parameter_updates,
            "oracle_gradient_evals_before": oracle_gradient_evals_before,
            "oracle_gradient_evals_after": total_oracle_gradient_evals,
            "objective_gradient_evals_before": objective_gradient_evals_before,
            "objective_gradient_evals_after": num_objectives * total_parameter_updates,
            "elapsed_wall_seconds_before": outer_elapsed_wall_seconds_before,
            "elapsed_wall_seconds_after": outer_elapsed_wall_seconds_after,
            "timing_outer_total_seconds": timing_outer_total_seconds,
            "timing_seconds_per_parameter_update": (
                timing_outer_total_seconds / timing_parameter_update_delta
            ),
            "timing_lambda_solver_seconds": timing_lambda_solver_seconds,
            "timing_lambda_raw_solver_seconds": timing_lambda_raw_solver_seconds,
            "timing_lambda_diagnostics_seconds": timing_lambda_diagnostics_seconds,
            "timing_inner_total_seconds": timing_inner_total_seconds,
            "timing_inner_update_seconds": timing_inner_update_seconds,
            "timing_inner_oracle_seconds": timing_inner_oracle_seconds,
            "timing_inner_accept_seconds": timing_inner_accept_seconds,
            "timing_prune_inner_seconds": timing_prune_inner_seconds,
            "timing_bundle_cap_seconds": timing_bundle_cap_seconds,
            "timing_bundle_cap_solver_seconds": bundle_cap_solver_seconds,
            "timing_bundle_cap_solver_calls": bundle_cap_solver_calls,
            "timing_other_overhead_seconds": timing_other_overhead_seconds,
            "checkpoint_dir": outer_checkpoint_dir,
        })

        record = {
            "outer": outer,
            "lambda": lam.tolist(),
            "update_rule": script_args.update_rule,
            "bundle_update_mode": script_args.bundle_update_mode,
            "update_data_source": script_args.update_data_source,
            "lambda_solver": script_args.lambda_solver,
            "lambda_normalization": script_args.lambda_normalization,
            "lambda_min": script_args.lambda_min,
            "lambda_entropy_tau": lambda_entropy_tau_current,
            "lambda_entropy": lambda_entropy,
            "lambda_before_diversity": gn_lam.tolist(),
            "lambda_diversity": lambda_diversity,
            "lambda_stall_selection": lambda_stall_selection,
            "lambda_stall_update": lambda_stall_update,
            "lambda_stall_states": lambda_stall_states,
            "lambda_projection": bundle.lambda_projection_info(),
            "gn_certificate_type": gn_certificate_type,
            "gn_star": pc_value,
            "raw_gn_star": pc_value,
            "raw_gn_star_lambda": raw_pc_lam.tolist(),
            "raw_gn_at_selected_lambda": raw_gn_at_selected_lam,
            "lambda_selection_objective_star": selection_pc_value,
            "lambda_selection_gn_star": selection_pc_value,
            "lambda_selection_gn_at_selected_lambda": selection_gn_at_selected_lam,
            "stop_rule": script_args.stop_rule,
            "outer_stop_threshold": outer_stop_threshold,
            "relative_outer_reference_gn": relative_outer_reference_gn,
            "inner_stop_threshold": inner_stop_threshold,
            "inner_stop_reached": inner_stop_reached,
            "inner_stop_reason": inner_stop_reason,
            "prune_inner": script_args.prune_inner,
            "pruned_inner_candidate_count": pruned_inner_candidate_count,
            "retained_bundle_index": retained_bundle_index,
            "max_bundle_size": script_args.max_bundle_size,
            "bundle_cap_mode": script_args.bundle_cap_mode,
            "bundle_cap_action": bundle_cap_action,
            "bundle_cap_source_idx": bundle_cap_source_idx,
            "bundle_cap_source_gn": bundle_cap_source_gn,
            "bundle_cap_candidate_idx": bundle_cap_candidate_idx,
            "bundle_cap_candidate_gn": bundle_cap_candidate_gn,
            "bundle_cap_swap_idx": bundle_cap_swap_idx,
            "bundle_cap_gn_star_before": bundle_cap_gn_star_before,
            "bundle_cap_gn_star_after": bundle_cap_gn_star_after,
            "bundle_cap_gn_star_lambda_before": bundle_cap_gn_star_lambda_before,
            "bundle_cap_gn_star_lambda_after": bundle_cap_gn_star_lambda_after,
            "bundle_cap_removed_count": bundle_cap_removed_count,
            "M_t": int(M_t),
            "gn_at_lambda_before": inner_gn_reference,
            "gn_at_lambda_after": inner_gn_at_lambda_after,
            "anchor_bundle_indices": outer_bundle_indices,
            "anchor_new_bundle_indices": outer_new_bundle_indices,
            "anchor_updated_bundle_indices": outer_updated_bundle_indices,
            "gradient_evals_before": oracle_gradient_evals_before,
            "gradient_evals_after": total_oracle_gradient_evals,
            "oracle_gradient_evals_before": oracle_gradient_evals_before,
            "oracle_gradient_evals_after": total_oracle_gradient_evals,
            "parameter_updates_before": parameter_updates_before,
            "parameter_updates_after": total_parameter_updates,
            "objective_gradient_evals_before": objective_gradient_evals_before,
            "objective_gradient_evals_after": num_objectives * total_parameter_updates,
            "elapsed_wall_seconds": outer_elapsed_wall_seconds_before,
            "elapsed_wall_seconds_before": outer_elapsed_wall_seconds_before,
            "elapsed_wall_seconds_after": outer_elapsed_wall_seconds_after,
            "timing_outer_total_seconds": timing_outer_total_seconds,
            "timing_seconds_per_parameter_update": (
                timing_outer_total_seconds / timing_parameter_update_delta
            ),
            "timing_lambda_solver_seconds": timing_lambda_solver_seconds,
            "timing_lambda_raw_solver_seconds": timing_lambda_raw_solver_seconds,
            "timing_lambda_diagnostics_seconds": timing_lambda_diagnostics_seconds,
            "timing_inner_total_seconds": timing_inner_total_seconds,
            "timing_inner_update_seconds": timing_inner_update_seconds,
            "timing_inner_oracle_seconds": timing_inner_oracle_seconds,
            "timing_inner_accept_seconds": timing_inner_accept_seconds,
            "timing_prune_inner_seconds": timing_prune_inner_seconds,
            "timing_bundle_cap_seconds": timing_bundle_cap_seconds,
            "timing_bundle_cap_solver_seconds": bundle_cap_solver_seconds,
            "timing_bundle_cap_solver_calls": bundle_cap_solver_calls,
            "timing_other_overhead_seconds": timing_other_overhead_seconds,
            "bundle_size": bundle.m,
            "bundle_size_before": bundle_size_before,
            "bundle_lambda_count": len(bundle_lambdas),
            "num_objectives": num_objectives,
            "objective_names": objective_names,
            "lambda_diagnostics": lambda_diagnostics,
            "l_scale_before": l_scale_before_outer,
            "l_scale_after": current_l_scale,
            "l_scale_final": current_l_scale,
            "safeguard_violations_before": safeguard_violations_before,
            "safeguard_violations_after": safeguard_violations,
            "safeguard_violations": safeguard_violations,
            "total_inner_steps": total_inner_steps,
            "inner": inner_records,
        }
        save_jsonl(history_path, record)
        final_gn_star = pc_value
        stopped_outer = outer
        status = (
            f"outer={outer} lambda={np.round(lam, 4).tolist()} "
            f"gn*={pc_value:.4e} updates={total_parameter_updates} "
            f"M_t={M_t} bundle={bundle.m} "
            f"update={script_args.update_rule}/{script_args.bundle_update_mode}"
            f" data={script_args.update_data_source}"
        )
        if inner_stop_threshold is not None:
            status += (
                f" inner_gn={inner_gn_at_lambda_after:.4e}"
                f"/{inner_stop_threshold:.4e}"
            )
        if bundle.lambda_projection_active:
            status += f" sel_gn={selection_pc_value:.4e}"
        if lambda_entropy_tau_current > 0.0:
            status += f" entropy={lambda_entropy:.4f} tau={lambda_entropy_tau_current:g}"
        if lambda_diversity.get("enabled"):
            status += (
                f" base_lambda={np.round(gn_lam, 4).tolist()} "
                f"diversity={script_args.lambda_diversity_strength:g}"
            )
        if script_args.lambda_stall_patience > 0:
            if lambda_stall_selection.get("selected_by") == "stall_unblocked_grid":
                status += (
                    f" stall_switch_from={np.round(lambda_stall_selection['base_lambda'], 4).tolist()}"
                )
            if lambda_stall_update.get("stalled"):
                status += (
                    f" stall={lambda_stall_update['stall_count']}"
                    f"/{script_args.lambda_stall_patience}"
                )
        if script_args.update_rule == "t_map":
            status += f" L_scale={current_l_scale:g}"
        if script_args.max_bundle_size and script_args.max_bundle_size > 0:
            status += (
                f" cap={script_args.max_bundle_size}"
                f" cap_action={bundle_cap_action}"
            )
        status += (
            f" sec/update={timing_outer_total_seconds / timing_parameter_update_delta:.2f}"
            f" lambda_sec={timing_lambda_solver_seconds + timing_lambda_raw_solver_seconds:.2f}"
            f" update_sec={timing_inner_update_seconds:.2f}"
            f" oracle_sec={timing_inner_oracle_seconds:.2f}"
        )
        if script_args.max_bundle_size and script_args.max_bundle_size > 0:
            status += (
                f" cap_sec={timing_bundle_cap_seconds:.2f}"
                f" cap_solver_sec={bundle_cap_solver_seconds:.2f}"
            )
        print_local_main(status)

        if outer_checkpoint_dir is not None:
            trainer.model.save_pretrained(outer_checkpoint_dir)
            trainer.tokenizer.save_pretrained(outer_checkpoint_dir)
        elapsed_wall_seconds_for_resume = time.perf_counter() - run_start_time
        save_resume_state(
            resume_state_files,
            bundle,
            bundle_lambdas,
            lambda_history,
            lambda_stall_states,
            prev_lam,
            next_outer=outer + 1,
            total_parameter_updates=total_parameter_updates,
            total_oracle_gradient_evals=total_oracle_gradient_evals,
            total_inner_steps=total_inner_steps,
            current_l_scale=current_l_scale,
            safeguard_violations=safeguard_violations,
            safeguard_warned=safeguard_warned,
            relative_outer_reference_gn=relative_outer_reference_gn,
            elapsed_wall_seconds=elapsed_wall_seconds_for_resume,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if (
            prefix_bundle_sizes
            and bundle.m in prefix_bundle_sizes
            and bundle.m not in saved_prefix_bundle_sizes
        ):
            os.makedirs(prefix_state_dir, exist_ok=True)
            prefix_state_files = prefix_resume_paths(prefix_state_dir, bundle.m)
            save_resume_state(
                prefix_state_files,
                bundle,
                bundle_lambdas,
                lambda_history,
                lambda_stall_states,
                prev_lam,
                next_outer=outer + 1,
                total_parameter_updates=total_parameter_updates,
                total_oracle_gradient_evals=total_oracle_gradient_evals,
                total_inner_steps=total_inner_steps,
                current_l_scale=current_l_scale,
                safeguard_violations=safeguard_violations,
                safeguard_warned=safeguard_warned,
                relative_outer_reference_gn=relative_outer_reference_gn,
                elapsed_wall_seconds=elapsed_wall_seconds_for_resume,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            copy_if_exists(history_path, f"{prefix_state_files['base']}_history.jsonl")
            save_json(f"{prefix_state_files['base']}_solution_path.json", solution_path)
            saved_prefix_bundle_sizes.add(bundle.m)
            print_local_main(
                f"saved prefix branch state bundle={bundle.m} "
                f"to {prefix_state_files['base']}"
            )

    final_dir = os.path.join(script_args.training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    trainer.tokenizer.save_pretrained(final_dir)
    solution_path["termination"] = {
        "stop_reason": stop_reason,
        "stopped_outer": stopped_outer,
        "final_gn_star": final_gn_star,
        "best_gn_star": None if not np.isfinite(best_gn_star) else float(best_gn_star),
        "best_gradient_norm": (
            None if not np.isfinite(best_gn_star) else float(np.sqrt(max(best_gn_star, 0.0)))
        ),
        "gn_target_norm": script_args.gn_target_norm,
        "target_reached": target_hit is not None,
        "parameter_updates": total_parameter_updates,
        "oracle_gradient_evals": total_oracle_gradient_evals,
        "objective_gradient_evals": num_objectives * total_parameter_updates,
        "bundle_size": bundle.m,
        "final_checkpoint": final_dir,
    }
    save_json(solution_path_file, solution_path)
    final_state_path = os.path.join(script_args.training_args.output_dir, "adaptive_final_state.json")
    with open(final_state_path, "w") as handle:
        json.dump(
            {
                "algorithm_mode": script_args.algorithm_mode,
                "update_rule": script_args.update_rule,
                "bundle_update_mode": script_args.bundle_update_mode,
                "stop_rule": script_args.stop_rule,
                "epsilon": script_args.epsilon,
                "relative_rho": script_args.relative_rho,
                "gn_target_norm": script_args.gn_target_norm,
                "stop_reason": stop_reason,
                "stopped_outer": stopped_outer,
                "final_gn_star": final_gn_star,
                "best_gn_star": None if not np.isfinite(best_gn_star) else float(best_gn_star),
                "best_gradient_norm": (
                    None
                    if not np.isfinite(best_gn_star)
                    else float(np.sqrt(max(best_gn_star, 0.0)))
                ),
                "target_reached": target_hit is not None,
                "target_hit": target_hit,
                "parameter_updates": total_parameter_updates,
                "oracle_gradient_evals": total_oracle_gradient_evals,
                "objective_gradient_evals": num_objectives * total_parameter_updates,
                "l_scale_final": current_l_scale,
                "safeguard_violations": safeguard_violations,
                "bundle_size": bundle.m,
                "lambda_stall_states": lambda_stall_states,
                "lambda_projection": bundle.lambda_projection_info(),
                "solution_path": solution_path_file,
            },
            handle,
            indent=2,
        )
    if script_args.gn_target_norm is not None:
        save_json(
            os.path.join(script_args.training_args.output_dir, "plateau_summary.json"),
            {
                "method": "Adaptive bundle",
                "gn_target_norm": float(script_args.gn_target_norm),
                "target_reached": target_hit is not None,
                "target_hit": target_hit,
                "best_gn_star": (
                    None if not np.isfinite(best_gn_star) else float(best_gn_star)
                ),
                "best_gradient_norm": (
                    None
                    if not np.isfinite(best_gn_star)
                    else float(np.sqrt(max(best_gn_star, 0.0)))
                ),
                "parameter_updates": total_parameter_updates,
                "objective_gradient_evals": num_objectives * total_parameter_updates,
            },
        )
    print_local_main(f"saved final checkpoint to {final_dir}")
    print_local_main(f"saved solution path to {solution_path_file}")


if __name__ == "__main__":
    main()
