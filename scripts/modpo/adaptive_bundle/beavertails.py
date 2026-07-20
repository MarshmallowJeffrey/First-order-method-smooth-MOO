from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from scripts.modpo.adaptive_bundle.bundle_core import (
    FirstOrderBundle,
    bundle_gradient_diagnostics,
    gn_grid_diagnostics,
    gn_value_at_lambda,
    ipopt_available,
    ipopt_import_error,
    maximise_gn,
    prune_last_candidates,
    t_map_step,
)
import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, get_scheduler

from scripts.modpo.adaptive_bundle.trainer import AdaptiveBundleMODPOTrainer
from src.data.configs import DATASET_CONFIGS
from src.trainer.modpo_trainer import MODPODataCollatorWithPadding, MODPODataMapFunc
from src.utils import (
    disable_progress_bar_non_local_main,
    param_sharding_enabled,
    print_local_main,
    set_seeds,
)

disable_progress_bar_non_local_main()


QWEN_PROMPT_TEMPLATE = "<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n"


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

    max_outer: Optional[int] = field(default=20)
    max_inner: Optional[int] = field(default=25)
    update_rule: Optional[str] = field(default="adamw")
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
    lambda_min: Optional[float] = field(default=0.05)
    smoothness: Optional[str] = field(default="1.0,1.0")
    l_scale: Optional[float] = field(default=1.0)
    descent_atol: Optional[float] = field(default=1e-6)
    descent_rtol: Optional[float] = field(default=1e-6)
    prune_inner: Optional[bool] = field(default=False)
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


def weighted_dpo_loss(trainer, helpful_batch, harmless_batch, helpful_weight, harmless_weight):
    helpful_batch = trainer._prepare_inputs(helpful_batch)
    harmless_batch = trainer._prepare_inputs(harmless_batch)

    helpful_loss = trainer.dpo_objective_loss(trainer.model, helpful_batch)
    harmless_loss = trainer.dpo_objective_loss(trainer.model, harmless_batch)
    loss = helpful_weight * helpful_loss + harmless_weight * harmless_loss
    return loss, helpful_loss.detach(), harmless_loss.detach()


def main():
    script_args = tyro.cli(ScriptArguments)
    set_seeds(script_args.seed)
    os.makedirs(script_args.training_args.output_dir, exist_ok=True)

    smoothness = parse_float_list(script_args.smoothness)
    if len(smoothness) != 2:
        raise ValueError("The first BeaverTails runner expects exactly two smoothness constants.")
    if script_args.bundle_dtype not in {"float32", "float64"}:
        raise ValueError("bundle_dtype must be either 'float32' or 'float64'.")
    if not np.isfinite(script_args.l_scale) or script_args.l_scale <= 0.0:
        raise ValueError("l_scale must be finite and strictly positive.")
    if not np.isfinite(script_args.descent_atol) or script_args.descent_atol < 0.0:
        raise ValueError("descent_atol must be finite and non-negative.")
    if not np.isfinite(script_args.descent_rtol) or script_args.descent_rtol < 0.0:
        raise ValueError("descent_rtol must be finite and non-negative.")
    if script_args.lambda_solver not in {"ipopt", "slsqp"}:
        raise ValueError("lambda_solver must be either 'ipopt' or 'slsqp'.")
    if script_args.lambda_normalization not in {"none", "global_mean"}:
        raise ValueError("lambda_normalization must be either 'none' or 'global_mean'.")
    if (
        not np.isfinite(script_args.lambda_min)
        or script_args.lambda_min < 0.0
        or script_args.lambda_min >= 1.0 / len(smoothness)
    ):
        raise ValueError("lambda_min must be finite and in [0, 1 / num_objectives).")
    if script_args.update_rule == "adam":
        script_args.update_rule = "adamw"
    if script_args.update_rule not in {"adamw", "t_map"}:
        raise ValueError("update_rule must be either 'adamw' or 't_map'.")
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
    if script_args.update_rule == "adamw" and script_args.prune_inner:
        warnings.warn(
            "prune_inner is ignored for AdamW updates because optimizer state "
            "follows the full trajectory.",
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
    better_pool = select_subset(
        better_rdp.get_preference_dataset(split="train"),
        script_args.train_subset_size_per_objective,
        script_args.seed,
        "helpful/better train pool",
    )
    safer_pool = select_subset(
        safer_rdp.get_preference_dataset(split="train"),
        script_args.train_subset_size_per_objective,
        script_args.seed + 1,
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
        script_args.seed + 3,
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
        total_optimizer_steps = max(1, script_args.max_outer * script_args.max_inner)
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

    config_path = os.path.join(script_args.training_args.output_dir, "adaptive_config.json")
    with open(config_path, "w") as handle:
        json.dump(asdict(script_args), handle, indent=2, default=str)

    print_local_main("building initial bundle point...")
    initial = oracle_call()
    num_objectives = 2
    total_parameter_updates = 0
    total_oracle_gradient_evals = 1
    bundle = FirstOrderBundle(
        K=num_objectives,
        d=int(initial["x"].shape[0]),
        L=np.asarray(smoothness, dtype=np.float64),
        dtype=np.dtype(script_args.bundle_dtype),
    )
    bundle.add(initial["x"], initial["fvals"], initial["grads"])
    initial_oracle_path = os.path.join(script_args.training_args.output_dir, "adaptive_initial_oracle.json")
    with open(initial_oracle_path, "w") as handle:
        json.dump(
            {
                "gradient_eval": 1,
                "oracle_gradient_eval": total_oracle_gradient_evals,
                "parameter_update": total_parameter_updates,
                "parameter_updates": total_parameter_updates,
                "objective_gradient_evals": num_objectives * total_parameter_updates,
                "num_objectives": num_objectives,
                "fvals": initial["fvals"].tolist(),
                "objective_names": initial["objective_names"],
            },
            handle,
            indent=2,
        )

    history_path = os.path.join(script_args.training_args.output_dir, "adaptive_history.jsonl")
    if os.path.exists(history_path) and script_args.training_args.overwrite_output_dir:
        os.remove(history_path)

    prev_lam = None
    total_inner_steps = 0
    current_l_scale = float(script_args.l_scale)
    safeguard_violations = 0
    safeguard_warned = False
    for outer in range(1, script_args.max_outer + 1):
        selection_pc_value, lam = maximise_gn(
            bundle,
            prev_lam=prev_lam,
            max_starts=script_args.lambda_max_starts,
            solver=script_args.lambda_solver,
            require_ipopt=script_args.require_ipopt,
            lambda_normalization=script_args.lambda_normalization,
            lambda_min=script_args.lambda_min,
        )
        if script_args.lambda_normalization == "none":
            pc_value = selection_pc_value
            raw_pc_lam = lam.copy()
        else:
            pc_value, raw_pc_lam = maximise_gn(
                bundle,
                prev_lam=prev_lam,
                max_starts=script_args.lambda_max_starts,
                solver=script_args.lambda_solver,
                require_ipopt=script_args.require_ipopt,
                lambda_normalization="none",
                lambda_min=script_args.lambda_min,
            )
        raw_gn_at_selected_lam = gn_value_at_lambda(
            bundle,
            lam,
            lambda_normalization="none",
            lambda_min=script_args.lambda_min,
        )
        selection_gn_at_selected_lam = gn_value_at_lambda(
            bundle,
            lam,
            lambda_normalization=script_args.lambda_normalization,
            lambda_min=script_args.lambda_min,
        )
        prev_lam = lam.copy()
        lambda_diagnostics = {
            "lambda_normalization": script_args.lambda_normalization,
            "lambda_min": script_args.lambda_min,
            "gradient": bundle_gradient_diagnostics(bundle),
            "gn_grid": gn_grid_diagnostics(
                bundle,
                lambda_normalization="none",
                lambda_min=script_args.lambda_min,
            ),
            "selection_gn_grid": gn_grid_diagnostics(
                bundle,
                lambda_normalization=script_args.lambda_normalization,
                lambda_min=script_args.lambda_min,
            ),
        }

        bundle_size_before = bundle.m
        parameter_updates_before = total_parameter_updates
        oracle_gradient_evals_before = total_oracle_gradient_evals
        objective_gradient_evals_before = num_objectives * total_parameter_updates
        l_scale_before_outer = current_l_scale
        safeguard_violations_before = safeguard_violations
        inner_records = []
        for inner_step in range(1, script_args.max_inner + 1):
            l_scale_before_step = current_l_scale
            source_idx = bundle.m - 1
            source_grad_norm_sq = None
            t_map_u_star = None
            L_lambda = None
            f_lambda_new = None
            descent_slack = None
            descent_tolerance = None
            safeguard_triggered = False
            train_loss_value = None
            train_helpful_loss = None
            train_harmless_loss = None
            learning_rate = None

            if script_args.update_rule == "t_map":
                x_new, source_idx, source_grad_norm_sq, t_map_u_star, L_lambda = t_map_step(
                    bundle,
                    lam,
                    L_scale=current_l_scale,
                )
                trainer.set_trainable_parameter_vector(x_new)
            else:
                micro_losses = []
                micro_helpful_losses = []
                micro_harmless_losses = []
                for _ in range(script_args.gradient_accumulation_steps):
                    loss, helpful_loss, harmless_loss = weighted_dpo_loss(
                        trainer,
                        helpful_loader.next(),
                        harmless_loader.next(),
                        float(lam[0]),
                        float(lam[1]),
                    )
                    micro_losses.append(float(loss.detach().cpu()))
                    micro_helpful_losses.append(float(helpful_loss.cpu()))
                    micro_harmless_losses.append(float(harmless_loss.cpu()))
                    scaled_loss = loss / script_args.gradient_accumulation_steps
                    scaled_loss.backward()
                if script_args.max_grad_norm and script_args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, script_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                learning_rate = float(scheduler.get_last_lr()[0])
                train_loss_value = float(np.mean(micro_losses))
                train_helpful_loss = float(np.mean(micro_helpful_losses))
                train_harmless_loss = float(np.mean(micro_harmless_losses))

            oracle = oracle_call()
            bundle.add(oracle["x"], oracle["fvals"], oracle["grads"])
            if script_args.update_rule == "t_map":
                f_lambda_new = float(np.asarray(oracle["fvals"], dtype=np.float64) @ lam)
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
            total_parameter_updates += 1
            total_oracle_gradient_evals += 1
            total_inner_steps += 1
            inner_records.append({
                "inner_step": inner_step,
                "update_rule": script_args.update_rule,
                "gradient_eval": total_oracle_gradient_evals,
                "oracle_gradient_eval": total_oracle_gradient_evals,
                "parameter_update": total_parameter_updates,
                "parameter_updates": total_parameter_updates,
                "objective_gradient_evals": num_objectives * total_parameter_updates,
                "bundle_size": bundle.m,
                "source_idx": source_idx,
                "source_grad_norm_sq": source_grad_norm_sq,
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
                "train_loss": train_loss_value,
                "train_helpful_loss": train_helpful_loss,
                "train_harmless_loss": train_harmless_loss,
                "learning_rate": learning_rate,
                "fvals": oracle["fvals"].tolist(),
            })

        if script_args.update_rule == "t_map" and script_args.prune_inner:
            prune_last_candidates(bundle, bundle_size_before, script_args.max_inner, lam)
            trainer.set_trainable_parameter_vector(bundle.points[-1])

        record = {
            "outer": outer,
            "lambda": lam.tolist(),
            "update_rule": script_args.update_rule,
            "lambda_normalization": script_args.lambda_normalization,
            "lambda_min": script_args.lambda_min,
            "gn_star": pc_value,
            "raw_gn_star": pc_value,
            "raw_gn_star_lambda": raw_pc_lam.tolist(),
            "raw_gn_at_selected_lambda": raw_gn_at_selected_lam,
            "lambda_selection_gn_star": selection_pc_value,
            "lambda_selection_gn_at_selected_lambda": selection_gn_at_selected_lam,
            "gradient_evals_before": oracle_gradient_evals_before,
            "gradient_evals_after": total_oracle_gradient_evals,
            "oracle_gradient_evals_before": oracle_gradient_evals_before,
            "oracle_gradient_evals_after": total_oracle_gradient_evals,
            "parameter_updates_before": parameter_updates_before,
            "parameter_updates_after": total_parameter_updates,
            "objective_gradient_evals_before": objective_gradient_evals_before,
            "objective_gradient_evals_after": num_objectives * total_parameter_updates,
            "bundle_size": bundle.m,
            "bundle_size_before": bundle_size_before,
            "num_objectives": num_objectives,
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
        status = (
            f"outer={outer} lambda={np.round(lam, 4).tolist()} "
            f"gn*={pc_value:.4e} updates={total_parameter_updates} "
            f"bundle={bundle.m} update={script_args.update_rule}"
        )
        if script_args.update_rule == "t_map":
            status += f" L_scale={current_l_scale:g}"
        print_local_main(status)

        if script_args.save_every_outer and outer % script_args.save_every_outer == 0:
            checkpoint_dir = os.path.join(script_args.training_args.output_dir, f"outer_{outer}")
            trainer.model.save_pretrained(checkpoint_dir)
            trainer.tokenizer.save_pretrained(checkpoint_dir)

    final_dir = os.path.join(script_args.training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    trainer.tokenizer.save_pretrained(final_dir)
    final_state_path = os.path.join(script_args.training_args.output_dir, "adaptive_final_state.json")
    with open(final_state_path, "w") as handle:
        json.dump(
            {
                "update_rule": script_args.update_rule,
                "parameter_updates": total_parameter_updates,
                "oracle_gradient_evals": total_oracle_gradient_evals,
                "objective_gradient_evals": num_objectives * total_parameter_updates,
                "l_scale_final": current_l_scale,
                "safeguard_violations": safeguard_violations,
                "bundle_size": bundle.m,
            },
            handle,
            indent=2,
        )
    print_local_main(f"saved final checkpoint to {final_dir}")


if __name__ == "__main__":
    main()
