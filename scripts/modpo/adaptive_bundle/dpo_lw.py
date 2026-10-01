from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from scripts.modpo.adaptive_bundle.bundle_core import (
    FirstOrderBundle,
    LAMBDA_SOLVERS,
    ipopt_available,
    ipopt_import_error,
    maximise_gn,
)
import numpy as np
import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
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


@dataclass
class ScriptArguments:
    sft_model_name: str = field(default="Qwen/Qwen2.5-0.5B-Instruct")
    use_flash_attention_2: Optional[bool] = field(default=False)
    prompt_template: Optional[str] = field(default=QWEN_PROMPT_TEMPLATE)
    helpful_dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K-better")
    harmless_dataset_name: Optional[str] = field(default="PKU-Alignment/PKU-SafeRLHF-10K-safer")
    dataset_caching: Optional[bool] = field(default=False)
    sanity_check: Optional[bool] = field(default=False)

    beta: Optional[float] = field(default=0.1)
    max_length: Optional[int] = field(default=384)
    num_proc: Optional[int] = field(default=4)
    train_subset_size_per_objective: Optional[int] = field(default=2000)
    per_objective_batch_size: Optional[int] = field(default=2)
    seed: Optional[int] = field(default=42)
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

    weight_resolution: Optional[int] = field(default=5)
    helpful_weights: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated helpful weights. Overrides weight_resolution."},
    )
    max_steps: Optional[int] = field(default=300)
    total_update_budget: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional total scalarized parameter-update budget across all "
                "uniform weights. If set, per-weight max_steps are distributed "
                "so sum(per-weight steps) equals this budget, matching the MOA "
                "gradient-eval axis where objective_gradient_evals=updates*K."
            )
        },
    )
    uniform_update_mode: Optional[str] = field(
        default="moa_cycle",
        metadata={
            "help": (
                "'moa_cycle' maintains one LoRA solution per uniform lambda and "
                "cycles over the grid, matching the MOA uniform-discretization "
                "baseline. 'independent' preserves the older behavior that trains "
                "each lambda from scratch once."
            )
        },
    )
    update_data_source: Optional[str] = field(
        default="oracle",
        metadata={
            "help": (
                "'oracle' updates each uniform lambda on the same fixed oracle "
                "subset used for GN* evaluation; 'train' updates on shuffled "
                "training minibatches."
            )
        },
    )
    chain_warm_start: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "For moa_cycle, initialize each first-pass grid point from the "
                "previous grid point's current solution, as in the MOA MLP baseline."
            )
        },
    )
    gradient_accumulation_steps: Optional[int] = field(default=1)
    warmup_ratio: Optional[float] = field(default=0.03)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    weight_decay: Optional[float] = field(default=0.0)
    max_grad_norm: Optional[float] = field(default=1.0)
    save_every_steps: Optional[int] = field(default=0)
    evaluate_uniform_gn: Optional[bool] = field(default=True)
    gn_target_norm: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional pre-specified worst-case gradient-norm target. "
                "For moa_cycle, stop at its first best-so-far GN hit."
            )
        },
    )
    oracle_subset_size_per_objective: Optional[int] = field(default=128)
    oracle_batch_size: Optional[int] = field(default=4)
    smoothness: Optional[str] = field(default="1.0,1.0")
    lambda_max_starts: Optional[int] = field(default=64)
    lambda_solver: Optional[str] = field(default="ipopt")
    require_ipopt: Optional[bool] = field(default=True)
    bundle_dtype: Optional[str] = field(default="float32")

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/dev/dpo_lw",
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


def format_weight(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_helpful_weights(args: ScriptArguments) -> List[Tuple[float, float]]:
    if args.helpful_weights:
        helpful_values = [float(item.strip()) for item in args.helpful_weights.split(",") if item.strip()]
    else:
        if args.weight_resolution < 1:
            raise ValueError("weight_resolution must be at least 1")
        helpful_values = [idx / args.weight_resolution for idx in range(args.weight_resolution + 1)]

    weights = []
    for helpful in helpful_values:
        if helpful < 0.0 or helpful > 1.0:
            raise ValueError(f"Helpful weight must be in [0, 1], got {helpful}")
        harmless = 1.0 - helpful
        weights.append((float(helpful), float(harmless)))
    return weights


def distribute_update_budget(num_weights: int, default_max_steps: int, total_update_budget: Optional[int]) -> List[int]:
    if num_weights < 1:
        raise ValueError("num_weights must be positive")
    if total_update_budget is None:
        if default_max_steps is None or default_max_steps < 0:
            raise ValueError(f"max_steps must be non-negative, got {default_max_steps}")
        return [int(default_max_steps)] * num_weights
    if total_update_budget < 0:
        raise ValueError(f"total_update_budget must be non-negative, got {total_update_budget}")

    base_steps = int(total_update_budget) // num_weights
    extra_steps = int(total_update_budget) % num_weights
    return [base_steps + (1 if idx < extra_steps else 0) for idx in range(num_weights)]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def select_subset(dataset, size, seed, name):
    if size is None or size <= 0 or size >= len(dataset):
        print_local_main(f"{name}: using {len(dataset)} samples")
        return dataset
    dataset = dataset.shuffle(seed=seed)
    print_local_main(f"{name}: selected {size} / {len(dataset)} samples")
    return dataset.select(range(size))


def make_fixed_batches(dataset, batch_size, collator):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
    )
    return list(dataloader)


def preprocess_preference_dataset(dataset, tokenizer, max_length, num_proc):
    dataset = dataset.map(
        MODPODataMapFunc(tokenizer),
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
    helpful_rdp,
    harmless_rdp,
    split: str,
    size,
    agreement_ratio: float,
    seed: int,
):
    raw_dataset = helpful_rdp._get_raw_dataset(split=split)
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
    helpful_pool = map_raw_preference_dataset(
        mixed_raw,
        helpful_rdp,
        "helpful train pool",
    )
    harmless_pool = map_raw_preference_dataset(
        mixed_raw,
        harmless_rdp,
        "harmless train pool",
    )
    return helpful_pool, harmless_pool


def prepare_datasets(args: ScriptArguments, tokenizer):
    if not args.dataset_caching:
        from datasets import disable_caching
        disable_caching()

    helpful_rdp = DATASET_CONFIGS[args.helpful_dataset_name](
        prompt_template=args.prompt_template,
        sanity_check=args.sanity_check,
    )
    harmless_rdp = DATASET_CONFIGS[args.harmless_dataset_name](
        prompt_template=args.prompt_template,
        sanity_check=args.sanity_check,
    )

    if args.agreement_ratio is not None:
        if (
            not np.isfinite(args.agreement_ratio)
            or args.agreement_ratio < 0.0
            or args.agreement_ratio > 1.0
        ):
            raise ValueError("agreement_ratio must be finite and in [0, 1].")
        helpful_pool, harmless_pool = make_controlled_agreement_pools(
            helpful_rdp,
            harmless_rdp,
            split="train",
            size=args.train_subset_size_per_objective,
            agreement_ratio=float(args.agreement_ratio),
            seed=args.seed,
        )
    else:
        helpful_pool = select_subset(
            helpful_rdp.get_preference_dataset(split="train"),
            args.train_subset_size_per_objective,
            args.seed,
            "helpful train pool",
        )
        harmless_pool = select_subset(
            harmless_rdp.get_preference_dataset(split="train"),
            args.train_subset_size_per_objective,
            args.seed + 1,
            "harmless train pool",
        )

    helpful_train = preprocess_preference_dataset(
        helpful_pool,
        tokenizer,
        args.max_length,
        args.num_proc,
    )
    harmless_train = preprocess_preference_dataset(
        harmless_pool,
        tokenizer,
        args.max_length,
        args.num_proc,
    )

    helpful_eval = preprocess_preference_dataset(
        helpful_rdp.get_preference_dataset(split="validation"),
        tokenizer,
        args.max_length,
        args.num_proc,
    )
    return helpful_train, harmless_train, helpful_eval


def build_trainer(args: ScriptArguments, tokenizer, data_collator, helpful_train, helpful_eval):
    device_kwargs = {}
    if torch.cuda.is_available() and not param_sharding_enabled():
        device_kwargs["device_map"] = {"": Accelerator().local_process_index}

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        use_flash_attention_2=args.use_flash_attention_2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        **device_kwargs,
    )
    model.config.update({
        "use_cache": False,
        "pad_token_id": model.config.eos_token_id,
    })

    trainer = AdaptiveBundleMODPOTrainer(
        model=model,
        beta=args.beta,
        args=args.training_args,
        train_dataset=helpful_train,
        eval_dataset=helpful_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        peft_config=args.peft_config,
        max_length=args.max_length,
        num_proc=args.num_proc,
        generate_during_eval=False,
    )
    trainer.model.train()
    return trainer


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


def dpo_batch_examples(batch) -> int:
    return int(batch["input_ids"].shape[0] // 2)


def backward_weighted_dpo_update(
    trainer,
    helpful_weight: float,
    harmless_weight: float,
    update_data_source: str,
    helpful_loader,
    harmless_loader,
    objective_batch_groups: Optional[Dict[str, Sequence[Dict]]],
    gradient_accumulation_steps: int,
) -> Dict:
    if update_data_source == "oracle":
        if objective_batch_groups is None:
            raise ValueError("update_data_source='oracle' requires fixed oracle batches.")

        losses = {}
        update_batches = 0
        update_examples = 0
        objective_specs = [
            ("helpful", float(helpful_weight)),
            ("harmless", float(harmless_weight)),
        ]
        for objective_name, objective_weight in objective_specs:
            batches = list(objective_batch_groups[objective_name])
            if not batches:
                raise ValueError(f"Objective {objective_name!r} has no oracle batches.")

            batch_examples = [dpo_batch_examples(batch) for batch in batches]
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

    if update_data_source != "train":
        raise ValueError("update_data_source must be either 'train' or 'oracle'.")

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
        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()
        micro_losses.append(float(loss.detach().cpu()))
        micro_helpful_losses.append(float(helpful_loss.cpu()))
        micro_harmless_losses.append(float(harmless_loss.cpu()))
        update_examples += dpo_batch_examples(helpful_batch) + dpo_batch_examples(harmless_batch)

    return {
        "train_loss": float(np.mean(micro_losses)),
        "train_helpful_loss": float(np.mean(micro_helpful_losses)),
        "train_harmless_loss": float(np.mean(micro_harmless_losses)),
        "update_batches": 2 * gradient_accumulation_steps,
        "update_examples": update_examples,
    }


def train_one_weight(
    args: ScriptArguments,
    tokenizer,
    data_collator,
    helpful_train,
    harmless_train,
    helpful_eval,
    helpful_weight: float,
    harmless_weight: float,
    objective_batch_groups: Optional[Dict[str, Sequence[Dict]]] = None,
    num_objectives: int = 2,
    max_steps_override: Optional[int] = None,
) -> Dict:
    run_start_time = time.perf_counter()
    run_max_steps = args.max_steps if max_steps_override is None else int(max_steps_override)
    if run_max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {run_max_steps}")

    run_name = (
        f"lambda_helpful_{format_weight(helpful_weight)}"
        f"_harmless_{format_weight(harmless_weight)}"
    )
    run_dir = os.path.join(args.training_args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "dpo_lw_config.json")
    with open(config_path, "w") as handle:
        json.dump(
            {
                **asdict(args),
                "lambda_helpful": helpful_weight,
                "lambda_harmless": harmless_weight,
                "run_max_steps": run_max_steps,
            },
            handle,
            indent=2,
            default=str,
        )

    print_local_main(
        f"training DPO-LW {run_name}: "
        f"loss={helpful_weight:.4f}*F_helpful+{harmless_weight:.4f}*F_harmless "
        f"steps={run_max_steps}"
    )

    set_seeds(args.seed)
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
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.training_args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler_steps = max(1, run_max_steps)
    warmup_steps = int(run_max_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=scheduler_steps,
    )

    history_path = os.path.join(run_dir, "training_history.jsonl")
    if os.path.exists(history_path) and args.training_args.overwrite_output_dir:
        os.remove(history_path)

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    total_micro_steps = run_max_steps * args.gradient_accumulation_steps
    progress = tqdm(range(1, total_micro_steps + 1), disable=not Accelerator().is_local_main_process)
    for micro_step in progress:
        if (micro_step - 1) % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        update_stats = backward_weighted_dpo_update(
            trainer=trainer,
            helpful_weight=helpful_weight,
            harmless_weight=harmless_weight,
            update_data_source=args.update_data_source,
            helpful_loader=helpful_loader,
            harmless_loader=harmless_loader,
            objective_batch_groups=objective_batch_groups,
            gradient_accumulation_steps=1,
        )

        if micro_step % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

        record = {
            "micro_step": micro_step,
            "optimizer_step": optimizer_step,
            "parameter_update": optimizer_step,
            "parameter_updates": optimizer_step,
            "objective_gradient_evals": num_objectives * optimizer_step,
            "micro_objective_gradient_evals": num_objectives * micro_step,
            "elapsed_wall_seconds": time.perf_counter() - run_start_time,
            "lambda_helpful": helpful_weight,
            "lambda_harmless": harmless_weight,
            "max_steps": run_max_steps,
            "loss": float(update_stats["train_loss"]),
            "helpful_loss": float(update_stats["train_helpful_loss"]),
            "harmless_loss": float(update_stats["train_harmless_loss"]),
            "update_data_source": args.update_data_source,
            "update_batches": int(update_stats["update_batches"]),
            "update_examples": int(update_stats["update_examples"]),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        save_jsonl(history_path, record)
        progress.set_description(
            f"loss={record['loss']:.4f} helpful={record['helpful_loss']:.4f} harmless={record['harmless_loss']:.4f}"
        )

        if (
            args.save_every_steps
            and optimizer_step > 0
            and micro_step % args.gradient_accumulation_steps == 0
            and optimizer_step % args.save_every_steps == 0
        ):
            checkpoint_dir = os.path.join(run_dir, f"step_{optimizer_step}")
            trainer.model.save_pretrained(checkpoint_dir)
            trainer.tokenizer.save_pretrained(checkpoint_dir)

    final_dir = os.path.join(run_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    trainer.tokenizer.save_pretrained(final_dir)
    print_local_main(f"saved DPO-LW checkpoint to {final_dir}")

    oracle = None
    if objective_batch_groups is not None:
        print_local_main(f"evaluating uniform oracle for {run_name}...")
        trainer.model.train()
        oracle = trainer.multi_objective_gradient_oracle_over_batches(
            objective_batch_groups,
            as_numpy=True,
        )

    del trainer
    del optimizer
    del scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "final_checkpoint": final_dir,
        "lambda_helpful": helpful_weight,
        "lambda_harmless": harmless_weight,
        "max_steps": run_max_steps,
        "optimizer_steps": optimizer_step,
        "parameter_updates": optimizer_step,
        "objective_gradient_evals": num_objectives * optimizer_step,
        "oracle": oracle,
    }


def train_uniform_grid_moa_cycle(
    args: ScriptArguments,
    tokenizer,
    data_collator,
    helpful_train,
    harmless_train,
    helpful_eval,
    weights: Sequence[Tuple[float, float]],
    per_weight_steps: Sequence[int],
    objective_batch_groups: Optional[Dict[str, Sequence[Dict]]],
    smoothness: Sequence[float],
    num_objectives: int,
) -> None:
    """MOA-style uniform discretization for DPO-LW.

    Each uniform-grid lambda owns a LoRA parameter vector and its own AdamW
    optimizer state. The runner cycles over the grid, updates one lambda's
    current solution, then refreshes that point in the GN* metric bundle.
    """
    run_start_time = time.perf_counter()
    if args.update_data_source == "oracle" and objective_batch_groups is None:
        raise ValueError("MOA-style uniform with update_data_source='oracle' requires oracle batches.")
    if len(weights) != len(per_weight_steps):
        raise ValueError("weights and per_weight_steps must have the same length.")
    if args.bundle_dtype not in {"float32", "float64"}:
        raise ValueError("bundle_dtype must be either 'float32' or 'float64'.")

    target_update_counts = [int(max(0, steps)) for steps in per_weight_steps]
    total_update_budget = int(sum(target_update_counts))
    if total_update_budget <= 0:
        print_local_main("MOA-style uniform DPO-LW has zero update budget; saving initial checkpoints only.")

    print_local_main(
        "running MOA-style uniform DPO-LW: "
        f"{len(weights)} lambdas, total_updates={total_update_budget}, "
        f"update_data={args.update_data_source}, AdamW unchanged"
    )

    set_seeds(args.seed)
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
    solution_vectors = [initial_vector.copy() for _ in weights]
    initialized = [False for _ in weights]
    update_counts = [0 for _ in weights]

    run_names = []
    run_dirs = []
    history_paths = []
    for idx, ((helpful_weight, harmless_weight), max_steps_for_weight) in enumerate(
        zip(weights, target_update_counts)
    ):
        run_name = (
            f"lambda_helpful_{format_weight(helpful_weight)}"
            f"_harmless_{format_weight(harmless_weight)}"
        )
        run_dir = os.path.join(args.training_args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        config_path = os.path.join(run_dir, "dpo_lw_config.json")
        with open(config_path, "w") as handle:
            json.dump(
                {
                    **asdict(args),
                    "lambda_helpful": helpful_weight,
                    "lambda_harmless": harmless_weight,
                    "run_max_steps": max_steps_for_weight,
                    "uniform_update_mode": args.uniform_update_mode,
                    "grid_index": idx,
                },
                handle,
                indent=2,
                default=str,
            )
        history_path = os.path.join(run_dir, "training_history.jsonl")
        if os.path.exists(history_path) and args.training_args.overwrite_output_dir:
            os.remove(history_path)
        run_names.append(run_name)
        run_dirs.append(run_dir)
        history_paths.append(history_path)

    optimizers = [
        torch.optim.AdamW(
            trainable_params,
            lr=args.training_args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for _ in weights
    ]
    schedulers = []
    for optimizer, target_steps in zip(optimizers, target_update_counts):
        scheduler_steps = max(1, target_steps)
        warmup_steps = int(target_steps * args.warmup_ratio)
        schedulers.append(
            get_scheduler(
                args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=scheduler_steps,
            )
        )

    uniform_gn_history_path = os.path.join(args.training_args.output_dir, "uniform_gn_history.jsonl")
    if args.evaluate_uniform_gn and os.path.exists(uniform_gn_history_path) and args.training_args.overwrite_output_dir:
        os.remove(uniform_gn_history_path)

    uniform_bundle = None
    uniform_oracle_gradient_evals = 0
    if args.evaluate_uniform_gn:
        trainer.set_trainable_parameter_vector(initial_vector)
        initial_oracle = trainer.multi_objective_gradient_oracle_over_batches(
            objective_batch_groups,
            as_numpy=True,
        )
        uniform_bundle = FirstOrderBundle(
            K=num_objectives,
            d=int(initial_oracle["x"].shape[0]),
            L=np.asarray(smoothness, dtype=np.float64),
            dtype=np.dtype(args.bundle_dtype),
        )
        for _ in weights:
            uniform_bundle.add(
                initial_oracle["x"],
                initial_oracle["fvals"],
                initial_oracle["grads"],
            )

    cumulative_parameter_updates = 0
    grid_cursor = 0
    best_gn_star = float("inf")
    target_reached = False
    target_hit_record = None
    progress = tqdm(
        total=total_update_budget,
        disable=not Accelerator().is_local_main_process,
    )
    while cumulative_parameter_updates < total_update_budget:
        weight_idx = grid_cursor % len(weights)
        grid_cursor += 1
        if update_counts[weight_idx] >= target_update_counts[weight_idx]:
            if all(done >= target for done, target in zip(update_counts, target_update_counts)):
                break
            continue

        helpful_weight, harmless_weight = weights[weight_idx]
        if not initialized[weight_idx]:
            if args.chain_warm_start and weight_idx > 0:
                solution_vectors[weight_idx] = solution_vectors[weight_idx - 1].copy()
            else:
                solution_vectors[weight_idx] = initial_vector.copy()
            initialized[weight_idx] = True

        trainer.set_trainable_parameter_vector(solution_vectors[weight_idx])
        optimizer = optimizers[weight_idx]
        scheduler = schedulers[weight_idx]
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

        update_counts[weight_idx] += 1
        cumulative_parameter_updates += 1
        solution_vectors[weight_idx] = trainer.get_trainable_parameter_vector(cpu=True).numpy()

        train_record = {
            "micro_step": update_counts[weight_idx] * args.gradient_accumulation_steps,
            "optimizer_step": update_counts[weight_idx],
            "parameter_update": update_counts[weight_idx],
            "parameter_updates": update_counts[weight_idx],
            "cumulative_parameter_updates": cumulative_parameter_updates,
            "objective_gradient_evals": num_objectives * cumulative_parameter_updates,
            "elapsed_wall_seconds": time.perf_counter() - run_start_time,
            "lambda_helpful": helpful_weight,
            "lambda_harmless": harmless_weight,
            "max_steps": target_update_counts[weight_idx],
            "uniform_update_mode": args.uniform_update_mode,
            "update_data_source": args.update_data_source,
            "grid_index": weight_idx,
            "loss": float(update_stats["train_loss"]),
            "helpful_loss": float(update_stats["train_helpful_loss"]),
            "harmless_loss": float(update_stats["train_harmless_loss"]),
            "update_batches": int(update_stats["update_batches"]),
            "update_examples": int(update_stats["update_examples"]),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        save_jsonl(history_paths[weight_idx], train_record)

        if args.evaluate_uniform_gn:
            oracle = trainer.multi_objective_gradient_oracle_over_batches(
                objective_batch_groups,
                as_numpy=True,
            )
            uniform_bundle.replace(weight_idx, oracle["x"], oracle["fvals"], oracle["grads"])
            uniform_oracle_gradient_evals += 1
            gn_star, lam_star = maximise_gn(
                uniform_bundle,
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
            gn_record = {
                "method": "DPO-LW uniform",
                "uniform_update_mode": args.uniform_update_mode,
                "update_data_source": args.update_data_source,
                "gradient_eval": uniform_oracle_gradient_evals,
                "oracle_gradient_eval": uniform_oracle_gradient_evals,
                "checkpoint_index": cumulative_parameter_updates,
                "updated_grid_index": weight_idx,
                "run_parameter_updates": update_counts[weight_idx],
                "parameter_updates": cumulative_parameter_updates,
                "cumulative_parameter_updates": cumulative_parameter_updates,
                "objective_gradient_evals": num_objectives * cumulative_parameter_updates,
                "elapsed_wall_seconds": time.perf_counter() - run_start_time,
                "num_objectives": num_objectives,
                "bundle_size": uniform_bundle.m,
                "run": run_names[weight_idx],
                "lambda_solver": args.lambda_solver,
                "gn_certificate_type": (
                    "exact_two_objective_full_simplex"
                    if args.lambda_solver == "exact_k2"
                    else "local_solver_lower_bound"
                ),
                "lambda_train": [helpful_weight, harmless_weight],
                "lambda_gn_star": lam_star.tolist(),
                "gn_star": float(gn_star),
                "best_gn_star": float(best_gn_star),
                "best_gradient_norm": best_gradient_norm,
                "gn_target_norm": args.gn_target_norm,
                "target_reached": target_reached,
                "fvals": oracle["fvals"].tolist(),
                "update_counts": list(update_counts),
            }
            save_jsonl(uniform_gn_history_path, gn_record)
            if target_reached and target_hit_record is None:
                target_hit_record = dict(gn_record)
            print_local_main(
                f"uniform update={cumulative_parameter_updates} "
                f"grid={weight_idx + 1}/{len(weights)} "
                f"train_lambda={[round(helpful_weight, 4), round(harmless_weight, 4)]} "
                f"steps_for_lambda={update_counts[weight_idx]}/{target_update_counts[weight_idx]} "
                f"gn*={gn_star:.4e} "
                f"gn_lambda={[round(float(x), 4) for x in lam_star]}"
            )

        progress.set_description(
            f"grid={weight_idx + 1}/{len(weights)} "
            f"loss={train_record['loss']:.4f}"
        )
        progress.update(1)
        if target_reached:
            print_local_main(
                "uniform GN target reached: "
                f"best_norm={best_gradient_norm:.4e}, "
                f"target={float(args.gn_target_norm):.4e}, "
                f"updates={cumulative_parameter_updates}"
            )
            break

    progress.close()

    for weight_idx, run_dir in enumerate(run_dirs):
        trainer.set_trainable_parameter_vector(solution_vectors[weight_idx])
        final_dir = os.path.join(run_dir, "final_checkpoint")
        trainer.model.save_pretrained(final_dir)
        trainer.tokenizer.save_pretrained(final_dir)
        print_local_main(f"saved DPO-LW checkpoint to {final_dir}")

    del trainer
    for optimizer in optimizers:
        del optimizer
    for scheduler in schedulers:
        del scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.gn_target_norm is not None:
        with open(os.path.join(args.training_args.output_dir, "plateau_summary.json"), "w") as handle:
            json.dump(
                {
                    "method": "DPO-LW uniform",
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
                handle,
                indent=2,
            )


def main():
    run_start_time = time.perf_counter()
    args = tyro.cli(ScriptArguments)
    set_seeds(args.seed)
    os.makedirs(args.training_args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print_local_main("preparing DPO-LW datasets...")
    helpful_train, harmless_train, helpful_eval = prepare_datasets(args, tokenizer)
    data_collator = MODPODataCollatorWithPadding(tokenizer)
    weights = parse_helpful_weights(args)
    per_weight_steps = distribute_update_budget(
        len(weights),
        int(args.max_steps),
        args.total_update_budget,
    )
    if args.uniform_update_mode not in {"moa_cycle", "independent"}:
        raise ValueError("uniform_update_mode must be either 'moa_cycle' or 'independent'.")
    if args.update_data_source not in {"train", "oracle"}:
        raise ValueError("update_data_source must be either 'train' or 'oracle'.")
    smoothness = parse_float_list(args.smoothness)
    if len(smoothness) != 2:
        raise ValueError("DPO-LW GN comparison expects exactly two smoothness constants.")
    num_objectives = len(smoothness)
    if args.bundle_dtype not in {"float32", "float64"}:
        raise ValueError("bundle_dtype must be either 'float32' or 'float64'.")
    if args.gn_target_norm is not None and float(args.gn_target_norm) <= 0.0:
        raise ValueError("gn_target_norm must be positive when provided.")
    if args.gn_target_norm is not None and args.uniform_update_mode != "moa_cycle":
        raise ValueError("gn_target_norm is currently supported only for uniform_update_mode='moa_cycle'.")
    if args.lambda_solver not in LAMBDA_SOLVERS:
        raise ValueError(
            "lambda_solver must be one of: "
            + ", ".join(sorted(LAMBDA_SOLVERS))
            + "."
        )
    if args.require_ipopt and args.lambda_solver == "ipopt" and not ipopt_available():
        raise RuntimeError(
            "IPOPT was required for DPO-LW GN evaluation, but cyipopt/IPOPT "
            "is unavailable. Install IPOPT + cyipopt on the training machine "
            "before running GN comparison. "
            f"Import error: {ipopt_import_error()!r}"
        )

    weights_path = os.path.join(args.training_args.output_dir, "weights.json")
    with open(weights_path, "w") as handle:
        json.dump(
            [
                {
                    "lambda_helpful": helpful,
                    "lambda_harmless": harmless,
                    "max_steps": per_weight_steps[idx],
                }
                for idx, (helpful, harmless) in enumerate(weights)
            ],
            handle,
            indent=2,
        )

    objective_batch_groups = None
    uniform_bundle = None
    cumulative_parameter_updates = 0
    uniform_oracle_gradient_evals = 0
    uniform_gn_history_path = os.path.join(args.training_args.output_dir, "uniform_gn_history.jsonl")
    if args.evaluate_uniform_gn or args.update_data_source == "oracle":
        helpful_oracle = select_subset(
            helpful_train,
            args.oracle_subset_size_per_objective,
            args.seed + 2,
            "helpful fixed oracle for uniform GN",
        )
        harmless_oracle = select_subset(
            harmless_train,
            args.oracle_subset_size_per_objective,
            args.seed + 2 if args.agreement_ratio is not None else args.seed + 3,
            "harmless fixed oracle for uniform GN",
        )
        objective_batch_groups = {
            "helpful": make_fixed_batches(helpful_oracle, args.oracle_batch_size, data_collator),
            "harmless": make_fixed_batches(harmless_oracle, args.oracle_batch_size, data_collator),
        }
        if os.path.exists(uniform_gn_history_path) and args.training_args.overwrite_output_dir:
            os.remove(uniform_gn_history_path)

    if args.uniform_update_mode == "moa_cycle":
        train_uniform_grid_moa_cycle(
            args=args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            helpful_train=helpful_train,
            harmless_train=harmless_train,
            helpful_eval=helpful_eval,
            weights=weights,
            per_weight_steps=per_weight_steps,
            objective_batch_groups=objective_batch_groups,
            smoothness=smoothness,
            num_objectives=num_objectives,
        )
        return

    for weight_idx, (helpful_weight, harmless_weight) in enumerate(weights):
        result = train_one_weight(
            args,
            tokenizer,
            data_collator,
            helpful_train,
            harmless_train,
            helpful_eval,
            helpful_weight,
            harmless_weight,
            objective_batch_groups=objective_batch_groups,
            num_objectives=num_objectives,
            max_steps_override=per_weight_steps[weight_idx],
        )
        run_parameter_updates = int(result.get("parameter_updates", 0))
        cumulative_parameter_updates += run_parameter_updates

        oracle = result.get("oracle")
        if oracle is None:
            continue

        if uniform_bundle is None:
            uniform_bundle = FirstOrderBundle(
                K=2,
                d=int(oracle["x"].shape[0]),
                L=np.asarray(smoothness, dtype=np.float64),
                dtype=np.dtype(args.bundle_dtype),
            )
        uniform_bundle.add(oracle["x"], oracle["fvals"], oracle["grads"])
        uniform_oracle_gradient_evals += 1
        gn_star, lam_star = maximise_gn(
            uniform_bundle,
            max_starts=args.lambda_max_starts,
            solver=args.lambda_solver,
            require_ipopt=args.require_ipopt,
        )
        record = {
            "method": "DPO-LW uniform",
            "gradient_eval": uniform_oracle_gradient_evals,
            "oracle_gradient_eval": uniform_oracle_gradient_evals,
            "checkpoint_index": uniform_bundle.m,
            "run_parameter_updates": run_parameter_updates,
            "parameter_updates": cumulative_parameter_updates,
            "cumulative_parameter_updates": cumulative_parameter_updates,
            "objective_gradient_evals": num_objectives * cumulative_parameter_updates,
            "elapsed_wall_seconds": time.perf_counter() - run_start_time,
            "num_objectives": num_objectives,
            "run": result["run_name"],
            "lambda_solver": args.lambda_solver,
            "gn_certificate_type": (
                "exact_two_objective_full_simplex"
                if args.lambda_solver == "exact_k2"
                else "local_solver_lower_bound"
            ),
            "lambda_train": [helpful_weight, harmless_weight],
            "lambda_gn_star": lam_star.tolist(),
            "gn_star": float(gn_star),
            "fvals": oracle["fvals"].tolist(),
        }
        save_jsonl(uniform_gn_history_path, record)
        print_local_main(
            f"uniform checkpoint={uniform_bundle.m} updates={cumulative_parameter_updates} "
            f"oracle_eval={uniform_oracle_gradient_evals} "
            f"train_lambda={[round(helpful_weight, 4), round(harmless_weight, 4)]} "
            f"gn*={gn_star:.4e}"
        )


if __name__ == "__main__":
    main()
