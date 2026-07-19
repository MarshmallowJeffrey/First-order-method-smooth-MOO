from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scripts.modpo.adaptive_bundle.bundle_core import (
    FirstOrderBundle,
    ipopt_available,
    ipopt_import_error,
    maximise_gn,
)
import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
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

    weight_resolution: Optional[int] = field(default=5)
    helpful_weights: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated helpful weights. Overrides weight_resolution."},
    )
    max_steps: Optional[int] = field(default=300)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    warmup_ratio: Optional[float] = field(default=0.03)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    weight_decay: Optional[float] = field(default=0.0)
    max_grad_norm: Optional[float] = field(default=1.0)
    save_every_steps: Optional[int] = field(default=0)
    evaluate_uniform_gn: Optional[bool] = field(default=True)
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
) -> Dict:
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
            },
            handle,
            indent=2,
            default=str,
        )

    print_local_main(
        f"training DPO-LW {run_name}: "
        f"loss={helpful_weight:.4f}*F_helpful+{harmless_weight:.4f}*F_harmless"
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
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    history_path = os.path.join(run_dir, "training_history.jsonl")
    if os.path.exists(history_path) and args.training_args.overwrite_output_dir:
        os.remove(history_path)

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    total_micro_steps = args.max_steps * args.gradient_accumulation_steps
    progress = tqdm(range(1, total_micro_steps + 1), disable=not Accelerator().is_local_main_process)
    for micro_step in progress:
        loss, helpful_loss, harmless_loss = weighted_dpo_loss(
            trainer,
            helpful_loader.next(),
            harmless_loader.next(),
            helpful_weight,
            harmless_weight,
        )
        scaled_loss = loss / args.gradient_accumulation_steps
        scaled_loss.backward()

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
            "lambda_helpful": helpful_weight,
            "lambda_harmless": harmless_weight,
            "loss": float(loss.detach().cpu()),
            "helpful_loss": float(helpful_loss.cpu()),
            "harmless_loss": float(harmless_loss.cpu()),
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
        "optimizer_steps": optimizer_step,
        "parameter_updates": optimizer_step,
        "objective_gradient_evals": num_objectives * optimizer_step,
        "oracle": oracle,
    }


def main():
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
    smoothness = parse_float_list(args.smoothness)
    if len(smoothness) != 2:
        raise ValueError("DPO-LW GN comparison expects exactly two smoothness constants.")
    num_objectives = len(smoothness)
    if args.bundle_dtype not in {"float32", "float64"}:
        raise ValueError("bundle_dtype must be either 'float32' or 'float64'.")
    if args.lambda_solver not in {"ipopt", "slsqp"}:
        raise ValueError("lambda_solver must be either 'ipopt' or 'slsqp'.")
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
                {"lambda_helpful": helpful, "lambda_harmless": harmless}
                for helpful, harmless in weights
            ],
            handle,
            indent=2,
        )

    objective_batch_groups = None
    uniform_bundle = None
    cumulative_parameter_updates = 0
    uniform_oracle_gradient_evals = 0
    uniform_gn_history_path = os.path.join(args.training_args.output_dir, "uniform_gn_history.jsonl")
    if args.evaluate_uniform_gn:
        helpful_oracle = select_subset(
            helpful_train,
            args.oracle_subset_size_per_objective,
            args.seed + 2,
            "helpful fixed oracle for uniform GN",
        )
        harmless_oracle = select_subset(
            harmless_train,
            args.oracle_subset_size_per_objective,
            args.seed + 3,
            "harmless fixed oracle for uniform GN",
        )
        objective_batch_groups = {
            "helpful": make_fixed_batches(helpful_oracle, args.oracle_batch_size, data_collator),
            "harmless": make_fixed_batches(harmless_oracle, args.oracle_batch_size, data_collator),
        }
        if os.path.exists(uniform_gn_history_path) and args.training_args.overwrite_output_dir:
            os.remove(uniform_gn_history_path)

    for helpful_weight, harmless_weight in weights:
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
            "num_objectives": num_objectives,
            "run": result["run_name"],
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
