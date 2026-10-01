from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import tqdm
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.modpo.beavertails.utils.score_model import LlamaForScore
from src.data.configs import DATASET_CONFIGS
from src.data.raw_data.safe_rlhf import agreement_stats, sample_agreement_mixture
from src.trainer.modpo_trainer import MODPODataMapFunc
from src.utils import disable_progress_bar_non_local_main, set_seeds


disable_progress_bar_non_local_main()

QWEN_PROMPT_TEMPLATE = (
    "<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n"
)


@dataclass
class CheckpointSpec:
    method: str
    run: str
    adapter_dir: Optional[Path]
    lambda_helpful: Optional[float]
    lambda_harmless: Optional[float]
    outer: Optional[int] = None
    parameter_updates: Optional[int] = None
    objective_gradient_evals: Optional[int] = None
    elapsed_wall_seconds: Optional[float] = None
    source: str = ""


@dataclass
class RewardPoint:
    method: str
    run: str
    mean_reward: float
    mean_cost: float
    lambda_helpful: Optional[float]
    lambda_harmless: Optional[float]
    outer: Optional[int]
    parameter_updates: Optional[int]
    objective_gradient_evals: Optional[int]
    elapsed_wall_seconds: Optional[float]
    adapter_dir: Optional[str]
    generation_dir: str
    score_dir: str
    source: str

    @property
    def mean_safety(self) -> float:
        """Safety score used for plotting; higher means lower raw cost."""
        return -float(self.mean_cost)


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


def config_first(config: Dict, *keys, default=None):
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y"}


def safe_name(value: str) -> str:
    value = value.strip().replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    return value.strip("_") or "checkpoint"


def select_shuffled_window(dataset, size, seed: int, name: str, start: int = 0):
    if start < 0:
        raise ValueError("start must be non-negative")
    shuffled = dataset.shuffle(seed=seed)
    if start >= len(shuffled):
        print(f"{name}: selected 0 / {len(shuffled)} samples from offset {start}")
        return shuffled.select([])
    if size is None or size <= 0:
        stop = len(shuffled)
    else:
        stop = min(len(shuffled), start + int(size))
    print(f"{name}: selected {stop - start} / {len(shuffled)} samples from offset {start}")
    return shuffled.select(range(start, stop))


def select_subset(dataset, size, seed: int, name: str):
    return select_shuffled_window(dataset, size, seed, name, start=0)


def resolve_logged_path(path_value: Optional[str], run_dir: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    if path.is_absolute():
        return path

    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate

    candidate = run_dir / path
    if candidate.exists():
        return candidate

    # Some logs store a relative ./output/... path. If the run directory was
    # moved, the basename is still useful for outer checkpoint directories.
    fallback = run_dir / path.name
    if fallback.exists():
        return fallback
    return path


def lambda_from_run_name(run_name: str) -> Tuple[Optional[float], Optional[float]]:
    patterns = [
        r"lambda_helpful_([^_]+)_harmless_([^_]+)",
        r"lambda_([^_]+)_([^_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, run_name)
        if not match:
            continue
        helpful = match.group(1).replace("p", ".")
        harmless = match.group(2).replace("p", ".")
        try:
            return float(helpful), float(harmless)
        except ValueError:
            return None, None
    return None, None


def collect_adaptive_checkpoints(
    adaptive_dir: Optional[Path],
    *,
    include_final: bool,
    method_label: Optional[str] = None,
) -> List[CheckpointSpec]:
    if adaptive_dir is None or not adaptive_dir.exists():
        return []

    specs: List[CheckpointSpec] = []
    history = read_jsonl(adaptive_dir / "adaptive_history.jsonl")
    is_surf_run = any(
        record.get("method") == "SURF" or record.get("surf_outer") is not None
        for record in history
    )
    final_surf_outer = None
    if is_surf_run:
        surf_outers = [
            int(record["surf_outer"])
            for record in history
            if record.get("surf_outer") is not None
        ]
        final_surf_outer = max(surf_outers) if surf_outers else None
    for record in history:
        if is_surf_run and final_surf_outer is not None:
            if record.get("surf_outer") is None or int(record["surf_outer"]) != int(final_surf_outer):
                continue
        checkpoint_dir = resolve_logged_path(record.get("checkpoint_dir"), adaptive_dir)
        if checkpoint_dir is None or not checkpoint_dir.exists():
            outer = record.get("outer")
            fallback = adaptive_dir / f"outer_{outer}"
            checkpoint_dir = fallback if fallback.exists() else None
        if checkpoint_dir is None:
            continue

        lam = record.get("lambda") or []
        lambda_helpful = float(lam[0]) if len(lam) > 0 else None
        lambda_harmless = float(lam[1]) if len(lam) > 1 else None
        method = str(method_label or record.get("method", "Adaptive bundle"))
        outer = int(record["outer"]) if record.get("outer") is not None else None
        run = str(record.get("run") or (f"outer_{outer:03d}" if outer is not None else checkpoint_dir.name))
        specs.append(
            CheckpointSpec(
                method=method,
                run=run,
                adapter_dir=checkpoint_dir,
                lambda_helpful=lambda_helpful,
                lambda_harmless=lambda_harmless,
                outer=outer,
                parameter_updates=record.get("parameter_updates_after"),
                objective_gradient_evals=record.get("objective_gradient_evals_after"),
                elapsed_wall_seconds=record.get("elapsed_wall_seconds_after"),
                source=record.get("source", "adaptive_outer_checkpoint"),
            )
        )

    if include_final:
        final_dir = adaptive_dir / "final_checkpoint"
        if final_dir.exists():
            last = history[-1] if history else {}
            lam = last.get("lambda") or []
            method = str(method_label or last.get("method", "Adaptive bundle"))
            specs.append(
                CheckpointSpec(
                    method=method,
                    run="final",
                    adapter_dir=final_dir,
                    lambda_helpful=float(lam[0]) if len(lam) > 0 else None,
                    lambda_harmless=float(lam[1]) if len(lam) > 1 else None,
                    outer=last.get("outer"),
                    parameter_updates=last.get("parameter_updates_after"),
                    objective_gradient_evals=last.get("objective_gradient_evals_after"),
                    elapsed_wall_seconds=last.get("elapsed_wall_seconds_after"),
                    source="adaptive_final_checkpoint",
                )
            )

    return dedupe_checkpoints(specs)


def collect_dpo_lw_checkpoints(dpo_lw_dir: Optional[Path]) -> List[CheckpointSpec]:
    if dpo_lw_dir is None or not dpo_lw_dir.exists():
        return []

    specs: List[CheckpointSpec] = []
    history_by_run: Dict[str, Dict] = {}
    for record in read_jsonl(dpo_lw_dir / "uniform_gn_history.jsonl"):
        run = record.get("run")
        if run:
            history_by_run[run] = record

    for final_dir in sorted(dpo_lw_dir.glob("lambda_helpful_*_harmless_*/final_checkpoint")):
        run_dir = final_dir.parent
        config = read_json(run_dir / "dpo_lw_config.json")
        record = history_by_run.get(run_dir.name, {})
        lambda_train = record.get("lambda_train") or []
        lambda_helpful = first_present(
            config.get("lambda_helpful"),
            lambda_train[0] if len(lambda_train) > 0 else None,
            lambda_from_run_name(run_dir.name)[0],
        )
        lambda_harmless = first_present(
            config.get("lambda_harmless"),
            lambda_train[1] if len(lambda_train) > 1 else None,
            lambda_from_run_name(run_dir.name)[1],
        )
        specs.append(
            CheckpointSpec(
                method="Uniform DPO-LW",
                run=run_dir.name,
                adapter_dir=final_dir,
                lambda_helpful=float(lambda_helpful) if lambda_helpful is not None else None,
                lambda_harmless=float(lambda_harmless) if lambda_harmless is not None else None,
                parameter_updates=record.get("parameter_updates"),
                objective_gradient_evals=record.get("objective_gradient_evals"),
                elapsed_wall_seconds=record.get("elapsed_wall_seconds"),
                source="dpo_lw_final_checkpoint",
            )
        )

    return dedupe_checkpoints(specs)


def load_oracle_reference_config(args: argparse.Namespace) -> Tuple[Dict, Path]:
    candidates: List[Path] = []
    reference_values: List[Optional[str]] = [args.oracle_reference_dir, args.adaptive_dir]
    if getattr(args, "adaptive_run", None):
        reference_values.extend(run_dir for run_dir, _ in args.adaptive_run)
    reference_values.append(args.dpo_lw_dir)

    for value in reference_values:
        if value:
            root = Path(value)
            candidates.append(root / "adaptive_config.json")
            candidates.append(root / "dpo_lw_config.json")
            candidates.extend(sorted(root.glob("lambda_helpful_*_harmless_*/dpo_lw_config.json")))

    for path in candidates:
        if path.exists():
            return read_json(path), path
    raise FileNotFoundError(
        "Could not find adaptive_config.json or dpo_lw_config.json for oracle prompt "
        "reconstruction. Pass --oracle_reference_dir to the training run directory."
    )


def map_raw_preference_dataset(raw_dataset, rdp, name: str):
    print(f"mapping {name} raw rows to preference format...")
    return raw_dataset.map(
        rdp._dataset_to_preference_formatter,
        num_proc=rdp.num_proc,
        remove_columns=raw_dataset.column_names,
    )


def make_controlled_agreement_pools(
    better_rdp,
    safer_rdp,
    *,
    split: str,
    size: int,
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
    print(
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


def preprocess_preference_dataset_for_oracle(dataset, tokenizer, max_length: int, num_proc: int):
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


def interleaved_prompt_dataset(
    helpful_dataset,
    harmless_dataset,
    *,
    limit: int,
) -> Dataset:
    rows = []
    seen_prompts = set()
    max_len = max(len(helpful_dataset), len(harmless_dataset))

    def maybe_add(sample, source: str) -> bool:
        prompt = sample.get("prompt")
        if prompt is None or prompt in seen_prompts:
            return False
        seen_prompts.add(prompt)
        rows.append({
            "prompt": prompt,
            "raw_prompt": sample.get("raw_prompt", ""),
            "objective_source": source,
        })
        return limit > 0 and len(rows) >= limit

    for idx in range(max_len):
        if idx < len(helpful_dataset) and maybe_add(helpful_dataset[idx], "helpful"):
            break
        if idx < len(harmless_dataset) and maybe_add(harmless_dataset[idx], "harmless"):
            break
    return Dataset.from_list(rows)


def load_oracle_prompt_dataset(args: argparse.Namespace) -> Dataset:
    config, config_path = load_oracle_reference_config(args)
    print(f"reconstructing oracle prompts from {config_path}")

    better_dataset_name = config_first(
        config,
        "better_dataset_name",
        "helpful_dataset_name",
        default="PKU-Alignment/PKU-SafeRLHF-10K-better",
    )
    safer_dataset_name = config_first(
        config,
        "safer_dataset_name",
        "harmless_dataset_name",
        default="PKU-Alignment/PKU-SafeRLHF-10K-safer",
    )
    prompt_template = config_first(config, "prompt_template", default=args.prompt_template)
    sanity_check = bool(config_first(config, "sanity_check", default=False))
    seed = int(config_first(config, "seed", default=args.seed))
    train_size = config_first(config, "train_subset_size_per_objective", default=None)
    oracle_size = int(config_first(config, "oracle_subset_size_per_objective", default=0) or 0)
    agreement_ratio = config_first(config, "agreement_ratio", default=None)
    shared_objective_subset = bool(config_first(config, "shared_objective_subset", default=False))
    max_length = int(config_first(config, "max_length", default=args.prompt_max_length))
    num_proc = int(config_first(config, "num_proc", default=1))

    if train_size is not None:
        train_size = int(train_size)

    better_rdp = DATASET_CONFIGS[better_dataset_name](
        prompt_template=prompt_template,
        sanity_check=sanity_check,
    )
    safer_rdp = DATASET_CONFIGS[safer_dataset_name](
        prompt_template=prompt_template,
        sanity_check=sanity_check,
    )

    if agreement_ratio is not None:
        better_pool, safer_pool = make_controlled_agreement_pools(
            better_rdp,
            safer_rdp,
            split="train",
            size=train_size,
            agreement_ratio=float(agreement_ratio),
            seed=seed,
        )
    else:
        better_pool = select_subset(
            better_rdp.get_preference_dataset(split="train"),
            train_size,
            seed,
            "helpful/better train pool",
        )
        safer_pool = select_subset(
            safer_rdp.get_preference_dataset(split="train"),
            train_size,
            seed if shared_objective_subset else seed + 1,
            "harmless/safer train pool",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    better_train = preprocess_preference_dataset_for_oracle(
        better_pool,
        tokenizer,
        max_length,
        num_proc,
    )
    safer_train = preprocess_preference_dataset_for_oracle(
        safer_pool,
        tokenizer,
        max_length,
        num_proc,
    )

    helpful_seed = seed + 2
    harmless_seed = (
        seed + 2
        if shared_objective_subset or agreement_ratio is not None
        else seed + 3
    )
    if args.eval_prompt_source == "oracle":
        offset = 0
        source_label = "fixed oracle"
    elif args.eval_prompt_source == "oracle_heldout":
        offset = oracle_size
        source_label = "oracle-heldout training-pool"
    else:
        raise ValueError(f"Unsupported oracle eval_prompt_source={args.eval_prompt_source!r}")

    better_eval = select_shuffled_window(
        better_train,
        oracle_size if args.eval_prompt_source == "oracle" else args.eval_size,
        helpful_seed,
        f"helpful/better {source_label}",
        start=offset,
    )
    safer_eval = select_shuffled_window(
        safer_train,
        oracle_size if args.eval_prompt_source == "oracle" else args.eval_size,
        harmless_seed,
        f"harmless/safer {source_label}",
        start=offset,
    )
    limit = int(args.eval_size or 0)
    eval_dataset = interleaved_prompt_dataset(better_eval, safer_eval, limit=limit)
    print(
        f"{args.eval_prompt_source}: selected {len(eval_dataset)} unique prompts "
        f"from helpful={len(better_eval)} harmless={len(safer_eval)}"
    )
    return eval_dataset


def dedupe_checkpoints(specs: Sequence[CheckpointSpec]) -> List[CheckpointSpec]:
    seen = set()
    deduped = []
    for spec in specs:
        key = (spec.method, spec.run, str(spec.adapter_dir))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def load_eval_prompts(args: argparse.Namespace) -> Dataset:
    if args.eval_prompt_source in {"oracle", "oracle_heldout"}:
        return load_oracle_prompt_dataset(args)
    if args.eval_prompt_source != "split":
        raise ValueError(
            "eval_prompt_source must be one of: split, oracle, oracle_heldout"
        )
    if args.dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset_name={args.dataset_name!r}. Use one of: "
            f"{', '.join(sorted(DATASET_CONFIGS))}"
        )
    rdp = DATASET_CONFIGS[args.dataset_name](prompt_template=args.prompt_template)
    eval_dataset = rdp.get_sft_dataset(split=args.eval_split)
    if args.eval_size and args.eval_size > 0:
        eval_dataset = eval_dataset.select(range(min(args.eval_size, len(eval_dataset))))
    return eval_dataset


def load_generation_model(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def active_generation_model(base_model, spec: CheckpointSpec, adapter_name: str):
    if spec.adapter_dir is None:
        return base_model
    if not isinstance(base_model, PeftModel):
        model = PeftModel.from_pretrained(
            base_model,
            str(spec.adapter_dir),
            adapter_name=adapter_name,
        )
        model.set_adapter(adapter_name)
        model.eval()
        return model
    if adapter_name not in base_model.peft_config:
        base_model.load_adapter(str(spec.adapter_dir), adapter_name=adapter_name)
    base_model.set_adapter(adapter_name)
    base_model.eval()
    return base_model


def generation_file(generation_dir: Path) -> Path:
    return generation_dir / "00001-of-00001.jsonl"


def generate_checkpoint_responses(
    specs: Sequence[CheckpointSpec],
    eval_dataset: Dataset,
    args: argparse.Namespace,
    output_dir: Path,
) -> List[Tuple[CheckpointSpec, Path]]:
    gen_root = output_dir / "generations"
    gen_root.mkdir(parents=True, exist_ok=True)

    missing = []
    outputs = []
    for spec in specs:
        run_dir = gen_root / safe_name(f"{spec.method}_{spec.run}")
        out_file = generation_file(run_dir)
        outputs.append((spec, run_dir))
        if args.force or not out_file.exists():
            missing.append((spec, run_dir))

    if not missing:
        return outputs

    model, tokenizer = load_generation_model(args)
    try:
        for spec, run_dir in missing:
            run_dir.mkdir(parents=True, exist_ok=True)
            out_file = generation_file(run_dir)
            adapter_name = safe_name(f"{spec.method}_{spec.run}")
            active_model = active_generation_model(model, spec, adapter_name)
            model = active_model

            results = []
            for idx in tqdm.tqdm(
                range(0, len(eval_dataset), args.generation_batch_size),
                desc=f"generate {spec.method} {spec.run}",
            ):
                batch = eval_dataset[idx : idx + args.generation_batch_size]
                tokenized = tokenizer(
                    batch["prompt"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.prompt_max_length,
                )
                tokenized = {key: value.cuda() for key, value in tokenized.items()}
                with torch.no_grad():
                    generated = active_model.generate(
                        **tokenized,
                        max_length=args.generation_max_length,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for sample in decoded:
                    results.append({"prompt_response": sample})

            Dataset.from_list(results).to_json(str(out_file))
            write_checkpoint_metadata(run_dir / "checkpoint.json", spec)
    finally:
        del model
        torch.cuda.empty_cache()

    return outputs


def write_checkpoint_metadata(path: Path, spec: CheckpointSpec) -> None:
    payload = {
        "method": spec.method,
        "run": spec.run,
        "adapter_dir": str(spec.adapter_dir) if spec.adapter_dir is not None else None,
        "lambda_helpful": spec.lambda_helpful,
        "lambda_harmless": spec.lambda_harmless,
        "outer": spec.outer,
        "parameter_updates": spec.parameter_updates,
        "objective_gradient_evals": spec.objective_gradient_evals,
        "elapsed_wall_seconds": spec.elapsed_wall_seconds,
        "source": spec.source,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def read_generation_jsonl(path: Path) -> List[str]:
    responses = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            responses.append(json.loads(line)["prompt_response"])
    return responses


def load_score_models(args: argparse.Namespace):
    reward = LlamaForScore.from_pretrained(
        args.reward_model_name,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        device_map="auto",
    )
    reward.eval()
    cost = LlamaForScore.from_pretrained(
        args.cost_model_name,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        device_map="auto",
    )
    cost.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return reward, cost, tokenizer


def load_one_score_model(model_name: str, args: argparse.Namespace):
    model = LlamaForScore.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def score_with_model(
    model,
    tokenizer,
    responses: Sequence[str],
    args: argparse.Namespace,
    desc: str,
) -> List[float]:
    scores: List[float] = []
    for idx in tqdm.tqdm(
        range(0, len(responses), args.score_batch_size),
        desc=desc,
    ):
        batch = list(responses[idx : idx + args.score_batch_size])
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.score_max_length,
        )
        tokenized = {key: value.cuda() for key, value in tokenized.items()}
        with torch.no_grad():
            output = model(tokenized["input_ids"], tokenized["attention_mask"])
        scores.extend(output.end_scores.view(-1).detach().cpu().tolist())
    return [float(score) for score in scores]


def write_score_outputs(
    spec: CheckpointSpec,
    score_dir: Path,
    responses: Sequence[str],
    reward_scores: Sequence[float],
    cost_scores: Sequence[float],
) -> None:
    score_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [
        {
            "prompt_response": prompt_response,
            "reward": float(reward_score),
            "cost": float(cost_score),
        }
        for prompt_response, reward_score, cost_score in zip(
            responses,
            reward_scores,
            cost_scores,
        )
    ]
    Dataset.from_list(raw_rows).to_json(str(score_dir / "raw.jsonl"))
    mean_reward = sum(row["reward"] for row in raw_rows) / len(raw_rows)
    mean_cost = sum(row["cost"] for row in raw_rows) / len(raw_rows)
    with (score_dir / "mean.csv").open("w") as handle:
        handle.write("mean reward,mean cost\n")
        handle.write(f"{mean_reward},{mean_cost}\n")
    write_checkpoint_metadata(score_dir / "checkpoint.json", spec)


def score_generations(
    generated: Sequence[Tuple[CheckpointSpec, Path]],
    args: argparse.Namespace,
    output_dir: Path,
) -> List[RewardPoint]:
    score_root = output_dir / "scores"
    score_root.mkdir(parents=True, exist_ok=True)

    pending = []
    for spec, gen_dir in generated:
        score_dir = score_root / gen_dir.name
        mean_path = score_dir / "mean.csv"
        if args.force or not mean_path.exists():
            pending.append((spec, gen_dir, score_dir))

    if pending and args.score_load_mode == "both":
        reward_model, cost_model, tokenizer = load_score_models(args)
        try:
            for spec, gen_dir, score_dir in pending:
                responses = read_generation_jsonl(generation_file(gen_dir))
                rewards = score_with_model(
                    reward_model,
                    tokenizer,
                    responses,
                    args,
                    desc=f"reward {spec.method} {spec.run}",
                )
                costs = []
                for idx in tqdm.tqdm(
                    range(0, len(responses), args.score_batch_size),
                    desc=f"cost {spec.method} {spec.run}",
                ):
                    batch = responses[idx : idx + args.score_batch_size]
                    tokenized = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.score_max_length,
                    )
                    tokenized = {key: value.cuda() for key, value in tokenized.items()}
                    with torch.no_grad():
                        cost_output = cost_model(
                            tokenized["input_ids"],
                            tokenized["attention_mask"],
                        )
                    costs.extend(cost_output.end_scores.view(-1).detach().cpu().tolist())
                write_score_outputs(spec, score_dir, responses, rewards, costs)
        finally:
            del reward_model
            del cost_model
            torch.cuda.empty_cache()
    elif pending:
        cached_responses = {
            gen_dir: read_generation_jsonl(generation_file(gen_dir))
            for _, gen_dir, _ in pending
        }
        reward_scores_by_dir: Dict[Path, List[float]] = {}
        reward_model, reward_tokenizer = load_one_score_model(args.reward_model_name, args)
        try:
            for spec, gen_dir, _ in pending:
                reward_scores_by_dir[gen_dir] = score_with_model(
                    reward_model,
                    reward_tokenizer,
                    cached_responses[gen_dir],
                    args,
                    desc=f"reward {spec.method} {spec.run}",
                )
        finally:
            del reward_model
            torch.cuda.empty_cache()

        cost_scores_by_dir: Dict[Path, List[float]] = {}
        cost_model, cost_tokenizer = load_one_score_model(args.cost_model_name, args)
        try:
            for spec, gen_dir, _ in pending:
                cost_scores_by_dir[gen_dir] = score_with_model(
                    cost_model,
                    cost_tokenizer,
                    cached_responses[gen_dir],
                    args,
                    desc=f"cost {spec.method} {spec.run}",
                )
        finally:
            del cost_model
            torch.cuda.empty_cache()

        for spec, gen_dir, score_dir in pending:
            write_score_outputs(
                spec,
                score_dir,
                cached_responses[gen_dir],
                reward_scores_by_dir[gen_dir],
                cost_scores_by_dir[gen_dir],
            )

    points = []
    for spec, gen_dir in generated:
        score_dir = score_root / gen_dir.name
        mean_reward, mean_cost = read_mean_score(score_dir / "mean.csv")
        points.append(
            RewardPoint(
                method=spec.method,
                run=spec.run,
                mean_reward=mean_reward,
                mean_cost=mean_cost,
                lambda_helpful=spec.lambda_helpful,
                lambda_harmless=spec.lambda_harmless,
                outer=spec.outer,
                parameter_updates=spec.parameter_updates,
                objective_gradient_evals=spec.objective_gradient_evals,
                elapsed_wall_seconds=spec.elapsed_wall_seconds,
                adapter_dir=str(spec.adapter_dir) if spec.adapter_dir is not None else None,
                generation_dir=str(gen_dir),
                score_dir=str(score_dir),
                source=spec.source,
            )
        )
    return points


def read_mean_score(path: Path) -> Tuple[float, float]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    return float(row["mean reward"]), float(row["mean cost"])


def nondominated_reward_cost(points: Sequence[RewardPoint]) -> List[RewardPoint]:
    if not points:
        return []
    keep = []
    values = [(point.mean_reward, point.mean_safety) for point in points]
    for idx, (reward, safety) in enumerate(values):
        dominated = False
        for jdx, (other_reward, other_safety) in enumerate(values):
            if idx == jdx:
                continue
            no_worse = other_reward >= reward and other_safety >= safety
            strict = other_reward > reward or other_safety > safety
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            keep.append(points[idx])
    keep.sort(key=lambda point: point.mean_reward)
    return keep


def write_points_csv(points: Sequence[RewardPoint], output_dir: Path) -> Path:
    path = output_dir / "reward_pareto_points.csv"
    fieldnames = [
        "method",
        "run",
        "mean_reward",
        "mean_cost",
        "mean_safety",
        "lambda_helpful",
        "lambda_harmless",
        "outer",
        "parameter_updates",
        "objective_gradient_evals",
        "elapsed_wall_seconds",
        "adapter_dir",
        "generation_dir",
        "score_dir",
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow({field: getattr(point, field) for field in fieldnames})
    return path


def plot_reward_pareto(
    points: Sequence[RewardPoint],
    output_dir: Path,
    annotate: bool,
    plot_style: str = "default",
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not points:
        return None

    if plot_style not in {"default", "surf", "publication"}:
        raise ValueError("plot_style must be one of: default, surf, publication")

    surf_style = plot_style in {"surf", "publication"}
    publication_style = plot_style == "publication"
    output_path = output_dir / (
        "reward_cost_pareto_front_publication.png"
        if publication_style
        else (
            "reward_cost_pareto_front_surf.png"
            if surf_style
            else "reward_cost_pareto_front.png"
        )
    )
    method_specs = {
        "Uniform DPO-LW": {"color": "#4C78A8", "marker": "s"},
        "Adaptive bundle": {"color": "#F58518", "marker": "o"},
        "Adaptive exact_k2": {"color": "#F58518", "marker": "o"},
        "Adaptive IPOPT": {"color": "#E45756", "marker": "^"},
        "SURF": {"color": "#54A24B", "marker": "P"},
        "SFT": {"color": "#8A8A8A", "marker": "D"},
    }
    fallback_colors = [
        "#B279A2",
        "#72B7B2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]
    fallback_markers = ["X", "v", "<", ">", "*"]

    if publication_style:
        method_specs = {
            "Uniform DPO-LW": {"color": "#1F77B4", "marker": "s", "linestyle": "--"},
            "Adaptive bundle": {"color": "#FF7F0E", "marker": "o", "linestyle": "-"},
            "Adaptive exact_k2": {"color": "#FF7F0E", "marker": "o", "linestyle": "-"},
            "Adaptive IPOPT": {"color": "#E45756", "marker": "o", "linestyle": "-"},
            "SURF": {"color": "#D62728", "marker": "^", "linestyle": "--"},
            "SFT": {"color": "#8A8A8A", "marker": "D", "linestyle": "None"},
        }

    def xy(point: RewardPoint) -> tuple[float, float]:
        if surf_style:
            return float(point.mean_safety), float(point.mean_reward)
        return float(point.mean_reward), float(point.mean_safety)

    fig, ax = plt.subplots(
        figsize=(7.2, 5.1) if publication_style else (7.4, 5.4),
        dpi=240 if publication_style else 180,
    )
    ordered_methods = []
    for point in points:
        if point.method not in ordered_methods:
            ordered_methods.append(point.method)
    if "SFT" in ordered_methods:
        ordered_methods = [method for method in ordered_methods if method != "SFT"] + ["SFT"]

    for method_idx, method in enumerate(ordered_methods):
        style = method_specs.get(method)
        if style is None:
            style = {
                "color": fallback_colors[method_idx % len(fallback_colors)],
                "marker": fallback_markers[method_idx % len(fallback_markers)],
            }
        method_points = [point for point in points if point.method == method]
        if not method_points:
            continue
        xs, ys = zip(*(xy(point) for point in method_points))
        ax.scatter(
            xs,
            ys,
            s=(26 if method != "SFT" else 58) if publication_style else (48 if method != "SFT" else 56),
            marker=style["marker"],
            color=style["color"],
            alpha=0.20 if publication_style and method != "SFT" else (0.85 if publication_style else 0.55),
            edgecolor="white" if publication_style else "black",
            linewidth=0.6 if publication_style else 0.35,
            label=("SFT" if method == "SFT" else "_nolegend_") if publication_style else (f"{method} checkpoints" if method != "SFT" else "SFT"),
            zorder=1 if publication_style and method != "SFT" else 2,
        )

        if method == "SFT":
            continue
        frontier = nondominated_reward_cost(method_points)
        frontier = sorted(frontier, key=lambda point: xy(point)[0])
        if len(frontier) >= 2:
            frontier_xs, frontier_ys = zip(*(xy(point) for point in frontier))
            ax.plot(
                frontier_xs,
                frontier_ys,
                color=style["color"],
                linewidth=2.35 if publication_style else 2.4,
                linestyle=style.get("linestyle", "-"),
                marker=style["marker"],
                markersize=6.2 if publication_style else 4.0,
                markeredgecolor="white" if publication_style else None,
                markeredgewidth=0.65 if publication_style else 0.0,
                label=method if publication_style else f"{method} reward-safety frontier",
                zorder=4 if publication_style else 4,
            )
        if annotate and not publication_style:
            for point in frontier:
                if point.lambda_helpful is None:
                    continue
                x_val, y_val = xy(point)
                ax.annotate(
                    f"{point.lambda_helpful:.2g}",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color=style["color"],
                )

    if surf_style:
        ax.set_title(
            "External reward-model Pareto front" if publication_style else "Reward-safety Pareto front on BeaverTails prompts",
            fontsize=12 if publication_style else None,
        )
        ax.set_xlabel("Mean safety / harmlessness = -cost (higher is better)")
        ax.set_ylabel("Mean reward / helpfulness (higher is better)")
    else:
        ax.set_title("Reward-safety Pareto front on BeaverTails prompts")
        ax.set_xlabel("Mean reward / helpfulness (higher is better)")
        ax.set_ylabel("Mean safety / harmlessness = -cost (higher is better)")
    ax.grid(True, alpha=0.28 if publication_style else 0.25, linewidth=0.7 if publication_style else 0.8)
    if publication_style:
        ax.tick_params(axis="both", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=9)
    else:
        ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def collect_specs(args: argparse.Namespace) -> List[CheckpointSpec]:
    specs: List[CheckpointSpec] = []
    if args.include_sft:
        specs.append(
            CheckpointSpec(
                method="SFT",
                run="sft",
                adapter_dir=None,
                lambda_helpful=None,
                lambda_harmless=None,
                source="base_sft_model",
            )
        )
    if args.adaptive_run:
        for run_dir, label in args.adaptive_run:
            specs.extend(
                collect_adaptive_checkpoints(
                    Path(run_dir),
                    include_final=args.include_adaptive_final,
                    method_label=label,
                )
            )
    else:
        specs.extend(
            collect_adaptive_checkpoints(
                Path(args.adaptive_dir) if args.adaptive_dir else None,
                include_final=args.include_adaptive_final,
            )
        )
    specs.extend(collect_dpo_lw_checkpoints(Path(args.dpo_lw_dir) if args.dpo_lw_dir else None))

    if args.max_checkpoints and args.max_checkpoints > 0:
        specs = specs[: args.max_checkpoints]
    return specs


def write_run_config(args: argparse.Namespace, specs: Sequence[CheckpointSpec], output_dir: Path) -> None:
    with (output_dir / "reward_pareto_config.json").open("w") as handle:
        json.dump(
            {
                "args": vars(args),
                "num_checkpoints": len(specs),
                "checkpoints": [
                    {
                        "method": spec.method,
                        "run": spec.run,
                        "adapter_dir": str(spec.adapter_dir) if spec.adapter_dir is not None else None,
                        "lambda_helpful": spec.lambda_helpful,
                        "lambda_harmless": spec.lambda_harmless,
                        "outer": spec.outer,
                        "parameter_updates": spec.parameter_updates,
                        "objective_gradient_evals": spec.objective_gradient_evals,
                        "elapsed_wall_seconds": spec.elapsed_wall_seconds,
                        "source": spec.source,
                    }
                    for spec in specs
                ],
            },
            handle,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and score adaptive-bundle / DPO-LW checkpoints with "
            "BeaverTails reward and cost models, then plot a reward-cost "
            "Pareto front."
        )
    )
    parser.add_argument("--adaptive_dir", default=None)
    parser.add_argument(
        "--adaptive_run",
        nargs=2,
        action="append",
        metavar=("DIR", "LABEL"),
        default=None,
        help=(
            "Adaptive-style output directory and plot label. Can be repeated "
            "for adaptive exact_k2, IPOPT, SURF, etc. When set, --adaptive_dir "
            "is ignored."
        ),
    )
    parser.add_argument("--dpo_lw_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sft_model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_name", default="PKU-Alignment/PKU-SafeRLHF-10K-safer")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument(
        "--eval_prompt_source",
        choices=["split", "oracle", "oracle_heldout"],
        default="split",
        help=(
            "'split' uses --dataset_name/--eval_split. 'oracle' reconstructs the "
            "fixed oracle prompts from the training run config. 'oracle_heldout' "
            "uses the same reconstructed training pool but starts after the oracle "
            "window under the same shuffle seeds."
        ),
    )
    parser.add_argument(
        "--oracle_reference_dir",
        default=None,
        help=(
            "Training run directory containing adaptive_config.json or dpo_lw_config.json "
            "for oracle/oracle_heldout prompt reconstruction. Defaults to --adaptive_dir "
            "then --dpo_lw_dir."
        ),
    )
    parser.add_argument("--prompt_template", default=QWEN_PROMPT_TEMPLATE)
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--prompt_max_length", type=int, default=384)
    parser.add_argument("--generation_max_length", type=int, default=512)
    parser.add_argument("--score_max_length", type=int, default=1024)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--score_batch_size", type=int, default=1)
    parser.add_argument("--reward_model_name", default="PKU-Alignment/beaver-7b-v1.0-reward")
    parser.add_argument("--cost_model_name", default="PKU-Alignment/beaver-7b-v1.0-cost")
    parser.add_argument("--score_load_mode", choices=["sequential", "both"], default="sequential")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_sft", type=parse_bool, default=True)
    parser.add_argument("--include_adaptive_final", type=parse_bool, default=False)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_scoring", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument(
        "--plot_style",
        choices=["default", "surf", "publication"],
        default="default",
        help=(
            "'publication' uses the reward-versus-safety orientation with muted "
            "checkpoint clouds and method-consistent Pareto frontiers."
        ),
    )
    parser.add_argument("--max_checkpoints", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = collect_specs(args)
    if not specs:
        raise SystemExit("No checkpoints found. Check --adaptive_dir and --dpo_lw_dir.")
    write_run_config(args, specs, output_dir)

    if args.skip_generation:
        generated = [
            (spec, output_dir / "generations" / safe_name(f"{spec.method}_{spec.run}"))
            for spec in specs
        ]
        missing = [gen_dir for _, gen_dir in generated if not generation_file(gen_dir).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing generation files while --skip_generation is set: "
                + ", ".join(str(path) for path in missing[:5])
            )
    else:
        eval_dataset = load_eval_prompts(args)
        generated = generate_checkpoint_responses(specs, eval_dataset, args, output_dir)

    if args.skip_scoring:
        score_root = output_dir / "scores"
        points = []
        for spec, gen_dir in generated:
            score_dir = score_root / gen_dir.name
            mean_reward, mean_cost = read_mean_score(score_dir / "mean.csv")
            points.append(
                RewardPoint(
                    method=spec.method,
                    run=spec.run,
                    mean_reward=mean_reward,
                    mean_cost=mean_cost,
                    lambda_helpful=spec.lambda_helpful,
                    lambda_harmless=spec.lambda_harmless,
                    outer=spec.outer,
                    parameter_updates=spec.parameter_updates,
                    objective_gradient_evals=spec.objective_gradient_evals,
                    elapsed_wall_seconds=spec.elapsed_wall_seconds,
                    adapter_dir=str(spec.adapter_dir) if spec.adapter_dir is not None else None,
                    generation_dir=str(gen_dir),
                    score_dir=str(score_dir),
                    source=spec.source,
                )
            )
    else:
        points = score_generations(generated, args, output_dir)

    csv_path = write_points_csv(points, output_dir)
    plot_path = plot_reward_pareto(
        points,
        output_dir,
        annotate=args.annotate,
        plot_style=args.plot_style,
    )
    print(f"saved reward/cost points to {csv_path}")
    if plot_path is not None:
        print(f"saved reward/cost Pareto front to {plot_path}")


if __name__ == "__main__":
    main()
