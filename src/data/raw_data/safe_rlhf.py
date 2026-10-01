from abc import ABC
from dataclasses import dataclass
import os
from typing import Dict, Literal, Optional

from datasets import concatenate_datasets, load_dataset

from .utils import RawDatasetPreprocessor

LOCAL_SAFE_RLHF_JSONL_ENV = "LOCAL_SAFE_RLHF_JSONL"
SAFE_RLHF_AGREEMENT_ONLY_ENV = "SAFE_RLHF_AGREEMENT_ONLY"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y"}


def _load_train_split(path: str):
    local_jsonl = os.environ.get(LOCAL_SAFE_RLHF_JSONL_ENV)
    if local_jsonl:
        if not os.path.exists(local_jsonl):
            raise FileNotFoundError(
                f"{LOCAL_SAFE_RLHF_JSONL_ENV} points to a missing file: {local_jsonl}"
            )
        return load_dataset("json", data_files=local_jsonl, split="train")
    return load_dataset(path, split="train")


def agreement_stats(dataset):
    agree = sum(
        int(sample["better_response_id"] == sample["safer_response_id"])
        for sample in dataset
    )
    total = len(dataset)
    disagree = total - agree
    return {
        "total": total,
        "agree": agree,
        "disagree": disagree,
        "agree_ratio": agree / total if total else 0.0,
        "disagree_ratio": disagree / total if total else 0.0,
    }


def sample_agreement_mixture(dataset, size, agreement_ratio: float, seed: int):
    if agreement_ratio < 0.0 or agreement_ratio > 1.0:
        raise ValueError(f"agreement_ratio must be in [0, 1], got {agreement_ratio}")
    if size is None or size <= 0 or size >= len(dataset):
        size = len(dataset)

    agree_pool = dataset.filter(
        lambda sample: sample["better_response_id"] == sample["safer_response_id"]
    )
    disagree_pool = dataset.filter(
        lambda sample: sample["better_response_id"] != sample["safer_response_id"]
    )

    target_agree = int(round(size * agreement_ratio))
    target_disagree = size - target_agree

    if target_agree > len(agree_pool):
        spill = target_agree - len(agree_pool)
        target_agree = len(agree_pool)
        target_disagree += spill
    if target_disagree > len(disagree_pool):
        spill = target_disagree - len(disagree_pool)
        target_disagree = len(disagree_pool)
        target_agree = min(len(agree_pool), target_agree + spill)

    if target_agree + target_disagree <= 0:
        raise ValueError("Agreement mixture has no samples to select.")

    parts = []
    if target_agree:
        parts.append(agree_pool.shuffle(seed=seed).select(range(target_agree)))
    if target_disagree:
        parts.append(disagree_pool.shuffle(seed=seed + 1).select(range(target_disagree)))
    mixed = concatenate_datasets(parts).shuffle(seed=seed + 2)
    return mixed


def _split_train_validation(path: str, split: str):
    dataset = _load_train_split(path)
    if _env_flag(SAFE_RLHF_AGREEMENT_ONLY_ENV):
        dataset = dataset.filter(
            lambda sample: sample["better_response_id"] == sample["safer_response_id"]
        )
    split_dataset = dataset.train_test_split(test_size=0.1, seed=0)
    if split == "train":
        return split_dataset["train"]
    if split == "validation":
        return split_dataset["test"]
    raise NotImplementedError

@dataclass
class PKUSafeRlhfRDPBase(RawDatasetPreprocessor, ABC):
    dimension: Literal["safer", "better"] = "better"

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        chosen_idx = example[f"{self.dimension}_response_id"]
        return {
            "raw_prompt": example["prompt"],
            "prompt":   self.prompt_template.format(raw_prompt=example["prompt"]),
            "chosen":   example[f"response_{chosen_idx}"],
            "rejected": example[f"response_{1-chosen_idx}"],
        }

@dataclass
class PKUSafeRlhfRDP(PKUSafeRlhfRDPBase):
    path: Optional[str] = "PKU-Alignment/PKU-SafeRLHF"

    def _get_raw_dataset(self, split):
        if split in {"train", "validation"}:
            return _split_train_validation(self.path, split)
        elif split == "test":
            if os.environ.get(LOCAL_SAFE_RLHF_JSONL_ENV):
                raise NotImplementedError(
                    f"{LOCAL_SAFE_RLHF_JSONL_ENV} provides only the train JSONL; "
                    "use split='train' or split='validation'."
                )
            return load_dataset(self.path, split="test")
        else:
            raise NotImplementedError


@dataclass
class PKUSafeRlhf10KRDP(PKUSafeRlhfRDPBase):
    path: Optional[str] = "PKU-Alignment/PKU-SafeRLHF-10K"

    def _get_raw_dataset(self, split):
        if split in {"train", "validation"}:
            return _split_train_validation(self.path, split)
        elif split == "test":
            raise NotImplementedError("PKU-Alignment/PKU-SafeRLHF-10K is for development, no test set available.")
        else:
            raise NotImplementedError


if __name__ == '__main__':
    safer10k_train_dataset = PKUSafeRlhf10KRDP(dimension="safer").get_preference_dataset(split="train")
    better10k_train_dataset = PKUSafeRlhf10KRDP(dimension="better").get_preference_dataset(split="train")
    breakpoint()
