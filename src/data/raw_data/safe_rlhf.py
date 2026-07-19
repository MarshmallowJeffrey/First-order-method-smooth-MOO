from abc import ABC
from dataclasses import dataclass
import os
from typing import Dict, Literal, Optional

from datasets import load_dataset

from .utils import RawDatasetPreprocessor

LOCAL_SAFE_RLHF_JSONL_ENV = "LOCAL_SAFE_RLHF_JSONL"


def _load_train_split(path: str):
    local_jsonl = os.environ.get(LOCAL_SAFE_RLHF_JSONL_ENV)
    if local_jsonl:
        if not os.path.exists(local_jsonl):
            raise FileNotFoundError(
                f"{LOCAL_SAFE_RLHF_JSONL_ENV} points to a missing file: {local_jsonl}"
            )
        return load_dataset("json", data_files=local_jsonl, split="train")
    return load_dataset(path, split="train")


def _split_train_validation(path: str, split: str):
    dataset = _load_train_split(path)
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
