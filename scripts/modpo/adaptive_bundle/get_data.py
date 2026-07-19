from datasets import load_dataset

ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-10K", split="train")
ds.to_json("PKU-SafeRLHF-10K-train.jsonl", force_ascii=False)
