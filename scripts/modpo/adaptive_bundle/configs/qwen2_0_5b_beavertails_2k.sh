#!/usr/bin/env bash

# First LLM-scale PoC configuration for adaptive-bundle MOO.
# Usage:
#   source scripts/modpo/adaptive_bundle/configs/qwen2_0_5b_beavertails_2k.sh
#   bash scripts/modpo/adaptive_bundle/run_beavertails.sh

export SFT_MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
export DATASET_NAME="PKU-Alignment/PKU-SafeRLHF-10K"
export PROMPT_TEMPLATE=$'<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n'

export MAX_LENGTH=384
export TRAIN_SUBSET_SIZE_PER_OBJECTIVE=2000
export ORACLE_SUBSET_SIZE_PER_OBJECTIVE=128
export ADAPTIVE_ORACLE_BATCH_SIZE=4

export ADAPTIVE_MAX_OUTER=20
export ADAPTIVE_MAX_INNER=25
export ADAPTIVE_LAMBDA_MAX_STARTS=64
export ADAPTIVE_LAMBDA_SOLVER="ipopt"
export ADAPTIVE_REQUIRE_IPOPT=True
export ADAPTIVE_LAMBDA_NORMALIZATION="none"
export ADAPTIVE_SMOOTHNESS="1.0,1.0"
export ADAPTIVE_L_SCALE=1.0
export ADAPTIVE_DESCENT_ATOL=1e-6
export ADAPTIVE_DESCENT_RTOL=1e-6
export ADAPTIVE_SAVE_EVERY_OUTER=1

export LORA_R=8
export LORA_ALPHA=16
export ADAPTIVE_LEARNING_RATE=1e-4

# DPO-LW / uniform baseline grid for the same two objectives.
export UNIFORM_WEIGHT_RESOLUTION=5

export ADAPTIVE_OUTPUT_DIR="./output/PKU-Alignment/PKU-SafeRLHF-10K/adaptive_bundle/qwen2_0_5b_2k"

export DPO_LW_MAX_STEPS=300
export DPO_LW_PER_OBJECTIVE_BATCH_SIZE=2
export DPO_LW_GRADIENT_ACCUMULATION_STEPS=1
export DPO_LW_LEARNING_RATE=1e-4
export DPO_LW_OUTPUT_DIR="./output/PKU-Alignment/PKU-SafeRLHF-10K/dpo_lw/qwen2_0_5b_2k_r5"
