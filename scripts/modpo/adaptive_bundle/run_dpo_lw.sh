#!/usr/bin/env bash
set -euo pipefail

# bash scripts/modpo/adaptive_bundle/run_dpo_lw.sh
#
# DPO loss-weighting baseline for the same two-objective Qwen/BeaverTails
# setup used by adaptive bundle.

default_prompt_template=$'<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n'

sft_model_name="${SFT_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
prompt_template="${PROMPT_TEMPLATE:-${default_prompt_template}}"
dataset_name="${DATASET_NAME:-PKU-Alignment/PKU-SafeRLHF-10K}"
sanity_check="${SANITY_CHECK:-False}"
output_root="${OUTPUT_DIR:-./output}"
output_dir="${DPO_LW_OUTPUT_DIR:-${output_root}/${dataset_name}/dpo_lw/qwen2_0_5b_2k_r5}"
max_length="${MAX_LENGTH:-384}"
train_subset_size="${TRAIN_SUBSET_SIZE_PER_OBJECTIVE:-2000}"
per_objective_batch_size="${DPO_LW_PER_OBJECTIVE_BATCH_SIZE:-2}"
max_steps="${DPO_LW_MAX_STEPS:-300}"
gradient_accumulation_steps="${DPO_LW_GRADIENT_ACCUMULATION_STEPS:-1}"
weight_resolution="${UNIFORM_WEIGHT_RESOLUTION:-5}"
learning_rate="${DPO_LW_LEARNING_RATE:-1e-4}"
lora_r="${LORA_R:-8}"
lora_alpha="${LORA_ALPHA:-16}"
evaluate_uniform_gn="${DPO_LW_EVALUATE_UNIFORM_GN:-True}"
oracle_subset_size="${ORACLE_SUBSET_SIZE_PER_OBJECTIVE:-128}"
oracle_batch_size="${DPO_LW_ORACLE_BATCH_SIZE:-${ADAPTIVE_ORACLE_BATCH_SIZE:-4}}"
smoothness="${ADAPTIVE_SMOOTHNESS:-1.0,1.0}"
lambda_max_starts="${ADAPTIVE_LAMBDA_MAX_STARTS:-64}"
lambda_solver="${DPO_LW_LAMBDA_SOLVER:-${ADAPTIVE_LAMBDA_SOLVER:-ipopt}}"
require_ipopt="${DPO_LW_REQUIRE_IPOPT:-${ADAPTIVE_REQUIRE_IPOPT:-True}}"

PYTHONPATH=. python scripts/modpo/adaptive_bundle/dpo_lw.py \
    --sft_model_name "${sft_model_name}" \
    --prompt_template "${prompt_template}" \
    --helpful_dataset_name "${dataset_name}-better" \
    --harmless_dataset_name "${dataset_name}-safer" \
    --sanity_check "${sanity_check}" \
    --max_length "${max_length}" \
    --train_subset_size_per_objective "${train_subset_size}" \
    --per_objective_batch_size "${per_objective_batch_size}" \
    --weight_resolution "${weight_resolution}" \
    --max_steps "${max_steps}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --evaluate_uniform_gn "${evaluate_uniform_gn}" \
    --oracle_subset_size_per_objective "${oracle_subset_size}" \
    --oracle_batch_size "${oracle_batch_size}" \
    --smoothness "${smoothness}" \
    --lambda_max_starts "${lambda_max_starts}" \
    --lambda_solver "${lambda_solver}" \
    --require_ipopt "${require_ipopt}" \
    --training_args.output_dir "${output_dir}" \
    --training_args.learning_rate "${learning_rate}" \
    --peft_config.r "${lora_r}" \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha "${lora_alpha}" \
    --peft_config.lora_dropout 0
