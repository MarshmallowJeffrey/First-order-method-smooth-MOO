#!/usr/bin/env bash
set -euo pipefail

# bash scripts/modpo/adaptive_bundle/run_beavertails.sh
#
# Two-objective adaptive-bundle runner on BeaverTails:
#   objective 1 = DPO loss on better/helpfulness preferences
#   objective 2 = DPO loss on safer/safety preferences
#
# This is intentionally single-process for the first LLM migration. Start with
# SANITY_CHECK=True and small ADAPTIVE_MAX_OUTER before scaling.

default_prompt_template=$'<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n'

sft_model_name="${SFT_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
prompt_template="${PROMPT_TEMPLATE:-${default_prompt_template}}"
dataset_name="${DATASET_NAME:-PKU-Alignment/PKU-SafeRLHF-10K}"
sanity_check="${SANITY_CHECK:-False}"
output_root="${OUTPUT_DIR:-./output}"
output_dir="${ADAPTIVE_OUTPUT_DIR:-${output_root}/${dataset_name}/adaptive_bundle/qwen2_0_5b_2k}"
max_length="${MAX_LENGTH:-384}"
train_subset_size="${TRAIN_SUBSET_SIZE_PER_OBJECTIVE:-2000}"
oracle_subset_size="${ORACLE_SUBSET_SIZE_PER_OBJECTIVE:-128}"
oracle_batch_size="${ADAPTIVE_ORACLE_BATCH_SIZE:-4}"
max_outer="${ADAPTIVE_MAX_OUTER:-20}"
max_inner="${ADAPTIVE_MAX_INNER:-25}"
update_rule="${ADAPTIVE_UPDATE_RULE:-adamw}"
per_objective_batch_size="${ADAPTIVE_PER_OBJECTIVE_BATCH_SIZE:-2}"
gradient_accumulation_steps="${ADAPTIVE_GRADIENT_ACCUMULATION_STEPS:-1}"
warmup_ratio="${ADAPTIVE_WARMUP_RATIO:-0.03}"
lr_scheduler_type="${ADAPTIVE_LR_SCHEDULER_TYPE:-cosine}"
weight_decay="${ADAPTIVE_WEIGHT_DECAY:-0.0}"
max_grad_norm="${ADAPTIVE_MAX_GRAD_NORM:-1.0}"
prune_inner="${ADAPTIVE_PRUNE_INNER:-False}"
lambda_max_starts="${ADAPTIVE_LAMBDA_MAX_STARTS:-64}"
lambda_solver="${ADAPTIVE_LAMBDA_SOLVER:-ipopt}"
require_ipopt="${ADAPTIVE_REQUIRE_IPOPT:-True}"
lambda_normalization="${ADAPTIVE_LAMBDA_NORMALIZATION:-none}"
smoothness="${ADAPTIVE_SMOOTHNESS:-1.0,1.0}"
l_scale="${ADAPTIVE_L_SCALE:-1.0}"
descent_atol="${ADAPTIVE_DESCENT_ATOL:-1e-6}"
descent_rtol="${ADAPTIVE_DESCENT_RTOL:-1e-6}"
save_every_outer="${ADAPTIVE_SAVE_EVERY_OUTER:-0}"
learning_rate="${ADAPTIVE_LEARNING_RATE:-1e-4}"
lora_r="${LORA_R:-8}"
lora_alpha="${LORA_ALPHA:-16}"

PYTHONPATH=. python scripts/modpo/adaptive_bundle/beavertails.py \
    --sft_model_name "${sft_model_name}" \
    --prompt_template "${prompt_template}" \
    --better_dataset_name "${dataset_name}-better" \
    --safer_dataset_name "${dataset_name}-safer" \
    --sanity_check "${sanity_check}" \
    --max_length "${max_length}" \
    --train_subset_size_per_objective "${train_subset_size}" \
    --oracle_subset_size_per_objective "${oracle_subset_size}" \
    --oracle_batch_size "${oracle_batch_size}" \
    --max_outer "${max_outer}" \
    --max_inner "${max_inner}" \
    --update_rule "${update_rule}" \
    --per_objective_batch_size "${per_objective_batch_size}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --warmup_ratio "${warmup_ratio}" \
    --lr_scheduler_type "${lr_scheduler_type}" \
    --weight_decay "${weight_decay}" \
    --max_grad_norm "${max_grad_norm}" \
    --prune_inner "${prune_inner}" \
    --lambda_max_starts "${lambda_max_starts}" \
    --lambda_solver "${lambda_solver}" \
    --require_ipopt "${require_ipopt}" \
    --lambda_normalization "${lambda_normalization}" \
    --smoothness "${smoothness}" \
    --l_scale "${l_scale}" \
    --descent_atol "${descent_atol}" \
    --descent_rtol "${descent_rtol}" \
    --save_every_outer "${save_every_outer}" \
    --training_args.output_dir "${output_dir}" \
    --training_args.learning_rate "${learning_rate}" \
    --peft_config.r "${lora_r}" \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha "${lora_alpha}" \
    --peft_config.lora_dropout 0
