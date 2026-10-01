#!/usr/bin/env bash
set -euo pipefail

# SURF baseline for the same two-objective Qwen/BeaverTails DPO-LW setup.
# SURF only changes the lambda schedule; data, DPO loss, AdamW updates, oracle
# diagnostics, checkpoints, reward eval, and plot compatibility stay in this repo.

default_prompt_template=$'<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n'

sft_model_name="${SFT_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
prompt_template="${PROMPT_TEMPLATE:-${default_prompt_template}}"
dataset_name="${DATASET_NAME:-PKU-Alignment/PKU-SafeRLHF-10K}"
sanity_check="${SANITY_CHECK:-False}"
seed="${SEED:-42}"
output_root="${OUTPUT_DIR:-./output}"
output_dir="${SURF_OUTPUT_DIR:-${output_root}/${dataset_name}/surf_dpo/qwen2_0_5b}"
max_length="${MAX_LENGTH:-384}"
beta="${SURF_BETA:-${DPO_LW_BETA:-${ADAPTIVE_BETA:-0.05}}}"
train_subset_size="${TRAIN_SUBSET_SIZE_PER_OBJECTIVE:-2000}"
agreement_ratio="${SURF_AGREEMENT_RATIO:-${DPO_LW_AGREEMENT_RATIO:-${ADAPTIVE_AGREEMENT_RATIO:-}}}"
per_objective_batch_size="${SURF_PER_OBJECTIVE_BATCH_SIZE:-${DPO_LW_PER_OBJECTIVE_BATCH_SIZE:-${ADAPTIVE_PER_OBJECTIVE_BATCH_SIZE:-2}}}"
update_data_source="${SURF_UPDATE_DATA_SOURCE:-${DPO_LW_UPDATE_DATA_SOURCE:-${ADAPTIVE_UPDATE_DATA_SOURCE:-oracle}}}"
gradient_accumulation_steps="${SURF_GRADIENT_ACCUMULATION_STEPS:-${DPO_LW_GRADIENT_ACCUMULATION_STEPS:-${ADAPTIVE_GRADIENT_ACCUMULATION_STEPS:-1}}}"
warmup_ratio="${SURF_WARMUP_RATIO:-${DPO_LW_WARMUP_RATIO:-${ADAPTIVE_WARMUP_RATIO:-0.03}}}"
lr_scheduler_type="${SURF_LR_SCHEDULER_TYPE:-${DPO_LW_LR_SCHEDULER_TYPE:-${ADAPTIVE_LR_SCHEDULER_TYPE:-cosine}}}"
weight_decay="${SURF_WEIGHT_DECAY:-${DPO_LW_WEIGHT_DECAY:-${ADAPTIVE_WEIGHT_DECAY:-0.0}}}"
max_grad_norm="${SURF_MAX_GRAD_NORM:-${DPO_LW_MAX_GRAD_NORM:-${ADAPTIVE_MAX_GRAD_NORM:-1.0}}}"
learning_rate="${SURF_LEARNING_RATE:-${DPO_LW_LEARNING_RATE:-${ADAPTIVE_LEARNING_RATE:-1e-4}}}"
oracle_subset_size="${ORACLE_SUBSET_SIZE_PER_OBJECTIVE:-128}"
oracle_batch_size="${SURF_ORACLE_BATCH_SIZE:-${DPO_LW_ORACLE_BATCH_SIZE:-${ADAPTIVE_ORACLE_BATCH_SIZE:-4}}}"
smoothness="${SURF_SMOOTHNESS:-${ADAPTIVE_SMOOTHNESS:-1.0,1.0}}"
lambda_max_starts="${SURF_LAMBDA_MAX_STARTS:-${ADAPTIVE_LAMBDA_MAX_STARTS:-64}}"
lambda_solver="${SURF_LAMBDA_SOLVER:-exact_k2}"
require_ipopt="${SURF_REQUIRE_IPOPT:-False}"
bundle_dtype="${SURF_BUNDLE_DTYPE:-${ADAPTIVE_BUNDLE_DTYPE:-float32}}"
gn_target_norm="${SURF_GN_TARGET_NORM:-${GN_TARGET_NORM:-}}"
lora_r="${LORA_R:-8}"
lora_alpha="${LORA_ALPHA:-16}"

# For a 300-update budget with one update per slot per outer, use:
#   SURF_NUM_SEGMENTS=9, SURF_MAX_OUTER=30, SURF_STEPS_PER_SLOT_PER_OUTER=1
# This creates 10 slots x 30 outers = 300 scalarized parameter updates.
surf_num_segments="${SURF_NUM_SEGMENTS:-9}"
surf_max_outer="${SURF_MAX_OUTER:-30}"
surf_steps_per_slot_per_outer="${SURF_STEPS_PER_SLOT_PER_OUTER:-1}"
surf_alpha="${SURF_ALPHA:-1.0}"
surf_cdf_grid_size="${SURF_CDF_GRID_SIZE:-2001}"
surf_use_pchip="${SURF_USE_PCHIP:-True}"
surf_monotone_eps="${SURF_MONOTONE_EPS:-1e-8}"
surf_force_endpoints="${SURF_FORCE_ENDPOINTS:-True}"
surf_save_every_outer="${SURF_SAVE_EVERY_OUTER:-1}"
surf_warm_start_strategy="${SURF_WARM_START_STRATEGY:-same_slot}"

agreement_args=()
if [[ -n "${agreement_ratio}" ]]; then
    agreement_args+=(--agreement_ratio "${agreement_ratio}")
fi
target_args=()
if [[ -n "${gn_target_norm}" ]]; then
    target_args+=(--gn_target_norm "${gn_target_norm}")
fi

PYTHONPATH=. python scripts/modpo/adaptive_bundle/surf_dpo.py \
    --sft_model_name "${sft_model_name}" \
    --prompt_template "${prompt_template}" \
    --helpful_dataset_name "${dataset_name}-better" \
    --harmless_dataset_name "${dataset_name}-safer" \
    --sanity_check "${sanity_check}" \
    --seed "${seed}" \
    --beta "${beta}" \
    --max_length "${max_length}" \
    --train_subset_size_per_objective "${train_subset_size}" \
    --per_objective_batch_size "${per_objective_batch_size}" \
    --update_data_source "${update_data_source}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --warmup_ratio "${warmup_ratio}" \
    --lr_scheduler_type "${lr_scheduler_type}" \
    --weight_decay "${weight_decay}" \
    --max_grad_norm "${max_grad_norm}" \
    --oracle_subset_size_per_objective "${oracle_subset_size}" \
    --oracle_batch_size "${oracle_batch_size}" \
    --smoothness "${smoothness}" \
    --lambda_max_starts "${lambda_max_starts}" \
    --lambda_solver "${lambda_solver}" \
    --require_ipopt "${require_ipopt}" \
    --bundle_dtype "${bundle_dtype}" \
    "${target_args[@]}" \
    --surf_num_segments "${surf_num_segments}" \
    --surf_max_outer "${surf_max_outer}" \
    --surf_steps_per_slot_per_outer "${surf_steps_per_slot_per_outer}" \
    --surf_alpha "${surf_alpha}" \
    --surf_cdf_grid_size "${surf_cdf_grid_size}" \
    --surf_use_pchip "${surf_use_pchip}" \
    --surf_monotone_eps "${surf_monotone_eps}" \
    --surf_force_endpoints "${surf_force_endpoints}" \
    --surf_save_every_outer "${surf_save_every_outer}" \
    --surf_warm_start_strategy "${surf_warm_start_strategy}" \
    "${agreement_args[@]}" \
    --training_args.output_dir "${output_dir}" \
    --training_args.seed "${seed}" \
    --training_args.learning_rate "${learning_rate}" \
    --peft_config.r "${lora_r}" \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha "${lora_alpha}" \
    --peft_config.lora_dropout 0
