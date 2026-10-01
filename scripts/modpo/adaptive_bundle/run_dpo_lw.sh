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
beta="${DPO_LW_BETA:-${ADAPTIVE_BETA:-0.1}}"
train_subset_size="${TRAIN_SUBSET_SIZE_PER_OBJECTIVE:-2000}"
agreement_ratio="${DPO_LW_AGREEMENT_RATIO:-${ADAPTIVE_AGREEMENT_RATIO:-}}"
per_objective_batch_size="${DPO_LW_PER_OBJECTIVE_BATCH_SIZE:-2}"
max_steps="${DPO_LW_MAX_STEPS:-300}"
total_update_budget="${DPO_LW_TOTAL_UPDATE_BUDGET:-}"
uniform_update_mode="${DPO_LW_UNIFORM_UPDATE_MODE:-moa_cycle}"
update_data_source="${DPO_LW_UPDATE_DATA_SOURCE:-${ADAPTIVE_UPDATE_DATA_SOURCE:-oracle}}"
chain_warm_start="${DPO_LW_CHAIN_WARM_START:-True}"
gradient_accumulation_steps="${DPO_LW_GRADIENT_ACCUMULATION_STEPS:-1}"
warmup_ratio="${DPO_LW_WARMUP_RATIO:-${ADAPTIVE_WARMUP_RATIO:-0.03}}"
lr_scheduler_type="${DPO_LW_LR_SCHEDULER_TYPE:-${ADAPTIVE_LR_SCHEDULER_TYPE:-cosine}}"
weight_decay="${DPO_LW_WEIGHT_DECAY:-${ADAPTIVE_WEIGHT_DECAY:-0.0}}"
max_grad_norm="${DPO_LW_MAX_GRAD_NORM:-${ADAPTIVE_MAX_GRAD_NORM:-1.0}}"
weight_resolution="${UNIFORM_WEIGHT_RESOLUTION:-5}"
learning_rate="${DPO_LW_LEARNING_RATE:-1e-4}"
lora_r="${LORA_R:-8}"
lora_alpha="${LORA_ALPHA:-16}"
evaluate_uniform_gn="${DPO_LW_EVALUATE_UNIFORM_GN:-True}"
gn_target_norm="${DPO_LW_GN_TARGET_NORM:-${GN_TARGET_NORM:-}}"
oracle_subset_size="${ORACLE_SUBSET_SIZE_PER_OBJECTIVE:-128}"
oracle_batch_size="${DPO_LW_ORACLE_BATCH_SIZE:-${ADAPTIVE_ORACLE_BATCH_SIZE:-4}}"
smoothness="${ADAPTIVE_SMOOTHNESS:-1.0,1.0}"
lambda_max_starts="${ADAPTIVE_LAMBDA_MAX_STARTS:-64}"
lambda_solver="${DPO_LW_LAMBDA_SOLVER:-${ADAPTIVE_LAMBDA_SOLVER:-ipopt}}"
require_ipopt="${DPO_LW_REQUIRE_IPOPT:-${ADAPTIVE_REQUIRE_IPOPT:-True}}"

extra_args=()
if [[ -n "${total_update_budget}" ]]; then
    extra_args+=(--total_update_budget "${total_update_budget}")
fi
if [[ -n "${agreement_ratio}" ]]; then
    extra_args+=(--agreement_ratio "${agreement_ratio}")
fi
if [[ -n "${gn_target_norm}" ]]; then
    extra_args+=(--gn_target_norm "${gn_target_norm}")
fi

PYTHONPATH=. python scripts/modpo/adaptive_bundle/dpo_lw.py \
    --sft_model_name "${sft_model_name}" \
    --prompt_template "${prompt_template}" \
    --helpful_dataset_name "${dataset_name}-better" \
    --harmless_dataset_name "${dataset_name}-safer" \
    --sanity_check "${sanity_check}" \
    --beta "${beta}" \
    --max_length "${max_length}" \
    --train_subset_size_per_objective "${train_subset_size}" \
    --per_objective_batch_size "${per_objective_batch_size}" \
    --weight_resolution "${weight_resolution}" \
    --max_steps "${max_steps}" \
    --uniform_update_mode "${uniform_update_mode}" \
    --update_data_source "${update_data_source}" \
    --chain_warm_start "${chain_warm_start}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --warmup_ratio "${warmup_ratio}" \
    --lr_scheduler_type "${lr_scheduler_type}" \
    --weight_decay "${weight_decay}" \
    --max_grad_norm "${max_grad_norm}" \
    --evaluate_uniform_gn "${evaluate_uniform_gn}" \
    --oracle_subset_size_per_objective "${oracle_subset_size}" \
    --oracle_batch_size "${oracle_batch_size}" \
    --smoothness "${smoothness}" \
    --lambda_max_starts "${lambda_max_starts}" \
    --lambda_solver "${lambda_solver}" \
    --require_ipopt "${require_ipopt}" \
    "${extra_args[@]}" \
    --training_args.output_dir "${output_dir}" \
    --training_args.learning_rate "${learning_rate}" \
    --peft_config.r "${lora_r}" \
    --peft_config.target_modules q_proj k_proj v_proj o_proj \
    --peft_config.lora_alpha "${lora_alpha}" \
    --peft_config.lora_dropout 0
