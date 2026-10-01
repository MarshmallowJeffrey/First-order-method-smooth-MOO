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
better_dataset_name="${BETTER_DATASET_NAME:-${dataset_name}-better}"
safer_dataset_name="${SAFER_DATASET_NAME:-${dataset_name}-safer}"
sanity_check="${SANITY_CHECK:-False}"
seed="${SEED:-42}"
output_root="${OUTPUT_DIR:-./output}"
output_dir="${ADAPTIVE_OUTPUT_DIR:-${output_root}/${dataset_name}/adaptive_bundle/qwen2_0_5b_2k}"
max_length="${MAX_LENGTH:-384}"
beta="${ADAPTIVE_BETA:-0.1}"
train_subset_size="${TRAIN_SUBSET_SIZE_PER_OBJECTIVE:-2000}"
oracle_subset_size="${ORACLE_SUBSET_SIZE_PER_OBJECTIVE:-128}"
oracle_batch_size="${ADAPTIVE_ORACLE_BATCH_SIZE:-4}"
consistent_preferences_only="${ADAPTIVE_CONSISTENT_PREFERENCES_ONLY:-False}"
shared_objective_subset="${ADAPTIVE_SHARED_OBJECTIVE_SUBSET:-False}"
agreement_ratio="${ADAPTIVE_AGREEMENT_RATIO:-}"
max_outer="${ADAPTIVE_MAX_OUTER:-20}"
max_inner="${ADAPTIVE_MAX_INNER:-25}"
algorithm_mode="${ADAPTIVE_ALGORITHM_MODE:-llm}"
if [[ "${algorithm_mode}" == "theory" ]]; then
    update_rule="${ADAPTIVE_UPDATE_RULE:-t_map}"
    bundle_update_mode="${ADAPTIVE_BUNDLE_UPDATE_MODE:-append}"
else
    update_rule="${ADAPTIVE_UPDATE_RULE:-adamw}"
    bundle_update_mode="${ADAPTIVE_BUNDLE_UPDATE_MODE:-lambda_aware}"
fi
stop_rule="${ADAPTIVE_STOP_RULE:-none}"
epsilon="${ADAPTIVE_EPSILON:-}"
relative_rho="${ADAPTIVE_RELATIVE_RHO:-0.5}"
gn_target_norm="${ADAPTIVE_GN_TARGET_NORM:-${GN_TARGET_NORM:-}}"
update_data_source="${ADAPTIVE_UPDATE_DATA_SOURCE:-oracle}"
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
lambda_min="${ADAPTIVE_LAMBDA_MIN:-0.0}"
lambda_entropy_tau="${ADAPTIVE_LAMBDA_ENTROPY_TAU:-0.0}"
lambda_diversity_strength="${ADAPTIVE_LAMBDA_DIVERSITY_STRENGTH:-0.0}"
lambda_diversity_grid_points="${ADAPTIVE_LAMBDA_DIVERSITY_GRID_POINTS:-101}"
lambda_diversity_recent_window="${ADAPTIVE_LAMBDA_DIVERSITY_RECENT_WINDOW:-3}"
lambda_projection_dim="${ADAPTIVE_LAMBDA_PROJECTION_DIM:-0}"
lambda_projection_seed="${ADAPTIVE_LAMBDA_PROJECTION_SEED:-0}"
lambda_match_tol="${ADAPTIVE_LAMBDA_MATCH_TOL:-1e-4}"
lambda_stall_patience="${ADAPTIVE_LAMBDA_STALL_PATIENCE:-0}"
lambda_stall_abs_delta="${ADAPTIVE_LAMBDA_STALL_ABS_DELTA:-0.0}"
lambda_stall_rel_delta="${ADAPTIVE_LAMBDA_STALL_REL_DELTA:-0.0}"
lambda_stall_cooldown="${ADAPTIVE_LAMBDA_STALL_COOLDOWN:-1}"
lambda_stall_grid_points="${ADAPTIVE_LAMBDA_STALL_GRID_POINTS:-101}"
lambda_stall_match_tol="${ADAPTIVE_LAMBDA_STALL_MATCH_TOL:-0.05}"
bundle_warm_start_steps="${ADAPTIVE_BUNDLE_WARM_START_STEPS:-0}"
bundle_warm_start_lambdas="${ADAPTIVE_BUNDLE_WARM_START_LAMBDAS:-0.0;0.25;0.5;0.75;1.0}"
resume="${ADAPTIVE_RESUME:-False}"
resume_state_path="${ADAPTIVE_RESUME_STATE_PATH:-}"
prefix_save_bundle_sizes="${ADAPTIVE_PREFIX_SAVE_BUNDLE_SIZES:-}"
prefix_state_dir="${ADAPTIVE_PREFIX_STATE_DIR:-}"
smoothness="${ADAPTIVE_SMOOTHNESS:-1.0,1.0}"
l_scale="${ADAPTIVE_L_SCALE:-1.0}"
descent_atol="${ADAPTIVE_DESCENT_ATOL:-1e-6}"
descent_rtol="${ADAPTIVE_DESCENT_RTOL:-1e-6}"
max_bundle_size="${ADAPTIVE_MAX_BUNDLE_SIZE:-0}"
bundle_cap_mode="${ADAPTIVE_BUNDLE_CAP_MODE:-replace_active_if_better}"
save_every_outer="${ADAPTIVE_SAVE_EVERY_OUTER:-0}"
learning_rate="${ADAPTIVE_LEARNING_RATE:-1e-4}"
lora_r="${LORA_R:-8}"
lora_alpha="${LORA_ALPHA:-16}"

epsilon_args=()
if [[ -n "${epsilon}" ]]; then
    epsilon_args+=(--epsilon "${epsilon}")
fi
target_args=()
if [[ -n "${gn_target_norm}" ]]; then
    target_args+=(--gn_target_norm "${gn_target_norm}")
fi
resume_state_args=()
if [[ -n "${resume_state_path}" ]]; then
    resume_state_args+=(--resume_state_path "${resume_state_path}")
fi
prefix_state_args=()
if [[ -n "${prefix_save_bundle_sizes}" ]]; then
    prefix_state_args+=(--prefix_save_bundle_sizes "${prefix_save_bundle_sizes}")
fi
if [[ -n "${prefix_state_dir}" ]]; then
    prefix_state_args+=(--prefix_state_dir "${prefix_state_dir}")
fi
agreement_args=()
if [[ -n "${agreement_ratio}" ]]; then
    agreement_args+=(--agreement_ratio "${agreement_ratio}")
fi

cmd=(
    python scripts/modpo/adaptive_bundle/beavertails.py
    --sft_model_name "${sft_model_name}"
    --prompt_template "${prompt_template}"
    --better_dataset_name "${better_dataset_name}"
    --safer_dataset_name "${safer_dataset_name}"
    --sanity_check "${sanity_check}"
    --seed "${seed}"
    --beta "${beta}"
    --max_length "${max_length}"
    --train_subset_size_per_objective "${train_subset_size}"
    --oracle_subset_size_per_objective "${oracle_subset_size}"
    --oracle_batch_size "${oracle_batch_size}"
    --consistent_preferences_only "${consistent_preferences_only}"
    --shared_objective_subset "${shared_objective_subset}"
    "${agreement_args[@]}"
    --max_outer "${max_outer}"
    --max_inner "${max_inner}"
    --algorithm_mode "${algorithm_mode}"
    --stop_rule "${stop_rule}"
    --relative_rho "${relative_rho}"
    "${target_args[@]}"
    --update_rule "${update_rule}"
    --bundle_update_mode "${bundle_update_mode}"
    --update_data_source "${update_data_source}"
    --per_objective_batch_size "${per_objective_batch_size}"
    --gradient_accumulation_steps "${gradient_accumulation_steps}"
    --warmup_ratio "${warmup_ratio}"
    --lr_scheduler_type "${lr_scheduler_type}"
    --weight_decay "${weight_decay}"
    --max_grad_norm "${max_grad_norm}"
    --prune_inner "${prune_inner}"
    --lambda_max_starts "${lambda_max_starts}"
    --lambda_solver "${lambda_solver}"
    --require_ipopt "${require_ipopt}"
    --lambda_normalization "${lambda_normalization}"
    --lambda_min "${lambda_min}"
    --lambda_entropy_tau "${lambda_entropy_tau}"
    --lambda_diversity_strength "${lambda_diversity_strength}"
    --lambda_diversity_grid_points "${lambda_diversity_grid_points}"
    --lambda_diversity_recent_window "${lambda_diversity_recent_window}"
    --lambda_projection_dim "${lambda_projection_dim}"
    --lambda_projection_seed "${lambda_projection_seed}"
    --lambda_match_tol "${lambda_match_tol}"
    --lambda_stall_patience "${lambda_stall_patience}"
    --lambda_stall_abs_delta "${lambda_stall_abs_delta}"
    --lambda_stall_rel_delta "${lambda_stall_rel_delta}"
    --lambda_stall_cooldown "${lambda_stall_cooldown}"
    --lambda_stall_grid_points "${lambda_stall_grid_points}"
    --lambda_stall_match_tol "${lambda_stall_match_tol}"
    --bundle_warm_start_steps "${bundle_warm_start_steps}"
    --bundle_warm_start_lambdas "${bundle_warm_start_lambdas}"
    --resume "${resume}"
    "${resume_state_args[@]}"
    "${prefix_state_args[@]}"
    --smoothness "${smoothness}"
    --l_scale "${l_scale}"
    --descent_atol "${descent_atol}"
    --descent_rtol "${descent_rtol}"
    --max_bundle_size "${max_bundle_size}"
    --bundle_cap_mode "${bundle_cap_mode}"
    --save_every_outer "${save_every_outer}"
    --training_args.output_dir "${output_dir}"
    --training_args.seed "${seed}"
    --training_args.learning_rate "${learning_rate}"
    --peft_config.r "${lora_r}"
    --peft_config.target_modules q_proj k_proj v_proj o_proj
    --peft_config.lora_alpha "${lora_alpha}"
    --peft_config.lora_dropout 0
    "${epsilon_args[@]}"
)

PYTHONPATH=. "${cmd[@]}"
