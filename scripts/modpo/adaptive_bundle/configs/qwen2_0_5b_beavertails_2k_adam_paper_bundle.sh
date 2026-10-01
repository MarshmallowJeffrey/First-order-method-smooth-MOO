#!/usr/bin/env bash

config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${config_dir}/qwen2_0_5b_beavertails_2k.sh"

# Paper-style adaptive bundle:
#   lambda_t = argmax_lambda min_i ||sum_k lambda_k grad F_k(theta_i)||^2
#   B_{t+1} = BundleUpdate_Mt(lambda_t; B_t)
#   stop outer when GN*_t < 2 epsilon / 3
#   stop inner when GN(lambda_t; B_{t+1}) < epsilon / 3
#
# Difference from the paper's T-map implementation:
#   BundleUpdate uses Adam/AdamW weighted-DPO steps from the active worst-case
#   bundle point, then keeps only the best inner candidate in B.
export ADAPTIVE_ALGORITHM_MODE="theory"
export ADAPTIVE_UPDATE_RULE="adam"
export ADAPTIVE_BUNDLE_UPDATE_MODE="append"
export ADAPTIVE_UPDATE_DATA_SOURCE="oracle"
export ADAPTIVE_PRUNE_INNER=True

# Conservative DPO temperature for the 70/30 shared-prompt run.
export ADAPTIVE_BETA="${ADAPTIVE_BETA:-0.05}"

# Controlled shared-prompt objective data: 70% agreement and 30% conflict.
# Set ADAPTIVE_AGREEMENT_RATIO="" before sourcing this config to recover
# the original independently sampled objective datasets.
export ADAPTIVE_SHARED_OBJECTIVE_SUBSET=True
if [[ -z "${ADAPTIVE_AGREEMENT_RATIO+x}" ]]; then
    export ADAPTIVE_AGREEMENT_RATIO=0.7
fi

# No extra lambda heuristics: use the raw GN maximization.
# To compare against the Gram/envelope two-objective solver, set:
#   export ADAPTIVE_LAMBDA_SOLVER=exact_k2
# after sourcing this config, or override it in the job environment.
export ADAPTIVE_LAMBDA_NORMALIZATION="none"
export ADAPTIVE_LAMBDA_ENTROPY_TAU=0.0
export ADAPTIVE_LAMBDA_DIVERSITY_STRENGTH=0.0
export ADAPTIVE_LAMBDA_PROJECTION_DIM=0
export ADAPTIVE_LAMBDA_MIN=0.0

# No grid warm-start points in the initial bundle; start from B0 only.
export ADAPTIVE_BUNDLE_WARM_START_STEPS=0
export ADAPTIVE_RESUME="${ADAPTIVE_RESUME:-False}"

# Enable the epsilon stopping logic from Algorithm 2.
# Tune this on the cloud after one smoke run if it stops too early/late.
export ADAPTIVE_STOP_RULE="absolute"
export ADAPTIVE_EPSILON="${ADAPTIVE_EPSILON:-1e-3}"

# Adam-specific training defaults.
export ADAPTIVE_LEARNING_RATE="${ADAPTIVE_LEARNING_RATE:-1e-4}"
export ADAPTIVE_LR_SCHEDULER_TYPE="constant_with_warmup"
export ADAPTIVE_WARMUP_RATIO="${ADAPTIVE_WARMUP_RATIO:-0.03}"
export ADAPTIVE_WEIGHT_DECAY="${ADAPTIVE_WEIGHT_DECAY:-0.0}"
export ADAPTIVE_MAX_GRAD_NORM="${ADAPTIVE_MAX_GRAD_NORM:-1.0}"
