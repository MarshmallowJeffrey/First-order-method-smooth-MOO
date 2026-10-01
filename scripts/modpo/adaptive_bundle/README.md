# Adaptive Bundle for MODPO Framework

This folder is the first LLM migration of the adaptive bundle method from the
MLP multi-objective alignment project.

For the initial two-objective version, the objectives are direct DPO losses:

- objective 1: helpful DPO loss on `better` preferences
- objective 2: harmless DPO loss on `safer` preferences

The lambda convention is:

```text
lambda = [lambda_helpful, lambda_harmless]
F_lambda = lambda_helpful * F_helpful + lambda_harmless * F_harmless
```

The runner maintains a first-order bundle of LoRA parameter vectors. Each bundle
entry stores the current trainable vector, both objective losses, and both
objective gradients. Each outer step:

1. maximizes GN over the two-objective simplex,
2. applies one or more AdamW weighted-DPO steps at the selected lambda,
3. evaluates both DPO objectives at the new LoRA vector,
4. appends the new first-order information to the bundle,
5. optionally saves the current adapter checkpoint.

`ADAPTIVE_UPDATE_RULE=adamw` is the default LLM runner. It keeps the adaptive
bundle lambda selection, but updates LoRA parameters with the same optimizer
family as the DPO-LW baseline:

```text
loss = lambda_helpful * F_helpful + lambda_harmless * F_harmless
```

The original smoothness-based T-map path is still available for ablations with
`ADAPTIVE_UPDATE_RULE=t_map`.

For the algorithm in the adaptive-bundle pseudocode, use the theory defaults:

```bash
export ADAPTIVE_ALGORITHM_MODE=theory
export ADAPTIVE_STOP_RULE=absolute
export ADAPTIVE_EPSILON=1e-4
```

`ADAPTIVE_ALGORITHM_MODE=theory` makes the shell runner default to
`ADAPTIVE_UPDATE_RULE=t_map` and `ADAPTIVE_BUNDLE_UPDATE_MODE=append`.  The
Python runner still accepts explicit overrides, but it will warn when theory
mode is paired with a non-theory update configuration.

Stopping rules:

- `ADAPTIVE_STOP_RULE=none`: legacy behavior, run `ADAPTIVE_MAX_OUTER` outer
  iterations and `ADAPTIVE_MAX_INNER` inner updates.
- `ADAPTIVE_STOP_RULE=absolute`: screenshot-style thresholds.  Stop before an
  outer update when `GN* < 2 * ADAPTIVE_EPSILON / 3`; for a selected
  `lambda_t`, stop the inner BundleUpdate loop at the first `M_t` where
  `GN(lambda_t; B) < ADAPTIVE_EPSILON / 3`.
- `ADAPTIVE_STOP_RULE=relative`: LLM-scale diagnostic rule.  Stop outer updates
  when `GN*` drops below `ADAPTIVE_RELATIVE_RHO` times the first outer `GN*`;
  stop an inner loop when `GN(lambda_t; B)` drops below
  `ADAPTIVE_RELATIVE_RHO` times its value before that outer update.

The runner writes `adaptive_solution_path.json`, which records the initial
centroid anchor, each selected `lambda_t`, the realized `M_t`, GN before/after
the inner loop, and the bundle indices used to build the approximate solution
path.

The lambda search uses the original full simplex by default:

```text
ADAPTIVE_LAMBDA_MIN=0.0
```

For an endpoint-avoidance ablation, set `ADAPTIVE_LAMBDA_MIN=0.05`. For two
objectives this restricts selection to `lambda_helpful in [0.05, 0.95]`,
preventing pure endpoint updates while still allowing strongly imbalanced
weights.

For LLM DPO runs where the GN maximizer repeatedly picks the same lambda, an
optional anti-collapse heuristic can be enabled:

```bash
export ADAPTIVE_LAMBDA_NORMALIZATION=global_mean
export ADAPTIVE_LAMBDA_MIN=0.05
export ADAPTIVE_LAMBDA_DIVERSITY_STRENGTH=0.5
export ADAPTIVE_LAMBDA_DIVERSITY_GRID_POINTS=101
export ADAPTIVE_LAMBDA_DIVERSITY_RECENT_WINDOW=3
```

This keeps the original GN maximization as the base choice, then scores a
two-objective lambda grid by normalized GN value plus a small distance bonus
from recent lambdas. The AdamW parameter update still uses the raw scalarized
DPO-LW loss. Setting `ADAPTIVE_LAMBDA_DIVERSITY_STRENGTH=0.0` recovers the
original behavior.

An optional entropy-regularized selector can also be enabled:

```bash
export ADAPTIVE_LAMBDA_NORMALIZATION=global_mean
export ADAPTIVE_LAMBDA_ENTROPY_TAU=0.3
```

This changes only the lambda-selection subproblem to
`GN(lambda) + tau * H(lambda)`, where
`H(lambda) = -sum_k lambda_k log(lambda_k)`. The DPO-LW AdamW update still uses
the raw scalarized DPO loss. `ADAPTIVE_LAMBDA_ENTROPY_TAU=0.0` recovers the
original GN selector.

To test whether high-dimensional LoRA gradients are making the GN selector
prefer simplex vertices, enable a low-dimensional CountSketch projection for
lambda selection only:

```bash
export ADAPTIVE_LAMBDA_PROJECTION_DIM=256
export ADAPTIVE_LAMBDA_PROJECTION_SEED=0
```

This changes only the GN lambda-selection geometry:

```text
GN(lambda) = min_i || sketch(sum_k lambda_k grad F_k(theta_i)) ||^2
```

The DPO-LW AdamW update still backpropagates the full raw scalarized loss over
all trainable LoRA parameters. Set `ADAPTIVE_LAMBDA_PROJECTION_DIM=0` to disable
the projection. Useful diagnostic values are `64`, `128`, `256`, and `512`.

To diagnose whether helpful/harmless disagreement in PKU-SafeRLHF is causing
near-orthogonal gradients, use only examples where the helpful and harmless
labels agree:

```bash
export ADAPTIVE_CONSISTENT_PREFERENCES_ONLY=True
export ADAPTIVE_SHARED_OBJECTIVE_SUBSET=True
```

This filters raw PKU samples to `better_response_id == safer_response_id` and
uses the same deterministic subset seed for both objectives. It is a diagnostic
data setting, not the default benchmark setting.

Run a small sanity pass first:

```bash
SANITY_CHECK=True ADAPTIVE_MAX_OUTER=2 TRAIN_SUBSET_SIZE_PER_OBJECTIVE=256 \
  ORACLE_SUBSET_SIZE_PER_OBJECTIVE=32 ADAPTIVE_ORACLE_BATCH_SIZE=2 \
  bash scripts/modpo/adaptive_bundle/run_beavertails.sh
```

The default configuration uses IPOPT for GN lambda maximization. For the
two-objective helpful/harmless runs, you can instead use the Gram/envelope
solver from the GNS note:

```bash
export ADAPTIVE_LAMBDA_SOLVER=exact_k2
```

`exact_k2` exactly maximizes the lower envelope on the two-objective simplex
from cached per-bundle-point Gram matrices for the selected GN geometry,
including `ADAPTIVE_LAMBDA_NORMALIZATION`, `ADAPTIVE_LAMBDA_MIN`, and optional
projection. It does not support `ADAPTIVE_LAMBDA_ENTROPY_TAU > 0`; use `ipopt`
or `slsqp` for entropy regularized selector ablations.

If you keep the default IPOPT solver, install it before running the scripts:

```bash
conda install -y -c conda-forge ipopt cyipopt=1.7.0
pip install -r requirements.txt
python - <<'PY'
from scripts.modpo.adaptive_bundle.bundle_core import ipopt_available
assert ipopt_available(), "cyipopt/IPOPT is not available"
print("IPOPT OK")
PY
```

Run the matching DPO-LW baseline:

```bash
source scripts/modpo/adaptive_bundle/configs/qwen2_0_5b_beavertails_2k.sh
bash scripts/modpo/adaptive_bundle/run_dpo_lw.sh
```

Plot the DPO-loss Pareto front after training:

```bash
python scripts/modpo/adaptive_bundle/plot_results.py \
  --dpo_lw_dir "$DPO_LW_OUTPUT_DIR" \
  --adaptive_dir "$ADAPTIVE_OUTPUT_DIR" \
  --output_dir "./output/PKU-Alignment/PKU-SafeRLHF-10K/figures/qwen2_0_5b_2k" \
  --annotate
```

The plotting script writes:

- `pareto_front.png`
- `pareto_representatives.csv`
- `adaptive_lambda_trajectories.png`
- `uniform_lambda_trajectories.png`
- `lambda_path.png`
- `gn_star_comparison.png`
- `dpo_lw_training_curves.png`
- `adaptive_training_trace.png`
- `results_summary.csv`

The adaptive runner also writes `adaptive_final_state.json` with
`l_scale_final` and `safeguard_violations`.

Budget accounting used in the plots:

- `parameter_updates` is the scalarized-update count, matching `total_iters`.
- `objective_gradient_evals` is the main fair-budget axis and equals
  `parameter_updates * K`.
- For uniform DPO-LW, one parameter update is one optimizer step for one fixed
  lambda run. A full pass over all uniform lambdas contains many updates.
- For adaptive bundle, choosing lambda from the cached bundle is not counted as
  an update. With the default AdamW update, each inner optimizer step counts as
  one parameter update. With the optional T-map update, each inner T-map
  candidate evaluation counts as one parameter update, even if `prune_inner=True`
  later removes that candidate from the retained bundle.
- In this two-objective setup, one parameter update corresponds to about
  `2` objective-gradient evaluations. The logs also keep `gradient_eval` /
  `oracle_gradient_eval` for backward compatibility and provenance, but
  `gn_star_comparison.png` uses `objective_gradient_evals` on the x-axis.

Default first-run configuration:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
prompt template: Qwen chat template
dtype: bfloat16
training: LoRA
LoRA r: 8
LoRA alpha: 16
LoRA target modules: q_proj,k_proj,v_proj,o_proj
max_length: 384
training pool: 2000 samples per objective
fixed oracle subset: 128 samples per objective
oracle batch size: 4
consistent preferences only: false
shared objective subset: false
adaptive max_outer: 20
adaptive max_inner: 25
adaptive algorithm_mode: llm
adaptive stop_rule: none
adaptive epsilon: unset
adaptive relative_rho: 0.5
adaptive update_rule: adamw
adaptive per_objective_batch_size: 2
adaptive prune_inner: false
adaptive lambda_min: 0.0
adaptive lambda_entropy_tau: 0.0
adaptive lambda_diversity_strength: 0.0
adaptive lambda_projection_dim: 0
lambda max starts: 64
lambda solver: ipopt
require ipopt: true
uniform baseline resolution: 5
DPO-LW max_steps per lambda: 300
DPO-LW uniform GN eval: enabled
```

Data handling:

- BeaverTails/PKU data is not stored in this repository.
- The scripts load `PKU-Alignment/PKU-SafeRLHF-10K` through Hugging Face
  `datasets` and the existing repo adapter in `src/data/raw_data/safe_rlhf.py`.
- To avoid Hugging Face network access on AutoDL, export
  `LOCAL_SAFE_RLHF_JSONL=/path/to/PKU-SafeRLHF-10K-train.jsonl`. The adapter
  will read that local JSONL and still create train/validation with the same
  `train_test_split(test_size=0.1, seed=0)` logic.
- The original adapter creates train/validation by
  `train_test_split(test_size=0.1, seed=0)`.
- This folder then selects deterministic objective subsets at runtime:
  helpful/better uses `seed`, harmless/safer uses `seed + 1`, and the adaptive
  fixed oracle subsets use `seed + 2` and `seed + 3`.

Important current assumptions:

- The default adaptive update is AdamW. `ADAPTIVE_SMOOTHNESS`,
  `ADAPTIVE_L_SCALE`, `ADAPTIVE_DESCENT_ATOL`, and `ADAPTIVE_DESCENT_RTOL`
  only affect the optional `ADAPTIVE_UPDATE_RULE=t_map` ablation.
- The optional T-map inner loop uses the same descent-lemma safeguard as the
  original bundle code: if the new point violates
  `F_lambda(x_new) <= F_lambda(x_i) - ||grad F_lambda(x_i)||^2/(2 L_lambda)`,
  the runner doubles the global `L_scale`, keeps the paid-for candidate in the
  bundle, and uses the smaller step size afterward. The LLM runner allows a
  numerical tolerance controlled by `ADAPTIVE_DESCENT_ATOL` and
  `ADAPTIVE_DESCENT_RTOL`, both defaulting to `1e-6`. Per-step diagnostics are
  logged in `adaptive_history.jsonl`.
- GN lambda maximization defaults to IPOPT through `cyipopt`, matching the
  original adaptive-bundle implementation. `ADAPTIVE_LAMBDA_SOLVER=exact_k2`
  switches the two-objective runs to the exact Gram/envelope solver, while
  `ipopt` and `slsqp` keep the previous local-NLP path. `ADAPTIVE_REQUIRE_IPOPT=True`
  makes missing IPOPT fail fast instead of silently using SLSQP when
  `ADAPTIVE_LAMBDA_SOLVER=ipopt`.
- `ADAPTIVE_LAMBDA_MIN` optionally constrains lambda selection away from simplex
  vertices. The default `0.0` keeps the original unconstrained MOA lambda
  search. Set it to `0.05` for a practical LLM ablation against endpoint
  collapse with nearly orthogonal helpful/harmless gradients.
- `ADAPTIVE_LAMBDA_NORMALIZATION=global_mean` enables an LLM-oriented
  scale-calibrated lambda selection: the GN maximization divides each
  objective gradient by its mean bundle norm before selecting lambda. AdamW
  updates still use the original raw scalarized DPO loss. The default `none`
  keeps the original MOA criterion unchanged.
- `ADAPTIVE_LAMBDA_DIVERSITY_STRENGTH` enables an explicit two-objective
  anti-collapse ablation for LLM DPO runs. The default `0.0` keeps the original
  GN-selected lambda. Values around `0.3` to `0.7` make the selector prefer a
  nearby high-GN lambda that is farther from the most recent selections.
- `ADAPTIVE_LAMBDA_PROJECTION_DIM` enables a CountSketch random projection for
  lambda selection only. This is intended to diagnose and mitigate the
  high-dimensional near-orthogonality that can make raw GN prefer endpoint
  lambdas. The default `0` disables projection and keeps the original
  full-gradient GN criterion.
- Each adaptive outer record includes `lambda_diagnostics`: per-objective
  gradient norm summaries, helpful/harmless gradient cosines, and GN values on
  a 21-point helpful-weight grid for both raw and lambda-selection metrics.
  These diagnostics are logging only and do not change parameter updates.
- This first migration uses a fixed oracle subset, so GN* is deterministic for
  the selected subset but still only approximates the full dataset objective.
- DPO-LW writes `uniform_gn_history.jsonl` by evaluating each trained uniform
  weight checkpoint on the same fixed oracle subset. The GN* comparison then
  aligns uniform and adaptive runs by cumulative `objective_gradient_evals`, not
  by outer iterations, uniform grid passes, or retained bundle size.
- The included DPO-loss plots use logged objective losses. For final
  reward/cost Pareto-front evaluation, use
  `scripts/modpo/adaptive_bundle/reward_pareto_eval.py`; it generates responses
  from adaptive and uniform checkpoints on the fixed validation prompts, scores
  them with `PKU-Alignment/beaver-7b-v1.0-reward` and
  `PKU-Alignment/beaver-7b-v1.0-cost`, and writes
  `reward_pareto_points.csv` plus `reward_cost_pareto_front.png`.

Example reward/cost Pareto-front evaluation:

```bash
export ADAPTIVE_OUTPUT_DIR="./output/PKU-Alignment/PKU-SafeRLHF-10K/adaptive_bundle/stall_switch_outer20_inner10"
export DPO_LW_OUTPUT_DIR="./output/PKU-Alignment/PKU-SafeRLHF-10K/dpo_lw/uniform_moa_cycle_oracle_matched_200_updates_r10"
export REWARD_FIG_DIR="./output/dev/figures/reward_pf_stall_switch_outer20_inner10_vs_uniform_r10"

python scripts/modpo/adaptive_bundle/reward_pareto_eval.py \
  --adaptive_dir "$ADAPTIVE_OUTPUT_DIR" \
  --dpo_lw_dir "$DPO_LW_OUTPUT_DIR" \
  --output_dir "$REWARD_FIG_DIR" \
  --sft_model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_name PKU-Alignment/PKU-SafeRLHF-10K-safer \
  --eval_size 200 \
  --generation_batch_size 4 \
  --score_batch_size 1 \
  --score_load_mode sequential \
  --annotate
```
