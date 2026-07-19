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
2. applies one or more T-map steps at the selected lambda,
3. evaluates both DPO objectives at the new LoRA vector,
4. appends the new first-order information to the bundle,
5. optionally saves the current adapter checkpoint.

Run a small sanity pass first:

```bash
SANITY_CHECK=True ADAPTIVE_MAX_OUTER=2 TRAIN_SUBSET_SIZE_PER_OBJECTIVE=256 \
  ORACLE_SUBSET_SIZE_PER_OBJECTIVE=32 ADAPTIVE_ORACLE_BATCH_SIZE=2 \
  bash scripts/modpo/adaptive_bundle/run_beavertails.sh
```

IPOPT is required by the default configuration for the GN lambda maximization.
On AutoDL/conda, install it before running the scripts:

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
  an update. Each inner T-map candidate evaluation counts as one parameter
  update, even if `prune_inner=True` later removes that candidate from the
  retained bundle.
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
adaptive max_outer: 20
adaptive max_inner: 25
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

- The smoothness constants `L_k` are user-provided via `ADAPTIVE_SMOOTHNESS`
  and default to `1.0,1.0`.
- The T-map inner loop uses the same descent-lemma safeguard as the original
  bundle code: if the new point violates
  `F_lambda(x_new) <= F_lambda(x_i) - ||grad F_lambda(x_i)||^2/(2 L_lambda)`,
  the runner doubles the global `L_scale`, keeps the paid-for candidate in the
  bundle, and uses the smaller step size afterward. The LLM runner allows a
  numerical tolerance controlled by `ADAPTIVE_DESCENT_ATOL` and
  `ADAPTIVE_DESCENT_RTOL`, both defaulting to `1e-6`. Per-step diagnostics are
  logged in `adaptive_history.jsonl`.
- GN lambda maximization defaults to IPOPT through `cyipopt`, matching the
  original adaptive-bundle implementation. `ADAPTIVE_REQUIRE_IPOPT=True` makes
  missing IPOPT fail fast instead of silently using SLSQP.
- `ADAPTIVE_LAMBDA_NORMALIZATION=global_mean` enables an LLM-oriented
  scale-calibrated lambda selection: the GN maximization divides each
  objective gradient by its mean bundle norm before selecting lambda. The
  T-map update still uses the original raw scalarized gradient. The default
  `none` keeps the original MOA criterion unchanged.
- Each adaptive outer record includes `lambda_diagnostics`: per-objective
  gradient norm summaries, helpful/harmless gradient cosines, and GN values on
  a 21-point helpful-weight grid for both raw and lambda-selection metrics.
  These diagnostics are logging only and do not change T-map updates.
- This first migration uses a fixed oracle subset, so GN* is deterministic for
  the selected subset but still only approximates the full dataset objective.
- DPO-LW writes `uniform_gn_history.jsonl` by evaluating each trained uniform
  weight checkpoint on the same fixed oracle subset. The GN* comparison then
  aligns uniform and adaptive runs by cumulative `objective_gradient_evals`, not
  by outer iterations, uniform grid passes, or retained bundle size.
- The included plots use logged DPO objective losses. Reward-model generation
  and external scoring can be added as a later evaluation layer.
