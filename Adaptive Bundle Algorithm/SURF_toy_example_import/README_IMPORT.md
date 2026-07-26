# Imported: SURF paper companion code (MOO_Uniform_PF)

Imported on 2026-07-24 for the bandit toy-example comparison task.

- Upstream: https://github.com/Liuyuan999/MOO_Uniform_PF
- Commit at import time: `867de3af2dd53570ee3a65a5c8f4446d78bf7d6e` (also in
  `UPSTREAM_COMMIT.txt`). The nested `.git/` was removed so this folder is a
  plain snapshot, not a nested repository.
- Upstream re-check on 2026-07-25: current head was
  `afc0c46d219cb262bda425ba982b98860abbbacf`. Comparing it against the imported
  commit showed three newer commits, but none changed `uniform_PF.ipynb`; the
  bandit toy snapshot is therefore still current for this task. Later changes
  were confined to Fishwood, Mountaincar, and `benchmark_moo`.
- Companion paper: `../Reference_essay/SURF.pdf` — "SURF: Steering the
  Scalarization Weight to Uniformly Traverse the Pareto Front"
  (Jiang, Huang, Chen; arXiv:2605.20619).

## Why this is here

Current task: compare our λ-bundle adaptive algorithm (paper Algorithm 2,
`../Original_py/algorithm_without_256_checkpoints.py`) against the uniform
discretization baseline (paper Algorithm 1,
`../Original_py/baseline_without_256_checkpoints.py`) on the **offline bandit
toy example** from the SURF paper (Appendix F.1, "Offline bandit"), measuring
epsilon-accuracy of the generated Pareto frontiers and CPU time.

## What matters for the task

| Path | Role for our task |
|---|---|
| `uniform_PF.ipynb` | THE reference notebook. Cell 1 defines the offline bandit (A=5 arms, R1(a)=x_a, R2(a)=1-x_a^4, KL-regularized objectives with tau=0.05, balanced offline dataset with Gaussian reward noise sigma=0.5, T=1000), the closed-form scalarized solution `softmax_policy`, the minimization objectives `f_components_policy`, the explicit PF-traversal speed (paper Eq. 9), and both weight rules: `run_baseline_offline` (uniform w) and `one_shot_arc_length_weights` (arc-length-uniform, paper Rule 1). Cell 2-3: error-vs-T replication study (paper Figure 5). |
| `README.md` | Upstream's own repo map. |
| `DST/`, `Fishwood/`, `Mountaincar/` | SURF's MO-Gymnasium experiments (paper F.2). Not used by the current task; kept because the overall project plan also compares on MO-Gymnasium later. |
| `LLM_alignment/` | SURF's LLM fine-tuning experiment (paper F.3). Not used now. |
| `Tchebycheff_nonconvex/`, `benchmark_moo/`, `additional_benchmark_experiments/` | Other SURF experiments. Not used now. |

## Conventions to keep straight when bridging to our code

- SURF's KL coefficient is written `beta` in the paper and `tau` in the
  notebook (value 0.05). Our code calls the scalarization weight `lambda`;
  SURF uses scalar `w` with `lambda = [w, 1-w]`.
- The notebook works in the minimization convention
  `f_m(pi) = tau*KL(pi||pi_ref) - <pi, R_m>`, matching our `min` convention.
- Appendix F.1 states reward-noise sigma = 0.5; the notebook's final run cell
  uses `noise_std=0.5`, `T_dataset=1000`, `N_target=12`, `N_dense=5000`
  (earlier helper defaults of 0.1/50000 are overridden there).
