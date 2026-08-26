# Experiment outputs

Layout updated Aug 25, 2026 after the repository-wide reorganisation
(grouping + renaming done by the user; mapping below). Every experiment
home keeps its own `README.md` — that file plus the dated `Note/`
record is the authoritative documentation of that experiment.

IMPORTANT: the runner scripts under `Original_py/` still carry the
PRE-reorganisation path constants and flat-layout imports, so nothing
is currently re-runnable in place — see the Aug-25 notice at the top of
`../CODE_MAP.md` before attempting any re-run or `--replot`.

## Layout (current)

```
with_256_checkpoints/                       ORIGINAL track (fixed 256-start yardstick)
  v1_IPOPT_vs_original_baseline_plateau/    July plateau sweep K=3..6 + K=6 budget study (30k/90k/240k)
  v1_IPOPT_vs_original_baseline_crossover/  July CPU-crossover sweep, widths 16x16..128x128
without_256_checkpoints_comparison/         A/B measurement-variant rerun (K5 plateau + 96x96
                                            crossover, *_self_report subfolders) + FINDINGS.md
lambda_path_K3_v1_IPOPT/                    K=3 lambda-path figure (July 9)
trial_K6_d11910_h96x96_tanh_n50000_B180180_without_256_checkpoints/
                                            July 11 K=6 reference trial — frozen reference curves
                                            reused by the fast trials (do not move)
fast_method_trials_v1v2v3_IPOPT/            fast-engine series: v1/v2/v3, cert_attempt_partial,
                                            v4 strict probe (trial_..._v4_strict_rel0.1)
pure_budget_without_256_checkpoints_SVRG_IPOPT_Baseline/
  baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/
                                            K=6 comparison home of record: original/ (verbatim copy
                                            of the deleted July-25 r-sweep home), tol0.02/, tol0.01/,
                                            adaptive_extended/ (v4 copy), diag_v3_plateau/,
                                            fixed_budget_B80912(+_SMOKE)/ (protocol 5e),
                                            pure_budget_B80912(+_SMOKE)/ (protocol 5f, headline)
  pure_budget_K2_without_256_checkpoints/   K=2 pure fixed budget with the EXACT 1-D meter
Bandit_toy/                                 SURF offline-bandit toys
  bandit_toy_surf_without_256_checkpoints/  K=2 (session 11; eps1e-2 revised by session 13)
  bandit_toy_K5_without_256_checkpoints/    K=5 centered-quartic variant
  bandit_toy_mv_without_256_checkpoints/    mean-variance nonconvex variant
CCP/                                        Aug 8-13 CCP campaigns
  ccp_compare_without_256_checkpoints/      Experiment 1 (K2_B20000, K6_B80912), MNIST K=10 trial
                                            (K10_mnist10k_B55000; report "Experiment 2", earlier
                                            numbering "3"), lambda_solver_bench (Study B) + SMOKEs
  ccp_smoke_sampler/                        Study A: Exp(1) vs Sobol seed-sampler ablation
  K2_mnist_pair_without_256_checkpoints/    Experiment 4: digit-pair conflict smoke + pair_3v5 /
                                            pair_7v9 campaigns + SMOKE
README.md                                   this file
```

Per-leg file conventions: `summary.json` (full curves + parameters +
health flags) and `grams.npz` in every leg; `thetas.npz` only in the
MNIST-pair legs; `raw_histories.npz` in bandit runs; `audit_v2.json` in
the audited K6/K10 legs; `campaign_manifest.json` at campaign level.
NOTE (Aug 25): the eight `pair_*_B20000/*/thetas.npz` files are 190–206
MB each — over GitHub's 100 MB per-file limit — so they are git-ignored
and exist ON THIS DISK ONLY (the SMOKE thetas stay tracked). Back them
up separately if the parameter vectors must survive this machine.

## Renamed / moved on Aug 25 (old top-level name → new location)

| old | new |
|---|---|
| `plateau/` | `with_256_checkpoints/v1_IPOPT_vs_original_baseline_plateau/` |
| `crossover/` | `with_256_checkpoints/v1_IPOPT_vs_original_baseline_crossover/` |
| `without_256_checkpoints/` | `without_256_checkpoints_comparison/` (inner `plateau`/`crossover` → `v1_IPOPT_vs_original_baseline_{plateau,crossover}_self_report/`) |
| `lambda_path_K3/` | `lambda_path_K3_v1_IPOPT/` |
| `fast_method_trials/` | `fast_method_trials_v1v2v3_IPOPT/` |
| `baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/` | `pure_budget_without_256_checkpoints_SVRG_IPOPT_Baseline/baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/` |
| `pure_budget_K2_without_256_checkpoints/` | `pure_budget_without_256_checkpoints_SVRG_IPOPT_Baseline/pure_budget_K2_without_256_checkpoints/` |
| `bandit_toy_{surf,K5,mv}_without_256_checkpoints/` | `Bandit_toy/…` |
| `ccp_compare_without_256_checkpoints/`, `ccp_smoke_sampler/`, `K2_mnist_pair_without_256_checkpoints/` | `CCP/…` |
| `trial_K6_…_B180180_…/` | unchanged |

## Deleted on Aug 25 (user decision — "deleted = not needed"; tracked items recoverable from git history only)

- `Pareto_front/` — the July-8 certified-Pareto K=2 experiment (both
  runs, FINDINGS_ZH, figures). No copy elsewhere in the tree.
- `baseline_svrg_multi_r_vs_fast_without_256_checkpoints/` — the old
  July-25 r-sweep home. Its verbatim copy SURVIVES at
  `pure_budget_…_Baseline/baseline_svrg_multi_r_vs_fast_v2_…/original/`.
- `calibration_speed_test_B2000/` — July 11 speed calibration (was
  never an experiment result).
- `plateau`'s `K6_budget_study.png` (cross-budget figure).
- In `CCP/ccp_compare_…/`: the three user-edit `.docx` backups and the
  K2/K6/K10 campaign logs (`Experiment1_report_K2_K6.pdf`,
  `audit_v2.log`, `bench_campaign.log` kept).
- In `CCP/K2_mnist_pair_…/`: `conflict_smoke.log`.
- `pure_budget_K2_…/B20000_SMOKE/` (empty shell).
- Repo level (same clean-up wave): `EXPERIMENTS.md`, `EXPERIMENTS_ZH.md`
  (the only full report of the July sweeps — git history has them),
  `Python_Change.md/` (4 change-record files), `tmp/` (paper-reading
  extracts), all `__pycache__/`.

## How to read the per-configuration sweep plots (July experiments)

Both plots show worst-case-over-weights squared gradient norm GN* (log
scale, lower is better) for the uniform-discretisation baseline and the
adaptive bundle method under the SAME total gradient budget on the SAME
problem instance.

- `gn_vs_grad_evals.png` — the oracle-complexity view: the paper
  predicts the adaptive method reaches lower GN* per gradient spent.
- `gn_vs_cpu_time.png` — the wall-clock view: the adaptive method pays
  an oracle-free per-round overhead (lambda search, T-map algebra), so
  with a cheap oracle the baseline wins this axis; the crossover sweep
  makes the oracle expensive to show where that reverses.

Checkpoint-metric cost is excluded from both axes for both methods.
The full July analysis lived in `EXPERIMENTS(_ZH).md` (deleted Aug 25;
recover from git history if needed).
