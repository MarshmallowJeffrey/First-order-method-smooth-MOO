# Results of the paper's runs

Written by `scripts/analyze.py`, `scripts/step_rules.py`, `scripts/warm_start.py` and `scripts/screening.py` from
the runs of the paper (NVIDIA RTX A5000, float64); read by `scripts/make_figures.py` and `scripts/make_tables.py`.

`k2.json`, `k3.json` ({4,9} and {4,7,9}, B = 480,000 gradient calls)
* `runs.<leg>`: `method` (adaptive, uniform, surf), `param` (r or N), `seed`, `budget`; the checkpoints `ck_grads`
  (gradient calls) and `ck_wall` (wall-clock seconds); `audit_gn`, the audited worst-case gradient norm
  max_lambda GN(lambda, B_t)^(1/2) at each checkpoint (K = 2 exact, K = 3 a lower bound); `levels` and `B_run`, the
  plateau test (`B_run` null: no plateau); `marker` {x, y, wall} (K = 2: located by bisection, `m` = bundle size;
  the checkpoint-based marker is in `marker_checkpoint`); timings, segments and rejections.
* `configs`: per configuration the geometric means over the seeds of the marker (`y_geomean`, `x_geomean`,
  `wall_geomean`), `x_median` and `n_plateau`, the number of seeds that plateau.
* `adaptive_final`, `adaptive_final_geomean`: the adaptive method at the end of the budget.

`k2_fronts.json`: per leg the non-dominated training objective values (F_4, F_9) of all visited points.

The K = 2 adaptive legs in `k2.json` and `k2_fronts.json` use the exact envelope lambda-search (`selector`, and
`adaptive_selector` at the top; since 2026-10-04, `scripts/update_k2_adaptive.py`).  `k2_ccp.json` and
`k2_fronts_ccp.json` are the earlier versions of both files, with the CCP lambda-search; their baseline entries are
the same.  `k2_envelope.json` (`scripts/compare_envelope.py`) compares the two: final values per seed, timings, the
values at 10k ... 480k gradient calls, and the ratios to the best baselines.
`k3_fronts.json`: per leg the non-dominated values (F_4, F_7, F_9) with every objective at most 0.5.

`step_rules_k2.json` ({4,9}): per step rule the mean curve over the three seeds (`ck_grads`, `ck_wall_mean`,
`gn_mean`), the final values per seed and their mean; `ranking` from the lowest mean.

`warm_start_k2.json`, `warm_start_k3.json`: per start rule A-D of the adaptive method (`start`, `reset`) the mean
curve over the three seeds, the final values per seed and their mean, rejections, and the number of decisions that
did not start at the last accepted point.

`screening_k2.json`, `screening_k3.json`: one record per digit pair or triple (most conflicting first): the
lookahead affinities `Z` at the checkpoints, `C`, `c_j`, `C_bal` and `C_mean`.
