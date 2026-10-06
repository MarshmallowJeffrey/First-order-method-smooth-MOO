# Results of the paper's runs

Written by `scripts/analyze.py`, `scripts/step_rules.py`, `scripts/warm_start.py` and `scripts/screening.py` from
the runs of the paper (NVIDIA RTX A5000, float64); read by `scripts/make_figures.py` and `scripts/make_tables.py`.

`k2.json`, `k3.json` ({4,9} and {4,7,9}, B = 480,000 gradient calls)
* `runs.<leg>`: `method` (adaptive = GRAB, uniform, surf), `param` (r or N), `seed`, `budget`; the checkpoints
  `ck_grads` (gradient calls) and `ck_wall` (wall-clock seconds); `audit_gn`, the audited worst-case gradient norm
  max_lambda GN(lambda, B_t) at each checkpoint (K = 2 exact, K = 3 a lower bound); `levels` and `B_run`, the plateau
  test (`B_run` null: no plateau); `marker` {x, y, wall} (K = 2: located by bisection, `m` = bundle size; the
  checkpoint-based marker is in `marker_checkpoint`); timings (`decision_seconds`: the lambda-search of GRAB),
  segments and rejections; for GRAB `selector`, its lambda-search (K = 2: `envelope`, the exact lower envelope;
  K = 3: `ccp_cg`, Algorithm 2 with constraint generation).
* `configs`: per configuration the geometric means over the seeds of the marker (`y_geomean`, `x_geomean`,
  `wall_geomean`), `x_median` and `n_plateau`, the number of seeds that plateau.
* `adaptive_final`, `adaptive_final_geomean`: GRAB at the end of the budget; `adaptive_selector`: its lambda-search.

`k2_fronts.json`: per leg the non-dominated training objective values (F_4, F_9) of all visited points.
`k3_fronts.json`: per leg the non-dominated values (F_4, F_7, F_9) with every objective at most 0.5.

`step_rules_k2.json`: per step rule the mean curve over the three seeds (`ck_grads`, `ck_wall_mean`, `gn_mean`), the
final values per seed and their mean; `ranking` from the lowest mean; `selector`, the lambda-search of the runs.

`warm_start_k2.json`: per start rule A-D of GRAB (`start`, `reset`) the mean curve over the three seeds, the final
values per seed and their mean, rejections, and the number of decisions that did not start at the last accepted point;
`selector` as above.

`screening_k2.json`, `screening_k3.json`: one record per digit pair or triple (most conflicting first): the
lookahead affinities `Z` at the checkpoints, `C`, `c_j`, `C_bal` and `C_mean`.
