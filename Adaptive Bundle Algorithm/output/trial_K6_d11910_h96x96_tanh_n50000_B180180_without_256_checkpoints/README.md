# K=6 parameter trial (96x96, budget mode, without-256 track)

One-off parameter-combination trial, July 11, 2026 — NOT part of the
plateau or crossover experiment families.  Everything here was produced
by `Original_py/run_trial_K6_without_256_checkpoints.py`; the module
docstring there derives every parameter and is the full specification.
Change record: `Note/Jul_11_note.md`.

- Problem: K=6, p=20, n=50000,
  hidden_sizes=[96, 96] (d=11910), tanh,
  seeds 7/8 — the crossover 96x96 instance with K=2
  replaced by K=6 (d moves 11522 -> 11910 because the output head is
  96*K+K).
- Budget mode, B=180180 gradient evaluations;
  checkpoints record SELF-REPORTED worst-case GN (baseline: max over its
  own grid nodes at their own weights, every
  4500 grads; adaptive: its own
  64-start lambda-search value, every
  600 grads) — no 256-start external
  solves.
- Baseline grid: r=10 (user-fixed floor) ->
  3003 nodes (C(K-1+r, K-1)); one pass costs 90090 grads, so
  the budget allows 2.0 passes — B
  was raised from the reference 2,000 precisely so the baseline can
  polish every node twice (within one pass its self-reported max still
  contains never-visited nodes at x0 and cannot move).
- Adaptive: max_inner=25 (plateau-family K>=3 value),
  lambda_max_starts=64 (K=6 precedent; the search value IS the plotted
  metric), and a BINDING round fuse max_outer=150
  (= 22500 grads,
  12%
  of B): a July 11 calibration run measured the lambda-search cost at
  ~0.0137 s/start/bundle-point per round, which makes full-budget rounds
  (1,201) infeasible (~176 h); the asymmetry is conservative AGAINST the
  adaptive method unless it has already plateaued — the curves and the
  plateau detector show whether it has.  All other parameters equal the
  reference run `output/crossover/d11522_h96x96_tanh_n50000_B2000/`.
- Figures: `gn_vs_grad_evals_without_256_checkpoints.png`,
  `gn_vs_cpu_time_without_256_checkpoints.png` (log time axis; the
  vertical line marks the smaller of the two final times — the
  equal-time comparison point).  `summary.json` has the curves, plateau
  detection (window 4, tol
  0.05, consecutive
  2), and the symmetric time-to-target
  block.

Caveats, stated once: the two curves are DIFFERENT self-reported
quantities (the baseline never looks between its grid nodes; the
adaptive value is a heuristic lower bound of an NP-hard max) — they are
each method's own honest progress meter, not a shared yardstick; for
cross-method quality claims at a shared metric, a GN* (256-start) run
would be needed.  Single instance, seeds 7/8;
prune_inner=True voids the full-bundle proof condition (warning recorded
in summary.json) but not the trajectories.

## Execution record (appended after completion)

Launched 2026-07-12 14:32:39 (nohup-detached; a first attempt at 02:20
was killed by the OS at ~02:26 under memory pressure and left no
artifacts).  Completed 18:36, DONE in 14,534 s: baseline 4,157.5 s
(oracle pace 0.1384 s/joint-call, within 6.5% of the 0.13 s
calibration), adaptive 10,375.0 s (150/150 rounds).  Extra files beyond
the standard set: `run_log.txt` (full stdout) and `machine_load_log.txt`
(uptime/vm_stat every 300 s for post-hoc audit of the serial-idle
assumption — 1-min load was 6–8 during the first ~75 min from
concurrent interactive use, ~3.5–5 afterwards).  Headline readings and
the hypothesis assessment: `Note/Jul_11_note.md` §7 of task 2.
