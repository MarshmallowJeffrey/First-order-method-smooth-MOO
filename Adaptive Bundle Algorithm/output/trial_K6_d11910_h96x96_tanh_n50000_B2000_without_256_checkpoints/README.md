> **SUPERSEDED — calibration data only (July 11, 2026).**  This run used
> r=2, which the user subsequently rejected (hard floor: r >= 10).  It is
> kept ONLY because its measured costs dimension the real r=10 trial:
> joint-oracle call ~0.13 s (K=6, n=50k, d=11910); lambda search
> ~0.0137 s per start per bundle point per round (64 starts).  Do NOT
> cite its figures/summary as experiment results.  See Note/Jul_11_note.md.

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
- Budget mode, B=2000 gradient evaluations for each
  method; checkpoints every 153 grads,
  SELF-REPORTED worst-case GN (baseline: max over its own grid nodes at
  their own weights; adaptive: its own 64-start
  lambda-search value) — no 256-start external solves.
- Baseline grid: r=2 -> 21 nodes
  (C(K-1+r, K-1)); one pass costs 630 grads, so the budget
  allows 3.2 passes.  r was reduced
  from the reference r=10 because at K=6 that grid would have 3003 nodes
  (45x the budget per pass); lambda_max_starts was raised 8 -> 64 (K=6
  precedent; the search domain is the 5-D simplex and its value IS the
  plotted metric).  All other parameters equal the reference run
  `output/crossover/d11522_h96x96_tanh_n50000_B2000/`.
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
