# Aug 9 note — CCP vs IPOPT comparison campaign (design + change record)

Session of Aug 9, 2026.  User-directed: run the two agreed comparison
experiments.  NEW FILES ONLY; the July outputs under
`pure_budget_K2_without_256_checkpoints/B20000` and
`baseline_svrg_multi_r_vs_fast_v2_without_256_checkpoints/pure_budget_B80912`
are never touched.  All campaign outputs live in a NEW home:

    output/ccp_compare_without_256_checkpoints/
        K2_B20000/           experiment 1, K = 2 (6 legs + figures)
        K6_B80912/           experiment 1, K = 6 (8 legs + figures)
        lambda_solver_bench/ experiment 2 (snapshots, paired 2a, fixed-time 2b)
        K2_campaign.log / K6_campaign.log

Design decisions locked in the Aug-9 Q&A (user-confirmed):

1. Both solvers run on the Gram form (already true everywhere).
2. Experiment 1 = full same-machine rerun of ALL legs (option A), so
   the CPU axis is clean.  Each method keeps its own targeting
   machinery (IPOPT: structured starts + single prev_lam chain; CCP:
   vertices + lambda_A + pool + Exp(1) seeds, screen -> r = 10
   polishes, N0 = 2000).  NO seed sharing in experiment 1 — the
   quality yardstick is the shared off-axis audit, not shared seeds.
3. prev_lam stays a single point for IPOPT (July design: each start is
   an expensive local solve; the pool-vs-single-point asymmetry is
   precisely the paper's §5 point that cheap restarts afford a pool).
   An "IPOPT + pool" arm is possible future work, not part of this
   campaign.
4. K = 2 audit: exact 1-D meter (unchanged).  K = 6 audit UPGRADED to
   two-instrument max: audit_v2 = max(strict-64 IPOPT search,
   heavy CCP solve (N = 8192, r = 20, single round, no pool)) per
   audited stack — both are lower bounds of the true max, the max is a
   tighter lower bound, and the instrument is method-symmetric.
   Computed post-hoc from grams.npz prefixes (off both cost axes).
5. Gap figures: K2 gap = phi_exact − phi(lambda chosen) per DECISION
   (exact meter, 20001-grid + polish for the curve); K6 gap at
   CHECKPOINT granularity vs audit_v2 (decision-granularity strict-64
   would be prohibitively slow; noted in the caption).
6. Experiment 2 runs on 6 frozen Gram snapshots (early/mid/late of the
   K2 and K6 adaptive-IPOPT legs' delivered stacks).
   2a paired: n_seed_batches = 20 batches (renamed from S to avoid the
   pure-budget "s" collision) x same screened top-10 starts to BOTH
   polishers; boxplots show ALL raw per-restart samples (200 per box),
   never batch means.
   2b fixed-time: ONE 60 s trajectory per (snapshot, method) from one
   shared ordered seed stream, no screening for either method;
   best@T for any T is read as a prefix with the STRICT completion
   rule t_complete <= T (in-flight restarts finish and are recorded,
   but never count toward a cutoff they crossed — no timeout bonus for
   slow restarts).  Reported: best@10s, best@60s, restarts completed,
   best-so-far vs time AND vs restart index, per-restart time
   distributions, distinct-maxima counts (same dedup for both).
7. Convergence bookkeeping: unified stationarity residual
   delta(lambda*) = val(M^(lambda*)) − phi(lambda*) for BOTH methods
   (one game LP; Lemma 11); IPOPT KKT/feasibility residuals dropped
   (cyipopt does not expose them cleanly) — kept: IPOPT success/status
   (+ iteration count if exposed), CCP iters + final delta_c.
8. Objective normalisation: s_k ≡ 1, one shared problem instance per K
   (data_seed = 7, init_seed = 8, shared L calibration); scaling
   sensitivity (e.g. s_k = L_k) is FUTURE WORK, separate report
   (also recorded in Aug_8_note §6).

## New files (this campaign)

* `run_pure_budget_K2_ccp_without_256_checkpoints.py` — the CCP
  targeting leg for the original K2 runner protocol (policy
  `adaptive_ccp`, leg dir `adaptive_s{s}_ccp`); per-decision CCP
  telemetry appended to summary.json as the `ccp` block.  Smoke: CCP
  targeting value == exact meter to 2.6e-16 rel on the final stack.
* `run_ccp_compare_K2_without_256_checkpoints.py` — campaign
  orchestrator: serial fresh rerun of baseline_r{10,20,40,80}_s5 +
  adaptive_s5_ts24 (ts64 twin is bit-identical at K2, Jul_30 §7a, not
  rerun) + adaptive_s5_ccp into K2_B20000; resumable (legs skip if
  summary exists); campaign_manifest.json records order + wall times.
* (to come, same session) K6 orchestrator + audit_v2 script + figure
  scripts + experiment-2 bench scripts — appended below as they land.

## Results (all runs complete)

### Experiment 1 — K2, B = 20000 (fresh same-machine rerun, 4981 s)

| leg | wall s | decision s | final EXACT audit |
|---|---|---|---|
| baseline r10/20/40/80, s5 | 706 / 682 / 797 / 749 | ~0 | 1.61e-4 / 6.20e-4 / 2.70e-4 / 2.98e-4 |
| adaptive_s5_ts24 (IPOPT) | 1243.6 | 560.1 | 9.518551e-5 (bit-reproduces July) |
| adaptive_s5_ccp | 802.7 | **103.5** | **7.934e-6** (12x below IPOPT) |

CCP telemetry: 641 decisions, 13398 CCP iters (~2.1 per restart — the
pool warm start is doing the work), 1 sandwich closure, pool pinned at
cap 30 with n_distinct 31 late (1 dropped/round — borderline; 4r is
the flagged ablation).  Gap-per-decision (exact meter, figure): CCP
targeting gap sits at ~1e-9; IPOPT's plateaus at 1e-3..1e-5 — the
structured 7-start search systematically misses the true worst lambda,
which is WHY the CCP leg's delivered set audits 12x better.

### Experiment 1 — K6, B = 80912

* **audit_v2 vindicated the instrument-bias concern**: on the IPOPT
  leg's stacks the CCP instrument was tighter on 42/42 checkpoints
  (final: strict-64 said 4.616e-2, true GN* >= 6.284e-2); on the CCP
  leg's own stacks 34/42.  All figures use v2.
* Final v2: **CCP 3.625e-2 vs IPOPT 6.284e-2 (42% lower)**; leg wall
  1575 vs 3978 s; decision time 484 vs 2799 s (5.8x).  IPOPT leg
  bit-reproduces July under strict-64 (0.046160).  Best baseline
  (r10_s1) v2 = 1.365e-1; the other baselines >= 3.18.
* Anomaly handled: first run of baseline_r15_s5 had wall 21252 s
  (suspected machine sleep; delivered set identical, audit unchanged)
  — archived as baseline_r15_s5_wallclock_artifact/, leg rerun clean
  (1338.8 s); manifest of run 1 kept as campaign_manifest_run1.json.

### Experiment 2 — λ-solver bench (6 snapshots, r = 10, 20 batches, T = 60 s)

2a paired, same screened starts (200 pairs per snapshot):

| snapshot | median restart (ccp / ipopt) | δ-converged (ccp / ipopt) | paired ccp wins/ties of 200 |
|---|---|---|---|
| K2_early_m41 | 0.5 ms / 117 ms | 1.00 / 0.75 | 84 / 11 |
| K2_mid_m1641 | 15 ms / 142 ms | 1.00 / 0.32 | 143 / 2 |
| K2_late_m3195 | 39 ms / 163 ms | 0.96 / 0.33 | 133 / 2 |
| K6_early_m108 | 2.8 ms / 82 ms | 1.00 / 0.02 | 183 / 0 |
| K6_mid_m2248 | 105 ms / 141 ms | 1.00 / 0.03 | 184 / 0 |
| K6_late_m4309 | 291 ms / 192 ms | 1.00 / 0.02 | 178 / 0 |

Reading: from identical starts CCP wins the paired value comparison
everywhere except the tiny K2_early stack (near-equivalent there), and
under the unified residual δ(λ*) IPOPT's single local solves terminate
non-stationary on 25–98% of starts (worst at K6) — the microscopic
cause of the K2 gap plateaus.  δ-convergence uses the same τ for both.

2b fixed-time, one 60 s shared stream, strict t_complete cutoffs:
throughput CCP/IPOPT = 176x / 4.3x / 2.1x / 24x / 1.2x / 0.63x
(ordered as the table); best@60s ties on K2/K6_early, CCP +9% at
K6_mid, +23% at K6_late.  The 0.63x at m=4309 is the `changeCoeff`
Python loop dominating a cold restart (~10–20 LP rewrites x 43k
entries); the production pipeline is unaffected because pool-warm
restarts need ~1.6 iterations (K6 leg: 0.48 s per full decision), but
a bulk-rewrite of the LP payoff is the flagged engineering follow-up.

### Figures

K2_B20000/: K2_{gn_vs_grads, gn_vs_cpu, gap_vs_decision, fronts}.png
K6_B80912/: K6_{gn_vs_grads, gn_vs_cpu, gap_vs_decision, fronts}.png
lambda_solver_bench/: fig_2a_time_box, fig_2b_best_vs_time,
fig_2b_best_vs_restarts (+ bench_2a/2b.csv, bench_summary.json,
summary.md, snapshots/).

### Aug-10 experiment-1 revisions (user-requested)

1. gn-vs-CPU (and grads) figures: both adaptive curves now start from
   one shared origin (shared pseudo-zero floor; the first audited
   stack is the identical {x0} bundle, so the starting point is the
   same in x AND y).
2. K2 fronts figure: three connected Pareto curves on log-log axes
   (was a collapsed linear scatter).  New honest reading recorded: on
   the FRONT view baseline_r10 traces the lowest central curve and CCP
   dominates IPOPT almost everywhere — front metrics and GN* measure
   different things.
3. K6 figures keep only baseline_r10_s1 (the other five ≥ 3.18
   compressed the axis).
4. K6 fronts evaluation replaced by the user's central-reference-front
   method (slide): R_central = union-front points with all objectives
   <= c; c = 1 is empty on this instance (every point has some class
   CE > 2), so c = median of per-point max_k = 6.95, |R_central| = 12,
   z_ideal from R_central; Central IGD / max-distance / HV (Sobol 2^19
   in [z_ideal, c]^6).  Numbers: IGD 0.751 (IPOPT) / 1.906 (CCP) /
   2.120 (bl); max-dist 3.23 / 4.70 / 7.14; HV 0.822 / 0.752 / 0.295.
   HONEST FINDING: the central-front metrics favour IPOPT — CCP's
   sharper targeting concentrates on worst-direction (often
   near-vertex) specialist points (8 central points vs 17), while GN*
   favours CCP by 42%.  Both views reported; possible future work:
   mixing a few grid lambdas into the CCP leg.
   Implemented in plot_ccp_compare_without_256_checkpoints.py
   (_central_front_metrics_K6; json K6_front_metrics_ccp_compare.json,
   figure K6_front_central_metrics.png).
5. The report file edited IN PLACE on the user's own revision
   (~/Desktop/"Experiment1_report_K2_K6 2.docx"; docx-skill XML
   surgery: 6 media swapped, image-6 extent resized, caption 2-4 +
   three analysis paragraphs updated, one sentence appended to 1-2;
   all other user edits preserved; validate.py PASSED).  Pre-edit
   backup: output/ccp_compare_without_256_checkpoints/
   Experiment1_report_user_edit_backup_aug10.docx.

### Reports

Experiment-2 report (simple Chinese, 2a/2b separately: algorithms,
definition/principle/idea, parameter tables, all figures with
analysis — 2b figures re-laid as 2x3 grids
``fig_2b_*_grid.png`` in lambda_solver_bench/ — and py-file tables):
``~/Desktop/Experiment2_report_2a_2b.docx``.

Experiment-1 report (Chinese, K2 & K6 separately: algorithm tables,
parameter tables, unified-audit definitions, all 8 figures with
per-figure analysis, py-file tables) generated Aug 9; the .docx now lives at ``~/Desktop/Experiment1_report_K2_K6.docx``
(user request); a .pdf render stays in
``output/ccp_compare_without_256_checkpoints/``.

### Recorded follow-ups

1. pool_cap 3r borderline at both K (n_distinct 31–32 vs cap 30 in
   late rounds) → 4r ablation candidate.
2. `_GameLP` bulk payoff rewrite for m ≳ 2000 (replaces the per-entry
   changeCoeff loop; would cut CCP decision time further).
3. τ / IPOPT-tol ×10/÷10 sensitivity ablation: designed, NOT yet run.
4. Objective upgrade discussion (MNIST per-class K=10 trial, softplus,
   λ-aware stratified sampling, small-batch sweep) — separate design
   thread, pending user go.
