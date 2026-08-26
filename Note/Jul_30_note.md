# Jul 30 note — K=2 pure-budget front experiment: new runner with an exact 1-D meter

Session of July 30, 2026.  User-directed.  This note is the change
record for all code written this session (user instruction: "所有的
code改动记录到Note里面，新的Jul_30_Note").  Engine files untouched.

## 1. Context and approved design

Goal (user's Experiment 1 requirement): 2-layer MLP, multi-class
classification, logistic-regression loss, our algorithm vs the uniform
discretization baseline, epsilon-Pareto front + grad/CPU efficiency —
now unified under ONE protocol.  The July-8 K=2 Pareto-front runs and
the July-27 K=6 pure-budget runs used different protocols/instruments;
this experiment redoes the front comparison under the July-27 pure
fixed-budget protocol, at K=2 where the front is plottable and the
meter is exactly computable.

Decisions taken with the user this session:

* r ladder = {10, 20, 40, 80} -> grids of 11 / 21 / 41 / 81 nodes
  (K=2: C(r+1, 1) = r+1).  The K=6 ladder {10,12,15,20} was rejected
  for K=2: it spans only 11->21 nodes (1.9x) and would artificially cap
  the baseline's front menu.
* The user's initial budget rule ("B such that >= 50% of baseline nodes
  get computed") was analysed and dropped as vacuous at K=2: one
  segment costs 13 * (2*4096*2/50000) + 2 ~= 6.26 grad-equivalents, so
  a full r=80 pass at s=5 costs ~2,540 and ANY sane budget covers every
  grid many times over.  Replacement rule (user asked for the
  rationale, given in conversation): B ~= 2-3x the flattening point of
  the largest-r baseline leg, measured by a cheap pilot; safety factor
  covers (a) the asymptotic approach to the floor (a visually-read
  flattening point understates true convergence — protects the
  endpoint verdict from a "baseline was cut early" objection, same
  fairness logic that rejected mid-run truncation on Jul 27), (b) the
  need for a visibly flat tail in the figure (the plateau claim should
  be graphical, not asserted), (c) single-realization noise (MLP torch
  runs are not bit-reproducible here).
* s: keep {1, 5} as sensitivity.  Pre-registered prediction (stated
  before any run): s=1 vs s=5 difference ~ zero at K=2, because s only
  matters when coverage is scarce (K=6: one pass never completed, s
  decided coverage itself); at K=2 every leg completes many passes, so
  per-node total work is s-independent and only the depth of the
  warm-start state handed along the chain changes, which washes out
  after the first pass.
* Meter: EXACT at K=2.  lam(w) = (w, 1-w) makes each delivered point's
  GN contribution q_i(w) = lam^T M_i lam a convex quadratic in w with
  coefficients read off the stored 2x2 Gram; GN(w) = min_i q_i(w) is
  evaluated by plain arithmetic everywhere, so max_w GN(w) is computed
  on a dense w-grid (200,001 points, vectorised, chunked) plus a
  closed-form polish of the winning cell (envelope max sits at an
  endpoint or an active-pair crossing; candidate crossings solved
  exactly, full envelope re-evaluated there).  No multistart search
  anywhere in this experiment -> the eps1e-4 false-certificate failure
  mode (Note/Jul_27_note.md section 8) is structurally impossible.

## 2. New file: `Original_py/run_pure_budget_K2_without_256_checkpoints.py`

Reused verbatim from `run_pure_budget_K6_without_256_checkpoints.py`
(imports, not copies, where possible): the pure-budget protocol
executor (shared segment unit = epoch_len(=13 at trial scale) MSVRG
minibatch steps + 1 charged full joint eval; shared s; chain warm
start; descent safeguard with MAX_SAFEGUARD_RETRIES; stop = budget),
`_Budget` meter, `_baseline_policy` (snake grid, cycling passes),
`_leg_dir`, seeds (data 7 / init 8 / sampler 41), instance family
(p=20, n=50,000, [96,96] tanh, b=4096) with K=2.

New in this runner:

* `_quad_coeffs` / `_env_at` / `exact_gn_1d(Ms, grid_points, polish)`:
  the exact 1-D meter described above.
* `_adaptive_policy_exact(decision_grid)`: adaptive targeting = exact
  worst-w on a COARSER grid (default 2,001 points, no polish), on-axis
  (inside the leg's wall clock).  Audits use the fine grid (default
  200,001) + polish, off-axis.  Grid sizes are CLI args
  (--decision-grid / --audit-grid) and are recorded in summary.json.
* Exact prefix audits at EVERY checkpoint for BOTH policies (K=6 gave
  the baseline only a final search; --backfill-audits is gone).  Exact
  prefix values are monotone non-increasing by mathematics, so the K6
  monotone-envelope repair is REMOVED and replaced by an assert
  (violation would mean a bug, not measurement noise).
* Per-segment recording for fronts: grams.npz gains `seg_grads`
  (cumulative grad-equivalent spend at each delivery) and `seg_lams`
  (lambda of each segment) next to the existing `gram_stack`, `fvals`,
  `lam_history` -> the delivered set and its front are reconstructable
  at ANY prefix budget.
* `--figure` now draws three artifacts into
  `output/pure_budget_K2_without_256_checkpoints/B<budget>/`:
  - `pure_budget_K2_gn_vs_grads.png` / `..._gn_vs_cpu.png`: FULL
    audited trajectories for every leg (baseline trajectories are
    plotted at K=2 — the per-r flattening points are a design readout
    of this experiment; deliberate departure from the K=6 figure's
    final-points-only style, disclosed here and in the README);
  - `pure_budget_K2_fronts.png`: delivered (F1, F2) clouds (faint) +
    per-method nondominated fronts + the union front (dotted, the
    mutual reference — no oracle front exists for this family);
  - `front_metrics.json` + README tables: per leg, delivered points,
    front size, IGD and max-dist from the union front (raw value-space
    Euclidean).
* Smoke mode (`--smoke`): tiny instance (K=2, p=6, n=300, h=[8],
  b=60, epoch 5), budget 400, r=4 legs at s=2/s=1 + adaptive s=2,
  full figure pass, then asserts: budget respected, finite audits,
  monotone exact prefix audits, fvals width 2, seg_grads aligned,
  front artifacts exist, and EXACT >= multistart-search on every leg's
  final stack (the search is a lower bound of the true max; run with
  tier="strict", 16 starts, when IPOPT is available).

## 3. Smoke result (July 30)

All checks passed ("SMOKE OK").  Readings (tiny instance, not
meaningful beyond validation): baseline r4 s2 audit 7.823190e-2;
baseline r4 s1 audit 1.775941e-1; adaptive s2 audit 5.189092e-3
(93 segments vs 76 — the adaptive spends fewer units on decisions than
the baseline's snake at this scale, so it fits more segments; decision
wall time ~0.0 s).  Exact-vs-search cross-check: exact >= search on
all three stacks, agreeing to 6-7 significant digits (the smoke stacks
are easy enough for the 16-start search to find the true max; the
exact meter confirms it from above rather than below).

## 4. Bookkeeping corrections found this session (for the next ledger rewrite)

* The July-27 K=6 pure-budget runner ALREADY stores the objective
  vector of every delivered point (`fvals` in grams.npz, shape
  (m, 6)).  An earlier statement in this session's conversation
  ("delivered F values were not recorded on Jul 27") was WRONG and is
  corrected: what is not stored are the parameter vectors theta (the
  ledger's "Grams only" caveat is about coordinates).  Consequence:
  K=6 post-hoc front METRICS are computable from the existing Jul-27
  artifacts without any re-run (6-D fronts cannot be plotted, but
  nondominated filtering and mutual-reference metrics work).  Not done
  this session (not requested).
* `Note/Jul_20_note.md` section 7: the wrong "~16x CPU" figure was
  ALREADY corrected to 8.8x on July 25 (in-place [CORRECTED
  2026-07-25] annotation; no other "16x" instance in the file).  The
  ledger's "STILL unfixed" line is stale.
* The ledger also predates the July-27 early-morning eps1e-3 / eps1e-4
  re-runs (Note/Jul_27_note.md sections 7-8, incl. the eps1e-4 false
  adaptive certificate) — known since the start of this session,
  restated here so the next ledger rewrite folds all three items in.

## 5. Same-session addition: certified two-sided bound on the exact meter

Trigger: the user challenged the word "exact" — "a 200,001-point grid
is just more points than 64 starts; it does not guarantee the true
result either."  The challenge is correct for an arbitrary black-box
function (dense sampling proves nothing between samples), and it
exposed an unredeemed corner of the first implementation: the
closed-form polish covered only the winning grid cell, so a spike in
another cell was excluded only up to an UNSTATED slope*h argument.

Fix (same file, `exact_gn_1d(..., certify=True)`): the meter now
returns a proven upper bound next to the true-value lower bound.
Mathematics: inside any grid cell [w_j, w_j + h], env(w) <= q_i(w)
for the quadratic i active at either cell end (env is a min, so any
member bounds it), and a quadratic with known coefficients moves by at
most max|q_i'| * h across the cell; taking the tighter of the
left-anchored and right-anchored bounds cell-wise and maximising over
cells yields a rigorous upper bound U.  The reported value V is a true
function value (grid + polished crossings), so the true GN* lies in
[V, U] — per audit, stored, asserted (U >= V), with the leg's widest
interval in `audit_certified_gap_max`.  What distinguishes this from
multistart search is not the point count but the existence of ANY
provable bound: search under-reports by unbounded amounts when it
misses a region (the eps1e-4 case); the grid+structure meter cannot,
and now says so quantitatively.

New summary keys: `final_audit_upper`, `audited_gn_upper_history`,
`audit_certified_gap_max`.  Smoke re-run after the change: all passes
again; sandwich verified per leg (16-start strict search <= exact <=
upper): gaps 6.92e-9 / 1.12e-8 / 8.20e-9 on the smoke audit grid
(20,001 points, h = 5e-5); the trial audit grid (200,001, h = 5e-6)
tightens h by 10x.  Figure/README template updated to state the
certified-interval semantics.

Follow-up (user: "这个修复有什么用? pareto front 图用得到吗?"): the
upper bound U is what SIGNS positive claims.  A search/grid value is a
lower bound — it can prove "quality no better than v" but never
"GN* <= eps"; quoting a lower-bound instrument for a positive claim is
exactly how the eps1e-4 false certificate happened.  Uses wired in:
(1) every audited curve point and final_audit now carries its stored
interval width, so quoting V as "the value" is quantitatively
justified per run; (2) the front table/front_metrics.json now quote
each leg's final U as the leg's **certified eps**: for every w in
[0, 1] the delivered set provably contains a point with
lam(w)-weighted gradient-norm^2 <= U — the "eps" of "eps-Pareto
front" in the stationarity sense.  The front FIGURE's plotted (F1, F2)
points do not pass through the GN meter at all (full-batch joint
evaluations, exact by construction); it is the front's eps LABEL that
consumes the certificate.  Smoke front metrics: baseline_r4 certified
eps 7.823191e-2, adaptive 5.189099e-3.

## 6. User decision (same day, later): adaptive targeting reverted to the strict multistart IPOPT search

After a long Q&A about what the 2,001-point decision grid does (see the
conversation; the ledger's next rewrite should keep only this summary),
the user ruled: the adaptive method's OWN lambda selection must stay
the strict multistart IPOPT worst-lambda search — the same mechanism
as the K = 6 runner — not a predefined candidate grid.  Directive:
"K=2的实验，你把adaptive bundle method改回 64 starts - IPOPT求解器，
不要用什么2001个点".

Changes (same file):
* `--decision-mode {search, grid}`, default **search**;
  `--targeting-starts`, default **64** (per the directive; K = 6's
  targeting used 24 — settable if parity is ever wanted).  The
  adaptive policy is now imported from the K6 runner
  (`_adaptive_policy`).  IPOPT is required in search mode.
* Grid targeting (`_adaptive_policy_exact`) is retained ONLY behind
  `--decision-mode grid` for a possible menu-density ablation leg;
  off by default; docstring marks it ablation-only.
* AUDITS UNCHANGED: exact 1-D meter + certified two-sided bound,
  200,001-point grid, off-axis, both methods.  The reverted search
  affects only the method's own navigation; its misses can hurt only
  the adaptive's efficiency, never the reported quality.
* summary.json now records `decision_mode` / `targeting_starts` /
  `decision_grid` (mode-dependent); README template describes the
  policy mode-aware.

Smoke re-run (targeting_starts=8 for speed): ALL PASS; sandwich
(search <= exact <= certified upper) holds on all three legs.
Two observations recorded honestly:
* On-axis decision cost is back, dominant even at toy scale: the
  adaptive smoke leg's wall went 0.4 s -> 16.0 s, of which 15.6 s is
  decision time (33 decisions, bundle <= 67 points, 8 starts).
* Segment counts differ between targeting modes at equal budget
  (66 vs 93): grid targeting had aimed exactly at vertices w = 0 / 1,
  where `_support_batch` restricts minibatch rows to the supported
  class, making segments cheaper; the search lands continuous
  (near-)interior lambdas charged at full rows.  Both are correct
  under the accounting; the policies simply aim differently.  Final
  smoke audits: 9.487851e-3 (search) vs 5.189092e-3 (grid) — single
  toy realization, no conclusion drawn.
* Pilot cost consequence at trial scale (forecast, to be measured):
  per-decision cost ~ 64 IPOPT starts x (fixed solve overhead +
  m-dependent Gram-path part); at s = 1 the leg makes ~3,195 decisions
  with bundles growing to ~3,200 points — the decision time could
  reach multi-hour scale.  Revised pilot plan: run the adaptive pilot
  leg at s = 5 first (~639 decisions) to MEASURE the true pace, then
  forecast and schedule the s = 1 leg from the measurement.

## 6a. User's pilot revisions (same day) + supporting code changes

User-ordered pilot changes: (1) baseline pilot legs run at s = 5 (not
s = 1); (2) BOTH targeting-start counts get their own adaptive leg —
24 and 64; (3) user asked for confirmation that everything, including
the Pareto-front figure, is fixed-budget (confirmed: every leg stops
exactly at budget B; the front figure is drawn from the delivered sets
at that same B for every leg; per-segment recording additionally
allows equal-prefix-budget front cuts post hoc).

Supporting code (same file): `_leg_dir_k2` names adaptive legs with
their targeting spec (`adaptive_s5_ts24` / `adaptive_s5_ts64` /
`adaptive_s5_grid81`), so several adaptive configurations coexist
under one budget home; the GN figures, front figure, front_metrics
keys and README rows are multi-adaptive-aware (per-leg labels
`adaptive 24-start` etc.; colors: 64-start green '^', 24-start teal
'v', grid-ablation purple 's'); `--s` default changed 1 -> 5 (the
protocol's main s).  Smoke re-run: ALL PASS (leg dir now
`adaptive_s2_ts8`).

## 7. Pilot execution log

* Attempt 1 (user go, evening): launched with third-party load on the
  machine (~35-50% CPU from other apps; disclosed to the user with the
  caveat that CPU-axis readings would be estimates only).  Leg r=10
  finished its core run (19,998.7 of 20,000 grad-equiv, 3,404
  segments, wall 546.1 s dirty-machine) but the host app restarted
  during its off-axis audits, killing the shell.  NO summaries were
  written (B20000/ empty) — nothing to discard; console log preserved
  as `pilot_B20000_console_attempt1_aborted.log`.
* The user then closed the load-generating apps and ordered a clean
  re-run of everything ("确保所有实验都是干净的").
* Attempt 2 (clean): same six-leg serial chain relaunched under
  caffeinate with a settle gate (wait up to 5 min for 1-min loadavg
  < 3 before the first leg).  Console: `pilot_B20000_console.log`.
  With the machine quiet, ALL pilot readings (grads AND CPU axes) are
  record-grade; the attempt-1 caveat about CPU numbers no longer
  applies.
* Stale folder note: `B20000_SMOKE/` is a leftover of the FIRST smoke
  iteration (before the smoke home was pinned to `B400_SMOKE`);
  harmless, clearly suffixed, left in place.

## 7a. Pilot RESULTS (clean run, B = 20,000; all readings record-grade)

| leg | final exact audit | w* | wall s | decision s | segments | distinct lambdas | cert gap max |
|---|---|---|---|---|---|---|---|
| baseline r10 s5 | 1.6071e-4 | 0.0109 | 680.3 | 0 | 3,404 | 11 | 6.5e-7 |
| baseline r20 s5 | 6.2015e-4 | 0.0120 | 546.8 | 0 | 3,302 | 21 | 7.4e-7 |
| baseline r40 s5 | 2.7010e-4 | 0.0323 | 504.3 | 0 | 3,247 | 41 | 1.2e-6 |
| baseline r80 s5 | 2.9811e-4 | 0.9933 | 490.1 | 0 | 3,220 | 81 | 1.1e-6 |
| adaptive s5 ts24 | **9.5186e-5** | 0.0232 | 1,140.7 | 568.0 | 3,194 | 617 | 6.5e-7 |
| adaptive s5 ts64 | **9.5186e-5** | 0.0232 | 1,093.3 | 566.2 | 3,194 | 617 | 6.5e-7 |

Findings (single realization each; MLP-K2 torch proved DETERMINISTIC
here — see finding 2):

1. **Equal-budget quality (grads axis): adaptive wins 1.69x** over the
   best baseline (9.52e-5 vs r10's 1.61e-4), and the gn-vs-grads
   figure shows the adaptive curve at-or-below every baseline at
   essentially every prefix budget (anytime dominance).  Baseline
   r-order is NON-MONOTONE (r10 < r40 < r80 < r20): the audit peak
   sits in a near-vertex hard zone (w ~ 0.01-0.03 / mirrored at
   w ~ 0.99), and a leg's quality is set by whether a node lands
   inside that zone (grid PHASE: r40 has a node at 0.025, r20 has
   nothing between 0 and 0.05) plus per-node depth (B spread over
   r+1 nodes).  "Finer grid is better" fails at this budget.
2. **The 24- vs 64-start adaptive twins are BIT-IDENTICAL** —
   lam_history, fvals, audit histories all equal, and decision time
   equal too (568.0 vs 566.2 s; ~0.89 s/decision average).  At K=2
   both start counts find the same worst-w every round.  Consequence:
   record runs need ONE adaptive leg; 24 starts (K=6 targeting
   parity) recommended.  Side finding: identical trajectories across
   two independent processes ⇒ the K=6 bit-non-reproducibility
   finding does NOT extend to this K=2 setup.
3. **No leg flattened**: every audited curve is still descending at
   budget end (first entry within 1.2x of final only at 86-96% of B).
   B = 20,000 is below the floor-exhibition regime; the record budget
   must be larger, and the "2-3x the r80 flattening point" rule could
   not be applied literally.
4. **Decision cost on-axis**: 568 s = 50% of the adaptive wall at
   B=20k.  Cost per decision grows ~linearly with bundle size, so
   decision TOTAL scales ~quadratically with B (forecast: ~2.6 h at
   B=80,912, s=5).
5. **Value-front vs stationarity-front diverge**: raw front metrics
   are dominated by degenerate specialist tails — baseline VERTEX
   nodes camp on one-class objectives for many passes and drive one
   loss toward 0 (F1 up to 27 on the other axis); the adaptive,
   steering by GN, stops visiting a region once it is near-stationary
   and never chases those tails.  Raw IGD-to-union therefore ranks
   the adaptive LAST (1.02) while r10 ranks first (0.21).  Restricted
   to the central region (both losses <= 1): r10 igd 0.020, adaptive
   0.043 (best maxdist 0.178), r80 0.045 — near-tie.  Record-run
   figure needs log axes or a zoom inset + BOTH raw and
   central-region metrics reported side by side.

## 7b. Front-figure revision + SURF-paper alignment (user-approved "直接做")

Figure iteration (all replot-only, no method re-runs): the first
revision (linear central-region main panel + full-range inset) proved
unreadable — the fronts hug the axes because the trade-off knee lives
BELOW loss 1e-2.  Final form: **log-log main panel** (losses are
strictly positive, spanning ~1.6e-5 .. 27), dashed lines marking the
central-metric bound; clouds + per-method fronts + union front all
visible, tails included.  Metrics now stored in THREE groups in
front_metrics.json / README: raw IGD/max-dist (union-front reference),
central IGD/max-dist (reference restricted to both-losses <= 1.0,
method fronts never clipped), and — after the user asked to check the
SURF paper's figures — SURF-Table-1-style **HV / CV / Gap Ratio**
computed on each front's central part with HV reference point
(1, 1) (conventions disclosed in the README; SURF: Fig 6 = linear PF
scatter, Table 1 = HV/IGD/CV/GapRatio over 8 seeds).

**Finding from the SURF-style numbers (B=20,000 pilot):** they are
DEGENERATE on this instance — HV saturates (0.993-0.999 of the whole
unit box for every leg) and Gap Ratio explodes (8e3 .. 2e5) — because
the planted K=2 instance is nearly NON-CONFLICTING at optimum: both
per-class losses are simultaneously drivable to ~1e-4 (final GN* 1e-4
scale), so the "front" collapses to a knee at 1e-3..1e-4 scale plus
degenerate specialist tails, unlike SURF's bounded RL fronts.
Value-space and stationarity-space verdicts genuinely diverge here:
r10's 11 deep-fed node rays yield the deepest VALUE front in the
middle band (central IGD 0.020, best), while the adaptive owns the
STATIONARITY meter (certified eps 9.5e-5, 1.69x better).  Pending
user decision for the record run: keep this instance (GN story
headline, front figure as log-log diagnostic) vs add genuine conflict
to the planted data (e.g. lower w_true_scale / class overlap so
per-class losses cannot both vanish -> bounded curved front at O(1)
scale, SURF-style linear figure + meaningful HV/CV/GapRatio; would
need a fresh ~1.5-2 h pilot).

## 7c. Figure restyle per user review (Jul 31, replot-only)

User reviewed the pilot figures against the SURF paper's Figure 6 and
the K=6 pure-budget figures, and ordered: (1) the two GN figures
follow the K=6 pure-budget STYLE — adaptive legs as audited trajectory
curves, baseline legs as FINAL POINTS only (x markers, staggered
connector-linked labels "r=N: value", proxy legend entries); baseline
trajectories remain in the summaries, unplotted.  (2) the fronts
figure shows the per-method NONDOMINATED FRONTS ONLY, SURF-Fig-6
style — one marker-line per method, no delivered clouds, no union
overlay, no central-bound lines, short title, legend upper right.
Log-log axes retained (knee below 1e-2 on this instance; linear axes
cannot show the fronts — linear becomes viable only if the instance
gains genuine conflict, pending decision in section 7b).  All metrics
(raw / central / SURF-style) still computed and stored — display-only
change.  Known cosmetic blemish: with four baseline endpoint values
within ~0.6 decade, the K=6 stagger stacks labels downward and the
lowest ("r=10") crowds the x-axis; acceptable for the pilot, revisit
at the record run if it bothers.

Pilot (user-revised composition): six legs, serial, idle machine —

    python run_pure_budget_K2_without_256_checkpoints.py --run baseline --r 10 --s 5 --budget 20000
    ... --r 20 / --r 40 / --r 80 ... (all s 5)
    python ... --run adaptive --s 5 --targeting-starts 24 --budget 20000
    python ... --run adaptive --s 5 --targeting-starts 64 --budget 20000
    python ... --figure --s 5 --budget 20000

Provisional B = 20,000 grad-equivalents (~3,195 segments/leg,
~123 full r=20 passes — far past any plausible flattening),
eval-every 250.  Time estimate: baseline legs ~0.08-0.15 s/segment ->
~5-8 min/leg + ~1 min instance build; adaptive legs (s = 5, on-axis
IPOPT targeting): forecast 0.2-0.6 h (24-start) and 0.5-1.5 h
(64-start), to be MEASURED; ~1.5-2.5 h serial for all six, plus
off-axis exact audits (~1-2 min/leg).  After the pilot: read the r=80
flattening point, set B ~= 2-3x that, then run the record legs at the
chosen B (s = 1 sensitivity legs and the optional grid-81 ablation leg
are decisions deferred to that point).  The pilot legs at B=20,000
remain valid record legs if B=20,000 turns out to be the chosen
budget; otherwise they stay as the pilot record under `B20000/`.

## 7d. Data-to-figure verification after the user's challenge (Jul 31)

The user asked "are you sure the two GN figures are not mis-drawn?".
Independent verification, all from raw archives:

1. **Re-audit**: 12 checkpoints across baseline_r10 and adaptive_ts64
   recomputed from grams.npz with a fresh exact_gn_1d call — all 12
   MATCH the stored `audited_gn_history` the figures plot (<= 1e-12
   relative).
2. **Instrument cross-check on the final stacks**: strict 64-start
   IPOPT search vs exact meter: r10 search 1.607108e-4 = exact
   (wide vertex-gap peak, easy for search); adaptive search
   **1.740e-5 vs exact 9.519e-5 — the search UNDER-reads the
   adaptive's own stack by 5.5x** (margin-tuned family, narrow
   pockets — the eps1e-4 failure mode reproduced on MLP data).  With
   a K=6-style search meter the figures would have FLATTERED the
   adaptive 5.5x; the exact meter kept them honest.  Sandwich
   search <= exact <= certified-upper holds everywhere.
3. **Visual claims re-read from arrays**: r10 plateau 6.54e-3 from
   ~24 s to ~200 s (real); adaptive 8.0e-4 @ ~98 s, 1.42e-4 @ ~682 s.
4. **CORRECTION of the conversational commentary** (figures were
   right; my reading of them was not): the earlier claim "r10 edges
   ahead at the matched ~680 s window" was WRONG — at r10's finish
   time (680.3 s, final 1.607e-4) the adaptive already reads
   1.419e-4, i.e. the adaptive leads at essentially EVERY wall-clock
   time from ~10 s on; baseline r10's only remaining CPU-axis edge is
   finishing its budget 1.7x sooner.  Early-pace fairness confirmed:
   adaptive pays 49.3 vs r10's 31.5 ms/grad-equiv over the first
   ~2,500 grads (decision cost genuinely on-axis) and dominates
   anyway.

## 7e. Display rule: one adaptive line (Jul 31, user-directed, replot-only)

"adaptive只保留24 start的线": since the 24- and 64-start adaptive legs
are bit-identical twins (7a finding 2), the figures now DISPLAY only
the 24-start line — `_figure` hides a 64-start leg whenever a same-s
24-start leg exists (`ad_show` filter, commented in code).  The hidden
leg's data, summary and README leg-table row remain; front_metrics.json
now carries the 24-start entry only (the 64-start entry was a
duplicate).  All three figures regenerated.

Same-day addition (user): companion figure
`pure_budget_K2_gn_vs_cpu_trajectories.png` — the CPU axis WITH full
baseline trajectory lines (makes the plateaus visible; the
endpoint-style `pure_budget_K2_gn_vs_cpu.png` stays the figure of
record).  Generated by the same `--figure` pass.

Follow-up (user asked why the curves started at different x): the true
start is identical for every leg (shared x0, GN ~2.33, time 0); the
horizontal spread was the per-leg log-axis placeholder for t=0 (first
positive checkpoint / 3, legs reach their first checkpoint at
different wall times).  User chose the fix "所有腿的 0 时刻统一钉在
同一个人造位置": new `_pseudo_zero` helper anchors every leg's zero at
ONE figure-wide position (global min positive x / 3), applied to the
main GN figures and the companion; figures regenerated.

## 7f. Union-front composition figure (Jul 31, user request, replot-only)

The user computed the union-front ownership breakdown and asked for a
figure of the CENTRAL (both losses <= 1) union front.  `--figure` now
also emits `pure_budget_K2_union_front_composition.png` (central union
front as a grey staircase, each point coloured by the contributing
leg) and stores the counts in front_metrics.json under
`union_front_composition`.  Verified counts match the user's numbers
exactly — raw union 45 pts: r10 34 / r20 3 / r40 6 / r80 0 / adaptive
2; central 30 pts: r10 27 / r40 2 / adaptive 1.  (README table loops
now skip the non-leg composition key.)  Reading: at this budget, on
this near-conflict-free instance, the central union front is 90%
r10's — the depth-concentration mechanism of section 7a finding 5 in
its starkest form; the K=6 companion result (note section 9: adaptive
owns 86% of the central union front there) is the mirror image.

## 9. (SEPARATE SESSION, started July 30) K=6 epsilon-Pareto front metrics from the stored July-27 fvals

User request to this second session: complete the epsilon-Pareto-front
piece of the Experiment-1 requirement for the K=6 pure-budget
experiment, taking the SURF paper's figures/metrics as the comparison
frame.  Pure post-processing of the July-27 artifacts; nothing re-run.

* NEW FILE `Original_py/front_metrics_K6_pure_budget_without_256_checkpoints.py`
  (numpy + matplotlib only — deliberately imports NO engine/torch
  module so it stays runnable while experiments occupy the machine).
  Outputs into `output/.../pure_budget_B80912/`: `FRONTS.md`,
  `front_metrics.json`, `pure_budget_K6_fronts.png`; existing files
  untouched (the K6 runner's `--figure` rewrites README.md only, so
  FRONTS.md survives any replot).
* Semantics (SURF Table 1 alignment + this track's records): front =
  nondominated subset of each leg's delivered cloud (all fvals rows);
  reference = union of all legs' fronts (mutual, session-13 query-free
  semantics); IGD / max-dist reported raw AND central (reference
  restricted to all six losses <= 1.0, the K2 CENTRAL_BOUND rationale;
  method fronts never clipped); HV = Monte-Carlo dominated fraction of
  [ideal, 1]^6 (100,000 common samples, seed 20260730, PAIRED across
  legs; raw-box HV deliberately not reported — tail-volume dominated);
  CV / Gap Ratio omitted (no canonical 1-D front ordering at K=6; K=5
  bandit precedent); no 6-D front scatter (same precedent; SURF itself
  only plots M=2 fronts) — the figure is a coverage-gap CDF + HV bars.
  eps labels = each leg's final strict 64-start audit, stated
  everywhere as search LOWER bounds — no certified meter exists at
  K=6, unlike the K=2 exact 1-D meter (sections 1 and 5 above).
* Smoke self-checks (all pass): vectorised nondominated filter ==
  O(m^2) brute force on 300x6 with planted duplicates/dominated rows;
  IGD hand case; MC HV within 5 SE of two exactly-known cases.
* RESULTS (final budget B = 80,912): central union front 246 pts =
  adaptive 215 + r10 s1 35 - 4 cross-dominated; the other five legs
  contribute ZERO central points.  IGD central: adaptive 0.0420 vs
  r10 s1 0.3302 (7.9x), r10 s5 1.1208, r15 s1 1.2169, r12/r15/r20 s5
  all 3.5695.  HV central: adaptive 0.0293 +- 0.0010, r10 s1 0.0270
  +- 0.0010, all others 0.0000; union envelope 0.0398; paired delta
  (common samples) adaptive - r10 s1 = 0.0023 +- 0.0009, small but
  clear of zero.  Pattern mirrors SURF's Fishwood table: near-equal
  HV, very different coverage (86% of the central union front is the
  adaptive's own point set).  The raw-IGD ranking INVERTS (adaptive
  last, 11.98) by exactly the section-7 finding-5 mechanism at K=2:
  the raw reference is mostly above-x0 wandering and specialist
  tails; FRONTS.md carries the warning paragraph.
* x0-plateau confirmed in VALUE space: the three 7.0916-audit legs
  have IGD central identically 3.5695 = the mean distance from the
  central union front to the shared initialization x0 (verified
  independently) — their fronts' nearest point to the entire genuine
  trade-off region is the starting point itself.  The per-objective
  minima table in FRONTS.md shows the mechanism (F1 never trained
  below the 2.199 x0 level for r12/r15/r20 s5; r15/r20 s5 also F2).
* Cross-reference to section 7b's degeneracy finding: at K=6 the
  SURF-style numbers do NOT saturate at this budget (central-box HV
  tops out at 0.0398, 246 distinct central trade-off points), so
  HV/IGD are informative here — but whether the K=6 instance is
  genuinely conflicting AT OPTIMUM is not established by these
  numbers (every leg is still far from stationarity; adaptive eps
  lower bound 4.6e-2).  The instance-design question of 7b stays a
  K=2 decision.
* Incident, recorded per the honest-reporting order: on the first real
  run the built-in union sanity assert fired.  Root cause was NOT the
  dominance filter: the matmul distance trick (|R|^2 + |F|^2 - 2 R.F)
  cancels catastrophically at loss scale O(10), so rows identical to
  union members measured d ~ 1e-8..1e-7 against a 1e-9 membership
  threshold.  Threshold raised to 1e-6 (commented in code); reported
  metrics unaffected (absolute distance error ~1e-6, quoted to 4
  decimals).
* Machine-conditions disclosure: this session's post-processing ran
  DURING the pilot's tail (r=20 leg onward, through both adaptive
  legs), always single-thread at nice 19 — about 4 CPU-minutes total
  spread over ~50 min of pilot wall time.  Interference with the
  pilot's CPU-axis readings is assessed negligible (lowest scheduler
  priority, one thread, idle cores available), but section 7's "clean
  machine" claim for B20000 carries this footnote.  (The pilot
  completed normally during this session; sections 7/7b analyse it.)
* Ledger-protocol closure by this session: all five gates re-run
  after the pilot freed the machine — verify_fixes 10/10,
  prefix_repro duplicates 39 / pc all 1.0, sanity_checks_fast 8/8,
  bandit toy 9/9, bandit K5 9/9; stored-result spot checks had
  passed earlier (pure-budget, fixed-budget, tol-sweep JSONs).
* July-31 addendum (user request: drop the SURF presentation frame
  for the figure, draw a NORMAL front with F1/F2 axes): added
  `pure_budget_K6_fronts_F1F2.png` — delivered clouds + per-leg
  nondominated fronts computed IN the (F1, F2) projection (class-1/2
  cross-entropy, log-log; union front dotted; x0 star; central-bound
  dashes), same script, `n_front_f1f2_projection` added to
  front_metrics.json.  Projection front sizes: adaptive 23, r10 s1
  26, r10 s5 39, r12 s5 37, r15 s1 23, r15 s5 13, r20 s5 1 — the
  r20 s5 "front" is x0 itself (training made both plotted classes
  worse than init everywhere).  Measured corner occupancy (both
  losses < 1): adaptive 8, r10 s1 9, r10 s5 1 (edge point
  (0.94, 0.21)), all others 0; r12/r15/r20 s5's leftmost front point
  IS x0.  FRONTS.md gained the projection-semantics paragraph
  (projection front != projection of the 6-D front; hidden
  objectives unconstrained; the projection compresses the coverage
  gap — 8 vs 9 corner points where the 6-D record shows 215 vs 35).
  CDF+HV figure and all 6-D metrics unchanged.  Same-day follow-up
  (user: pairwise projections, all pairs): added
  `pure_budget_K6_fronts_pairwise.png` — all 15 objective pairs,
  lower-triangle matrix, log-log; per-panel projection fronts +
  per-panel union front + x0 star + central-bound dashes; same
  per-panel reading rule (hidden objectives unconstrained).  Visible
  there: every panel involving F1 keeps the collapsed legs pinned on
  the high-F1 side (r15/r20 s5 likewise for F2), while in panels
  among F3..F6 their specialist fronts do reach low corners (the
  snake prefixes trained those classes); the adaptive front and
  r10 s1 are the only legs spanning the trade-off range in all 15
  panels (per-objective minima <= 0.02 on every objective for both).
  Second follow-up (user: too crowded — keep only adaptive and
  r10 s1, and show Pareto-optimality more clearly): projection
  figures redrawn as two-leg ATTAINMENT views — staircase =
  per-projection attainment boundary (steps-post), shaded = the
  region dominated by that leg's delivered set (colour overlap =
  both), per-panel joint-front composition printed on the panel and
  stored per pair in front_metrics.json
  (`projection_joint_front_composition`); new CLI `--proj-legs`
  (default `adaptive_s5,baseline_r10_s1`; `all` restores the 7-leg
  matrix).  Notable readout across the 15 panels (from the JSON):
  the adaptive owns the (F1, F2) joint front 20/2 and the F1/F2-pair
  panels broadly, while r10 s1 holds more joint-front points in the
  F3..F6 panels (extreme: F5-F6 5/17; totals 123 vs 154) —
  pair-marginal depth (grid nodes live on low-dimensional simplex
  faces, breeding pair specialists) vs the adaptive's 6-D joint
  coverage (central front 215 vs 35, IGD 7.9x); the projection
  caveat cuts both ways and is stated on the figures and in
  FRONTS.md.
  Third follow-up (user: two lines only, nothing else in the
  legend): all overlays removed from both projection figures —
  no shading, no x0 star, no central-bound dashes, no per-panel
  joint-front counts; each panel is exactly the two attainment
  staircases, legend has exactly the two leg entries.  The
  joint-front composition remains recorded in front_metrics.json
  (unplotted); dead code (Patch import, x0_all, _short) removed.
