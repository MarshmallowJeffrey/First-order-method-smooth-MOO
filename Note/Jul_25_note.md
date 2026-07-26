# Jul 25 note — restore baseline trajectory lines; full r-sweep design

User request (Jul 25), three items:

1. Complete the Momentum-SVRG baseline vs Momentum-SVRG adaptive
   experiment on the without-256-checkpoints track — parameters and
   reasoning to be presented FIRST and checked together before any run
   is launched.
2. Restore the baseline's trajectory LINE in the figures (the Jul-20
   sweep figures showed the baseline only as endpoint scatter points).
3. Code edits for these two items MAY be made in place on the Jul-20
   files (user authorisation, superseding the Jul-20 "new files only"
   rule for these files), with every edit recorded here.

## 1. Edits made (in place, user-authorised)

`Original_py/baseline_svrg_certified_without_256_checkpoints.py`:

* `_serve_sweep` gained a `mark: bool = True` parameter — `mark=False`
  updates `best_val` only (metric bookkeeping) without granting
  service.  Used once, at x0, in share_mode="none".
* share_mode="none" now initialises every node's `best_val` at its x0
  score via that call (metric initialisation only; service in "none"
  mode is still granted strictly at visit).  This replicates the
  July-8 baseline's metric initialisation exactly.
* New per-checkpoint history `cov_history` = max over ALL grid nodes
  of `best_val` (the best value known so far for that node,
  x0-initialised).  This is the July-8 baseline's lag-semantics grid
  meter, restored.  It is pure cached bookkeeping (an O(N) max on a
  resident array, no oracle calls), so both cost axes are unaffected.
  Returned in the result dict; also shown in the verbose checkpoint
  line as `worst=`.

`Original_py/run_baseline_svrg_r_sweep_without_256_checkpoints.py`:

* `_plot_sweep` rewritten: each r is now drawn as a trajectory LINE
  (its `cov_history` against grads / cpu, log axes, shared
  pseudo-abscissa convention for the x=0 checkpoint) in a per-r shade
  of red, ending in a dotted vertical connector up to the endpoint
  SQUARE = the delivered set's strict full-simplex 64-start score.
  The vertical connector IS the measured between-node error of that
  grid (r=10: 0.0199 -> 0.1635, factor 8.2).  Filled square =
  certified complete; open square = fused/censored (cost lower
  bound).  Legend uses family proxies; y-label states the two meters.
* README/README_ZH templates updated to describe the restored lines.
* Dead constant `_BL_PT_KW` removed.

Semantics note (why the line and its endpoint differ on purpose): the
LINE is the method's own grid meter — weights between grid nodes never
enter it; the SQUARE is the full-simplex score on the family strict
search.  Plotting both, joined by the connector, shows the grid
certificate AND what it fails to control, in one glyph.  The fast
curve remains the v3 cheap-tier trajectory, unchanged, per the user's
standing decision.

## 2. Verification

* Smoke (r in {2,3}, --force): PASSED; delivered GN* bitwise equal to
  the Jul-20 smoke (0.447 / 0.8683) — the edits add bookkeeping only,
  the trajectory is untouched.  New `worst=` column descends as
  expected and ends above/below the strict score per the two-meter
  semantics.
* r=10 re-run launched with --force (the Jul-20 r=10 summary lacks
  `cov_history`; the run is deterministic — fresh sampler seed 41 —
  so the delivered GN* is expected to reproduce 0.1635 exactly; wall
  time will differ).  The old r10/summary.json is REPLACED; this note
  is the record of that replacement.

## 3. Experiment design presented for the user's check (item 1 —
##    NOT launched; awaiting the user's confirmation)

Sides: adaptive side = the completed v3 fast trial (Gram +
Momentum-SVRG, eps=0.001, round_fuse at 500) — reused, not re-run; it
IS the Momentum-SVRG adaptive run.  Baseline side = the r-sweep,
remaining legs r in {20, 30, 40, 50} (r=10 done/re-run above).

Parameters (all identical to the completed r=10 leg):

* node_tol=0.02, solve_target=0.005 (0.25x), share_mode="gram"
* inner solver: v3 verbatim — b=4096, epoch=ceil(n/b)=13,
  step_const=0.1, momentum=0.5, rho=0.7, patience=2, max_segments=10,
  descent safeguard x2 with <=4 retries/segment, global L_scale cap 2^60
* instance: K=6, p=20, n=50000, h=[96,96], tanh, seeds 7/8; fresh
  sampler (seed 41) per r
* fuses per r: 4 h wall + 2,000,000 grad-equivalents; checkpoint
  cadence 4500 grads; figures/READMEs regenerate after each r
* accounting, metric, and track rules unchanged (without-256)

Timing forecast (basis: r=10 measured 0.202 s/segment; sharing above
r=10 unknown): r=20 0.5-3 h; r=30 1-4 h (may fuse); r=40 1.5-4 h (may
fuse); r=50 2-4 h (likely fuses).  Total realistic 5-15 h, hard-capped
by the fuses at 16 h; sequential; each completed r lands
incrementally.  Expected figure shape: 1/r^2 descent of the squares
with a bend toward the node_tol floor around r~25-30.

## 4. Execution log

* Jul 25: engine + runner edits above; smoke PASSED; r=10 re-run
  launched (background).  r in {20,30,40,50}: PENDING the user's check
  of Sec. 3.
* Jul 25, later: that r=10 re-run was killed at ~node 808/3003 when the
  previous session process exited.  Separately, the ENTIRE sweep output
  directory `output/baseline_svrg_r_sweep_without_256_checkpoints`
  (and the _SMOKE variant) was deleted outside the session (output/
  mtime 12:41; repo-wide find shows no trace), taking the Jul-20 r=10
  results and figures with it.  Everything is deterministically
  re-derivable; r=10 relaunched from scratch — expected to reproduce
  delivered GN* = 0.1635 exactly and regenerate figures/READMEs with
  the restored trajectory lines.
* Jul 25: v3 two-tier VERIFIED from its summary.json per the user's
  request: config lambda_tier_mode="two_tier", strict tier 64 starts,
  tier_history_counts {'cheap': 500}, stop_reason=round_fuse — the
  two-tier machinery was live (all ordinary rounds cheap; stop-verify
  present but never triggered since values never approached 2eps/3).
  Adaptive side is therefore REUSED as-is, no re-run.
* Jul 25: user asked to shrink the r list (runtime concern) and to
  couple node_tol to the chosen r afterwards.  Recommendation
  presented: r in {12, 15, 20} (+ the existing 10) — no-sharing
  worst-case costs 22 min / 55 min / 3.1 h (all converge, no fuse
  ambiguity), equal-quality bracket of the fast plateau (0.058)
  expected between r=15 and r=20, large-r wall carried by the analytic
  floor verticals instead of 16 h of fused runs; optional stretch leg
  r=25 to be decided AFTER r=20's measured sharing rate.  Coupling
  preview: grid_norm(r) ~ 3.3/r from the r=10 measurement gives grid
  term ~ 0.027 at r=20, so node_tol=0.02 is what the balance rule
  picks for r_max=20 anyway.  AWAITING the user's decision.

## 5. Jul 25, final round — user decisions and launch

User confirmations (Jul 25):

* The Jul-20 sweep output folder was deleted BY THE USER (points-only
  figures were not what they wanted); new figures supersede it — no
  reconstruction of the old image needed beyond the restored lines.
* r list FIXED: {10, 12, 15, 20}.
* node_tol delegated to me.  DECIDED: node_tol = 0.02 (solve_target =
  0.005).  Reasons: (i) coupling rule — the measured grid term
  grid_norm(r) ~ 3.3/r gives (3.3/20)^2 ~ 0.027 at r_max=20, so 0.02
  places the grid->tolerance bend at the window's edge, letting the
  sweep show the 1/r^2 regime across 10-20 while r=20 just starts to
  feel the floor; (ii) it is ~3x below the fast method's delivered
  plateau 0.058, so the baseline cannot be said to lose on laxer
  per-node quality; (iii) the sharing radius ~ sqrt(node_tol)/(2G)
  stays large enough for the gram sweeps to matter at r=15-20;
  (iv) per-node cost stays ~1 segment (measured at r=10).
* Figure spec confirmed: ONE figure per axis containing ALL r
  baselines as trajectory lines + endpoint squares, plus the single
  fast adaptive curve.  This is what the Jul-25 `_plot_sweep` rewrite
  already produces.
* Adaptive side: v3 two-tier verified (Sec. 4 above) — reused, not
  re-run.

Code edits this round (both in
`run_baseline_svrg_r_sweep_without_256_checkpoints.py`):

* `SWEEP_DIRNAME` renamed to
  `baseline_svrg_multi_r_vs_fast_without_256_checkpoints` — the
  folder name now states the content (multiple-r baselines vs fast),
  per the user's item 6.
* `--r-list` default changed from "10,20,30,40,50" to "10,12,15,20"
  (user's item 2).

## 6. Jul 25, evening — cross-meter misread and the comparable-meter fix

Incident: with r=10 and r=12 on the figure, the user read the
baseline's GRID-meter line (which ends just under node_tol = 0.02) as
the baseline BEATING the fast curve (plateau 0.058) and concluded the
result was wrong ("I need adaptive to win").  The DATA in fact says
adaptive wins decisively on the honest, same-meter reading:

* to quality 0.1635 (r=10's delivery): fast ~2.4k grad-equivalents vs
  baseline 41.3k -> 17x;
* to quality 0.0954 (r=12's delivery): fast ~3.6k vs 64.4k -> 18x;
* to quality 0.058 (fast's delivery): NO baseline point reaches it yet;
  r=20 projected ~0.035-0.05 at ~0.5M grads -> ~50-80x.

The misread is the figure's fault, not the data's: the grid-meter line
(3003 nodes' own scores) and the full-simplex curves shared one y-axis,
and any reader — a reviewer included — would make the same mistake.

Fix (visualisation only; NO measurement changed): the baseline's
plotted line is now the strict full-simplex 64-start GN* of the
delivered-set PREFIX at each checkpoint, computed post-hoc on the
cached Grams (warm-started along the prefix chain; cost recorded in
metric_seconds, excluded from both cost axes like all measurement on
this track).  Line and endpoint square are now the same meter as the
fast curve's y-axis family; the line ends exactly at the square.  The
grid meter stays in the summaries (`cov_history`) and in the README
table (new "grid cert end" column); "grid cert end" vs "delivered GN*"
is the between-node gap, now disclosed as numbers instead of a
confusing glyph.  Where two strict searches score the same final set,
the LARGER value is kept (both lower-bound a maximum; never understate
the criterion).

Edits: engine — `delivered_gn_strict_history` computed at metric time,
final square = max of the two final-set searches; runner — line
switched to the new history, gap connector and grid-meter line removed
from the main figure, legend/y-label/README templates updated
accordingly.  Smoke re-passed, delivered GN* values bitwise unchanged
(0.447 / 0.8683).  The in-flight sweep (old code, r=15 mid-leg) was
killed and ALL four legs relaunched detached with --force so every
summary carries the comparable-meter history; r=10/12 are
deterministic reproductions of the numbers above.

Launch: two background runs were killed mid-r=10 by session exits
(node 808, then node 1404 — the harness kills its children when the
app closes).  The sweep is therefore now launched DETACHED
(`nohup caffeinate -i ... &`, PID recorded in the output folder's
`sweep_run.pid`, log in `sweep_run.log`): it survives session/app
exit.  Completed r legs persist their summaries individually and the
runner skips existing summaries on relaunch, so any interruption
loses at most the in-flight leg.  Expected: r=10 ~11 min, r=12
<=22 min, r=15 <=55 min, r=20 <=3.1 h (zero-sharing upper bounds);
figures + READMEs regenerate after every completed leg.

## 7. Jul 25, late — between-node-gap showcase figures (user redirect)

User: set the r-sweep aside for now; using r=10, produce the two
figures designed in the "prove the missed λ fails" discussion.  The
user first proposed a one-off 256-point check for the worst missed λ;
resolved to the FAMILY strict 64-start search instead (their own
without-256 rule bans the external 256-start yardstick; the family
searcher is the same instrument that signs the adaptive certificates,
and under-search can only UNDER-state the gap — conservative
direction).  A one-off 256-START run of the family searcher was
offered as an optional robustness audit, pending their say-so.

Actions:

* Sweep process killed mid-r=15 (r=10/r=12 summaries persist; r=15/20
  legs to be resumed later on the user's word — runner skips completed
  legs).
* Engine: new ``return_grams`` flag returns the delivered Gram stack,
  every node's final certified value (best_val) and the grid.
  Runner: new ``--save-grams`` flag dumps these to
  ``r{r}/delivery_audit.npz`` (a few MB) and strips them from the JSON.
* New post-processing file
  ``plot_between_node_gap_without_256_checkpoints.py``: exact Gram
  arithmetic only (no oracle calls, outside both cost axes), produces
  (1) ``between_node_gap_path_r10.png`` — g(λ)=min_i λᵀM_iλ along
  nearest-node → witness λ* → 2nd-nearest node, dips certified ≤
  node_tol at the nodes, peak = delivered GN* at λ*;
  (2) ``between_node_gap_nodes_r10.png`` — all 3003 certified node
  values sorted below the node_tol line vs the witness level above it.
* r=10 relaunched detached with ``--force --save-grams`` (deterministic
  reproduction, ~12 min) to materialise the audit npz.

## 8. Jul 25, close — `--replot` flag

Figures are normally rewritten only after a leg COMPLETES, so an
invocation whose r values are all cached (e.g. re-drawing r=10 together
with r=12 after they were produced by separate invocations) redrew
nothing — the on-disk sweep figures showed only r=10.  Added
`--replot` to `run_baseline_svrg_r_sweep_without_256_checkpoints.py`:
loads the stored per-r summaries, regenerates figures + READMEs, exits
without running anything (`--force` is ignored while replotting, so the
cached summaries are still loaded).  Used to produce the current
two-leg figures.  No measurement is affected — plotting only.

## 9. Jul 25/26 — sweep COMPLETE (r = 10, 12, 15, 20) and the two-knob floor

User gave the go; r=15 and r=20 ran detached with --save-grams.  All
four legs: stop_reason=completed, censored=0, certificates verified
(worst served value <= node_tol on every leg).

| r | N | delivered GN* | grad-equiv | wall s | solved | share | delivered |
|---|---|---|---|---|---|---|---|
| 10 | 3,003 | 0.16352 | 41,327 | 598.6 | 2,879 | 4.1% | 3,155 |
| 12 | 6,188 | 0.09542 | 64,428 | 986.3 | 4,441 | 28.2% | 4,768 |
| 15 | 15,504 | **0.05946** | 80,912 | 1,172.5 | 4,758 | 69.3% | 5,667 |
| 20 | 53,130 | 0.06342 | 241,721 | 3,689.0 | 11,820 | 77.8% | 16,487 |

Fast adaptive (v3) final: 0.05811 at 9,226 grad-equivalents / 292.6 s.

### PREDICTION FAILED, and the failure is the finding

I forecast r=20 would land at 0.03-0.045 and BRACKET the fast method's
0.0581 from below, giving an interpolated headline ratio.  It did not:
r=20 came in at 0.06342, i.e. slightly WORSE than r=15 while costing
2.99x more.  **No baseline leg ever went below 0.05946**, so the
bracket does not exist and the equal-quality number must be quoted
from the r=10/12/15 legs only.

Mechanism (structural, not noise): a node stops being worked as soon as
it certifies at <= node_tol, so the grid-max pins to the tolerance on
EVERY leg (measured: 0.019911 / 0.020000 / 0.019997 / 0.019999).  The
full-simplex value behaves like (sqrt(node_tol) + c/r)^2: the grid term
dominates for r=10..15 (0.1635 -> 0.0954 -> 0.0595) and by r=20 it has
shrunk below the tolerance term, so refinement buys nothing while cost
keeps growing as C(r+5,5).  The 6.7% gap between r=15 and r=20 sits
inside the heuristic lambda-search's own noise band (the v3 curve's own
sawtooth spans 0.06-0.12), so the honest reading is "r=15 and r=20 are
the same level", i.e. **the curve has bent onto the node_tol floor at
about 3x node_tol** — the bend I predicted for r~25-30 when choosing
node_tol=0.02 (Sec. 5), arriving earlier than expected.

Consequence worth writing up: Algorithm 1 has **two knobs that must be
tightened together with multiplicative cost** — refining r alone stalls
at ~3x node_tol while cost explodes combinatorially; lowering node_tol
alone costs 3-4x per node while the grid term is untouched.  The
adaptive method has no such coupling (its lambda is searched, not
gridded).  Over four configurations, 241k grad-equivalents and ~1 h of
compute, the baseline's best delivered quality was 0.05946; the fast
method reached 0.05811 in 9,226 grad-equivalents / 293 s.

### Equal-quality cost ratios (read horizontally)

| level | baseline | fast | grads | CPU |
|---|---|---|---|---|
| 0.16352 | 41,327 g / 598.6 s | 2,409 g / 68.1 s | 17.2x | 8.8x |
| 0.09542 | 64,428 g / 986.3 s | 4,813 g / 140.3 s | 13.4x | 7.0x |
| 0.05946 | 80,912 g / 1,172.5 s | 5,413 g / 158.6 s | 14.9x | 7.4x |
| 0.06342 | 241,721 g / 3,689.0 s | 5,413 g / 158.6 s | 44.7x | 23.3x |

**Quote 13-17x (gradients) / 7-9x (CPU), stable across the three
r=10/12/15 legs.**  Do NOT headline the r=20 row: its 44.7x is large
because that leg wasted budget on a floored configuration, not because
the adaptive method got better.  CPU rows are cross-run (v3 measured
Jul 16, baseline Jul 25) — state that when quoting them.

### Other read-outs

* Sharing rate climbs 4.1% -> 28.2% -> 69.3% -> 77.8% as spacing 2/r
  falls; this is what kept r=15/20 feasible at all (r=15 has 5.2x the
  nodes of r=10 but only 1.65x the solves).
* r=20 is the ONLY leg where the descent safeguard fired: L_scale
  1.0 -> 4.0 (two doublings).  Certificates are unaffected (all
  acceptance on full gradients), but it records that some lambda
  directions on the finer grid have curvature above the L estimate.
* Also this session: the wrong "~16x CPU" figure in Jul_20_note Sec. 7
  was corrected in place to 8.8x (see the bracketed note there).

### Offered, not yet decided by the user

A dedicated "two-knob floor" figure: delivered GN* vs r, with the
node_tol floor line and a 1/r^2 reference slope, making the bend
explicit.  Data is complete; the figure is seconds of work.  AWAITING
the user's word.
