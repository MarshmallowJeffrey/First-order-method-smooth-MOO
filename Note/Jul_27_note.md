# Jul 27 note — K=2 bandit toy, eps1e-2 rung: presentation + measurement revision

Session date: 2026-07-27 (session 12; follows `Jul_26_note.md`).
Scope (user-directed): the figures and the checkpoint resolution of ONE rung,
`Adaptive Bundle Algorithm/output/bandit_toy_surf_without_256_checkpoints/eps1e-2/`.
No objective or solver logic was changed.  The eps1e-3 / eps1e-4 folders were
NOT re-run and still hold the Jul-26 figure style; if they are re-run later
with the current driver they will pick up the new style automatically.

## 1. The six user-ordered items (all done)

1. **Equal-budget marker removed** from fig1/fig2 (vertical line + legend
   entry).  The matched-budget numbers still live in
   `summary.json -> readouts.fixed_budget`.
2. **Finer checkpoints** to fix the fig1 "early adaptive lead" interpolation
   artifact: run with `--eval-every 0` (cadence None = record at every
   processed node / round) PLUS new per-segment recording (see §2).
   Checkpoints: baseline 4 -> 21, adaptive 3 -> 8.
3. **fig1 stop-verification tails removed**: each CPU curve now ends at its
   last GN improvement; the trailing flat segment (stop-verification cost
   only, no descent) is not displayed.  The x-label says so; full CPU totals
   remain in `summary.json`.
4. **Why the adaptive stops earlier than the baseline** — organized write-up
   in §5.
5. **fig3 redesigned as "nondominated discovered fronts"** with two
   recomputed, query-free metrics (§4).
6. **New figure `fig4b_value_gap_profile_matched.png`** — same folder, only
   the two matched-budget (solid) value-gap profiles from fig4.

## 2. Code changes (3 files, all in `Original_py/`)

- `run_bandit_toy_without_256_checkpoints.py`
  - new CLI flag `--eval-every` (default 10.0; `<= 0` maps to `None` =
    finest cadence), stored in `cfg["eval_every_n_grads"]`;
  - passes `record_segment_checkpoints=True` to BOTH methods;
  - `fig_gn_curves`: equal-budget marker removed; CPU variant truncates each
    curve at its last GN improvement and the x-label discloses it;
  - fig3: new helpers `nondominated_2d` (staircase scan, weak dominance and
    duplicates dropped) and `pf_front_metrics` (see §4); the old
    `_solution_map_front`-based 12-query figure code removed; annotation now
    shows the query-free metrics; `summary.json` gains
    `pf_front_nondominated`, `raw_histories.npz` gains
    `bl_front_nd` / `ad_front_nd`;
  - new `fig_value_profile_matched` (fig4b);
  - the OLD query-based `pf_metrics` block is still computed and stored
    unchanged (cross-rung comparability with eps1e-3 / eps1e-4).
- `algorithm_fast_without_256_checkpoints.py`
  - `_bundle_update_msvrg(..., on_full_eval=None)`: optional hook invoked
    after EVERY segment-end full evaluation joins the bundle;
  - `algorithm_adaptive_fast(..., record_segment_checkpoints=False)`: when
    True, a cadence-gated checkpoint is recorded from that hook.
  - RECORDING ONLY: default off = bit-identical histories; no solver
    decision reads the records (grad_equiv is recomputed from scratch after
    the inner loop, so the hook's counter refresh cannot drift).
  - Regression gate after the edit: `sanity_checks_fast.py` ALL PASS (8/8),
    MSVRG degeneration still bitwise.
- `baseline_svrg_certified_without_256_checkpoints.py`
  - `record_segment_checkpoints=False` kwarg; per-segment checkpoint under
    the same cadence, inserted right after the delivery + serve-sweep
    bookkeeping.  The node-level checkpoint block — and with it the
    equal-level stop-check cadence — is UNCHANGED.  Default off =
    bit-identical histories (MLP track unaffected).

## 3. Re-run of the rung + invariants

Command (from `Original_py/`):

    KMP_DUPLICATE_LIB_OK=TRUE python run_bandit_toy_without_256_checkpoints.py \
        --epsilon 1e-2 --eval-every 0

Unchanged trajectory invariants (same seeds 7/41, deterministic):
totals 85.2 / 30.48 grad-equivalents; final GN* 3.358e-3 / 6.649e-3;
stop reasons `global_stop_gn` / `epsilon_certified`; eps_value final
0.4758 / 0.3088; adaptive bundle 6 -> 1 after delivery pruning; the old
query-based `pf_metrics` numbers reproduce exactly (0.9414 / 0.7415 max,
0.3379 / 0.4798 IGD).

Changed by the finer measurement (same trajectories, more sample points):

| readout | old (cadence 10, node-level) | new (per segment) |
|---|---|---|
| checkpoints bl / ad | 4 / 3 | 21 / 8 |
| grads-to-eps bl / ad | 85.2 / 30.48 | **48.62 / 12.19** |
| cpu-to-eps bl / ad (s) | 0.059 / 0.076 | 0.036 / 0.086 (wall jitter) |
| GN at matched 30.48 grads bl / ad | 0.01366 (interp.) / 0.00665 | **0.01403 (measured) / 0.00665** |

The old first-crossing readouts were coarse-cadence artifacts (first
checkpoint at-or-after the true crossing); `Jul_26_note.md`'s eps1e-2
"grads-to-eps 85 / 30" line is superseded by 48.62 / 12.19.  The fig4/fig4b
"baseline @ matched budget" profile is now a real 8-point prefix (x0 + 7
segment points; max 0.4758 at w=1, min 0.056 at w=0.28) instead of the
x0-only prefix the coarse recording produced.

Structural fact exposed by the re-run: only **2 of 12 nodes** were
chain-processed (17 segments: 9 + 8); ALL 12 node certificates were signed
by the gram-share sweep (`served_by_share = 12`, `served_by_chain = 0`).
Node-level recording therefore could never produce more than 2 interior
baseline checkpoints — that ceiling, not the cadence number, was the root
cause of the old interpolation artifact.  The equal-level stop check ran at
BOTH processed nodes and passed only at the second, so the stop still
coincides with completion at per-node granularity — a stronger version of
the Jul-26 statement.

## 4. fig3: nondominated discovered fronts (query-free)

Sets: per method, the nondominated subset (2-D minimisation) of ALL its
evaluated points — baseline: all 18 delivered points; adaptive: the 6-point
pre-prune bundle.  No query weights, no delivery/pruning policy involved:
this compares the methods' ability to DISCOVER the front.

Metrics (against the 5000-sample dense closed-form front), shown under the
figure and stored in `summary.json -> pf_front_nondominated`:

- `max_front_to_oracle` — max over the method's front points of the
  distance to the oracle front: the worst off-front error of a point the
  method presents.
- `igd` — mean over oracle-front samples of the distance to the nearest
  front point: how much of the true front the discovered front covers.

Values at eps1e-2: baseline front 11 of 18 — max dist 8.777e-2,
IGD 3.379e-1; adaptive front 6 of 6 — max dist 8.180e-2, IGD 3.615e-1.
Reading: near-tie on off-front error (both discovered fronts hug the true
front within ~0.09); the baseline covers slightly more arc (11 points vs 6,
lower IGD); both miss both ENDS of the front at this loose rung.  Note the
semantics differ from the old query-based numbers: the old
`max_point_to_oracle` (0.94 / 0.74) measured worst QUERY answers under the
solution map on DELIVERED sets; the new max (0.088 / 0.082) measures
off-front error of the discovered fronts.  Do not mix the two when quoting.

## 5. Why the adaptive stops at 30 grads while the baseline runs to 85 (figs 1-2)

Both runs are independent and stop against the SAME line: strict
full-simplex GN* <= 2eps/3 (adaptive: its certificate check every round,
`epsilon_certified`; baseline: the equal-level stop `global_stop_gn`,
checked at checkpoint cadence).  The different stopping TIMES come from how
each method schedules its work relative to the global meter:

1. **The adaptive schedules against the global worst case.**  Each round it
   searches for the current worst λ over the whole simplex and runs the
   inner solver exactly there, so the plotted meter is the quantity it
   attacks; when the strict search signs GN* <= 2eps/3 it terminates
   immediately (here after 1 round / 5 segments, 30.48 grads, at 6.649e-3
   vs the line 6.667e-3 — no overshoot beyond one segment).
2. **The baseline schedules by grid nodes, not by the global meter.**  It
   walks the snake order serving nodes; the global meter is dominated by
   whichever region is still unserved.  On this instance the worst simplex
   region is the one its chain fixes LAST (now verified at per-node check
   granularity: the stop check ran at both processed nodes and passed only
   at the second).  At 36 grads its global GN* was still 1.37e-2 — twice
   the line — so it HAD to keep running.
3. **The overshoot below the line is by design and by quantum.**  Each
   node's inner target is solve_target = node_tol/4 = eps/6, four times
   stricter than the stop line (margin for between-node lambdas), and
   progress arrives in segment-sized quanta with checks at checkpoint
   cadence — so when the last region is fixed, the meter lands at 3.358e-3,
   one quantum past the line, not on it.
4. **Anytime-vs-terminal corollary, now measured (not interpolated):** on
   the grads axis the adaptive sits below the baseline at every shared
   budget until its stop (to-eps 12.19 vs 48.62 = 4.0x; at the 30.48-grad
   matched budget 0.00665 vs 0.01403 = 2.1x); on the CPU axis the baseline
   leads from ~2 ms on because the oracle is nearly free while the
   adaptive's CPU is 99% λ-search (0.244 of 0.247 s).  fig5 shows the value
   counterpart: the baseline's worst-query value gap stays pinned at 0.4758
   across all 21 checkpoints while the adaptive cuts it to 0.3088.

## 6a. Addendum (same day): fig1 display reverted to milestone cadence

After viewing the dense fig1 the user asked for the previous look.
Decision: the dense per-segment DATA stays (npz, fig2, fig4b, all §3
readouts), but fig1 now displays only the node/round MILESTONE
checkpoints — recovered as the duplicate records the node/round-end
blocks write right after a segment record (same grads, same count);
fallback = all points when no duplicates exist (numeric cadences).
Items 1 and 3 (no equal-budget marker, verification tails truncated)
remain in force; the x-label discloses the milestone display.  fig1/fig2
were re-rendered from the stored arrays (no method re-run; fig2 keeps
the dense style).  Rationale: at per-segment density the CPU picture
collapses into vertical bursts (segments cost ~0.4 ms each) separated
by overhead plateaus (the 32 ms stop screen, the 86 ms strict λ-search)
— honest but unreadable; the milestone view keeps fig1 readable while
the dense evidence lives in fig2 and raw_histories.npz.

## 6b. Addendum (same day): fig4b legend

Per user request the fig4b legend labels are now the method names —
"Momentum-SVRG baseline" / "Momentum-SVRG adaptive bundle method" — and
the matched-budget information moved into the title ("at matched budget
30 grads").  Re-rendered from the stored arrays; no data change.

## 6. Bookkeeping

- `output/bandit_toy_surf_without_256_checkpoints/README.md` gained a
  "Jul 27 revision" paragraph (eps1e-2 only; other rungs keep the old
  style until re-run).
- The LEDGER's K=2 table row for eps1e-2 (to-eps readouts) is superseded as
  per §3; to be corrected at the next ledger rewrite.
- Old eps1e-2 artifacts (Jul-26 style) remain recoverable from commit
  `df75b55`.

## 7. Same day, later session: eps1e-3 rung brought to the Jul-27 style + fig4c

Scope (user-directed): apply items 1/2/3/5 of §1 to
`output/bandit_toy_surf_without_256_checkpoints/eps1e-3/` (marker
removal, fig1 tail truncation, denser fig2 checkpoints, discovered-front
fig3 with recomputed metrics) and add ONE new figure — fig4's baseline
solid + adaptive dashed curves isolated.  Items 1/2/4-of-§1's-list were
already in the driver from the eps1e-2 pass; the only code change this
session is the new figure (below).  No solver logic touched.

Command (from `Original_py/`; the venv interpreter, NOT the default
`python` — the miniconda 3.9 default's cyipopt fails to dlopen and the
adaptive REQUIRES IPOPT):

    KMP_DUPLICATE_LIB_OK=TRUE /Users/shirch/vscode101/.venv/bin/python \
        run_bandit_toy_without_256_checkpoints.py --epsilon 1e-3 --eval-every 0

Code change (driver only): `fig_value_profile_owndelivery` in
`run_bandit_toy_without_256_checkpoints.py` ->
`fig4c_value_gap_profile_owndelivery.png` — baseline FULL-delivery
profile + adaptive FULL-delivery profile, both solid (the adaptive was
initially dashed to mirror fig4's line style and changed to solid on a
follow-up user request, re-rendered from the stored arrays), per-curve
grad totals in the legend.  These are exactly fig4's "red solid" and
"blue dashed" curves (on this rung the matched budget IS the baseline's
own endpoint, so fig4's solid-red and dashed-red coincide); the
own-delivery framing keeps the figure meaningful on rungs where the
adaptive is the shorter run.  All rungs gain fig4c on their next re-run.

Unchanged trajectory invariants (seeds 7/41): totals 328.704 / 481.584
grad-equivalents; joint calls 64 / 79; stop reasons `global_stop_gn` /
`epsilon_certified`; baseline 65 delivered, 4 solved nodes,
`served_by_share = 12`; adaptive bundle 80 -> 3 after pruning; final
GN* 5.1178e-4 / 4.9532e-4; eps_value final 0.47578 / 0.03505; the old
query-based `pf_metrics` reproduce bitwise (0.94139 / 0.66444 max,
0.32154 / 0.12034 IGD).

Changed by the finer measurement (same trajectories, more sample
points; supersedes the Jul-26 eps1e-3 readouts):

| readout | old (cadence 10, node-level) | new (per segment) |
|---|---|---|
| checkpoints bl / ad | 6 / 8 | 70 / 87 |
| grads-to-eps bl / ad | 328.704 / 408.432 | **304.32 / 359.664** |
| matched-budget prefix m bl / ad | 65 / 44 | 65 / 54 |
| GN at matched 328.704 grads bl / ad | 5.118e-4 / 1.569e-3 | 5.118e-4 / **1.158e-3** |
| eps_value at matched budget bl / ad | 0.47578 / 0.06887 | 0.47578 / **0.06494** |

fig3 discovered-front metrics at this rung (stored in
`summary.json -> pf_front_nondominated`): baseline front 37 of 65 —
max front-to-oracle 9.085e-2, IGD 3.216e-1; adaptive front 62 of 80 —
max 9.116e-2, IGD **5.611e-2**.  Reading: off-front error is a
near-tie, but the adaptive's discovered front covers ~5.7x more of the
true front (IGD); the baseline's front is confined to the right corner
F1 in [-0.486, -0.18] — the saturation-plateau geometry documented in
the eps1e-3 analysis (its chain never reached the F1-good region).

Caveat recorded for the future (corrects the blanket wording of §2):
the baseline's per-segment checkpoint and the node-level block share
the `grad_at_ckpt` cadence tracker, and the node-level block gates the
equal-level STOP CHECK.  With a NUMERIC cadence plus
`record_segment_checkpoints=True`, a segment record landing < cadence
grads before a node end defers that node's stop check — potentially a
trajectory change.  At cadence None (`--eval-every 0`, used for ALL
Jul-27 re-runs) the node-end check always fires, so recording-only
holds there; recording-off runs (all committed history) are likewise
unaffected.  Give the segment records their own tracker before ever
combining a numeric cadence with segment recording.

Bookkeeping: README revision paragraph now says "eps1e-2 and eps1e-3"
and lists fig4c + the eps1e-3 reproduce line; the LEDGER's K=2 eps1e-3
to-eps readouts are superseded per the table above (correct at the next
ledger rewrite); old eps1e-3 artifacts (Jul-26 style) remain
recoverable from commit `df75b55`.

Addendum (same session): after viewing the dense fig2 (70 / 87 markers)
the user asked for the previous look with a few added points.  fig2 now
DISPLAYS the milestone checkpoints plus segment records spaced ~1/10 of
the x-span (12 / 14 markers on this rung); duplicate-x records collapse
to one marker, endpoints always shown, and the x-label discloses the
thinning.  Display only — the dense arrays stay untouched in
`raw_histories.npz`; fig1/fig2 were re-rendered from the stored arrays
(no method re-run, §6a precedent).  eps1e-2's fig2 keeps the dense
style until its next re-run or re-render.

## 8. Same day, third session: eps1e-4 rung re-run — the denser meter EXPOSES A FALSE ADAPTIVE CERTIFICATE at this rung

Scope (user-directed): bring `eps1e-4/` to the Jul-27 style — §1's
items 1 (no equal-budget marker), 3 (fig1 verification-tail
truncation), the fig2 milestone-plus-spaced display, and the
discovered-front fig3 with recomputed metrics.  The user also asked
for eps1e-3's fig4 companion (red solid + blue dashed isolated, blue
made solid) — that is exactly §7's `fig4c`, already on disk since
01:36; no action needed.  NO code changes this pass: the §1/§6a/§7
driver applied verbatim.  eps1e-4 additionally gains fig4b + fig4c.

Command (from `Original_py/`; venv interpreter, cyipopt constraint as
in §7):

    KMP_DUPLICATE_LIB_OK=TRUE /Users/shirch/vscode101/.venv/bin/python \
        run_bandit_toy_without_256_checkpoints.py --epsilon 1e-4 --eval-every 0

Unchanged trajectory invariants (seeds 7/41, bit-identical to Jul-26):
totals 1215.664 / 1383.792 grad-equivalents; joint calls 231 / 227;
stop reasons `global_stop_gn` / `epsilon_certified`; baseline 232
delivered, 12/12 `served_by_share`, `delivered_gn_strict`
2.828917887e-5 bitwise; adaptive bundle 228 -> 5 after pruning; the
old query-based `pf_metrics` reproduce bitwise (0.941385 / 0.647397
max, 0.317485 / 0.105155 IGD).

Changed by the finer measurement (same trajectories, denser strict
scoring — supersedes the Jul-26 eps1e-4 readouts AND §5-adjacent
statements made about this rung in conversation):

| readout | old (cadence 10, node/round-level) | new (per segment) |
|---|---|---|
| checkpoints bl / ad | 11 / 14 | 242 / 241 |
| grads-to-eps bl | 1215.664 | **1118.128** |
| grads-to-eps ad | 987.552 | **none — never crosses eps (finding below)** |
| ad final_gn_strict (min over history) | 5.294e-5 | 1.0709e-4 |
| GN at matched 1215.664 grads bl / ad | 2.829e-5 / 8.373e-5 | 2.829e-5 / 1.0709e-4 |

### The finding

The adaptive's strict certificate at THIS rung is FALSE — a
search-limited miss, now proven by an exact computation:

1. **Mechanism.**  Both the internal certificate and the post-hoc
   scorer maximise GN(lambda) with the 64-start warm-chained search;
   a search can only UNDER-report a maximum, so higher readings are
   authoritative.  The Jul-26 sparse 14-prefix chain read the final
   228-point family at 5.294e-5 (fresh-search cross-check 6.677e-5);
   the new 241-prefix chain reads it at 1.280e-4.  Spot-checks at
   identical prefixes (m = 6 / 89 / 153) reproduce the old values
   BITWISE; the divergence starts at m ~= 163, i.e. exactly when the
   family is tuned to margin ~0 and the violating pockets narrow.
2. **Exact verification (no search).**  K = 2 makes lambda 1-D, so
   the strict meter is computable EXACTLY on a dense grid: rebuild the
   Gram rows M_i = J_i J_i^T from the stored points via
   `make_bandit_toy(T=1000, noise_std=0.5, data_seed=7, A=5, tau=0.05,
   alpha=4.0).joint_oracle` (exact jacobians, deterministic), then
   GN(w) = min_i lam^T M_i lam on a 200001-point w grid.  Results:
   - adaptive PRE-PRUNE (228 pts): true max **1.279968e-4 at
     w = 0.09430** — the new scorer's endpoint reading equals the true
     max to 6 digits;
   - adaptive DELIVERED (5 pts): identical 1.279968e-4 at the same w
     (pruning irrelevant — the pocket was never covered);
   - the eps-violating pocket: w in [0.0796, 0.1132], 3.36% of the
     simplex; GN* there exceeds eps = 1e-4, peak 1.28e-4 = 1.92x the
     2eps/3 certificate line;
   - baseline (232 pts): true max **2.828918e-5 at w = 1.0** — equals
     its stored `delivered_gn_strict` bitwise (its meter was and is
     exact-search-robust).
3. **Cross-rung check (same exact recipe).**  eps1e-2 adaptive:
   6.6491e-3 <= 2eps/3 = 6.6667e-3 (pass; recorded final = true max).
   eps1e-3 adaptive: 4.9531e-4 <= 6.667e-4 (pass; ditto).  The false
   certificate is SPECIFIC to the tightest rung — narrow pockets +
   margin-tuned family is exactly the regime a 64-start multistart
   misses.
4. **What must no longer be quoted.**  At eps1e-4: "epsilon_certified"
   does NOT mean eps was reached; the adaptive never certifiably
   crosses eps on the strict meter (fig2: blue plateaus at ~1.07e-4,
   above the dashed line).  The Jul-26 readouts "ad grads-to-eps
   987.552" and "ad final 5.294e-5" are superseded as measurement
   optimism; the "adaptive reaches eps ~19% cheaper" reading for this
   rung is WITHDRAWN.  On this rung only the baseline certifiably
   reaches eps (1118.128 grads).  The eps1e-2 / eps1e-3 conclusions
   stand.
5. **What is NOT affected.**  Value-gap results (closed-form oracle,
   no lambda search): eps_value final 0.02684 vs 0.47578 — the
   adaptive remains 17.7x better; the discovered-front metrics (fig3,
   below); the CPU-axis picture; both other rungs.

fig3 discovered-front metrics at this rung (stored in
`summary.json -> pf_front_nondominated`): baseline front 137 of 232 —
max front-to-oracle 9.085e-2, IGD 3.176e-1; adaptive front 182 of
228 — max 9.106e-2, IGD **3.557e-2**.  Same pattern as eps1e-3:
off-front error a near-tie, adaptive covers ~8.9x more of the true
front; the baseline's front stays confined to the right corner.

Display notes: fig2 shows 13 / 15 markers (milestones + ~1/10-span
spacing per §7's addendum); fig1 milestone-only + last-improvement
truncation per §6a; the plotted strict curves are lower-bound
envelopes (per-prefix search), so small non-monotone wiggles on the
blue plateau are search variation, not descent — the dense-grid value
above is the validated endpoint.

Open item (decision owner: user — solver behaviour, not touched this
session): harden the adaptive's strict certificate for K = 2 — the
dense-grid exact check costs ~1 s per certification on this problem
(200001-point grid over 228 Gram rows) and would make the certificate
exact rather than search-limited; alternatives are more/denser lambda
starts.  Either way the eps1e-4 rung would need one re-run (~4 min
wall) and the adaptive would stop LATER (honestly) at this rung.

Bookkeeping: README revision heading now says "all three rungs", the
reproduce block gains the 1e-4 line, and a dedicated "eps1e-4 caveat"
section flags the false certificate; the LEDGER's K=2 eps1e-4 to-eps
row is superseded per the table + finding above (correct at the next
ledger rewrite: bl 1118.128, ad none-pending-certificate-fix); old
eps1e-4 artifacts (Jul-26 style) remain recoverable from commit
`df75b55`.

Addendum (same session): fig4c's legend renamed to plain method names
— "momentum-SVRG baseline" / "momentum-SVRG adaptive bundle method"
(user request; the per-curve grad totals stay in `summary.json`).
Driver label change in `fig_value_profile_owndelivery` + eps1e-4's
fig4c re-rendered from the stored arrays (no method re-run, §6a
precedent); eps1e-3's fig4c re-rendered the same way in a follow-up
session, so both fig4c-bearing rungs now carry the renamed legend.

---

## Jul 27, MLP (session 12 continued): PURE fixed-budget protocol — design + results

Separate strand from the bandit sections above (which are the parallel
session's record).  User-designed revision of the fixed-budget
comparison: **no tolerance input anywhere** — the earlier
`fixed_budget_B80912/` experiment (tolerance-family points + an
adaptive run that still carried epsilon/rel_target) remains archived as
its own protocol; this one replaces tolerances with structure only.

### Protocol (user-approved; the user also fixed s and the sensitivity legs)

- Work unit: one SEGMENT = 13 minibatch Momentum-SVRG steps (trigger
  early-stop REMOVED — it compares to a target) + 1 full joint
  evaluation (delivered + charged).  Budget B = 80,912 grad-equivalents
  for EVERY leg; gate "spent + 18.78 <= B"; stop = budget, nothing else.
- Shared execution loop, chain warm start for BOTH methods (the T-map
  anchor needs per-point Jacobians — ~2.5 GB at these bundle sizes —
  so the pure protocol equalises the warm-start policy instead;
  disclosed).  Shared s = segments per allocation decision.
- The ONLY difference is the next-lambda policy: snake grid order
  (baseline, r in {10,12,15,20}) vs strict worst-lambda search
  (adaptive, 24 targeting starts, decision time ON its CPU axis).
- Measurement: strict 64-start prefix audits at ~2000-grad checkpoints
  for ALL legs (baseline lines added on user request, backfilled from
  stored Gram stacks), monotone lower-bound envelope, never-understate
  merge with each leg's delivery search.  Audits off both axes.
- Driver: `run_pure_budget_K6_without_256_checkpoints.py` (new file;
  --run baseline/adaptive, --figure, --backfill-audits, --smoke;
  legs skip themselves if their summary exists).  Engines untouched.
- Boundary discount (standing accounting rule) predicts the baseline
  fits MORE segments per budget (grid nodes are boundary-heavy);
  registered in advance, observed: 5,431-5,904 vs adaptive 4,308.
- The user registered the expected outcome in advance (adaptive should
  win) and asked for a symmetric implementation check if it lost.

### Results (output: v2 home, `pure_budget_B80912/`)

| leg | segments | node coverage | final strict audit |
|-----|----------|---------------|--------------------|
| adaptive s=5 | 4,308 | 862 distinct lambdas | **4.616e-2** |
| baseline r10 s=1 (sens.) | 5,557 | 3,003/3,003 (1.85 passes) | 1.114e-1 |
| baseline r10 s=5 | 5,897 | 1,180/3,003 (39%) | 3.178 |
| baseline r15 s=1 (sens.) | 5,431 | 5,431/15,504 (35%) | 4.904 |
| baseline r12 s=5 | 5,821 | 1,165/6,188 (19%) | 7.092 (= x0 plateau) |
| baseline r15 s=5 | 5,846 | 1,170/15,504 (7.5%) | 7.092 |
| baseline r20 s=5 | 5,904 | 1,181/53,130 (2.2%) | 7.092 |

All legs clean; descent safeguard fired 1-2 times in four baseline
legs (L_scale end 2-4; first firings of the whole MLP track — fixed
13-step segments without the trigger overshoot occasionally at some
boundary lambdas; negligible counts, recorded).  r10_s1's final audit
rose 0.1039 -> 0.1114 in the never-understate merge (the prefix-audit
chain found a higher peak than the single delivery search).

**VERDICT — the user's registered prediction CONFIRMED on the
protocol's primary (grads) axis, with margin:**

1. Adaptive final 0.0462 vs best baseline (r10 s=1) 0.1114 = **2.4x
   better at equal budget**; the margin dwarfs the ~13% audit-noise
   scale observed this session.  Adaptive crosses the best baseline's
   FINAL value at 28,132 grads = **35% of the budget**.
2. Adaptive 0.0462 is the best value ANY configuration produced at
   this budget all session — it beats the tolerance-machinery knee
   (r15@0.02, 0.0595 with gram share + certificates + deep targets) at
   70,329 grads with no tolerance machinery at all.
3. Baseline mechanics: coverage is everything.  At s=5 every r
   collapses (>= r12: audit pinned at the x0 plateau 7.09 — whole
   simplex regions hold no delivered point better than x0).  Its only
   viable configuration is coarsest-grid + finest interleaving
   (r10 s=1), whose audited trajectory is BACK-LOADED: pinned near the
   plateau until the snake nears full coverage (~35-40k grads), then a
   terminal staircase — the same anytime-vs-terminal shape as the K=5
   bandit finding 3.  The adaptive curve is anytime-smooth.
4. CPU axis (honest counterpoint, unchanged story): adaptive wall
   3,782 s (74% = strict-search decisions) vs baseline legs
   ~1,110-1,260 s; at ~1,150 s of CPU the r10 s=1 baseline already has
   its 0.1114 while the adaptive is still ~0.13.  Cheap-gradient
   problems keep the baseline competitive on wall clock.
5. Boundary statement for the paper (both experiments together): with
   tolerance machinery (certificates + gram share + deep targets) the
   baseline wins the terminal grads comparison at its knee
   (fixed_budget_B80912: 0.0595 vs 0.102); strip the machinery to a
   pure allocation-policy contest at equal budget and the adaptive
   wins 2.4x with half the budget to parity.  Neither result
   overclaims the other; together they LOCATE the K-advantage: it
   lives in the allocation policy, and the baseline's tolerance-mode
   strength lives in memoisation, which needs certificates to exist.
