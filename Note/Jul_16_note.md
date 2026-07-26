# July 16, 2026 — parameter renaming in the `_fast` set; v3 launch reverted

Scope: identifier renaming across the five `_fast` files and the two
Desktop plan documents; no algorithmic change of any kind (semantics,
defaults, math all identical).  Context section records the same-day
reverted launch.  Original (non-`_fast`) files untouched, as always.

## 1. Renaming (user request)

Motivation: the step-size constant was named with a bare "c"
(`msvrg_step_c`) and the early-trigger patience count was informally
abbreviated "c"/"c_consec" in discussion — two different parameters
colliding on one letter.  New names:

| old | new | meaning (unchanged) | default (unchanged) |
|---|---|---|---|
| `msvrg_step_c` (internal `step_c`, CLI `--step-c`) | `msvrg_step_const` (internal `step_const`, CLI `--step-const`) | step size η = step_const / (L_λ·L_scale); with momentum β the long-run effective step ≈ step_const/(1−β) | 0.1 |
| `msvrg_trigger_consec` (internal `trigger_consec`) | `msvrg_trigger_patience` (internal `trigger_patience`) | how many CONSECUTIVE small-‖v‖ steps (‖v‖² ≤ ρ·ε/3) before an early segment-end full check | 2 |

Files touched: `Original_py/algorithm_fast_without_256_checkpoints.py`,
`Original_py/run_trial_K6_fast_without_256_checkpoints.py`,
`Original_py/sanity_checks_fast.py`,
`~/Desktop/MOO_bundle_acceleration_plan_{ZH,EN}.md` (parameter tables,
pseudo-code, tuning notes).

NOT touched (historical records keep the old key names, by convention):
the v1/v2 run folders' `summary.json` / READMEs, `Note/Jul_15_note.md`,
the LEDGER session-9 body (an addendum line was appended instead).

Verification: `sanity_checks_fast.py` 8/8 PASS after the rename
(including the bitwise MSVRG-degeneration check, now stated as
step_const=1); `--step-const` CLI flag verified.

## 2. Context: v3 launch cancelled and fully reverted (same day)

Earlier on July 16 a follow-up experiment pair was launched and then
CANCELLED at the user's request (design to be approved first):

* fast v3 (two-tier λ-search + max_outer=500, chasing eps=1e-3) — killed
  at ~round 55; partial output folder and log deleted;
* the accompanying code changes (round-start checkpoint-due logic that
  forced strict tier on recorded rounds; CLI flags --tier-mode /
  --max-outer / --orig-dir; a new runner
  `run_trial_K6_orig_eps001_without_256_checkpoints.py` for an original-
  method epsilon-mode reference re-run) — ALL reverted/deleted the same
  hour.  The `_fast` code is back to the exact state that produced the
  v1/v2 results (sanity 8/8 re-confirmed), except for the §1 renaming.

Standing decisions recorded from the same exchange: all experiments
(old and new curves, all future runs) stay on the
without-256-checkpoints track; no long runs are launched before the
user approves the experiment design.

## 3. Open items (carried from Jul_15_note §9; updated after the user's
## design review later on Jul 16)

* eps=1e-3 certification run (fast v3, APPROVED DESIGN): two-tier
  λ-search with the CHEAP-TIER VALUES PLOTTED DIRECTLY (user decision) +
  max_outer 150 → 500.  The v3 outcome is open BY DESIGN and both
  endings are valid results: either the strict verify signs an epsilon
  certificate (first live use of the stop-verify path), or the 500-round
  fuse fires and we get rate data ("how low by round 500").
* FUTURE-WORK / BACKUP — "recorded-rounds-strict" plotting guard: force
  the strict tier on checkpoint rounds so every plotted point is on the
  legacy 64-start yardstick.  Held in reserve; enable only if the
  mixed-tier curve proves misleading (the 8-start meter under-searches a
  maximiser, so it under-reports — a flattering bias — and mixing meters
  can add artificial sawtooth on top of the real new-worst-direction
  oscillation).
* PROPOSAL (pending user decision) — relative inner target, response to
  v2's cap_hits=150 (every round exhausted its 10 segments because the
  absolute eps/3 = 3.3e-4 target is unreachable from a cold start):
  eps_inner_eff = max(eps/3, gamma * pc_val) with gamma ~ 1/4, segment
  cap kept as a fuse.  Restores an achievable per-round postcondition
  ("cut the current worst direction to a quarter"); the stopping
  certificate is unaffected (still the strict-tier 2eps/3 line); the
  counting-argument variant remains to be written up.  Simply raising
  msvrg_max_segments does NOT fix cap_hits (the absolute target stays
  unreachable early on; more segments only spend more per wasted round).
* Original-method epsilon re-run CANCELLED (user decision, Jul 16): the
  original method cannot reach eps=1e-3 in feasible time anyway
  (quadratic λ-search cost in the round count), so the July 11 curve
  stays as the reference, with the reuse disclosure unchanged.
* Optional rhythm experiment (v4): `msvrg_max_segments=1` (one segment →
  one full evaluation → one new bundle point per round, λ re-chosen
  every round; a segment is still up to p_seg minibatch steps, NOT one
  step).  Pairs naturally with the relative inner target and with the
  cheap tier (λ-search runs 10x as often at that rhythm).
  [RESOLVED by §4: with rel_target=0.25 the run took exactly 1 segment
  per round on its own — the adaptive target subsumes the manual
  segments=1 experiment on this instance.]

## 4. v3 EXECUTED (later on Jul 16) — results and observation read-outs

Run: `run_trial_K6_fast_without_256_checkpoints.py --variant-tag v3`,
approved design (two_tier, cheap values plotted, msvrg_rel_target=0.25,
max_outer=500, v2 MSVRG parameters b=4096/β=0.5/step_const=0.1).
Folder: `output/..._eps0.001_fast_msvrg_without_256_checkpoints_v3/`.
Pre-run sanity 8/8 including a new per-round check that the inner
target equals max(eps/3, 0.25*pc_val).

Headline numbers: **wall 293 s for 500 rounds** (λ-search 136 s = 46%);
final best-so-far GN **0.0581** — 2.5x below both v2 (0.1526) and the
original method's final (0.1473) — on only **9,226 grad-equivalents**
(v2: 28,169; original: 22,500).  Time-to-the-original's-final-quality
(0.1473): v3 at 68 s / 2,409 grad-equivalents vs original 10,375 s /
22,500 grads (**~153x CPU, ~9.3x gradients**).  Vs baseline
(time-to-target): CPU 138x, gradients 171x.  Pruning 502 -> 498
(nearly every point λ-activated, consistent with one-point-per-round).
L_scale stayed 1.0.

Observation read-outs against §3:

* cap_hits: **0 / 500** (v2: 150/150) — the relative target restored an
  achievable per-round postcondition, as intended.
* **Segments per round: exactly 1, every round.**  γ=0.25 turned out to
  sit at the segments=1 rhythm on its own (one 13-step segment suffices
  to cut the worst direction to a quarter at every stage reached) — the
  user's "fast λ-turnover" intuition emerged automatically from the
  adaptive target; the planned manual v4 is subsumed on this instance.
* Endgame regime NOT reached: search values bottomed around 0.058–0.12,
  far above 4eps/3 ≈ 1.3e-3, so the max(·) floor never took over and
  the "more segments near the end" mechanism is still untested.
* Cheap-values-plotted verdict: the mixed... (no mixing occurred — all
  500 rounds were cheap; no stop-verify was ever triggered since values
  never approached 2eps/3).  The curve's sawtooth (0.06–0.12) matches
  the original curve's own new-worst-direction oscillation in shape; no
  misleading artifact — the strict-on-recorded-rounds BACKUP stays
  unused.
* Stop-verify still has never fired in production; it will only be
  exercised by a longer run that actually approaches the stop line.
* eps=1e-3 still NOT certified (fuse ending, as designed for).  Rate
  read: best-so-far fell ~0.41 dex over rounds 150→500; extrapolating
  ~1.2 dex per 1,000 rounds, certification (2eps/3 ≈ 6.7e-4) needs
  roughly 1,500–2,500 more rounds ≈ **15–25 min wall at v3's pace**
  (each round ≈ 0.59 s).  Natural next run: same config,
  max_outer 2,500–3,000 — pending user go.
  [SUPERSEDED by §5: the extrapolation failed — the slope decayed.]

## 5. Certification attempt (max_outer=3000): stopped, salvaged, and two
## bottlenecks found (later on Jul 16)

Run `--variant-tag cert --max-outer 3000` (v3 config otherwise) was
STOPPED BY THE USER at round ~1441 (~7,170 s CPU) after a progress
check.  Salvage: every checkpoint line in run_log.txt was parsed by the
one-off `Original_py/rebuild_cert_partial.py` into full curves (123
checkpoints, nothing recomputed), two comparison figures vs the original
method, `summary_partial.json`, and a bilingual README — all in
`output/..._fast_msvrg_without_256_checkpoints_cert/`.

Partial result: best-so-far GN **2.907e-2** at 7,173 s / 80,794
grad-equivalents (extends v3's 5.81e-2; original: 1.47e-1 at 10,375 s).

What the attempt revealed:

* **Bottleneck 1 — stacked-copy waste (FIXED same day).**  The MSVRG
  inner loop re-stacked the whole bundle every segment (three
  O(m·K·d) np.asarray copies + a full einsum: ~2.3 GB copied per
  segment at m ≈ 4000, ~17 GB/round).  Fix: per-round scalarised
  caches — F_λ, ∇F_λ, ‖∇F_λ‖² of existing points are built once per
  inner call in 128-point chunks (no (m,K,d) temporary) and appended
  incrementally as segments add points; u_vals is an O(m) vector op
  per segment.  Same math, different bookkeeping; sanity 8/8 after the
  fix with the degeneration check still BITWISE (|Δx| = 0.0).
  Expected effect: tail rounds ~24 s → ~3–5 s.
* **Bottleneck 2 — segments/round grew 1 → ~7.6 (design, not a bug):**
  at GN ≈ 0.03–0.05 the relative target "cut to a quarter" genuinely
  needs several segments.  After fix 1 this costs ~3–4 s/round and may
  be acceptable; further levers if wanted: (a) enlarge p_seg in the
  endgame (fewer, longer segments — fewer full evals per round; low
  risk at b=4096's 8% sampling), (b) relax gamma (0.25 → 0.4: lighter
  rounds, more of them — roughly work-conserving, limited gain),
  (c) endgame b increase if the floor is variance-driven (see below).
* **The real blocker is slope decay, now plateau-grade evidence:** the
  salvaged curve oscillates sideways in the 0.03–0.08 band from
  ~1,000 s on, and the plateau detector fires at level 2.907e-2.
  Best-so-far slope: ~1.2 dex/1000 rounds (150→500) → ~0.32 (500→1441).
  Certification at eps=1e-3 is NOT reachable by round-count alone at
  this pace; candidate explanations to separate next: variance floor of
  b=4096 (test: endgame with larger b / smaller steps) vs intrinsic
  max-min landscape difficulty at this level (test: a few rounds with a
  much deeper inner budget at fixed λ).  OPEN — next-step decision with
  the user.
