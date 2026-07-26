# fast_method_trials — the accelerated-method (Gram + Momentum-SVRG) series

All four runs here share the SAME problem instance and settings unless
noted: K=6, p=20, n=50,000, hidden [96,96], tanh, seeds 7/8, d=11,910,
epsilon=0.001, without-256-checkpoints track.  They differ ONLY in the
knobs listed below.  Reference curves (baseline r=10 and the ORIGINAL
adaptive method) come from the July 11 run one level up:
`../trial_K6_d11910_h96x96_tanh_n50000_B180180_without_256_checkpoints/`
(NOT part of this series — do not move it; every summary here points to
it).  `../calibration_speed_test_B2000/` (renamed Jul 16 from
`trial_K6_..._B2000_without_256_checkpoints`) is the July 11
speed-calibration pre-run that dimensioned the B180180 trial, also
unrelated to this series.

| folder | date | what it is | key knobs | result | role |
|---|---|---|---|---|---|
| `v1_plan_defaults` | Jul 15 | first fast run, plan-default parameters | b=1024, β=0.9, step_const=0.1, strict λ-search, max_outer=150 | best GN 0.774 — tail SATURATED (variance floor) | tuning record; superseded by v2 |
| `v2_tuned_b4096_beta0.5` | Jul 15 | tuning-table response to v1 | b=4096, β=0.5 (rest as v1) | best 0.1526 ≈ original's 0.1473 at 1,225 s vs 10,375 s (**~8.5×**) | headline result of the plan-default design |
| `v3_rel_target_two_tier` | Jul 16 | user-approved redesign | + two-tier λ-search (cheap values plotted), + rel_target=0.25, max_outer=500 | best 0.0581 in **293 s** / 9,226 grad-equiv; cap_hits 0/500; exactly 1 segment/round | headline result of the adaptive-target design |
| `cert_attempt_partial` | Jul 16 | certification attempt, max_outer=3000 — **user-stopped at round ~1441** | v3 config, longer fuse | best 0.0291 at 7,173 s (salvaged from checkpoint log; nothing recomputed) | evidence: slope decayed to plateau-grade at ~2.9e-2 → eps=1e-3 not reachable by round count alone; also exposed the stacked-copy waste (fixed same day) |

Progression of the self-reported best-so-far GN (original method:
0.1473 at 10,375 s): v1 0.774 → v2 0.1526 → v3 0.0581 → cert 0.0291.

## What exactly changed between versions

**Shared base (all four runs):** Gram-path λ-search (exact rewrite),
Momentum-SVRG segmented inner loop with full-gradient acceptance,
grad-equivalent accounting, delivery-time pruning, same instance/seeds,
epsilon=0.001.

* **v1 → v2 — parameters only, no algorithmic change.**
  `msvrg_batch` 1024 → 4096 (gradient variance ÷4) and
  `msvrg_momentum` 0.9 → 0.5 (heavy-ball noise amplification
  1/(1−β): 10× → 2×); `p_seg = ⌈n/b⌉` follows automatically 49 → 13,
  so per-segment cost is unchanged.  Motivation: v1's tail saturated on
  a variance floor (safeguard silent, 148/150 rounds at the segment
  cap, oscillating tail) — exactly the plan's tuning-table symptoms.
* **v2 → v3 — two algorithmic changes + one fuse change.**
  (1) λ-search: all-strict (64 starts every round) → TWO-TIER — cheap
  tier (≈K+2 starts, tol 1e-4) on ordinary rounds with its values
  plotted directly, strict tier (64 starts, tol 1e-8) reserved for
  signing the stopping certificate (stop-verify).
  (2) Inner-loop target: absolute eps/3 → RELATIVE
  max(eps/3, 0.25·pc_val) — "cut the round's worst direction to a
  quarter", flooring to the paper's eps/3 in the endgame (Algorithm-2
  variant; the stopping certificate is untouched).  This is what
  removed v2's cap_hits (150/150 → 0/500).
  (3) `max_outer` 150 → 500.  MSVRG parameters stay at v2 values.
  Emergent behaviour: exactly 1 segment per round (the adaptive target
  auto-realised the fastest λ-turnover rhythm).
* **v3 → cert — one number: `max_outer` 500 → 3000.**  No other change;
  stopped by the user at ~round 1441 (per-round cost had grown with the
  bundle — the stacked-copy waste found here was fixed the same day —
  and the slope had decayed to plateau-grade at ~2.9e-2).

## Old folder names (referenced by Notes/LEDGER written before this
## reorganisation, July 16)

| old path (under `output/`) | now |
|---|---|
| `trial_K6_d11910_h96x96_tanh_n50000_eps0.001_fast_msvrg_without_256_checkpoints` | `fast_method_trials/v1_plan_defaults` |
| `..._fast_msvrg_without_256_checkpoints_v2` | `fast_method_trials/v2_tuned_b4096_beta0.5` |
| `..._fast_msvrg_without_256_checkpoints_v3` | `fast_method_trials/v3_rel_target_two_tier` |
| `..._fast_msvrg_without_256_checkpoints_cert` | `fast_method_trials/cert_attempt_partial` |

Code: the `_fast` set under `Original_py/` (`bundle_fast.py`,
`objectives_torch_fast.py`, `algorithm_fast_without_256_checkpoints.py`,
`sanity_checks_fast.py`, `run_trial_K6_fast_without_256_checkpoints.py`).
Design documents: `~/Desktop/MOO_bundle_acceleration_plan_{ZH,EN}.md`.
Narrative records: `Note/Jul_15_note.md` (v1/v2), `Note/Jul_16_note.md`
(rename, v3, cert attempt, fixes).
