# July 15, 2026 — fast method (Gram + Momentum-SVRG): implementation + K=6 trial

Scope: FIVE new code files under `Original_py/` (suffix `_fast`), two new
run folders under `output/`, two reference documents exported to the
user's Desktop, and a LEDGER session-9 entry.  **No original file was
modified** — `bundle.py`, `algorithm*.py`, `objectives_torch.py`,
`baseline*.py` and every existing run folder are untouched (the only
edits to existing artifacts are additive: this note, the LEDGER entry).
Design reference: the July 15 bilingual plan documents (see §1).

## 1. Purpose and plan documents

User request: speed up the adaptive bundle method along two routes —
(1) a variance-reduced stochastic inner loop whose anchors are the
full-gradient bundle points ("only full-GD points enter the bundle" was
refined during discussion to the inclusive policy of §4), and (2) a
cheaper λ-selection.  During code review a third, dominant move was
found: the exact Gram identity for GN evaluation.  The frozen plan
(approach, steps, parameter tables with tuning guidance and importance
ranking, file list, implementation order) was exported bilingual to

    ~/Desktop/MOO_bundle_acceleration_plan_ZH.md
    ~/Desktop/MOO_bundle_acceleration_plan_EN.md

with SVRG upgraded to **Momentum-SVRG** (heavy-ball on the SVRG
estimator) per the user's follow-up.  Governing principle settled in
discussion: **during the run the bundle is inclusive** (GN is a min over
a growing set — adding a point only ever weakly improves the criterion;
"passing eps/3" is a property of a (point, λ) pair, not of a point);
**pruning happens only at delivery time**, by λ-activation, with bitwise
GN invariance.

## 2. New code files (originals untouched)

| file | contents |
|---|---|
| `bundle_fast.py` | `BundleFast(Bundle)` maintaining Gram cache `M_i = J_i J_i^T` on add/pop/init; `simplex_grid`; `prune_inactive` (delivery-time pruning; asserts every probe-λ GN value bitwise unchanged) |
| `objectives_torch_fast.py` | original full oracles verbatim (delegates to `make_mlp_nonconvex`) + `StochLamOracle`: stratified per-class minibatch (b_k ∝ n_k), single-backward λ-scalarized gradients, persistent anchor-net pair, IFO counter; dataset re-derived from the same seed (first-rng-consumer argument) and verified bit-identical |
| `algorithm_fast_without_256_checkpoints.py` | (a) Gram-path GN: `GN(λ)=min_i λ^T M_i λ`, `∇=2M_{i*}λ` — exact rewrite, O(m·K·d)→O(m·K²); (b) two-tier λ-search (cheap = centroid+vertices+prev_lam, tol 1e-4/30 it; strict = original start set, tol 1e-8/100 it) with stop-verify: only a strict value ≤ 2ε/3 stops the run, a failed verify hands the strict λ to the inner loop, `sticky_strict` stays strict afterwards; (c) Momentum-SVRG inner loop in segments (see §3); (d) grad-equivalent accounting: joint call = K, minibatch step = 2b·K/n (classes partition the samples) |
| `sanity_checks_fast.py` | 6/6 PASS — Gram≡einsum to 1e-16 (value+jac+argmin); stoch full-batch ≡ J^Tλ to 1e-16 (data identity); **Momentum-SVRG degeneration (β=0, p_seg=1, b=n, c=1) ≡ original T-map inner loop bitwise (worst Δx = 0.0)**; prune invariance; strict fast search ≡ original maximiser (rel 0); end-to-end smoke in both tier modes |
| `run_trial_K6_fast_without_256_checkpoints.py` | trial runner: `--smoke`, CLI overrides `--batch/--beta/--step-c/--variant-tag`, reuses the July 11 reference curves (§5), produces 4 figures + summary.json + bilingual README |

## 3. The Momentum-SVRG inner loop (what actually runs)

At fixed λ the inner task is GN(λ;B) ≤ ε/3 — a single smooth nonconvex
problem in F_λ.  Segment recipe:

    anchor a = argmin_i {F_λ(x_i) − ‖∇F_λ(x_i)‖²/(2L_λ)}   (T-map rule, bundle cache)
    u←0, y←a, g_a = J_a^T λ  (snapshot gradient is FREE — it is in the bundle)
    ≤ p_seg steps: v = ∇f_{λ,S}(y) − ∇f_{λ,S}(a) + g_a ;  u = βu+v ;  y ← y − (c/(L_λ·L_scale))·u
        early trigger: ‖v‖² ≤ ρ·(ε/3) for `consec` consecutive steps
    segment end: FULL joint evaluation of y → ALWAYS added to the bundle →
        exact ε/3 check on full gradients (Gram path)
    safeguard at segment end: F_λ(y) > F_λ(a)(1+slack) ⇒ L_scale×2, momentum
        reset, re-run segment from the same anchor; violating point stays.

Randomness can never fake a certificate (all acceptance tests are on
full gradients); it can only delay the inner loop.  p is a CAP, not a
fixed length — the ‖v‖² trigger ends a segment as soon as stationarity
is plausible, and an overshot ε/3 point is harmless (the goal is to
produce SOME passing point, not the first one).

## 4. Trial setup (K=6, 96x96, eps mode)

Config: K=6, p=20, n=50,000, hidden [96,96], tanh, seeds 7/8 — the SAME
problem instance as the July 11 B180180 trial — epsilon=1e-3, round fuse
max_outer=150 (matching the old adaptive run), nominal budget cap
180,180 grad-equivalents, checkpoint cadence 600, λ-search ALL-STRICT at
64 starts (after the Gram rewrite the strict tier is ~2.9 s/round even
at bundle size ~1500, and all-strict keeps the self-reported metric
exactly on the legacy 64-start yardstick; the two-tier machinery is
implemented and sanity-covered, available for larger K).  Delivery
pruning grid r=10.

## 5. Reused reference curves (disclosed here and in both READMEs)

The baseline (r=10) and ORIGINAL-adaptive curves in the comparison
figures are READ from
`output/trial_K6_d11910_h96x96_tanh_n50000_B180180_without_256_checkpoints/summary.json`
(July 11/12 run), not re-run.  Validity: same instance and x0 (seeds
7/8), same fuse, same 64-start self-reported metric; the old adaptive
run was budget mode (epsilon=None), but with epsilon=1e-3 the recorded
trajectory is IDENTICAL — the two epsilon-mode tests (outer stop at
2ε/3≈6.7e-4, inner target ε/3≈3.3e-4) never fire on a run whose GN never
went below 0.147, and budget mode's CPU axis if anything flatters the
ORIGINAL method (epsilon mode would charge it an extra per-step GN
check).  Caveat kept: the CPU comparison is cross-run on the same
machine (the July 11 folder logs machine load; oracle pace within 6.5%
of calibration).  Re-running all three same-day would cost ~70 min
(baseline) + ~3 h (original adaptive) with the old scripts, unchanged.

## 6. Parameter iteration: v1 → v2 (both folders kept)

* **v1** = plan defaults (b=1024, β=0.9, c=0.1), folder
  `..._eps0.001_fast_msvrg_without_256_checkpoints`.  Wall 1,399 s at
  the 150-round fuse (λ-search 432 s = 31%, vs ~95% in the original
  run) — but the quality tail SATURATED at best-so-far 0.774: a
  variance floor (2% sampling × heavy-ball amplification 1/(1−β)=10×),
  flagged by exactly the plan's tuning-table signals (safeguard silent
  at L_scale=1.0; 148/150 rounds at the segment cap; oscillating tail).
  Kept as the tuning record, README annotated.
* **v2** = tuning-table response (b=4096 → variance /4; β=0.5 →
  amplification 2×; c unchanged; per-segment cost unchanged since
  p_seg=⌈n/b⌉ drops 49→13), folder `..._v2`.  **No saturation.**

## 7. Results (v2, self-reported track)

| quantity | original adaptive (Jul 11) | fast v2 (this note) |
|---|---|---|
| final best-so-far GN | 0.1473 | **0.1526** (same level) |
| wall at the same 150-round fuse | 10,375 s | **1,225 s (≈8.5×)** |
| λ-search share | ~95% | 35% (429 s) |
| grads consumed | 22,500 | 28,169 grad-equivalents (1,500 joint + 159.7M IFO) |
| stop | round fuse | round fuse (ε not reached, disclosed) |
| bundle at delivery | — | 1,501 → 913 after activation pruning (probe GN bitwise unchanged) |

Cross-method (fast v2 vs baseline, symmetric time-to-target at the
common target 0.1526): CPU 13.2×, gradients 15.3× in favour of fast.
Against the original adaptive at the same target: 1,063 s vs 10,375 s.
On the GRADIENT axis fast v2 is roughly at par with the original
(0.153 @ 28.2k equiv vs 0.147 @ 22.5k) — the net win is the TIME axis
(Gram rewrite + cheap minibatch steps), consistent with the plan's
prediction that idea 2 (λ-search) was the dominant lever and idea 1
buys oracle-time, not oracle-count.

Figures (per folder): `gn_vs_grad_evals_baseline_vs_fast.png`,
`gn_vs_cpu_time_baseline_vs_fast.png`,
`gn_vs_grad_evals_adaptive_orig_vs_fast.png`,
`gn_vs_cpu_time_adaptive_orig_vs_fast.png`; plus `summary.json`,
bilingual README, `run_log.txt`.

## 8. Caveats, stated once

Self-reported meters (the λ-search value is a heuristic lower bound of
an NP-hard max; under-search UNDERSTATES the criterion).  Single
instance, single machine; reused curves are a cross-run CPU comparison
(§5).  Momentum-SVRG's inner guarantee is expectation-type; all
acceptance tests are exact.  inner_cap_hits=150 — every round hit the
segment cap before ε/3 at the active λ (expected at ε=1e-3 from a cold
start), so the Algorithm-2 termination argument does not apply to these
rounds; the certificate machinery (strict stop line at 2ε/3) never
triggered.  L_scale stayed 1.0 in both fast runs (the segment-end
safeguard is coarser than the original per-step descent-lemma check —
by design; documented in the plan).

## 9. TODO (user-flagged, next session)

* **Reach ε=1e-3 for real: raise max_outer.**  At the 150-round fuse
  the best-so-far is still ~0.15 and DESCENDING (no plateau found by
  the detector on the fast curve either).  At v2's pace (~8 s/round),
  doubling the fuse costs only ~40 min wall; certification requires
  driving the strict 64-start search value under 2ε/3 ≈ 6.7e-4, so a
  substantially larger fuse (and possibly cheap-tier rounds plus the
  stop-verify path, already implemented) is the natural next
  experiment.
* Optional follow-ups noted in the plan: enable the two-tier λ-search
  in production once bundles grow (v2 measured strict at ~2.9 s/round
  ≈ 35% of wall — the cheap tier would cut ~380 s); incremental anchor
  selection to trim the O(m·K·d) per-segment einsum as m grows.

## 10. Files added / touched

| path | change |
|---|---|
| `Original_py/bundle_fast.py`, `objectives_torch_fast.py`, `algorithm_fast_without_256_checkpoints.py`, `sanity_checks_fast.py`, `run_trial_K6_fast_without_256_checkpoints.py` | NEW (the `_fast` set) |
| `output/trial_K6_d11910_h96x96_tanh_n50000_eps0.001_fast_msvrg_without_256_checkpoints/` | NEW run folder (v1, tuning record; README annotated) |
| `output/trial_K6_d11910_h96x96_tanh_n50000_eps0.001_fast_msvrg_without_256_checkpoints_v2/` | NEW run folder (v2, headline result) |
| `~/Desktop/MOO_bundle_acceleration_plan_{ZH,EN}.md` | NEW (outside the repo, per user request) |
| `Note/Jul_15_note.md` | this note |
| `/Users/shirch/vscode101/.venv/LEDGER.md` | session-9 entry + state block (outside the repo, per standing rule) |
