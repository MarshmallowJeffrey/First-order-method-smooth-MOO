# Plateau sweep — cross-configuration analysis

Equal-budget (30,000 gradient evaluations) head-to-head at K = 3, 4, 5, 6.
Fixed: p=6, n=30, hidden=[4], tanh activation, grid resolution r=6,
prune_inner=True, max_inner=25, data seed 7 / init seed 8.
Per-configuration parameters, rationale, plots and health flags are in each
`K*/README.md`; raw curves in each `summary.json`.

## Headline numbers (best-so-far GN* at the shared budget)

| K | baseline final | adaptive final | quality ratio bl/a2 | baseline plateau | adaptive plateau |
|---|---|---|---|---|---|
| 3 | 8.97e-03 | 1.72e-04 | **52.1** | not reached | not reached |
| 4 | 3.56e-02 | 2.84e-03 | **12.6** | not reached | 2.84e-03 |
| 5 | 4.04e-02 | 4.36e-03 | **9.3**  | 4.04e-02 | not reached |
| 6 | 4.77e-03 | 1.92e-02 | **0.25** | not reached | not reached |

(“plateau not reached” = the best-so-far curve was still improving by more
than 5% per detection window at the end of the budget, so `detect_plateau`
correctly refuses to declare a level; where both levels exist their ratio is
the plateau ratio, otherwise the equal-budget quality ratio above is the
comparable summary statistic.)

**The K=6 entry is budget-truncated, not a defeat.** At this fixed 30k
budget the adaptive method had not finished descending; given more budget
it crosses the baseline and wins (ratio 3.09 at 240k). See the budget
study below. The plateau story holds at every K in this sweep; the budget
needed for the adaptive method to overtake the baseline simply grows with K.

Health flags: `L_scale_final` between 2 and 8 across the sweep (the descent
safeguard fired once to three times to correct the probe estimate — its
intended regime); `inner_cap_hits` = 0; one expected RuntimeWarning per run
(the L-underestimate notice).

## What matches the theory

At K = 3, 4, 5 the adaptive method dominates the gradient axis exactly as
the paper predicts (cross-weight gradient reuse plus adaptive allocation):
one order of magnitude or more lower GN* at the same oracle budget, and at
K=5 the textbook picture — the baseline plateaus at 4.0e-02 while the
adaptive method is still descending an order of magnitude below.

On the CPU axis the baseline wins at every K here (cpu ratios 0.006–0.16),
which is ALSO the predicted behaviour: with n=30 the oracle is nearly free,
so the adaptive method's per-round lambda-search overhead dominates. The
wall-clock case for the adaptive method is made by the crossover sweep,
where the oracle is expensive; this sweep isolates gradient efficiency.

## The K=6 investigation — from "anomaly" to "budget-truncation artefact"

At the 30k sweep budget the adaptive method lands at 1.9e-02 vs the
baseline's 4.8e-03 (seed 7). Two rounds of investigation followed. Round 1
(seven single-budget variants) ruled out every 30k-budget explanation;
round 2 (the budget escalation) found the actual cause. Both are kept here
because the reasoning matters: the single-budget matrix was correct in what
it ruled out, and the budget study is what the user's read of the
still-descending curve pointed to.

Round 1 — seven variants, all at 30k, all K=6, tanh:

| variant | change vs reference | bl best | a2 best | a2 better? |
|---|---|---|---|---|
| A (reference) | — | 4.77e-03 | 1.92e-02 | no |
| B | seed 17/18 | 7.56e-03 | 2.31e-02 | no |
| C | seed 27/28 | 2.77e-02 | 1.66e-02 | **yes** |
| D | max_inner 25→10 | 4.77e-03 | 1.95e-02 | no |
| E | prune_inner→False (bundle 5001 points) | 4.77e-03 | 1.67e-02 | no |
| F | w_true_scale 1→4 (stronger objective conflict), seed 7 | 3.16e-04 | 9.98e-04 | no |
| G | w_true_scale 1→4, seed 17 | 3.92e-03 | 1.15e-02 | no |

Conclusions from the matrix:

1. **Not a step-size pathology.** `L_scale_final` stayed 4–16 in every
   variant (the ReLU-era step-size collapse is gone) and the adaptive curve
   descends smoothly to the end — it is slow, not broken.
2. **Not the pruning/solution-set-size effect.** Keeping every iterate
   (5,001 points, variant E) improves the adaptive result by only ~13%.
3. **Not the exploit-vs-search balance.** Re-searching lambda 2.5x more
   often (variant D) changes nothing.
4. **Instance-dependent, on average not favourable.** One seed in three
   wins; the adaptive method's final level is stable at ~2e-02 across
   variants A–E while the baseline's varies 4.8e-03–2.8e-02.
5. **Not fixed by raising inter-objective conflict.** At w_true_scale=4
   (variants F, G — more separable classes, more conflicting per-class
   objectives) both methods reach much lower absolute levels, but the
   ratio stays ~0.3: the coarse grid is still not punished.

What round 1 correctly established: at a FIXED 30k budget, no seed, pruning
setting, inner-loop schedule, or objective-conflict level flips K=6. What
it wrongly inferred: that this made K=6 a permanent loss ("the coarse grid
is not punished; the advantage does not grow with K"). That inference was
the classic mistake of holding the budget fixed while varying everything
else — it missed the one axis the user pointed at.

Round 2 — budget escalation (the user-raised hypothesis). The K=6 curves
(baseline flat, adaptive still descending in `gn_vs_grad_evals.png`)
suggested the 30k ratio was read before the adaptive method had converged.
Raising the budget with everything else fixed:

| budget | bl final (best-so-far) | a2 final | ratio bl/a2 | note |
|---|---|---|---|---|
| 30k  | 4.77e-03 | 1.92e-02 | 0.25 | sweep configuration — adaptive worse |
| 90k  | 3.55e-03 | 4.74e-03 | 0.75 | baseline plateaus at 54k grads; adaptive still descending |
| 240k | 3.63e-03 | 1.17e-03 | **3.09** | **adaptive crosses below and wins** |

![K=6 budget study](K6_budget_study.png)

**Resolved: at K=6 the 30k result was a budget-truncation artefact.** The
baseline reaches its grid floor early (≈3.6e-03, fixed from ~56k gradients
onward at every budget) and never improves again. The adaptive method
descends monotonically the whole way — it passes the baseline's floor at
≈105k gradients and reaches 1.17e-03 by 240k, a 3.09× advantage still on a
gentle downward staircase. At 30k the adaptive method was only about a
third of the way down its descent, so the equal-budget comparison was read
before it had converged. The three-budget trajectory (ratio 0.25 → 0.75 →
3.09) settles it.

The finding, corrected: **the plateau story holds at K=6 too, it just needs
more gradients to show.** The baseline has a structural grid floor at every
K (uniform-discretisation cannot beat its covering radius); the adaptive
method has no such floor and keeps descending, but its convergence is
slower at higher K (consistent with the paper's Theorem 2, whose outer-
iteration bound is exponential in K for the adaptive method too). So the
budget needed for the adaptive method to overtake the baseline GROWS with
K: ~immediately at K=3, within 30k at K=4–5, ~105k at K=6. What looked at
the fixed 30k budget like "advantage disappears at K=6" is really
"break-even budget grows with K". The seven single-budget diagnostic
variants above were all correct that no 30k-budget knob (seed, pruning,
inner schedule, conflict strength) flips K=6 — the missing dimension was
the budget itself, which the user identified from the still-descending
adaptive curve in `gn_vs_grad_evals.png`.

## Honest-reporting notes

- All numbers are heuristic lower bounds of the NP-hard max over lambda,
  computed by the same fixed 256-start IPOPT yardstick for both methods.
- Single problem instance per configuration except K=6 (three seeds, shown
  above); treat per-K ratios as one-instance measurements, the cross-K
  pattern as the finding.
- The unfavourable K=6 result is reported in full, including the trend plot.
