# Offline-bandit toy: MSVRG uniform baseline vs MSVRG adaptive bundle

SURF paper's offline bandit (Appendix F.1; identical to our paper's
Section 5.2) run through the Momentum-SVRG pair on the
without-256-checkpoints track. Design record: `Note/Jul_26_note.md`
(repo root) and `Adaptive Bundle Algorithm/OFFLINE_BANDIT_TOY_INTEGRATION_ZH.md`.

## Problem

Reduced logits theta in R^4, pi = softmax([theta, 0]);
F_k(theta) = 0.05 * KL(pi || uniform) - <pi, Rk_hat>, K = 2;
R1(a) = x_a, R2(a) = 1 - x_a^4 observed through a balanced offline
dataset (T = 1000 rows, reward noise std 0.5, data_seed 7). Everything
both solvers see is built from the plug-in estimates (R1_hat, R2_hat);
the closed-form softmax solution is a NEVER-TIMED oracle used only for
evaluation (exact plug-in PF, scalarized optima, SURF Eq. (9) arc CDF).

## Folder map

- `smoke/`    — plumbing-validation run (small caps; not a result).
- `eps1e-2/`, `eps1e-3/`, `eps1e-4/` — the formal runs, one per rung.
  Each contains `summary.json`, `raw_histories.npz`, and the figures
  (fig1–fig5; rungs re-run on Jul 27 also carry the fig4b/fig4c
  companions — see the revision section below):
  - `fig1_gn_vs_cpu.png`, `fig2_gn_vs_grads.png` — common-meter
    worst-case GN* (strict in-family search over the WHOLE simplex,
    off both cost axes) vs CPU / grad-equivalents.
  - `fig3_pareto_front.png` — delivered fronts over the exact plug-in
    PF, SURF 12 arc-uniform reference points, max-dist + IGD printed
    under the axes.
  - `fig4_value_gap_profile.png` — delta_value(w) per query weight at
    the matched grad budget (solid) and at each method's own delivery
    (dashed); light verticals mark the 12 grid nodes.
  - `fig5_value_gap_convergence.png` — eps_value (best-so-far) vs CPU
    and vs grad-equivalents.
- `FUTURE_WORK.md` — user-defined follow-ups (K up, non-convex rewards,
  r sweep).

## Reproduce

From `Adaptive Bundle Algorithm/Original_py/`:

    KMP_DUPLICATE_LIB_OK=TRUE python sanity_checks_bandit_toy.py
    KMP_DUPLICATE_LIB_OK=TRUE python run_bandit_toy_without_256_checkpoints.py --epsilon 1e-2
    KMP_DUPLICATE_LIB_OK=TRUE python run_bandit_toy_without_256_checkpoints.py --epsilon 1e-3
    KMP_DUPLICATE_LIB_OK=TRUE python run_bandit_toy_without_256_checkpoints.py --epsilon 1e-4

Deterministic given (data_seed 7, sampler_seed 41); randomness affects
only how long solving takes, never a certificate (all acceptances are
on full gradients).

## Jul 27 revision (applies to all three rungs)

All three rungs were re-run with `--eval-every 0` (finest checkpoint
cadence: every processed node/round PLUS every MSVRG segment-end
delivery — recording only, trajectories bit-identical; see
`Note/Jul_27_note.md`).  Presentation changes in the current driver:

- fig1/fig2: equal-budget marker removed; fig1 CPU curves end at each
  curve's last GN improvement (stop-verification tails not shown; full
  totals stay in `summary.json`) and display node/round MILESTONE
  checkpoints only; fig2 displays the milestones plus segment
  checkpoints spaced ~1/10 of the budget axis — the dense per-segment
  data stays in `raw_histories.npz`.
- fig3: now "nondominated discovered fronts" — the nondominated subset
  of ALL evaluated points per method (baseline: delivered set;
  adaptive: pre-prune bundle), query-free.  Metrics under the axes:
  `max_front_to_oracle` (worst off-front error of a front point) and
  `igd` (oracle-front coverage), stored in
  `summary.json -> pf_front_nondominated`.  The old query-based
  `pf_metrics` block is still stored for cross-rung comparability —
  the two max-dist numbers have DIFFERENT semantics; do not mix.
- new `fig4b_value_gap_profile_matched.png`: fig4's two matched-budget
  (solid) profiles alone.
- new `fig4c_value_gap_profile_owndelivery.png` (eps1e-3 and eps1e-4;
  eps1e-2 gains it on its next re-run): the two FULL-delivery
  value-gap profiles alone, both solid — the own-endpoint comparison;
  legend uses the plain method names ("momentum-SVRG baseline" /
  "momentum-SVRG adaptive bundle method"); grad totals stay in
  `summary.json`.

Reproduce the revised rungs:

    KMP_DUPLICATE_LIB_OK=TRUE python run_bandit_toy_without_256_checkpoints.py --epsilon 1e-2 --eval-every 0
    KMP_DUPLICATE_LIB_OK=TRUE python run_bandit_toy_without_256_checkpoints.py --epsilon 1e-3 --eval-every 0
    KMP_DUPLICATE_LIB_OK=TRUE python run_bandit_toy_without_256_checkpoints.py --epsilon 1e-4 --eval-every 0

## eps1e-4 caveat (Jul 27): the adaptive's certificate at this rung is FALSE

The denser meter exposed that the adaptive's `epsilon_certified` stop at
the eps1e-4 rung is a search-limited false certificate: the EXACT
dense-grid worst case of its family (pre-prune AND delivered — both) is
GN* = 1.279968e-4 > eps at w = 0.0943, with the violating pocket
w in [0.080, 0.113] (~3.4% of the simplex).  The internal 64-start
lambda search missed it; searches only ever under-report a maximum.  Do
NOT quote the adaptive as having reached eps = 1e-4 on this rung; on
the strict meter only the baseline certifiably reaches eps (at
1118.128 grad-equivalents).  The eps1e-2 / eps1e-3 adaptive
certificates PASS the same exact check (their recorded finals equal
the true maxima).  Full analysis + verification recipe:
`Note/Jul_27_note.md` section 8.
