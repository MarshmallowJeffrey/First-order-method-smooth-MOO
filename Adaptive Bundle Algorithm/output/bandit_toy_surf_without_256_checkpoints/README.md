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
  Each contains `summary.json`, `raw_histories.npz`, and five figures:
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
