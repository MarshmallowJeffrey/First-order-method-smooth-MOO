# Future work — SURF offline-bandit toy comparison

User-defined follow-ups (July 26, 2026), in priority order as given.
Baseline for all of them: the current setup in this folder
(reduced logits, MSVRG pair, without-256-checkpoints track, r=11,
eps in {1e-2, 1e-3}, b=256, data_seed 7).

## 1. Increase K (number of objectives)

The K=2 toy cannot show the baseline's combinatorial grid growth
(|G_r| = C(r+K-1, K-1)); with K=2 the grid is only r+1 nodes. Design
needs: a K>2 reward family on the same bandit (e.g. K reward vectors
R_k(a) with distinct argmax arms so the front is genuinely
K-dimensional), then the same pipeline. Expected story: the adaptive
method's budget allocation advantage should appear on the
grad-equivalent axis as K grows, matching the paper's O((1/eps)^K)
comparison and the MLP-track findings (K=6 sessions).

## 2. Make the objectives genuinely non-convex ("R non-convex")

Currently F_k is tau-strongly convex in pi (hidden convexity), which is
what gives SURF its closed form. Replace the linear reward term
<pi, R_k> with a non-convex functional of pi (or parameterize through a
small network), so that: our smoothness assumptions (Assumption 2.1-2.3)
still hold, but the closed-form softmax oracle and SURF's Eq. (9) speed
die. The bundle method's guarantees survive; SURF's Rule 1 needs its
general Algorithm 1 (Rule 2) instead. This isolates exactly what the
bundle method buys beyond the hidden-convex regime. Ground truth then
comes from a heavily-converged reference solver, not a formula — budget
the reference runs accordingly.

## 3. Increase r (grid resolution sweep)

Deferred from the current run at the user's request (r=11 only for
now). The ladder r in {11, 23, 47} doubles the node count per rung
(12 -> 24 -> 48) and should drop the baseline's between-node geometric
floor by ~4x per rung (floor proportional to (grid gap)^2). Reading the
common-meter plateau against r on a log-log plot tests the 1/r^2
prediction directly; the equal-quality CPU/grad brackets per rung are
the headline numbers.

## Also worth noting (not user-ordered)

- Multi-seed statistics: current runs are data_seed 7 / sampler_seed 41
  single-instance, consistent with the track's style. 30 (later 100)
  data seeds would put error bars on time-to-eps, IGD, and eps_value.
- b = n = 1000 degeneration line (deterministic momentum full-gradient)
  was dropped from the main plan with user consent; the module-level
  full-batch exactness check in `sanity_checks_bandit_toy.py` covers
  the correctness half of it.
