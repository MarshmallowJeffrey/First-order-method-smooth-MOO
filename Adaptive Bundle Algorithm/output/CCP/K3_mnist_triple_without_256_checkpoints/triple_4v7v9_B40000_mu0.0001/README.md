# K3 MNIST triple 4 vs 7 vs 9 — pure fixed budget campaign (Aug 26, 2026)

Legs: baseline simplex grids r in [10, 20, 30] + adaptive CCP, all at
s = 5, B = 40,000 grad-equivalents, batch
1024, per_class =
5842 (balanced maximum), d = 8195.
Quality: audit_v2 two-instrument meter (IPOPT strict multistart + heavy
CCP) at every checkpoint + dense simplex-grid lower-bound cross-check at
the final stack (the K=2 exact 1-D meter has no K=3 analogue).  Test
values: ALL official t10k rows of the three digits.  Figures (Aug-26
restyle, modelled on the breakable-bottles reference layout):
gn_vs_grads, gn_vs_cpu, front_train + front_test (adaptive CCP vs the
best baseline only: ONE row of three 3-D views at fixed angles
(22,-60)/(18,-140)/(34,115), linear axes, window <= ln 3; pairwise
log-log projections in the companion *_proj.png), front_err_test (all
legs, same layout + companion projections), test_ce_vs_budget.
Divergence arms of vertex / edge grid nodes lie outside the ln-3
window by design (no regularisation).

| leg | delivered pts | train front | test front | HV central train | HV central test |
|-----|---------------|-------------|------------|------------------|-----------------|
| adaptive CCP | 4,388 | 460 | 1941 | 1.2561 | 1.3085 |
| baseline r=10 | 4,841 | 371 | 1792 | 1.2594 | 1.2962 |
| baseline r=20 | 4,573 | 879 | 2264 | 1.2553 | 1.2915 |
| baseline r=30 | 4,477 | 1397 | 1822 | 1.2537 | 1.2828 |
