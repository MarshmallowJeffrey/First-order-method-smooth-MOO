# K3 MNIST triple 3 vs 5 vs 8 — pure fixed budget campaign (Aug 26, 2026)

Legs: baseline simplex grids r in [10, 20, 30] + adaptive CCP, all at
s = 5, B = 40,000 grad-equivalents, batch
1024, per_class =
5421 (balanced maximum), d = 8195.
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
| adaptive CCP | 4,543 | 298 | 886 | 1.3256 | 1.2942 |
| baseline r=10 | 4,982 | 351 | 961 | 1.3258 | 1.2723 |
| baseline r=20 | 4,730 | 919 | 1210 | 1.3258 | 1.2709 |
| baseline r=30 | 4,605 | 268 | 988 | 1.3244 | 1.2377 |
