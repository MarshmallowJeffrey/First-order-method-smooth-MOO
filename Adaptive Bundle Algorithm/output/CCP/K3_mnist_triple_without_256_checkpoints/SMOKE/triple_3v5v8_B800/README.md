# K3 MNIST triple 3 vs 5 vs 8 — pure fixed budget campaign (Aug 26, 2026)

Legs: baseline simplex grids r in [4] + adaptive CCP, all at
s = 2, B = 800 grad-equivalents, batch
256, per_class =
300 (balanced maximum), d = 8195.
Quality: audit_v2 two-instrument meter (IPOPT strict multistart + heavy
CCP) at every checkpoint + dense simplex-grid lower-bound cross-check at
the final stack (the K=2 exact 1-D meter has no K=3 analogue).  Test
values: ALL official t10k rows of the three digits.  Figures:
gn_vs_grads, gn_vs_cpu, front_train + front_test (each showing adaptive
CCP vs the best baseline only: two 3-D views on log10 axes + three
pairwise log-log projections, window <= ln 3), front_err_test (all
legs), test_ce_vs_budget.  Divergence arms of vertex / edge grid nodes
lie outside the ln-3 window by design (no regularisation).

| leg | delivered pts | train front | test front | HV central train | HV central test |
|-----|---------------|-------------|------------|------------------|-----------------|
| adaptive CCP | 106 | 40 | 41 | 0.3732 | 0.2618 |
| baseline r=4 | 106 | 40 | 46 | 0.5994 | 0.4887 |
