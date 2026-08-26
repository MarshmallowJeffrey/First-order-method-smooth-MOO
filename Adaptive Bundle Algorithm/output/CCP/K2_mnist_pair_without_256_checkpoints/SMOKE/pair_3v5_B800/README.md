# K2 MNIST pair 3 vs 5 — pure fixed budget campaign (Aug 13, 2026)

Legs: baseline grids r in [4] + adaptive CCP, all at s = 2,
B = 800 grad-equivalents, batch 256,
per_class = 300 (balanced maximum), d = 8098.
Quality: EXACT 1-D meter at every checkpoint (certified).  Test values:
ALL official t10k rows of digits 3 and 5.  Figures: gn_vs_grads,
gn_vs_cpu, fronts_train_test (window <= ln 2), front_err_test,
test_ce_vs_budget.  Divergence arms of vertex grid nodes lie outside
the ln-2 window by design (no regularisation — see Note/Aug_13_note.md).

| leg | delivered pts | train front | test front | HV central train | HV central test |
|-----|---------------|-------------|------------|------------------|-----------------|
| adaptive CCP | 114 | 28 | 28 | 0.3218 | 0.3006 |
| baseline r=4 | 132 | 21 | 22 | 0.2608 | 0.2385 |
