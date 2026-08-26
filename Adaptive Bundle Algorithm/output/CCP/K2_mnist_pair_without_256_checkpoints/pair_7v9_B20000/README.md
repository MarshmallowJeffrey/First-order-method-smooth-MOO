# K2 MNIST pair 7 vs 9 — pure fixed budget campaign (Aug 13, 2026)

Legs: baseline grids r in [10, 20, 40] + adaptive CCP, all at s = 5,
B = 20,000 grad-equivalents, batch 1024,
per_class = 5949 (balanced maximum), d = 8098.
Quality: EXACT 1-D meter at every checkpoint (certified).  Test values:
ALL official t10k rows of digits 7 and 9.  Figures: gn_vs_grads,
gn_vs_cpu, front_train + front_test (each showing adaptive CCP vs the
best baseline only, window <= ln 2 — Aug-13 user revision),
front_err_test, test_ce_vs_budget.  Divergence arms of vertex grid
nodes lie outside the ln-2 window by design (no regularisation — see
Note/Aug_13_note.md).

| leg | delivered pts | train front | test front | HV central train | HV central test |
|-----|---------------|-------------|------------|------------------|-----------------|
| adaptive CCP | 3,266 | 98 | 207 | 0.4804 | 0.4765 |
| baseline r=10 | 3,476 | 72 | 333 | 0.4804 | 0.4769 |
| baseline r=20 | 3,372 | 82 | 309 | 0.4804 | 0.4761 |
| baseline r=40 | 3,318 | 110 | 293 | 0.4804 | 0.4769 |
