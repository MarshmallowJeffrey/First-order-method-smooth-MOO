# Pure fixed-budget comparison (no tolerance inputs; without-256 track)

Produced by `Original_py/run_pure_budget_K6_without_256_checkpoints.py`
(July 27, 2026).  Protocol (user-designed): every leg spends the SAME
budget B = 400 grad-equivalents in identical work units
(one segment = 5 minibatch Momentum-SVRG steps, no trigger,
+ 1 full joint evaluation), s = 2 consecutive segments per
allocation decision, chain warm start for BOTH methods.  The ONLY
difference between methods is the next-lambda policy: snake grid order
(baseline, per r) vs strict worst-lambda search (adaptive,
8 starts, decision time inside its CPU axis).
No node_tol / solve_target / epsilon / rel_target anywhere; stop =
budget.  Quality appears only in the post-hoc strict 64-start audits
(adaptive curve: prefix audits, monotone lower-bound envelope).

| r | s | distinct lambdas visited | grad-equiv | wall s | strict audit |
|---|---|--------------------------|------------|--------|--------------|
| 2 | 1 | 21 | 388 | 0 | 5.1296e-01 |
| 2 | 2 | 21 | 388 | 0 | 4.2658e-01 |
| — | 2 | 11 | 396 | 3 | 5.4683e-01 |

(last row = adaptive; "distinct lambdas" for the baseline = grid nodes
actually visited = its coverage at this budget.)

Sensitivity legs (s=1, open circles) let the baseline run at its own
preferred decision granularity — its decisions are free, so s>1 is a
constraint it inherits from the shared-chunk design; disclosed.

CPU-axis caveat: adaptive decision time (strict searches over a growing
Gram stack) is method work, on-axis; audits are off-axis for everyone.
MLP torch runs are not bit-reproducible in this environment
(session-12 finding); one realization each.
