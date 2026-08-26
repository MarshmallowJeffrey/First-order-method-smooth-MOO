# Pure fixed-budget comparison (no tolerance inputs; without-256 track)

Produced by `Original_py/run_pure_budget_K6_without_256_checkpoints.py`
(July 27, 2026).  Protocol (user-designed): every leg spends the SAME
budget B = 80,912 grad-equivalents in identical work units
(one segment = 13 minibatch Momentum-SVRG steps, no trigger,
+ 1 full joint evaluation), s = 5 consecutive segments per
allocation decision, chain warm start for BOTH methods.  The ONLY
difference between methods is the next-lambda policy: snake grid order
(baseline, per r) vs strict worst-lambda search (adaptive,
24 starts, decision time inside its CPU axis).
No node_tol / solve_target / epsilon / rel_target anywhere; stop =
budget.  Quality appears only in the post-hoc strict 64-start audits
(adaptive curve: prefix audits, monotone lower-bound envelope).

| r | s | distinct lambdas visited | grad-equiv | wall s | strict audit |
|---|---|--------------------------|------------|--------|--------------|
| 10 | 1 | 3,003 | 80895 | 1149 | 1.1144e-01 |
| 10 | 5 | 1,180 | 80898 | 1247 | 3.1783e+00 |
| 12 | 5 | 1,165 | 80898 | 1194 | 7.0916e+00 |
| 15 | 1 | 5,431 | 80894 | 1116 | 4.9037e+00 |
| 15 | 5 | 1,170 | 80907 | 1256 | 7.0916e+00 |
| 20 | 5 | 1,181 | 80898 | 1113 | 7.0916e+00 |
| — | 5 | 862 | 80902 | 3782 | 4.6160e-02 |

(last row = adaptive; "distinct lambdas" for the baseline = grid nodes
actually visited = its coverage at this budget.)

Sensitivity legs (s=1, open circles) let the baseline run at its own
preferred decision granularity — its decisions are free, so s>1 is a
constraint it inherits from the shared-chunk design; disclosed.

CPU-axis caveat: adaptive decision time (strict searches over a growing
Gram stack) is method work, on-axis; audits are off-axis for everyone.
MLP torch runs are not bit-reproducible in this environment
(session-12 finding); one realization each.
