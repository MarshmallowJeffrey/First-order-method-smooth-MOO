# Fixed-budget comparison (single strict instrument, without-256 track)

Produced by `Original_py/run_fixed_budget_K6_without_256_checkpoints.py`
(July 26, 2026, session 12; design record `Note/Jul_26_note.md` part 2).

Protocol: every completed baseline configuration (r, node_tol) is one
POINT — x = its realized cost, y = the strict full-simplex 64-start
delivery audit from its own summary.  The adaptive method is ONE
budget-mode run (B = 80,912 grad-equivalents, stop =
`max_grad_evals`; stop_reason=`budget`); its curve is the
strict 64-start GN* of the bundle PREFIX at each checkpoint, audited
post-hoc (audit cost 161.2s, OFF both axes).
One instrument everywhere; every abscissa is an equal-budget read.

In-run searches are TARGETING only (tier=strict, 24
starts, time inside the adaptive CPU axis); the meter is always the
64-start audit.  Inner loop: Momentum-SVRG (b=4096,
step_const=0.1, beta=0.5,
rel_target=0.05, max_segments=10);
segment-cap rounds are acceptable in budget mode (budget burns, point
still delivered) — cap rounds this run: 414.

Adaptive endpoint: audited GN* 1.0209e-01 at
80871 grad-equivalents /
2907.1 s (bundle m=4318,
post-prune audit 1.0826e-01).

## Baseline points (loaded, not re-run)

| r | node_tol | N nodes | grad-equiv | wall s | strict audit | stop |
|---|----------|---------|------------|--------|--------------|------|
| 10 | 0.02 | 3,003 | 41327 | 599 | 1.6352e-01 | completed |
| 12 | 0.02 | 6,188 | 64428 | 986 | 9.5415e-02 | completed |
| 15 | 0.02 | 15,504 | 80912 | 1173 | 5.9456e-02 | completed |
| 20 | 0.02 | 53,130 | 241721 | 3689 | 6.3415e-02 | completed |
| 10 | 0.01 | 3,003 | 55416 | 882 | 1.4993e-01 | completed |
| 15 | 0.01 | 15,504 | 254197 | 3628 | 5.8878e-02 | completed |

## Figures

- `fixed_budget_gn_vs_grads.png` — grad-equivalents axis (headline).
- `fixed_budget_gn_vs_cpu.png` — CPU axis (adaptive pays its search
  time on-axis; baseline points include their solve time).

Annotations "xN.N vs curve" = baseline audit / adaptive audited value
at the SAME abscissa (>1: adaptive better at that budget).
"beyond budget" = the point lies past the adaptive run's end.

The plotted adaptive curve is the MONOTONE LOWER-BOUND ENVELOPE of the
raw prefix audits (at each checkpoint, the max over that and all later
audits).  Valid because the true prefix GN* is non-increasing in m and
every audit is a lower bound; raw per-checkpoint audits stay in
`summary.json` (`audited_gn_history`).

## Caveats

Single instance (seeds 7/8/41), single machine,
serial runs.  Audits are heuristic lower bounds of an NP-hard max
(64 starts, warm-started); under-search can only under-report — and it
is the SAME instrument for both methods.  MLP torch runs are not
bit-reproducible in this environment (session-12 finding); trajectories
are one realization each.
