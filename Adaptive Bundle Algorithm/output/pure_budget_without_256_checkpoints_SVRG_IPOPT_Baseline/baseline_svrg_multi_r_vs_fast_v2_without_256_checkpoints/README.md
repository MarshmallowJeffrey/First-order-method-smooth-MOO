# Baseline r-sweep vs fast adaptive — v2 comparison home (without-256 track)

Created July 26, 2026 (session 12; design record: `Note/Jul_26_note.md`,
part 2).  This folder collects the July-26 re-presentation and the two
extensions of the K=6 MLP baseline-vs-fast comparison in one place.
The original experiment folder
`output/baseline_svrg_multi_r_vs_fast_without_256_checkpoints/` remains
in place and untouched; `original/` below is a verbatim copy of it.

## Layout

- `original/` — verbatim archive of the original folder (node_tol 0.02,
  r in {10, 12, 15, 20}; figures in the July-25 comparable-meter
  presentation).
- `tol0.02/` — the SAME four runs (summaries copied, nothing re-run);
  figures REDRAWN in the July-26 presentation: baseline lines on their
  NATIVE grid meter (`cov_history`), endpoint circle = the run's own
  grid certificate, x marker = a SEPARATE strict full-simplex delivery
  audit, dotted connector = the measured between-node error.
- `tol0.01/` — fresh baseline runs at node_tol 0.01 (solve_target
  0.0025 = tol/4), r in {10, 15}: how do the trajectory and the end
  audit move when the node certificate is tightened?  (Created when the
  runs land.)
- `adaptive_extended/` — the fast adaptive re-run with max_outer raised
  500 -> 2000.  v3 stopped at round_fuse with best 0.0581 vs eps=1e-3;
  its best-so-far had been flat from grad-equivalent 5413 to 9226, so
  this run tests the round-cap hypothesis against the variance-floor
  hypothesis flagged in `Note/Jul_16_note.md`.  (Created when the run
  lands.)

## Meter discipline (track rules unchanged)

No external 256-start yardstick anywhere.  Baseline trajectory lines
speak the baseline's own contract (max over grid nodes of best-known
value); the fast curve speaks its own cheap-tier search; ONLY the x
audit markers are strict-tier full-simplex scores.  Cross-curve reads
are strict at the audit markers, indicative elsewhere.
