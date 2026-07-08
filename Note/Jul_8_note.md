# July 8, 2026 — Baseline certification mode (`node_tol`)

Scope: ONE change. `baseline.py`'s `uniform_discretisation` gains an
optional certification mode (per-node tolerance stopping), in preparation
for future ε experiments. No experiment design, no experiment runs, no
changes to `run_experiments.py`'s two official sweeps. Companion records:
`Note/Jul_5_note.md` (July 4 soundness fixes),
`Adaptive Bundle Algorithm/Python_Change.md/PYTHON_CHANGES.md` (earlier
change history), `Note/Jul_6_note.md` (July 6 paper-conformance fixes).

## 1. Motivation

The adaptive method (`algorithm.py`, `algorithm_adaptive`) has had an
`epsilon` parameter from the start (outer stop at 2ε/3, inner target ε/3),
i.e. it can run in a certified mode: "stop when the quality criterion is
provably met". The baseline (paper's Algorithm 1) had no counterpart: it
could only stop on a budget (`max_grad_evals`) or a pass schedule
(`n_passes`), never on a per-node quality criterion. Any future ε-style
comparison ("what does it COST each method to certify a given tolerance?")
therefore had no baseline side to stand on. This change adds it.

## 2. What was added

New optional parameter of `uniform_discretisation`:

    node_tol: Optional[float] = None

Meaning: the per-node acceptance level. A grid node i whose own weighting
λ_i satisfies ‖∇F_{λ_i}(x_i)‖² ≤ node_tol counts as "served".

Semantics when `node_tol` is set (certification mode):

1. **Entry check, one per visit.** At each visit to a node, BEFORE any
   step is taken, the gradient that the first gradient-descent step would
   consume anyway is checked. If its squared norm is ≤ node_tol, the node
   is marked served and NOT moved; the visit ends there. If not, the same
   gradient is used for the first step exactly as before — the check never
   causes an extra oracle call.
2. **Served nodes are frozen.** Later passes skip served nodes entirely
   (no oracle calls, no movement). Because a served node's point never
   moves again, the mark stays valid forever — no re-checks needed.
   (A mark can only be set during a visit, so the skip can never interfere
   with the pass-1 warm-start chain.)
3. **Stopping.** The run ends as soon as ALL nodes are served; the
   cumulative gradient-evaluation count at that moment is the
   certification cost (`certified_grad_evals`). `max_grad_evals` is kept
   as a fuse: if it (or the pass schedule) runs out first, the run stops
   and reports certification failure honestly.
4. **Accounting (paper protocol unchanged).** Every algorithm-side
   gradient evaluation counts toward `grad_evals`, including the one that
   triggers a mark (it is real oracle work). The check itself only reuses
   that gradient (one extra dot product, never an oracle call). On
   success, a final checkpoint is recorded as usual, so `cov_history[-1]`
   is the measured GN* of the delivered point set at certification time;
   checkpoint-metric cost stays excluded from both reported axes, exactly
   as before.

With `node_tol=None` (the default) the code path is behaviourally
identical to the previous budget mode — verified bit-for-bit (§4).

## 3. New result-dictionary fields

All pre-existing keys are unchanged. New keys (all None when
`node_tol=None`):

- `node_tol` — the acceptance level used (None = budget mode).
- `certified` — True iff every node was served before the fuse/schedule
  ran out; False on failure; None when the mode is off.
- `certified_grad_evals` — cumulative gradient evaluations at the moment
  the last node certified (the certification cost); None on failure.
- `node_served` — list of N booleans, service status per node.
- `node_grad_sq` — list of N floats: last measured ‖∇F_{λ_i}‖² per node.
  For a served node this is the certifying value at its frozen point
  (exact from then on). For an unserved node it is the squared norm of the
  gradient consumed by its most recent step, i.e. it lags that node's
  final position by one step — reported as-is rather than spending extra
  oracle calls on a fresh sweep. NaN = the node was never visited.
- `unserved_nodes` — list of node indices not yet served (empty on
  success). This is the "fuse blew" diagnostic.

Also updated: the module docstring (the old sentence "The inner loop does
NOT stop based on any per-point tolerance" now describes the two modes)
and the function docstring (parameter + returned keys + semantics).

## 4. Verification (all on K=3, p=6, n=30, hidden=[4], tanh, seeds 7/8, r=6)

1. **Regression, `node_tol=None`.** A fixed small configuration
   (budget 3,000, checkpoints every 600) was run twice BEFORE the change
   (run-to-run deterministic, including the metric) and once AFTER:
   `grad_evals_history`, `total_iters_history`, `cov_history`, and the
   SHA-256 of `final_solutions` are all IDENTICAL to the pre-change run.
   The only difference is the six new keys, all None.
2. **Certification smoke tests** (24 checks, all passed):
   - Loose tolerance (1e6): every node certifies at its first visit;
     total cost exactly N·K = 28·3 = 84 gradient evaluations (one check
     eval per node); no point moved; final checkpoint measured the
     delivered set.
   - Moderate tolerance (1e-2): certification success at 909 gradient
     evaluations (strictly between 84 and the fuse); history tail equals
     the certification cost; all served nodes' recorded values ≤ tol;
     spot-checked nodes' gradients recomputed independently at their
     frozen points match the recorded certifying values exactly (proof
     the freeze works).
   - Strict tolerance (1e-12) + small fuse (1,000): honest failure —
     `certified=False`, no certification cost, stop at 999 evals (the
     pre-existing budget-check granularity), consistent
     `unserved_nodes`/`node_served` bookkeeping.
   - Input validation: node_tol of 0, negative, inf, NaN, and bool are
     rejected with ValueError.
3. **Full suite:** `ledger-artifacts/verify_fixes.py` — 10 passed,
   0 failed (unchanged).

## 5. What this change deliberately does NOT do

- No change to `run_experiments.py` or either official sweep; no
  experiment in `EXPERIMENTS.md` uses `node_tol` (a one-line pointer was
  added to its §6 `epsilon` entry, and to the `baseline.py` rows of
  `MANUAL.md`/`MANUAL_ZH.md`).
- No ε/δ/r value choices and no ε-experiment design — explicitly out of
  scope for this change; to be planned separately.
- No mid-visit acceptance checks: the check runs once per visit, at
  entry, per the task specification. A node that crosses the tolerance
  during its steps is caught at its next visit's entry check (cost: K
  evaluations), which slightly overstates certification cost in exchange
  for zero extra oracle calls and unchanged step behaviour.
