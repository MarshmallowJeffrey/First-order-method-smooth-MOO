# bandit_toy_mv_without_256_checkpoints — mean-variance K=2 bandit toy

July 31, 2026.  The July-26 K=2 offline-bandit toy
(`bandit_toy_surf_without_256_checkpoints`) with the reward term
replaced by a MEAN-VARIANCE utility; everything else (instance data,
MSVRG pair, eps ladder, equal-level stop at 2eps/3, figure set,
summary layout) is the July-26 pipeline verbatim.  Change record:
`Note/Jul_31_note.md`; mathematics: `objectives_bandit_toy_mv.py`;
driver: `run_bandit_toy_mv_without_256_checkpoints.py`.

Objective (gamma recorded in every summary.json; record runs use
gamma = 1.0, chosen from `gamma_scan.json` — see the note):

    F_k(theta) = tau*KL(pi||pi_ref) - <pi, Rk_hat> + gamma*VarHat_k(pi),
    VarHat_k(pi) = <pi, Sk_hat> - <pi, Rk_hat>^2,

pi = softmax([theta, 0]).  gamma = 0 reproduces the July-26 objective
bit-for-bit.  The -gamma*<pi,Rk_hat>^2 terms are concave in pi: the
closed-form softmax oracle is DEAD, scalarizations are nonconvex, and
at gamma = 1 the reference front JUMPS at w ~ 0.665 (the global
optimum teleports between a commit-to-arm-4 and a commit-to-arm-5
basin; both basins persist on the upper w range).

Ground truth semantics (READ BEFORE QUOTING VALUE METRICS): the
"reference PF" and every delta_value(w) are measured against a
NEVER-TIMED multi-start + path-relaxation reference table — a
best-known pool, NOT a certificate.  After both methods run, every
method-delivered point seeds a polish at every evaluation weight; any
improvement > 1e-9 updates the table (counted in
`summary.json: mean_variance.reference_refresh`).  Value gaps are
therefore >= -1e-9 by construction, and "delta_value = 0" means
"matches the best solution ANY procedure found", not "matches the
global optimum".  GN* metrics are unchanged in meaning (strict
in-family 64-start search, a heuristic lower bound of an NP-hard max,
off both cost axes).

The SURF Eq.-9 speed formula died with the closed form: the arc CDF
and the Rule-1 arc-uniform reference weights use the CHORDAL arc
length of the reference front (a front jump concentrates reference
points at the jump — deliberate, that is where the structure lives).

Layout:
* `eps1e-2/`, `eps1e-3/` — record rungs (gamma = 1.0,
  --eval-every 0 per-segment cadence, both rungs run serially in ONE
  invocation and scored against the SAME final reference).
* `smoke/` — tiny-fuse pipeline validation (gamma = 1.0 default).
* `gamma_scan.json` — nonconvexity evidence per candidate gamma
  (mind the theta-plateau caveat in the note: quote front jumps, not
  raw pool basin counts).
* `reference_gamma*_n*.npz` — cached reference tables (plug-in +
  true-parameter twins).
* `record_run_console.log` — console of the record invocation.

Machine-conditions caveat (July 31 record runs): third-party load was
present (1-min loadavg 3.3-11.3; settle gate did not settle) — CPU-axis
readings (fig1, cpu-to-eps, wall/process seconds) are estimates only;
grads-axis readings, GN*, value metrics and fronts are load-independent
and record-grade.  A clean re-run supersedes the CPU numbers.
