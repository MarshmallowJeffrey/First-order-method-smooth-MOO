# Note — Soundness fixes to the Adaptive Bundle Algorithm

Fixes applied: July 4, 2026 · This note: July 5, 2026
Files changed: `Original_py/algorithm.py`, `Original_py/baseline.py`, `Original_py/objectives_numpy.py`
Companion record: `PYTHON_CHANGES.md` → "Soundness fixes — July 4, 2026"
Public APIs unchanged, except two new keys in `algorithm_adaptive`'s result dict: `L_scale_final`, `inner_cap_hits`.

---

## 1. Descent-lemma safeguard — fixing the smoothness-constant problem

### Why L matters here

The bundle method uses the smoothness constants L in two places in the algorithm, not one:

* **Step size.** For weighting λ, the scalarized objective F_λ = Σ λ_k F_k is L_λ-smooth with
  L_λ = Σ λ_k L_k, and the T-map steps with size 1/L_λ.
* **Point selection** — the bundle-specific part. The T-map scores every bundle point by
  u_i = F_λ(x_i) − ‖∇F_λ(x_i)‖² / (2 L_λ) — the certified post-step value if you stepped from
  x_i — and steps from the argmin, ties broken by lowest index (`algorithm.py:100-129`).

L also enters the theory: the paper's inner-loop iteration bound is
M_t ≤ ⌈3 · C_λ · L_λ / ε⌉ with C_λ ≥ 2·[F_λ(x₁) − F_λ*] (Corollary 4.1 + Appendix A.1).

So L is not just a step size; it is the yardstick by which the algorithm predicts the future and
chooses where to work.

Crucially, the theory only needs L to be an **upper bound** on the true curvature. Overestimate it
and you take smaller steps — slower, but every guarantee still holds. Underestimate it and the
guaranteed-decrease formula becomes a lie the algorithm keeps believing.

### The frozen loop (the failure this fix removes)

With L too small, the step 1/L_λ is too big, so the step can land **higher** than the promised
value u*. Here is the vicious part: the selection rule ranks points by their promise, and an
understated L inflates every promise. The new (bad) point's promise is **no better than the old
anchor's — worse, or exactly tied, and ties break to the lowest index, i.e. the older point**
(that tie-break is what locks the reproduced 2-cycle). So the next iteration re-selects the exact
same anchor, takes the exact same step, and appends a byte-identical point to the bundle —
forever. Each duplicate still costs K gradient evaluations (`algorithm.py:774`). Nothing errors,
because nothing fails numerically; the run just silently burns its entire budget making no
progress. One reproduced run had 375 copies of one point.

This is the mechanism that froze the flagship `mlp_crossover_h64x64/h80x80/h96x96` and
`run_plateau8` notebook runs for hours at their first adaptive checkpoint (their output
directories stayed empty), while the baseline stage of the same notebooks completed.

### Where the wrong L came from (two separate causes in the MLP testbed)

* **The quantity does not exist: ReLU.** A ReLU MLP's gradient is discontinuous across every
  activation-kink hyperplane, so no finite global gradient-Lipschitz constant exists; any finite
  estimate is an estimate of an infinite quantity, and it stays finite only because random probes
  rarely straddle a kink at small separation. (In addition, the logits are bilinear in the
  weights, so the Hessian grows with ‖θ‖ — global L-smoothness fails even ignoring kinks.)
* **The probes measured the wrong region.** The code estimated L empirically: 40 random pairs of
  nearby parameter vectors *per objective* (t1 = 0.5·randn(d), t2 = t1 + 0.1·randn(d)), take the
  worst gradient-difference ratio, double it for safety (`objectives_torch.py:299-314`; same
  scheme in the NumPy backend). Those probes live at a scale of ~0.5 per coordinate, while the
  actual optimization starts at He initialization — per-layer std roughly 0.14–0.32 for the
  crossover nets, ≈ 0.18 for the width-64 hidden layers, hidden biases 1e-2, output bias 0
  (`experiments.py:123-163`) — and then wanders along its own trajectory. Measuring curvature
  where the probes happened to land tells you little about curvature where the optimizer
  actually walks.

(A third instance — an analytically mis-derived constant in the logistic-regression testbed — is
covered in §2.)

### The fix (in `algorithm.py`)

* **A runtime safety check — the important one.** After every inner step, the code compares the
  actual new value F_λ(x_new) against the promised value u*. Both numbers were already being
  computed, so the check is free (`algorithm.py:509`, relative tolerance 1e-10·(1+|u*|)). If the
  promise is broken, that is mathematical proof that L was too small, so the algorithm doubles a
  global multiplier `L_scale` on L (halving future steps), issues a Python `RuntimeWarning` once,
  and records the final multiplier in the results (`L_scale_final`). If it has to double past
  2⁶⁰, it raises a `RuntimeError` with a clear message that the objectives simply are not
  L-smooth along the iterates (`algorithm.py:737-757`).
* **Fully deterministic and self-adjusting** — nothing random, nothing you pick. It starts at 1.
  After each step: promise kept → leave it alone; promise broken → double it. It only ever
  increases. So the final value is always a power of 2: the first one large enough that the
  scaled L stopped being violated **along the actual trajectory**. "Settled at 2" decodes as:
  exactly one violation occurred during that run — the probe estimate proved too small once, was
  doubled, and 2× the probe was never contradicted again.
* **It is a global multiplier** — one scalar multiplying the whole vector of K constants at once.
  Conservative: if only one objective's constant was too low, everyone's steps shrink; still
  valid, just slightly slower for the others.
* **The violating point stays in the bundle.** Its oracle evaluation is already paid for, and
  under a valid (rescaled) L a bad point is simply never re-selected.
* This is faithful to the paper, which itself sketches online L re-estimation (its Eq. 22) that
  was never implemented.

### Scope and suggestion

* The safeguard protects the **adaptive method only**. The baseline's fixed-step 1/L_λ gradient
  descent still trusts the supplied L with no check. It cannot freeze (it has no selection rule),
  but under the same bad L its subproblem solves can silently take invalid steps — keep this in
  mind when reading head-to-head comparisons on ReLU MLPs.
* One honest caveat: on ReLU networks the warning will fire in essentially every run, because no
  finite L exists — the safeguard adapts to the curvature along the actual trajectory, which is
  the best any first-order method can do there. Smooth activations (tanh, GELU, softplus) in the
  test problems remove the kink problem and restore C^∞ — but note that even then an MLP is not
  *globally* gradient-Lipschitz (bilinear logits ⇒ the Hessian grows with ‖θ‖); smoothness then
  holds on bounded regions, which is the standard regime for local analyses.

---

## 2. Corrected logistic-regression smoothness constant (`objectives_numpy.py`)

*(This fix was part of the same changeset and belongs in any summary of it.)*

The strongly-convex logreg testbed's analytic constant was provably wrong on two counts:

* it used the factor **1/4** — the binary-sigmoid Hessian bound — where the K-class softmax
  needs **1/2** (λ_max(diag(p) − ppᵀ) ≤ 1/2);
* it used the spectral norm of the **full** data matrix X, where F_i's data term only involves
  the class-i rows X_i.

Old: `L_i = ‖X‖² / (4 n_i) + reg`. This can undershoot the true constant (e.g. under class
imbalance, where ‖X_i‖ → ‖X‖), and the resulting too-large steps reproduced an optimization
stall at the 6.665e-04 floor.

New: `L_i = ‖X_i‖² / (2 n_i) + reg` (`objectives_numpy.py:154-157`) — a valid upper bound,
verified against numerical Hessians (max eig(H)/L over probes = 0.9449; was 1.454 with the old
formula, i.e. the old "constant" was genuinely below the true curvature).

Note: the current experiment drivers use the MLP testbed, not this problem — but this fix matters
for two reasons. It shows the wrong-L failure was not only a probe-estimation issue (an
*analytic* constant was mis-derived too), and while the §1 safeguard would rescue such a run at
runtime, the corrected formula restores valid theory constants without any rescaling.

---

## 3. λ-search consistency (`_maximise_GN`)

### The problem

* IPOPT and SLSQP obey the constraints ("sum = 1", "each component ≥ 0") only approximately, up
  to a numerical tolerance — and when a solve stops early or fails, the violation can be bigger.
  So the solver's final answer `res.x` might be, for example, λ = (0.55, 0.65), whose sum is 1.2.
  That point is outside the simplex; it is not a legal weighting.
* GN(λ) is homogeneous of degree 2 in λ: multiply λ by a constant c and the value multiplies by
  c². So the illegal λ with sum 1.2 reports a value 1.44× larger (1.2² = 1.44) than the true
  value at the legal, normalized version of the same direction. And the search compares many
  candidates and keeps the largest — inflated illegal candidates tend to win the competition
  unfairly.

### What the old code did wrong

It scored `res.x` raw (illegal point, inflated value), let it win, and normalized λ only at the
very end. Result: the number it returned was measured at one point, but the λ it returned was a
different point. That is what "the pair (value, λ) was inconsistent" means: value ≠ GN(λ) for the
very λ in the same result.

### The fix (`algorithm.py:314-327`, applied at lines 330, 332, 370-371)

Before scoring **any** candidate — the starting points and the solver outputs — first project it
onto the simplex: clip negative components to 0, then divide by the sum so it is exactly 1
(centroid fallback on degenerate input). Now every score is measured at a legal point, and the
returned value is literally GN evaluated at the returned λ. The monotone-in-starts guarantee
(start points are scored before their solves, so a failed solve never loses ground) is preserved.
The docstring at `algorithm.py:317-321` states exactly this reason.

Residual, by design: the maximization itself remains a multistart **heuristic lower bound** of an
NP-hard non-concave max. This fix makes the answer internally consistent; it does not make it
globally optimal (see §7).

---

## 4. Warm-start ordering in the baseline (snake / boustrophedon)

### What the old code did

It generated all the grid weightings and visited them in dictionary (lexicographic) order via
`np.lexsort`, warm-starting each subproblem from the previous subproblem's solution. Its own
docstring claimed consecutive points are ℓ₁-close (≤ 2/r apart).

### What was wrong with that

Dictionary order is fine for K = 2, but for K ≥ 3 it has "carry" moments — like counting
099 → 100, where several digits change at once. At those moments the next λ is far from the
previous one, so the warm start begins from a nearly opposite trade-off and is useless there.
Two consequences:

* (a) it violates the ≤ 2/r consecutive-adjacency assumption behind the paper's cost analysis
  for the baseline (Algorithm 1's enumeration guarantee) — an assumption the old code's own
  docstring asserted;
* (b) it wastes the baseline's work: at K = 6, r = 9, **494 of 2001 consecutive hops (~25%)
  broke the ≤ 2/r bound**, with jump sizes ranging up to the simplex diameter ℓ₁ = 2.0 (only a
  handful of the outermost carries hit the full 2.0; the rest are intermediate sizes). A baseline
  that is accidentally weakened also makes the comparison against the adaptive method unfair.

### How the new code solves it

`_snake_compositions` (`baseline.py:57`) builds an ordering that walks through all the integer
splits moving exactly one unit at a time. The trick: it counts the first coordinate upward, and
enumerates the remaining coordinates forward on even blocks and backward on odd blocks — back and
forth, like plowing a field row by row — so that neighboring blocks meet at adjacent points.
`_sort_grid_for_warmstart` (`baseline.py:76`) then reorders the existing grid to follow that
path. Every hop is now exactly 2/r; this was checked exhaustively for every K from 2 to 7.
Nothing else changed — same grid points, same solver, only the visiting order.

---

## 5. ε-mode honesty (no more silent early stops)

### The fix, in three pieces

1. `_bundle_update_adaptive` now reports back whether the job was finished: it returns
   `target_met` (`algorithm.py:449-450` — `None` in budget mode, `True`/`False` in ε-mode).
   Before the fix the function returned only the number of steps; the concept "did I actually
   reach the target?" did not exist anywhere in the code, so the final certificate could be
   false with no visible sign.
2. The main loop checks it (`algorithm.py:758-770`): if `target_met` is `False`, it issues a
   `RuntimeWarning` once, with a clear message — the termination argument of Algorithm 2 does not
   apply to that round; raise `max_inner` if you want a certified run.
3. Every such event is counted in `inner_cap_hits` and returned in the result dictionary
   (`algorithm.py:809`).

### Why the silent early stop only matters in ε-mode

In budget mode, stopping at the cap is exactly what was asked for — there is no target, so
nothing can be "missed." In ε-mode, if the safety cap fires before the ε/3 target is reached,
that round did not do its required job — but the old code continued exactly as if it had.

### Usage rule — necessary conditions, not sufficient

After an ε-mode run, look at `result["inner_cap_hits"]`:

* **> 0** → the certificate is not trustworthy; increase `max_inner` and run again.
* **== 0** → the inner-loop part of the termination argument holds, **provided**
  `prune_inner=False` — pruning discards inner candidates and breaks the full-bundle condition
  the ε-proof assumes (the code warns about this separately). And even then, the outer stopping
  test GN* < (2/3)ε trusts the multistart λ-maximization, which is a heuristic lower bound of an
  NP-hard max — if the maximizer undershoots, the run can stop believing coverage it does not
  have. Treat the certificate as heuristic-grade unless the λ-max is strengthened.

Also for context: the shipped experiment drivers never pass `epsilon` — everything still runs in
budget mode, so ε-mode currently only matters for future certified runs.

---

## 6. Documentation corrections (also part of the changeset)

* Stale draft references fixed: Algorithm 6 → Algorithm 2, Eq. 13 → Eq. 10,
  Appendix B.1 → A.1, removed a stale Eq. 17 reference.
* `pc_star`'s docstring no longer claims it uses "the same" maximizer as the outer loop; it now
  documents the deliberate fixed-strength yardstick design (IPOPT, 256 multistarts, independent
  of the run's λ-solver settings, so it is comparable across methods and configurations) and
  states that the value is a heuristic lower bound on an NP-hard maximum, not a certificate.

---

## 7. What these fixes do NOT change

* GN* and the ε-certificate remain heuristic lower bounds (the exact max is NP-hard; multistart
  local solves).
* Experiment drivers still run budget mode, never ε-mode.
* Flagged experimental-design biases were left as-is: the time-to-target asymmetry
  (`target_cov` = the baseline's final value, while the baseline is always charged its full
  schedule), `detect_plateau`'s level = median of raw (non-monotone) values, and the
  per-checkpoint 256-start IPOPT `pc_star` cost being excluded from the reported CPU axes.
* Evidence status: the plateau claim has real small-K support; the CPU-time-crossover claim has
  **zero valid data points** until the three crossover notebooks are re-run on the fixed code.

---

## 8. How the fixes were verified

* `verify_fixes.py`: **10/10 pass** — the wrong-L 2-cycle now converges to 0.0 (L_scale settles
  at 2; duplicate points 39 → 0), a 3×-understated L converges to 1.5e-05 with no crash, the
  corrected logreg L bounds the numerical Hessian (ratio 0.9449, was 1.454), the theory-faithful
  no-prune run passes its old 6.665e-04 stall floor (reaching 4.8e-05), the ε-stop fires on a
  healthy problem (13/40 outers), value/λ are consistent on both solver backends, and snake
  adjacency is exact (worst jump 0.2222 = 2/9 at K=6, r=9; lex sort gave 2.0).
* End-to-end tiny plateau comparison (K=3, p=4, n=60, h=8, 3000 gradients): completes cleanly;
  adaptive GN* falls 4.50 → 0.0303, strictly decreasing (it froze at the first checkpoint
  pre-fix), 4.8× below the baseline's best; the safeguard fired once (L_scale_final = 2.0). A
  deliberately crippled L/10 run rescued itself automatically (L_scale_final = 64, final GN*
  1.37× of normal, no crash).
* An independent three-lens review of the fixes found no critical or major defects introduced.