# Note — Paper-conformance review and fixes

Review performed: July 6, 2026 · This note: July 6, 2026
Scope: line-by-line comparison of the implementation against the paper draft
("A First-Order Bundle Method for Smooth Non-convex Multi-objective Optimization",
extracted text at `Adaptive Bundle Algorithm/tmp/pdfs/paper.txt`).
Files changed: `Original_py/objectives_numpy.py`, `Original_py/objectives_torch.py`,
`Original_py/experiments.py`, `Original_py/baseline.py` (docstring only),
`Original_py/bundle.py` (docstring only).
Public APIs unchanged except one new optional keyword (`activation`, §3b) with
a backward-compatible default. No result-dictionary keys changed.

Companion records: the July 4 soundness fixes are documented in
`Note/Jul_5_note.md` and `PYTHON_CHANGES.md`; this note covers the *remaining*
paper-vs-code differences found in a fresh July 6 review, after those fixes.

---

## 0. Verdict of the review

The algorithmic core matches the paper:

* **T-map (Eq. 10)** — `_T_map_batched` selects
  i* = argmin_i {F_λ(x_i) − ‖∇F_λ(x_i)‖²/(2L_λ)} with ties broken to the
  lowest index, then steps x_{i*} − (1/L_λ)∇F_λ(x_{i*}). Matches Eq. 10
  including the tie-break rule.
* **Progress criterion** — GN(λ;B) = min_i ‖∇F_λ(x_i)‖² as in Section 4;
  the λ-gradient used by the maximiser is the Danskin gradient.
* **Algorithm 2 loop** — outer step maximises GN over the simplex, stops at
  GN* ≤ 2ε/3 in ε-mode, runs BundleUpdate at the maximising λ with inner
  target ε/3 and every T-map iterate appended (the proof's semantics);
  budget mode (`epsilon=None`) reproduces the experiment protocol of the
  paper's Section 5 (fixed inner step count, checkpoint cadence by gradient
  count). The paper's strict inequalities ("< 2ε/3", "< ε/3") are
  implemented as "≤"; a boundary-only difference with no practical effect.
* **Inner-loop pruning** — `prune_inner=True` implements the Section 5
  implementation note (keep only the candidate with the smallest
  ‖∇F_λ‖); `prune_inner=False` restores the proof-faithful full bundle.
* **Algorithm 1 baseline** — uniform simplex grid G_r, snake enumeration
  satisfying the ‖λ^(g+1)−λ^(g)‖₁ ≤ 2/r adjacency requirement, first-pass
  chain warm-starting and per-node continuation afterwards, exactly as the
  paper's "Warm-starting" paragraph describes.
* **Cost accounting** — one gradient evaluation per ∇F_k call, so one
  scalarised step costs K evaluations; checkpoint metric work (the fixed
  256-start IPOPT GN* solve) is excluded from both reported axes for both
  methods. Matches Section 5 "Two budget axes".

Three genuine differences were found and fixed (§1–§3), and several
paper-side ambiguities are documented (§4) so they are not mistaken for
implementation errors later.

---

## 1. Planted-data distribution restored to the paper's U[−1, 1] (both backends)

**What changed.** In `objectives_numpy.py::_sample_planted_data` and
`objectives_torch.py::_sample_planted_data`, the ground-truth weight matrix
is now drawn as

```python
W_true = rng.uniform(-w_true_scale, w_true_scale, size=(K, p))
```

**What the code did before.** Both backends drew `W_true = rng.randn(K, p)`
(standard normal). In the NumPy backend the paper-conformant uniform line
was present but commented out, and the docstring *claimed*
"W_true ~ Uniform[−w_true_scale, w_true_scale]" while the code did something
else. The torch backend's docstring honestly said N(0,1) and noted it
matched the NumPy reference — i.e. the two backends were consistent with
each other but both inconsistent with the paper.

**Why this matters / what was wrong with the old behaviour.**

* The paper's Section 5.1.1 specifies (W⋆)_{iℓ} ~ U[−1, 1] and explicitly
  reasons from it: "Because X is standard Gaussian and W⋆ has entries in
  [−1, 1], the ground-truth logits have coordinate-wise standard deviation
  √(p/3), so the softmax outputs are moderately peaked and the realised
  per-class sample counts n_i are close to, but not exactly, n/K."
* With N(0,1) entries the logit standard deviation is √p instead of
  √(p/3) — a factor √3 ≈ 1.73 larger. The softmax targets are noticeably
  more peaked, so classes are more imbalanced and the per-class objectives
  are farther from the "moderately peaked" regime the paper argues from.
  All experiment problem instances silently differed from the ones the
  paper describes.
* The `w_true_scale` knob was dead (accepted but ignored), and the NumPy
  docstring lied about the sampling distribution.

**The new behaviour.** Entries of W⋆ are uniform on
[−w_true_scale, w_true_scale]; the default `w_true_scale=1.0` reproduces
the paper's U[−1, 1] exactly. The class-coverage resampling loop
("reject datasets with an empty class") is unchanged and corresponds to
the paper's "we enforce n_i ≥ 1" remark. The torch docstring now states
the uniform law; the knob is live again.

**Impact.** Every experiment generates a (slightly easier, more
class-balanced) problem instance that now matches the paper's spec.
Numerical results from runs made before this change are not comparable
with new runs and were already invalidated by the July 4 soundness fixes.

---

## 2. `detect_plateau` reports the level of the curve it detects on

**What changed.** In `experiments.py::detect_plateau`, the reported level is
now

```python
"plateau_level": float(np.median(best_so_far[start:]))
```

**What the code did before.** Detection ran on the best-so-far
(monotone, non-increasing) GN* curve, but `plateau_level` was
`np.median(cov[start:])` — the median of the *raw* checkpoint values from
the onset onward.

**Why the old definition was wrong.** The raw GN* sequence is not
monotone:

* For the baseline the effect is real: its checkpoint metric re-evaluates
  the *current* per-node solutions, and non-convex gradient steps can make
  the worst-case λ genuinely worse between checkpoints.
* For both methods the reported value carries multistart-maximiser noise
  (each checkpoint's GN* is a heuristic lower bound found by local solves;
  the found value can fluctuate even when the true quantity is monotone).

So after the onset the raw values can drift *upward* while best-so-far —
the curve the detector actually certified as flat — stays put. The
reported "plateau level" could then be substantially higher than any level
the method had actually achieved, and the baseline/adaptive plateau-ratio
inherited this bias in whichever direction the noise fell. In short: the
detector answered "has the best-so-far curve stopped improving?", but the
level it reported was measured on a different, noisier curve.

**The new behaviour.** Level and detection now use the same best-so-far
curve. Because the detector's tail condition already guarantees the
best-so-far curve improves by less than `relative_improvement_tol` (5%)
after the onset, the median-of-best-so-far lies within 5% of both the
onset value and the final value — the reported number is stable and means
"the quality the method had actually reached and held".

**Impact.** `plateau_level` values and plateau ratios from earlier runs are
not comparable with new ones. Detection (onset index / found flag) is
unchanged — only the reported level.

---

## 3b. Selectable activation function — the testbed now can satisfy the paper's smoothness assumption

**What changed.** `objectives_torch.make_mlp_nonconvex` (and the experiment
drivers in `experiments.py`) accept `activation`: one of `"relu"` (default,
backward compatible), `"tanh"`, `"softplus"`, `"identity"`. The experiment
runner (`run_experiments.py`) runs its sweeps with `"tanh"`.

**What the code did before.** The hidden activation was hard-coded ReLU.

**Why this matters — an empirical finding, not just a conformance point.**
The paper's entire analysis assumes each F_k has a Lipschitz gradient
(L_k-smooth). A ReLU network violates this: the gradient jumps across every
activation kink, so no finite smoothness constant exists (the July 5 note
§1 flagged this). The first full equal-budget plateau sweep on the fixed
code showed how bad the consequences are in practice:

* At K=3 and K=4 the descent-lemma safeguard, hitting genuine kink
  violations, doubled the step-size divisor to `L_scale_final` = 2^24 and
  2^25 respectively — the adaptive method's steps were effectively frozen
  to zero length, and its GN* stalled ABOVE the baseline's plateau.
* At K=5 and K=6 `L_scale_final` settled at 32 — a 32-fold step-size
  penalty applied ONLY to the adaptive method. The baseline runs fixed
  1/L_λ steps with no safeguard at all: on the same non-smooth objectives
  its (equally invalid) long steps go unchecked. The comparison is then
  structurally unfair: one method obeys the theory's step-size discipline
  on a problem where the theory's premise is false, the other simply
  ignores it.
* Net effect at equal budget: the K=3 and K=6 headline directions inverted
  (adaptive worse than baseline), contradicting the theory that assumes
  smoothness — because smoothness did not hold.

The fix follows the paper: Section 5.1 introduces the activation as a free
choice ("Let σ be the activation function, e.g: identity, ReLU"), and the
paper's assumptions require L-smoothness, which tanh/softplus satisfy on
bounded regions (C-infinity, bounded second derivatives through the softmax
head) and ReLU does not. Running the benchmark suite with `tanh` puts the
experiments inside the theory's assumptions. Verified effect on the re-run
sweep: `L_scale_final` drops from 2^24–2^25 to 2–16 (the safeguard's
intended occasional-correction regime), the step-size collapse is gone,
and the K=3 direction inverts back to a 52x adaptive advantage. One
configuration (K=6) remains unfavourable to the adaptive method after the
fix — that residue is a genuine instance-structure phenomenon, not a
step-size artefact; it was investigated to closure with a seven-variant
diagnosis matrix and is documented in `EXPERIMENTS.md` §5.1 and
`output/plateau/README.md`.

The ReLU sweep that exposed this is archived at
`/Users/shirch/vscode101/.venv/ledger-artifacts/relu_sweep_archive/` as
evidence; it is diagnostic material, not citable benchmark data.

---

## 3. Stale documentation corrected (no behaviour change)

* `baseline.py::uniform_discretisation` docstring said the grid is walked
  "in warm-start order (lex sort)". The code has used the snake
  (boustrophedon) order since the July 4 fix — the lexicographic order is
  exactly what violated the paper's ≤ 2/r adjacency requirement for K ≥ 3.
  The docstring now describes the snake order and the guarantee.
* `bundle.py` header cited "Section 3 of the paper" for the bundle
  definition; the bundle is defined in Section 4 (Section 3 is the
  uniform-discretisation baseline). It also cited "Assumption 3.1" for the
  monotonicity property; the paper's monotonicity assumption is unnumbered
  in the current draft ("Assumption ??"), so the docstring now describes
  the property instead of citing a number that does not exist.

---

## 4. Paper-side ambiguities and deliberate implementation choices (documented, not "fixed")

These are places where the paper is ambiguous, self-inconsistent, or
describes a suggestion rather than a requirement. Recording them here so
they are not repeatedly re-investigated.

1. **Smoothness-constant estimation.** The paper (Eq. 22, stated for the
   LLM testbed "as in" the MLP section) estimates L̂_i by gradient-difference
   ratios over pairs drawn from the *recent bundle*, refreshed every K_est
   outer iterations, with unspecified K_est and N_pairs. The code instead
   estimates L once, before optimisation, from 40 random parameter-space
   probe pairs, and then relies on the runtime descent-lemma safeguard
   (July 4 fix; doubles a monotone `L_scale` whenever the certified-decrease
   inequality fails — see `Note/Jul_5_note.md` §1). We keep the code's
   scheme deliberately: Eq. 22's parameters are unspecified; a max-ratio
   estimate along the trajectory still carries no validity guarantee
   (on ReLU networks no finite global L exists at all), whereas the
   safeguard detects and repairs every underestimate it actually
   encounters, which subsumes Eq. 22's purpose. `L_scale_final > 1` in a
   result dictionary is the flag that the initial estimate was too small.
2. **Baseline per-node stopping tolerance.** Algorithm 1 stops each
   grid-node solve at grad_norm_tol = √(ε/(2L_max))·µ_λ̂ — a quantity
   defined by the strong-convexity analysis (Theorem 1). The MLP testbed
   is non-convex (µ does not exist), and the paper's own experiment
   protocol (Section 5, "Checkpoint cadence") runs the baseline as
   fixed-step gradient descent under a gradient budget, exactly as
   `uniform_discretisation` does. No change.
3. **Checkpoint boundaries.** The paper says checkpoints land at "the next
   natural boundary (a pass-completion for the baseline; an outer iteration
   for Algorithm 2)". The code additionally supports mid-pass checkpoints
   for the baseline (`eval_every_n_grads`), because at realistic budgets a
   full pass can consume the whole budget, leaving a one-point curve.
   Checkpointing frequency cannot change the optimisation trajectory (the
   metric bundle is separate and its cost is excluded from both axes);
   it only changes measurement density. Kept.
4. **Draft inconsistencies in the experiment spec.** Section 5.1.1 fixes
   K=3, p=10, n=50, seed=7; the figure captions in 5.1.2 say n=20. The
   experiment drivers keep all of these configurable and record the values
   used. The torch backend's default `seed=7` matches the paper.
5. **λ-maximisation.** The paper prescribes no solver for the (NP-hard,
   non-concave) max–min weight selection. The code uses deterministic
   multistart local solves (IPOPT, SLSQP fallback), scores every candidate
   after projection onto the simplex, and reports the best — a heuristic
   lower bound, as the July 4 notes already document. Any reported GN* can
   therefore understate the true worst case; this caveat applies equally
   to both methods, which are scored by the same fixed-strength maximiser.
6. **"CPU time" axis.** Both drivers measure wall-clock time
   (`time.time()`) with checkpoint-metric time subtracted. On a quiet
   machine running one experiment at a time this approximates CPU time;
   comparison plots should (and our runs do) come from serial, unloaded
   runs.

---

## 5. How the changes were verified

* Full verification suite (`ledger-artifacts/verify_fixes.py`, 10 checks
  covering the July 4 fixes) re-run after the changes: **10/10 pass**.
  Data-dependent figures moved as expected with the new data distribution
  (e.g. the logreg Hessian/L ratio check now reads 0.9575, still < 1 as
  required; the no-prune stall check converges to 6.5e-06, still past the
  old 6.665e-04 failure floor); all pass criteria are mathematical
  properties, not memorised constants.
* Dedicated smoke test (scratchpad `smoke_after_paper_fixes.py`):
  1. `detect_plateau` on a synthetic curve whose raw values drift upward
     after convergence: the reported level now equals the best-so-far
     median (1.0e-03) instead of the inflated raw median (~4.6e-03).
  2. Both backends' W⋆ samples satisfy max|W| ≤ 1 with standard deviation
     ≈ 0.577 (= 1/√3, the U[−1,1] value); a standard normal would exceed
     |W|=1 almost surely at these sample sizes.
  3. A tiny end-to-end `experiment_mlp_plateau_comparison` (K=3, p=4,
     n=60, h=8, 3000 gradients, both methods) runs to completion with
     finite, decreasing GN* histories.
