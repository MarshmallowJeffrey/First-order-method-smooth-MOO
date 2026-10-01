# Python Code Change Log

Date: July 1, 2026

## Notes

- This document records only code changes actually made during the chats. It does not describe features that already existed in the files.
- The user later renamed `algorith-main.py` to `algorithm.py` and `experiments-main.py` to `experiments.py`. These renames did not change the code logic. This document uses the current file names throughout.
- Temporary changes that were later reverted are not included.

## `algorithm.py`

### Removed

- Removed the unused `_T_map_grid_batched`; only `_T_map_batched`, which is used by the algorithm, remains.
- Removed imports of `T_map` and `GN` from `bundle.py`.
- Removed the `pc_fn` parameter from `_bundle_update_adaptive`, together with `pc_fn = GN` and the related argument passing.
- Removed the unused `mode` parameter from `algorithm_adaptive` and `pc_star`.
- Removed the `_pc_star_metric` alias and obsolete references to GAP, `_maximise_GAP`, and `bundle.mu`.
- Removed `bundle_from_points` from this file; it was moved to `baseline.py`.
- Removed the unused `Fmat` and `L` parameters from `_gn_value_and_jac_batched`.

### Added and changed

- Changed the `_gn_value_and_jac_batched` interface to:

  ```python
  _gn_value_and_jac_batched(Jmat, lam)
  ```

  Updated its call sites and documentation, removed the nonexistent strongly-convex scaling formula, and documented nondifferentiability when multiple bundle points tie for the minimum.
- Added `_gn_value_batched(Jmat, lam)` to calculate only the scalar GN value. The inner-loop stopping check now reuses `Jbuf` and no longer calculates an unused weight gradient.
- Split the previous weight into `selector_prev_lam` and `metric_prev_lam`, preventing checkpoint frequency from changing the algorithm's weight-selection results.
- Local IPOPT/SLSQP failures now report the failed starting weight and exception before continuing with the remaining starts.
- `algorithm_adaptive` now calls `prefer_fused_joint_oracle` automatically, so direct algorithm calls also prefer `joint.fused`.
- Added `ipopt_available()` and expanded IPOPT import handling to detect both missing packages and binary-library loading failures.
- Added `lambda_solver` and `require_ipopt` to `algorithm_adaptive`, allowing explicit IPOPT/SLSQP selection and preventing silent IPOPT fallback when required.
- Added `max_grad_evals` to `algorithm_adaptive`. The method stops at the gradient-evaluation budget and returns the actual lambda solver and configured budget. A final checkpoint is also recorded when the budget is not divisible by `K`.

## `bundle.py`

### Removed

- Removed the unused loop-based `T_map`.
- Removed the unused loop-based `GN`.

### Added

- Added the shared `prefer_fused_joint_oracle` function:
  - Uses `joint_oracle.fused` when available.
  - Uses the standard `joint_oracle` when no fused implementation exists.
  - If the fused implementation raises `MemoryError` or `NotImplementedError`, it warns once and permanently falls back to the standard implementation.

## `baseline.py`

### Removed

- Removed the unused `worst_errs` list.
- Removed the always-`NaN` `err` variable and the `"worst_errs"` result field.
- Removed obsolete comments about the previous suboptimality metric, `mu_lams`, A2, and NAG.
- Removed the unused `Tuple` import.

### Added and changed

- Replaced `coverage_mode` with the Boolean `evaluate_coverage` parameter. A temporary Bundle is built and GN* is evaluated only when it is `True`.
- Moved `bundle_from_points` from `algorithm.py` to this file because it is used only by Baseline checkpoints.
- Retained `_nearest_coarse_index` and documented that it is currently unused.
- `uniform_discretisation` and `bundle_from_points` now call `prefer_fused_joint_oracle` automatically.
- Added `max_grad_evals`. Baseline stops before the next step would exceed the budget, records a final checkpoint even when stopping in the middle of a grid pass, and returns the configured budget.

## `experiments.py`

### Removed

- Removed the unused `time` import.
- Removed `mode="gn"` from plotting and algorithm calls.
- Replaced the old `coverage_mode="gn"` argument with `evaluate_coverage=True`.
- Removed the unused `checkpoint_every` parameter.
- Removed the temporary local `_prefer_fused_joint_oracle` implementation and switched to the shared implementation in `bundle.py`.

### Added and changed

- Baseline and Adaptive now receive the same fused-preferred `joint_oracle`.
- `_plot_coverage` now plots GN* directly and no longer contains an unused GN/GAP mode branch.
- Added `detect_plateau`, which detects the GN* plateau onset and level using consecutive, non-overlapping checkpoint windows.
- Added `_plot_plateau_pair` to generate three pairwise comparison plots:
  - Baseline versus Sample Adaptive.
  - Baseline versus IPOPT Adaptive.
  - Sample Adaptive versus IPOPT Adaptive.
- Added `experiment_mlp_plateau_comparison`. All three methods share the same problem, initial point, oracle, and gradient budget. The Adaptive runs use `target_cov=None` so they do not stop as soon as they reach the Baseline's final value.
- The plateau experiment requires IPOPT to be genuinely available, preventing SLSQP results from being mislabeled as IPOPT results.
- The plateau experiment returns each method's plateau information, plateau-level ratios, and the three plot paths.

## `objectives.py`

### Removed

- Removed the unused PyTorch import and global PyTorch dtype/thread settings. The file now uses only NumPy.

### Added and changed

- Documented that the logistic-regression factory is not called by the current `experiments.py` and is retained only for future strongly-convex experiments.
- Added `K > 0`, `p > 0`, and `n >= K` validation to `_sample_planted_data`.
- Datasets containing an empty class are resampled up to 1,000 times; a `RuntimeError` is raised if all attempts fail.
- Forced the final cumulative class probability to exactly `1.0` to prevent inverse-CDF sampling failures caused by floating-point rounding.
- Class counts now use the actual number of samples instead of replacing empty-class counts with one.
- Added `h > 0` validation to `make_mlp_nonconvex`.
- Corrected the MLP documentation example to unpack four return values and documented the `joint_oracle` output arrays and their shapes.

## Verification

- All five Python files pass bytecode compilation.
- The scalar GN implementation was checked against the implementation that also returns its Jacobian.
- The fused oracle, standard oracle, and fused-fallback paths were tested.
- Input validation, empty-class resampling, joint-oracle output shapes, and the basic plateau-detection behavior were tested.
- The complete three-method plateau experiment has not been run because the local `cyipopt` binary currently fails to load. Strict IPOPT validation was confirmed to raise a clear error instead of falling back and mislabeling the result as IPOPT.

---

# Soundness fixes — July 4, 2026

A full audit (paper-vs-code fidelity, math soundness, line-level bug hunt, and
executable adversarial tests) confirmed the core method is implemented
faithfully, but found failure modes that the following changes fix. The public
APIs (`algorithm_adaptive`, `uniform_discretisation`, `pc_star`) are unchanged
except for two new keys in `algorithm_adaptive`'s result dict.

## `algorithm.py`

### Descent-lemma safeguard (adaptive L rescaling)

- **Problem (reproduced):** the T-map trusted the supplied smoothness
  constants `L` unconditionally. When `L` underestimates the true curvature —
  as with the repo's own (previously mis-derived) logistic-regression
  constants, or the probe-based estimates on ReLU MLPs where no finite global
  L exists — the method entered an exact no-progress cycle (the same point
  appended to the bundle hundreds of times, e.g. 375 duplicates in one run,
  GN* frozen for the entire budget) or diverged to an uncaught
  `OverflowError`/`ValueError`. This is the mechanism behind the frozen
  adaptive curves in the `mlp_crossover_h*` and `run_plateau8` notebook runs.
- **Fix:** `_T_map_batched` now also returns the descent-lemma prediction
  `u_star`; after each oracle evaluation the inner loop checks
  `F_lambda(x_new) <= u_star` (both sides were already computed — the check is
  free) and doubles a monotone `L_scale` multiplier on violation, in the
  spirit of the paper's Eq. 22 online L re-estimation. A `RuntimeWarning` is
  emitted on first trigger; a `RuntimeError` is raised if the scale exceeds
  2^60 (objectives demonstrably not L-smooth). The result dict gains
  `L_scale_final`.
- **Verified:** the previously stalling configurations now converge (wrong-L
  2-cycle → GN* 0.0; 3x-understated L → 1.5e-05 with no crash; the
  theory-faithful `prune_inner=False` logistic-regression run passes its old
  6.7e-04 stall floor, reaching 4.8e-05).

### `_maximise_GN` feasibility consistency

- **Problem:** local-solve outputs were scored at the raw solver iterate and
  only the final winner was renormalised, so (a) an infeasible iterate could
  win on a value inflated by `(sum lambda)^2` (GN is degree-2 homogeneous),
  and (b) the returned value did not correspond to the returned lambda.
- **Fix:** every candidate (start point or solver output) is projected onto
  the simplex and scored at the projection. The returned `(value, lambda)`
  pair is now always internally consistent; the monotone-in-starts guarantee
  is preserved.

### Epsilon-mode honesty

- The inner loop reports whether the `eps/3` target was actually met;
  `algorithm_adaptive` warns once when the `max_inner`/budget cap binds first
  (the Algorithm 2 termination argument does not apply to such iterations) and
  records `inner_cap_hits` in the result dict.

### Documentation corrections

- Stale draft references fixed (Algorithm 6 → Algorithm 2, Eq. 13 → Eq. 10,
  Appendix B.1 → A.1, removed Eq. 17).
- `pc_star`'s docstring no longer claims it uses "the same" maximiser as the
  outer loop; it documents the deliberate fixed-strength yardstick design and
  that the value is a heuristic lower bound on an NP-hard maximum, not a
  certificate.

## `baseline.py`

- **Problem (measured):** `_sort_grid_for_warmstart` used a plain
  lexicographic sort, whose "carry" boundaries jump up to the full simplex
  diameter (l1 = 2.0) for K >= 3 — at the default coverage config (K=6, r=9),
  494 of 2001 consecutive grid transitions violated the documented `<= 2/r`
  bound, silently degrading the chained warm start the paper's Algorithm 1
  enumeration guarantees.
- **Fix:** boustrophedon (snake) composition enumeration
  (`_snake_compositions`), which provably keeps consecutive grid points
  within l1 <= 2/r. Verified exhaustively for K = 2..7, r in {1,2,3,5,9,10}:
  worst consecutive jump now exactly 2/r, and the ordering is a permutation of
  the original grid.

## `objectives_numpy.py`

- **Problem (proved and measured):** `make_logreg_strongly_convex` set
  `L_i = ||X||^2 / (4 n_i) + reg`. The 1/4 factor is the binary-sigmoid
  Hessian bound; the multiclass-softmax bound is 1/2, and the correct data
  matrix is the class-i submatrix. At seed=0, K=2, p=6, n=40 the true Hessian
  max-eigenvalue exceeded the claimed constant by 22%, invalidating the
  1/L_lambda step size (this is the constant that triggered the reproduced
  stall above).
- **Fix:** `L_i = ||X_i||^2 / (2 n_i) + reg`. Verified against numerical
  Hessians (max eig(H)/L now 0.94 < 1 over probe points).

## Verification (July 4)

- `verify_fixes.py` (session scratchpad): 10/10 checks pass, covering the
  wrong-L cycle, wrong-L divergence, the logistic-regression stall floor, the
  epsilon-stop on a healthy problem, `(value, lambda)` consistency of
  `_maximise_GN` under both IPOPT and SLSQP, and the snake-ordering adjacency
  contract.
- Known limitations that remain by design: the outer GN maximisation is still
  a multi-start local heuristic (the exact problem is NP-hard), so GN* and the
  epsilon certificate remain heuristic lower bounds; and the epsilon
  certificate machinery is still unused by the experiment drivers, which run
  the budget-driven mode.
