# Aug 8 note — CCP λ-solver to replace IPOPT: locked design + implementation plan

Session of Aug 8, 2026.  User-directed: implement the multistart
convex–concave procedure (CCP) from `Reference_essay/gns-ccp.pdf` as a
new λ-solver for the adaptive bundle method, replacing the IPOPT
λ-maximisation, for a later CCP-vs-IPOPT comparison.  Without-256
track.  NEW FILES ONLY — no existing file is modified.  This note is
the design record agreed in the Aug 8 Q&A; the files listed in §3 are
what will be written.

## 1. Criterion and naming

The λ-subproblem is the paper's GNS maximisation, identical to the
existing `_maximise_GN_fast` target on the Gram path:

    phi_i(λ) = λᵀ Q_i λ,   phi(λ) = min_i phi_i(λ),
    GNS = max_{λ ∈ Δ_K} phi(λ),
    Q_i = J_i J_iᵀ  (K×K)  ==  `BundleFast.grams[i]`  (paper's Q_i).

Naming convention in the new code, to avoid the collision with
`bundle_fast`'s `Ms`:

* `Q`  — the (m, K, K) Gram stack, `bundle.gram_stack()`.  This is what
  gets SAVED whenever instances are dumped (smoke test, tests).
* `Mc` — the per-CCP-iterate LP payoff matrix M^(c) (m, K).  Never
  saved; it changes at every CCP iteration (user point 2.1).

## 2. Decisions locked (Aug 8 Q&A)

1. **Single λ̂ out.**  The solver returns one `(pc_val, lam)` pair —
   same interface/semantics as `_maximise_GN_fast` — and λ̂ =
   argmax_{λ ∈ pool} phi(λ) feeds the msvrg inner loop.  No
   multi-candidate parallel bundle generation.  The pool is internal
   state only (next-round seeds).
2. **LP solver: HiGHS via `highspy`** (installed, 1.15.1), persistent
   model + warm start; guarded import with cold `scipy.optimize.linprog
   (method="highs")` fallback (same pattern as the cyipopt guard).
   Verified on the real LP shape (m=100, K=6): cold solve 13 simplex
   iters; rewrite all M coefficients (consecutive CCP iterate) → 1
   iter (cold reference 14); append one row (outer round m→m+1) → 0
   iters.  Values cross-checked against scipy to 1e-9.  Options:
   `output_flag=off`, `presolve=off`, `solver=simplex`.
3. **CCP stop:** delta_c = val(M^(c)) − phi(λ_c);  stop at
   `tau = min(1e-8 * max(1, phi(λ_c)), 0.01 * epsilon)` or at
   iteration cap T=100.  Note the code's epsilon is already on the
   squared scale (GN = min‖·‖² is compared to 2ε/3 / ε/3 directly;
   runs use ε = 1e-2, 1e-3), so the 0.01·ε cap is a safety term that
   never binds at current ε.  "Fairness" vs IPOPT's tol=1e-8 is NOT
   numeral-matching (different quantities: KKT residual vs predicted
   model improvement); it is that both solvers sit in the saturated
   regime, to be demonstrated by a ×10/÷10 tolerance ablation in the
   comparison phase.
4. **Adaptive seed schedule** behind `adaptive_seed_schedule` (default
   OFF — confirmed Aug 9: main runs use static N_new = N_0; the rule
   below activates only when the switch is explicitly on, as an
   ablation arm): rho = h/r = fraction of top-r restarts coming from
   this round's fresh random seeds.
       rho = 0 two consecutive rounds → N_new ← max(10 r, N_new / 2)
       0 < rho ≤ 0.25               → N_new unchanged
       rho > 0.25                    → N_new ← min(N_0, 2 N_new)
       pool collapse                 → N_new ← N_0
   Pool-collapse trigger: after re-screening under the new bundle,
   ALL old-pool points fall out of the top r, OR best old-pool value
   < 0.5 × last round's winner.  Vertices + λ_A + old pool always
   enter screening unconditionally; N_new governs only the random
   batch.  Log (N_new, rho) every round.
5. **Pool:** carry the previous round's deduplicated local maximisers
   (λ only — phi values are stale across bundles and are re-screened),
   cap `pool_cap = 3 r` (r default 10), current winner always kept.
   Per-round log fields (user point 1):
       pool_size, pool_cap, n_distinct_before_cap, n_dropped_by_cap.
   Escalation rule: if many rounds show n_distinct_before_cap >
   pool_cap, raise the cap to 4r or 5r in the experiment configs.
6. **Dedup keys** (greedy, phi-descending, keep-best): ℓ1 distance,
   tolerance-based active set A_η(λ) = {i : phi_i ≤ phi + η max(1,
   phi)}, and phi-value proximity.  Prefer under-merging.
7. **Seed sampler:** Exp(1)-normalised vs scrambled Sobol (sorted-gaps
   map from [0,1]^{K−1}) — undecided; the smoke test below decides.
   If indistinguishable, Exp(1) wins on simplicity.  Both implemented
   behind `seed_sampler`.
8. **val(A):** no diagnostic curve (dropped Aug 8).  The tiny game LP
   is kept ONLY for what Algorithm 1 itself uses: the λ_A seed and the
   sandwich-closure early exit (max_k min_i A_ik == val(A) → return
   vertex exactly).
9. **ε semantics unchanged:** outer stop stays pc_val ≤ 2ε/3 with
   pc_val = the CCP solver's returned value — same heuristic status as
   the IPOPT multistart value, keeping the two arms comparable.

## 3. New files (Original_py/ unless said otherwise)

1. `ccp_lambda_solver.py` — core, track-agnostic.
   * `CCPConfig` dataclass: N0=2000, r=10, pool_cap_factor=3,
     tau_rel=1e-8, tau_eps_frac=0.01, T_max=100, seed_sampler,
     adaptive_seed_schedule=False, dedup tolerances, rng seed.
   * `class CCPLambdaSolver(K, config)` — holds cross-round state
     (pool, N_new, rho counters, Sobol stream, persistent HiGHS model).
     `solve(Q, epsilon=None) -> (pc_val, lam)`; `stats_last` dict.
   * `_HighsGameLP` — persistent epigraph LP  max t s.t. Mλ ≥ t·1,
     Σλ = 1, λ ≥ 0; `resolve(Mc)` via `changeCoeff` (warm),
     `add_row()` across rounds; scipy cold fallback.
   * `phi_batch(Q, lams)` — one-einsum screening evaluator.
   * `sample_simplex_exp` / `sample_simplex_sobol`.
   * `exact_gns_K2(Q)` — K=2 parabola-envelope exact solver
     (pairwise-crossing enumeration): test oracle / gold standard.
2. `algorithm_ccp_without_256_checkpoints.py` — driver.  Imports the
   msvrg inner loop, Gram helpers, checkpoint machinery from
   `algorithm_fast_without_256_checkpoints` (no fork of those);
   reimplements only the outer function (`algorithm_adaptive_ccp`)
   with the λ-search call sites swapped from `_maximise_GN_fast` to
   one per-run `CCPLambdaSolver` instance.  Tier machinery
   (`lambda_tier_mode`) does not carry over — CCP is single-tier.
   History dict gains `ccp_stats_history` (per-round solver stats);
   everything else (checkpoint semantics, grad-equivalent accounting,
   stopping) inherited unchanged.  Alignment target confirmed Aug 9:
   the CCP arm mirrors `algorithm_adaptive_fast`, and the comparison's
   IPOPT arm IS `algorithm_adaptive_fast` (lambda_tier_mode="strict");
   the non-fast `algorithm_without_256_checkpoints.py` stays a
   historical reference only.
3. `run_ccp_smoke_sampler_without_256_checkpoints.py` — the
   Sobol-vs-Exp ablation (§4).
4. `sanity_checks_ccp.py` — §5.
5. Output folder: `output/ccp_smoke_sampler/` (new) — results.csv,
   summary.md, instances/*.npz (saved Q stacks only).

Build order: (1) solver + sanity checks, (2) smoke test → fix the
sampler default, (3) driver, (4) comparison harness (separate phase;
long runs only after user go-ahead, per convention).

## 4. Smoke test: Exp(1) vs scrambled Sobol

Static-bundle ablation, no outer loop; doubles as the solver test bed.

* Instances: synthetic Q stacks (K ∈ {3, 6} × m ∈ {30, 200} × 3
  instance seeds, random J_i); real Q stacks harvested from short
  in-process runs of the existing fast pipeline (early/mid/late
  bundles via small max_outer runs; SAVE `gram_stack()` = Q, never
  M^(c)); a few K = 2 instances where `exact_gns_K2` provides truth.
* Grid: sampler ∈ {exp, sobol} × N ∈ {128, 512, 2048} × 50 replicates
  (different seeds/scrambles).  N = 128 emulates the post-shrink
  N_new ≈ 10r regime.  Both arms share screening rule (r = 10), dedup,
  τ; only the sampler differs.  Deterministic seeds (vertices, λ_A)
  included in both arms as in production.
* Metrics vs GNS_ref (= best value ever seen on the instance ∪ one
  N = 2^15 heavy reference run): regret mean / 95th percentile, miss
  rate (relative shortfall > 1e-6), distinct-local-maxima count.
  Paired per (instance, N).
* Pre-registered decision: at N = 2048 AND N = 128, if both arms miss
  < 2% and the paired difference is not significant → default
  `seed_sampler="exp"`.  If Sobol wins only at N = 128 → still "exp",
  and raise the schedule floor 10r → 20r instead.  Sobol becomes the
  default only if it wins at the production sizes.
* Cost: ms per screen+polish; whole grid minutes.

## 5. Sanity checks (`sanity_checks_ccp.py`)

1. Monotone ascent: phi(λ_c) nondecreasing along every CCP run.
2. Termination invariant: at delta_c ≤ τ, active set has ≥ 2 members
   unless λ̂ is a vertex / degenerate flat piece (Lemma 2) — warn.
3. K = 2: solver value == `exact_gns_K2` within tolerance.
4. highspy path == scipy fallback path (same values).
5. Warm LP == cold LP on the same M^(c).
6. Agreement with `_maximise_GN_fast` (strict) on small instances.
7. `phi_batch` einsum == per-point loop.

## 6a. Implementation record (Aug 9)

All §3 files are written and green; no existing file was modified.

* Sanity: 10/10 PASS (`sanity_checks_ccp.py`).  Highlights: K=2 vs the
  exact envelope, worst relative shortfall 4.6e-11; CCP >= IPOPT-strict
  on all 6 real quadratic bundles (strictly better on 2, max rel gain
  2.4e-5); zero interior single-active terminations (Lemma-2 tie
  invariant); monotone ascent on every traced run.
* Smoke test (19 instances incl. 3 real K=5 bandit-toy Gram prefixes,
  50 reps): **seed_sampler default = "exp"**.  N=2048: 0/950 misses in
  both arms.  N=512: exp 0.84% vs sobol 0.42% (McNemar p=0.39).
  N=128: exp 2.42% vs sobol 1.89% (p=0.44).  No significant difference
  anywhere, so Exp(1) wins on simplicity (pre-registered rule).  Note
  for the (default-off) adaptive-schedule ablation: at the shrunk
  floor N ~ 10r both arms miss ~2% per solve; raising the floor to
  20r is the knob if that ever matters.
* End-to-end (K=5 bandit toy, eps=1e-2, same seeds): the CCP arm and
  the IPOPT-strict arm produce IDENTICAL pc trajectories
  (2.888e-2 -> 6.832e-3 -> 5.981e-3), both stop epsilon_certified with
  the same grad_equiv (137.2) and final bundle (m=10); λ-search wall
  time 2.50 s (IPOPT, 64 starts x 3 rounds) vs 0.004 s (CCP).  The
  sandwich closed in rounds 0-1 (exact vertex solutions for free).
* Bug found by the E2E and fixed: `_GameLP` appended grown-bundle rows
  AFTER the trailing simplex-equality row, so the SECOND growth's
  coefficient rewrite overwrote the equality row and made the LP
  unbounded (HiGHS kUnbounded + silent scipy fallback; values stayed
  correct, warm start silently lost).  Fix: any row-count change
  rebuilds the model (at most once per outer round); the warm path
  serves the same-shape rewrites, i.e. every CCP iterate.  Regression
  test added (consecutive growths, sanity check 4).

## 6. Deferred / dropped

* CCP-vs-IPOPT end-to-end comparison harness (bandit toy K5/K6, pure
  budget): next phase, plus the ×10/÷10 tolerance ablation of §2.3.
* val(A) diagnostic curve: dropped (Aug 8).
* Certified stopping: only realistic at K = 2 via the exact envelope;
  not pursued for K ≥ 4 (branch-and-bound cost ε^{−(K−1)/2}).
* Future work (user-confirmed Aug 9): objective-scaling sensitivity
  experiment (e.g. s_k = L_k instead of the main experiments' fixed
  s_k = 1) — GNS is not scale-invariant (paper Prop. 5), so this runs
  as a SEPARATE experiment with its own report, never mixed into the
  main CCP-vs-IPOPT comparison.
* Per-round log fields emitted by the solver, for the record:
  N_new, rho, pool_size, pool_cap, n_distinct_before_cap,
  n_dropped_by_cap, ccp_iters, lp_simplex_iters, sandwich_closed,
  lambda_search_wall_time.
