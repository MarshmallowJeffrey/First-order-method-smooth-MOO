# SURF baseline for the K = 2 pair campaign — design spec

Date: September 2, 2026. Status: **design approved (user sign-off Sep 2),
implementation pending (campaign stage S3)**.
Chinese version: `Zh/BASELINE_SURF_ZH.md`; when one file changes, change the
other to match. Companion documents: `ADAPTIVE_STEPPERS.md` (optimization
cores; revised Sep 2), `CODE_MAP.md`, `MANUAL.md`; the campaign-level plan
(stages S0–S5, all legs, judged metrics) is recorded in
`../Note/Sep_2_note.md`.

This document specifies the SURF leg ("baseline1") of the K = 2 MNIST pair
campaign: the SURF weight-allocation strategy of Jiang–Huang–Chen wrapped
around this project's own inner machinery, plus the campaign's evaluation
protocol. Design principle: **SURF's "where to put λ" mathematics is kept
verbatim; the inner solve, oracle, accounting and safeguard are ours** — so
the only difference between legs is the λ-selection strategy.

---

## 0. Theory anchors

- Bundle paper v2 (`Reference_essay/A_first_order_bundle_method_for_smooth_
  multi_objective_optimization__MAnalytics_ (2).pdf`):
  - headline quantity (its eq. 4): ε_sm−stat(θ̂) = max_{λ∈Δ_K} ‖∇F_λ(θ̂(λ))‖ —
    the **norm-scale** worst-case gradient norm, our "worst GN";
  - **Proposition 3**: GN(·; B) is uniformly Lipschitz in λ (ℓ₁), the property
    the covering analysis needs — it holds for the norm, NOT for the square;
    this is why all tolerances and reports are on the norm scale;
  - **Proposition 4**: uniform discretization needs ~eK(1 + C·LipGN/ε)^{K−1}
    grid points (linear in 1/ε for K = 2) — the reason grid sizes are swept,
    not guessed.
- SURF paper (`Reference_essay/SURF.pdf`, arXiv:2605.20619): Algorithm 1
  (Rule 2), chord estimate (its eq. 12), damped CDF update (its eq. 13,
  condition α ≤ 0.5), Remark 1 (inexact warm-started inner solves suffice).

## 1. Setting and notation

K = 2; weights on a dial: λ(w) = [w, 1−w], w ∈ [0, 1].

    F_w(x) = w·f₁(x) + (1−w)·f₂(x),    ∇F_w(x) = J(x)ᵀλ(w)
    L_w = w·L₁ + (1−w)·L₂,             L̂ = L_scale·L_w   (global, doubles only)

x_n^(t) = the point slot n delivers in round t; y_τ = inner iterate; u_τ =
momentum accumulator; D = the cumulative delivered set (each point with its
f-vector and Gram G = JJᵀ).

SURF's geometric objects (kept verbatim):

    front path    f_PF(w) = f(x*_w) ∈ R²
    speed         v(w) = ‖∂f_PF(w)/∂w‖
    arc length    s(w) = ∫₀ʷ v(p) dp,    CDF  Φ(w) = s(w)/s(1)
    allocation    w_n = Φ⁻¹(n/N)  ⟹  s(w_{n+1}) − s(w_n) = s(1)/N

N is swept over the same ladder as the uniform leg's r (see §4).

## 2. Algorithm (single phase, runs to budget exhaustion)

**Init**: Φ₀(w) = w (round 0 = uniform grid); all slots x_n^(−1) = x₀; one
charged full evaluation at x₀ (every leg does this — the shared t = 0 anchor
for the metric curves); D = {(x₀, f(x₀), G(x₀))}; L_scale = 1.

**Round t = 0, 1, 2, …**

1. **Allocate**: w_n^(t) = Φ_t⁻¹(n/N), n = 0…N (endpoints w = 0, 1 included —
   the front's wings). Φ_t is stored on a 1001-point w-grid; the inverse is
   monotone interpolation.
2. **Solve one segment per slot** with the campaign's winning optimization
   core (per `ADAPTIVE_STEPPERS.md`; no tol-based stopping — fixed budget).
   Anchor x̃ = x_n^(t−1); its Jacobian is cached from the previous round's
   full evaluation, so the new-weight full gradient is free:

       g̃ = J(x̃)ᵀλ(w_n^(t))

   Inner loop τ = 0…m−1 (stratified batch S, early-stop trigger active):

       v_τ = g_S(y_τ) − g_S(x̃) + g̃            (SVRG corrected gradient)
       step per the winning core (const shown):  u_τ = β·u_{τ−1} + v_τ,
       y_{τ+1} = y_τ − (0.1/L̂)·u_τ

   Segment end: one charged full evaluation → f_n^(t) = f(y_m), J_n^(t);
   descent check F(y_m) ≤ F(x̃) + 1e-10·(1+|F(x̃)|); on failure L_scale ×= 2,
   momentum reset, retry from the same anchor (≤ 4); on success
   x_n^(t) = y_m. The paid-for point always enters D with its f and Gram.
   This evaluation triple-serves: descent check, front measurement h(u_n),
   and the metric layer's Gram.
3. **Chord arc estimate** (SURF eq. 12 + strict-monotonicity guard):

       s̃(w₀) = 0,   s̃(w_{n+1}) = s̃(w_n) + max(‖f_{n+1}^(t) − f_n^(t)‖₂, ε_arc)

   ε_arc = 1e-12 only prevents a flat plateau in Φ (coincident points) from
   breaking the inverse.
4. **Monotone interpolation + damped update** (SURF eq. 13): PCHIP through
   {(w_n, s̃(w_n))} → s̃ on [0,1]; Φ̃_t = s̃/s̃(1);

       Φ_{t+1} = α·Φ̃_t + (1−α)·Φ_t,    α = 0.3  (paper condition α ≤ 0.5)

5. **Termination**: budget check before each slot (grad_fuse convention);
   stop when grad_equiv ≥ max_grad_evals. The round count T is emergent:
   T ≈ budget / [(N+1)·(mean inner rows + 1 full eval)].

Why one coarse segment per slot is legitimate: SURF's Remark 1 — inexact,
warm-started, finite-step inner solves with a small α suffice; a slot is
progressively refined across rounds (vertical warm start).

`certify_final` flag (default OFF): freeze the final weights and solve each
slot to a target — kept only as an epilogue option; the fixed-budget
campaign does not use it.

## 3. Evaluation protocol (identical for every leg)

**y-axis — worst GN** (three-layer GN convention, user-approved Sep 2):

    per point/direction   GN(x, λ) = ‖J(x)ᵀλ‖ = √(λᵀ G(x) λ)
    per direction         gn(λ; D) = min_{x∈D} GN(x, λ)
    worst GN(D)           = max_{λ∈Δ₂} min_{x∈D} ‖J(x)ᵀλ‖

- numerical kernel unchanged: Grams and val = λᵀQλ are computed as always
  (√ commutes with min and max, so every argmax/argmin is unchanged);
- every prescribed tolerance is stated on the GN (norm) scale and squared
  once at code entry (solve_target = tol/4 on the squared scale is exactly
  ε/2 on the norm scale);
- all aggregation across λ and ALL reporting use the norm;
- **exact computation**: for K = 2, worst GN is evaluated exactly on a
  **200,001-point w-grid** (chunked BLAS; smoke tier 20,001) — zero
  approximation noise; CCP λ-search is only the adaptive leg's internal
  selection engine, never the metric;
- fixed budget ⟹ **no threshold line on any plot and no
  certified/uncertified marking** on the front figure.

**x-axes**: (1) total gradient evaluations = grad_equiv (λ-search and
interpolation excluded); (2) CPU time (process time; λ-search and CDF
overheads charged to the leg that spends them). Checkpoints log
(grad_equiv, cpu_time, worst GN) at the eval_every cadence; all legs share
the t = 0 anchor point so the curves start together.

**Figures**: ① best-so-far worst GN vs total gradient evaluations (main);
② best-so-far worst GN vs CPU time; ③ Pareto front — best-per-λ scatter
(within a rounded-λ group, argmin of w·f₁ + (1−w)·f₂), non-dominated
frontier polyline, color = λ₁, gray diamond = f(x₀).

**Secondary table**: frontier chord-length CV (coverage uniformity),
ε-accuracy vs the reference front, safeguard retries, minibatch step
counts, per-leg overhead time share.

## 4. Parameters

| Parameter | Default | Note |
|---|---|---|
| N | swept over {10, 20, 30, 40} | same ladder as uniform's r; each leg fights at its own best (N\*, r\*) |
| α | 0.3 | CDF damping, ≤ 0.5 |
| segment / trigger / β / step const | = campaign baseline | c = 0.1, β = 0.5, ρ = 0.7, patience 2, retries ≤ 4 (const core; the winning core may replace the step rule) |
| ε_arc | 1e-12 | strict monotonicity guard |
| Φ grid | 1001 points | CDF storage + inverse |
| budget | B = 20,000 grad-equiv, eval_every = 250 | campaign values (smoke 400/25) |
| worst-GN grid | 200,001 (smoke 20,001) | exact metric |
| certify_final | off | epilogue option only |
| Rule-1 sub-variant | bandit toy only | closed-form Φ one-shot weights (imported notebook has it); upper-bound reference arm |

## 5. Differences from original SURF (final list)

**Kept verbatim**: the Rule-2 loop structure, quantile allocation
Φ_t⁻¹(n/N), chord estimate (12), PCHIP, damped update (13) with α ≤ 0.5,
endpoints included, per-slot vertical warm start.

**Changed**: ① inner solve = this project's Momentum-SVRG segment family
with the descent safeguard (original: any off-the-shelf K-step solver) — to
match the other legs exactly and isolate "how λ is chosen"; ② front
measurement = the charged exact full oracle, shared with the descent check
(original: cheap approximations allowed); ③ single phase, budget
termination (original: round count T); ④ the worst-GN/CPU/front evaluation
layer is bolted on (measurement only, dynamics untouched); ⑤ ε_arc guard;
⑥ K = 2 only (M > 2 is the SURF paper's own declared future work).

Related but distinct: the MODPO-branch SURF (classmate's LLM experiment,
AdamW inner, no exact metric layer) is a sibling implementation on another
battlefield; plotting and budget-axis conventions are aligned with it.

## 6. File plan (add new, never edit existing)

- `Original_py/baseline/baseline_surf_without_256_checkpoints.py` — the leg
  (loop of §2, stepper injected from `stepper_core`);
- `Original_py/experiment_plot/run_surf_compare_K2_without_256_checkpoints.py`
  — campaign runner: three legs, checkpoint logging, the three figures and
  the secondary table.
