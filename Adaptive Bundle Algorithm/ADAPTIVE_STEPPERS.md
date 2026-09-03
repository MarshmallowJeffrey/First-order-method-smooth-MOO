# Adaptive steppers for the certified Momentum-SVRG baseline — design spec

Date: September 1, 2026; revised September 2, 2026 (K2-campaign binding —
see the revision notice below). Status: **design approved, implementation
pending (campaign stage S1)**.
Chinese version: `Zh/ADAPTIVE_STEPPERS_ZH.md`; when one file changes, change
the other to match. Companion documents: `CODE_MAP.md` (file map),
`MANUAL.md` (how to run), `VARIABLES.md` (naming).

This document is the agreed experimental basis for adding three adaptive
step-size variants ("steppers") to the certified Momentum-SVRG baseline.
It is written to be self-contained: a future session should be able to
implement and run the experiments from this file alone.

**Scope rule.** Only the inner-loop *walk rule* — what is done with the
corrected gradient v_t — changes. The SVRG correction itself, the
certification logic, the descent safeguard, the early-stop trigger, and
the grad_equiv accounting are all untouched, for every stepper.

## ⚠ Sep 2 revision — binding to the K = 2 pair campaign

The stepper mathematics below is unchanged. Four planning items are
superseded by the K2 campaign plan (`../Note/Sep_2_note.md`, user-approved
Sep 2):

1. **Home of the 4-way switch** (as built, Sep 2): the walk rules live in
   the shared module `Original_py/Core Engine/stepper_core.py`; the
   switch-carrying executor is `Original_py/experiment_plot/
   run_stepper_pre_experiment_K2_without_256_checkpoints.py` — a
   stepper-parameterized copy of the v1 pair campaign's adaptive-CCP
   executor (`_run_leg_pair`). The campaign executors live in the runner
   layer, not in Core Engine, so no `algorithm_ccp_stepper_*` engine file
   exists. The SURF and uniform legs never carry the switch; they import
   only the winning rule from `stepper_core`. (§5's baseline-engine file
   plan is superseded.)
2. **Gate 0** accordingly compares stepper="const" against the v1 pair
   executor's adaptive-CCP leg. **PASSED Sep 2**: bit-exact on all 12
   checks (gram/theta/fval/λ/cost stacks, checkpoint arrays, audit
   histories) on the 4v9 smoke instance.
3. **The smoke protocol of §7 is superseded** by the campaign pre-experiment
   (stage S2): the stepper executor on the locked pair 4v9, B = 2,500
   (1/8 budget), eval_every = 50, 11 configs (const 1 + bb 1 + adagrad 3 +
   adam 6) × 3 sampler seeds {41, 141, 241}; judge = best-so-far worst GN
   (norm scale) vs grad_equiv (CPU time secondary; ties go to the simpler
   stepper). The winner is frozen and used by every leg of the campaign.
   **Gate 1** (safety) runs on the same 4v9 smoke instance instead of the
   bandit toy — same intent (cheap NaN/divergence catch), on exactly the
   S2 machinery. **PASSED Sep 2**: bb / adagrad(×3) / adam all finite,
   L_scale bounded (bb reached 8 via safeguard, by design), audits
   monotone, progress on every arm.
4. **GN scale**: every prescribed tolerance is stated on the gradient-NORM
   scale and squared once at code entry; internal Gram/val computation and
   every argmax/argmin are unchanged; all reporting is on the norm scale
   (bundle paper v2, Proposition 3; see `BASELINE_SURF.md` §3).

---

## 0. Baseline recap (stepper = "const")

Reference implementation:
`Original_py/baseline/baseline_svrg_certified_without_256_checkpoints.py`
(the active `_without_256_checkpoints` track; see `CODE_MAP.md`).

Per node, λ ∈ Δ_K is fixed. Notation:

- F_λ(x) = Σ_k λ_k f_k(x); smoothness L_λ = Σ_k λ_k L_k; L̂ = L_scale·L_λ
  (L_scale is the safeguard multiplier, starts at 1, only doubles).
- Anchor x̃ with full Jacobian J(x̃) (already computed by the pipeline);
  full gradient g̃ = J(x̃)ᵀλ, obtained at no extra oracle cost.
- Inner loop, t = 0 … m−1 (m = epoch length), stratified batch S:

      v_t = g_S(y_t) − g_S(x̃) + g̃          (SVRG corrected gradient)
      u_t = β·u_{t−1} + v_t                  (heavy-ball momentum)
      y_{t+1} = y_t − η·u_t,   η = c/L̂      (constant scalar step)

- Early stop: ‖v_t‖² ≤ ρ·solve_target for `patience` consecutive steps.
- Segment end: one charged full evaluation at y_m; descent check
  F_λ(y_m) ≤ F_λ(x̃) + 1e-10·(1+|F_λ(x̃)|). On failure: L_scale ×= 2,
  momentum reset, retry from the same anchor (≤ max_segment_retries).
  On success: anchor ← y_m (Option I), certification val = ‖J(y_m)ᵀλ‖²
  checked against node_tol.

Current defaults: c = msvrg_step_const = 0.1 (the η = 0.1/L rule of
Johnson & Zhang 2013), β = msvrg_momentum = 0.5, msvrg_max_segments = 10,
msvrg_trigger_rho = 0.7, msvrg_trigger_patience = 2,
max_segment_retries = 4.

All three new steppers keep the descent safeguard verbatim, and every
stepper consumes exactly the same `grad_pair` calls per inner step, so
grad_equiv accounting is identical across steppers by construction.

---

## 1. stepper = "bb" — SVRG-BB, regularized + clipped, momentum kept

Scalar step, recomputed once per segment; within a segment the walk rule
is exactly the const rule with η_k in place of η.

At the start of segment k, using the last two **accepted** anchors of the
current node (both full gradients are already available — zero extra
oracle cost):

    s = x̃_k − x̃_{k−1},      r = g̃_k − g̃_{k−1}
    D = max(sᵀr, δ‖s‖²)                     with δ = bb_delta_rel · L_λ
    η_k = clip( (1−β)·‖s‖² / (m·D),  c_min/L̂,  c_max/L̂ )

Design notes:

1. **(1−β) momentum correction.** Tan et al.'s formula η = ‖s‖²/(m·sᵀr)
   assumes plain SVRG steps. Heavy-ball amplifies each gradient by
   ≈ 1/(1−β) (β = 0.5 ⇒ ×2), and BB calibrates the *total* epoch
   displacement to 1/curvature: m·η/(1−β) = ‖s‖²/D, hence the factor.
2. **δ only enforces positivity** of the denominator (nonconvex objectives
   can give sᵀr ≤ 0). Safety against η_k blow-up is the clip's job, not
   δ's. With the max() form, a failed curvature estimate degrades to the
   constant fallback step (1−β)/(m·δ), which the clip then bounds.
3. **Clip window anchored to the trusted rule**: c_min = 0.01,
   c_max = 1.0, i.e. BB may raise the step at most ×10 above and lower
   ×10 below the current 0.1/L̂ rule. The window moves down automatically
   when the safeguard doubles L_scale.
4. **Fallback to the const rule** η = 0.1/L̂ in three cases: first segment
   of a node (no anchor pair yet); ‖s‖² ≈ 0 (trigger fired immediately);
   any safeguard-retry segment. On retry the BB proposal for that segment
   is discarded — failure mode = exactly the current algorithm.
5. **Per-node memory.** The pair (x̃_prev, g̃_prev) is per-node state;
   reset on node switch (λ changes ⇒ secant pair is meaningless). Only
   accepted anchors advance the pair; retries do not. m in the formula is
   the planned epoch length (the curvature estimate does not depend on
   how many steps the previous segment actually took).

New parameters: `bb_delta_rel = 1e-3`, `bb_clip = (0.01, 1.0)`.
Tuning tax: **0 configurations** (all constants are structural).

Provenance: SVRG-BB, Tan–Ma–Dai–Qian, NeurIPS 2016
(`Reference_essay/SVRG-BB_Barzilai-Borwein_step_size_for_SGD.pdf`,
arXiv:1605.04131) — linear convergence proven for strongly convex
objectives only. The regularized denominator follows the idea of
Li & Giannakis 2019 (arXiv:1910.06532), who bridge nonconvex to strongly
convex via a quadratic regularizer. Our objectives are nonconvex, so BB
here is a safeguarded accelerator, not a certified rate: the theory
anchor remains the fixed-step nonconvex SVRG analysis (Reddi et al.
2016), which the fallback path reproduces.

## 2. stepper = "adagrad" — AdaGrad-on-SVRG, per-coordinate, warm-started

Per-coordinate step; momentum in the numerator, AdaGrad scaling in the
denominator. G accumulates v (not u), so momentum amplification is not
double-counted:

    G_t = G_{t−1} + v_t ⊙ v_t               (cumulative, never decreases)
    u_t = β·u_{t−1} + v_t
    y_{t+1} = y_t − α_mult · u_t ⊘ (√G_t + ε)

**G₀ warm start (core design).** Initialize, per node,

    G₀ = (L̂/c)² · 𝟙                         (c = 0.1, same value each coordinate)

so that the first step equals the trusted const rule 0.1/L̂ on every
coordinate. Afterwards AdaGrad can only *selectively shrink*: coordinates
with a large gradient history slow down, flat coordinates keep ≈ 0.1/L̂.
Stability follows from "monotone non-increasing steps from a trusted
starting point". This removes the free step-size knob that plain AdaGrad
would introduce.

`α_mult` (the one coarse knob, default 1.0) raises the whole starting
level to α_mult·0.1/L̂. α_mult = 1 is the zero-tuning fallback; the
speed-up potential is probed with α_mult ∈ {1, 3, 10} (overreach is
caught by the descent safeguard).

Reset rules: **keep G across segments within a node** (the v_t stream is
continuous; variance shrinks at anchor updates rather than jumping);
reset to the G₀ rule on node switch; on safeguard retry reset u and G —
the re-initialized G₀ uses the doubled L̂, so the restart step is halved,
matching current retry semantics. ε = 1e-12 (inactive given the large G₀).

New parameters: `adagrad_alpha_mult = 1.0`, `adagrad_eps = 1e-12`.
Tuning tax: **3 configurations** (α_mult grid).

Provenance: SVRG-3 in Allen-Zhu & Hazan 2016
(`Reference_essay/Variance Reduction for-Faster-Non-Convex-Optimization.pdf`)
— empirical recommendation, no theorem. Convex theory for the
combination: AdaSVRG, Dubois-Taine et al., Machine Learning 2022
(arXiv:2102.09645). Nonconvex: no ready-made theorem; safeguard covers.

## 3. stepper = "adam" — VR + Adam (variance reduction feeds Adam)

The EMA first moment *is* the momentum (do not stack u on top):

    m_t = β₁·m_{t−1} + (1−β₁)·v_t
    G_t = β₂·G_{t−1} + (1−β₂)·v_t ⊙ v_t
    m̂ = m_t/(1−β₁ᵗ),   Ĝ = G_t/(1−β₂ᵗ)     (t = within-node step count)
    y_{t+1} = y_t − α · m̂ ⊘ (√Ĝ + ε)

Rules:

- β₁ = 0.9 fixed. **β₂ ∈ {0.9, 0.99} only** — memory length 1/(1−β₂) of
  10–100 steps, matched to the epoch length; the deep-learning default
  0.999 (memory 1000 steps) is banned: anchor moves would drag stale
  statistics for far longer than a segment.
- ε = 1e-8. The near-certification region where v_t → 0 would make the
  update ε-dominated is never entered: the existing trigger
  ‖v_t‖² ≤ ρ·solve_target stops the inner loop first (the trigger reads
  v_t, not the Adam direction, so it is optimizer-independent).
- State (m, G, t) is kept across segments within a node, cleared on node
  switch. On safeguard retry: clear moments and **α ← α/2** (α is not
  L-based, so retry-halving acts on α directly), preserving the "retry
  means smaller steps" semantics.
- Feed Adam **only v_t**. Full gradients enter only inside v_t as the
  correction term; segment-end full evaluations stay outside the
  optimizer (certification / descent check / checkpoints), never as a
  step. This keeps Adam's input stream statistically homogeneous.

Why VR is required at all for the certified track: with raw minibatch
gradients, E‖g_S‖² = ‖∇F_λ‖² + Var, so Adam's Ĝ measures noise near a
solution and the iterate stalls at a noise floor above node_tol. With
v_t, Var → 0 as y → x̃ and certification stays reachable.

New parameters: `adam_alpha = 3e-4`, `adam_beta1 = 0.9`,
`adam_beta2 = 0.99`, `adam_eps = 1e-8`.
Tuning tax: **6 configurations** (α ∈ {1e-4, 3e-4, 1e-3} × β₂ ∈ {0.9, 0.99}),
selected on the smoke protocol below before the main comparison.

## 4. Resolved design decisions

| Question | Decision | Reason |
|---|---|---|
| Reset the accumulator (G / moments)? | Keep across segments within a node; reset on node switch; on safeguard retry reset (AdaGrad re-inits G₀ with the doubled L̂; Adam halves α) | The v_t stream is continuous across segments (variance shrinks at anchor updates rather than jumping); a node switch changes the objective itself |
| Scalar or per-coordinate accumulation? | Per-coordinate by default; scalar AdaGrad-Norm (b_t² += ‖v_t‖²) kept only as an ablation flag | Per-coordinate is where the new capability lies (anisotropy); the scalar version duplicates BB's role with weaker information |
| Next anchor: last iterate or average? | Last iterate (Option I); no averaging variant | The certified point, the delivered point and the anchor stay the same point; averaging would cost one extra charged full evaluation per segment; averaging theory is convex-only |

## 5. Implementation plan

Per project convention, existing files are never edited; each stage adds
new files.

1. **New engine**
   `Original_py/baseline/baseline_svrg_adaptive_certified_without_256_checkpoints.py`:
   a copy of the certified baseline plus a `stepper` switch
   ∈ {"const", "bb", "adagrad", "adam"} and the new parameters above
   (`import _layout` before sibling imports). `"const"` reproduces the
   current algorithm and exists for the equivalence gate.
2. **New runner**
   `Original_py/experiment_plot/run_adaptive_stepper_smoke_without_256_checkpoints.py`
   for the gate + smoke protocol of §6–7.
3. Per-node stepper state to implement: BB — (x̃_prev, g̃_prev, η_k);
   AdaGrad — (G, u); Adam — (m, G, t). Result-dict additions: stepper
   name, per-segment step-size trace (BB: η_k list; AdaGrad/Adam:
   min/median/max of the effective per-coordinate step per segment),
   plus the existing counters.
4. **Implementation caution for Gate 0**: on the const path the stepper
   switch must not consume any extra RNG draws and must not reorder
   floating-point operations, otherwise bit-identity with the reference
   baseline is lost.

## 6. Validation gates (run before any experiment)

- **Gate 0 (equivalence).** stepper="const" vs the existing certified
  baseline, same seeds: segment-end F values, grad_equiv, safeguard
  counters and delivered points must match exactly.
- **Gate 1 (safety).** Each adaptive stepper on the bandit-toy
  objectives, 2 nodes each: no NaN/Inf, bounded safeguard retries,
  certification reached.

## 7. Smoke and main comparison protocol

- **Setting**: the K = 3 MNIST triple node grid with the Exp 5 (μ = 0)
  configuration, 1/8 of the full budget, 3 seeds. Full-budget runs only
  after the user signs off on the smoke results (house rule).
- **Fairness**: identical seeds and stratified batch streams, identical
  grad_equiv accounting (automatic — same `grad_pair` calls), identical
  epoch length and trigger settings across steppers.
- **Arms**: const / bb (1 config) / adagrad (α_mult ∈ {1,3,10}) /
  adam (6 configs, best one advances to the main comparison).
  The **tuning tax** (0 / 0 / 3 / 6 configurations) is reported as part
  of the results, not hidden.
- **Metrics.** Primary: served nodes vs grad_equiv curve (house
  standard) and per-node cost to certification. Diagnostics: step-size
  traces (η_k for BB; effective-step percentiles for AdaGrad/Adam),
  safeguard_retries, minibatch_steps_total. Compliance: re-check
  val ≤ node_tol for every served node.
- **Expected signatures** (what would confirm each design): BB — a
  majority of segments with η_k > 0.1/L̂ (amplification = acceleration
  evidence); AdaGrad — gains concentrated at α_mult > 1; Adam — if it
  does not beat BB/AdaGrad at equal tax, that settles "not worth it on
  the certified track".

## 8. Environment notes

- Launch with the venv's Python 3.13 interpreter explicitly; `run.sh`
  points at an inner 3.11 venv (known gotcha).
- Long (full-budget) runs require the user's go-ahead first.

## 9. References

In `Reference_essay/`:

- `accelerating-stochastic-gradient-descent-using-predictive-variance-reduction-Paper.pdf` — Johnson & Zhang 2013 (SVRG; the η = 0.1/L rule; Option I/II).
- `Stochastic_Variance_Reduction_for_Nonconvex_Optimization.pdf` — Reddi et al. 2016 (nonconvex fixed-step theory anchor; η = μ₁/(L n^{2/3})).
- `Variance Reduction for-Faster-Non-Convex-Optimization.pdf` — Allen-Zhu & Hazan 2016 (SVRG-3 = AdaGrad-on-SVRG, empirical).
- `SVRG-BB_Barzilai-Borwein_step_size_for_SGD.pdf` — Tan et al. 2016 (BB step; strongly convex theory; first Option-I proof).

External:

- Li & Giannakis 2019, *Adaptive Step Sizes in Variance Reduction via Regularization*, arXiv:1910.06532 (regularized BB denominator for nonconvex).
- Dubois-Taine et al. 2022, *SVRG Meets AdaGrad: Painless Variance Reduction*, Machine Learning (arXiv:2102.09645) — convex theory for AdaGrad-on-SVRG.
- Kingma & Ba 2015, *Adam* (arXiv:1412.6980).
