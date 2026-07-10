# Findings: what changed when we removed the 256-start checkpoint metric

Date: July 8, 2026. Companion files: `README.md` in this folder (exact
semantics of the variant), the four `ab_*.png` figures (the A/B
comparisons), and the two `summary.json` files (all numbers below).
A Chinese version is kept at `FINDINGS_ZH.md`; when one file changes,
change the other to match.

## 1. What was changed, in one paragraph

In the original protocol, every checkpoint pauses the run and scores the
method's current point set with an EXTERNAL yardstick: a fixed 256-start
IPOPT search for the worst weighting, the same for both methods
(`pc_star`). In this A/B variant, that external solve is removed. Instead,
each checkpoint records what the method ITSELF already computed:

- the baseline records the worst (largest) most-recently-computed
  ‖∇F_{λ_i}‖² across its grid nodes — each node scored at its own weight,
  using the gradient it already computed for its own steps;
- the adaptive method records the value of its own most recent λ-search
  (the search it runs every outer round anyway, with 64 starts in the
  plateau run and 8 in the crossover run).

Nothing about the algorithms changed. We verified this directly: both
variants produce checkpoint positions at exactly the same gradient counts
and end with exactly the same safeguard state (`L_scale_final` 4.0 and
8.0). The trajectories are identical; only the scoring rule differs.
Two configurations were run: plateau K=5 (30k gradient budget) and
crossover 96x96 (2k gradient budget).

## 2. The numbers

Final best values (lower is better; "yardstick" = original 256-start GN*,
"self" = this variant):

| run | curve | yardstick | self-reported |
|---|---|---|---|
| plateau K=5   | baseline | 4.04e-02 | 2.32e-02 (still falling) |
| plateau K=5   | adaptive | 4.36e-03 | 4.60e-03 |
| crossover 96x96 | baseline | 2.71e-01 | 9.31e-01 (raw last 2.05) |
| crossover 96x96 | adaptive | 2.30e-04 | 1.53e-04 |

## 3. Finding 1 — the adaptive method's self-report matches the yardstick

Across both runs, the adaptive method's dashed (self-reported) curve sits
on top of its solid (yardstick) curve: the ratio between the two has
median 1.00 (plateau) and 1.07 (crossover).

**Why.** Both numbers estimate the SAME quantity — the worst weighting's
best squared gradient norm over the bundle, max over λ of min over points.
The only differences are search strength (the method's own 64 or 8 starts
versus the yardstick's 256) and a small lag (the self-report scores the
bundle at the start of the current outer round). On these problems the
smaller start counts were evidently enough to find essentially the same
maximum, so the external yardstick adds almost nothing FOR THIS METHOD.

## 4. Finding 2 — the baseline's self-report disagrees with the yardstick, in opposite directions in the two runs

### 4.1 Plateau K=5: the self-report is flattering (too optimistic)

The baseline's yardstick curve stalls at 4.04e-02 from ~4,800 gradients on
(its discretisation floor) and even drifts slightly up. Its self-reported
curve keeps falling and ends at 2.32e-02 — about 1.7x BELOW the yardstick
— crossing under it near 24,000 gradients, still descending at the budget
end.

**Why.** The self-report only looks at each node at its OWN weight λ_i.
Gradient descent keeps grinding each node's own gradient toward a local
stationary value, so this score can keep improving essentially without
limit — it has no floor. But the quality claim the experiments care about
is uniform over ALL weightings, including those BETWEEN grid nodes, and
that is exactly where the discretisation floor lives. The yardstick
searches over all weightings and finds it; the self-report cannot see it
by construction. (Early in the run the self-report is instead HIGHER than
the yardstick: it takes the max over all nodes, including nodes not yet
visited that still sit at the initial point, while the yardstick may take
whichever node happens to serve each weighting best.)

### 4.2 Crossover 96x96: the self-report is damning (too pessimistic)

Here the baseline's self-reported curve runs one to two orders of
magnitude ABOVE the yardstick for the entire run (spiking to ~150 at one
checkpoint, oscillating between ~1 and ~30 afterwards, raw end value
2.05), while the yardstick sits between 0.3 and 1.

**Why.** At this width the probe-based estimate of the smoothness constant
L underestimates the true curvature by roughly a factor of 8 (the adaptive
method's safeguard, which shares the same L input, had to scale it by 8).
The baseline has no safeguard — per the paper — so its fixed steps are far
too long, and its node iterates oscillate instead of converging. The
self-report takes the max over nodes, so it is pinned by the single worst
oscillating node at every checkpoint. The yardstick is more forgiving: for
every weighting it scores the BEST available node, so a few badly
oscillating nodes do not dominate it as long as some node serves each
weighting acceptably.

## 5. What this means

1. **The baseline's self-report is not a usable substitute for the
   uniform metric.** It errs in both directions, and which direction
   depends on the regime: when its steps are healthy it flatters itself
   (blind to between-node weightings — the "no floor" effect); when its
   steps are unhealthy it condemns itself (pinned by its worst node). No
   fixed correction maps one to the other.
2. **The adaptive method's self-report is essentially the same
   measurement** as the yardstick, because the method's own inner
   machinery already solves the yardstick's problem every round. For
   adaptive-only quick looks, the 256-start checkpoint solve is close to
   redundant.
3. **So the external yardstick's real, irreplaceable job is scoring the
   BASELINE honestly and putting both methods on one comparable scale.**
   The 256-start solve is not what creates the baseline's plateau in the
   official results — the plateau is a property of the grid; the solve is
   merely the instrument that can see it.
4. Practical side effect: with the checkpoint solves removed, total wall
   time drops noticeably (the solves were always excluded from the
   REPORTED axes, but they were real waiting time). This variant is
   usable as a fast preview mode for the adaptive curve — but its
   baseline curve means something different and must not be mixed with
   official results.

## 6. Where everything is

- Variant code: `Original_py/baseline_without_256_checkpoints.py`,
  `Original_py/algorithm_without_256_checkpoints.py`,
  `Original_py/run_experiments_without_256_checkpoints.py`
  (original files untouched).
- Runs: `plateau/K5_p6_n30_h4_tanh_r6_B30000/` and
  `crossover/d11522_h96x96_tanh_n50000_B2000/` in this folder.
- A/B figures (four curves each — solid = yardstick, dashed =
  self-reported; red = baseline, purple = adaptive):
  `ab_plateau_K5_gn_vs_grad_evals.png`, `ab_plateau_K5_gn_vs_cpu_time.png`,
  `ab_crossover_h96x96_gn_vs_grad_evals.png`,
  `ab_crossover_h96x96_gn_vs_cpu_time.png`.
