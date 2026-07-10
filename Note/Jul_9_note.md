# July 9, 2026 — Lambda-adaptivity figure runner (K=3 weight-path plot)

Scope: ONE new file, `Original_py/run_lambda_path_without_256_checkpoints.py`.
No existing file was modified. Companion records: `Note/Jul_8_note.md`
(baseline certification mode; `require_ipopt` default; the certified
Pareto-front runner).

## 1. Purpose

Visualise HOW the adaptive bundle method (paper Algorithm 2) chooses its
weights, versus the baseline (Algorithm 1), at K=3 where the weight
domain — the 2-simplex — can be drawn as a flat triangle:

- The baseline's weights are fixed BEFORE seeing the problem: the
  uniform grid of resolution r (C(r+2, 2) nodes at K=3; r=10 → 66 nodes
  as requested). They are pure combinatorics of (K, r), so NO baseline
  run is needed (none is performed); the figure computes the grid
  directly with `_uniform_simplex_grid`.
- The adaptive method's weight at each outer round is the argmax of its
  own max-min search — the currently worst-served weight — so its weight
  sequence exists only by running the algorithm. `algorithm_adaptive`
  already records it (`lambda_history`, one weight per outer round); the
  runner runs once and draws that sequence.

## 2. What the runner does

Configuration: the plateau sweep's K=3 run
(`output/plateau/K3_p6_n30_h4_tanh_r6_B30000`) unchanged — K=3, p=6,
n=30, hidden [4], tanh, seeds 7/8, BUDGET mode (epsilon=None), 30,000
gradient evaluations, max_inner=25, lambda_max_starts=64,
prune_inner=True, eval_every=1,200 — which yields exactly
30,000/(3*25) = 400 outer rounds, i.e. a 400-point weight sequence.
Only the figure's grid dots use r=10 (the user-requested density);
the adaptive method does not use r at all. Runs on the
without-256-checkpoints track (self-reported checkpoints; irrelevant to
which weights the search selects, but keeps wall time down).

Outputs to `output/lambda_path_K3/` (originals untouched):

- `lambda_path_triangle.png` — equilateral-triangle rendering of the
  simplex (vertices e1, e2, e3 = the pure objectives; projection
  x = lam2 + lam3/2, y = (sqrt(3)/2) lam3). Red dots = the baseline's
  66 uniform grid nodes. Coloured markers connected in round order = the
  adaptive weight sequence (colour = round index; star = round 1, X =
  final round).
- `summary.json` — run record, the full `lambda_history`, and
  consecutive-step statistics (L1 distances between successive weights;
  simplex L1 diameter = 2, r=10 grid spacing = 0.1).
- `README.md` (English, auto-generated) and `EXPLANATION_ZH.md`
  (hand-written mechanism analysis in Chinese: why the sequence is
  adaptive rather than uniform, and in what sense it is / is not a
  "continuous line").

## 3. Verification

`--smoke` mode (same problem, budget cut to 3,000 → 40 rounds) ran
end-to-end in ~40 s and produced the figure, summary, and README; the
smoke folder was deleted afterwards. The smoke run already showed the
honest headline: the round-ordered weight sequence JUMPS across the
triangle (median consecutive L1 step ~0.87 on a diameter-2 domain)
rather than sliding smoothly — the mechanism analysis treats this as a
finding about what adaptivity looks like (instant reallocation to the
new worst region), not as a defect.

## 4. Same-day revision (user feedback on the first figure)

Two changes to the runner, results regenerated in place:

1. **Single colour for the adaptive weights.** The first version
   coloured markers by outer-round index (with a colourbar); with 400
   rounds that read as visual noise. Now all adaptive markers are one
   colour; round order is carried by the connecting line and the
   start/stop markers, and the colourbar is gone.
2. **Certified mode instead of budget mode.** `epsilon=0.001` (stop as
   soon as the method's own search value <= 2*eps/3); the 30,000-grad
   budget stays as the fuse. Outcome of the regenerated run: CERTIFIED
   at round 180, 13,155 gradient evaluations (44% of the fuse), stop
   value 6.57e-4. The sequence no longer exhausts the budget, which
   also demonstrates the certification semantics on the figure itself.

`EXPLANATION_ZH.md` and `README.md` in the output folder were updated
to the certified run's numbers (180 rounds; median consecutive L1 step
0.47; 23% of steps below 0.2; first-quarter mean step 0.93 vs
last-quarter 0.57; 156 distinct 0.05-cells; 76% interior / 24% edge /
5% vertex).

## 5. Second same-day revision (make the mechanism VISIBLE)

User feedback: the colours were still too close, and the
"local chains + global jumps + shrinking steps" story was asserted in
text but hard to SEE.  Three changes, all in the runner's plotting code
(same run, figures regenerated from the identical deterministic run):

1. **High-contrast colours.** Baseline grid = hollow BLUE circles;
   adaptive weights = solid ORANGE dots.
2. **The main triangle now encodes the two movement types.**
   Consecutive-round movements with L1 step < 0.2 (2x the grid spacing)
   are drawn as thick orange segments — the local chains pop out;
   steps >= 0.2 are thin light-grey lines — the jumps recede.
3. **Two new figures.** `lambda_step_sizes.png`: per-round L1 step bars
   (orange = chain, grey = jump) with the 1/r spacing and the 0.2
   threshold as reference lines, a rolling median, and quarter means.
   `lambda_path_quarters.png`: the sequence split into four consecutive
   quarters, one small triangle each.

The quarter view surfaced an honest nuance now recorded in
EXPLANATION_ZH.md: the shrinking-step trend is not monotone — quarter
means 0.93 / 0.58 / 0.36 / 0.57 with chain shares 2% / 25% / 43% / 25%.
The fourth-quarter bounce is informative: near certification only a few
scattered above-threshold pockets remain, so the search slides WITHIN a
pocket and hops BETWEEN pockets; the rolling median still falls below
0.25 over the final ~20 rounds.
