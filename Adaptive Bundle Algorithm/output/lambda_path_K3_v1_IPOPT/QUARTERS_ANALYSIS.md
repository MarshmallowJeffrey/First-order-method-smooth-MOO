# Reading `lambda_path_quarters.png`: the weight sequence, quarter by quarter

Date: July 10, 2026. Chinese version: `QUARTERS_ANALYSIS_ZH.md`.
Companion: `EXPLANATION_ZH.md` (the full mechanism write-up),
`lambda_step_sizes.png` (the same trend as one bar chart),
`summary.json` (all numbers).

## 1. How to read the figure

The certified run selected one weight per outer round — 180 weights in
total (epsilon=0.001, certified at 13,155 gradients, 44% of the fuse).
This figure cuts that sequence into four consecutive time segments
(rounds 1–45, 46–90, 91–135, 136–180) and draws each segment on its own
copy of the K=3 weight simplex, rendered as an equilateral triangle
whose corners are the pure objectives (e1, e2, e3). In every panel:

- **Hollow blue circles** — the baseline's weights: the uniform grid of
  resolution r=10 (66 nodes). They are IDENTICAL in all four panels by
  construction.
- **Orange dots** — the weights the adaptive method's max–min search
  selected during that quarter (one per outer round).
- **Thick orange segments** — movements between consecutive rounds with
  L1 step < 0.2 (twice the grid spacing): "local chains", the worst
  weight sliding to a neighbour.
- **Thin grey lines** — consecutive-round movements with L1 step >= 0.2:
  "global jumps", the budget teleporting to a new worst region.
- Each panel's subtitle gives the quarter's mean step and the share of
  chain steps.

## 2. Parameter settings

Problem instance (the plateau sweep's K=3 testbed): K=3, p=6, n=30,
hidden_sizes=[4], activation=tanh, data seed 7, init seed 8,
W* ~ U[-1,1].

Adaptive run (the only run performed): CERTIFIED mode epsilon=0.001
(stop as soon as the method's own search value <= 2*eps/3 = 6.67e-4);
max_grad_evals=30,000 kept as the fuse; max_inner=25 (so one round
costs 3x25 = 75 gradient evaluations); lambda_max_starts=64;
prune_inner=True; eval_every_n_grads=1,200; max_outer=1,000,000 (never
binding). Measurement on the without-256-checkpoints track (irrelevant
to which weights the search selects).

Baseline (not run — combinatorics only): uniform grid at r=10, giving
C(r+2, 2) = 66 nodes on the K=3 simplex; grid spacing 1/r = 0.1 per
coordinate.

Figure constant: the chain/jump threshold is an L1 step of 0.2 — twice
the grid spacing; the simplex L1 diameter is 2.

## 3. What the baseline's points look like

The blue lattice is the whole story of Algorithm 1's weight choice: a
pattern fixed BEFORE seeing the problem, with density set by one number
(r), identical in every panel because nothing about it ever reacts to
anything. Every node receives the same optimisation schedule whether it
needs it or not. The lattice is also blind between its own points: the
covering radius (~1/(2r) in each coordinate) is the resolution below
which it simply cannot express a preference.

## 4. How the adaptive points move, and why

Each round's weight is the argmax of min_i ||grad F_lambda(x_i)||^2 over
the current bundle — the currently WORST-served weight. Serving it adds
a point that suppresses the GN landscape in a neighbourhood around it
(each branch of the min is quadratic in lambda, so the new "dip" has
width), which expels the next argmax from the region just served. The
four panels show the four phases of that process:

- **Rounds 1–45 (mean step 0.93, chains 2%): reconnaissance.** The
  bundle is tiny, almost every weight is badly served, and the GN
  landscape is high everywhere. After one spot is served, the next
  maximum is usually far away, so nearly every movement is a long grey
  jump; the orange dots scatter across the whole triangle, edges and
  corners included.
- **Rounds 46–90 (0.58, 25%): first chains.** Enough of the triangle
  has been suppressed that the remaining high plateaus are fewer and
  wider apart; more often the next maximum sits at the RIM of the dip
  just created, so the argmax slides — the first thick orange worms
  appear (top-centre, left flank).
- **Rounds 91–135 (0.36, 43%): polishing — the most local quarter.**
  Only a few contiguous hard regions remain above the certification
  level; the search works them down step by step. Nearly half of all
  movements are chain steps; the long chains (lower-left cluster,
  centre) are exactly the "continuous line" the original intuition
  expected — it exists, but as a LATE-phase, local phenomenon.
- **Rounds 136–180 (0.57, 25%): scattered pockets, then stop.** What is
  left above threshold is a handful of SEPARATED pockets (lower-left,
  right edge, upper-right). The method slides WITHIN a pocket (chains)
  and hops BETWEEN pockets (mid-size jumps), so the mean step bounces
  back up even though the process is finishing — an informative
  non-monotonicity, not noise. Over the final ~20 rounds the rolling
  median step falls below 0.25, and at round 180 the search can no
  longer find any weight above 2*eps/3: certification.

## 5. The one-sentence contrast

The blue lattice is a budget allocation decided once, uniformly,
blindly; the orange sequence is a per-round diagnosis — go to the
current worst weight, extinguish it, repeat — which first surveys the
whole triangle, then chains through the hard regions, and stops the
moment no weight anywhere is left above the certified level.
