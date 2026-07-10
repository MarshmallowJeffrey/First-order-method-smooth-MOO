# Certified Pareto-front comparison (K=2) — experiment overview

Date: July 8, 2026 (runs), overview written July 10. Chinese version:
`README_ZH.md`. This folder holds two runs of the same experiment:

- `pareto_certified_without_256_checkpoints/` — the main run (grid r=10),
  with the full findings write-up (`FINDINGS_ZH.md`, Chinese) and the
  per-run specification (`README.md`, English).
- `pareto_certified_without_256_checkpoints_r20/` — the follow-up re-run
  with ONLY the baseline grid resolution raised to r=20 (its own
  `FINDINGS_ZH.md` records the prediction-vs-outcome check).

Producer script: `Original_py/run_pareto_certified_without_256_checkpoints.py`
(`--resolution 20` for the second run).

## 1. Experiment name

Certified-mode Pareto-front comparison at K=2: uniform-grid baseline
(paper Algorithm 1) versus adaptive bundle method (paper Algorithm 2),
on the without-256-checkpoints measurement track.

## 2. What is being compared — which "Pareto front"

The problem has two per-class cross-entropy losses F1, F2, both to be
MINIMISED. After each method finishes its certified run, it is asked,
for every weighting lambda = (t, 1-t) on a 1,001-point sweep of t in
[0, 1]: "which point do you deliver for this trade-off?" Both methods
answer with the SAME rule — the point in their own point set with the
smallest ||grad F_lambda||^2 (computed from stored gradients for the
adaptive bundle, and from one post-run evaluation per grid node for the
baseline; no re-optimisation anywhere). Plotting the delivered points'
(F1, F2) values as t sweeps traces each method's EMPIRICAL front: the
trade-off menu it can actually hand you.

So the comparison is NOT against the problem's true Pareto front (which
is unknown); it is delivered-front versus delivered-front, judged by:

- **Dominance**: both axes are losses, so lower-left is better. A front
  that lies below-left of the other offers a better F2 at every F1 (and
  vice versa).
- **Resolution/smoothness**: how many distinct points the front has and
  how evenly it covers the trade-off range — a grid method can only
  answer with one of its C(r+1, 1) = r+1 node solutions.
- **Certification cost**: both methods run in certified mode; what did
  the certificate cost, and was it achieved at all?

Certification semantics per method: the adaptive method stops when its
own lambda-search value satisfies max_lambda min_i ||grad F_lambda||^2
<= 2*eps/3. The baseline runs the certification mode added on July 8:
a node is "served" once ||grad F_{lambda_i}||^2 <= node_tol at its own
weight, and the run stops when all nodes are served. Its eps is split
in norm space between the nodes and the grid: node_tol = eps/4 (half
the norm budget), the other half reserved for the between-node
degradation h*D with h = 1/(2r) and D = ||grad F1 - grad F2|| at the
delivered points; the exact all-lambda certificate
(sqrt(node value) + h*D_i)^2 is audited per node after the run.

Combos requested: (1) baseline eps 0.01 vs adaptive eps 0.01;
(2) baseline eps 0.01 vs adaptive eps 0.001 (the baseline run is shared
— its eps is the same in both).

## 3. Parameter settings

Problem instance (identical for all runs): K=2, p=20, n=50,000,
hidden_sizes=[96,96] (d = 11,522), activation=tanh, data seed 7,
init seed 8, W* ~ U[-1,1].

Method parameters: coarse_resolution r=10 (main run) / r=20 (re-run,
baseline only — the adaptive method does not use r and its runs are
bit-identical across the two folders); n_passes=100,000;
steps_per_point_per_pass=5; max_grad_evals=2,000 kept as the FUSE (not
the stopper) in certified mode; baseline/adaptive eval_every_n_grads=153;
max_outer=1,000,000; max_inner=5; lambda_max_starts=8; prune_inner=True;
plateau_window=4, plateau_consecutive_windows=2.

Certification parameters: baseline eps 0.01 -> node_tol = eps/4 =
0.0025; adaptive eps in {0.01, 0.001}, stop at own-search value
<= 2*eps/3.

Measurement track: without-256-checkpoints (checkpoints record each
method's own worst-case value; no external 256-start solves). This
affects wall time and checkpoint curves only, not the trajectories or
the delivered fronts.

## 4. Results and conclusions

Run outcomes (identical adaptive results in both folders):

| run | outcome | cost |
|---|---|---|
| baseline eps 0.01, r=10 | FAILED: 3/11 nodes served; audited all-lambda level 3.57 | fuse exhausted (2,000 grads) |
| baseline eps 0.01, r=20 | FAILED: 5/21 nodes served; audited level 3.81 | fuse exhausted |
| adaptive eps 0.01 | CERTIFIED (stop value 6.08e-3) | 132 grads = 6.6% of the fuse, bundle m=17 |
| adaptive eps 0.001 | CERTIFIED (stop value 6.63e-4) | 700 grads = 35% of the fuse, m=71 |

Front geometry (log-log figures in each run folder):

- **Smoothness/resolution — expectation confirmed.** The adaptive front
  is a dense, smooth curve (14 delivered points at eps 0.01, 54 at
  eps 0.001); the baseline front is a jagged polyline through 6 (r=10)
  / 8 (r=20) of its nodes.
- **Dominance — confirmed across the trade-off interior.** Wherever
  both losses are non-extreme, the adaptive front lies below-left of
  the baseline polyline by a wide margin. The only exceptions are the
  two pure endpoints: the baseline's t=0 / t=1 nodes converge to
  single-objective minimisers and reach deeper on ONE axis at
  catastrophic cost on the other (e.g. its t=1 node at r=10:
  F1 = 0.041 but F2 = 16.7) — expected endpoint behaviour, not
  dominance.
- **Tighter eps buys a better menu.** Going from eps 0.01 to 0.001
  (5.3x the gradients) densifies the front 14 -> 54 points and extends
  it ~3.5-3.8x deeper at both ends (min F1 0.033 -> 0.0088, min F2
  0.032 -> 0.0093).

Why the baseline could not certify (two independent layers, both
audited):

1. **Grid share too large (structural).** Measured D_max ≈ 6.1 at the
   delivered nodes, so the grid term alone is (h*D)^2 = 0.093 at r=10
   and 0.024 at r=20 — both above eps = 0.01. NO node_tol can certify
   0.01 at these resolutions; the formula says r >= 31 is needed for
   the grid share alone, r ≈ 62 under the equal split. This is the
   plateau study's "discretisation floor" restated in certified-mode
   language.
2. **Interior nodes do not converge (robustness).** Only the near-pure
   end nodes passed node_tol (3/11 at r=10, 5/21 at r=20); interior
   mixed-weight nodes oscillate (own gradients 0.04-3.5) because the
   probe estimate of L is ~8x too small at this width (the adaptive
   safeguard's L_scale_final = 8 is the witness) and Algorithm 1 has no
   safeguard, per the paper. Raising r dilutes per-node polish further
   (same fuse over more nodes) and cannot fix this.

Raising r=10 -> 20 (the follow-up): improved the FRONT exactly as the
"denser lambda menu" intuition predicts (8 delivered points, several
segments now touching the adaptive front) but did not — and per the
audit formula could not — rescue the certificate. Fronts measure
delivered function values; certificates measure worst-case gradient
norms over ALL weights. They are different quantities, and the
experiment shows both.

**Overall conclusion.** In certified mode on this expensive-oracle
testbed, the adaptive method delivers a denser, smoother, and (in the
trade-off interior) dominating Pareto menu, certified at 6.6% / 35% of
the gradient fuse for eps 0.01 / 0.001 respectively — while the
uniform-grid baseline exhausts the full fuse and fails certification on
two independent grounds at both grid resolutions tried. The r-sweep
additionally quantifies what it would take for the baseline to certify
eps 0.01 at all: r ≈ 62 nodes-worth of grid (equal split) AND a fix for
its interior-node step-size fragility, i.e. substantially more budget
and a safeguard the paper's Algorithm 1 does not have.

Honest-reporting notes: the adaptive certificate's witness is its own
8-start search (a heuristic lower bound of an NP-hard max);
prune_inner=True voids the epsilon-mode convergence-TIME proof
condition (warning recorded in every summary.json) but not the stopping
rule; D is measured post-run at the delivered nodes (one oracle call
per node, presentation work); one problem instance, seeds 7/8 — the
cross-run pattern, not the exact ratios, is the finding.
