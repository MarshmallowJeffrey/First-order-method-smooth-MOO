# Crossover sweep — cross-configuration analysis

Equal-budget (2,000 gradient evaluations) head-to-head at hidden widths
[16,16] → [128,128] (parameter count d = 642 → 19,458). Fixed: K=2, p=20,
n=50,000, tanh activation, grid resolution r=10 (11 nodes), prune_inner=True,
max_inner=5, data seed 7 / init seed 8. Runs strictly serial on an idle
machine. Per-configuration parameters, rationale, plots and health flags in
each `d*/README.md`.

## Headline numbers

| width | d | equal-TIME quality ratio | equal-GRADIENT quality ratio | bl final GN* | a2 final GN* | L_scale |
|---|---|---|---|---|---|---|
| 16x16   | 642    | ≈1 (1.3) | 64.3  | 1.14e-02 | 1.77e-04 | 4 |
| 32x32   | 1,794  | 3.7   | 60.7  | 8.98e-03 | 1.48e-04 | 4 |
| 64x64   | 5,634  | 18.8  | 332.5 | 9.85e-02 | 2.96e-04 | 8 |
| 96x96   | 11,522 | 46.5  | 1,180 | 2.71e-01 | 2.30e-04 | 8 |
| 128x128 | 19,458 | 101.4 | 1,941 | 9.27e-01 | 4.78e-04 | 16 |

Definitions:

- **Equal-TIME quality ratio** — give both methods the same wall-clock time
  (the baseline's total, which is far shorter) and compare best-so-far GN*
  (baseline / adaptive; >1 means the adaptive method is better). Read off
  the curves with log-GN* interpolation between checkpoints (checkpoints
  are ~13 per run; a raw step-function reading is unstable exactly at the
  comparison time — at 16x16 it swung the ratio by 200x because the
  comparison time fell 0.2 s before the adaptive method's first
  checkpoint). The 16x16 value is interpolated across the run's first
  measurement interval, so read it as "parity within measurement
  granularity", not as a precise 1.3.
- **Equal-GRADIENT quality ratio** — best-so-far GN* ratio when both have
  consumed the full 2,000-gradient budget. This is the oracle-efficiency
  statistic.

## The two findings

1. **On this expensive-oracle testbed (n=50,000) the adaptive method wins
   the wall-clock axis from the smallest width tested, and the advantage
   grows monotonically with d** — parity at d=642, 3.7x at d≈1,800, 101x
   at d≈19,500. The crossover point itself (equal-time ratio = 1) lies at
   or below d≈642 here: with 50k-sample gradients even a 642-parameter
   network is already past the cheap-oracle regime. (The cheap-oracle
   side, where the baseline wins wall-clock outright, is demonstrated by
   the plateau sweep's n=30 runs — CPU ratios 0.006–0.16 — so the full
   crossover picture spans the two sweeps.) This is the paper's crossover
   thesis measured end-to-end: once gradients are expensive, reusing every
   gradient across all weightings and spending them only where the
   worst case sits beats grid replication in wall-clock terms, not just
   oracle counts.
2. **Gradient efficiency is large at every width and grows with d.** 61x
   to 1,941x. The growth is itself informative: at fixed budget the bigger
   networks leave the baseline barely below its starting point (its 11
   nodes each get ~91 steps of plain GD on a 19k-parameter landscape),
   while the adaptive method still reaches the 1e-04 range.

## Why the time-to-common-target statistic is NOT the headline here

`summary.json` also contains the symmetric time-to-target ratios (both
methods race to the WORSE of the two final levels). Those read 0.38, 0.45,
2.02, 1.37, 0.215 across the sweep — non-monotone with a collapsed last
point. The reason is a boundary artefact of the statistic, not of the
methods: at large d the baseline's final level is so poor (0.93 at
128x128, barely below its starting value) that "first to reach the
baseline's final level" measures the adaptive method's fixed start-up cost
(a few outer rounds of lambda search) against a target the baseline sits
at almost immediately. When the two final qualities are orders of
magnitude apart, that target stops being meaningful. The equal-time and
equal-gradient quality ratios above have no such degeneracy (they compare
at a common resource point instead of a common quality point), which is
why the summary plot uses them. All three statistics are in every
`summary.json`.

## Reading the per-config CPU plots

`gn_vs_cpu_time.png` uses a log time axis (the two methods' total times
differ by 30–100x; on a linear axis the baseline's whole curve collapses
into the left margin) and a dotted vertical line at the moment the
shorter-running method (the baseline) has consumed its full budget — the
equal-time comparison is the vertical gap between the curves at that line.
Both curves start from the same leftmost point: that is the shared t=0
checkpoint (the same initial parameter vector scored by the same metric).
A log axis cannot place x=0, so that shared starting point is drawn at a
pseudo-abscissa left of the first real measurement (one third of it); every
other point sits at its true time. The plotted curves are the raw
per-checkpoint GN* values (they can bounce upward, especially the
baseline's — its node solutions genuinely move); the statistics above use
the best-so-far envelope.

## Health flags

`L_scale_final` rises 4 → 16 with width: the pre-run probe estimate of the
smoothness constants degrades on larger parameter spaces and the descent
safeguard corrects it with 2–4 doublings. This is the safeguard's intended
regime (compare 2^24–2^25 on the archived ReLU runs). `inner_cap_hits` = 0
everywhere; the only runtime warning is the expected L-underestimate
notice.

## Honest-reporting notes

- The adaptive method's wall time for the SAME 2,000-gradient budget is
  30–100x the baseline's (e.g. 9,040 s vs 95 s at 128x128): the lambda
  search per outer round is real work and grows with bundle size and d.
  The equal-time statistic already accounts for exactly this; the point of
  the sweep is that past d ≈ 1,800 the overhead is worth it.
- CPU times exclude the checkpoint metric (256-start IPOPT GN* solve) for
  both methods, per the protocol; total wall time including metrics is
  larger for both.
- One instance per width (seed 7); the cross-d trend is the finding.
- All GN* values are heuristic lower bounds of an NP-hard max, same fixed
  yardstick for both methods.
