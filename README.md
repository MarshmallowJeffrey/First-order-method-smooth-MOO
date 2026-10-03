# MO-Gymnasium experiments: GRAB vs. uniform discretization and SURF

This folder reproduces the MO-Gymnasium results of the paper *GRAB: Gradient Reuse with Adaptive Bundles for
Smooth Nonconvex Multi-Objective Optimization*:
the table in Section "MO-Gymnasium Benchmarks" and the four convergence figures of the appendix
(FishWood and Deep Sea Treasure with K=2, Breakable Bottles with K=3, Fruit Tree of depth 6 with K=6).

The metric is the worst-case gradient norm of a bundle B,

    max_{lambda in simplex} GN(lambda, B),   GN(lambda, B) = min_{theta in B} || grad F_lambda(theta) ||,

plotted against Gradient Calls and training CPU time.

## Setup

Python 3.13 was used; install the packages with

```bash
pip install -r requirements.txt
```

`mo-gymnasium` is used only to read the Deep Sea Treasure map and the Fruit Tree rewards; all objectives
and gradients are computed exactly from the finite models (no sampled trajectories).

## Reproducing everything

```bash
./run_all.sh
```

The script runs every task serially with one numerical thread and takes about 7 minutes on the machine below
(`check_doubling.py` for all tasks takes about 8 more minutes).
Outputs:

- `results/<task>/...`: one JSON file per run (settings, checkpoints with Gradient Calls, CPU time and GN, timings) and one `.npz` file with the returned policies.
- `figures/<task>_convergence.png`: the figure.
- `figures/<task>_summary.json`: the number report.

Reported runs: Apple M1 (8 CPU cores, 4 performance and 4 efficiency), 8 GB memory, macOS 15.7, Python 3.13.5,
NumPy 2.1.3 (OpenBLAS 0.3.21), SciPy 1.15.3 (HiGHS), Gymnasium 1.3.0, MO-Gymnasium 1.3.2; one numerical thread,
no GPU.

The runs are deterministic.
CPU times depend on the machine and vary by a few percent between repeated runs.  Comparisons in the
Time panel between points that are only a few milliseconds apart can therefore flip.

## Running a single task

```bash
python scripts/run_uniform.py  dst            # Uniform, every plotted r, each run to its GN plateau
python scripts/run_surf.py     dst            # SURF (K=2 tasks), every plotted N
python scripts/run_adaptive.py dst            # GRAB, to the fixed budget of the task
python scripts/make_figure.py  dst            # figure + summary
python scripts/check_doubling.py dst          # optional: rerun each plotted baseline run for twice its length
```

Tasks are `fishwood`, `dst`, `bb` and `fruittree_d6`. Each run script accepts `--values` (a subset of r or N)
and `--results <dir>`; runs whose output already exists are skipped.

## Code

| File | Contents |
|---|---|
| `mogym/envs.py` | finite MDP models of the four tasks, with γ and τ |
| `mogym/oracle.py` | exact objectives F_k (discounted, KL-regularized) and their Jacobians |
| `mogym/adam.py` | Adam, the inner solver of all methods |
| `mogym/adaptive.py` | GRAB, Gradient Reuse with Adaptive Bundles (paper Algorithm 1) |
| `mogym/uniform.py` | uniform discretization (paper Algorithm 7), each grid weight run to the GN plateau |
| `mogym/surf.py` | SURF (Algorithm 1 of the SURF paper), K=2 |
| `mogym/lambda_solvers.py` | preference-weight solvers: exact envelope (K=2), simplicial branch-and-bound (K=3), multistart CCP of the paper appendix (K=6; the seed screening is one matrix product, large LPs by constraint generation) |
| `mogym/metrics.py` | reported metric: exact (K=2), certified interval (K=3), fixed pool of 23,992 weights (K=6) |
| `mogym/plateau.py` | stopping rule of the baseline runs and definition of the plotted point |
| `mogym/recorder.py` | checkpoints; training time excludes the metric evaluation |
| `mogym/config.py` | settings of the reported runs |
| `mogym/points.py` | reading runs back: plotted points, GRAB curve |
| `scripts/make_figure.py` | figure in the style of the MNIST figures of the paper, and the number report |
| `scripts/labels.py` | placement of the number labels next to the points (taken from the MNIST code) |

## Protocol

**Common.**
- All methods start at θ₀ = 0 and use Adam with (β₁, β₂, ε) = (0.9, 0.999, 1e-8).
- One evaluation of the K gradients counts as K Gradient Calls; for SURF, one scalarized step counts as K.
- Time is the training process CPU time. It includes preference selection and the SURF weight updates, and excludes the metric evaluation at checkpoints.
- Adam state: a method keeps the Adam state of an inner solve while its weight is unchanged, i.e. moves by at most 0.005 in every coordinate, and starts a new state otherwise.

**GRAB.**
- At each outer iteration it selects λ_t maximizing GN(λ, B) over the current bundle.
- It runs M_A Adam steps from the Algorithm 2 warm start and adds the iterate with the smallest ‖∇F_{λ_t}‖.
- It runs until the budget B is spent (no tolerance ε); every inner solve takes exactly M_A steps (fewer only when the budget runs out).
- Preference-weight solvers and their stopping conditions:
  - K=2: with λ = (x, 1−x) every bundle point gives a convex quadratic in x; the lower envelope is kept as pieces (split at the crossing points when a point is added), and its maximum is the largest value at a piece end. Exact, no iteration or tolerance.
  - K=3: branch-and-bound over triangles of the simplex. Lower bound: the best φ at the vertices, the center, a grid of resolution 12, the previous maximizer and the vertices of the examined triangles; upper bound on a triangle: min_i max_vertex q_i (q_i convex). The triangle with the largest upper bound is split into four at its edge midpoints. A selection stops once the certified relative gap in GN is at most 0.05 or after 1,000 splits (the reported metric uses 0.005, 1e5 splits and a grid of resolution 24).
  - K=6: multistart CCP (paper appendix). A selection returns at once if the best screened seed is within a relative 1e-8 of the upper bound val(A); otherwise each start stops when the predicted improvement is at most 1e-8·max(φ, 1e-6) (values normalized by the best seed value) or after the iteration cap. LPs: HiGHS with feasibility tolerances 1e-9, warm-started from the previous basis; LPs with at least 400 rows are solved by constraint generation (rows added until none is violated by more than 1e-10), which returns an optimum of the full LP.

**Uniform discretization.**
- The grid G_r is visited in snake order. In the first sweep each weight starts from the preceding weight's last iterate; afterwards each weight continues its own trajectory.
- The bundle is θ₀ plus the smallest-gradient iterate of every grid weight.

**SURF.**
- N+1 slots are placed at the quantiles of the reward-space arc length, with PCHIP interpolation and damping α = 0.3.
- Each slot runs K_S Adam steps per round, and the output is the last round.

**Budget and checkpoints.**
- Each task has one fixed budget B, the same for all methods (table below): 1.05 × the Gradient Calls of the farthest comparator point measured previously, rounded up to a multiple of 10³ (K=2), 3·10³ (Breakable Bottles) or 6·10³ (Fruit Tree); the factor 1.05 is a heuristic margin.
- All methods are checkpointed on one Gradient-Call schedule: every B/600 (K=2) or B/120 (K>2) calls up to B and every ten times that afterwards, at the end of the first unit that completes after the mark: a GRAB outer iteration, the M_U steps of one Uniform grid weight, or the K_S steps of one SURF slot. These checkpoints have `"kind": "calls"` in the run files.
- Uniform also records the GN after every sweep and SURF after every round (`"kind": "sweep"` / `"round"`); the stopping rule uses only these.

**Stopping rule and plotted points.**
- Every Uniform r and SURF N is a separate run. It stops once its per-sweep (per-round) GN has changed by at most 1% three times in a row, the run is ready, and three further sweeps (rounds) confirm the plateau.
  - Ready means: the 95th percentile of the gradient norms at the policies' own weights is at most max(1e-5, 0.25·GN). For SURF the slot weights must also have settled.
- The plotted point: among the Gradient-Call checkpoints before the stop and the stopping checkpoint, the earliest after which the GN stays within 5% of its value at stopping. Gradient Calls, CPU time and GN are read from that checkpoint.
- A run is plotted if its point lies within B. The values of r and N are powers of two from 2 (FishWood, DST, Breakable Bottles) and r = 1, …, 6 (Fruit Tree); the next value of every family has its point beyond B (`mogym/config.py`).
- `check_doubling.py` reruns every plotted run for twice its length; in the reported runs the GN changed by less than 1.7% after the stop (between −1.63% and +1.51%).

**Settings** (`mogym/config.py`):

| Task | Uniform | SURF | GRAB | Budget B |
|---|---|---|---|---|
| FishWood | M_U=50, lr 0.03, r = 2,4,…,512 | K_S=25, lr 0.03, N = 2,4,…,128 | M_A=10, lr 0.01, envelope | 108,000 |
| DST | M_U=5, lr 0.3, r = 2,4,…,512 | K_S=25, lr 0.1, N = 4,8,…,64 | M_A=10, lr 0.1, envelope | 93,000 |
| Breakable Bottles | M_U=5, lr 0.03, r = 2,4,…,32 | – | M_A=25, lr 0.03, K=3 subdivision (gap 0.05, 1,000 splits) | 18,000 |
| Fruit Tree (d=6) | M_U=5, lr 0.1, r = 1,…,6 | – | M_A=10, lr 0.03, periodic CCP | 60,000 |

The Fruit Tree preference selector is the multistart CCP of the paper appendix with these settings:
- Ordinary selections: 64 random seeds (redrawn at every selection), one polished start, at most 15 CCP updates.
- Every tenth selection: 1,024 seeds, eight starts, 100 updates.
- Extra starts: the center, the previous maximizers, a pool of at most 64 earlier local maximizers (newest first, pairwise distance above 0.08, shared by both settings), 19 points on each two-objective edge, and the centers of the three-objective faces.
- The seeds are screened against the whole bundle in one batched contraction, computed as a single matrix product.
- Every LP is warm-started from the previous basis; LPs with at least 400 rows are solved by constraint generation.

**Figures.** The style is that of the MNIST figures of the paper: GRAB as an orange curve, Uniform as blue
squares, SURF as red triangles, each family with a dashed descriptive trend c + a (x/s)^(-p) (s the median x),
fitted by least squares on the logarithms and drawn from the leftmost to the rightmost point.

## Expected results

`make_figure.py` prints these values. The ratio is the baseline's lowest point divided by the final GRAB value; in
parentheses, divided by the lowest GRAB value within the Gradient Calls and within the CPU time of that point
(the time ratio varies by a few percent with the CPU timing).

| Task | GRAB (end of budget) | Uniform, lowest point | SURF, lowest point |
|---|---|---|---|
| DST | 6.6235e-6 | 4.8588e-5 (r=512), 7.34× (6.36× / 5.50×) | 1.3043e-4 (N=64), 19.69× (19.24× / 16.63×) |
| FishWood | 1.0211e-4 | 2.5027e-4 (r=512), 2.45× (2.23× / 1.58×) | 1.0050e-3 (N=128), 9.84× (3.17× / 3.07×) |
| Breakable Bottles | [5.0486, 5.0738]e-4 | [1.5639, 1.5713]e-3 (r=32), 3.10× (3.09× / 1.99×) | – |
| Fruit Tree | 8.3210e-4 | 1.0594e-3 (r=6), 1.27× (1.22× / 1.00×) | – |

## Notes

- `lambda_solvers.lp` calls HiGHS through SciPy's internal interface (SciPy 1.15), with the same model and options as `scipy.optimize.linprog(method="highs")`. If that interface is unavailable it falls back to `linprog`, which gives the same solutions but runs more slowly.
- The Fruit Tree metric is a lower estimate: the maximum over a fixed pool of 23,992 weights, the same pool for all methods.
