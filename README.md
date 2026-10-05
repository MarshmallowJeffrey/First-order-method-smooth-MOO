# MO-Gymnasium experiments: GRAB vs. uniform discretization and SURF

This folder reproduces the MO-Gymnasium results of the paper *GRAB: Gradient Reuse with Adaptive Bundles for
Smooth Nonconvex Multi-Objective Optimization*: the convergence figures and the numbers for FishWood (K=2) and
Fruit Tree of depth 6 (K=6).

The metric is the worst-case gradient norm of a bundle B,

    max_{lambda in simplex} GN(lambda, B),   GN(lambda, B) = min_{theta in B} || grad F_lambda(theta) ||,

plotted against Gradient Calls and training CPU time.

## Setup

Python 3.13 was used. Install the packages with

```bash
pip install -r requirements.txt
```

`mo-gymnasium` is used only to read the Fruit Tree rewards. All objectives and gradients are computed exactly
from the finite models; no trajectories are sampled.

## Reproducing everything

```bash
./run_all.sh
```

The script runs every configuration of both tasks, then the CPU-time repeats, then the figures. Everything runs
serially with one numerical thread and takes about 20 minutes on the machine below: about 6 minutes for the runs, the rest for the four timing repeats.

Outputs:

- `results/<task>/...`: one JSON file per run (settings, checkpoints with Gradient Calls, CPU time and GN, timings) and one `.npz` file with the returned policies. `results/<task>/timing.json` holds the CPU times of the repeats.
- `figures/<task>_convergence.png`: the figure.
- `figures/<task>_summary.json`: the numbers.

The reported runs used an Apple M1 (8 CPU cores: 4 performance, 4 efficiency) with 8 GB memory, macOS 15.7,
Python 3.13.5, NumPy 2.1.3 (OpenBLAS 0.3.21), SciPy 1.15.3 (HiGHS), Gymnasium 1.3.0 and MO-Gymnasium 1.3.2,
with one numerical thread and no GPU.

The runs are deterministic: Gradient Calls, GN and the selected weights are identical in every repeat. Only CPU
time varies. The figures plot the median CPU time over 5 runs, and the summary also reports the range.

## Running a single task

```bash
python scripts/run_uniform.py  fishwood       # Uniform, every configured r, each run to its GN plateau
python scripts/run_surf.py     fishwood       # SURF (K=2), every configured N, each run to its GN plateau
python scripts/run_adaptive.py fishwood       # GRAB, to the budget B of the task
python scripts/time_repeats.py fishwood fruittree_d6 --repeats 5   # CPU time over 5 runs
python scripts/make_figure.py  fishwood       # figure + summary
```

The tasks are `fishwood` and `fruittree_d6`; SURF is run for FishWood only.

Each run script accepts `--values` (a subset of r or N) and `--results <dir>`. A stored run is reused only if its
identity matches: settings, source code, package versions, model and arrays (`mogym/identity.py`). If the
identity differs, the script stops instead of skipping or overwriting the run.

`time_repeats.py` reruns every configuration and checks that each repeat reproduces the stored run exactly.
Without it, `make_figure.py` uses the CPU times of the single stored run.

## Code

| File | Contents |
|---|---|
| `mogym/envs.py` | finite MDP models of the two tasks, with γ and τ |
| `mogym/oracle.py` | exact objectives F_k (discounted, KL-regularized) and their Jacobians |
| `mogym/adam.py` | Adam, the inner solver of all methods |
| `mogym/adaptive.py` | GRAB (paper Algorithm 1) |
| `mogym/uniform.py` | uniform discretization (paper Algorithm 6), run to the GN plateau |
| `mogym/surf.py` | SURF (Algorithm 1 of the SURF paper), K=2, run to the GN plateau |
| `mogym/lambda_solvers.py` | preference-weight selection: exact envelope (K=2, paper Appendix A.4.1) and multi-start CCP (K=6, paper Algorithm 2) |
| `mogym/metrics.py` | reported metric: exact for K=2, a fixed pool of 23,992 weights for K=6 |
| `mogym/plateau.py` | stopping rule of the baseline runs and the plotted point |
| `mogym/recorder.py` | checkpoints; training time excludes the metric evaluation |
| `mogym/identity.py` | identity of a stored run |
| `mogym/config.py` | settings of the reported runs |
| `mogym/points.py` | reading runs back: plotted points and the GRAB curve |
| `scripts/make_figure.py` | figure and summary |
| `scripts/labels.py` | placement of the number labels next to the points |

## Protocol

**Common.**
- All methods start at θ₀ = 0. Its objective values and gradients are evaluated once and counted.
- All methods use Adam with (β₁, β₂, ε) = (0.9, 0.999, 1e-8).
- One evaluation of the K gradients counts as K Gradient Calls; for SURF, one scalarized step also counts as K.
- Preference selection, the starting-point rule and the SURF weight updates use stored values only and make no oracle calls.
- Time is the CPU time of the training process. It includes preference selection, the starting-point rule and the SURF weight updates. It excludes the metric evaluation at checkpoints.
- An Adam state is kept only while a method continues the same trajectory at a weight that has moved by at most 0.005 in every coordinate. Otherwise the solve starts with a new Adam state.

**GRAB** (Algorithm 1, no tolerance ε; the run stops when the budget B is spent). Each outer iteration does the
following:
1. Select λ_t, an approximate maximizer of GN(λ, B) over the current bundle (see below).
2. Choose the starting point θ₀^(t): the bundle point minimizing F_λ − ‖∇F_λ‖²/(2L_λ), with L_λ = Σ_k λ_k L_k from empirical smoothness estimates L_k. Ties go to the first minimizer in the stored order.
3. Run M_A Adam steps on F_{λ_t}. Every inner solve belongs to a trajectory, which keeps its last iterate and Adam state.
   - θ₀^(t) and the point added in step 4 belong to the same trajectory.
   - If θ₀^(t) belongs to a trajectory last run at a weight within 0.005 of λ_t in every coordinate, the solve resumes that trajectory from its last iterate with its Adam state.
   - Otherwise a new trajectory starts at θ₀^(t) with a new Adam state.
4. Add the iterate of the solve with the smallest ‖∇F_{λ_t}‖ to the bundle.

**Preference selection.**
- K=2: with λ = (x, 1−x), every bundle point gives a convex quadratic in x. The lower envelope is kept as pieces, which are split at the crossing points when a point is added. Its maximum is the largest value at a piece end (Lemma 4). This is the envelope of Algorithm 3, built by inserting one point at a time instead of by divide and conquer. It is exact, with no iteration or tolerance.
- K=6: multi-start CCP (Algorithm 2) with these steps:
  - **Upper bound.** The upper bound val(A) (Proposition 10) and its maximizer λ_A come from one LP. The LP is solved again only when a new bundle point cuts λ_A; otherwise λ_A, val(A) and the dual bound are unchanged.
  - **Seeds.** The seed set has these points:
    - the K vertices and λ_A;
    - N = 64 points drawn uniformly from the simplex, drawn anew at every selection;
    - the maximizers of the previous selection;
    - a pool of at most 64 earlier local maximizers, newest first, pairwise distance above 0.08;
    - the center, the 19 interior points of each two-objective edge (multiples of 1/20), and the centers of the three-objective faces.
  - **Screening.** φ is evaluated on all seeds in one batched contraction. Values are normalized by the largest φ at the vertices, the center and the edge and face points. If the best seed is within a relative 1e-8 of val(A), it is returned.
  - **Polishing.** Otherwise the r = 1 best seed is polished by CCP steps. It stops when the predicted improvement δ_c ≤ τ = 1e-8·max{1, φ}, or after c_max = 15 LPs.
  - **LPs.** HiGHS (dual simplex, feasibility tolerances 1e-9), warm-started from the basis of the previous LP when that LP has the same size. LPs with at least 400 rows are solved by constraint generation, which returns an optimum of the full LP:
    - HiGHS solves the LP on a working set of rows: the rows active at the previous solution, the 40 smallest at the previous maximizer, and the smallest row of each column.
    - The 40 most violated remaining rows are added until no row is violated by more than 1e-10.

**Uniform discretization** (Algorithm 6). The grid G_r = {λ ∈ Δ: rλ integral} is visited in snake order, in sweeps of
M_U Adam steps per weight.
- **Sweep 1.** Each weight starts from the starting point of Algorithm 6, Step 2. The rule is the same as in GRAB and runs over θ₀ and the incumbents of the weights visited so far. The Adam state is new, except when the chosen point is the incumbent of a weight within 0.005; the solve then starts from a copy of the Adam state that incumbent was produced with.
- **Later sweeps.** Each weight continues its own trajectory: its own last iterate and Adam state.
- **Bundle.** θ₀ plus the smallest-gradient iterate of every grid weight visited.

**SURF** (K=2).
- N+1 slots are placed at the quantiles of the arc length of the front of objective values F = (F₁, F₂). This is SURF Algorithm 1, eq. (12); the values come with the last gradient of each slot, so there is no extra evaluation. The front uses PCHIP interpolation and damping α = 0.3.
- Each slot runs K_S Adam steps per round from its previous iterate; all slots start at θ₀.
- The output is the N+1 policies of the last round.

**Budget and checkpoints.**
- Each task has one budget B, the same for all methods: 1.05 × the Gradient Calls of the farthest Uniform point (FishWood r=512, Fruit Tree r=6), rounded up.
- All methods are checkpointed on one Gradient-Call schedule: every B/600 calls (K=2) or every B/120 calls (K=6) up to B, and ten times that interval afterwards. Each checkpoint is taken at the end of the first unit that completes after the mark. A unit is a GRAB outer iteration, the M_U steps of one Uniform weight, or the K_S steps of one SURF slot.
- Uniform also records the GN after every sweep and SURF after every round. The stopping rule uses only these values.

**Stopping rule and plotted points.**
- Every Uniform r and SURF N is a separate run. It stops once three conditions hold:
  - its per-sweep (per-round) GN has changed by at most 1% three times in a row;
  - the run is ready: the 95th percentile of the gradient norms at the policies' own weights is at most max(1e-5, 0.25·GN), and for SURF the slot weights have also settled;
  - three further sweeps (rounds) confirm the plateau.
- The plotted point is taken among the Gradient-Call checkpoints before the stop and the stopping checkpoint. It is the earliest one after which the GN stays within 5% of its value at stopping. Gradient Calls, CPU time and GN are read from that checkpoint.
- A run is plotted if it reached its plateau and its point lies within B.

**Settings** (`mogym/config.py`):

| Task | Uniform | SURF | GRAB | Budget B |
|---|---|---|---|---|
| FishWood | M_U=50, lr 0.03, r = 2, 4, …, 512 | K_S=25, lr 0.03, N = 2, 4, …, 128 and 208 | M_A=1, lr 0.01, envelope | 108,000 |
| Fruit Tree (d=6) | M_U=5, lr 0.1, r = 1, …, 6 | – | M_A=5, lr 0.1, CCP (N=64, r=1, c_max=15) | 60,000 |

**Metric.**
- FishWood: exact.
- Fruit Tree: the maximum over a fixed pool of 23,992 weights. The pool has the vertices, the center, 199 points per edge, and 500 random points per face with 3 to 6 objectives. It is a lower estimate, the same for all methods.

**Figures.**
- GRAB is drawn as an orange curve, Uniform as blue squares and SURF as red triangles.
- Each family has a dashed descriptive trend c + a (x/s)^(-p), where s is the median x. The trend is fitted by least squares on the logarithms and drawn from the leftmost to the rightmost point.

## Expected results

`make_figure.py` prints these values.
- The ratio is the baseline's lowest point divided by the final GRAB value.
- In parentheses: the same point divided by the lowest GRAB value within the Gradient Calls of that point, and within its CPU time. The time ratio is the median over the 5 timing runs.

| Task | GRAB (end of budget) | Uniform, lowest point | SURF, lowest point |
|---|---|---|---|
| FishWood | 2.0235e-5 at 108,000 calls / 41.8 s | 2.4495e-4 (r=512), 12.11× (11.50× / 4.94×) | 6.3154e-4 (N=208), 31.21× (29.31× / 12.44×) |
| Fruit Tree | 7.3540e-4 at 60,000 calls / 11.8 s | 1.0273e-3 (r=6), 1.40× (1.40× / 1.01×) | – |

Over the 5 timing runs, the same-time ratios range over [4.89, 5.10] (FishWood Uniform), [12.19, 12.54] (FishWood
SURF) and [1.01, 1.03] (Fruit Tree Uniform). The Gradient-Call values are exact. The CPU times depend on the
machine.

## Notes

- `lambda_solvers.lp` calls HiGHS through SciPy's internal interface (SciPy 1.15), with the same model and options as `scipy.optimize.linprog(method="highs")`. If that interface is unavailable, it falls back to `linprog`, which gives the same solutions but runs more slowly.
