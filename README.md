# MO-Gymnasium experiments: GRAB vs. uniform discretization and SURF

This folder reproduces the MO-Gymnasium results of the paper *GRAB: Gradient Reuse with Adaptive Bundles for
Smooth Nonconvex Multi-Objective Optimization* for FishWood (K=2) and Fruit Tree of depth 6 (K=6): the convergence
figures of both tasks, the per-resolution comparisons for Fruit Tree, and the numbers.

## What is compared

Every method builds a bundle B of points with their objective values and component gradients. Its quality is the
worst-case gradient norm

    GN*(B) = max_{λ ∈ simplex} GN(λ, B),   GN(λ, B) = min_{θ ∈ B} ‖∇F_λ(θ)‖,

plotted against Gradient Calls and training CPU time.

- **FishWood (K=2).** GN*(B) is computed exactly (lower envelope of parabolas) for every method.
- **Fruit Tree (K=6).** GN*(B) is not computed exactly. Instead:
  - every **baseline** value is GN(λ, B) at an actual weight λ (the best found over a fixed pool of weights refined by
    local CCP ascent), hence a **lower bound** on the baseline's GN*;
  - for **GRAB**, an **upper bound** on GN*(B_t) is computed at every checkpoint by branch and bound over the simplex.

  Where GRAB's upper bound lies below a baseline's lower bound, GRAB's bundle has the smaller worst-case gradient
  norm. The figures draw exactly these two quantities.

## Setup

Python 3.13 was used. Install the packages with

```bash
pip install -r requirements.txt
```

`mo-gymnasium` is used only to read the Fruit Tree rewards. All objectives and gradients are computed exactly from
the finite models; no trajectories are sampled.

## Reproducing everything

```bash
./run_all.sh
```

The script runs every configuration of both tasks, the CPU-time repeats, the Fruit Tree upper bounds and the figures.
The runs and the timing repeats are serial with one numerical thread (about 15 minutes on the machine below,
mostly the evaluation of the Uniform runs, which is not part of the measured time; each further timing repeat takes
about as long again). The upper bounds are computed afterwards with six processes (about 11 minutes).

Outputs:

- `results/<task>/...`: one JSON file per run (settings, checkpoints with Gradient Calls, CPU time and GN, timings)
  and one `.npz` file with the returned policies (GRAB and SURF: their Jacobians; Uniform: the Gram matrices J J' of
  its bundle points).
- `results/fruittree_d6/adaptive/adaptive_metric.json`, `adaptive_upper.json`: GRAB's lower estimate and upper bound
  at every checkpoint.
- `results/<task>/timing.json`: the CPU times of the timing repeats.
- `figures/<task>_convergence.png` and `figures/<task>_summary.json`: the convergence figure and its numbers.
- `figures/fruittree_d6_bounds_grid_calls.png`, `..._cpu.png`, `fruittree_d6_bounds_grid.json`: the comparison along
  every Uniform run, one panel per resolution r.

The reported runs used an Apple M1 (8 CPU cores: 4 performance, 4 efficiency) with 8 GB memory, macOS 15.7,
Python 3.13.5, NumPy 2.1.3 (OpenBLAS 0.3.21), SciPy 1.15.3 (HiGHS), Gymnasium 1.3.0 and MO-Gymnasium 1.3.2,
with one numerical thread and no GPU.

The runs are deterministic: Gradient Calls, GN and the selected weights are identical in every repeat; only CPU
time varies. With `time_repeats.py` the figures plot the median CPU time over the repeats and the summaries report
the range; without it they use the CPU time of the single stored run. The upper bounds stop at a time limit per
checkpoint and therefore depend slightly on the machine. **The figures and numbers below are from a single timed
run; the timing repeats are still to be run.**

## Running a single task

```bash
python scripts/run_uniform.py     fishwood       # Uniform, every configured r, each run to its GN plateau
python scripts/run_surf.py        fishwood       # SURF (K=2), every configured N, each run to its GN plateau
python scripts/run_adaptive.py    fishwood       # GRAB, to the budget B of the task
python scripts/time_repeats.py    fishwood fruittree_d6 --repeats 5   # CPU time over 5 runs
python scripts/upper_bounds.py    fruittree_d6   # K>2: upper bounds on GRAB's GN* at every checkpoint
python scripts/make_figure.py     fishwood       # convergence figure + summary
python scripts/make_bounds_grid.py fruittree_d6  # K>2: comparison along every Uniform run
```

The tasks are `fishwood` and `fruittree_d6`; SURF is run for FishWood only.

Each run script accepts `--values` (a subset of r or N) and `--results <dir>`. A stored run is reused only if its
identity matches: settings, source code, package versions, model and arrays (`mogym/identity.py`); otherwise the
script stops instead of skipping or overwriting the run. The upper bounds are tied in the same way to the arrays of
the GRAB run, the evaluator and the bound settings, and the figure scripts stop if they do not match.
`time_repeats.py` reruns every configuration and checks that each repeat reproduces the stored run exactly.

## Code

| File | Contents |
|---|---|
| `mogym/envs.py` | finite MDP models of the two tasks, with γ and τ |
| `mogym/oracle.py` | exact objectives F_k (discounted, KL-regularized) and their Jacobians |
| `mogym/adam.py` | Adam, the inner solver of all methods |
| `mogym/adaptive.py` | GRAB |
| `mogym/uniform.py` | uniform discretization, run to the GN plateau |
| `mogym/surf.py` | SURF (Jiang et al.), K=2, run to the GN plateau |
| `mogym/lambda_solvers.py` | preference selection: exact envelope (K=2) and multistart CCP (K>2), with its options; the LP solver |
| `mogym/metrics.py` | GN*(B): exact for K=2; for K>2 the lower bound from a fixed pool of weights refined by CCP |
| `mogym/bounds.py` | K>2: upper bounds on GRAB's GN*(B_t) at every checkpoint (branch and bound) |
| `mogym/plateau.py` | stopping rule of the baseline runs and the plotted point |
| `mogym/recorder.py` | checkpoints; training time excludes the metric evaluation |
| `mogym/identity.py` | identity of a stored run |
| `mogym/config.py` | settings of the reported runs and of the upper bounds |
| `mogym/points.py` | reading runs back: plotted points, the GRAB curve and its upper bounds |
| `scripts/make_figure.py` | convergence figure and summary |
| `scripts/make_bounds_grid.py` | K>2: comparison along every Uniform run |
| `scripts/upper_bounds.py` | K>2: computes and stores the upper bounds |
| `scripts/labels.py` | placement of the number labels next to the points |

## Protocol

**Common.**
- All methods start at θ₀ = 0 (the uniform policy). Its objective values and gradients are evaluated once and counted.
- All methods use Adam with (β₁, β₂, ε) = (0.9, 0.999, 1e-8) and bias correction.
- One evaluation of the K component gradients counts as K Gradient Calls; for SURF, one scalarized step also counts
  as K. A point added to a bundle is an iterate whose gradients were already evaluated; it costs no further calls.
- Preference selection, the starting-point rule and the SURF weight updates use stored values only and make no
  oracle calls.
- Time is the CPU time of the training process. It includes preference selection, the starting-point rule and the
  SURF weight updates. It excludes the metric evaluation at checkpoints and the upper bounds.

**Trajectories and Adam states.** Every method advances optimization trajectories; a trajectory has its last
iterate and its Adam state (m, v, t). A trajectory is continued, from its last iterate with its Adam state, only
while its weight has moved by at most 0.005 in every coordinate since it last ran; otherwise a solve starts with a
new Adam state. In GRAB a trajectory is resumed when the starting point chosen in step 2 belongs to it; in Uniform
every grid weight continues its own trajectory after the first sweep; in SURF every slot continues its own iterate.

**Bundles.** Every inner solve of GRAB and Uniform returns the iterate with the smallest scalarized gradient norm
among its iterates, and the bundle keeps every returned point: GRAB adds one point per outer iteration (M_A Adam
steps), Uniform one point per sweep of every grid weight (M_U Adam steps). SURF's output is, by its definition, the
N+1 current slot policies.

**GRAB.** No tolerance ε is used: the run stops when the budget B is spent, and preference selection only has to
return a weight with positive GN, so neither the LP certificate nor a global fallback is used. Each outer iteration:
1. Select λ_t (see below).
2. Choose the starting point θ₀^(t): the bundle point minimizing F_λ − ‖∇F_λ‖²/(2L_λ), with L_λ = Σ_k λ_k L_k from
   empirical smoothness estimates L_k (finite-difference curvature around θ₀; they enter only this rule). Ties go to
   the first minimizer in the stored order.
3. Run M_A Adam steps on F_{λ_t}. If θ₀^(t) belongs to a trajectory last run at a weight within 0.005 of λ_t in every
   coordinate, the solve resumes that trajectory from its last iterate with its Adam state; otherwise a new
   trajectory starts at θ₀^(t) with a new Adam state.
4. Add the iterate with the smallest ‖∇F_{λ_t}‖ to the bundle.

**Preference selection.**
- K=2: with λ = (x, 1−x), every bundle point gives a convex quadratic in x. The lower envelope is kept as pieces and
  updated when a point is added (each piece is split where the new quadratic crosses it). The maximum is attained at
  a piece end; the selected weight is the piece end of largest value (the first one in case of ties). The selection
  is exact up to rounding, with two numerical guards (a piece is left unchanged if the new quadratic exceeds the
  stored one on it by more than 1e-9 times the sum of the absolute coefficients of their difference; roots within
  2e-15 of a piece end are ignored).
- K=6: multistart CCP, with these steps at a *full* selection:
  - **Upper bound.** The LP max_λ min_i (Aλ)_i with A_ik = ‖∇F_k(θ_i)‖² gives its maximizer λ_A and, from its dual, an
    upper bound on the selection problem. It is solved again only when a new bundle point i has (Aλ_A)_i below the
    stored value; otherwise λ_A, the value and the bound are unchanged.
  - **Seeds.** The K vertices and λ_A; N = 64 points drawn uniformly from the simplex anew at every selection; the
    polished point of the previous selection; a pool of at most 16 earlier local maximizers (newest first, pairwise
    distance above 0.08); the center, the 19 interior points of each two-objective edge (multiples of 1/20) and the
    centers of the three-objective faces.
  - **Seed values.** φ(λ) = min_i λ'Q_iλ is evaluated on all seeds, normalized by its largest value at the vertices,
    the center and the edge and face points. If the best seed is within a relative 1e-8 of the upper bound, it is
    returned.
  - **CCP.** Otherwise the best seed is improved by CCP steps (an LP over the linearizations, then a move to its
    maximizer). It stops when the predicted improvement δ_c ≤ 1e-8·max{1, φ} (normalized φ), or after c_max = 3 LPs.
    The selected weight is the point of largest φ found; it updates the pool.
  - **Selections without LPs.** A full selection is made at every 100th call. At the other calls the selected weight
    is the best seed (the fixed seeds, the last λ_A, the previous selection, the pool and 64 fresh draws), without any
    LP, unless its φ is below 0.5 times that of the last full selection; then a full selection is made.
  - **Seed screening.** A fresh seed is evaluated on blocks of bundle points, newest first, and dropped as soon as its
    value cannot exceed that of the seeds before it. This gives the same selected weight as evaluating every seed in
    full.
  - **LPs.** HiGHS (dual simplex, presolve off, feasibility tolerances 1e-9), warm-started from the basis of the
    previous LP of the same size. An LP with at least 400 rows is solved on a working set of rows (those active at the
    previous optimum, the 40 smallest at the previous maximizer, the smallest of each column, and the rows of the
    previous LP whose slacks were nonbasic, with their basis statuses), without adding further rows. This is a
    relaxation: its maximizer need not be optimal for the full LP, but its dual still gives a valid upper bound, and φ
    is always evaluated exactly on every bundle point; a CCP step that lowers φ ends the run.
  - These options are parameters of `lambda_solvers.CCP` (`lazy`, `lazy_rho`, `screen`, `presolve`, `warm_cg`,
    `relaxed`, `maxiter`, `tau`). The defaults give the plain procedure with exact LPs at every call; the evaluation of
    GN* and the upper bounds always use exact LPs.

**Uniform discretization.** The grid G_r = {λ ∈ simplex: rλ integral} (C(r+K−1, K−1) weights) is visited in snake
order, in sweeps of M_U Adam steps per weight.
- **Sweep 1.** Each weight starts from the starting point of GRAB's step 2, chosen over θ₀ and the points returned so
  far, with a new Adam state, unless the chosen point was returned by a weight within 0.005; the solve then starts
  from a copy of the Adam state that point was produced with.
- **Later sweeps.** Each weight continues its own trajectory, from its own last iterate with its Adam state, until
  the run stops (see the stopping rule).
- **Bundle.** θ₀ and the smallest-gradient iterate of every sweep of every weight, kept in the order returned.

**SURF** (K=2).
- N+1 slots are placed at the quantiles of the arc length of the front of objective values F = (F₁, F₂) (Jiang et
  al., eq. (12)); the values come with the last gradient of each slot, so there is no extra evaluation. The front
  uses PCHIP interpolation and damping α = 0.3.
- Each slot runs K_S Adam steps per round from its previous iterate; all slots start at θ₀.
- The output is the N+1 policies of the last round.

**Budget and checkpoints.**
- Each task has one budget B, the same for all methods: 1.05 × the Gradient Calls of the farthest Uniform point
  (FishWood r=512, Fruit Tree r=6), rounded up to a multiple of 1e3 (FishWood) or 6e3 (Fruit Tree).
- All methods are checkpointed on one Gradient-Call schedule: every B/600 calls (K=2) or every B/120 calls (K=6) up
  to B, and ten times that interval afterwards. Each checkpoint is taken at the end of the first unit that completes
  after the mark. A unit is a GRAB outer iteration, the M_U steps of one Uniform weight, or the K_S steps of one SURF
  slot. With M_A = M_U the GRAB and Uniform checkpoints fall on the same Gradient Calls.
- Uniform also records the GN after every sweep and SURF after every round.

**Stopping rule and plotted points.**
- Every Uniform r and SURF N is a separate run. The stopping rule is checked after every block of sweeps (rounds)
  that gives each weight (slot) 50 Adam steps: every 50/M_U sweeps, every 50/K_S rounds. The run stops once
  - its GN at these checks has changed by at most 1% three times in a row;
  - the run is ready: the 95th percentile of the gradient norms at the policies' own weights is at most
    max(1e-5, 0.25·GN), and for SURF the slot weights have moved by at most 0.005 from round to round;
  - three further checks confirm the plateau.
- The plotted point is taken among the Gradient-Call checkpoints before the stop and the stopping checkpoint. It is
  the earliest one after which the GN stays within 5% of its value at stopping. Gradient Calls, CPU time and GN are
  read from that checkpoint.
- A run is plotted if it reached its plateau and its point lies within B.

**Settings** (`mogym/config.py`):

| Task | Uniform | SURF | GRAB | Budget B |
|---|---|---|---|---|
| FishWood | M_U=5, lr 0.03, r = 2, 4, …, 512 | K_S=25, lr 0.03, N = 2, 4, …, 256 | M_A=5, lr 0.003, envelope | 157,000 |
| Fruit Tree (d=6) | M_U=5, lr 0.1, r = 1, …, 6 | – | M_A=5, lr 0.1, CCP (N=64, one start, c_max=3, pool 16; full selection every 100th call, screening, relaxed LPs) | 60,000 |

SURF uses the doubling values of N whose point lies within B. The step counts and learning rates were chosen once
per method from M ∈ {5, 10, 25, 50} and lr ∈ {1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3} by one rule (pilot runs to a fixed
budget, scored by the mean log GN over Gradient Calls and CPU time); for Fruit Tree the CCP options of GRAB were then
compared in the same way.

**Evaluation of GN\* for Fruit Tree.**
- *Lower bound (baselines; GRAB's lower estimate).* A fixed pool of 23,992 weights: the vertices, the center, 199
  points per edge, and 500 random points (fixed seed) per face with 3 to 6 objectives. From each of the 32 best pool
  weights that are more than 0.08 apart (Euclidean), CCP steps with exact LPs climb to a local maximum, stopping when
  δ_c ≤ 1e-8·max{1, φ} or after 200 LPs. The value is the largest GN found, at an actual weight.
- *Upper bound (GRAB).* On a subsimplex S with vertices v_1, …, v_K, every λ'Q_iλ is convex, so it is at most the
  linear interpolation of its vertex values; hence max_S min_i λ'Q_iλ is at most the value of the LP with
  A_ij = v_j'Q_iv_j, and any distribution w over its rows bounds that value by max_j (w'A)_j. w is the LP dual
  (projected onto the simplex), so the bound does not depend on the accuracy of the LP solution. The subsimplex of
  largest bound is split at the midpoint of its longest edge. The bound of a bundle is the largest bound over the
  subsimplices.
  - *Along the checkpoints.* The bundle only grows, so a subsimplex bound computed for an earlier bundle stays valid
    (its w, extended by zeros, is still a distribution over the rows). One branch and bound is therefore carried
    along the checkpoints: at each checkpoint the subsimplices of largest bound are re-bounded with the current
    bundle or split, until the bound is within 1% of the lower estimate or 20 s (six processes) are spent.
  - Subsimplices whose bound is within 1% of the smallest lower estimate are dropped (they cannot matter later); the
    reported bound is at least that value. It is multiplied by 1 + 1e-9 against floating-point rounding (a numerically
    certified bound) and is non-increasing over the checkpoints.

**Figures.**
- `<task>_convergence.png`: GRAB in orange (FishWood: exact curve; Fruit Tree: its upper bound as a step line),
  Uniform as blue squares and SURF as red triangles at their plotted points (Fruit Tree: their values, lower bounds).
  Each baseline family has a dashed descriptive trend c + a (x/s)^(-p), s the median x, fitted by least squares on the
  logarithms and drawn from the leftmost to the rightmost point. Fruit Tree comparisons match a baseline point with
  GRAB's last checkpoint within the same Gradient Calls or the same CPU time.
- `fruittree_d6_bounds_grid_*.png`: one panel per r. At every checkpoint of the Uniform run within B, the Uniform value
  lb(UD) against GRAB's upper bound ub(GRAB) at its last checkpoint within the same Gradient Calls (`_calls`) or CPU
  time (`_cpu`); green where lb(UD) > ub(GRAB). The panel title is the smallest ratio lb(UD)/ub(GRAB) over these
  checkpoints, marked and enlarged in an inset. In the time figure, Uniform checkpoints earlier than GRAB's first
  checkpoint after θ₀ are drawn as open circles and left out of the ratio (GRAB has only θ₀ there).

## Expected results

`make_figure.py` and `make_bounds_grid.py` print these values. The ratio is the baseline's lowest point divided by
GRAB's value at B; in parentheses, the same point divided by GRAB's value at its last checkpoint within the Gradient
Calls of that point, and within its CPU time. For Fruit Tree, GRAB's value is its upper bound, so the ratios are lower
bounds on the ratios of the true GN*.

| Task | GRAB at B | Uniform, lowest point | SURF, lowest point |
|---|---|---|---|
| FishWood | 3.1905e-5 at 157,000 calls / 18.4 s | 1.8808e-4 (r=512, 149,032 calls / 13.9 s): 5.89× (5.65× / 4.87×) | 5.0727e-4 (N=256, 150,902 calls / 13.1 s): 15.90× (15.34× / 12.48×) |
| Fruit Tree | upper bound 8.1185e-4 at 60,000 calls / 2.8 s | 1.0012e-3 (r=6, 55,026 calls / 2.5 s): ≥ 1.23× (≥ 1.23× / ≥ 1.23×) | – |

- Baseline points that GRAB's curve (FishWood) or upper bound (Fruit Tree) lies below, within the same Gradient
  Calls / the same CPU time: FishWood Uniform 4/9 / 4/9 (r = 64, …, 512; the coarse grids reach their plateau within
  23,102 calls, while GRAB is still descending), SURF 7/8 / 6/8; Fruit Tree Uniform 6/6 / 5/6 (r = 1 at 0.02 s, before
  GRAB's first checkpoint after θ₀).
- Fruit Tree upper bounds: within 0.9%–4.9% of GRAB's lower estimate at checkpoints 2–121 (median 1.0%).
- Fruit Tree along every Uniform run within B (`make_bounds_grid.py`), smallest lb(UD)/ub(GRAB) for r = 1, …, 6:
  same Gradient Calls 1.200, 1.190, 1.212, 1.186, 1.089, 1.064; same CPU time (from GRAB's first checkpoint after θ₀,
  0.023 s; one Uniform checkpoint per r is earlier) 1.200, 1.003, 1.096, 1.186, 1.144, 1.148.

The Gradient-Call values are exact. The CPU times depend on the machine.

## Notes

- `lambda_solvers.lp` calls HiGHS through SciPy's internal interface (SciPy 1.15), with the same model and options as
  `scipy.optimize.linprog(method="highs")`. If that interface is unavailable, it falls back to `linprog`, which gives
  the same solutions but runs more slowly.
- HiGHS runs on a global task scheduler fixed by the first solve in the process. If another HiGHS user in the same
  process (e.g. `scipy.optimize.linprog` without a `threads` option) started it with another number of threads,
  `lambda_solvers` resets it once and solves again.
