# MO-Gymnasium experiments: adaptive bundle method vs. uniform discretization and SURF

This folder reproduces the MO-Gymnasium results of the paper *An Adaptive First Order Bundle Method*:
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

The script runs every task serially with one numerical thread and takes about 12 minutes on a laptop.
Outputs:

- `results/<task>/...`: one JSON file per run (settings, checkpoints with Gradient Calls, CPU time and GN, timings) and one `.npz` file with the returned policies.
- `figures/<task>_convergence.png`: the figure.
- `figures/<task>_summary.json`: the number report.

The runs are deterministic.
CPU times depend on the machine and vary by a few percent between repeated runs.  Comparisons in the
Time panel between points that are only a few milliseconds apart can therefore flip.

## Running a single task

```bash
python scripts/run_uniform.py  dst            # Uniform, every plotted r, each run to its GN plateau
python scripts/run_surf.py     dst            # SURF (K=2 tasks), every plotted N
python scripts/run_adaptive.py dst            # adaptive bundle method (budget from the baseline points)
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
| `mogym/adaptive.py` | adaptive bundle method (paper Algorithm 1) |
| `mogym/uniform.py` | uniform discretization (paper Algorithm 7), each grid weight run to the GN plateau |
| `mogym/surf.py` | SURF (Algorithm 1 of the SURF paper), K=2 |
| `mogym/lambda_solvers.py` | preference-weight solvers: exact envelope (K=2), simplicial branch-and-bound (K=3), multistart CCP of the paper appendix (K=6) |
| `mogym/metrics.py` | reported metric: exact (K=2), certified interval (K=3), fixed pool of 23,992 weights (K=6) |
| `mogym/plateau.py` | stopping rule of the baseline runs and definition of the plotted point |
| `mogym/recorder.py` | checkpoints; training time excludes the metric evaluation |
| `mogym/config.py` | settings of the reported runs |
| `mogym/points.py` | reading runs back: plotted points, adaptive curve, adaptive budget |

## Protocol

**Common.**
- All methods start at θ₀ = 0 and use Adam with (β₁, β₂, ε) = (0.9, 0.999, 1e-8).
- One evaluation of the K gradients counts as K Gradient Calls; for SURF, one scalarized step counts as K.
- Time is the training process CPU time. It includes preference selection and the SURF weight updates, and excludes the metric evaluation at checkpoints.
- Adam state: a method keeps the Adam state of an inner solve while its weight is unchanged, i.e. moves by at most 0.005 in every coordinate, and starts a new state otherwise.

**Adaptive bundle method.**
- At each outer iteration it selects λ_t maximizing GN(λ, B) over the current bundle.
- It runs M_A Adam steps from the Algorithm 2 warm start and adds the iterate with the smallest ‖∇F_{λ_t}‖.

**Uniform discretization.**
- The grid G_r is visited in snake order. In the first sweep each weight starts from the preceding weight's last iterate; afterwards each weight continues its own trajectory.
- The bundle is θ₀ plus the smallest-gradient iterate of every grid weight.

**SURF.**
- N+1 slots are placed at the quantiles of the reward-space arc length, with PCHIP interpolation and damping α = 0.3.
- Each slot runs K_S Adam steps per round, and the output is the last round.

**Stopping rule and plotted points.**
- Every Uniform r and SURF N is a separate run. It stops once its GN has changed by at most 1% three times in a row, the run is ready, and three further checkpoints confirm the plateau.
  - Ready means: the 95th percentile of the gradient norms at the policies' own weights is at most max(1e-5, 0.25·GN). For SURF the slot weights must also have settled.
- The plotted point is the earliest checkpoint after which the GN stays within 5% of its value at stopping.
- The adaptive budget is 1.05 × the Gradient Calls of the farthest plotted point, rounded up.

**Settings** (`mogym/config.py`; chosen by the tuning procedure of the paper appendix):

| Task | Uniform | SURF | Adaptive | Adaptive budget |
|---|---|---|---|---|
| FishWood | M_U=50, lr 0.03, r = 2,4,…,256 | K_S=25, lr 0.03, N = 2,4,…,128 | M_A=10, lr 0.01, envelope | 81,000 |
| DST | M_U=5, lr 0.3, r = 2,4,…,512 | K_S=25, lr 0.1, N = 4,8,…,64 | M_A=10, lr 0.1, envelope | 96,000 |
| Breakable Bottles | M_U=25, lr 0.1, r = 1,…,24 | – | M_A=25, lr 0.03, K=3 subdivision (gap 0.1, 500 nodes) | 27,000 |
| Fruit Tree (d=6) | M_U=25, lr 0.03, r = 1,…,6 | – | M_A=10, lr 0.1, periodic CCP with LP warm start | 150,000 |

The Fruit Tree preference selector is the multistart CCP of the paper appendix with these settings:
- Ordinary selections: 128 random seeds (redrawn at every selection), up to two polished starts, at most 30 CCP updates.
- Every tenth selection: 1,024 seeds, eight starts, 100 updates.
- Extra starts: the center, the previous maximizers, 19 points on each two-objective edge, and the centers of the three-objective faces.
- Every LP is warm-started from the previous basis.

## Expected results

`make_figure.py` prints these values; the ratio is the baseline's lowest point divided by the final adaptive value.

| Task | Adaptive (end of budget) | Uniform, lowest point | SURF, lowest point |
|---|---|---|---|
| DST | 6.5277e-6 | 4.8588e-5 (r=512), 7.44× | 1.3043e-4 (N=64), 19.98× |
| FishWood | 1.9520e-4 | 4.8630e-4 (r=256), 2.49× | 1.0092e-3 (N=128), 5.17× |
| Breakable Bottles | [4.4503, 4.4714]e-4 | [1.9295, 1.9382]e-3 (r=24), 4.34× | – |
| Fruit Tree | 6.6208e-4 | 1.0100e-3 (r=6), 1.53× | – |

## Notes

- `lambda_solvers.lp` calls HiGHS through SciPy's internal interface (SciPy 1.15), with the same model and options as `scipy.optimize.linprog(method="highs")`. If that interface is unavailable it falls back to `linprog`, which gives the same solutions but runs more slowly.
- The Fruit Tree metric is a lower estimate: the maximum over a fixed pool of 23,992 weights, the same pool for all methods.
