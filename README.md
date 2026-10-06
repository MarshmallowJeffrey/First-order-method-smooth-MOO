# GRAB: MNIST experiments

Code, data and results for the multiclass-classification experiments of the paper *GRAB: Gradient Reuse with Adaptive
Bundles for Smooth Nonconvex Multi-Objective Optimization* (Section 5.1 and Appendix D.1).  The repository reproduces

* **Figure 1**: the worst-case gradient norm max_λ GN(λ, B_t) of GRAB, uniform discretization and SURF against
  gradient calls and wall-clock time, (a) on the digit pair {4,9} (K = 2) and (b) on the digit triple {4,7,9} (K = 3);
* **Figure 2**: the linear scalarization fronts of the same methods;

and the numbers behind them, from the raw runs to the figures.

## Contents

1. [Requirements](#requirements)
2. [Repository structure](#repository-structure)
3. [Quick check](#quick-check)
4. [Figures and tables from the included results](#figures-and-tables-from-the-included-results)
5. [Running the experiments](#running-the-experiments)
6. [Results](#results)
7. [Experimental setup](#experimental-setup)
8. [Reproducibility notes](#reproducibility-notes)
9. [Citation](#citation)

## Requirements

    pip install -r requirements.txt

Python 3.11 or newer with numpy, scipy, PyTorch, matplotlib and highspy (versions in `requirements.txt`).  The tests
and the figures run on a CPU; the full experiments need a GPU with CUDA and float64 support (the paper's runs used one
NVIDIA RTX A5000 per run with 4 CPU cores, Python 3.11, PyTorch 2.3.1, numpy 2.0, scipy 1.14, highspy 1.15).  Apple
MPS is not supported (no float64).  The MNIST training files are included in `data/mnist/` (they are downloaded
again if missing).

## Repository structure

| path | content |
|---|---|
| `abm/` | the package (below) |
| `scripts/` | command-line entry points: runs, analysis, figures, tables, screening, step rules, warm start |
| `results/` | the numbers of the paper's runs (see `results/README.md`) |
| `figures/`, `tables/` | the figures (PDF, PNG) and tables (LaTeX), generated from `results/` |
| `tests/` | reproduction checks |
| `data/mnist/` | the MNIST training files |

| module of `abm/` | content |
|---|---|
| `config.py` | all settings of the experiments (digits, ρ, μ, budget, seeds, checkpoints, configurations, λ-search settings) |
| `data.py`, `model.py`, `objective.py` | MNIST, the network, the objectives F_k and their full and stochastic oracles |
| `steppers.py`, `training.py` | step rules (Adam and others), one SVRG segment, the budget meter, the run record |
| `envelope.py` | Step 1 of GRAB for K = 2: the exact lower envelope (Appendix A.4.1, Algorithms 3-5) |
| `ccp_cg.py` | Step 1 of GRAB for K = 3: Algorithm 2 (multistart CCP, Appendix A.4), its linear programs solved by constraint generation |
| `methods.py` | GRAB, uniform discretization and SURF |
| `ccp.py`, `meter.py` | the worst-case gradient norm of a bundle: exact for K = 2, a lower bound for K = 3 (multistart CCP, simplex grids, SLSQP) |
| `analysis.py`, `fronts.py`, `labels.py` | plateau test, markers, trend fits; fronts; label placement in the figures |
| `screening.py`, `grid.py` | conflict screening of digit pairs and triples; simplex grids |

## Quick check

    python tests/test_reproduce.py      # about 5 minutes on a CPU
    python tests/test_envelope.py       # about 30 seconds
    python tests/test_ccp_cg.py         # a few seconds

`test_reproduce.py` reruns four short runs (K = 2: GRAB, uniform r = 3, SURF N = 3; K = 3: GRAB) against reference
values, reruns the screening of {4,9}, and checks that the configuration statistics and the numbers quoted in the paper
follow from `results/`.  `test_envelope.py` checks the exact K = 2 envelope against the pointwise minimum of the
parabolas, a rebuild from scratch and the meter.  `test_ccp_cg.py` checks the K = 3 λ-search: constraint generation
against the full linear program, the same decisions with both, the steps of Algorithm 2, and when the step rule keeps
its state.

## Figures and tables from the included results

    python scripts/make_figures.py
    python scripts/make_tables.py

| in the paper | file |
|---|---|
| Figure 1(a): worst-case gradient norm, {4,9} | `figures/mnist_worst_gn_k2.pdf` |
| Figure 1(b): worst-case gradient norm, {4,7,9} | `figures/mnist_worst_gn_k3.pdf` |
| Figure 2: linear scalarization fronts, {4,9} (left) and {4,7,9} (right) | `figures/mnist_fronts.pdf` |

| further material | file |
|---|---|
| GRAB and the best configuration of each baseline | `tables/mnist_main.tex` |
| all configurations, {4,9} and {4,7,9} | `tables/mnist_k2_full.tex`, `tables/mnist_k3_full.tex` |
| conflict screening of pairs and triples | `tables/screening_pairs.tex`, `tables/screening_triples.tex` |
| step rules of the improvement step, {4,9} | `tables/step_rules_k2.tex`, `figures/mnist_step_rules_k2.pdf` |
| warm start of GRAB, {4,9} | `tables/warm_start.tex` |

Both scripts read only `results/`; the PDFs are written without a creation date, so a rerun gives the same files.

## Running the experiments

One *leg* is one method with one resolution and one seed, run for B = 480,000 gradient calls:

    python scripts/run.py --K 2 --legs all --device cuda
    python scripts/run.py --K 3 --legs all --device cuda
    python scripts/run.py --K 2 --legs adaptive,uniform:60,surf:38 --seeds 41 --device cuda

K = 2 has 111 legs (uniform discretization r ∈ {2, ..., 64}: 21 values; SURF N ∈ {2, ..., 40}: 15 values; GRAB; three
seeds each) and K = 3 has 48 legs (uniform discretization r ∈ {4, ..., 24}: 15 values; GRAB).  On one RTX A5000 a
K = 2 leg takes about 1 hour and a K = 3 leg about 1.7 hours including the audits (GRAB: about 1 and 1.8 hours),
about 110 and 82 GPU-hours in total.  Legs are independent and can run in parallel; finished legs are skipped.  Each leg writes
`runs/k<K>/<leg>/summary.json` (checkpoints, audited worst-case gradient norm, timings) and `grams.npz` (Gram
matrices, objective values, budget and λ of every bundle point).  Then

    python scripts/analyze.py --K 2 --workers 6
    python scripts/analyze.py --K 3
    python scripts/make_figures.py
    python scripts/make_tables.py

write `results/` and the figures and tables.  The further experiments:

    python scripts/screening.py --K 2          # CPU, about 5 minutes
    python scripts/screening.py --K 3          # CPU, about 25 minutes
    python scripts/step_rules.py --device cuda # eleven step rules x three seeds, B = 10,000, {4,9}
    python scripts/warm_start.py --device cuda # four starts x three seeds, B = 10,000, {4,9}, about 20 minutes

## Results

Final worst-case gradient norm of GRAB (geometric mean over the seeds 41, 42, 43, B = 480,000) and the plateau level of
the best configuration of each baseline (`tables/mnist_main.tex`):

| problem | method | max_λ GN(λ, B_t) | baseline / GRAB |
|---|---|---|---|
| {4,9}, K = 2 | GRAB | 9.82e-4 | |
| | uniform discretization, r = 60 | 6.86e-3 | 7.0x |
| | SURF, N = 38 | 6.27e-3 | 6.4x |
| {4,7,9}, K = 3 | GRAB | 1.57e-2 | |
| | uniform discretization, r = 24 | 9.04e-2 | 5.8x |

A GRAB run takes about 0.9 hours for K = 2 (λ-search: 16 s in total) and 1.2 hours for K = 3 (λ-search: 12 % of the
time), without the audits.

## Experimental setup

* **Problems.**  Objective k is F_k = L_k + ρ L_pool + (μ/2)‖θ‖², where L_k is the mean cross-entropy on the images
  of digit k and L_pool the mean cross-entropy on all images; ρ = 1/9 (K = 2) or 3/17 (K = 3), μ = (1 + ρ) 10⁻³.  The
  first 5,842 training images of each digit (the size of the smallest class), pixels scaled to [0, 1].  The digits
  were chosen by a conflict screening of all pairs and triples (`scripts/screening.py`).
* **Network.**  A locally connected layer of 64 units (each sees one 5x5 block of the 28x28 image; the blocks start
  on an 8x8 grid), a dense layer of 96 units and K outputs, softplus activations, float64; d = 8,098 (K = 2) and
  8,195 (K = 3).  He initialization; all methods and seeds start from the same point.
* **Improvement steps (all methods).**  A segment is one SVRG epoch: the full gradient at the anchor, then 12 (K = 2)
  or 18 (K = 3) steps on class-stratified mini-batches of 1,024 images with Adam (α = 10⁻³, β₁ = β₂ = 0.9), then
  one full evaluation at the end point, which enters the bundle.  A segment that increases F_λ is rejected: the point
  stays, Adam's moments are cleared and α is halved (the fifth rejection in a row is accepted).  Five segments per
  decision (GRAB), grid visit (uniform discretization) or slot and round (SURF).  Budget: one full Jacobian costs K
  gradient calls, a mini-batch gradient b·K/n.
* **GRAB.**  Every decision chooses λ on the current bundle and runs five segments from the last accepted point; Adam's
  state is kept while λ is unchanged (for K = 3, a change of at most 10⁻⁸ in ℓ1 counts as none: Algorithm 2 returns a
  point chosen again with rounding noise).  The runs use no tolerance ε and stop when the budget is spent.
  * K = 2: λ maximizes the exact lower envelope of the bundle's parabolas (Appendix A.4.1); the envelope is updated
    with the new points of each decision.
  * K = 3: Algorithm 2 with N = 2,000 uniform seeds, the K vertices, λ_A and the distinct local maximizers of the
    previous decision; the r = 10 best seeds more than 0.05 apart in ℓ1 are polished by CCP steps, each stopping when
    the predicted improvement is at most 10⁻⁸ max{1, φ(λ_c)} or after 100 linear programs.  Every linear program is
    solved by HiGHS (dual simplex) with constraint generation: a working set of the 200 rows smallest at the current
    point and the smallest row of each column, enlarged by the most violated rows until no row is violated by more
    than 10⁻¹² max{1, |t|}, which yields an optimum of the full program.
* **Baselines.**  Uniform discretization: the snake-ordered grid of resolution r on Δ_K, visited forward and
  backward; the first visit of a node starts from the last accepted point of the run, later visits continue from the
  node's own point and Adam state.  SURF (K = 2): N + 1 slots at the quantiles of the
  arc-length distribution of the front, updated after every round (damping 0.3).
* **Worst-case gradient norm.**  At checkpoints (K = 2: every 250 gradient calls up to 20,000, every 1,000 up to
  80,000, then every 2,000; K = 3: every 1,000 up to 80,000, then every 2,000), audited after training on the bundle
  prefixes.  K = 2: exact (the lower envelope on 200,001
  weights, polished, with a certified upper bound).  K = 3: a lower bound (two multistart CCP searches with 8,192
  seeds and a simplex grid of resolution 500 at every checkpoint; SLSQP, a third CCP search and a grid of resolution
  1,000 at B/8, B/4, B/2 and B).  Every series is repaired by its suffix maximum.
* **Markers and plateau.**  A run plateaus if g(B/4) ≤ 1.05 g(B).  Its marker is y = g(B) and x = the first budget with
  g ≤ 1.05 y (K = 2: located to one segment by bisection over the bundle prefixes).  A configuration is drawn if at
  least two of its three seeds plateau; its marker is the geometric mean over the seeds.  The dashed curves in
  Figure 1 are descriptive fits c + a (x/s)^(-p) of the markers.
* **Fronts.**  Per method the non-dominated training objective values of all visited points, averaged over the
  seeds; uniform discretization r = 60 and SURF N = 38 (K = 2) and r = 24 (K = 3).

All settings are in `abm/config.py`.

## Reproducibility notes

* On a CPU the code is deterministic: the quick check reproduces its reference values bit for bit on the machine that
  produced them, and up to floating-point differences of the BLAS elsewhere.  Floating-point reductions on a GPU are
  not bit-reproducible across GPU models and library versions, so rerun values agree with `results/` up to the
  run-to-run variation; wall-clock times depend on the hardware.
* Some baseline legs of the paper were run in two parts (the first 240,000 or 320,000 gradient calls, then continued
  from the saved state).  The continuation reproduces a single run bit for bit, so `scripts/run.py` runs every leg in
  one go.

## Citation

    @inproceedings{grab2027,
      title     = {{GRAB}: Gradient Reuse with Adaptive Bundles for Smooth Nonconvex Multi-Objective Optimization},
      author    = {Anonymous},
      booktitle = {Under review at AISTATS 2027},
      year      = {2027}
    }

The MNIST data: Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, Gradient-based learning applied to document
recognition, Proceedings of the IEEE 86(11), 1998.
