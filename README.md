# Adaptive bundle method: MNIST experiments

Code and data for the MNIST experiments of the paper (Section 4.1 and Appendix C.1): the adaptive bundle method
compared with uniform discretization and SURF on multiclass classification, with one objective per digit class.

* {4,9} (K = 2) and {4,7,9} (K = 3), chosen by a conflict screening of all digit pairs and triples;
* objective k: F_k = L_k + rho L_pool + (mu/2) ||theta||^2, where L_k is the cross-entropy on class k and L_pool the
  cross-entropy on all images; rho = 1/9 (K = 2) or 3/17 (K = 3), mu = (1 + rho) 1e-3;
* a small network (a locally connected layer of 64 units on 5x5 blocks, a dense layer of 96 units, softplus;
  d = 8,098 for K = 2 and 8,195 for K = 3), 5,842 training images per class, float64;
* all methods: the same initial point, SVRG segments (one full gradient, then 12 or 18 mini-batch steps of 1,024
  images) with Adam (alpha = 1e-3, beta_1 = beta_2 = 0.9) and a descent safeguard, 5 segments per decision / grid
  visit / SURF slot and round, a budget of 480,000 gradient calls, sampling seeds 41, 42, 43;
* the score is the worst-case gradient norm of the bundle of all visited points, max over lambda of GN(lambda, B_t):
  exact for K = 2, a lower bound for K = 3.

## Layout

| path | content |
|---|---|
| `abm/` | the package: data, network, objectives, step rules, SVRG segments, CCP lambda-search, the three methods, the worst-case gradient norm meter, analysis, fronts, screening |
| `scripts/` | command-line entry points (below) |
| `results/` | the numbers of the paper's runs: per run the audited worst-case gradient norm at every checkpoint, the plateau test and the marker; configuration statistics; front points; step-rule and screening results |
| `figures/`, `tables/` | the figures (PDF, PNG) and tables (LaTeX) of the paper, generated from `results/` |
| `tests/` | short reproduction checks |
| `data/mnist/` | the MNIST training files (downloaded if missing) |

## Install

    pip install -r requirements.txt

A GPU with CUDA is needed for the full experiments (the paper used NVIDIA RTX A5000 GPUs); everything else runs on a
CPU.  Apple MPS is not supported (no float64).

## Check the installation (CPU, about 6 minutes)

    python tests/test_reproduce.py

It reruns four short runs against reference values of the original code, reruns the screening of {4,9}, and checks
that the numbers quoted in the paper follow from `results/`.

## Figures and tables from the included results

    python scripts/make_figures.py
    python scripts/make_tables.py

| paper | file |
|---|---|
| Figure: worst-case gradient norm, {4,9} and {4,7,9} | `figures/mnist_worst_gn_k2.pdf`, `figures/mnist_worst_gn_k3.pdf` |
| Figure: linear scalarization fronts | `figures/mnist_front_k2.pdf`, `figures/mnist_front_k3.pdf` |
| Table: best configuration of each baseline | `tables/mnist_main.tex` |
| Appendix: screening of pairs and triples | `tables/screening_pairs.tex`, `tables/screening_triples.tex` |
| Appendix: step rules (table and figure) | `tables/step_rules_k2.tex`, `figures/mnist_step_rules_k2.pdf` |
| Appendix: all runs, {4,9} and {4,7,9} | `tables/mnist_k2_full.tex`, `tables/mnist_k3_full.tex` |

## Rerun the experiments

1. Screening (CPU, 5 min for the pairs, 25 min for the triples):

       python scripts/screening.py --K 2
       python scripts/screening.py --K 3

2. Step rules on {4,9} (the adaptive method with eleven step rules, three seeds each; 33 short runs):

       python scripts/step_rules.py --device cuda

3. Main runs.  One leg = one method, one resolution, one seed:

       python scripts/run.py --K 2 --legs all --device cuda
       python scripts/run.py --K 3 --legs all --device cuda
       python scripts/run.py --K 2 --legs adaptive,uniform:60,surf:38 --seeds 41 --device cuda

   K = 2 has 111 legs (uniform r in {2, ..., 64}: 21 values, SURF N in {2, ..., 40}: 15 values, adaptive; three
   seeds each), K = 3 has 54 legs (uniform r in {2, ..., 24}: 17 values, adaptive).  On one RTX A5000 a K = 2 leg
   takes about 1 hour (adaptive: 5 hours; the CCP lambda-search runs on the CPU), a K = 3 leg about 1.7 hours
   (adaptive: 3.8 hours) including the audits: about 120 and 95 GPU-hours in total.  Legs are independent and can
   run in parallel; finished legs are skipped.

4. Analysis, figures and tables:

       python scripts/analyze.py --K 2 --workers 6
       python scripts/analyze.py --K 3
       python scripts/make_figures.py
       python scripts/make_tables.py

The settings of all experiments are in `abm/config.py`.

## Notes on reproducibility

* On a CPU the code is deterministic.  Floating-point reductions on a GPU are not bit-reproducible across GPU models
  and library versions, so rerun values agree with `results/` up to the run-to-run variation, not bit for bit; the
  wall-clock times depend on the hardware.
* Some legs of the paper were run in two parts (the first 240,000 or 320,000 gradient calls, then continued with the
  saved state).  The continuation reproduces a single run bit for bit, so `scripts/run.py` runs every leg in one go.
* The K = 3 audits use SLSQP (scipy) at the budget levels B/8, B/4, B/2 and B, as in the paper's runs.
