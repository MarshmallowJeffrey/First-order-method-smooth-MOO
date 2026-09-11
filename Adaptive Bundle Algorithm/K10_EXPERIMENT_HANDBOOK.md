# K = 10 MNIST All-Digits Experiment Handbook

Experiment handbook for the K = 10 fixed-budget campaign: adaptive λ-bundle with CCP decisions versus uniform simplex grids, all ten MNIST digits, B = 500,000, seed 41. Version v1, September 10, 2026.
Chinese version: `Zh/K10_EXPERIMENT_HANDBOOK_ZH.md`; the two files must be kept in sync.

This handbook is written for the person who runs the experiment on a GPU machine. Part 1 explains what the experiment is; Part 2 explains how to run it, where the output goes and what to send back. All code is written and has been checked end to end at small scale on a CPU. The runner does not need to change any code, only to execute the commands of Part 2 in order.

---

## Part 1: What the experiment is

### 1.1 Summary

The problem is ten-objective training on all ten MNIST digits. Objective k, F_k(θ), is the mean cross-entropy on the training images of digit k, plus a ridge term (μ/2)‖θ‖² with μ = 1e-4. The network is a small three-layer net (patch(5×5)-64 → dense-96 → 10 logits, 8,874 parameters) with softplus activations, so every objective is smooth. To train the ten objectives together, a weight vector λ = (λ₁, …, λ₁₀) combines them into one scalar objective F_λ = Σ_k λ_k F_k; λ lives on the nine-dimensional simplex (non-negative entries summing to one). Every λ is one training problem, and "multi-objective training" means solving the problems of the whole simplex well.

The experiment compares two ways of choosing where on the simplex to train. Both use the same network, the same data, the same inner solver, the same computational budget and the same random seed; the only difference is which λ is trained in each round:

* **uniform grid** (baseline): the simplex is tiled with the C(r+9, 9) grid points of resolution r, and the points are visited in a snake order, over and over, until the budget is spent.
* **adaptive λ-bundle with CCP decisions** (CCP below): in every round, the weight λ that is "least solved" by the models delivered so far is found by solving a max-min problem with the convex-concave procedure (CCP), and training happens at that λ.

There are two evaluation metrics:

* **Primary: worst GN** (worst-case gradient norm). For every λ on the simplex, take the delivered model with the smallest gradient ‖Σ_k λ_k ∇F_k(θ)‖; then take the maximum over all λ. It measures how well the hardest weight has been solved; smaller is better.
* **Secondary: hypervolume of the training front** (HV). All models delivered by a run are placed in the ten-dimensional loss space, the non-dominated set is taken, and the volume it dominates below the reference point (ln 10, …, ln 10) is estimated by Monte Carlo. Larger means better coverage of the front.

The budget is B = 500,000 "gradient-evaluation units" (see the budget accounting in 1.2). The uniform grid uses r ∈ {3, 4, 5, 6, 7}, i.e. 220, 715, 2,002, 5,005 and 11,440 grid points; one run per r plus one CCP run, six runs in total, all with seed 41. There is no test-set evaluation, but every delivered parameter vector θ is saved for a later test-set experiment.

### 1.2 Overall idea and steps

The experiment has four stages, the four `--stage` values of one script:

1. **train**: the six runs, one after another. Within its budget, a run keeps delivering models (one θ at the end of every training "segment") and records a checkpoint every 12,500 budget units (budget spent, wall-clock time, number of delivered models). Training runs on the GPU; the CCP decisions run on the CPU. Output: summary.json, grams.npz (the 10×10 gradient Gram matrix and the ten loss values of every delivered point), thetas.npz (all θ).
2. **audit**: afterwards, the worst GN of the model set at every checkpoint is computed. There is no exact method at K = 10, so a CCP-based instrument searches for it (it returns a lower bound); at the end point a heavier instrument searches once more and an exact evaluation on random λ's is added as a check; the maximum is the value that goes into the table. This stage uses the CPU only and is run after all training, so that its CPU load does not disturb the wall clock of another run.
3. **tables**: Table 1 (final worst GN and ratios) and Table 2 (hypervolume).
4. **figures**: Figure 1 (all curves) and Figure 2 (one dot per baseline resolution).

The idea is the same as in the earlier K = 2 and K = 3 fixed-budget experiments: compare under a fixed budget, summarise every baseline resolution as one dot, and check whether the CCP curve lies below all dots. The points specific to K = 10:

* **A structural fact about grids.** Grid coordinates are multiples of 1/r; all ten coordinates can be non-zero only when r ≥ 10, so for r < 10 no grid point trains all ten classes at once: a grid point of r = 3 trains at most 3 classes, of r = 7 at most 7. The ℓ₁ distance from the uniform weight (the centre of the simplex) to the nearest grid point falls from 1.4 to 0.6 as r grows, while the diameter of the simplex is 2. The first resolution that covers the centre is r = 10 with 92,378 grid points; one pass costs about 9.5 million units, 19 times the budget. This is what K = 10 shows: no affordable resolution covers the interior of the simplex, while CCP finds the worst weight by itself.
* **Why B = 500,000.** A full-support segment costs 30 units at K = 10 versus 9 at K = 3, so the same budget buys three times fewer segments; at 500,000 the CCP run gets about 22,000 segments. Passes completed by the grids: r = 3 about 30, r = 4 about 8.7, r = 5 about 2.9, r = 6 about 1.1, r = 7 visits only 47% of its points. r = 7 is the "finer does not help" exhibit.
* **Budget accounting and the support effect.** One segment = one anchor full gradient (all ten objectives once, 10 units) + 53 minibatch correction steps (1,024 rows per step; the minibatch contains only the classes with non-zero weight in λ, and the cost counts the rows actually used). A segment therefore costs 10 + 2.0 × (number of weighted classes): grid segments contain 3 to 7 classes, 16 to 19 units each; CCP mostly trains near full support, close to 30 units. All runs are measured with the same ruler; grids consequently buy more segments. The table reports grid points and passes, not segment counts.
* **Definition of the dots (Figure 2).** The best-so-far curve of a baseline is a staircase; the dot is placed at the start of its last step, i.e. the checkpoint of the curve's last decrease; its height is the final value.

### 1.3 Parameters

| Item | Value |
|---|---|
| Data | MNIST training set, 5,421 images of each digit (limited by digit 5, the smallest class; balanced), n = 54,210; no test set |
| Model | patch(5×5)-64 → dense-96 → 10 logits, softplus activations, d = 8,874 parameters |
| Objectives | mean cross-entropy per digit + (μ/2)‖θ‖², μ = 1e-4 on all parameters including biases; λ ∈ Δ₉ |
| Inner solver | SVRG segment: 1 anchor full gradient + m = 53 minibatch correction steps (one pass over the data), batch 1,024 (102 to 103 rows per class); smoothness constants L from 40 random probe pairs, plus μ |
| Step rule | Adam for every method, α = 1e-3, β₁ = 0.9, β₂ = 0.9; moments reset when λ changes; on a non-descending segment the moments are reset and α is halved |
| Uniform grid | r ∈ {3, 4, 5, 6, 7}, 220 / 715 / 2,002 / 5,005 / 11,440 points, snake order, s = 5 segments per visit, warm start from the previous point's solution |
| CCP | per decision N₀ = 2,000 sampled λ's, the best 10 polished by CCP ascent, pool cap 30, stopping threshold τ = 1e-8·max(1, φ); the chosen weight is trained for s = 5 segments |
| Budget | B = 500,000 gradient-evaluation units per run; a checkpoint every 12,500 units, 40 in total |
| Worst-GN meter | every checkpoint: CCP instrument N₀ = 8,192, r = 20, fresh start. End point: CCP N₀ = 32,768, r = 20 with two sampler seeds, plus the exact envelope on 100,000 uniform random λ's as a check; the maximum is reported. Tables report the norm scale (square root) |
| Hypervolume | training loss space (ridge included); reference point (ln 10, …, ln 10); non-dominated set of all points delivered by a run; Monte Carlo with one million samples |
| Time axis | wall-clock time of each run, training on the GPU and decisions on the CPU, audits excluded; six runs in series on an idle machine |
| Seeds | sampler 41, initialisation 8, probes 7, identical for all runs; same initial point |
| Hardware | training: NVIDIA GPU, float64; decisions and audits: CPU |

### 1.4 Expected results

* The final worst GN of CCP is clearly below every uniform grid, and its curve lies below all dots. The ratio is expected to exceed the 6x seen at K = 3, because of the structural fact of 1.2: grids with r ≤ 7 never train a model that sees all ten classes.
* The ratio need not be monotone in r: coarse grids sit at the floor set by the gap between grid points; fine grids do not complete a pass within the budget.
* In hypervolume, uniform is far below CCP; see 1.6.
* On the time axis the CCP curve is about 2 to 2.5 times longer than the uniform curves (the cost of the λ search), but should still lie below all dots.

### 1.5 The smoke test

`--smoke` is a one-minute miniature that only checks that the whole pipeline runs; it produces no scientific result: 300 images per class, budget 800, 2 segments per visit, uniform only at r = 2, reduced audit and hypervolume sample counts, everything on the CPU. It runs train → audit → tables → figures and prints `SMOKE OK` at the end. Its output goes to a separate SMOKE directory and never touches the real experiment's directory. The runner executes it once after setting up the environment on the GPU machine (Step 0 of Part 2) and continues only if it passes.

### 1.6 Things to state in advance

1. **The structural result in Table 2.** The hypervolume reference point is (ln 10, …, ln 10), the loss of random guessing. A uniform grid point trains at most r classes, and the cross-entropy of an untrained class exceeds ln 10 = 2.30, so such a delivered point lies outside the reference box and contributes nothing to the hypervolume. In the August K = 10 trial, 0% of the delivered points that trained 2 classes or fewer lay inside the box, 25% of those with 3 classes, 57% with 5, 99% with 8 or more. The hypervolume of uniform will therefore be far below CCP's; the table records the fact that "grids never deliver a model that handles all ten classes", not a subtle difference in the shape of the front. Table 2 therefore carries two diagnostic columns: the fraction of delivered points inside the reference box, and the largest class loss of the most balanced model, min_θ max_k F_k(θ).
2. **Meaning of the time axis.** The time in the figures is the wall clock of one machine: training on the GPU, the λ-search decisions on the CPU, audits excluded. The CCP curve is expected to be about 2 to 2.5 times longer than the uniform curves because of the decision cost; the caption must state the hardware and this point.
3. **Worst GN is a lower bound.** There is no exact meter at K = 10; an instrument returns the value at the worst λ it found, and the true value can only be larger. All runs use the same instruments, so the comparison is fair; the heavier end-point instrument plus the random check exist to make the number that enters the table as tight as possible.
4. **Implementation of the CCP decision step.** Every CCP iteration solves a linear programme (LP): max t subject to 2(M_i λ_c)ᵀλ − λ_cᵀM_iλ_c ≥ t for every delivered point i, λ on the simplex. The original implementation hands the LP coefficients to HiGHS one entry at a time, i.e. m×10 Python-to-C++ calls for m delivered points; at m ≈ 20,000 one decision takes 40 s and the decisions of one CCP run 20 hours. This experiment uses the new file `ccp_lambda_solver_bulk.py`, which passes the whole LP in one block and restores the previous basis for a warm start. The LP itself, the CCP iteration rule, τ, the number of restarts and the pool are unchanged. The gate (`sanity_checks_ccp_bulk.py`) verifies on 1,371 recorded LPs that both transports return the same optimal value (relative difference 8.7e-13) and the same λ. What must be stated: the full decision sequence is not bit-identical. The multistart CCP contains discrete choices (seed screening order, pool de-duplication, the stopping test); a last-bit difference in an LP solution flipped one of them at decision 170, after which the two paths visit different but equally valid local maximisers — the same effect as running the original solver on another CPU or with another HiGHS build. On the 600 decisions of the K = 3 leg the two sequences agree bit for bit. On the 475 K = 10 decisions the relative difference of φ between the two sequences has median 0, mean +0.8% and maximum 20%; the new transport's value is at least as high on 93% of the decisions; the final φ's are 7.90e-2 and 7.78e-2, so the new transport does not search systematically worse. The gate report is at `output/CCP/ccp_compare_without_256_checkpoints/K10_mnist10k_B55000/adaptive_s5_ccp/gate_ccp_bulk.json`.
5. **Single seed.** All runs use seed 41, as in K = 2 and K = 3. The conclusions are single-seed comparisons.

---

## Part 2: How to run the experiment

### 2.1 Environment

A Linux machine with an NVIDIA GPU (any CUDA card; float64 is kept: the network is tiny, so the time per segment is set by the number of kernel launches, not by float64 throughput; 1 GB of GPU memory is enough). The Mac GPU (MPS) has no float64 and cannot be used.

```bash
git clone https://github.com/MarshmallowJeffrey/First-order-method-smooth-MOO.git
cd First-order-method-smooth-MOO
git checkout mlp-comparison-results   # the experiment code lives on this branch, not on main
python3 -m venv .venv            # Python 3.10 or newer
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib highspy
# CUDA build of torch: pick the command matching the machine's CUDA version at https://pytorch.org/get-started/locally/, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch, highspy; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU model. cyipopt is not needed (K = 10 does not use IPOPT), nor is torchvision. The MNIST data (10 MB) is downloaded automatically on first use into `Adaptive Bundle Algorithm/data/mnist/`; on a machine without internet access, copy that directory from elsewhere.

All commands are executed from this directory:

```bash
cd "First-order-method-smooth-MOO/Adaptive Bundle Algorithm/Original_py/experiment_plot"
```

### 2.2 Code files

New files written for this experiment (all under `Adaptive Bundle Algorithm/Original_py/`); the runner does not modify them:

| File | Role |
|---|---|
| `objective/objectives_mnist_patch_k10.py` | the K = 10 objectives: 5,421 rows per class, ridge term, per-class-forward joint gradient, device selection (cpu / cuda) |
| `Core Engine/ccp_lambda_solver_bulk.py` | the CCP decision solver with block LP transport and basis warm start (see 1.6, item 4) |
| `experiment_plot/run_dots_K10_without_256_checkpoints.py` | the main script: the four stages train / audit / tables / figures, and `--smoke` |
| `experiment_plot/calibrate_gpu_K10_without_256_checkpoints.py` | GPU calibration: numerics against the CPU, determinism, time per segment, decision time; writes `calibration_K10.json` |
| `sanity_check/sanity_checks_k10_objective.py` | objective gates (passed on the development machine) |
| `sanity_check/sanity_checks_ccp_bulk.py` | solver equivalence gate (passed on the development machine; see 1.6, item 4) |

Existing files that are reused (no need to look inside): `plot_curves_family_without_256_checkpoints.py` and `plot_dots_figure_without_256_checkpoints.py` (Figures 1 and 2), `baseline/baseline_without_256_checkpoints.py` (simplex grid and snake order), `Core Engine/stepper_core.py` (Adam), `Core Engine/ccp_lambda_solver.py` (the CCP base class), `objective/objectives_mnist_patch.py` (network and data loader).

### 2.3 Steps

Every command below is executed in `experiment_plot/` with the venv's `python`. Long steps run under `nohup … &` or `tmux` so that an SSH disconnect does not kill them; logs are written with `tee`.

**Step 0: smoke test (1 minute, CPU)**

```bash
python run_dots_K10_without_256_checkpoints.py --smoke 2>&1 | tee smoke_K10.log
```

The last line must be `SMOKE OK`. Output goes to `output/CCP/K10_mnist_without_256_checkpoints/SMOKE/`; it may be deleted or kept.

**Step 1: GPU calibration (2 to 3 minutes)**

```bash
python calibrate_gpu_K10_without_256_checkpoints.py --device cuda 2>&1 | tee calibrate_K10.log
```

The last line must be `[calib] PASS -> …/calibration_K10.json`. It checks four things: C1, the ten losses and the gradients at one θ agree between the GPU and the CPU to a relative 1e-10; C2, one segment run twice on the GPU is bit-identical; C3, the time per segment and the projected total training time at B = 500,000 (see `projected training for all legs` in the log, in minutes); C4, one CCP decision uses the `highspy-bulk` backend. **If C2 fails**, run again with `--deterministic` and note in RUN_NOTES.md that this flag was needed; if it still fails, stop and send `calibrate_K10.log` back. The training script refuses to start on CUDA without a passed `calibration_K10.json`.

**Step 2: training (about 4 to 5 hours, GPU; the machine must be idle)**

```bash
nvidia-smi        # no other process on the GPU
top -bn1 | head   # CPU idle
nohup python run_dots_K10_without_256_checkpoints.py --stage train --device cuda \
      > train_K10.log 2>&1 &
tail -f train_K10.log
```

Order: the CCP run, then uniform r = 3, 4, 5, 6, 7. Every run prints one progress line per 10% of budget (segments, wall clock, decision time); at the end it prints the budget spent, segment count, wall clock, decision share and support histogram, and writes summary.json, grams.npz and thetas.npz. If the process dies, rerun the same command: runs that are complete (summary.json present) are skipped, incomplete ones restart from scratch. **Nothing else may run on the machine during training**, because the wall clock enters the time axis of Figure 2.

**Step 3: audit (about 1 hour, CPU)**

```bash
nohup python run_dots_K10_without_256_checkpoints.py --stage audit > audit_K10.log 2>&1 &
```

Computes the worst GN at the 40 checkpoints and the end point of every run, writes audit.json and merges the results into summary.json; the log has one line per run with the final value and which of the three end-point instruments won. Audited runs are skipped. This step does not have to run on the GPU machine: copy every run's summary.json and grams.npz to any machine with the Python environment and run the same command there.

**Step 4: tables (about 30 minutes, CPU)**

```bash
python run_dots_K10_without_256_checkpoints.py --stage tables 2>&1 | tee tables_K10.log
```

Writes `table1_K10.md/json` and `table2_K10.md/json`. Table 2 uses one million Monte Carlo samples, a few minutes per run.

**Step 5: figures (1 minute)**

```bash
python run_dots_K10_without_256_checkpoints.py --stage figures 2>&1 | tee figures_K10.log
```

Writes `worst_gn_curves_uniform_all_laststep.png` (Figure 1) and `worst_gn_dots_paper.png` with its .json/.md (Figure 2).

**Step 6: package and deliver** (see 2.9)

Steps 2 to 5 can also run as one command with `--stage all`. Running them separately allows the small files to be delivered as soon as training ends.

### 2.4 Output directory and files

All output lives inside the repository in the directory below (the "home"), next to the K = 2 and K = 3 outputs in `output/CCP/`:

```
Adaptive Bundle Algorithm/output/CCP/K10_mnist_without_256_checkpoints/
├── calibration_K10.json                 result of Step 1 (machine, timings, PASS)
├── SMOKE/                               output of Step 0 (may be deleted)
└── mu0.0001/dots_B500000/adam_1e-3_b0.9/          <- home
    ├── campaign_manifest.json           per run: wall clock, decision time, segments, machine info, git commit
    ├── RUN_NOTES.md                     written by the runner (see 2.5)
    ├── adaptive_ccp_seed41/
    │   ├── summary.json                 parameters, budget spent, checkpoints, wall clock, decision time, CCP telemetry, audit results
    │   ├── grams.npz                    Gram matrices (m,10,10), loss values (m,10), λ history of every delivered point
    │   ├── thetas.npz                   parameters θ of every delivered point (m, 8874), float64, 1.5 to 2.4 GB per run
    │   └── audit.json                   details of Step 3
    ├── uniform_r3_seed41/  … uniform_r7_seed41/      the same, one directory per r
    ├── table1_K10.md / table1_K10.json  Table 1
    ├── table2_K10.md / table2_K10.json  Table 2
    ├── worst_gn_curves_uniform_all_laststep.png      Figure 1
    └── worst_gn_dots_paper.png / .json / .md         Figure 2 and its numbers
```

The directory names `mu0.0001`, `dots_B500000` and `adam_1e-3_b0.9` record the ridge coefficient, the budget and the step rule, the same naming as the K = 3 directory `K3_mnist_triple_without_256_checkpoints/v2_stepper_mu0.0001/dots_B100000/adam_1e-3_b0.9/`. `thetas.npz` is excluded by the repository's .gitignore (above GitHub's single-file limit); everything else can be committed.

### 2.5 Logs, JSON files and notes to keep

Produced by the scripts: every run's `summary.json` (with a `machine` field: platform, torch version, GPU model, CPU count, thread count, git commit, start and end times), `audit.json`, `campaign_manifest.json`, `calibration_K10.json`, and the console logs of every stage (`smoke_K10.log`, `calibrate_K10.log`, `train_K10.log`, `audit_K10.log`, `tables_K10.log`, `figures_K10.log`, written by `tee`/`nohup`; please send them back too).

Written by the runner: create `RUN_NOTES.md` in the home directory and fill in this template:

```markdown
# K10 run notes
- Machine: (hostname, GPU model, CUDA version, CPU model, cores, RAM)
- Python / torch / highspy versions: (from `pip list`)
- Repo commit: (git rev-parse --short HEAD)
- Step 0 smoke: date, SMOKE OK?
- Step 1 calibration: date, PASS? deterministic flag used? projected minutes printed:
- Step 2 train: start/end time of each run; anything else running on the machine? any restart?
- Step 3 audit: date, machine used (same GPU box or another), wall time
- Step 4/5 tables and figures: date
- Anomalies: (crashes, warnings in the logs, re-runs, machine sleep, other users)
```

### 2.6 CPU and GPU: division of work and cautions

The forward and backward passes of the training segments run on the GPU; θ, the gradients, the Adam update, the budget meter, the CCP decisions (the HiGHS linear programmes), the audits (CCP instruments and the random check), the tables and the figures run on the CPU. Every training step copies θ to the GPU and the gradient back to the CPU (71 KB each), about 0.3 ms of overhead. The GPU idles during decisions and the CPU idles during training; this is the sequential dependency of the algorithm and cannot be overlapped.

Cautions:

1. **Numerics and reproducibility.** Float64 matrix products on a GPU differ from the CPU in the last bits (summation order, fused multiply-add), so a GPU training trajectory cannot be bit-identical to a CPU one, and need not be. The calibration checks that the losses and gradients at one θ agree between GPU and CPU to a relative 1e-10, and that one segment run twice on the GPU is bit-identical. If the latter fails, rerun the calibration with `--deterministic` (it switches on torch's deterministic algorithms) and record it. The backward pass of this network has no atomic additions, so it is expected to be deterministic.
2. **Timing.** CUDA is asynchronous; the script calls `torch.cuda.synchronize()` at every checkpoint and at the end of a run before reading the clock, and the per-step copy of the gradient to the CPU synchronises anyway. The runner only has to keep the machine idle.
3. **Meaning of the time axis.** The time in Figure 2 is the wall clock of one machine: training on the GPU, decisions on the CPU; the caption must state the machine and this point.
4. **The machine must be idle.** The six runs are serial; no other task may use the GPU or the CPU during training (check `nvidia-smi` and `top`). Do not run the audit while training. On a shared machine, choose a slot when nobody else is using it. On a laptop, plug in the power and disable sleep.
5. **Audit after training**, or on another machine (it only needs every run's summary.json and grams.npz, about 200 MB in total).
6. **Threads.** torch's default thread count is used; to fix it, add `--threads 8`. HiGHS is single-threaded.
7. **GPU memory and RAM.** The data (340 MB) plus the forward activations of 5,421 rows per class (about 70 MB) fit on any card; the largest object in RAM is the θ stack (1.5 to 2.4 GB per run, compressed to disk at the end of the run); 16 GB of RAM is enough.
8. **Disk.** The six thetas.npz files take about 10 to 12 GB; everything else is under 300 MB. If disk is short, add `--thetas-float32` (θ stored as float32, half the size; this only affects the precision of a later test-set experiment, not any number of this experiment).
9. **Interruptions and reruns.** The training script resumes per run: runs with a summary.json are skipped, the others restart from scratch. Killing a run mid-way loses that run's progress (about 40 minutes), nothing else.

### 2.7 Standards for the results

**Figure 1** (`worst_gn_curves_uniform_all_laststep.png`): 1 × 2 panels. Left, the abscissa is gradient evaluations (linear); right, wall-clock seconds; both ordinates are the best-so-far worst GN (norm scale) on a log scale. One thick CCP curve and five thin uniform curves for r = 3 to 7, with a dot on every baseline curve at its last decrease. All curves start from the same point (only the initial point θ₀ at x = 0).

**Figure 2** (`worst_gn_dots_paper.png`): 1 × 2 panels with the same two abscissae (log scale by default). CCP as a curve; uniform as one square per r, placed at the checkpoint of the curve's last decrease with the final value as its height, joined by a dashed line in abscissa order and labelled with r. The .md file of the same name gives the numbers of every dot (final value, abscissa of the dot, the CCP value at the same budget and the ratio).

**Table 1** (`table1_K10.md`): columns method, r, grid points, passes, final worst GN (norm scale), method / CCP. Passes = the budget actually spent by the run divided by the budget of one pass of that grid (support-aware cost per segment). No segment counts.

**Table 2** (`table2_K10.md`): columns run, HV (train, Monte Carlo, with standard error), gap versus CCP ((HV_CCP − HV_run)/HV_CCP), fraction of delivered points inside the reference box, number of non-dominated points inside the box, largest class loss of the most balanced model. The paper uses the first three columns; the other three are diagnostics that explain where the gap comes from.

### 2.8 Troubleshooting

* `SMOKE OK` does not appear: read the error at the end of `smoke_K10.log`. Usual causes: missing packages (`ModuleNotFoundError`: install per 2.1), MNIST download failed (no internet: copy the `Adaptive Bundle Algorithm/data/mnist/` directory over).
* Calibration C1 fails (GPU versus CPU difference above 1e-10): send `calibrate_K10.log` back; do not continue.
* Calibration C4 reports `highspy missing`: `pip install highspy`; without it the decisions fall back to scipy's cold-start solver, tens of times slower.
* The training script reports `GPU training requires a passed calibration`: run Step 1 first.
* The machine reboots or the process is killed during training: rerun the Step 2 command; finished runs are skipped.
* `nvidia-smi` shows someone else on the GPU: wait for the machine to be free; if a finished run was disturbed (clearly inflated wall clock), delete that run's directory and rerun.
* The earlier experiments are only needed when debugging the code itself: the K = 3 counterpart of the main script is `run_dots_K3_without_256_checkpoints.py` (its executor is almost identical, but the problem is three digits and the audit uses IPOPT), the K = 2 one is `run_dots_K2_without_256_checkpoints.py`; their outputs are under `output/CCP/K3_mnist_triple_without_256_checkpoints/` and `output/CCP/K2_mnist_pair_without_256_checkpoints/`, and their summary.json fields can be compared. Apart from that, nothing about K = 2 or K = 3 needs to be known.

### 2.9 Delivery checklist

After training (a first delivery is possible as soon as Step 2 is complete; a second one after audit, tables and figures):

1. everything in the home directory except `thetas.npz` (the six runs' summary.json, grams.npz, audit.json, campaign_manifest.json, tables and figures), about 300 MB;
2. `calibration_K10.json` and all `*.log` files;
3. `RUN_NOTES.md`;
4. the six `thetas.npz` (about 10 to 12 GB), which can follow later via cloud storage or a drive, or stay on the machine for now.

Packaging example (from the repository root):

```bash
cd "Adaptive Bundle Algorithm/output/CCP/K10_mnist_without_256_checkpoints"
tar --exclude='thetas.npz' -czf K10_results_$(date +%Y%m%d).tgz mu0.0001 calibration_K10.json
```

---

## Appendix: command summary

```bash
cd First-order-method-smooth-MOO && git checkout mlp-comparison-results
cd "Adaptive Bundle Algorithm/Original_py/experiment_plot"
source ../../../.venv/bin/activate
python run_dots_K10_without_256_checkpoints.py --smoke 2>&1 | tee smoke_K10.log                 # Step 0, 1 min
python calibrate_gpu_K10_without_256_checkpoints.py --device cuda 2>&1 | tee calibrate_K10.log  # Step 1, 3 min
nohup python run_dots_K10_without_256_checkpoints.py --stage train --device cuda > train_K10.log 2>&1 &   # Step 2, ~4-5 h
nohup python run_dots_K10_without_256_checkpoints.py --stage audit > audit_K10.log 2>&1 &                 # Step 3, ~1 h
python run_dots_K10_without_256_checkpoints.py --stage tables 2>&1 | tee tables_K10.log         # Step 4, ~30 min
python run_dots_K10_without_256_checkpoints.py --stage figures 2>&1 | tee figures_K10.log       # Step 5, 1 min
```

The projected minutes printed by the calibration script are the authoritative time estimate; the hours above assume 60 to 90 ms per segment.
