# July 11, 2026 — two tasks: (1) adaptive-pair Pareto figure; (2) K=6 96x96 trial runner

# Task 1 — Adaptive-pair Pareto figure (eps 0.01 vs eps 0.001, no baseline)

Scope: ONE code file modified,
`Original_py/run_pareto_certified_without_256_checkpoints.py`.
Also touched (docs/artifacts only, listed in §5): the two overview
READMEs in `output/Pareto_front/` and the new figure itself. No other
script, no algorithm file, no experiment parameter was changed, and
nothing inside the two existing run folders was modified or
regenerated. Companion record: `Note/Jul_8_note.md` (the certified
Pareto-front runner this extends).

## 1. Purpose

User request on the K=2 certified Pareto-front comparison: keep the
experiment exactly as it is, and add ONE figure that puts the adaptive
eps=0.01 and eps=0.001 fronts on the same axes, with NO baseline —
i.e. the two error tolerances of Algorithm 2 compared directly. The
figure must live under `output/Pareto_front/`.

## 2. How the figure was produced — replot, NOT re-run

The figure is drawn from the July 8 main run's stored
`pareto_data.json` (`fronts` = the t-segments per delivered point with
their F1/F2; `labels` = the certified-outcome strings), not from a new
run. Rationale:

- "All parameters unchanged" is satisfied in the strongest sense: the
  curves come from the SAME runs already on disk, not from a
  reproduction of them.
- The adaptive runs are deterministic — the overview README records
  them as bit-identical across the r=10 and r=20 folders — so a re-run
  would spend hours of oracle time to reproduce known numbers.

Exactness of the reconstruction: `front_segments` compressed the
uniform 1,001-point t-sweep into maximal runs of constant delivered
point, so a segment [t_from, t_to] covers exactly
`round((t_to - t_from) * 1000) + 1` consecutive grid values; replaying
the segments in order rebuilds `sel` exactly, hence both the polyline
and each distinct point's mean-served-t (the marker colour) are
reproduced bit-for-bit (floats survive the JSON round-trip exactly).
This is not assumed but AUDITED in-process: the reconstructed arrays
must regenerate the stored segments through `front_segments` again; any
mismatch raises before a figure is drawn.

## 3. What changed in the script (function by function)

- `_draw_fronts` (new) — the drawing body of `plot_combo` extracted
  verbatim (delivered-point polyline; one marker per distinct delivered
  point, viridis-coloured by mean served t; colourbar; log-log axes;
  grid; legend). Single source of truth so all figures read the same
  way.
- `plot_combo` — now calls `_draw_fronts`; its two combo figures are
  visually unchanged (they were NOT regenerated).
- `plot_adaptive_pair` (new) — the requested figure: eps=0.01 (purple
  `^`, as in the combo figures) and eps=0.001 (blue `o`) via the same
  grammar; certified outcomes go into the title. The denser eps=0.001
  front is drawn FIRST so the 14 eps=0.01 markers stay visible on top
  where the two fronts coincide; the legend still lists eps=0.01 first.
  Saves to every path it is given.
- `adaptive_pair_title` (new) — builds the title from the config dict,
  which works both for a live run's cfg and for the `config` stored in
  `pareto_data.json`.
- `packs_from_saved_fronts` (new) — the exact reconstruction + round
  trip audit of §2.
- `main()` — new flag `--replot-adaptive-pair RUN_DIR`: loads
  `RUN_DIR/pareto_data.json`, rebuilds the packs, draws ONLY the pair
  figure into `output/Pareto_front/` (keeping the source folder's
  `_smoke`/`_r<R>` suffix in the filename), runs nothing, touches
  nothing in RUN_DIR. The full-run path now also draws the pair figure
  after the two combos (into the run folder AND `output/Pareto_front/`)
  so future runs stay self-contained; the run-README template lists the
  third figure. Module docstring and two constants (`PARETO_FRONT_DIR`,
  `ADAPTIVE_PAIR_STEM`) updated accordingly.

## 4. Verification

- Replot on the main run folder
  (`--replot-adaptive-pair ../output/Pareto_front/pareto_certified_without_256_checkpoints`):
  round-trip audit passed; figure written. Visual check against the
  July 8 findings: the fronts coincide over the shared trade-off range;
  eps=0.001 is denser (54 vs 14 delivered points) and extends deeper at
  both ends (min F1 0.033 -> 0.0088, min F2 0.032 -> 0.0093) at 5.3x
  the certification cost (132 -> 700 grads) — exactly the recorded
  "tighter eps buys a better menu" conclusion, now visible on one
  figure.
- Full-run path: `--smoke` end-to-end (58 s) produced all three figures
  in the smoke run folder plus the `_smoke` pair copy in
  `output/Pareto_front/`; the smoke artifacts were deleted afterwards
  (repo convention, as in the Jul 9 note).
- `py_compile` clean; existing run folders and their combo figures
  untouched throughout.

## 5. Files touched (task 1)

| file | kind of change |
|---|---|
| `Original_py/run_pareto_certified_without_256_checkpoints.py` | the ONLY code change of task 1 (§3) |
| `output/Pareto_front/pareto_front_adaptive_eps0.01_vs_eps0.001.png` | NEW artifact — the requested figure |
| `output/Pareto_front/README.md`, `README_ZH.md` | docs: one bullet each recording the new figure and its provenance |
| `Note/Jul_11_note.md` | this note |

# Task 2 — K=6 96x96 budget-curve trial (runner READY; long run NOT yet launched)

## 1. Purpose

User request: a one-off parameter-combination trial, not part of any
existing experiment family — the crossover 96x96 configuration
(`output/crossover/d11522_h96x96_tanh_n50000_B2000/summary.json` as the
parameter reference) with K moved from 2 to 6, run in budget mode on the
without-256-checkpoints track, producing the two standard figures
(self-reported worst-case GN vs total gradient evaluations, and vs CPU
time on a log axis).  Hard user constraints, in final form: K=6,
hidden_sizes=[96,96], **r >= 10**; max_grad_evals may be RAISED;
everything else adjustable with reasons.  Output goes to a NEW folder
under `output/`, figure filenames must carry the without-256 marker.

## 2. What was created — a NEW script, no existing file modified

`Original_py/run_trial_K6_without_256_checkpoints.py` (new).  A new
runner was chosen over editing `run_experiments_without_256_checkpoints.py`
because that script is a purpose-built A/B harness: it always runs its
two fixed configurations AND draws A/B figures against the existing
with-256 summaries — a K=6 trial does not belong in it.  The new script
reuses the same machinery (`uniform_discretisation` /
`algorithm_adaptive` from the *_without_256_checkpoints modules,
`detect_plateau`, `_plot_plateau_pair`, `symmetric_time_to_target`,
`_result_curves`) and mirrors its `run_config` structure, with: output
folder `output/trial_K6_d<d>_h96x96_tanh_n<n>_B<B>_without_256_checkpoints/`,
figures `gn_vs_grad_evals_without_256_checkpoints.png` and
`gn_vs_cpu_time_without_256_checkpoints.png`, `summary.json` (same
schema, plus a `trial_note` provenance field), and an auto-generated
README.  Smoke mode (`--smoke`) verified the script end-to-end (34 s).

## 3. Parameter history — first proposal, user correction, calibration

- First proposal kept the reference budget B=2000 and cut r 10 -> 2,
  reasoning that at K=6, r=10 gives C(15,5)=3003 grid nodes and one
  baseline pass = 3003*5*6 = 90,090 grads = 45x that budget.  The user
  REJECTED reducing r: r >= 10 is a hard floor; raising the budget is
  the permitted lever instead.
- The first-proposal run (B=2000, r=2, max_inner=5, 64-start search) had
  already been launched and was left to finish as a CALIBRATION run; its
  folder `output/trial_K6_d11910_h96x96_tanh_n50000_B2000_without_256_checkpoints/`
  is kept with a SUPERSEDED banner (calibration data only, not an
  experiment result).  Measured on it:
  - joint-oracle call ~0.13 s (43.3 s per 2,000 grads; K=6, n=50k,
    d=11,910);
  - lambda-search cost ~0.0137 s per start per bundle point per round —
    cleanly linear in bundle size m (64 starts: 2.2 s/round at m~7 up
    to 59 s/round at m~66);
  - adaptive self-reported GN 7.32 -> 0.285 in 66 rounds (still falling
    slowly; L_scale_final=8 — the step-size safeguard fired again at
    this width, as in every 96x96 run).

## 4. Final configuration (encoded in TRIAL_CONFIG) and reasons

Search-cost model calibrated above: total lambda-search time over R
rounds with S starts ~ S*(0.03*R + 0.0137*R^2/2) seconds (m grows +1
per round under prune_inner).

| parameter | reference (K=2) | trial (K=6) | reason |
|---|---|---|---|
| coarse_resolution r | 10 | 10 | user floor; 3003 nodes, one pass = 90,090 grads |
| max_grad_evals B | 2,000 | 180,180 | exactly TWO baseline passes: within pass 1 the baseline's self-reported max still contains never-visited nodes at x0, so its curve cannot move before the pass ends; two passes show polish + re-polish. ~65 min of baseline gradient work at 0.13 s/call |
| max_inner | 5 | 25 | plateau-family K>=3 precedent; a round = K*25 = 150 grads — fewer, heavier rounds keep the bundle (and the per-round search cost, linear in it) in check |
| lambda_max_starts | 8 | 64 | repo's K=6 precedent; the search domain is the 5-D simplex and on this track the search value IS the plotted metric — 8 starts would bias it optimistically. CPU-only cost |
| max_outer | 1e6 (inert) | **150 (binding fuse)** | full-budget rounds are infeasible: R=1,201 at 64 starts ~ 176 h by the cost model; R=150 (= 22,500 grads = 12.5% of B) costs ~2.8 h. DISCLOSED ASYMMETRY: baseline gets the full budget, the adaptive method stops at its round fuse — conservative against the adaptive method unless it has already plateaued (figures + detector will show) |
| eval_every | 153 both (~13 pts) | baseline 4,500 (~40 pts), adaptive 600 (~37 pts) | cadence scaled to keep checkpoint COUNT in the reference regime, not the raw interval; self-reported checkpoints cost no extra solves and are excluded from CPU axes by protocol |
| everything else | — | unchanged | p=20, n=50,000, tanh, seeds 7/8, steps_per_point_per_pass=5, prune_inner=True, n_passes fuse 1e5, plateau detection (window 4, tol 0.05, consecutive 2) |

d moves 11,522 -> 11,910 with K (output head 96K+K) — a consequence,
not a choice.  Estimated wall time ~4-5 h serial (baseline ~1.2 h,
adaptive ~3 h of which ~2.8 h lambda search).

## 5. Status and next step

Script compiled and smoke-tested; TRIAL_CONFIG holds the final table
above.  The 4-5 h run was NOT launched: the user chose to think it over
first ("先别跑，我再想想").  To launch when decided (serial, idle
machine — concurrent load distorts the CPU axes):

    cd "Adaptive Bundle Algorithm/Original_py"
    KMP_DUPLICATE_LIB_OK=TRUE \
      ../../.venv/bin/python run_trial_K6_without_256_checkpoints.py

## 6. Files touched (task 2)

| file | kind of change |
|---|---|
| `Original_py/run_trial_K6_without_256_checkpoints.py` | NEW script (the only code artifact of task 2; no existing .py modified) |
| `output/trial_K6_d11910_h96x96_tanh_n50000_B2000_without_256_checkpoints/` | calibration-run output, kept with a SUPERSEDED banner in its README |
| `Note/Jul_11_note.md` | this section |

## 7. Execution history and results (July 12)

Status update to §5: the run was launched and completed on July 12.
Full history, kept honest:

- 02:20 — first launch (after the user's go-ahead), as a
  session-background task.  Killed silently by the OS at ~02:26 during
  baseline pass 1, node ~450/3003 (no traceback, no crash report;
  memory-pressure evidence: 16 GB machine, swap ~5.8/7 GB, ~15 stale
  Jupyter kernels from July 1–7 resident).  Deliberately not relaunched
  into the starved machine.
- 14:29 — with the user's explicit authorization, the 15 stale
  ipykernels and BaiduNetdisk were killed.  (A first kill attempt
  silently failed — the assistant harness sandbox blocks signals and
  `2>/dev/null` hid the errors; it succeeded with the sandbox disabled.)
- 14:32:39 — relaunch, nohup-detached, log at `run_log.txt` inside the
  output folder; a side sampler appended uptime/vm_stat to
  `machine_load_log.txt` every 300 s for post-hoc audit of the
  serial-idle assumption.
- 18:36 — DONE in 14,534 s (4 h 02 m): baseline 4,157.5 s, adaptive
  10,375.0 s.

Correction to §3 found while re-verifying: the calibration run's
adaptive final was 0.287 (summary.json: 0.28709), not 0.285.

### Run health

- Baseline oracle pace 0.1384 s/joint-call (30,030 joint calls in
  4,157.5 s) — within 6.5% of the 0.13 s calibration, so competing load
  did not materially distort the time axis.  Load audit (49 samples):
  1-min load 6–8 during the first ~75 min (the user was using the
  machine interactively), settling to ~3.5–5 during the adaptive phase;
  the run's own torch threads account for ~3 of that.
- RSS ~0.8 GB at start; no memory incident this time.  The
  descent-lemma safeguard fired as at every 96x96 width;
  L_scale_final = 16 (the shorter 66-round calibration ended at 8).
  inner_cap_hits = 0.  The one recorded runtime warning is the standard
  safeguard notice.

### Results (self-reported track — each method's own meter, NOT a shared yardstick)

| quantity | baseline (Algorithm 1, r=10) | adaptive (Algorithm 2) |
|---|---|---|
| final best-so-far GN | 0.6161 | **0.1473** (final-quality ratio 4.18) |
| budget used | 180,180 grads / 4,157.5 s | 22,500 grads = 12.5% of B (150-round fuse) / 10,375 s |
| plateau (window 4, tol 0.05, consec 2) | FOUND: onset 94,500 grads / 2,226 s, level 0.6161 | NOT found — still descending at the fuse |
| curve shape | pinned at 7.0916 through all of pass 1 (unvisited nodes sit at x0); first post-pass-1 checkpoint (94,500) reads 0.6246; pass 2 polishes only to 0.6161, flat for the last 76,680 grads | descent 7.32 → 0.147; end-phase search value oscillates 0.17–0.25 (the λ-search is a heuristic max), best-so-far monotone |

Cross-curve readings (best-so-far step interpolation, as in the
crossover analysis):

- **Equal-TIME quality ratio** at the baseline's total (4,157.5 s):
  0.6161 / 0.2143 = **2.88**.
- **Time-to-target** (symmetric; target = the worse final = baseline's
  0.6161): adaptive reaches it at 6,000 grads / 954.7 s, baseline at
  103,500 grads / 2,406.8 s → **17.25× fewer gradients, 2.52× less
  CPU**.
- Equal-GRADIENT ratio at the adaptive's total (22,500 grads):
  7.0916 / 0.1473 = 48.1 — structurally inflated (within pass 1 the
  baseline's meter cannot move below the x0 ceiling); prefer
  2.88 / 17.25 / 2.52 / 4.18 when quoting.

### Reading against the hypothesis (user message of July 12)

Hypothesis: on both figures the adaptive curve should lie LEFT of the
baseline's (better GN at equal gradients and at equal CPU time).
**CONFIRMED on this instance, on both axes**: the curves never cross;
at every abscissa the adaptive best-so-far is below the baseline's,
and every level the baseline ever reaches, the adaptive method reached
earlier (its final 0.6161: adaptive at 955 s vs baseline 2,407 s; on
gradients 6,000 vs 103,500).

Why this differs from the cheap-oracle K=6 case (plateau study at
n=30, where the baseline won the CPU axis by ~200×): two mechanisms
compound here.

1. **Oracle price.** At 0.138 s/joint-call, the λ-search overhead
   (~95% of the adaptive method's 10,375 s) is amortised against
   35-minute baseline passes instead of micro-second grid steps.
2. **Grid combinatorics.** |G_r| = C(r+K−1, K−1): at K=2, r=10 the
   grid has 11 nodes (330 grads/pass); at K=6 it has 3,003 nodes
   (90,090 grads/pass).  The baseline's per-pass cost grows like
   r^(K−1) at fixed r, while the adaptive method's per-step cost grows
   only linearly in K.

So at expensive gradients, raising K to 6 does not restore the
baseline's cheap-oracle advantage — it enlarges the adaptive lead: the
adaptive method's large-K disadvantage is a CPU-overhead phenomenon,
and expensive gradients price it away.

Caveats (standing): the two curves are per-method self-reported
meters — the baseline's max never looks between its grid nodes (its
true sup over the simplex can only be worse than reported), the
adaptive value is its own 64-start search; this bias direction is
generous TO THE BASELINE, so the conclusion is robust in the direction
that matters, but talk-grade cross-method claims should still use a
256-start GN* rerun.  The round fuse proved conservative against the
adaptive method (stopped at 12.5% of B, not plateaued).  Single
instance, seeds 7/8; single machine; elevated interactive load during
the baseline phase (documented in `machine_load_log.txt`, oracle pace
within 6.5% of calibration).

### Files added by the run (task 2, final)

| file | kind of change |
|---|---|
| `output/trial_K6_d11910_h96x96_tanh_n50000_B180180_without_256_checkpoints/` | NEW results folder: `summary.json`, `gn_vs_grad_evals_without_256_checkpoints.png`, `gn_vs_cpu_time_without_256_checkpoints.png`, auto-README (plus execution record), `run_log.txt`, `machine_load_log.txt` |
