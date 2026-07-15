# Presentation Figures

This folder collects the final figures recommended for slides.

## Main Figures

- `00_overview_contact_sheet.png`
  Quick visual overview of the selected figures.

- `01_K2_hybrid_aligned.png`
  K=2 original hybrid Chebyshev comparison. Use this for the first successful case.

- `02_K3_highk_tuned.png`
  K=3 high-k tuned comparison. Chebyshev beats the uniform baseline.

- `03_K4_highk_maxinner1_tuned.png`
  K=4 tuned comparison using Chebyshev `high-k` with `max_inner=1`. This is the final K=4 figure.

- `04_selected_final_best_gn_vs_K.png`
  Final selected stationarity coverage across K=2,3,4.

- `05_selected_final_cpu_vs_K.png`
  CPU cost across K=2,3,4. Useful for explaining the adaptive bundle overhead.

- `06_K4_chebyshev_tuning_ablation.png`
  K=4 tuning ablation showing why `max_inner=1` works best.

## Data Tables

- `results_table.csv`
  Final selected method results for K=2,3,4.

- `k4_tuning_table.csv`
  K=4 Chebyshev tuning variants.

## Final Selected Results

| K | Method | Final best GN* | CPU time |
|---|---|---:|---:|
| 2 | baseline | 2.7403e-01 | 77.28s |
| 2 | adaptive bundle | 2.3804e-04 | 3631.13s |
| 2 | tuned Chebyshev | 1.5236e-02 | 122.28s |
| 3 | baseline | 5.4020e-01 | 115.17s |
| 3 | adaptive bundle | 1.0122e-02 | 1902.14s |
| 3 | tuned Chebyshev | 2.0291e-01 | 95.25s |
| 4 | baseline | 1.6088e+00 | 60.96s |
| 4 | adaptive bundle | 5.7056e-02 | 3716.76s |
| 4 | tuned Chebyshev | 3.7347e-01 | 1296.60s |

## Suggested Slide Usage

Use `01`, `02`, and `03` as the main per-K result slides. Use `04` as the generalization summary. Use `06` only when explaining why K=4 needed a shallower Chebyshev inner loop.
