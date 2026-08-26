# lambda-solver bench (experiment 2)

T = 60s, batches = 20; rules: Note/Aug_9_note.md §6.

## K2_early_m41

| metric | ccp | ipopt |
|---|---|---|
| 2a_converged_frac | 1 | 0.75 |
| 2a_n | 200 | 200 |
| 2a_phi_best | 0.0143958 | 0.0143958 |
| 2a_phi_mean | 0.0114269 | 0.0105598 |
| 2a_phi_median | 0.0101308 | 0.0101308 |
| 2a_time_median_s | 0.000488 | 0.116803 |
| 2a_time_p95_s | 0.000655433 | 0.127928 |
| 2b_best_at_10s | 0.0143958 | 0.0143958 |
| 2b_best_at_T | 0.0143958 | 0.0143958 |
| 2b_distinct_maxima | 13 | 55 |
| 2b_restarts | 91375 | 519 |
| 2b_restarts_within_10s | 15244 | 87 |
| paired ccp wins / ties / total | 84 / 11 / 200 | |

## K2_late_m3195

| metric | ccp | ipopt |
|---|---|---|
| 2a_converged_frac | 0.96 | 0.33 |
| 2a_n | 200 | 200 |
| 2a_phi_best | 9.51855e-05 | 9.47439e-05 |
| 2a_phi_mean | 5.87357e-05 | 5.66169e-05 |
| 2a_phi_median | 5.04346e-05 | 5.04346e-05 |
| 2a_time_median_s | 0.038681 | 0.162958 |
| 2a_time_p95_s | 0.0401575 | 0.236484 |
| 2b_best_at_10s | 9.51755e-05 | 9.51855e-05 |
| 2b_best_at_T | 9.51854e-05 | 9.51855e-05 |
| 2b_distinct_maxima | 229 | 129 |
| 2b_restarts | 712 | 334 |
| 2b_restarts_within_10s | 133 | 52 |
| paired ccp wins / ties / total | 133 / 2 / 200 | |

## K2_mid_m1641

| metric | ccp | ipopt |
|---|---|---|
| 2a_converged_frac | 1 | 0.315 |
| 2a_n | 200 | 200 |
| 2a_phi_best | 0.000235601 | 0.000235601 |
| 2a_phi_mean | 0.000123226 | 0.000120424 |
| 2a_phi_median | 0.000115667 | 0.000107361 |
| 2a_time_median_s | 0.0149936 | 0.141461 |
| 2a_time_p95_s | 0.0156252 | 0.188298 |
| 2b_best_at_10s | 0.000235601 | 0.000235601 |
| 2b_best_at_T | 0.000235601 | 0.000235601 |
| 2b_distinct_maxima | 231 | 125 |
| 2b_restarts | 1750 | 407 |
| 2b_restarts_within_10s | 243 | 67 |
| paired ccp wins / ties / total | 143 / 2 / 200 | |

## K6_early_m108

| metric | ccp | ipopt |
|---|---|---|
| 2a_converged_frac | 1 | 0.02 |
| 2a_n | 200 | 200 |
| 2a_phi_best | 1.79952 | 1.79952 |
| 2a_phi_mean | 1.71285 | 1.57412 |
| 2a_phi_median | 1.79754 | 1.57731 |
| 2a_time_median_s | 0.00275427 | 0.0819016 |
| 2a_time_p95_s | 0.00367202 | 0.109671 |
| 2b_best_at_10s | 1.79952 | 1.79952 |
| 2b_best_at_T | 1.79952 | 1.79952 |
| 2b_distinct_maxima | 103 | 655 |
| 2b_restarts | 16949 | 698 |
| 2b_restarts_within_10s | 2828 | 116 |
| paired ccp wins / ties / total | 183 / 0 / 200 | |

## K6_late_m4309

| metric | ccp | ipopt |
|---|---|---|
| 2a_converged_frac | 1 | 0.02 |
| 2a_n | 200 | 200 |
| 2a_phi_best | 0.0656887 | 0.0656571 |
| 2a_phi_mean | 0.0497217 | 0.0433103 |
| 2a_phi_median | 0.0489792 | 0.0420925 |
| 2a_time_median_s | 0.291132 | 0.192439 |
| 2a_time_p95_s | 0.411497 | 0.285878 |
| 2b_best_at_10s | 0.0556785 | 0.0518791 |
| 2b_best_at_T | 0.0656887 | 0.0534384 |
| 2b_distinct_maxima | 143 | 242 |
| 2b_restarts | 153 | 242 |
| 2b_restarts_within_10s | 29 | 39 |
| paired ccp wins / ties / total | 178 / 0 / 200 | |

## K6_mid_m2248

| metric | ccp | ipopt |
|---|---|---|
| 2a_converged_frac | 1 | 0.025 |
| 2a_n | 200 | 200 |
| 2a_phi_best | 0.108109 | 0.106658 |
| 2a_phi_mean | 0.0886828 | 0.0762499 |
| 2a_phi_median | 0.0882554 | 0.0739151 |
| 2a_time_median_s | 0.105042 | 0.141433 |
| 2a_time_p95_s | 0.131164 | 0.178382 |
| 2b_best_at_10s | 0.102227 | 0.0924832 |
| 2b_best_at_T | 0.108109 | 0.0995621 |
| 2b_distinct_maxima | 364 | 361 |
| 2b_restarts | 434 | 362 |
| 2b_restarts_within_10s | 77 | 58 |
| paired ccp wins / ties / total | 184 / 0 / 200 | |
