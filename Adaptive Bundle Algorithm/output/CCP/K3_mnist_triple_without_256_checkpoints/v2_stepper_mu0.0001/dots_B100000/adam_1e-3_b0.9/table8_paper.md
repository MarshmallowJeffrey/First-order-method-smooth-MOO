| Method | Final worst-case gradient norm | Budget ratio to reach the same worst-case gradient norm (baseline / adaptive) |
|---|---|---|
| Adaptive λ-bundle | 8.55e-03 | 1× |
| Uniform grid, r = 10 | 7.39e-02 | 8.7× |
| Uniform grid, r = 20 | 5.89e-02 | 10.0× |
| Uniform grid, r = 25 | 5.80e-02 | 15.2× |
| Uniform grid, r = 30 | 5.35e-02 | 12.2× |
| Uniform grid, r = 35 | 5.12e-02 | 12.3× |
| Uniform grid, r = 40 | 5.33e-02 | 6.7× |

B = 100,000 gradient evaluations for every run, seed 41; worst-case gradient norm from the IPOPT + CCP audit with the final resolution-500 grid check. Dot rule: the last step of the best-so-far staircase.
