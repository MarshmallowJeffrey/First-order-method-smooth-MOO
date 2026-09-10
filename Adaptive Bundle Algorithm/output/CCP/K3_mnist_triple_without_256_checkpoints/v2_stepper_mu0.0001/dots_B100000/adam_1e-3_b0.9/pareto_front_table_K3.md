| Method | HV, training loss space (exact) | Gap to adaptive (train) | HV, test cross-entropy space (exact) | Gap to adaptive (test) |
|---|---|---|---|---|
| Adaptive λ-bundle | 1.2760 | 0.00% | 1.3076 | 0.00% |
| Uniform grid, r = 40 | 1.2825 | −0.51% | 1.2941 | +1.03% |
| Uniform grid, r = 20 | 1.2821 | −0.48% | 1.2954 | +0.93% |
| Uniform grid, r = 35 | 1.2821 | −0.48% | 1.2960 | +0.89% |
| Uniform grid, r = 30 | 1.2819 | −0.46% | 1.2882 | +1.48% |
| Uniform grid, r = 25 | 1.2816 | −0.44% | 1.2869 | +1.58% |
| Uniform grid, r = 10 | 1.2800 | −0.32% | 1.2902 | +1.33% |

HV = EXACT hypervolume (sweep over the third coordinate with 2-D staircase slabs) of the non-dominated set of all delivered points, reference point (ln 3, ln 3, ln 3), full reference box (independent of the figure's display window); rows ordered by training HV; representative of Figure 6 = uniform_r40_seed41 (largest exact hypervolume in the training loss space (same r for train and test)). Gap = (HV_adaptive − HV_method) / HV_adaptive; negative = the method's HV is larger. Single seed (41).
