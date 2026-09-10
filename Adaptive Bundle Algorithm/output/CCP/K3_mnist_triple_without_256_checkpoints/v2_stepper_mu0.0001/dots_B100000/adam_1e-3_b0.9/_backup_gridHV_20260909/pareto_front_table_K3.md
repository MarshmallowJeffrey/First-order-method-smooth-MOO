| Method | HV, training loss space | Gap to adaptive (train) | HV, test cross-entropy space | Gap to adaptive (test) |
|---|---|---|---|---|
| Adaptive λ-bundle | 1.2766 | +0.0% | 1.3080 | +0.0% |
| Uniform grid, r = 10 | 1.2803 | -0.3% | 1.2905 | +1.3% |
| Uniform grid, r = 20 | 1.2819 | -0.4% | 1.2958 | +0.9% |
| Uniform grid, r = 25 | 1.2826 | -0.5% | 1.2886 | +1.5% |
| Uniform grid, r = 30 | 1.2829 | -0.5% | 1.2872 | +1.6% |
| Uniform grid, r = 35 | 1.2823 | -0.4% | 1.2970 | +0.8% |
| Uniform grid, r = 40 | 1.2829 | -0.5% | 1.2923 | +1.2% |

HV = hypervolume of the non-dominated set of all delivered points, reference point (ln 3, ln 3, ln 3); gap = (HV_adaptive - HV) / HV_adaptive, negative means the baseline is larger.
