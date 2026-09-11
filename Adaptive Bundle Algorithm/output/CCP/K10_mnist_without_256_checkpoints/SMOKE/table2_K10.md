| run | HV (train, MC, ref (ln 10)^10 = 4189.4) | gap vs CCP | delivered points inside the reference box | non-dominated inside | best max-class CE |
|---|---|---|---|---|---|
| adaptive_ccp_seed41 | 0.0000 ± 0.0000 | +nan% | 0.0% | 0 | 4.479 |
| uniform_r2_seed41 | 0.0000 ± 0.0000 | +nan% | 0.0% | 0 | 5.275 |

Monte Carlo with 20,000 uniform samples in [0, ln 10]^10 (seed 2026); gap = (HV_CCP − HV_run) / HV_CCP. A delivered point with any class CE ≥ ln 10 lies outside the reference box and contributes nothing.