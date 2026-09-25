"""Settings of the paper's MNIST experiments (Section 4.1 and Appendix C.1)."""

from __future__ import annotations

from .ccp import CCPConfig

DIGITS = {2: (4, 9), 3: (4, 7, 9)}               # chosen by the conflict screening
RHO = {2: 1.0 / 9.0, 3: 3.0 / 17.0}              # every class keeps a weight of at least 5 %
MU_PLAIN = 1e-3


def mu_of(K: int) -> float:
    """mu = (1 + rho) 1e-3: the pooled problem is (1 + rho) x the ridge problem with mu = 1e-3."""
    return (1.0 + RHO[K]) * MU_PLAIN


BATCH_SIZE = 1024
N_PROBES, PROBE_SEED = 40, 7                     # smoothness estimates
SEEDS = (41, 42, 43)                             # mini-batch sampling seeds (theta_0 is the same for all)
BUDGET = 480_000.0                               # gradient calls
SEGMENTS = 5                                     # per decision / grid visit / SURF slot and round
STEP_RULE = "adam_alpha0.001_beta20.9"           # Adam(1e-3, beta2 = 0.9), the winner of the step-rule experiment
CCP_DECISIONS = CCPConfig(N0=2000, r=10, seed=0)

INF = float("inf")
SCHEDULE = {2: [(20_000.0, 250.0), (80_000.0, 1_000.0), (INF, 2_000.0)],     # checkpoint every ... gradient calls
            3: [(80_000.0, 1_000.0), (INF, 2_000.0)]}
AUDIT_GRID_K2 = 200_001

# the runs of the paper
UNIFORM_R = {2: [2, 3, 4, 6, 8, 10, 15, 16, 20, 25, 30, 32, 36, 40, 45, 50, 52, 55, 56, 60, 64],
             3: [2, 3, 4, 5, 6, 8, 10, 12, 13, 14, 15, 16, 17, 19, 20, 22, 24]}
SURF_N = [2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 35, 37, 38, 39, 40]

# step-rule experiment on {4,9} (adaptive method, all eleven rules)
STEP_RULE_SEEDS = (41, 141, 241)
STEP_RULE_BUDGET = 10_000.0
STEP_RULE_CADENCE = 200.0
STEP_RULE_AUDIT_GRID = 20_001

# drawn (and fitted) in the worst-case gradient norm figure; the other configurations are listed in the appendix
# tables (marked with a dagger)
FIGURE_UNIFORM_R = {2: [2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 36, 45, 52, 60, 64],
                    3: [5, 6, 8, 10, 12, 14, 16, 17, 19, 22, 24]}
FIGURE_SURF_N = [3, 6, 8, 10, 15, 20, 25, 30, 37, 38]

# linear scalarization fronts: the best configurations that plateau in at least two of three seeds
FRONT_LEGS = {2: {"uniform": 60, "surf": 38, "seeds": (41, 42, 43)},
              3: {"uniform": 24, "seeds": (41, 42, 43)}}
FRONT_WINDOW = {2: 0.13, 3: 0.5}

# the best configuration of each baseline (Table in Section 4.1)
BEST = {2: {"uniform": 60, "surf": 38}, 3: {"uniform": 24}}
