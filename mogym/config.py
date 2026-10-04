"""Settings of the reported runs (paper Table "Selected settings")."""
import numpy as np

# Empirical directional smoothness estimates L_k of F_k; they enter only the starting-point rule of GRAB and Uniform.
L_ESTIMATES = {
    "fishwood": [0.240111632014359, 0.34796415276597487],
    "fruittree_d6": [0.13867360882755067, 0.1386705991183074, 0.1386719770172668,
                     0.13867074332994217, 0.13867109068783343, 0.1386744703018394],
}

# Run-to-plateau rule (mogym.plateau); SURF additionally requires its slot weights to have settled.
RULE = dict(tol=.01, window=3, confirm=3, own_ratio=.25, own_floor=1e-5)
SURF_RULE = dict(RULE, weight_tol=.005)
MAX_SWEEPS = 50000          # Uniform safety cap
MAX_ROUNDS = 1000           # SURF safety cap
ADAM_KEEP_TOL = .005        # a weight counts as unchanged if it moves by at most this in every coordinate
POINT_BAND = .05            # plotted point: earliest checkpoint after which the GN stays within 5% of the value at stopping

# Values of r and N: FishWood r = 2, 4, ..., 512 and N = 2, 4, ..., 128; Fruit Tree r = 1, ..., 6.  One fixed budget B
# per task, the same for all methods: 1.05 x the Gradient Calls of the farthest of these points (as measured before B
# was fixed: FishWood Uniform r=512, Fruit Tree r=6), rounded up to a multiple of 1e3 (FishWood) or 6e3 (Fruit Tree).
# Checkpoints of every method follow one Gradient-Call schedule: every B/600 (K=2) or B/120 (K=6) calls up to B, 10x
# that afterwards (GRAB: checkpoint_count = 600 / 120 over B; Uniform and SURF: `every`, which is floor(B/600) = 268
# for FishWood).  A run is plotted if its point lies at <= B calls.  FishWood SURF also uses N=272, the largest
# multiple of 16 whose point lies within B, to place one SURF point near the end of the budget.  The next values
# (FishWood r=1024 and N=288, Fruit Tree r=7) have their point beyond B.
TASKS = {
    "fishwood": dict(
        uniform=dict(M=50, lr=.03, values=[2, 4, 8, 16, 32, 64, 128, 256, 512]),
        surf=dict(K_S=25, lr=.03, values=[2, 4, 8, 16, 32, 64, 128, 272]),
        adaptive=dict(inner_steps=2, lr=.03, lambda_method="envelope", checkpoint_count=600),
        budget=161000, every=268),
    "fruittree_d6": dict(
        uniform=dict(M=1, lr=.03, values=[1, 2, 3, 4, 5, 6]),
        adaptive=dict(inner_steps=5, lr=.03, lambda_method="periodic_strong_ccp", hybrid_period=10,
                      weak_ccp=(64, 1, 15), strong_ccp=(1024, 8, 100), boundary_ccp_seeds=True,
                      boundary_seed_resolution=20, fresh_ccp_seeds=True, ccp_keep_pool=64, lp_warm_start=True,
                      lp_cg=True, checkpoint_count=120),
        budget=150000, every=1250),
}


def smoothness(name):
    return np.asarray(L_ESTIMATES[name], float)
