"""Settings of the reported runs (paper Table "Selected settings")."""
import numpy as np

# Empirical directional smoothness estimates L_k of F_k; they enter only the adaptive warm-start score.
L_ESTIMATES = {
    "fishwood": [0.240111632014359, 0.34796415276597487],
    "dst": [0.16001718180814467, 0.16001719646012644],
    "bb": [0.07334441503294956, 0.06938799010915637, 0.06884899017331293],
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

# Values of r and N: powers of two from 2 (FishWood, DST, Breakable Bottles) and r = 1, ..., 6 (Fruit Tree).  One
# fixed budget B per task, the same for all methods: 1.05 x the Gradient Calls of the farthest of these points
# (as measured before B was fixed), rounded up to a multiple of 1e3 (K=2), 3e3 (Breakable Bottles) or 6e3 (Fruit
# Tree).  Checkpoints of every method follow one Gradient-Call schedule: every B/600 (K=2) or B/120 (K>2) calls up
# to B, 10x that afterwards (GRAB: checkpoint_count = 600 / 120 over B).  A run is plotted if its point lies at <= B
# calls.  The next values (FishWood r=1024 and N=256, DST r=1024 and N=128, Breakable Bottles r=64, Fruit Tree r=7)
# have their point beyond B; DST N=2 is not plotted (its middle slot weight cycles without settling, so the rule
# never stops it).
TASKS = {
    "fishwood": dict(
        uniform=dict(M=50, lr=.03, values=[2, 4, 8, 16, 32, 64, 128, 256, 512]),
        surf=dict(K_S=25, lr=.03, values=[2, 4, 8, 16, 32, 64, 128]),
        adaptive=dict(inner_steps=10, lr=.01, lambda_method="envelope", checkpoint_count=600),
        budget=108000, every=180),
    "dst": dict(
        uniform=dict(M=5, lr=.3, values=[2, 4, 8, 16, 32, 64, 128, 256, 512]),
        surf=dict(K_S=25, lr=.1, values=[4, 8, 16, 32, 64]),
        adaptive=dict(inner_steps=10, lr=.1, lambda_method="envelope", checkpoint_count=600),
        budget=93000, every=155),
    "bb": dict(
        uniform=dict(M=5, lr=.03, values=[2, 4, 8, 16, 32]),
        adaptive=dict(inner_steps=25, lr=.03, lambda_method="k3_special", k3_rtol=.05, k3_max_nodes=1000,
                      checkpoint_count=120),
        budget=18000, every=150),
    "fruittree_d6": dict(
        uniform=dict(M=5, lr=.1, values=[1, 2, 3, 4, 5, 6]),
        adaptive=dict(inner_steps=10, lr=.03, lambda_method="periodic_strong_ccp", hybrid_period=10,
                      weak_ccp=(64, 1, 15), strong_ccp=(1024, 8, 100), boundary_ccp_seeds=True,
                      boundary_seed_resolution=20, fresh_ccp_seeds=True, ccp_keep_pool=64, lp_warm_start=True,
                      lp_cg=True, checkpoint_count=120),
        budget=60000, every=500),
}

def smoothness(name):
    return np.asarray(L_ESTIMATES[name], float)
