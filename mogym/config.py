"""Settings of the reported runs (paper Table "Selected settings").

The learning rates and step counts were chosen by the tuning procedure described in the paper appendix
(pilot budget 5e4 Gradient Calls, 1.5e5 for Fruit Tree); only the resulting values are listed here.
"""
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

# One fixed budget B per task.  Checkpoints of every method follow one Gradient-Call schedule: B/600 (K=2) or
# B/120 (K>2) calls up to B, 10x that afterwards (GRAB: checkpoint_count = 600 / 120 over B).  A Uniform r or
# SURF N is plotted if its point lies at <= B calls; r and N were increased until two consecutive values lie
# beyond B (not plotted: FishWood r=512, 1024 and N=256, 512; DST r=1024, 2048 and N=128, 256, and N=2, whose
# middle slot weight cycles without settling, so the rule never stops it; Breakable Bottles r=27, 28; Fruit
# Tree r=7, 8).
TASKS = {
    "fishwood": dict(
        uniform=dict(M=50, lr=.03, values=[2, 4, 8, 16, 32, 64, 128, 256]),
        surf=dict(K_S=25, lr=.03, values=[2, 4, 8, 16, 32, 64, 128]),
        adaptive=dict(inner_steps=10, lr=.01, lambda_method="envelope", checkpoint_count=600),
        budget=81000, every=135),
    "dst": dict(
        uniform=dict(M=5, lr=.3, values=[2, 4, 8, 16, 32, 64, 128, 256, 512]),
        surf=dict(K_S=25, lr=.1, values=[4, 8, 16, 32, 64]),
        adaptive=dict(inner_steps=10, lr=.1, lambda_method="envelope", checkpoint_count=600),
        budget=96000, every=160),
    "bb": dict(
        uniform=dict(M=25, lr=.1, values=list(range(1, 27))),
        adaptive=dict(inner_steps=25, lr=.03, lambda_method="k3_special", k3_rtol=.05, k3_max_nodes=1000,
                      checkpoint_count=120),
        budget=27000, every=225),
    "fruittree_d6": dict(
        uniform=dict(M=25, lr=.03, values=[1, 2, 3, 4, 5, 6]),
        adaptive=dict(inner_steps=10, lr=.03, lambda_method="periodic_strong_ccp", hybrid_period=20,
                      weak_ccp=(64, 1, 15), strong_ccp=(1024, 8, 100), boundary_ccp_seeds=True,
                      boundary_seed_resolution=20, fresh_ccp_seeds=True, lp_warm_start=True, checkpoint_count=120),
        budget=150000, every=1250),
}


def smoothness(name):
    return np.asarray(L_ESTIMATES[name], float)
